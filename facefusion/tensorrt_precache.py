import os
import threading
from contextlib import contextmanager
from time import time
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy

from facefusion import face_detector, logger, state_manager
from facefusion.execution import resolve_cache_path
from facefusion.processors.core import get_processors_modules, resolve_processor_name
from facefusion.time_helper import calculate_end_time
from facefusion.types import ErrorCode
from facefusion.vision import unpack_resolution

HEARTBEAT_INTERVAL = 30
INPUT_DTYPE_SET : Dict[str, Any] =\
{
	'tensor(float)': numpy.float32,
	'tensor(float16)': numpy.float16,
	'tensor(double)': numpy.float64,
	'tensor(int64)': numpy.int64,
	'tensor(int32)': numpy.int32,
	'tensor(int8)': numpy.int8,
	'tensor(uint8)': numpy.uint8,
	'tensor(bool)': numpy.bool_
}


def run() -> ErrorCode:
	if 'tensorrt' not in state_manager.get_item('execution_providers'):
		logger.error('tensorrt precache needs tensorrt in the execution providers, got ' + ', '.join(state_manager.get_item('execution_providers')) + ' (a webcam config settings block overrides --execution-providers)', __name__)
		return 2

	execution_device_ids = state_manager.get_item('execution_device_ids')
	precache_modules = collect_precache_modules()
	cache_path = resolve_cache_path()
	step_total = len(execution_device_ids) * len(precache_modules)
	step_index = 0
	start_time = time()
	start_cache_files = collect_cache_files(cache_path)

	logger.info('caching into ' + os.path.abspath(cache_path) + ', ' + str(len(start_cache_files)) + ' file(s) already there', __name__)
	logger.info(str(step_total) + ' step(s): ' + ', '.join(resolve_processor_name(module) for module in precache_modules) + ' on device(s) ' + ', '.join(map(str, execution_device_ids)), __name__)

	try:
		for execution_device_id in execution_device_ids:
			# One device at a time, so the pool returned is the one for this device and not a random pick
			state_manager.set_item('execution_device_ids', [ execution_device_id ])

			for precache_module in precache_modules:
				step_index += 1
				step_label = '[' + str(step_index) + '/' + str(step_total) + '] ' + resolve_processor_name(precache_module) + ' (device ' + str(execution_device_id) + ')'
				precache_step(precache_module, step_label, cache_path)
	finally:
		state_manager.set_item('execution_device_ids', execution_device_ids)

	end_cache_files = collect_cache_files(cache_path)
	cache_size = sum(os.path.getsize(os.path.join(cache_path, cache_file)) for cache_file in end_cache_files)
	logger.info('precache finished in ' + str(calculate_end_time(start_time)) + ' seconds, ' + str(len(end_cache_files - start_cache_files)) + ' new file(s), ' + str(len(end_cache_files)) + ' total (' + str(round(cache_size / 1024 / 1024, 1)) + ' MB)', __name__)
	return 0


def collect_precache_modules() -> List[ModuleType]:
	processor_modules = get_processors_modules(state_manager.get_item('processors'))
	common_modules : List[ModuleType] = []

	for processor_module in processor_modules:
		for common_module in processor_module.get_common_modules():
			if common_module not in common_modules:
				common_modules.append(common_module)

	return common_modules + processor_modules


def precache_step(precache_module : ModuleType, step_label : str, cache_path : str) -> None:
	step_start_time = time()
	step_cache_files = collect_cache_files(cache_path)

	logger.info(step_label + ': loading models, tensorrt builds every engine that is not cached yet', __name__)
	with log_heartbeat(step_label + ': still loading'):
		inference_pool = precache_module.get_inference_pool()
	logger.info(step_label + ': loaded ' + (', '.join(inference_pool.keys()) or 'no models') + ' in ' + str(calculate_end_time(step_start_time)) + ' seconds', __name__)

	for model_name, inference_session in inference_pool.items():
		warm_up_session(precache_module, model_name, inference_session, step_label)

	if hasattr(precache_module, 'clear_inference_pool'):
		precache_module.clear_inference_pool()
	logger.info(step_label + ': done in ' + str(calculate_end_time(step_start_time)) + ' seconds, ' + str(len(collect_cache_files(cache_path) - step_cache_files)) + ' new cache file(s)', __name__)


# Models with dynamic input shapes only get their engine on the first run, so one run at the shape
# the real pipeline feeds is what makes the cache complete
def warm_up_session(precache_module : ModuleType, model_name : str, inference_session : Any, step_label : str) -> None:
	model_label = step_label + ' > ' + model_name

	if 'TensorrtExecutionProvider' not in inference_session.get_providers():
		logger.info(model_label + ': not on tensorrt (' + ', '.join(inference_session.get_providers()) + '), nothing to cache', __name__)
		return

	input_feed = {}

	for session_input in inference_session.get_inputs():
		input_shape = resolve_input_shape(precache_module, model_name, session_input.shape)
		input_dtype = INPUT_DTYPE_SET.get(session_input.type)

		if not input_shape or not input_dtype:
			logger.warn(model_label + ': input ' + session_input.name + ' ' + str(session_input.shape) + ' cannot be resolved, its engine is built on the first real frame', __name__)
			return
		input_feed[session_input.name] = numpy.zeros(input_shape, dtype = input_dtype)

	warm_up_start_time = time()
	input_description = ', '.join(input_name + ' ' + 'x'.join(map(str, input_value.shape)) for input_name, input_value in input_feed.items())
	logger.info(model_label + ': warm-up run with ' + input_description, __name__)

	try:
		with log_heartbeat(model_label + ': still warming up'):
			inference_session.run(None, input_feed)
		logger.info(model_label + ': warm-up done in ' + str(calculate_end_time(warm_up_start_time)) + ' seconds', __name__)
	except Exception as exception:
		logger.warn(model_label + ': warm-up failed, its engine is built on the first real frame (' + str(exception).splitlines()[0] + ')', __name__)


def resolve_input_shape(precache_module : ModuleType, model_name : str, input_shape : List[Any]) -> Optional[List[int]]:
	resolved_shape = []
	model_size = resolve_model_size(precache_module, model_name)

	for dimension_index, dimension in enumerate(input_shape):
		if isinstance(dimension, int) and dimension > 0:
			resolved_shape.append(dimension)
		elif dimension_index == 0:
			resolved_shape.append(1)
		elif model_size and len(input_shape) == 4 and dimension_index in [ 2, 3 ]:
			resolved_shape.append(model_size[3 - dimension_index])
		else:
			return None

	return resolved_shape


# Width and height the pipeline feeds this model, used only for dimensions the model leaves open
def resolve_model_size(precache_module : ModuleType, model_name : str) -> Optional[Tuple[int, int]]:
	if precache_module is face_detector:
		return unpack_resolution(state_manager.get_item('face_detector_size'))

	if hasattr(precache_module, 'create_static_model_set'):
		model_set = precache_module.create_static_model_set('full').get(model_name)

		if model_set and model_set.get('size'):
			return model_set.get('size')

	if hasattr(precache_module, 'get_model_options'):
		model_options = precache_module.get_model_options()

		if model_options and model_options.get('size'):
			return model_options.get('size')

	return None


def collect_cache_files(cache_path : str) -> Set[str]:
	if os.path.isdir(cache_path):
		return set(os.listdir(cache_path))
	return set()


# Building an engine blocks for minutes without a word from tensorrt, so say it is still alive
@contextmanager
def log_heartbeat(message : str) -> Iterator[None]:
	stop_event = threading.Event()
	start_time = time()

	def heartbeat() -> None:
		while not stop_event.wait(HEARTBEAT_INTERVAL):
			logger.info(message + ', ' + str(int(time() - start_time)) + ' seconds so far', __name__)

	heartbeat_thread = threading.Thread(target = heartbeat, daemon = True)
	heartbeat_thread.start()

	try:
		yield
	finally:
		stop_event.set()
		heartbeat_thread.join()
