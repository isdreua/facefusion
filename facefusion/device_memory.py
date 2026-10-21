import ctypes
import threading
import time
from typing import Any, Dict, List, Optional

from facefusion import logger, state_manager

# Read through nvml rather than the cuda driver api, because nvml needs no current context: a
# cuMemGetInfo would have to make the primary context current on whichever thread samples, just to
# read a number, and nvml reports the same device wide figures nvidia-smi does
NVML_SUCCESS = 0
NVML_LIBRARY_NAMES = [ 'libnvidia-ml.so.1', 'libnvidia-ml.so', 'nvml.dll' ]
SAMPLE_INTERVAL = 1.0
SUMMARY_INTERVAL = 60.0
# A conv whose workspace allocation fails reports CUDNN_BACKEND_API_FAILED rather than an out of
# memory error, and the cuda context is gone by the time anything else notices, so the warning has
# to come while there is still room to act on it
LOW_FREE_FRACTION = 0.05
LOW_FREE_BYTES = 256 * 1024 * 1024

DEVICE_MEMORY_LOCK = threading.Lock()
DEVICE_MEMORY_LIBRARY : Dict[str, Any] = { 'library': None, 'is_resolved': False, 'handles': {} }
DEVICE_MEMORY_METRICS : Dict[str, float] = { 'samples': 0, 'free_bytes': 0, 'total_bytes': 0, 'min_free_bytes': 0, 'first_free_bytes': 0, 'last_sample_time': 0, 'next_sample_time': 0, 'next_summary_time': 0, 'has_warned': 0 }


class NvmlMemory(ctypes.Structure):
	_fields_ = [ ('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong) ]


def resolve_nvml_library() -> Optional[Any]:
	if DEVICE_MEMORY_LIBRARY.get('is_resolved'):
		return DEVICE_MEMORY_LIBRARY.get('library')
	DEVICE_MEMORY_LIBRARY['is_resolved'] = True

	for nvml_library_name in NVML_LIBRARY_NAMES:
		try:
			nvml_library = ctypes.CDLL(nvml_library_name)
		except OSError:
			continue
		nvml_init = resolve_nvml_function(nvml_library, [ 'nvmlInit_v2', 'nvmlInit' ])

		if nvml_init and nvml_init() == NVML_SUCCESS:
			DEVICE_MEMORY_LIBRARY['library'] = nvml_library
			return nvml_library

	logger.debug('nvml could not be loaded, so the stream reports no device memory', __name__)
	return None


# The v2 symbols are the current abi; the older names stay as a fallback for drivers that predate them
def resolve_nvml_function(nvml_library : Any, function_names : List[str]) -> Optional[Any]:
	for function_name in function_names:
		if hasattr(nvml_library, function_name):
			return getattr(nvml_library, function_name)
	return None


def resolve_nvml_device_handle(device_id : int) -> Optional[Any]:
	nvml_library = resolve_nvml_library()

	if not nvml_library:
		return None
	if device_id in DEVICE_MEMORY_LIBRARY.get('handles'):
		return DEVICE_MEMORY_LIBRARY.get('handles').get(device_id)
	nvml_device_get_handle = resolve_nvml_function(nvml_library, [ 'nvmlDeviceGetHandleByIndex_v2', 'nvmlDeviceGetHandleByIndex' ])

	if not nvml_device_get_handle:
		return None
	device_handle = ctypes.c_void_p()

	if nvml_device_get_handle(ctypes.c_uint(device_id), ctypes.byref(device_handle)) != NVML_SUCCESS:
		logger.debug('nvml has no handle for device ' + str(device_id) + ', so the stream reports no device memory', __name__)
		DEVICE_MEMORY_LIBRARY['handles'][device_id] = None
		return None
	DEVICE_MEMORY_LIBRARY['handles'][device_id] = device_handle
	return device_handle


# Free and total bytes on the device, or zeros when nvml is not there to ask
def resolve_device_memory(device_id : int) -> Dict[str, int]:
	nvml_library = resolve_nvml_library()
	device_handle = resolve_nvml_device_handle(device_id)

	if not nvml_library or not device_handle:
		return { 'free_bytes': 0, 'total_bytes': 0 }
	nvml_device_get_memory_info = resolve_nvml_function(nvml_library, [ 'nvmlDeviceGetMemoryInfo' ])

	if not nvml_device_get_memory_info:
		return { 'free_bytes': 0, 'total_bytes': 0 }
	nvml_memory = NvmlMemory()

	if nvml_device_get_memory_info(device_handle, ctypes.byref(nvml_memory)) != NVML_SUCCESS:
		return { 'free_bytes': 0, 'total_bytes': 0 }
	return { 'free_bytes': int(nvml_memory.free), 'total_bytes': int(nvml_memory.total) }


def has_device_memory_provider() -> bool:
	execution_providers = state_manager.get_item('execution_providers') or []
	return 'cuda' in execution_providers or 'tensorrt' in execution_providers


def reset_device_memory_metrics() -> None:
	with DEVICE_MEMORY_LOCK:
		for metric_name in DEVICE_MEMORY_METRICS:
			DEVICE_MEMORY_METRICS[metric_name] = 0


# Called from the stream loop every frame and throttled here, because the loop spins on an empty
# camera queue and a reading is only worth taking once a second
def sample_device_memory() -> None:
	if not has_device_memory_provider():
		return
	current_time = time.perf_counter()

	with DEVICE_MEMORY_LOCK:
		if current_time < DEVICE_MEMORY_METRICS.get('next_sample_time'):
			return
		DEVICE_MEMORY_METRICS['next_sample_time'] = current_time + SAMPLE_INTERVAL
	execution_device_ids = state_manager.get_item('execution_device_ids') or [ 0 ]
	device_memory = resolve_device_memory(execution_device_ids[0])
	free_bytes = device_memory.get('free_bytes')
	total_bytes = device_memory.get('total_bytes')

	if not free_bytes or not total_bytes:
		return

	with DEVICE_MEMORY_LOCK:
		if not DEVICE_MEMORY_METRICS.get('samples'):
			DEVICE_MEMORY_METRICS['first_free_bytes'] = free_bytes
			DEVICE_MEMORY_METRICS['min_free_bytes'] = free_bytes
			DEVICE_MEMORY_METRICS['next_summary_time'] = current_time + SUMMARY_INTERVAL
		DEVICE_MEMORY_METRICS['samples'] += 1
		DEVICE_MEMORY_METRICS['free_bytes'] = free_bytes
		DEVICE_MEMORY_METRICS['total_bytes'] = total_bytes
		DEVICE_MEMORY_METRICS['min_free_bytes'] = min(DEVICE_MEMORY_METRICS.get('min_free_bytes'), free_bytes)
		DEVICE_MEMORY_METRICS['last_sample_time'] = current_time
		# The first reading is logged as a summary as well, so the log carries the baseline the drift
		# is measured against rather than only the deltas
		is_summary_due = DEVICE_MEMORY_METRICS.get('samples') == 1 or current_time >= DEVICE_MEMORY_METRICS.get('next_summary_time')
		if is_summary_due:
			DEVICE_MEMORY_METRICS['next_summary_time'] = current_time + SUMMARY_INTERVAL
		is_low = free_bytes < max(LOW_FREE_BYTES, total_bytes * LOW_FREE_FRACTION)
		has_warned = bool(DEVICE_MEMORY_METRICS.get('has_warned'))
		if is_low and not has_warned:
			DEVICE_MEMORY_METRICS['has_warned'] = 1
		device_memory_metrics = dict(DEVICE_MEMORY_METRICS)

	logger.debug('device memory ' + format_device_memory(device_memory_metrics), __name__)

	# The drift is the reading that matters: the arenas grow while the stream runs, so a monotonic
	# decline is the leak, and a flat figure clears it
	if is_summary_due:
		logger.info('device memory ' + format_device_memory(device_memory_metrics), __name__)
	if is_low and not has_warned:
		logger.warn('device memory is nearly exhausted (' + format_device_memory(device_memory_metrics) + '); the next convolution that cannot allocate its workspace takes the cuda context down with it', __name__)


def format_device_memory(device_memory_metrics : Dict[str, float]) -> str:
	free_mb = device_memory_metrics.get('free_bytes') / (1024 * 1024)
	total_mb = device_memory_metrics.get('total_bytes') / (1024 * 1024)
	min_free_mb = device_memory_metrics.get('min_free_bytes') / (1024 * 1024)
	drift_mb = (device_memory_metrics.get('free_bytes') - device_memory_metrics.get('first_free_bytes')) / (1024 * 1024)

	return str(round(free_mb)) + ' MB free of ' + str(round(total_mb)) + ' MB, min ' + str(round(min_free_mb)) + ' MB, drift ' + format(drift_mb, '+.0f') + ' MB'


def get_device_memory_metrics() -> Dict[str, float]:
	with DEVICE_MEMORY_LOCK:
		return dict(DEVICE_MEMORY_METRICS)
