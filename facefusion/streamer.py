import os
import queue
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from types import ModuleType
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Tuple

import cv2
import numpy
from tqdm import tqdm

from facefusion import ffmpeg_builder, logger, state_manager, translator
from facefusion.app_context import detect_app_context, set_app_context_override
from facefusion.audio import create_empty_audio_frame
from facefusion.camera_manager import CameraSwitch
from facefusion.common_helper import get_first, is_windows
from facefusion.content_analyser import analyse_frame
from facefusion.device_memory import reset_device_memory_metrics, sample_device_memory
from facefusion.face_creator import set_face_analysis_features
from facefusion.face_detector import set_stream_worker_count
from facefusion.face_helper import set_paste_in_place
from facefusion.face_selector import begin_face_selection_context, end_face_selection_context
from facefusion.ffmpeg import open_ffmpeg
from facefusion.filesystem import has_image, is_directory
from facefusion.inference_timer import get_inference_timer, reset_inference_timer
from facefusion.processors.core import get_processors_modules, resolve_processor_name
from facefusion.stream_abort import StreamAborted, abort_stale_stream_frame, set_stream_abort
from facefusion.types import AppContext, AudioFrame, Fps, Mask, Resolution, StreamMode, VisionFrame
from facefusion.vision import extract_vision_mask, is_vision_frame, read_static_images, restrict_frame
from facefusion.webcam_config import DEFAULT_SOURCE_FACE_NAME, NO_SOURCE_FACE_NAME

CONTENT_ANALYSIS_METRICS_LOCK = threading.Lock()
CONTENT_ANALYSIS_METRICS = { 'runs': 0, 'last_ms': 0.0, 'max_ms': 0.0 }
WARM_BARRIER_TIMEOUT = 10.0
IDLE_LISTENER_INTERVAL = 0.1
# The longest the scheduler sleeps on its wake event; a camera frame or a finished frame wakes it at once
STREAM_WAKE_TIMEOUT = 0.005
# How long the delivering side waits for a result before it checks whether the scheduler is still alive
STREAM_DELIVERY_POLL = 0.1
# How long stopping a stream waits for the scheduler to release the camera and the workers
STREAM_SCHEDULER_JOIN_TIMEOUT = 5.0
# The processors the running stream left out because their stream check failed, so the overlay can
# tell a processor that is selected from one that is actually touching the frame
STREAM_SKIPPED_PROCESSORS : List[str] = []
STREAM_SKIPPED_PROCESSOR_HINTS : Dict[str, str] =\
{
	'face_swapper': ': it needs a source image with a detectable face',
	'lip_syncer': ': it needs an audio file among the sources, and the stream has no live microphone so it would sync to silence'
}


# A frame older than the one the output has already shown is certain to be dropped: the scheduler
# trims everything before the latest completed future without yielding it. That makes the index of
# the last yielded frame the one fact a worker can act on, so it is published here for the workers
# to read between processors. An int assignment is atomic under the gil, and a worker reading the
# previous value only costs one more processor before it notices
class StreamFrameCursor:
	def __init__(self) -> None:
		self.yielded_frame_index = 0

	def has_passed(self, frame_index : int) -> bool:
		return frame_index < self.yielded_frame_index


class CameraCaptureThread(threading.Thread):
	MAX_CONSECUTIVE_READ_FAILURES = 30
	READ_FAILURE_RETRY_DELAY = 0.01

	def __init__(self, camera_capture: cv2.VideoCapture):
		super().__init__()
		self.camera_capture = camera_capture
		self.raw_frame_listener : Optional[Callable[[VisionFrame], None]] = None
		self.camera_switch : Optional[CameraSwitch] = None
		# Set after every queued frame, so the scheduler waits on one event for a camera frame or a
		# finished frame instead of polling the queue
		self.frame_event : Optional[threading.Event] = None
		self.frame_queue = queue.Queue(maxsize=1)
		self.running = True
		self.daemon = True

	# With a camera switch the thread never ends on its own: a camera that stops delivering (unplugged,
	# taken by another program) is released and opened again every 100 ms, and a switch is tried
	# between reads. Without one, as in the tests, it ends after a run of failed reads as before
	def run(self):
		consecutive_read_failures = 0
		try:
			while self.running:
				if self.camera_switch and self.try_camera_switch():
					consecutive_read_failures = 0
				if not self.camera_capture or not self.camera_capture.isOpened():
					if not self.camera_switch:
						break
					self.lose_camera()
					time.sleep(self.READ_FAILURE_RETRY_DELAY)
					continue
				capture_time = time.perf_counter()
				ret, frame = self.camera_capture.read()
				capture_read_end = time.perf_counter()
				if not ret:
					consecutive_read_failures += 1
					if not self.camera_capture.isOpened() or consecutive_read_failures >= self.MAX_CONSECUTIVE_READ_FAILURES:
						if not self.camera_switch:
							break
						self.lose_camera()
						consecutive_read_failures = 0
						continue
					time.sleep(self.READ_FAILURE_RETRY_DELAY)
					continue
				consecutive_read_failures = 0
				# Every frame the camera delivers, before any processing or skipping decides its fate,
				# so the raw view runs at the camera's own rate. The listener only stores a reference
				if self.raw_frame_listener:
					self.raw_frame_listener(frame)
				try:
					self.frame_queue.put_nowait((capture_time, capture_read_end, frame))
				except queue.Full:
					try:
						self.frame_queue.get_nowait()
					except queue.Empty:
						pass
					try:
						self.frame_queue.put_nowait((capture_time, capture_read_end, frame))
					except queue.Full:
						pass
				if self.frame_event:
					self.frame_event.set()
		finally:
			self.running = False

	# One attempt when one is due. The current camera keeps delivering until the new one is open,
	# and is released only then. Swapped here because this thread is the capture's only reader
	def try_camera_switch(self) -> bool:
		camera_id = self.camera_switch.take_due_target()
		if camera_id is None:
			return False
		next_capture = self.camera_switch.try_open(camera_id)
		if not next_capture:
			return False
		previous_capture = self.camera_capture
		self.camera_capture = next_capture
		if previous_capture is not None and previous_capture is not next_capture:
			self.camera_switch.release_capture(previous_capture)
		self.camera_switch.commit(camera_id)
		return True

	# Released so the device is free to be opened again, by this stream or another program
	def lose_camera(self) -> None:
		if self.camera_capture is not None:
			self.camera_switch.release_capture(self.camera_capture)
			self.camera_capture = None
		self.camera_switch.mark_lost()

	def stop(self):
		self.running = False

def analyse_frame_background(vision_frame: VisionFrame, stop_event: threading.Event):
	started = time.perf_counter()
	try:
		if analyse_frame(vision_frame):
			stop_event.set()
	finally:
		elapsed_ms = (time.perf_counter() - started) * 1000
		with CONTENT_ANALYSIS_METRICS_LOCK:
			CONTENT_ANALYSIS_METRICS['runs'] += 1
			CONTENT_ANALYSIS_METRICS['last_ms'] = elapsed_ms
			CONTENT_ANALYSIS_METRICS['max_ms'] = max(CONTENT_ANALYSIS_METRICS['max_ms'], elapsed_ms)


def get_content_analysis_metrics() -> Dict[str, float]:
	with CONTENT_ANALYSIS_METRICS_LOCK:
		return dict(CONTENT_ANALYSIS_METRICS)


# The processors' own reasons stay muted because they are worded for a file run, but a processor that
# is left out is named: dropping one silently made toggling it in the browser look like it did nothing.
# Only runs on start and on a change of the processor selection, so the warning cannot repeat per frame
def validate_stream_processors(processor_names : List[str]) -> List[ModuleType]:
	processor_modules = get_processors_modules(processor_names)
	validated_processor_modules = []
	skipped_processor_names = []

	for processor_module in processor_modules:
		logger.disable()
		try:
			is_processor_ready = processor_module.pre_process('stream')
		finally:
			logger.enable()
		if is_processor_ready:
			validated_processor_modules.append(processor_module)
		else:
			skipped_processor_names.append(resolve_processor_name(processor_module))

	for processor_name in skipped_processor_names:
		logger.warn(processor_name + ' cannot run in the webcam stream with the current settings and was left out' + STREAM_SKIPPED_PROCESSOR_HINTS.get(processor_name, ''), __name__)
	STREAM_SKIPPED_PROCESSORS[:] = skipped_processor_names
	return validated_processor_modules


def get_stream_skipped_processors() -> List[str]:
	return list(STREAM_SKIPPED_PROCESSORS)


def prepare_stream_inputs(processor_modules : List[ModuleType], source_vision_frames : List[VisionFrame]) -> Dict[str, Dict[str, Any]]:
	processor_stream_inputs = {}

	for processor_module in processor_modules:
		if hasattr(processor_module, 'prepare_stream_inputs'):
			processor_stream_inputs[processor_module.__name__] = processor_module.prepare_stream_inputs(source_vision_frames)

	return processor_stream_inputs


def prepare_stream_processors(processor_names : List[str], source_vision_frames : List[VisionFrame]) -> Tuple[List[ModuleType], Dict[str, Dict[str, Any]]]:
	validated_processor_modules = validate_stream_processors(processor_names)
	return validated_processor_modules, prepare_stream_inputs(validated_processor_modules, source_vision_frames)


# One face group is detected, recognized and embedded here, so switching to it later is a dictionary
# lookup rather than any inference work
def prepare_stream_face_set(processor_modules : List[ModuleType], face_name : str, source_paths : List[str]) -> Dict[str, Any]:
	source_vision_frames = read_static_images(source_paths)

	if source_paths and not source_vision_frames:
		logger.warn('source face ' + face_name + ' has no readable image', __name__)

	return\
	{
		'source_paths': list(source_paths),
		'source_vision_frames': source_vision_frames,
		'processor_stream_inputs': prepare_stream_inputs(processor_modules, source_vision_frames)
	}


# The no source face entry is written out rather than prepared, because prepare_stream_inputs
# would run every processor's own against an empty list and only face_swapper is known to
# tolerate that. Keeping it a real member of the set is what lets every lookup below stay
# unchanged: it is found instead of falling back to the first group
def create_no_source_face_set() -> Dict[str, Any]:
	return\
	{
		'source_paths': [],
		'source_vision_frames': [],
		'processor_stream_inputs': {}
	}


def prepare_stream_face_sets(processor_modules : List[ModuleType], source_face_groups : Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
	stream_face_sets = { face_name: prepare_stream_face_set(processor_modules, face_name, source_paths) for face_name, source_paths in source_face_groups.items() }
	stream_face_sets[NO_SOURCE_FACE_NAME] = create_no_source_face_set()
	return stream_face_sets


# Faces added while the stream runs must not cost the ones already prepared their embeddings again,
# so a group is only re-prepared when its own paths changed
def update_stream_face_sets(stream_face_sets : Dict[str, Dict[str, Any]], processor_modules : List[ModuleType], source_face_groups : Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
	updated_stream_face_sets = {}

	for face_name, source_paths in source_face_groups.items():
		stream_face_set = stream_face_sets.get(face_name)
		if stream_face_set and stream_face_set.get('source_paths') == source_paths:
			updated_stream_face_sets[face_name] = stream_face_set
		else:
			updated_stream_face_sets[face_name] = prepare_stream_face_set(processor_modules, face_name, source_paths)
	updated_stream_face_sets[NO_SOURCE_FACE_NAME] = create_no_source_face_set()

	return updated_stream_face_sets


# Groups loaded from the config and groups uploaded in the browser are kept apart in state so that
# provenance survives, and are merged only here
def resolve_source_face_groups() -> Dict[str, List[str]]:
	resolved_source_face_groups = {}

	for state_key in [ 'source_face_groups', 'source_face_uploads' ]:
		source_face_groups = state_manager.get_item(state_key)
		if isinstance(source_face_groups, dict):
			for face_name, source_paths in source_face_groups.items():
				resolved_source_face_groups[face_name] = list(source_paths)

	if not resolved_source_face_groups:
		resolved_source_face_groups[DEFAULT_SOURCE_FACE_NAME] = list(state_manager.get_item('source_paths') or [])

	return resolved_source_face_groups


# Groups that came from a config or a face directory never pass through the upload path that fills
# source_paths, and validate_stream_processors drops the face swapper when no detectable face is
# there, so a stream started from such a group would silently swap nothing
def apply_stream_source_paths() -> None:
	source_paths = [ source_path for source_paths in resolve_source_face_groups().values() for source_path in source_paths ]

	if has_image(source_paths):
		state_manager.init_item('source_paths', source_paths)


# A stream has no reference frame, so reference mode resolves no reference face and selects no
# target face at all: every processor would then run over a frame with nothing to work on. many is
# left alone because it is a deliberate choice that works frame by frame
def apply_stream_face_selector_mode() -> None:
	if state_manager.get_item('face_selector_mode') != 'many':
		state_manager.init_item('face_selector_mode', 'one')


def resolve_stream_face_set(stream_face_sets : Dict[str, Dict[str, Any]], source_face_name : Optional[str]) -> Tuple[Optional[str], List[VisionFrame], Dict[str, Dict[str, Any]]]:
	if source_face_name not in stream_face_sets:
		source_face_name = get_first(list(stream_face_sets))
	stream_face_set = stream_face_sets.get(source_face_name) or {}
	return source_face_name, stream_face_set.get('source_vision_frames') or [], stream_face_set.get('processor_stream_inputs') or {}


# Every worker thread that touches nvidia-vfx builds and loads its own effect, which takes seconds,
# and with ordered output that load stalls every frame behind it: left alone a run freezes once per
# worker over its first seconds. The warmup moves those loads to one startup pause
def warm_stream_workers(executor : ThreadPoolExecutor, worker_count : int, processor_modules : List[ModuleType], warm_vision_frame : VisionFrame) -> None:
	warm_processor_modules = [ processor_module for processor_module in processor_modules if hasattr(processor_module, 'warm_stream_worker') ]

	if not warm_processor_modules or worker_count < 1:
		return
	logger.debug('warming ' + str(worker_count) + ' stream workers for ' + ', '.join(processor_module.__name__ for processor_module in warm_processor_modules), __name__)
	# One task per worker, each holding its thread until every other task has one too, because an
	# executor hands out threads rather than letting a caller address them: an idle thread would
	# otherwise take a second task and leave another worker cold
	warm_barrier = threading.Barrier(worker_count)
	app_context = detect_app_context()
	warm_futures = [ executor.submit(warm_stream_worker, warm_barrier, warm_processor_modules, warm_vision_frame, app_context) for _ in range(worker_count) ]

	for warm_future in warm_futures:
		warm_future.result()


def warm_stream_worker(warm_barrier : threading.Barrier, processor_modules : List[ModuleType], warm_vision_frame : VisionFrame, app_context : AppContext) -> None:
	set_app_context_override(app_context)
	try:
		# Timed out rather than waited on forever, so a pool that cannot seat every task still
		# warms the threads it did seat instead of hanging the stream before its first frame
		warm_barrier.wait(WARM_BARRIER_TIMEOUT)
	except threading.BrokenBarrierError:
		pass
	try:
		for processor_module in processor_modules:
			logger.disable()
			try:
				processor_module.warm_stream_worker(warm_vision_frame)
			except Exception as exception:
				logger.enable()
				logger.debug('warming ' + processor_module.__name__ + ' failed, its first frame pays the load instead: ' + str(exception), __name__)
			finally:
				logger.enable()
	finally:
		set_app_context_override(None)


def has_stream_capacity(pending_total : int, buffered_total : int, max_queue_size : int) -> bool:
	return pending_total < max_queue_size and buffered_total < max_queue_size


# Only unfinished work occupies a worker, so results awaiting their turn in the ordered output are not
# counted; abandoned work that is still running is
def count_pending_futures(futures : List[Tuple[int, Future[Any]]], discarded_futures : List[Future[Any]]) -> int:
	pending_total = len(discarded_futures)

	for _, future in futures:
		if not future.done():
			pending_total += 1
	return pending_total


# Hands results from the scheduler thread to the thread that delivers them. Adaptive mode keeps one
# frame waiting, the newest: a frame the delivering side has not taken yet is replaced by the next
# one, the same latest-wins rule the camera queue and the writer apply. A deeper queue there handed
# out its oldest frame first, so a delivering side that fell behind showed frames one interval old
# for every extra place. The ordered modes queue up to maxsize and wait for room instead, so every
# frame still arrives in order
class StreamDelivery:
	def __init__(self, maxsize : int) -> None:
		self.maxsize = max(1, maxsize)
		self.items : Deque[Tuple[str, Any]] = deque()
		self.condition = threading.Condition()
		self.is_closed = False

	def put(self, stream_frame : Any, is_latest_wins : bool) -> None:
		with self.condition:
			while not is_latest_wins and len(self.items) >= self.maxsize and not self.is_closed:
				self.condition.wait(STREAM_DELIVERY_POLL)
			if self.is_closed:
				return
			# Only frames are queued until the end, so clearing never drops the end of the stream
			if is_latest_wins:
				self.items.clear()
			self.items.append(('frame', stream_frame))
			self.condition.notify_all()

	# The end of the stream, or why it failed; always queued, whatever the size
	def finish(self, failure : Optional[BaseException]) -> None:
		with self.condition:
			if not self.is_closed:
				self.items.append(('end', failure))
				self.condition.notify_all()

	def take(self, timeout : float) -> Optional[Tuple[str, Any]]:
		with self.condition:
			if not self.items:
				self.condition.wait(timeout)
			if not self.items:
				return None
			stream_item = self.items.popleft()
			self.condition.notify_all()
			return stream_item

	def close(self) -> None:
		with self.condition:
			self.is_closed = True
			self.items.clear()
			self.condition.notify_all()


# Runs the scheduler generator on its own thread. Stopped at a yield, it is closed there, which runs
# its own cleanup (camera, capture thread, executor) on this thread before the caller is told
def pump_stream_frames(stream_frames : Iterator[Tuple[VisionFrame, float, bool, Dict[str, Any]]], stream_delivery : StreamDelivery, stop_event : threading.Event) -> None:
	failure : Optional[BaseException] = None

	try:
		for stream_frame in stream_frames:
			if stop_event.is_set():
				break
			stream_delivery.put(stream_frame, (state_manager.get_item('webcam_frame_skipping') or 'adaptive') == 'adaptive')
	except BaseException as exception:
		failure = exception
	finally:
		try:
			stream_frames.close() #type:ignore[attr-defined]
		except BaseException as exception:
			failure = failure or exception
		stream_delivery.finish(failure)


# The scheduler used to be this generator itself, so while the caller worked on a delivered frame
# (colour conversion, frame generation, overlay, hand off to the writer) no finished frame was
# collected and no camera frame was submitted, and the workers waited. It now runs on a thread of
# its own and keeps the workers fed whatever the caller does; the caller still sees one frame at a
# time, in delivery order. The caller's app context goes with it, since that thread cannot detect
# it from its own stack
def multi_process_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps, webcam_execution_thread_count : Optional[int] = None, capture_resolution : Optional[Resolution] = None, raw_frame_listener : Optional[Callable[[VisionFrame], None]] = None, camera_switch : Optional[CameraSwitch] = None, idle_listener : Optional[Callable[[], bool]] = None) -> Iterator[Tuple[VisionFrame, float, bool, Dict[str, Any]]]:
	app_context = detect_app_context()
	stop_event = threading.Event()
	stream_delivery = StreamDelivery(max(2, webcam_execution_thread_count or state_manager.get_item('execution_thread_count') or 1))
	stream_frames = schedule_stream_frames(camera_capture, camera_fps, webcam_execution_thread_count, capture_resolution, raw_frame_listener, camera_switch, idle_listener, app_context, stop_event)
	scheduler_thread = threading.Thread(target = pump_stream_frames, args = (stream_frames, stream_delivery, stop_event), name = 'stream_scheduler', daemon = True)
	scheduler_thread.start()

	try:
		while True:
			stream_item = stream_delivery.take(STREAM_DELIVERY_POLL)
			if stream_item is None:
				# The scheduler always queues its end, so a dead thread with nothing queued is only the
				# moment between the two; one more look settles it
				if not scheduler_thread.is_alive():
					stream_item = stream_delivery.take(0.0)
					if stream_item is None:
						return
				else:
					continue
			item_kind, item_value = stream_item
			if item_kind == 'end':
				if item_value is not None:
					raise item_value
				return
			yield item_value
	finally:
		stop_event.set()
		stream_delivery.close()
		scheduler_thread.join(STREAM_SCHEDULER_JOIN_TIMEOUT)
		if scheduler_thread.is_alive():
			logger.warn('the stream scheduler did not stop within ' + str(STREAM_SCHEDULER_JOIN_TIMEOUT) + ' seconds', __name__)


def schedule_stream_frames(camera_capture : cv2.VideoCapture, camera_fps : Fps, webcam_execution_thread_count : Optional[int], capture_resolution : Optional[Resolution], raw_frame_listener : Optional[Callable[[VisionFrame], None]], camera_switch : Optional[CameraSwitch], idle_listener : Optional[Callable[[], bool]], app_context : AppContext, stop_event : threading.Event) -> Iterator[Tuple[VisionFrame, float, bool, Dict[str, Any]]]:
	# First, so the setup below reads and writes the caller's state rather than the cli one
	set_app_context_override(app_context)
	try:
		yield from schedule_stream_frames_in_context(camera_capture, camera_fps, webcam_execution_thread_count, capture_resolution, raw_frame_listener, camera_switch, idle_listener, stop_event)
	finally:
		set_app_context_override(None)


def schedule_stream_frames_in_context(camera_capture : cv2.VideoCapture, camera_fps : Fps, webcam_execution_thread_count : Optional[int], capture_resolution : Optional[Resolution], raw_frame_listener : Optional[Callable[[VisionFrame], None]], camera_switch : Optional[CameraSwitch], idle_listener : Optional[Callable[[], bool]], stop_event : threading.Event) -> Iterator[Tuple[VisionFrame, float, bool, Dict[str, Any]]]:
	webcam_execution_thread_count = webcam_execution_thread_count or state_manager.get_item('execution_thread_count')
	# Published so the overlay can name the count the stream actually runs on instead of the global
	# execution thread count, which no webcam frame is processed with
	state_manager.set_item('webcam_execution_thread_count', webcam_execution_thread_count)
	max_queue_size = max(1, webcam_execution_thread_count)
	processor_names = list(state_manager.get_item('processors') or [])
	source_face_groups_version = state_manager.get_item('source_face_groups_version')
	source_face_groups = resolve_source_face_groups()
	processor_modules = validate_stream_processors(processor_names)
	stream_face_sets = prepare_stream_face_sets(processor_modules, source_face_groups)
	source_face_name, source_vision_frames, processor_stream_inputs = resolve_stream_face_set(stream_face_sets, state_manager.get_item('source_face_name'))
	stream_vision_mask = None
	face_swapper_model = state_manager.get_item('face_swapper_model')
	face_swapper_weight = state_manager.get_item('face_swapper_weight')
	source_audio_frame = create_empty_audio_frame()
	source_voice_frame = create_empty_audio_frame()
	source_audio_frame.setflags(write = False)
	source_voice_frame.setflags(write = False)

	face_analysis_features = collect_stream_face_analysis_features(processor_modules)

	frame_index = 0
	nsfw_frame_index = 0
	has_logged_capture_downscale = False
	is_stream_bypassed = False
	is_stream_warmed = False
	stream_frame_cursor = StreamFrameCursor()
	idle_listener_time = time.perf_counter()

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		# Add +1 to max_workers to accommodate the background NSFW analysis without stalling frame processing
		executor = ThreadPoolExecutor(max_workers = webcam_execution_thread_count + 1)
		capture_thread = CameraCaptureThread(camera_capture)
		# Assigned rather than passed, so the thread keeps its one argument constructor
		capture_thread.raw_frame_listener = raw_frame_listener
		capture_thread.camera_switch = camera_switch
		# A camera frame or a finished frame sets it, so an idle scheduler wakes at once for either
		wake_event = threading.Event()
		capture_thread.frame_event = wake_event
		capture_thread.start()

		try:
			# Sizes the detector's session pool and turns on its angle fan out for as long as this
			# stream runs
			set_stream_worker_count(webcam_execution_thread_count)
			# Zeroed here so the drift reported for this stream is measured from its own first
			# reading rather than from a previous one
			reset_device_memory_metrics()
			futures = []
			discarded_futures = []
			analysis_future = None

			while capture_thread.running and not stop_event.is_set():
				# Cleared before anything is looked at, so a camera frame or a finished frame that lands
				# from here on wakes the wait below instead of being slept through
				wake_event.clear()
				# Throttled to once a second inside the sampler, because this loop also runs on every
				# wake up of an idle scheduler
				sample_device_memory()
				discarded_futures = [ future for future in discarded_futures if not future.done() ]
				skipping_mode = state_manager.get_item('webcam_frame_skipping') or 'adaptive'
				current_processor_names = list(state_manager.get_item('processors') or [])
				if current_processor_names != processor_names:
					processor_names = current_processor_names
					processor_modules = validate_stream_processors(processor_names)
					stream_face_sets = prepare_stream_face_sets(processor_modules, source_face_groups)
					source_face_name, source_vision_frames, processor_stream_inputs = resolve_stream_face_set(stream_face_sets, source_face_name)
					face_analysis_features = collect_stream_face_analysis_features(processor_modules)
					is_stream_warmed = False

				# 1. Preserve ordered output normally; adaptive mode abandons stale work still running
				# once a newer result is ready
				if skipping_mode == 'adaptive':
					completed_indices = [ index for index, (_, future) in enumerate(futures) if future.done() ]
					if completed_indices:
						latest_completed_index = completed_indices[-1]
						latest_frame_index, _ = futures[latest_completed_index]
						# Published before the stale futures are abandoned, so the ones still running
						# can give up at their next processor instead of finishing a frame that has
						# already been overtaken
						stream_frame_cursor.yielded_frame_index = latest_frame_index
						drained_futures = futures[:latest_completed_index + 1]
						futures = futures[latest_completed_index + 1:]
						for _, drained_future in drained_futures:
							# Only work that is still running is abandoned. Dropping a frame that has
							# already finished buys no latency, because the newest result is delivered
							# in this same pass either way, and it costs a frame that was fully paid for
							if not drained_future.done():
								if not drained_future.cancel():
									discarded_futures.append(drained_future)
								continue
							stream_result = drained_future.result()
							if stream_result:
								capture_vision_frame, capture_time, timing = stream_result
								progress.update()
								yield capture_vision_frame, capture_time, False, timing
				else:
					while futures and futures[0][1].done():
						oldest_frame_index, oldest_future = futures.pop(0)
						stream_frame_cursor.yielded_frame_index = oldest_frame_index
						stream_result = oldest_future.result()
						if stream_result:
							capture_vision_frame, capture_time, timing = stream_result
							progress.update()
							yield capture_vision_frame, capture_time, False, timing

				# 2. Read the latest frame from the camera thread, but only when a worker can take it.
				# Taken while every worker was busy, it was dropped, and a worker that freed up a moment
				# later then waited a whole camera interval for the next one; left in the camera's
				# latest-wins queue, it is there for that worker. Raw passthrough uses no worker
				is_stream_bypass_requested = bool(state_manager.get_item('webcam_stream_bypass'))
				queue_item = None
				if is_stream_bypass_requested or has_stream_capacity(count_pending_futures(futures, discarded_futures), len(futures), max_queue_size):
					try:
						queue_item = capture_thread.frame_queue.get(timeout = 0)
					except queue.Empty:
						queue_item = None
				if queue_item is None:
					# While no frame comes (a camera being reopened or switched to) the caller's loop
					# body never runs, so its window keys would go unread: Ctrl+0-9 is exactly what
					# gets out of a camera that is gone. Returning False ends the stream
					if idle_listener and time.perf_counter() - idle_listener_time > IDLE_LISTENER_INTERVAL:
						idle_listener_time = time.perf_counter()
						if not idle_listener():
							break
					wake_event.wait(STREAM_WAKE_TIMEOUT)
					continue
				if len(queue_item) == 3:
					capture_time, capture_read_end, capture_vision_frame = queue_item
				else:
					capture_time, capture_vision_frame = queue_item
					capture_read_end = capture_time
				idle_listener_time = time.perf_counter()

				# The camera may negotiate a mode larger than the one requested, so bring the frame down
				# to the working resolution before anything processes it
				if capture_resolution and is_vision_frame(capture_vision_frame):
					capture_height, capture_width = capture_vision_frame.shape[:2]
					if capture_width > capture_resolution[0] or capture_height > capture_resolution[1]:
						if not has_logged_capture_downscale:
							has_logged_capture_downscale = True
							logger.debug('camera delivered ' + str(capture_width) + 'x' + str(capture_height) + ' for requested ' + str(capture_resolution[0]) + 'x' + str(capture_resolution[1]) + ', downscaling to the working resolution', __name__)
						capture_vision_frame = restrict_frame(capture_vision_frame, capture_resolution)

				# Warmed on a real frame at the resolution the workers will see, so each one builds
				# the effect the stream goes on using rather than a second one on its first frame
				if not is_stream_warmed and is_vision_frame(capture_vision_frame):
					is_stream_warmed = True
					warm_stream_workers(executor, webcam_execution_thread_count, processor_modules, capture_vision_frame)

				# 3. Sample content analysis on its configured interval, only copying/submitting the
				# sampled frame. The three detector models cost around 190ms of device time per
				# sample, which is a large share of a stream that is already saturated, so the
				# interval sets how often that is paid rather than whether the check runs
				content_analysis_interval = state_manager.get_item('webcam_content_analysis_interval') or 1
				#nsfw_frame_index += 1
				#if nsfw_frame_index % max(1, int(camera_fps * content_analysis_interval)) == 0 and (analysis_future is None or analysis_future.done()):
				#	analysis_future = executor.submit(analyse_frame_background, capture_vision_frame.copy(), stop_event)

				# 4. Raw passthrough skips the pipeline entirely, which is the whole point of it: the
				# lowest latency the camera can deliver, and the reference the processed output is judged against
				if is_stream_bypass_requested:
					# Work admitted before the switch would otherwise land as a processed frame after
					# it, so it is dropped rather than shown
					if not is_stream_bypassed:
						is_stream_bypassed = True
						for _, stale_future in futures:
							if not stale_future.cancel() and not stale_future.done():
								discarded_futures.append(stale_future)
						futures = []
					if is_vision_frame(capture_vision_frame):
						bypass_time = time.perf_counter()
						timing = { 'capture_read_start': capture_time, 'capture_read_end': capture_read_end, 'scheduler_admitted': bypass_time, 'frame_interval_ms': 1000.0 / max(1, float(camera_fps)), 'processor_times': {}, 'processing_started': bypass_time, 'processing_finished': bypass_time }
						progress.update()
						yield capture_vision_frame, capture_time, False, timing
					continue
				is_stream_bypassed = False

				# 5. Process frame or apply temporal skipping to sustain target FPS
				if is_vision_frame(capture_vision_frame):
					if capture_vision_frame.ndim == 3 and capture_vision_frame.shape[2] == 3 and (stream_vision_mask is None or stream_vision_mask.shape != capture_vision_frame.shape[:2]):
						stream_vision_mask = numpy.full(capture_vision_frame.shape[:2], 255, dtype = numpy.uint8)
						stream_vision_mask.setflags(write = False)

					# A bumped version means faces were uploaded or cleared, so the set is reconciled
					# before the selection below is resolved against it
					current_source_face_groups_version = state_manager.get_item('source_face_groups_version')
					if current_source_face_groups_version != source_face_groups_version:
						source_face_groups_version = current_source_face_groups_version
						source_face_groups = resolve_source_face_groups()
						stream_face_sets = update_stream_face_sets(stream_face_sets, processor_modules, source_face_groups)
						source_face_name, source_vision_frames, processor_stream_inputs = resolve_stream_face_set(stream_face_sets, source_face_name)

					# Switching the selected face only rebinds the inputs prepared at startup, so it costs
					# nothing beyond this lookup and takes effect on the next frame
					current_source_face_name = state_manager.get_item('source_face_name')
					if current_source_face_name != source_face_name and current_source_face_name in stream_face_sets:
						source_face_name, source_vision_frames, processor_stream_inputs = resolve_stream_face_set(stream_face_sets, current_source_face_name)

					# Refresh the cached source inputs if the face swapper model changed mid-stream,
					# since the cached embedding/prepared-frame format is tied to the previous model
					current_face_swapper_model = state_manager.get_item('face_swapper_model')
					current_face_swapper_weight = state_manager.get_item('face_swapper_weight')
					face_swapper_model_changed = current_face_swapper_model != face_swapper_model
					face_swapper_weight_changed = current_face_swapper_weight != face_swapper_weight
					if face_swapper_model_changed:
						face_swapper_model = current_face_swapper_model
						stream_face_sets = prepare_stream_face_sets(processor_modules, source_face_groups)
						source_face_name, source_vision_frames, processor_stream_inputs = resolve_stream_face_set(stream_face_sets, source_face_name)
					if face_swapper_weight_changed:
						face_swapper_weight = current_face_swapper_weight
					if face_swapper_model_changed or face_swapper_weight_changed:
						face_analysis_features = collect_stream_face_analysis_features(processor_modules)

					# Only unfinished work occupies a worker, so results awaiting their turn in the ordered
					# output must not block submission and leave the executor idle behind a slow frame
					pending_total = count_pending_futures(futures, discarded_futures)

					frame_index += 1
					should_skip = False
					if skipping_mode == '1-in-2' and frame_index % 2 != 0:
						should_skip = True
					elif skipping_mode == '1-in-3' and frame_index % 3 != 0:
						should_skip = True
					elif skipping_mode == 'adaptive' and pending_total >= max_queue_size:
						should_skip = True

					if not should_skip and has_stream_capacity(pending_total, len(futures), max_queue_size):
						timing = { 'capture_read_start': capture_time, 'capture_read_end': capture_read_end, 'scheduler_admitted': time.perf_counter(), 'frame_interval_ms': 1000.0 / max(1, float(camera_fps)), 'processor_times': {} }
						# Only adaptive mode drops a frame once it is in flight, so only adaptive
						# mode hands the worker a cursor to give up against
						future = executor.submit(process_stream_frame, source_vision_frames, capture_vision_frame, capture_time, processor_modules, processor_stream_inputs, stream_vision_mask, source_audio_frame, source_voice_frame, face_analysis_features, timing, frame_index, stream_frame_cursor if skipping_mode == 'adaptive' else None)
						# Wakes the scheduler the moment this frame is done, instead of on its next poll
						future.add_done_callback(lambda _: wake_event.set())
						futures.append((frame_index, future))

			# Stopped by the caller, nobody takes the frames still in flight, so they are not waited for;
			# a camera that ended on its own still hands over what was in flight, in order. The camera is
			# released in the finally below, after the capture thread has been told to stop: released
			# here, before that, the capture thread could still take it for lost and start reopening it
			if not stop_event.is_set():
				for _, future in futures:
					stream_result = future.result()
					if stream_result:
						capture_vision_frame, capture_time, timing = stream_result
						progress.update()
						yield capture_vision_frame, capture_time, False, timing
		finally:
			set_stream_worker_count(0)
			capture_thread.stop()
			if hasattr(camera_capture, 'release'):
				camera_capture.release()
			# A camera switched to mid stream is the thread's capture now, not the one passed in
			switched_capture = getattr(capture_thread, 'camera_capture', camera_capture)
			if switched_capture is not None and switched_capture is not camera_capture and hasattr(switched_capture, 'release'):
				switched_capture.release()
			capture_thread.join(timeout = 1.0)
			if capture_thread.is_alive():
				logger.warn(translator.get('stream_camera_capture_hung'), __name__)
			executor.shutdown(wait=False)


# Returns None for a frame it gave up on, which the scheduler was going to drop anyway
def process_stream_frame(source_vision_frames : List[VisionFrame], target_vision_frame : VisionFrame, capture_time : float, processor_modules : List[ModuleType], processor_stream_inputs : Optional[Dict[str, Dict[str, Any]]] = None, stream_vision_mask : Optional[Mask] = None, source_audio_frame : Optional[AudioFrame] = None, source_voice_frame : Optional[AudioFrame] = None, face_analysis_features : Optional[Set[str]] = None, timing : Optional[Dict[str, Any]] = None, frame_index : int = 0, stream_frame_cursor : Optional[StreamFrameCursor] = None) -> Any:
	if source_audio_frame is None:
		source_audio_frame = create_empty_audio_frame()
	if source_voice_frame is None:
		source_voice_frame = create_empty_audio_frame()
	temp_vision_frame = target_vision_frame.copy()
	if stream_vision_mask is not None and temp_vision_frame.ndim == 3 and temp_vision_frame.shape[2] == 3 and stream_vision_mask.shape == temp_vision_frame.shape[:2]:
		temp_vision_mask = stream_vision_mask
	else:
		temp_vision_mask = extract_vision_mask(temp_vision_frame)

	# Resolve the app context once instead of walking the stack on every state lookup of this frame
	set_app_context_override(detect_app_context())
	set_paste_in_place(True)
	set_face_analysis_features(face_analysis_features)
	# Published for the whole chain rather than consulted only here, so the processors can give up
	# between their own expensive steps instead of only between one another
	if stream_frame_cursor:
		set_stream_abort(lambda: stream_frame_cursor.has_passed(frame_index))
	try:
		if timing is not None:
			timing['processing_started'] = time.perf_counter()
		# Zeroed per frame on this worker, so what is read back below is this frame's model time only
		reset_inference_timer()
		begin_face_selection_context()
		# Read once for the frame rather than per processor, and never used to rebuild the chain:
		# a bypassed processor keeps its inference pool loaded, so toggling it is instant and the
		# frames on either side of the toggle are comparable
		bypassed_processors = state_manager.get_item('webcam_bypassed_processors') or []
		for processor_module in processor_modules:
			if resolve_processor_name(processor_module) in bypassed_processors:
				continue
			# Checked between processors rather than only up front, because the frame is overtaken
			# while the chain runs: this is what keeps a stale frame from paying for the swapper and
			# everything after it
			abort_stale_stream_frame()
			logger.disable()
			processor_started = time.perf_counter()
			processor_inputs =\
			{
				'source_vision_frames': source_vision_frames,
				'source_audio_frame': source_audio_frame,
				'source_voice_frame': source_voice_frame,
				'target_vision_frames': [ target_vision_frame ],
				'temp_vision_frame': temp_vision_frame,
				'temp_vision_mask': temp_vision_mask
			}
			if processor_stream_inputs:
				processor_inputs.update(processor_stream_inputs.get(processor_module.__name__, {}))
			temp_vision_frame, temp_vision_mask = processor_module.process_frame(processor_inputs)
			if timing is not None:
				timing['processor_times'][processor_module.__name__] = (time.perf_counter() - processor_started) * 1000
			logger.enable()
	except StreamAborted:
		return None
	finally:
		logger.enable()
		end_face_selection_context()
		set_face_analysis_features(None)
		set_paste_in_place(False)
		set_stream_abort(None)
		set_app_context_override(None)

	if timing is not None:
		timing['processing_finished'] = time.perf_counter()
		timing['model_ms'], timing['model_runs'] = get_inference_timer()
		return temp_vision_frame, capture_time, timing
	return temp_vision_frame, capture_time


def collect_stream_face_analysis_features(processor_modules : List[ModuleType]) -> Set[str]:
	face_analysis_features : Set[str] = set()
	for processor_module in processor_modules:
		if hasattr(processor_module, 'get_stream_face_analysis_features'):
			face_analysis_features.update(processor_module.get_stream_face_analysis_features())

	if state_manager.get_item('face_selector_mode') == 'reference':
		face_analysis_features.add('embedding')
	if state_manager.get_item('face_selector_gender') or state_manager.get_item('face_selector_race'):
		face_analysis_features.add('demographics')
	face_selector_age_start = state_manager.get_item('face_selector_age_start') or 0
	face_selector_age_end = state_manager.get_item('face_selector_age_end') or 100
	if face_selector_age_start > 0 or face_selector_age_end < 100:
		face_analysis_features.add('demographics')
	return face_analysis_features


class WindowsVirtualCameraStream:
	def __init__(self, resolution : str, fps : int):
		import pyvirtualcam
		from facefusion.vision import unpack_resolution
		width, height = unpack_resolution(resolution)
		self.width = width
		self.height = height
		try:
			self.cam = pyvirtualcam.Camera(width = width, height = height, fps = fps, device = "facefusion")
		except Exception:
			self.cam = pyvirtualcam.Camera(width = width, height = height, fps = fps)
		self.stdin = self

	def write(self, data : bytes) -> None:
		import numpy
		frame = numpy.frombuffer(data, dtype = numpy.uint8).reshape((self.height, self.width, 3))
		self.cam.send(frame)

	def __del__(self) -> None:
		self.close()

	def close(self) -> None:
		if hasattr(self, 'cam') and self.cam:
			self.cam.close()
			self.cam = None


def open_stream(stream_mode : StreamMode, stream_resolution : str, stream_fps : Fps) -> Any:
	if stream_mode == 'v4l2' and is_windows():
		return WindowsVirtualCameraStream(stream_resolution, stream_fps)

	commands = ffmpeg_builder.chain(
		ffmpeg_builder.capture_video(),
		ffmpeg_builder.set_media_resolution(stream_resolution),
		ffmpeg_builder.set_input_fps(stream_fps)
	)

	if stream_mode == 'udp':
		commands.extend([
			'-fflags', 'nobuffer',
			'-flags', 'low_delay'
		])
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('udp'))
		commands.extend(ffmpeg_builder.set_stream_quality(2000))
		commands.extend([
			'-c:v', 'libx264',
			'-preset', 'ultrafast',
			'-tune', 'zerolatency',
			'-g', '30',
			'-pix_fmt', 'yuv420p'
		])
		commands.extend(ffmpeg_builder.set_output('udp://127.0.0.1:27000?pkt_size=1316&connect=0'))

	if stream_mode == 'v4l2':
		device_directory_path = '/sys/devices/virtual/video4linux'
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('v4l2'))

		if is_directory(device_directory_path):
			device_names = os.listdir(device_directory_path)

			for device_name in device_names:
				device_path = '/dev/' + device_name
				commands.extend(ffmpeg_builder.set_output(device_path))

		else:
			logger.error(translator.get('stream_not_loaded').format(stream_mode = stream_mode), __name__)

	return open_ffmpeg(commands, capture_stdout = False)
