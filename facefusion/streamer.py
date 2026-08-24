import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import cv2
import numpy
from tqdm import tqdm

from facefusion import ffmpeg_builder, logger, state_manager, translator
from facefusion.app_context import detect_app_context, set_app_context_override
from facefusion.audio import create_empty_audio_frame
from facefusion.common_helper import is_windows
from facefusion.content_analyser import analyse_frame
from facefusion.face_creator import set_face_analysis_features
from facefusion.face_helper import set_paste_in_place
from facefusion.face_selector import begin_face_selection_context, end_face_selection_context
from facefusion.ffmpeg import open_ffmpeg
from facefusion.filesystem import is_directory
from facefusion.processors.core import get_processors_modules
from facefusion.types import AudioFrame, Fps, Mask, StreamMode, VisionFrame
from facefusion.vision import extract_vision_mask, is_vision_frame, read_static_images

CONTENT_ANALYSIS_METRICS_LOCK = threading.Lock()
CONTENT_ANALYSIS_METRICS = { 'runs': 0, 'last_ms': 0.0, 'max_ms': 0.0 }


class CameraCaptureThread(threading.Thread):
	MAX_CONSECUTIVE_READ_FAILURES = 30
	READ_FAILURE_RETRY_DELAY = 0.01

	def __init__(self, camera_capture: cv2.VideoCapture):
		super().__init__()
		self.camera_capture = camera_capture
		self.frame_queue = queue.Queue(maxsize=1)
		self.running = True
		self.daemon = True

	def run(self):
		consecutive_read_failures = 0
		try:
			while self.running and self.camera_capture.isOpened():
				capture_time = time.perf_counter()
				ret, frame = self.camera_capture.read()
				capture_read_end = time.perf_counter()
				if not ret:
					consecutive_read_failures += 1
					if not self.camera_capture.isOpened() or consecutive_read_failures >= self.MAX_CONSECUTIVE_READ_FAILURES:
						break
					time.sleep(self.READ_FAILURE_RETRY_DELAY)
					continue
				consecutive_read_failures = 0
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
		finally:
			self.running = False

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


def prepare_stream_processors(processor_names : List[str], source_vision_frames : List[VisionFrame]) -> Tuple[List[ModuleType], Dict[str, Dict[str, Any]]]:
	processor_modules = get_processors_modules(processor_names)
	validated_processor_modules = []
	processor_stream_inputs = {}

	for processor_module in processor_modules:
		logger.disable()
		try:
			is_processor_ready = processor_module.pre_process('stream')
		finally:
			logger.enable()
		if is_processor_ready:
			validated_processor_modules.append(processor_module)
			if hasattr(processor_module, 'prepare_stream_inputs'):
				processor_stream_inputs[processor_module.__name__] = processor_module.prepare_stream_inputs(source_vision_frames)

	return validated_processor_modules, processor_stream_inputs


def has_stream_capacity(pending_total : int, buffered_total : int, max_queue_size : int) -> bool:
	return pending_total < max_queue_size and buffered_total < max_queue_size


def multi_process_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps, webcam_execution_thread_count : Optional[int] = None) -> Iterator[Tuple[VisionFrame, float, bool, Dict[str, float]]]:
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	webcam_execution_thread_count = webcam_execution_thread_count or state_manager.get_item('execution_thread_count')
	max_queue_size = max(1, webcam_execution_thread_count)
	processor_names = list(state_manager.get_item('processors') or [])
	processor_modules, processor_stream_inputs = prepare_stream_processors(processor_names, source_vision_frames)
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

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		# Add +1 to max_workers to accommodate the background NSFW analysis without stalling frame processing
		executor = ThreadPoolExecutor(max_workers = webcam_execution_thread_count + 1)
		capture_thread = CameraCaptureThread(camera_capture)
		capture_thread.start()
		stop_event = threading.Event()
		
		try:
			set_app_context_override(detect_app_context())
			futures = []
			discarded_futures = []
			analysis_future = None

			while capture_thread.running and not stop_event.is_set():
				discarded_futures = [ future for future in discarded_futures if not future.done() ]
				skipping_mode = state_manager.get_item('webcam_frame_skipping') or 'adaptive'
				current_processor_names = list(state_manager.get_item('processors') or [])
				if current_processor_names != processor_names:
					processor_names = current_processor_names
					processor_modules, processor_stream_inputs = prepare_stream_processors(processor_names, source_vision_frames)
					face_analysis_features = collect_stream_face_analysis_features(processor_modules)

				# 1. Preserve ordered output normally; adaptive mode drops stale work when a newer result is ready
				if skipping_mode == 'adaptive':
					completed_indices = [ index for index, (_, future) in enumerate(futures) if future.done() ]
					if completed_indices:
						latest_completed_index = completed_indices[-1]
						_, latest_future = futures[latest_completed_index]
						for _, stale_future in futures[:latest_completed_index]:
							if not stale_future.cancel() and not stale_future.done():
								discarded_futures.append(stale_future)
						futures = futures[latest_completed_index + 1:]
						capture_vision_frame, capture_time, timing = latest_future.result()
						progress.update()
						yield capture_vision_frame, capture_time, False, timing
				else:
					while futures and futures[0][1].done():
						_, oldest_future = futures.pop(0)
						capture_vision_frame, capture_time, timing = oldest_future.result()
						progress.update()
						yield capture_vision_frame, capture_time, False, timing

				# 2. Read the latest frame from the camera thread (non-blocking yield delay)
				try:
					queue_item = capture_thread.frame_queue.get(timeout=0.005)
					if len(queue_item) == 3:
						capture_time, capture_read_end, capture_vision_frame = queue_item
					else:
						capture_time, capture_vision_frame = queue_item
						capture_read_end = capture_time
				except queue.Empty:
					continue

				# 3. Sample NSFW analysis to ~once per second, only copying/submitting the sampled frame
				nsfw_frame_index += 1
				if nsfw_frame_index % max(1, int(camera_fps)) == 0 and (analysis_future is None or analysis_future.done()):
					analysis_future = executor.submit(analyse_frame_background, capture_vision_frame.copy(), stop_event)

				# 4. Process frame or apply temporal skipping to sustain target FPS
				if is_vision_frame(capture_vision_frame):
					if capture_vision_frame.ndim == 3 and capture_vision_frame.shape[2] == 3 and (stream_vision_mask is None or stream_vision_mask.shape != capture_vision_frame.shape[:2]):
						stream_vision_mask = numpy.full(capture_vision_frame.shape[:2], 255, dtype = numpy.uint8)
						stream_vision_mask.setflags(write = False)

					# Refresh the cached source inputs if the face swapper model changed mid-stream,
					# since the cached embedding/prepared-frame format is tied to the previous model
					current_face_swapper_model = state_manager.get_item('face_swapper_model')
					current_face_swapper_weight = state_manager.get_item('face_swapper_weight')
					face_swapper_model_changed = current_face_swapper_model != face_swapper_model
					face_swapper_weight_changed = current_face_swapper_weight != face_swapper_weight
					if face_swapper_model_changed:
						face_swapper_model = current_face_swapper_model
						for processor_module in processor_modules:
							if hasattr(processor_module, 'prepare_stream_inputs'):
								processor_stream_inputs[processor_module.__name__] = processor_module.prepare_stream_inputs(source_vision_frames)
					if face_swapper_weight_changed:
						face_swapper_weight = current_face_swapper_weight
					if face_swapper_model_changed or face_swapper_weight_changed:
						face_analysis_features = collect_stream_face_analysis_features(processor_modules)

					# Only unfinished work occupies a worker, so results awaiting their turn in the ordered
					# output must not block submission and leave the executor idle behind a slow frame
					pending_total = len(discarded_futures)
					for _, future in futures:
						if not future.done():
							pending_total += 1

					frame_index += 1
					should_skip = False
					if skipping_mode == '1-in-2' and frame_index % 2 != 0:
						should_skip = True
					elif skipping_mode == '1-in-3' and frame_index % 3 != 0:
						should_skip = True
					elif skipping_mode == 'adaptive' and pending_total >= max_queue_size:
						should_skip = True

					if not should_skip and has_stream_capacity(pending_total, len(futures), max_queue_size):
						timing = { 'capture_read_start': capture_time, 'capture_read_end': capture_read_end, 'scheduler_admitted': time.perf_counter(), 'frame_interval_ms': 1000.0 / max(1, float(camera_fps)) }
						future = executor.submit(process_stream_frame, source_vision_frames, capture_vision_frame, capture_time, processor_modules, processor_stream_inputs, stream_vision_mask, source_audio_frame, source_voice_frame, face_analysis_features, timing)
						futures.append((frame_index, future))

			if stop_event.is_set():
				camera_capture.release()

			# Yield any remaining frames in order
			for _, future in futures:
				capture_vision_frame, capture_time, timing = future.result()
				progress.update()
				yield capture_vision_frame, capture_time, False, timing
		finally:
			set_app_context_override(None)
			capture_thread.stop()
			if hasattr(camera_capture, 'release'):
				camera_capture.release()
			capture_thread.join(timeout = 1.0)
			if capture_thread.is_alive():
				logger.warn(translator.get('stream_camera_capture_hung'), __name__)
			executor.shutdown(wait=False)


def process_stream_frame(source_vision_frames : List[VisionFrame], target_vision_frame : VisionFrame, capture_time : float, processor_modules : List[ModuleType], processor_stream_inputs : Optional[Dict[str, Dict[str, Any]]] = None, stream_vision_mask : Optional[Mask] = None, source_audio_frame : Optional[AudioFrame] = None, source_voice_frame : Optional[AudioFrame] = None, face_analysis_features : Optional[Set[str]] = None, timing : Optional[Dict[str, float]] = None) -> Any:
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
	try:
		if timing is not None:
			timing['processing_started'] = time.perf_counter()
		begin_face_selection_context()
		for processor_module in processor_modules:
			logger.disable()
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
			logger.enable()
	finally:
		logger.enable()
		end_face_selection_context()
		set_face_analysis_features(None)
		set_paste_in_place(False)
		set_app_context_override(None)

	if timing is not None:
		timing['processing_finished'] = time.perf_counter()
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
