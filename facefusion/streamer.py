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
from facefusion.audio import create_empty_audio_frame
from facefusion.common_helper import is_windows
from facefusion.content_analyser import analyse_frame
from facefusion.face_creator import set_face_analysis_features
from facefusion.face_selector import begin_face_selection_context, end_face_selection_context
from facefusion.ffmpeg import open_ffmpeg
from facefusion.filesystem import is_directory
from facefusion.processors.core import get_processors_modules
from facefusion.types import AudioFrame, Fps, Mask, StreamMode, VisionFrame
from facefusion.vision import extract_vision_mask, is_vision_frame, read_static_images


class CameraCaptureThread(threading.Thread):
	def __init__(self, camera_capture: cv2.VideoCapture):
		super().__init__()
		self.camera_capture = camera_capture
		self.frame_queue = queue.Queue(maxsize=1)
		self.running = True
		self.daemon = True

	def run(self):
		while self.running and self.camera_capture.isOpened():
			capture_time = time.perf_counter()
			ret, frame = self.camera_capture.read()
			if not ret:
				self.running = False
				break
			if self.frame_queue.full():
				try:
					self.frame_queue.get_nowait()
				except queue.Empty:
					pass
			self.frame_queue.put((capture_time, frame))

	def stop(self):
		self.running = False

def analyse_frame_background(vision_frame: VisionFrame, stop_event: threading.Event):
	if analyse_frame(vision_frame):
		stop_event.set()

def multi_process_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps) -> Iterator[Tuple[VisionFrame, float]]:
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	max_queue_size = max(1, state_manager.get_item('execution_thread_count'))
	processor_modules = get_processors_modules(state_manager.get_item('processors'))
	processor_stream_inputs = {}
	stream_vision_mask = None
	face_swapper_model = state_manager.get_item('face_swapper_model')
	source_audio_frame = create_empty_audio_frame()
	source_voice_frame = create_empty_audio_frame()
	source_audio_frame.setflags(write = False)
	source_voice_frame.setflags(write = False)

	# Pre-validate processors once before streaming starts to avoid per-frame disk I/O and face detection
	validated_processor_modules = []
	for processor_module in processor_modules:
		logger.disable()
		is_processor_ready = processor_module.pre_process('stream')
		logger.enable()
		if is_processor_ready:
			validated_processor_modules.append(processor_module)
			if hasattr(processor_module, 'prepare_stream_inputs'):
				processor_stream_inputs[processor_module.__name__] = processor_module.prepare_stream_inputs(source_vision_frames)
	processor_modules = validated_processor_modules
	face_analysis_features = collect_stream_face_analysis_features(processor_modules)

	frame_index = 0
	nsfw_frame_index = 0
	last_processed_frame = None

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		# Add +1 to max_workers to accommodate the background NSFW analysis without stalling frame processing
		executor = ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count') + 1)
		capture_thread = CameraCaptureThread(camera_capture)
		capture_thread.start()
		stop_event = threading.Event()
		
		try:
			futures = []
			discarded_futures = []

			while capture_thread.running and not stop_event.is_set():
				discarded_futures = [ future for future in discarded_futures if not future.done() ]
				skipping_mode = state_manager.get_item('webcam_frame_skipping') or 'disabled'

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
						capture_vision_frame, capture_time = latest_future.result()
						last_processed_frame = capture_vision_frame
						progress.update()
						yield capture_vision_frame, capture_time
				else:
					while futures and futures[0][1].done():
						_, oldest_future = futures.pop(0)
						capture_vision_frame, capture_time = oldest_future.result()
						last_processed_frame = capture_vision_frame
						progress.update()
						yield capture_vision_frame, capture_time

				# 2. Read the latest frame from the camera thread (non-blocking yield delay)
				try:
					capture_time, capture_vision_frame = capture_thread.frame_queue.get(timeout=0.005)
				except queue.Empty:
					continue

				# 3. Sample NSFW analysis to ~once per second, only copying/submitting the sampled frame
				nsfw_frame_index += 1
				if nsfw_frame_index % max(1, int(camera_fps)) == 0:
					executor.submit(analyse_frame_background, capture_vision_frame.copy(), stop_event)

				# 4. Process frame or apply temporal skipping to sustain target FPS
				if is_vision_frame(capture_vision_frame):
					if capture_vision_frame.ndim == 3 and capture_vision_frame.shape[2] == 3 and (stream_vision_mask is None or stream_vision_mask.shape != capture_vision_frame.shape[:2]):
						stream_vision_mask = numpy.full(capture_vision_frame.shape[:2], 255, dtype = numpy.uint8)
						stream_vision_mask.setflags(write = False)

					# Refresh the cached source inputs if the face swapper model changed mid-stream,
					# since the cached embedding/prepared-frame format is tied to the previous model
					current_face_swapper_model = state_manager.get_item('face_swapper_model')
					if current_face_swapper_model != face_swapper_model:
						face_swapper_model = current_face_swapper_model
						for processor_module in processor_modules:
							if hasattr(processor_module, 'prepare_stream_inputs'):
								processor_stream_inputs[processor_module.__name__] = processor_module.prepare_stream_inputs(source_vision_frames)
						face_analysis_features = collect_stream_face_analysis_features(processor_modules)

					frame_index += 1
					should_skip = False
					if skipping_mode == '1-in-2' and frame_index % 2 != 0:
						should_skip = True
					elif skipping_mode == '1-in-3' and frame_index % 3 != 0:
						should_skip = True
					elif skipping_mode == 'adaptive' and len(futures) + len(discarded_futures) >= max_queue_size:
						should_skip = True

					if should_skip and last_processed_frame is not None:
						yield last_processed_frame, capture_time
					elif len(futures) + len(discarded_futures) < max_queue_size:
						future = executor.submit(process_stream_frame, source_vision_frames, capture_vision_frame, capture_time, processor_modules, processor_stream_inputs, stream_vision_mask, source_audio_frame, source_voice_frame, face_analysis_features)
						futures.append((frame_index, future))

			if stop_event.is_set():
				camera_capture.release()

			# Yield any remaining frames in order
			for _, future in futures:
				capture_vision_frame, capture_time = future.result()
				progress.update()
				yield capture_vision_frame, capture_time
		finally:
			capture_thread.stop()
			capture_thread.join(timeout = 1.0)
			if capture_thread.is_alive():
				logger.warn(translator.get('stream_camera_capture_hung'), __name__)
			executor.shutdown(wait=False)


def process_stream_frame(source_vision_frames : List[VisionFrame], target_vision_frame : VisionFrame, capture_time : float, processor_modules : List[ModuleType], processor_stream_inputs : Optional[Dict[str, Dict[str, Any]]] = None, stream_vision_mask : Optional[Mask] = None, source_audio_frame : Optional[AudioFrame] = None, source_voice_frame : Optional[AudioFrame] = None, face_analysis_features : Optional[Set[str]] = None) -> Tuple[VisionFrame, float]:
	if source_audio_frame is None:
		source_audio_frame = create_empty_audio_frame()
	if source_voice_frame is None:
		source_voice_frame = create_empty_audio_frame()
	temp_vision_frame = target_vision_frame.copy()
	if stream_vision_mask is not None and temp_vision_frame.ndim == 3 and temp_vision_frame.shape[2] == 3 and stream_vision_mask.shape == temp_vision_frame.shape[:2]:
		temp_vision_mask = stream_vision_mask
	else:
		temp_vision_mask = extract_vision_mask(temp_vision_frame)

	set_face_analysis_features(face_analysis_features)
	begin_face_selection_context()
	try:
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
		self.cam.sleep_until_next_frame()

	def __del__(self) -> None:
		if hasattr(self, 'cam') and self.cam:
			self.cam.close()


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

	return open_ffmpeg(commands)
