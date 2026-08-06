import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List

import cv2
from tqdm import tqdm

from facefusion import ffmpeg_builder, logger, state_manager, translator
from facefusion.audio import create_empty_audio_frame
from facefusion.common_helper import is_windows
from facefusion.content_analyser import analyse_stream
from facefusion.ffmpeg import open_ffmpeg
from facefusion.filesystem import is_directory
from facefusion.processors.core import get_processors_modules
from facefusion.types import Fps, StreamMode, VisionFrame
from facefusion.vision import extract_vision_mask, is_vision_frame, read_static_images


def multi_process_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps) -> Iterator[VisionFrame]:
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	max_queue_size = max(1, state_manager.get_item('execution_thread_count'))

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		executor = ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count'))
		try:
			futures = []

			while camera_capture and camera_capture.isOpened():
				# 1. Yield any completed futures in chronological order
				while futures and futures[0].done():
					oldest_future = futures.pop(0)
					capture_vision_frame = oldest_future.result()
					progress.update()
					yield capture_vision_frame

				# 2. Read the latest frame from the camera to keep the buffer fresh
				_, capture_vision_frame = camera_capture.read()
				if analyse_stream(capture_vision_frame, camera_fps):
					camera_capture.release()
					break

				# 3. Process frame if we have queue capacity, otherwise drop it to prevent delay
				if is_vision_frame(capture_vision_frame):
					if len(futures) < max_queue_size:
						future = executor.submit(process_stream_frame, source_vision_frames, capture_vision_frame)
						futures.append(future)

			# Yield any remaining frames in order
			for future in futures:
				capture_vision_frame = future.result()
				progress.update()
				yield capture_vision_frame
		finally:
			executor.shutdown(wait=False)


def process_stream_frame(source_vision_frames : List[VisionFrame], target_vision_frame : VisionFrame) -> VisionFrame:
	source_audio_frame = create_empty_audio_frame()
	source_voice_frame = create_empty_audio_frame()
	temp_vision_frame = target_vision_frame.copy()
	temp_vision_mask = extract_vision_mask(temp_vision_frame)

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		logger.disable()
		if processor_module.pre_process('stream'):
			logger.enable()
			temp_vision_frame, temp_vision_mask = processor_module.process_frame(
			{
				'source_vision_frames': source_vision_frames,
				'source_audio_frame': source_audio_frame,
				'source_voice_frame': source_voice_frame,
				'target_vision_frames': [ target_vision_frame ],
				'temp_vision_frame': temp_vision_frame,
				'temp_vision_mask': temp_vision_mask
			})
		logger.enable()

	return temp_vision_frame


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
