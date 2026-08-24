from typing import Any

import cv2

from facefusion import logger, state_manager
from facefusion.camera_manager import clear_camera_pool, get_local_camera_capture
from facefusion.overlay import PERFORMANCE_OVERLAY
from facefusion.streamer import multi_process_capture, open_stream
from facefusion.stream_writer import LatestFrameWriter
from facefusion.vision import fit_cover_frame, unpack_resolution
from facefusion.webcam_config import load_webcam_config


def run() -> int:
	webcam_config = load_webcam_config(state_manager.get_item('webcam_config'))
	webcam_device_id = webcam_config.get('device_id')
	webcam_mode = webcam_config.get('headless_mode')
	webcam_resolution = webcam_config.get('resolution')
	webcam_fps = webcam_config.get('fps')
	webcam_execution_thread_count = webcam_config.get('execution_thread_count')
	stream_writer : Any = None

	if webcam_mode not in [ 'udp', 'v4l2' ]:
		logger.error('headless webcam mode requires "udp" or "v4l2" output', __name__)
		return 2

	state_manager.sync_state()
	webcam_width, webcam_height = unpack_resolution(webcam_resolution)
	camera_capture = get_local_camera_capture(webcam_device_id, webcam_width, webcam_height, webcam_fps)
	if not camera_capture or not camera_capture.isOpened():
		logger.error('webcam device could not be opened', __name__)
		return 1

	stream_writer = LatestFrameWriter(lambda: open_stream(webcam_mode, webcam_resolution, webcam_fps), webcam_width, webcam_height, webcam_mode == 'v4l2')
	stream_writer.start()

	try:
		for capture_vision_frame, capture_time, is_duplicate in multi_process_capture(camera_capture, webcam_fps, webcam_execution_thread_count):
			capture_vision_frame = cv2.cvtColor(capture_vision_frame, cv2.COLOR_BGR2RGB)
			capture_vision_frame = fit_cover_frame(capture_vision_frame, (webcam_width, webcam_height))
			overlay_mode = state_manager.get_item('webcam_performance_overlay')
			if overlay_mode in [ 'simple', 'advanced' ]:
				capture_vision_frame = PERFORMANCE_OVERLAY.render(capture_vision_frame, capture_time, overlay_mode, is_duplicate)
			stream_writer.submit(capture_vision_frame)
	finally:
		clear_camera_pool()
		if stream_writer:
			stream_writer.close()

	return 0
