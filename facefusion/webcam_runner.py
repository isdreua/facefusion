import time
from typing import Any

import cv2

from facefusion import logger, state_manager
from facefusion.camera_manager import clear_camera_pool, get_local_camera_capture
from facefusion.common_helper import get_first
from facefusion.overlay import PERFORMANCE_OVERLAY
from facefusion.streamer import multi_process_capture, open_stream
from facefusion.stream_writer import LatestFrameWriter
from facefusion.vision import fit_cover_frame, pack_resolution, unpack_resolution
from facefusion.webcam_config import DEFAULT_SOURCE_FACE_NAME, WEBCAM_HEADLESS_MODES, get_source_faces, load_webcam_config
from facefusion.window_output import WindowStream, apply_window_intents, has_window_output, log_window_output_requirement


def run() -> int:
	webcam_config = load_webcam_config(state_manager.get_item('webcam_config'))
	webcam_device_id = webcam_config.get('device_id')
	webcam_mode = webcam_config.get('headless_mode')
	webcam_resolution = webcam_config.get('resolution')
	webcam_fps = webcam_config.get('fps')
	webcam_execution_thread_count = webcam_config.get('webcam_execution_thread_count')
	webcam_camera_backend = webcam_config.get('webcam_camera_backend')
	webcam_output_resolution = webcam_config.get('webcam_output_resolution')
	source_face_groups = get_source_faces(webcam_config)
	stream_writer : Any = None
	window_stream = None

	if webcam_mode not in WEBCAM_HEADLESS_MODES:
		logger.error('headless webcam mode requires one of ' + ', '.join(WEBCAM_HEADLESS_MODES) + ' output', __name__)
		return 2
	if webcam_mode == 'window' and not has_window_output():
		log_window_output_requirement()
		return 2

	state_manager.init_item('source_face_groups', source_face_groups)
	state_manager.init_item('webcam_stream_bypass', False)
	state_manager.init_item('source_face_name', webcam_config.get('webcam_source_face') or get_first(list(source_face_groups)) or DEFAULT_SOURCE_FACE_NAME)
	state_manager.sync_state()
	webcam_width, webcam_height = unpack_resolution(webcam_resolution)
	if webcam_output_resolution == 'copy':
		output_resolution = webcam_resolution
		output_width, output_height = webcam_width, webcam_height
	elif webcam_output_resolution.startswith('x'):
		output_scale = int(webcam_output_resolution[1:])
		output_resolution = pack_resolution((webcam_width * output_scale, webcam_height * output_scale))
		output_width, output_height = unpack_resolution(output_resolution)
	else:
		output_resolution = webcam_output_resolution
		output_width, output_height = unpack_resolution(webcam_output_resolution)
	if webcam_width * output_height != output_width * webcam_height:
		logger.warn('webcam output resolution ' + output_resolution + ' does not match the aspect ratio of capture resolution ' + webcam_resolution + ', edges will be cropped', __name__)
	camera_capture = get_local_camera_capture(webcam_device_id, webcam_width, webcam_height, webcam_fps, webcam_camera_backend)
	if not camera_capture or not camera_capture.isOpened():
		logger.error('webcam device could not be opened', __name__)
		return 1

	try:
		# The window is the control channel headless never had: it carries the hotkeys that switch
		# the source face mid run, which no config file can do
		if webcam_mode == 'window':
			window_stream = WindowStream(output_width, output_height, webcam_fps, 'FaceFusion', webcam_config.get('webcam_window_fullscreen'))
			stream_writer = LatestFrameWriter(lambda: window_stream, output_width, output_height, False, webcam_fps)
		else:
			stream_writer = LatestFrameWriter(lambda: open_stream(webcam_mode, output_resolution, webcam_fps), output_width, output_height, webcam_mode == 'v4l2', webcam_fps)
		stream_writer.start()
		for capture_vision_frame, capture_time, is_duplicate, timing in multi_process_capture(camera_capture, webcam_fps, webcam_execution_thread_count, (webcam_width, webcam_height)):
			capture_vision_frame = cv2.cvtColor(capture_vision_frame, cv2.COLOR_BGR2RGB)
			capture_vision_frame = fit_cover_frame(capture_vision_frame, (output_width, output_height))
			timing['post_processing_finished'] = time.perf_counter()
			overlay_mode = state_manager.get_item('webcam_performance_overlay')
			if overlay_mode in [ 'simple', 'advanced' ]:
				capture_vision_frame = PERFORMANCE_OVERLAY.render(capture_vision_frame, capture_time, overlay_mode, is_duplicate, timing)
			stream_writer.submit(capture_vision_frame, timing)
			if window_stream and not apply_window_intents(window_stream):
				break
	finally:
		clear_camera_pool()
		if stream_writer:
			stream_writer.close()
		if window_stream:
			window_stream.close()

	return 0
