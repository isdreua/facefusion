import time
from typing import Any

import cv2

from facefusion import logger, state_manager
from facefusion.camera_manager import CameraSwitch, clear_camera_pool, resolve_camera_window_title
from facefusion.common_helper import get_first
from facefusion.frame_generator import StreamFrameGenerator, resolve_output_fps
from facefusion.overlay import PERFORMANCE_OVERLAY, ControlOverlay
from facefusion.streamer import apply_stream_face_selector_mode, apply_stream_source_paths, multi_process_capture, open_stream
from facefusion.stream_writer import LatestFrameWriter
from facefusion.vision import fit_cover_frame, pack_resolution, unpack_resolution
from facefusion.webcam_config import DEFAULT_SOURCE_FACE_NAME, WEBCAM_HEADLESS_MODES, get_source_faces, load_webcam_config
from facefusion.webcam_crop import StreamCrop, describe_webcam_crop_error, format_webcam_crop_size, has_webcam_crop_aspect_ratio, resolve_webcam_crop_box, resolve_webcam_stream_resolution
from facefusion.window_output import OUTPUT_WINDOW_TITLE, WindowIntentPump, WindowStream, open_raw_camera_view, has_window_output, log_window_output_requirement


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
	control_overlay = None
	window_intent_pump = None
	raw_camera_view = None

	if webcam_mode not in WEBCAM_HEADLESS_MODES:
		logger.error('headless webcam mode requires one of ' + ', '.join(WEBCAM_HEADLESS_MODES) + ' output', __name__)
		return 2
	if webcam_mode == 'window' and not has_window_output():
		log_window_output_requirement()
		return 2

	state_manager.init_item('source_face_groups', source_face_groups)
	state_manager.init_item('source_face_uploads', {})
	state_manager.init_item('source_face_groups_version', 0)
	state_manager.init_item('webcam_stream_bypass', False)
	state_manager.init_item('source_face_name', webcam_config.get('webcam_source_face') or get_first(list(source_face_groups)) or DEFAULT_SOURCE_FACE_NAME)
	apply_stream_source_paths()
	# The browser layout seeds these from the same config through webcam_options.render, which
	# headless never reaches, so the config's skipping and overlay would be ignored here
	state_manager.init_item('webcam_frame_skipping', webcam_config.get('frame_skipping'))
	state_manager.init_item('webcam_content_analysis_interval', webcam_config.get('content_analysis_interval'))
	state_manager.init_item('webcam_bypassed_processors', webcam_config.get('bypassed_processors'))
	state_manager.init_item('webcam_performance_overlay', webcam_config.get('performance_overlay'))
	state_manager.init_item('webcam_face_preview_window', webcam_config.get('webcam_face_preview_window'))
	# A live switch, not a setting: every run starts with the window shown
	state_manager.init_item('webcam_window_hidden', False)
	state_manager.init_item('webcam_output_blackout', False)
	state_manager.init_item('webcam_blackout_overlay', webcam_config.get('webcam_blackout_overlay'))
	state_manager.init_item('webcam_frame_generation', webcam_config.get('webcam_frame_generation'))
	state_manager.init_item('webcam_face_enhancer_worker_count', webcam_config.get('webcam_face_enhancer_worker_count'))
	# Read by the scheduler on every frame, the same as the browser layout's, though nothing changes
	# them here once the run has started
	state_manager.init_item('webcam_crop_size', webcam_config.get('webcam_crop_size', 'none'))
	state_manager.init_item('webcam_crop_offset_x', webcam_config.get('webcam_crop_offset_x', 0))
	state_manager.init_item('webcam_crop_offset_y', webcam_config.get('webcam_crop_offset_y', 0))
	state_manager.init_item('webcam_crop_upscale', webcam_config.get('webcam_crop_upscale', True))
	apply_stream_face_selector_mode()
	state_manager.sync_state()
	webcam_width, webcam_height = unpack_resolution(webcam_resolution)
	# Refused rather than moved inside: the config asked for a part of the picture it does not have, and
	# a box shifted to fit would frame something it never asked for
	crop_box = resolve_webcam_crop_box(webcam_config.get('webcam_crop_size'), webcam_config.get('webcam_crop_offset_x'), webcam_config.get('webcam_crop_offset_y'))
	is_crop_upscale = bool(webcam_config.get('webcam_crop_upscale', True))
	if crop_box and describe_webcam_crop_error((webcam_width, webcam_height), crop_box):
		logger.error(describe_webcam_crop_error((webcam_width, webcam_height), crop_box), __name__)
		return 2
	if crop_box and is_crop_upscale and not has_webcam_crop_aspect_ratio((webcam_width, webcam_height), crop_box):
		logger.warn('webcam crop ' + format_webcam_crop_size(crop_box) + ' does not match the aspect ratio of capture resolution ' + webcam_resolution + ', the upscale will stretch it', __name__)
	# The output follows the frames the processors get, which are the crop's size when it is not upscaled
	stream_resolution = resolve_webcam_stream_resolution(webcam_resolution, crop_box, is_crop_upscale)
	stream_width, stream_height = unpack_resolution(stream_resolution)
	if webcam_output_resolution == 'copy':
		output_resolution = stream_resolution
		output_width, output_height = stream_width, stream_height
	elif webcam_output_resolution.startswith('x'):
		output_scale = int(webcam_output_resolution[1:])
		output_resolution = pack_resolution((stream_width * output_scale, stream_height * output_scale))
		output_width, output_height = unpack_resolution(output_resolution)
	else:
		output_resolution = webcam_output_resolution
		output_width, output_height = unpack_resolution(webcam_output_resolution)
	if stream_width * output_height != output_width * stream_height:
		logger.warn('webcam output resolution ' + output_resolution + ' does not match the aspect ratio of ' + stream_resolution + ', edges will be cropped', __name__)
	# Same first open as the browser's, retried until the camera opens (Ctrl+C ends it)
	camera_switch = CameraSwitch(webcam_device_id, webcam_width, webcam_height, webcam_fps, webcam_camera_backend)
	camera_capture = camera_switch.open_selected_capture()
	if not camera_capture or not camera_capture.isOpened():
		logger.error('webcam device could not be opened', __name__)
		return 1
	webcam_device_id = camera_switch.camera_id

	# The camera keeps its own rate; only the transport runs at the generated one. A generated stream
	# is also paced by capture time, since uneven worker finishing is what makes doubled frames judder
	output_fps = resolve_output_fps(webcam_fps)
	frame_generator = StreamFrameGenerator()

	try:
		# The window is the control channel headless never had: it carries the hotkeys that switch
		# the source face mid run, which no config file can do
		if webcam_mode == 'window':
			window_stream = WindowStream(output_width, output_height, output_fps, OUTPUT_WINDOW_TITLE, webcam_config.get('webcam_window_fullscreen'), webcam_config.get('webcam_window_display'), window_resolution = webcam_config.get('webcam_window_resolution'), is_high_dpi = webcam_config.get('webcam_window_high_dpi'))
			if webcam_config.get('webcam_control_window'):
				window_stream.open_control()
			if webcam_config.get('webcam_mirror_window'):
				window_stream.open_mirror(webcam_config.get('webcam_mirror_window_display'), webcam_config.get('webcam_mirror_window_fullscreen'))
			control_overlay = ControlOverlay(window_stream, output_width, output_height)
			stream_writer = LatestFrameWriter(lambda: window_stream, output_width, output_height, False, output_fps, output_fps != webcam_fps)
		else:
			stream_writer = LatestFrameWriter(lambda: open_stream(webcam_mode, output_resolution, output_fps), output_width, output_height, webcam_mode == 'v4l2', output_fps, output_fps != webcam_fps)
		stream_writer.start()
		if webcam_config.get('webcam_raw_window'):
			raw_camera_view = open_raw_camera_view(resolve_camera_window_title(webcam_device_id), webcam_fps, webcam_config.get('webcam_raw_window_fullscreen'), webcam_config.get('webcam_raw_window_display'), camera_switch)
		# Keys are read on their own thread from here on, so a stalled pipeline cannot hold them back
		if window_stream:
			window_intent_pump = WindowIntentPump(window_stream, camera_switch)
		# Without the upscale and without a crop the frames pass untouched, so no crop is needed at all
		stream_crop = StreamCrop((webcam_width, webcam_height), is_crop_upscale, crop_box) if crop_box or is_crop_upscale else None
		for capture_vision_frame, capture_time, is_duplicate, timing in multi_process_capture(camera_capture, webcam_fps, webcam_execution_thread_count, (webcam_width, webcam_height), raw_camera_view.submit if raw_camera_view else None, camera_switch, (lambda: not window_intent_pump.has_stopped()) if window_intent_pump else None, stream_crop):
			capture_vision_frame = cv2.cvtColor(capture_vision_frame, cv2.COLOR_BGR2RGB)
			capture_vision_frame = fit_cover_frame(capture_vision_frame, (output_width, output_height))
			# Here and not in the workers: this loop sees the frames one at a time in delivery order,
			# and before the overlay so the hud is never blended between two readings
			generated_vision_frame = frame_generator.generate(capture_vision_frame, is_duplicate)
			timing['post_processing_finished'] = time.perf_counter()
			overlay_mode = state_manager.get_item('webcam_performance_overlay')
			is_control_open = bool(window_stream and window_stream.has_control())
			if overlay_mode in [ 'simple', 'advanced' ] or is_control_open:
				timing['frame_generation'] = frame_generator.get_stats()
				timing['writer_stats'] = stream_writer.get_stats()
			# The window mode blacks out in its window processes, which keeps the control window live.
			# Every other mode blacks out the frame: before the overlay is drawn when the overlay is kept
			# on the black screen, so none of the picture shows through its boxes, and after it otherwise
			is_frame_blackout = not window_stream and bool(state_manager.get_item('webcam_output_blackout'))
			is_frame_blackout_overlay = is_frame_blackout and bool(state_manager.get_item('webcam_blackout_overlay'))
			if is_frame_blackout_overlay:
				capture_vision_frame = create_blackout_frame(capture_vision_frame)
				generated_vision_frame = create_blackout_frame(generated_vision_frame) if generated_vision_frame is not None else None
			if overlay_mode in [ 'simple', 'advanced' ]:
				if generated_vision_frame is not None:
					generated_vision_frame = PERFORMANCE_OVERLAY.render(generated_vision_frame, capture_time, overlay_mode, is_duplicate, timing, True)
				capture_vision_frame = PERFORMANCE_OVERLAY.render(capture_vision_frame, capture_time, overlay_mode, is_duplicate, timing)
			# The control window shows the output's own frames; only the readings go to its overlay
			# thread, which draws the boxes the output leaves out, and the output's boxes on black for
			# a black screen that keeps the overlay
			if control_overlay and (is_control_open or window_stream.is_blackout_overlay):
				control_overlay.submit(capture_time, overlay_mode, is_duplicate, timing, is_control_open)
			if is_frame_blackout and not is_frame_blackout_overlay:
				capture_vision_frame = create_blackout_frame(capture_vision_frame)
				generated_vision_frame = create_blackout_frame(generated_vision_frame) if generated_vision_frame is not None else None
			stream_writer.submit(capture_vision_frame, timing, generated_vision_frame)
			if window_intent_pump and window_intent_pump.has_stopped():
				break
	finally:
		if window_intent_pump:
			window_intent_pump.close()
		if control_overlay:
			control_overlay.close()
		if stream_writer:
			stream_writer.close()
		if window_stream:
			window_stream.close()
		if raw_camera_view:
			raw_camera_view.close()
		clear_camera_pool()

	return 0
