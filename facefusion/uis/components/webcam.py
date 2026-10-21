import time
from typing import Iterator, List, Optional, Tuple

import cv2
import gradio

from facefusion import logger, state_manager, translator
from facefusion.camera_manager import clear_camera_pool, get_local_camera_capture
from facefusion.filesystem import has_image
from facefusion.overlay import PERFORMANCE_OVERLAY
from facefusion.streamer import apply_stream_face_selector_mode, multi_process_capture, open_stream, resolve_source_face_groups
from facefusion.stream_writer import LatestFrameWriter
from facefusion.types import Fps, VisionFrame, WebcamMode
from facefusion.uis.components.webcam_options import update_source_face_gallery
from facefusion.uis.core import get_ui_component
from facefusion.uis.types import File
from facefusion.vision import fit_cover_frame, pack_resolution, restrict_frame, unpack_resolution
from facefusion.webcam_config import append_source_face_uploads, load_webcam_config
from facefusion.window_output import WindowStream, apply_window_intents, has_window_output, log_window_output_requirement

SOURCE_FILE : Optional[gradio.File] = None
WEBCAM_IMAGE : Optional[gradio.Image] = None
WEBCAM_START_BUTTON : Optional[gradio.Button] = None
WEBCAM_STOP_BUTTON : Optional[gradio.Button] = None


def render() -> None:
	global SOURCE_FILE
	global WEBCAM_IMAGE
	global WEBCAM_START_BUTTON
	global WEBCAM_STOP_BUTTON

	# Starts and stays empty: it is an intake for adding faces, and configured source paths already
	# show up in the gallery as a selectable face
	SOURCE_FILE = gradio.File(
		label = translator.get('uis.source_file'),
		file_count = 'multiple',
		value = None
	)
	WEBCAM_IMAGE = gradio.Image(
		label = translator.get('uis.webcam_image'),
		format = 'jpeg',
		visible = False
	)
	WEBCAM_START_BUTTON = gradio.Button(
		value = translator.get('uis.start_button'),
		variant = 'primary',
		size = 'sm'
	)
	WEBCAM_STOP_BUTTON = gradio.Button(
		value = translator.get('uis.stop_button'),
		size = 'sm',
		visible = False
	)


def listen() -> None:
	webcam_source_face_gallery = get_ui_component('webcam_source_face_gallery')
	webcam_source_face_clear_button = get_ui_component('webcam_source_face_clear_button')

	if webcam_source_face_gallery:
		SOURCE_FILE.upload(update_source, inputs = SOURCE_FILE, outputs = [ SOURCE_FILE, webcam_source_face_gallery ])
		if webcam_source_face_clear_button:
			webcam_source_face_clear_button.click(clear_source, outputs = [ SOURCE_FILE, webcam_source_face_gallery ])
	webcam_device_id_dropdown = get_ui_component('webcam_device_id_dropdown')
	webcam_mode_radio = get_ui_component('webcam_mode_radio')
	webcam_resolution_dropdown = get_ui_component('webcam_resolution_dropdown')
	webcam_fps_slider = get_ui_component('webcam_fps_slider')
	webcam_inline_preview_resolution_dropdown = get_ui_component('webcam_inline_preview_resolution_dropdown')
	webcam_execution_thread_count_slider = get_ui_component('webcam_execution_thread_count_slider')
	webcam_camera_backend_dropdown = get_ui_component('webcam_camera_backend_dropdown')
	webcam_output_resolution_dropdown = get_ui_component('webcam_output_resolution_dropdown')

	if webcam_device_id_dropdown and webcam_mode_radio and webcam_resolution_dropdown and webcam_fps_slider and webcam_inline_preview_resolution_dropdown and webcam_execution_thread_count_slider and webcam_camera_backend_dropdown and webcam_output_resolution_dropdown:
		WEBCAM_START_BUTTON.click(pre_start, outputs = [ SOURCE_FILE, WEBCAM_IMAGE, WEBCAM_START_BUTTON, WEBCAM_STOP_BUTTON ])
		start_event = WEBCAM_START_BUTTON.click(start, inputs = [ webcam_device_id_dropdown, webcam_mode_radio, webcam_resolution_dropdown, webcam_fps_slider, webcam_inline_preview_resolution_dropdown, webcam_execution_thread_count_slider, webcam_camera_backend_dropdown, webcam_output_resolution_dropdown ], outputs = WEBCAM_IMAGE)
		start_event.then(pre_stop)
		WEBCAM_STOP_BUTTON.click(stop, cancels = start_event, outputs = WEBCAM_IMAGE)
		WEBCAM_STOP_BUTTON.click(pre_stop, outputs = [ SOURCE_FILE, WEBCAM_IMAGE, WEBCAM_START_BUTTON, WEBCAM_STOP_BUTTON ])


def listen_auto_start(ui : gradio.Blocks) -> None:
	webcam_config = load_webcam_config(state_manager.get_item('webcam_config'))
	webcam_device_id_dropdown = get_ui_component('webcam_device_id_dropdown')
	webcam_mode_radio = get_ui_component('webcam_mode_radio')
	webcam_resolution_dropdown = get_ui_component('webcam_resolution_dropdown')
	webcam_fps_slider = get_ui_component('webcam_fps_slider')
	webcam_inline_preview_resolution_dropdown = get_ui_component('webcam_inline_preview_resolution_dropdown')
	webcam_execution_thread_count_slider = get_ui_component('webcam_execution_thread_count_slider')
	webcam_camera_backend_dropdown = get_ui_component('webcam_camera_backend_dropdown')
	webcam_output_resolution_dropdown = get_ui_component('webcam_output_resolution_dropdown')

	if webcam_config.get('auto_start') and webcam_device_id_dropdown and webcam_mode_radio and webcam_resolution_dropdown and webcam_fps_slider and webcam_inline_preview_resolution_dropdown and webcam_execution_thread_count_slider and webcam_camera_backend_dropdown and webcam_output_resolution_dropdown:
		load_event = ui.load(pre_start, outputs = [ SOURCE_FILE, WEBCAM_IMAGE, WEBCAM_START_BUTTON, WEBCAM_STOP_BUTTON ])
		start_event = load_event.then(start, inputs = [ webcam_device_id_dropdown, webcam_mode_radio, webcam_resolution_dropdown, webcam_fps_slider, webcam_inline_preview_resolution_dropdown, webcam_execution_thread_count_slider, webcam_camera_backend_dropdown, webcam_output_resolution_dropdown ], outputs = WEBCAM_IMAGE)
		start_event.then(pre_stop)


# Each uploaded image becomes its own selectable face instead of being averaged with the others, and
# uploads add to what is already there rather than replacing it
def update_source(files : List[File]) -> Tuple[gradio.File, gradio.Gallery]:
	file_names = [ file.name for file in files ] if files else []
	current_source_face_uploads = state_manager.get_item('source_face_uploads')
	source_face_uploads = append_source_face_uploads(current_source_face_uploads, file_names)

	if source_face_uploads != current_source_face_uploads:
		state_manager.set_item('source_face_uploads', source_face_uploads)
		apply_source_paths(bool(source_face_uploads))

	# Holding on to the uploaded files leaves the widget full and with no room to drop the next one
	# in; the gallery is the record of what has been loaded, so the intake is emptied again
	return gradio.File(value = None), update_source_face_gallery()


def clear_source() -> Tuple[gradio.File, gradio.Gallery]:
	state_manager.set_item('source_face_uploads', {})
	apply_source_paths(False)
	return gradio.File(value = None), update_source_face_gallery()


# pre_process refuses to run the face swapper unless source_paths holds a detectable face, so it
# carries every path of every group while the groups themselves drive what is selectable
def apply_source_paths(has_source_face_uploads : bool) -> None:
	source_paths = [ source_path for source_paths in resolve_source_face_groups().values() for source_path in source_paths ]
	state_manager.set_item('source_face_groups_version', (state_manager.get_item('source_face_groups_version') or 0) + 1)

	if has_image(source_paths):
		state_manager.set_item('source_paths', source_paths)
	else:
		state_manager.clear_item('source_paths')
	if has_source_face_uploads:
		state_manager.set_item('source_paths_from_ui', True)
	else:
		state_manager.clear_item('source_paths_from_ui')


def pre_start() -> Tuple[gradio.File, gradio.Image, gradio.Button, gradio.Button]:
	return gradio.File(visible = False), gradio.Image(visible = True), gradio.Button(visible = False), gradio.Button(visible = True)


def pre_stop() -> Tuple[gradio.File, gradio.Image, gradio.Button, gradio.Button]:
	return gradio.File(visible = True), gradio.Image(visible = False), gradio.Button(visible = True), gradio.Button(visible = False)


def start(webcam_device_id : int, webcam_mode : WebcamMode, webcam_resolution : str, webcam_fps : Fps, inline_preview_resolution : str = '640x480', webcam_execution_thread_count : int = 2, webcam_camera_backend : str = 'auto', webcam_output_resolution : str = 'copy') -> Iterator[VisionFrame]:
	apply_stream_face_selector_mode()
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
	# Refused rather than downgraded: falling back to inline would hand back the very encoder the
	# window exists to remove
	if webcam_mode == 'window' and not has_window_output():
		log_window_output_requirement()
		return
	camera_capture = get_local_camera_capture(webcam_device_id, webcam_width, webcam_height, webcam_fps, webcam_camera_backend)
	stream_writer = None
	window_stream = None
	# Headless says so and exits; the browser layout used to just hand back an empty image, leaving
	# a failed camera indistinguishable from a stream that produced nothing
	if not camera_capture or not camera_capture.isOpened():
		logger.error('webcam device could not be opened', __name__)
	if camera_capture and camera_capture.isOpened():
		try:
			if webcam_mode in [ 'udp', 'v4l2' ]:
				stream_writer = LatestFrameWriter(lambda: open_stream(webcam_mode, output_resolution, webcam_fps), output_width, output_height, webcam_mode == 'v4l2', webcam_fps) #type:ignore[arg-type]
				stream_writer.start()
			# Created here rather than inside the factory so that poll_events stays reachable from
			# this loop and a broken pygame fails before the camera is opened. repeat_latest is off
			# because the window goes on showing the last frame by itself
			if webcam_mode == 'window':
				window_stream = WindowStream(output_width, output_height, webcam_fps, 'FaceFusion', state_manager.get_item('webcam_window_fullscreen'))
				stream_writer = LatestFrameWriter(lambda: window_stream, output_width, output_height, False, webcam_fps)
				stream_writer.start()
			for capture_vision_frame, capture_time, is_duplicate, timing in multi_process_capture(camera_capture, webcam_fps, webcam_execution_thread_count, (webcam_width, webcam_height)):
				capture_vision_frame = cv2.cvtColor(capture_vision_frame, cv2.COLOR_BGR2RGB)
				capture_vision_frame = fit_cover_frame(capture_vision_frame, (output_width, output_height))
				timing['post_processing_finished'] = time.perf_counter()

				overlay_mode = state_manager.get_item('webcam_performance_overlay')
				if overlay_mode in [ 'simple', 'advanced' ]:
					capture_vision_frame = PERFORMANCE_OVERLAY.render(capture_vision_frame, capture_time, overlay_mode, is_duplicate, timing)

				if webcam_mode == 'inline':
					if inline_preview_resolution != 'native':
						capture_vision_frame = restrict_frame(capture_vision_frame, unpack_resolution(inline_preview_resolution))
					yield capture_vision_frame
				if stream_writer:
					stream_writer.submit(capture_vision_frame, timing)
				if window_stream and not apply_window_intents(window_stream):
					break
		finally:
			if stream_writer:
				stream_writer.close()
			if window_stream:
				window_stream.close()
			clear_camera_pool()


def stop() -> gradio.Image:
	clear_camera_pool()
	return gradio.Image(value = None)
