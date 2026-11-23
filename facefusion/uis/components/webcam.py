import time
from typing import Iterator, List, Optional, Tuple

import cv2
import gradio

from facefusion import logger, state_manager, translator
from facefusion.camera_manager import CameraSwitch, cancel_camera_open, clear_camera_pool, get_active_camera_id, is_camera_open_cancelled, request_camera_switch, reset_camera_open, resolve_camera_window_title
from facefusion.filesystem import has_image
from facefusion.frame_generator import StreamFrameGenerator, resolve_output_fps
from facefusion.overlay import PERFORMANCE_OVERLAY
from facefusion.streamer import apply_stream_face_selector_mode, multi_process_capture, open_stream, resolve_source_face_groups
from facefusion.stream_writer import LatestFrameWriter
from facefusion.types import Fps, VisionFrame, WebcamMode
from facefusion.uis.components.webcam_options import get_local_camera_ids, update_source_face_gallery
from facefusion.uis.core import get_ui_component
from facefusion.uis.types import File
from facefusion.vision import fit_cover_frame, pack_resolution, restrict_frame, unpack_resolution
from facefusion.webcam_config import append_source_face_uploads, load_webcam_config, normalize_webcam_resolution
from facefusion.window_output import SharedWindowTransport, acquire_idle_camera_window, acquire_shared_camera_window, acquire_shared_window_stream, open_raw_camera_view, release_shared_camera_window, apply_window_intents, has_window_output, log_window_output_requirement, release_shared_window_stream

SOURCE_FILE : Optional[gradio.File] = None
WEBCAM_IMAGE : Optional[gradio.Image] = None
WEBCAM_START_BUTTON : Optional[gradio.Button] = None
WEBCAM_STOP_BUTTON : Optional[gradio.Button] = None
WEBCAM_CAMERA_TIMER : Optional[gradio.Timer] = None


def render() -> None:
	global SOURCE_FILE
	global WEBCAM_IMAGE
	global WEBCAM_START_BUTTON
	global WEBCAM_STOP_BUTTON
	global WEBCAM_CAMERA_TIMER

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
	# The server cannot push to the page, so the page asks: a camera switched with Ctrl+0-9 (or one
	# being waited for) shows up in the camera dropdown within half a second
	WEBCAM_CAMERA_TIMER = gradio.Timer(0.5)


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
	if webcam_mode_radio and webcam_resolution_dropdown and webcam_fps_slider and webcam_output_resolution_dropdown:
		for window_component in [ webcam_mode_radio, webcam_resolution_dropdown, webcam_output_resolution_dropdown ]:
			window_component.change(update_window_output, inputs = [ webcam_mode_radio, webcam_resolution_dropdown, webcam_fps_slider, webcam_output_resolution_dropdown ])
	# Picking another camera while streaming switches it in place, no stop and start needed
	if webcam_device_id_dropdown:
		webcam_device_id_dropdown.change(switch_running_camera, inputs = webcam_device_id_dropdown)
		if WEBCAM_CAMERA_TIMER:
			WEBCAM_CAMERA_TIMER.tick(sync_camera_dropdown, inputs = webcam_device_id_dropdown, outputs = webcam_device_id_dropdown)
	camera_window_inputs = resolve_camera_window_inputs()
	if camera_window_inputs:
		for camera_window_component in camera_window_inputs[:4]:
			camera_window_component.change(update_camera_window, inputs = camera_window_inputs)


# The checkbox values are passed in rather than read from state: their own change handlers write the
# state, and gradio does not order two handlers of the same event
def resolve_camera_window_inputs() -> Optional[List[gradio.components.Component]]:
	camera_window_inputs =\
	[
		get_ui_component('webcam_raw_window_checkbox'),
		get_ui_component('webcam_raw_window_fullscreen_checkbox'),
		get_ui_component('webcam_device_id_dropdown'),
		get_ui_component('webcam_resolution_dropdown'),
		get_ui_component('webcam_fps_slider')
	]

	if all(camera_window_inputs):
		return camera_window_inputs #type:ignore[return-value]
	return None


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

	# Whether or not the stream starts by itself, a page opened with window mode selected has its window
	if webcam_mode_radio and webcam_resolution_dropdown and webcam_fps_slider and webcam_output_resolution_dropdown:
		ui.load(update_window_output, inputs = [ webcam_mode_radio, webcam_resolution_dropdown, webcam_fps_slider, webcam_output_resolution_dropdown ])
	# Likewise the camera window, whenever its option is on
	camera_window_inputs = resolve_camera_window_inputs()
	if camera_window_inputs:
		ui.load(update_camera_window, inputs = camera_window_inputs)
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

	# Checked again here because start can be clicked before the dropdown's change has normalized a
	# typed value, and unpack_resolution raises on anything that is not WIDTHxHEIGHT
	webcam_resolution = normalize_webcam_resolution(webcam_resolution)
	if not webcam_resolution:
		logger.error('webcam resolution must be WIDTHxHEIGHT with even sides between 16 and 7680', __name__)
		return
	webcam_width, webcam_height = unpack_resolution(webcam_resolution)
	output_resolution, output_width, output_height = resolve_output_resolution(webcam_resolution, webcam_output_resolution)
	if webcam_width * output_height != output_width * webcam_height:
		logger.warn('webcam output resolution ' + output_resolution + ' does not match the aspect ratio of capture resolution ' + webcam_resolution + ', edges will be cropped', __name__)
	# Refused rather than downgraded: falling back to inline would hand back the very encoder the
	# window exists to remove
	if webcam_mode == 'window' and not has_window_output():
		log_window_output_requirement()
		return
	# Tried every 100 ms until the camera opens; only the stop button gives up on it
	# Active before the first attempt, so a camera picked in the dropdown while this start is still
	# waiting is the one it goes on trying
	reset_camera_open()
	camera_switch = CameraSwitch(webcam_device_id, webcam_width, webcam_height, webcam_fps, webcam_camera_backend)
	camera_switch.activate()
	camera_capture = camera_switch.open_selected_capture(is_camera_open_cancelled)
	if not camera_capture:
		camera_switch.deactivate()
	# The camera that actually opened, which may not be the one start() was given
	webcam_device_id = camera_switch.camera_id
	stream_writer = None
	window_stream = None
	window_transport = None
	raw_camera_view = None
	# The camera keeps its own rate; only the transport runs at the generated one. A generated stream
	# is also paced by capture time, since uneven worker finishing is what makes doubled frames judder
	output_fps = resolve_output_fps(webcam_fps)
	frame_generator = StreamFrameGenerator()
	# Headless says so and exits; the browser layout used to just hand back an empty image, leaving
	# a failed camera indistinguishable from a stream that produced nothing
	if not camera_capture or not camera_capture.isOpened():
		logger.error('webcam device could not be opened', __name__)
	if camera_capture and camera_capture.isOpened():
		try:
			if webcam_mode in [ 'udp', 'v4l2' ]:
				stream_writer = LatestFrameWriter(lambda: open_stream(webcam_mode, output_resolution, output_fps), output_width, output_height, webcam_mode == 'v4l2', output_fps, output_fps != webcam_fps) #type:ignore[arg-type]
				stream_writer.start()
			# The window outlives the stream: it is the one already open since window mode was selected,
			# or opened now, and the writer gets a transport whose close blanks it instead of closing it.
			# Acquired here rather than inside the factory so that poll_events stays reachable from this
			# loop. repeat_latest is off because the window goes on showing the last frame by itself
			if webcam_mode == 'window':
				window_stream = acquire_shared_window_stream(output_width, output_height, output_fps, state_manager.get_item('webcam_window_fullscreen'), state_manager.get_item('webcam_window_display') or 0)
				window_transport = SharedWindowTransport(window_stream)
				stream_writer = LatestFrameWriter(lambda: window_transport, output_width, output_height, False, output_fps, output_fps != webcam_fps)
				stream_writer.start()
			# The shared camera window, open already while the option is on; acquired here too in case
			# it was closed, so it is up before the camera delivers its first frame
			if state_manager.get_item('webcam_raw_window') and has_window_output():
				acquire_shared_camera_window(webcam_width, webcam_height, webcam_fps, resolve_camera_window_title(webcam_device_id), bool(state_manager.get_item('webcam_raw_window_fullscreen')), state_manager.get_item('webcam_raw_window_display') or 0)
			if state_manager.get_item('webcam_raw_window'):
				raw_camera_view = open_raw_camera_view(resolve_camera_window_title(webcam_device_id), webcam_fps, bool(state_manager.get_item('webcam_raw_window_fullscreen')), state_manager.get_item('webcam_raw_window_display') or 0, camera_switch, True)
			for capture_vision_frame, capture_time, is_duplicate, timing in multi_process_capture(camera_capture, webcam_fps, webcam_execution_thread_count, (webcam_width, webcam_height), raw_camera_view.submit if raw_camera_view else None, camera_switch, (lambda: apply_window_intents(window_stream, camera_switch)) if window_stream else None):
				capture_vision_frame = cv2.cvtColor(capture_vision_frame, cv2.COLOR_BGR2RGB)
				capture_vision_frame = fit_cover_frame(capture_vision_frame, (output_width, output_height))
				# Here and not in the workers: this loop sees the frames one at a time in delivery
				# order, and before the overlay so the hud is never blended between two readings. The
				# inline preview has no writer to pace a generated frame, so it never gets one
				generated_vision_frame = frame_generator.generate(capture_vision_frame, is_duplicate) if stream_writer else None
				timing['post_processing_finished'] = time.perf_counter()

				overlay_mode = state_manager.get_item('webcam_performance_overlay')
				if overlay_mode in [ 'simple', 'advanced' ]:
					timing['frame_generation'] = frame_generator.get_stats()
					timing['writer_stats'] = stream_writer.get_stats() if stream_writer else None
					if generated_vision_frame is not None:
						generated_vision_frame = PERFORMANCE_OVERLAY.render(generated_vision_frame, capture_time, overlay_mode, is_duplicate, timing, True)
					capture_vision_frame = PERFORMANCE_OVERLAY.render(capture_vision_frame, capture_time, overlay_mode, is_duplicate, timing)

				if webcam_mode == 'inline':
					if inline_preview_resolution != 'native':
						capture_vision_frame = restrict_frame(capture_vision_frame, unpack_resolution(inline_preview_resolution))
					yield capture_vision_frame
				if stream_writer:
					stream_writer.submit(capture_vision_frame, timing, generated_vision_frame)
				if window_stream and not apply_window_intents(window_stream, camera_switch):
					break
		finally:
			if stream_writer:
				stream_writer.close()
			# Also closed by the writer when its thread finishes; this covers one that did not
			if window_transport:
				window_transport.close()
			if raw_camera_view:
				raw_camera_view.close()
			camera_switch.deactivate()
			clear_camera_pool()


def resolve_output_resolution(webcam_resolution : str, webcam_output_resolution : str) -> Tuple[str, int, int]:
	webcam_width, webcam_height = unpack_resolution(webcam_resolution)

	if webcam_output_resolution == 'copy':
		return webcam_resolution, webcam_width, webcam_height
	if webcam_output_resolution.startswith('x'):
		output_scale = int(webcam_output_resolution[1:])
		output_resolution = pack_resolution((webcam_width * output_scale, webcam_height * output_scale))
		return output_resolution, webcam_width * output_scale, webcam_height * output_scale
	output_width, output_height = unpack_resolution(webcam_output_resolution)
	return webcam_output_resolution, output_width, output_height


# Selecting the window mode opens the window and keeps it, so it can be placed and captured before
# anything streams into it; selecting any other mode closes it. Also run on page load and when the
# size changes, which reopens it at the size the next start will deliver
def update_window_output(webcam_mode : WebcamMode, webcam_resolution : str, webcam_fps : Fps, webcam_output_resolution : str) -> None:
	if webcam_mode != 'window' or not has_window_output():
		release_shared_window_stream()
		return
	webcam_resolution = normalize_webcam_resolution(webcam_resolution)
	if webcam_resolution:
		_, output_width, output_height = resolve_output_resolution(webcam_resolution, webcam_output_resolution)
		acquire_shared_window_stream(output_width, output_height, resolve_output_fps(webcam_fps), state_manager.get_item('webcam_window_fullscreen'), state_manager.get_item('webcam_window_display') or 0)


# The camera window stays open while its option is on, black until a stream feeds it, the same way
# the output window does in window mode. Opened at the capture resolution asked for, under the name
# of the selected camera; a stream reopens it if the camera grants another size
# Shows the camera the running stream is on, or the one it is waiting for, so it can never disagree
# with the stream and send it back. Setting the value fires the dropdown's change, which asks for the
# camera already selected and so changes nothing. A camera not detected when the page was built is
# added to the choices, since gradio refuses a value outside them
def sync_camera_dropdown(webcam_device_id : int) -> gradio.Dropdown:
	active_camera_id = get_active_camera_id()

	if active_camera_id is None or str(active_camera_id) == str(webcam_device_id):
		return skip_component_update()
	local_camera_ids = list(get_local_camera_ids())
	if active_camera_id not in local_camera_ids:
		local_camera_ids.append(active_camera_id)
		return gradio.Dropdown(value = active_camera_id, choices = local_camera_ids)
	return gradio.Dropdown(value = active_camera_id)


def skip_component_update() -> gradio.Dropdown:
	if hasattr(gradio, 'skip'):
		return gradio.skip()
	return gradio.Dropdown()


# Nothing to do without a running stream: the next start opens whatever the dropdown then shows
def switch_running_camera(webcam_device_id : int) -> None:
	if str(webcam_device_id).isdigit():
		request_camera_switch(int(webcam_device_id))


def update_camera_window(raw_window : bool, raw_window_fullscreen : bool, webcam_device_id : int, webcam_resolution : str, webcam_fps : Fps) -> None:
	if not raw_window or not has_window_output():
		release_shared_camera_window()
		return
	webcam_resolution = normalize_webcam_resolution(webcam_resolution)
	if webcam_resolution:
		webcam_width, webcam_height = unpack_resolution(webcam_resolution)
		camera_title = resolve_camera_window_title(int(webcam_device_id)) if str(webcam_device_id).isdigit() else 'CAMERA'
		acquire_idle_camera_window(webcam_width, webcam_height, webcam_fps, camera_title, bool(raw_window_fullscreen), state_manager.get_item('webcam_raw_window_display') or 0)


def stop() -> gradio.Image:
	cancel_camera_open()
	clear_camera_pool()
	return gradio.Image(value = None)
