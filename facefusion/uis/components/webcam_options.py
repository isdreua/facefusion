from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, translator
from facefusion.camera_manager import detect_local_camera_ids
from facefusion.filesystem import is_image
from facefusion.common_helper import get_first
from facefusion.uis import choices as uis_choices
from facefusion.streamer import apply_stream_source_paths, resolve_source_face_groups
from facefusion.types import WebcamMode
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.webcam_config import DEFAULT_SOURCE_FACE_NAME, NO_SOURCE_FACE_NAME, WEBCAM_BYPASSED_PROCESSORS, WEBCAM_CAMERA_BACKENDS, WEBCAM_CONTENT_ANALYSIS_INTERVALS, WEBCAM_FRAME_SKIPPING_MODES, WEBCAM_INLINE_PREVIEW_RESOLUTIONS, WEBCAM_OUTPUT_RESOLUTIONS, WEBCAM_PERFORMANCE_OVERLAYS, WEBCAM_STATE_KEYS, format_webcam_config, get_source_faces, load_webcam_config, resolve_source_face_directory
from facefusion.window_output import has_window_output, log_window_output_requirement

WEBCAM_DEVICE_ID_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_MODE_RADIO : Optional[gradio.Radio] = None
WEBCAM_RESOLUTION_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_FPS_SLIDER : Optional[gradio.Slider] = None
WEBCAM_PERFORMANCE_OVERLAY_RADIO : Optional[gradio.Radio] = None
WEBCAM_FRAME_SKIPPING_RADIO : Optional[gradio.Radio] = None
WEBCAM_CONTENT_ANALYSIS_INTERVAL_SLIDER : Optional[gradio.Slider] = None
WEBCAM_BYPASSED_PROCESSORS_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None
WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_EXECUTION_THREAD_COUNT_SLIDER : Optional[gradio.Slider] = None
WEBCAM_CAMERA_BACKEND_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_OUTPUT_RESOLUTION_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_WINDOW_FULLSCREEN_CHECKBOX : Optional[gradio.Checkbox] = None
WEBCAM_WINDOW_HINT_MARKDOWN : Optional[gradio.Markdown] = None
SOURCE_FACE_SELECTED_MARKER = '\u25cf '
WEBCAM_SOURCE_FACE_GALLERY : Optional[gradio.Gallery] = None
WEBCAM_SOURCE_FACE_CLEAR_BUTTON : Optional[gradio.Button] = None
WEBCAM_GENERATE_CONFIG_BUTTON : Optional[gradio.Button] = None
WEBCAM_GENERATE_CONFIG_TEXTBOX : Optional[gradio.Textbox] = None


def render() -> None:
	global WEBCAM_DEVICE_ID_DROPDOWN
	global WEBCAM_MODE_RADIO
	global WEBCAM_RESOLUTION_DROPDOWN
	global WEBCAM_FPS_SLIDER
	global WEBCAM_PERFORMANCE_OVERLAY_RADIO
	global WEBCAM_FRAME_SKIPPING_RADIO
	global WEBCAM_CONTENT_ANALYSIS_INTERVAL_SLIDER
	global WEBCAM_BYPASSED_PROCESSORS_CHECKBOX_GROUP
	global WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN
	global WEBCAM_EXECUTION_THREAD_COUNT_SLIDER
	global WEBCAM_CAMERA_BACKEND_DROPDOWN
	global WEBCAM_OUTPUT_RESOLUTION_DROPDOWN
	global WEBCAM_WINDOW_FULLSCREEN_CHECKBOX
	global WEBCAM_WINDOW_HINT_MARKDOWN
	global WEBCAM_SOURCE_FACE_GALLERY
	global WEBCAM_SOURCE_FACE_CLEAR_BUTTON
	global WEBCAM_GENERATE_CONFIG_BUTTON
	global WEBCAM_GENERATE_CONFIG_TEXTBOX

	webcam_config = load_webcam_config(state_manager.get_item('webcam_config'))
	local_camera_ids = detect_local_camera_ids(0, 10, webcam_config.get('webcam_camera_backend')) or [ 'none' ] #type:ignore[list-item]
	webcam_device_id = webcam_config.get('device_id')
	if webcam_device_id not in local_camera_ids:
		webcam_device_id = get_first(local_camera_ids)
	source_face_groups = get_source_faces(webcam_config)
	source_face_name = webcam_config.get('webcam_source_face') or get_first(list(source_face_groups)) or DEFAULT_SOURCE_FACE_NAME
	state_manager.init_item('source_face_groups', source_face_groups)
	state_manager.init_item('source_face_uploads', {})
	state_manager.init_item('source_face_groups_version', 0)
	state_manager.init_item('source_face_name', source_face_name)
	apply_stream_source_paths()
	state_manager.init_item('webcam_frame_skipping', webcam_config.get('frame_skipping'))
	state_manager.init_item('webcam_content_analysis_interval', webcam_config.get('content_analysis_interval'))
	state_manager.init_item('webcam_bypassed_processors', resolve_bypassed_processors(webcam_config.get('bypassed_processors'), state_manager.get_item('processors')))
	state_manager.init_item('webcam_performance_overlay', webcam_config.get('performance_overlay'))
	state_manager.init_item('webcam_stream_bypass', False)
	state_manager.init_item('webcam_window_fullscreen', webcam_config.get('webcam_window_fullscreen'))
	WEBCAM_DEVICE_ID_DROPDOWN = gradio.Dropdown(
		value = webcam_device_id,
		label = translator.get('uis.webcam_device_id_dropdown'),
		choices = local_camera_ids
	)
	# gradio cannot gray out one choice of a radio, so an unavailable window is removed from the
	# choices and explained underneath instead of being offered and then refused on start
	webcam_modes, webcam_mode = resolve_webcam_modes(webcam_config.get('mode'))
	WEBCAM_MODE_RADIO = gradio.Radio(
		label = translator.get('uis.webcam_mode_radio'),
		choices = webcam_modes,
		value = webcam_mode
	)
	# Read from state at window creation like frame skipping and the overlay are, which keeps it off
	# start()'s signature and out of both of its inputs lists
	WEBCAM_WINDOW_FULLSCREEN_CHECKBOX = gradio.Checkbox(
		label = translator.get('uis.webcam_window_fullscreen_checkbox'),
		value = webcam_config.get('webcam_window_fullscreen'),
		visible = has_window_output()
	)
	WEBCAM_WINDOW_HINT_MARKDOWN = gradio.Markdown(
		value = translator.get('uis.webcam_window_hint_markdown'),
		visible = not has_window_output()
	)
	WEBCAM_RESOLUTION_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.webcam_resolution_dropdown'),
		choices = uis_choices.webcam_resolutions,
		value = webcam_config.get('resolution')
	)
	WEBCAM_FPS_SLIDER = gradio.Slider(
		label = translator.get('uis.webcam_fps_slider'),
		value = webcam_config.get('fps'),
		step = 1,
		minimum = 1,
		maximum = 30
	)
	WEBCAM_PERFORMANCE_OVERLAY_RADIO = gradio.Radio(
		label = translator.get('uis.webcam_performance_overlay_radio'),
		choices = WEBCAM_PERFORMANCE_OVERLAYS,
		value = webcam_config.get('performance_overlay')
	)
	WEBCAM_FRAME_SKIPPING_RADIO = gradio.Radio(
		label = translator.get('uis.webcam_frame_skipping_radio'),
		choices = WEBCAM_FRAME_SKIPPING_MODES,
		value = webcam_config.get('frame_skipping')
	)
	WEBCAM_CONTENT_ANALYSIS_INTERVAL_SLIDER = gradio.Slider(
		label = translator.get('uis.webcam_content_analysis_interval_slider'),
		value = webcam_config.get('content_analysis_interval'),
		step = 1,
		minimum = min(WEBCAM_CONTENT_ANALYSIS_INTERVALS),
		maximum = max(WEBCAM_CONTENT_ANALYSIS_INTERVALS)
	)
	WEBCAM_BYPASSED_PROCESSORS_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = translator.get('uis.webcam_bypassed_processors_checkbox_group'),
		choices = state_manager.get_item('processors'),
		value = state_manager.get_item('webcam_bypassed_processors')
	)
	WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.webcam_inline_preview_resolution_dropdown'),
		choices = WEBCAM_INLINE_PREVIEW_RESOLUTIONS,
		value = webcam_config.get('webcam_inline_preview_resolution')
	)
	WEBCAM_EXECUTION_THREAD_COUNT_SLIDER = gradio.Slider(
		label = translator.get('uis.webcam_execution_thread_count_slider'),
		value = webcam_config.get('webcam_execution_thread_count'),
		step = 1,
		minimum = 1,
		maximum = 8
	)
	WEBCAM_CAMERA_BACKEND_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.webcam_camera_backend_dropdown'),
		choices = WEBCAM_CAMERA_BACKENDS,
		value = webcam_config.get('webcam_camera_backend')
	)
	WEBCAM_OUTPUT_RESOLUTION_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.webcam_output_resolution_dropdown'),
		choices = WEBCAM_OUTPUT_RESOLUTIONS,
		value = webcam_config.get('webcam_output_resolution')
	)
	source_face_name, source_face_names = resolve_source_face_selection()
	WEBCAM_SOURCE_FACE_GALLERY = gradio.Gallery(
		label = resolve_source_face_label(source_face_name, source_face_names),
		value = create_source_face_items(source_face_name),
		columns = 6,
		height = 140,
		object_fit = 'cover',
		allow_preview = False,
		selected_index = source_face_names.index(source_face_name) if source_face_name in source_face_names else None
	)
	WEBCAM_SOURCE_FACE_CLEAR_BUTTON = gradio.Button(
		value = translator.get('uis.webcam_source_face_clear_button'),
		size = 'sm'
	)
	WEBCAM_GENERATE_CONFIG_BUTTON = gradio.Button(
		value = translator.get('uis.webcam_generate_config_button'),
		size = 'sm'
	)
	WEBCAM_GENERATE_CONFIG_TEXTBOX = gradio.Textbox(
		label = translator.get('uis.webcam_generate_config_textbox'),
		lines = 20,
		max_lines = 20,
		show_copy_button = True,
		interactive = False,
		visible = False
	)
	register_ui_component('webcam_device_id_dropdown', WEBCAM_DEVICE_ID_DROPDOWN)
	register_ui_component('webcam_mode_radio', WEBCAM_MODE_RADIO)
	register_ui_component('webcam_resolution_dropdown', WEBCAM_RESOLUTION_DROPDOWN)
	register_ui_component('webcam_fps_slider', WEBCAM_FPS_SLIDER)
	register_ui_component('webcam_performance_overlay_radio', WEBCAM_PERFORMANCE_OVERLAY_RADIO)
	register_ui_component('webcam_frame_skipping_radio', WEBCAM_FRAME_SKIPPING_RADIO)
	register_ui_component('webcam_content_analysis_interval_slider', WEBCAM_CONTENT_ANALYSIS_INTERVAL_SLIDER)
	register_ui_component('webcam_bypassed_processors_checkbox_group', WEBCAM_BYPASSED_PROCESSORS_CHECKBOX_GROUP)
	register_ui_component('webcam_inline_preview_resolution_dropdown', WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN)
	register_ui_component('webcam_execution_thread_count_slider', WEBCAM_EXECUTION_THREAD_COUNT_SLIDER)
	register_ui_component('webcam_camera_backend_dropdown', WEBCAM_CAMERA_BACKEND_DROPDOWN)
	register_ui_component('webcam_output_resolution_dropdown', WEBCAM_OUTPUT_RESOLUTION_DROPDOWN)
	register_ui_component('webcam_window_fullscreen_checkbox', WEBCAM_WINDOW_FULLSCREEN_CHECKBOX)
	register_ui_component('webcam_window_hint_markdown', WEBCAM_WINDOW_HINT_MARKDOWN)
	register_ui_component('webcam_source_face_gallery', WEBCAM_SOURCE_FACE_GALLERY)
	register_ui_component('webcam_source_face_clear_button', WEBCAM_SOURCE_FACE_CLEAR_BUTTON)
	register_ui_component('webcam_generate_config_button', WEBCAM_GENERATE_CONFIG_BUTTON)


def listen() -> None:
	if WEBCAM_PERFORMANCE_OVERLAY_RADIO:
		WEBCAM_PERFORMANCE_OVERLAY_RADIO.change(update_performance_overlay, inputs = WEBCAM_PERFORMANCE_OVERLAY_RADIO)
	if WEBCAM_FRAME_SKIPPING_RADIO:
		WEBCAM_FRAME_SKIPPING_RADIO.change(update_frame_skipping, inputs = WEBCAM_FRAME_SKIPPING_RADIO)
	if WEBCAM_CONTENT_ANALYSIS_INTERVAL_SLIDER:
		WEBCAM_CONTENT_ANALYSIS_INTERVAL_SLIDER.change(update_content_analysis_interval, inputs = WEBCAM_CONTENT_ANALYSIS_INTERVAL_SLIDER)
	if WEBCAM_BYPASSED_PROCESSORS_CHECKBOX_GROUP:
		WEBCAM_BYPASSED_PROCESSORS_CHECKBOX_GROUP.change(update_bypassed_processors, inputs = WEBCAM_BYPASSED_PROCESSORS_CHECKBOX_GROUP)
		processors_checkbox_group = get_ui_component('processors_checkbox_group')
		if processors_checkbox_group:
			processors_checkbox_group.change(remote_update_bypassed_processors, inputs = processors_checkbox_group, outputs = WEBCAM_BYPASSED_PROCESSORS_CHECKBOX_GROUP)
	if WEBCAM_WINDOW_FULLSCREEN_CHECKBOX:
		WEBCAM_WINDOW_FULLSCREEN_CHECKBOX.change(update_window_fullscreen, inputs = WEBCAM_WINDOW_FULLSCREEN_CHECKBOX)
	if WEBCAM_GENERATE_CONFIG_BUTTON and WEBCAM_GENERATE_CONFIG_TEXTBOX:
		WEBCAM_GENERATE_CONFIG_BUTTON.click(generate_config, inputs =\
		[
			WEBCAM_DEVICE_ID_DROPDOWN, WEBCAM_MODE_RADIO, WEBCAM_RESOLUTION_DROPDOWN, WEBCAM_FPS_SLIDER,
			WEBCAM_PERFORMANCE_OVERLAY_RADIO, WEBCAM_FRAME_SKIPPING_RADIO, WEBCAM_CONTENT_ANALYSIS_INTERVAL_SLIDER,
			WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN, WEBCAM_EXECUTION_THREAD_COUNT_SLIDER, WEBCAM_CAMERA_BACKEND_DROPDOWN,
			WEBCAM_OUTPUT_RESOLUTION_DROPDOWN
		], outputs = WEBCAM_GENERATE_CONFIG_TEXTBOX)
	if WEBCAM_SOURCE_FACE_GALLERY:
		WEBCAM_SOURCE_FACE_GALLERY.select(select_source_face, outputs = WEBCAM_SOURCE_FACE_GALLERY)
	if WEBCAM_CAMERA_BACKEND_DROPDOWN and WEBCAM_DEVICE_ID_DROPDOWN:
		WEBCAM_CAMERA_BACKEND_DROPDOWN.change(update_camera_backend, inputs = WEBCAM_CAMERA_BACKEND_DROPDOWN, outputs = WEBCAM_DEVICE_ID_DROPDOWN)


# Applied when the window opens, so a change while one is already running takes effect on the next
# start rather than resizing it underneath the viewer
def update_window_fullscreen(window_fullscreen : bool) -> None:
	state_manager.set_item('webcam_window_fullscreen', window_fullscreen)


def update_performance_overlay(performance_overlay : str) -> None:
	state_manager.set_item('webcam_performance_overlay', performance_overlay)


def update_frame_skipping(frame_skipping : str) -> None:
	state_manager.set_item('webcam_frame_skipping', frame_skipping)


# Read per iteration by the capture loop, so a change applies to the running stream
def update_content_analysis_interval(content_analysis_interval : float) -> None:
	state_manager.set_item('webcam_content_analysis_interval', int(content_analysis_interval))


# Only names that are both real processors and in the chain the caller names, so a config written
# with a different processor selection cannot bypass something the chain never runs
def resolve_bypassed_processors(bypassed_processors : Optional[List[str]], processors : Optional[List[str]]) -> List[str]:
	return [ processor for processor in bypassed_processors or [] if processor in WEBCAM_BYPASSED_PROCESSORS and processor in (processors or []) ]


# Read per frame by the capture loop, so a change applies to the running stream without touching
# the inference pools: the model stays on the device and the next frame simply skips it
def update_bypassed_processors(bypassed_processors : List[str]) -> None:
	state_manager.set_item('webcam_bypassed_processors', resolve_bypassed_processors(bypassed_processors, state_manager.get_item('processors')))


# The processor selection is what can be bypassed, so the choices follow it, and a processor that
# leaves the chain drops out of the bypass list with it. The new selection arrives as the event's
# own value rather than being read back from state: processors.update_processors writes that state
# from a separate listener on this same event, and two listeners on one event have no order
def remote_update_bypassed_processors(processors : List[str]) -> gradio.CheckboxGroup:
	bypassed_processors = resolve_bypassed_processors(state_manager.get_item('webcam_bypassed_processors'), processors)

	state_manager.set_item('webcam_bypassed_processors', bypassed_processors)
	return gradio.CheckboxGroup(value = bypassed_processors, choices = processors)


# Faces uploaded through the browser live in a temporary directory that no later run can resolve, so
# only groups loaded from a local config are carried into the generated one. Groups that came from
# the face directory are left out too, since the directory path already reproduces them
def generate_config(webcam_device_id : str, webcam_mode : str, webcam_resolution : str, webcam_fps : float, webcam_performance_overlay : str, webcam_frame_skipping : str, webcam_content_analysis_interval : float, webcam_inline_preview_resolution : str, webcam_execution_thread_count : float, webcam_camera_backend : str, webcam_output_resolution : str) -> gradio.Textbox:
	webcam_config = load_webcam_config(state_manager.get_item('webcam_config'))
	source_face_directory = webcam_config.get('webcam_source_faces_directory')
	directory_source_faces = resolve_source_face_directory(source_face_directory)
	source_face_groups = state_manager.get_item('source_face_groups') or {}
	settings = {}

	for state_key in WEBCAM_STATE_KEYS:
		if state_key == 'source_paths' and state_manager.get_item('source_paths_from_ui'):
			continue
		settings[state_key] = state_manager.get_item(state_key)

	webcam_options =\
	{
		'auto_start': webcam_config.get('auto_start'),
		'device_id': int(webcam_device_id) if str(webcam_device_id).isdigit() else 0,
		'mode': webcam_mode,
		'headless_mode': webcam_config.get('headless_mode'),
		'resolution': webcam_resolution,
		'fps': int(webcam_fps),
		'frame_skipping': webcam_frame_skipping,
		'content_analysis_interval': int(webcam_content_analysis_interval),
		'bypassed_processors': state_manager.get_item('webcam_bypassed_processors') or [],
		'performance_overlay': webcam_performance_overlay,
		'webcam_inline_preview_resolution': webcam_inline_preview_resolution,
		'webcam_execution_thread_count': int(webcam_execution_thread_count),
		'webcam_camera_backend': webcam_camera_backend,
		'webcam_output_resolution': webcam_output_resolution,
		'webcam_window_fullscreen': bool(state_manager.get_item('webcam_window_fullscreen')),
		'webcam_source_faces': { face_name: source_paths for face_name, source_paths in source_face_groups.items() if face_name not in directory_source_faces },
		'webcam_source_faces_directory': source_face_directory,
		'webcam_source_face': state_manager.get_item('source_face_name')
	}
	webcam_config_content = format_webcam_config(webcam_options, settings)

	if webcam_config_content:
		return gradio.Textbox(value = webcam_config_content, visible = True)
	return gradio.Textbox(value = translator.get('uis.webcam_generate_config_failed'), visible = True)


# One entry per face, pairing the name it is selected by with the image that stands for it. The first
# image stands in for a group of several photos, since they are averaged into one identity anyway
def create_source_face_entries() -> List[Tuple[str, str]]:
	source_face_entries = []

	for face_name, source_paths in resolve_source_face_groups().items():
		source_path = get_first(source_paths)
		if source_path and is_image(source_path):
			source_face_entries.append((face_name, source_path))

	return source_face_entries


# selected_index alone is not a reliable cue once previews are off, so the selected face is marked in
# its own caption and named in the label, which no theme can style away
def create_source_face_items(source_face_name : Optional[str]) -> List[Tuple[str, str]]:
	return [ (source_path, SOURCE_FACE_SELECTED_MARKER + face_name if face_name == source_face_name else face_name) for face_name, source_path in create_source_face_entries() ]


# An unavailable window is dropped from the choices rather than offered and then refused on start.
# A config that asks for it is warned about once and falls back, so a profile written on a machine
# that has pygame still opens on one that does not
def resolve_webcam_modes(webcam_mode : Optional[str]) -> Tuple[List[WebcamMode], Optional[str]]:
	if has_window_output():
		return list(uis_choices.webcam_modes), webcam_mode
	webcam_modes = [ mode for mode in uis_choices.webcam_modes if mode != 'window' ]

	if webcam_mode == 'window':
		log_window_output_requirement()
		return webcam_modes, get_first(webcam_modes)

	return webcam_modes, webcam_mode


# Repairs the selection before anything renders it: a face that disappeared, or one named by a config
# that holds no such face, cannot stay selected or the caption would mark a face that is not shown.
# No source face is a selection rather than a face, so it is not in the gallery and must survive
# the repair instead of being treated as a face that disappeared
def resolve_source_face_selection() -> Tuple[Optional[str], List[str]]:
	source_face_names = [ face_name for face_name, _ in create_source_face_entries() ]
	source_face_name = state_manager.get_item('source_face_name')

	if source_face_name != NO_SOURCE_FACE_NAME and source_face_name not in source_face_names:
		source_face_name = get_first(source_face_names) or DEFAULT_SOURCE_FACE_NAME
		state_manager.set_item('source_face_name', source_face_name)

	return source_face_name, source_face_names


def resolve_source_face_label(source_face_name : Optional[str], source_face_names : List[str]) -> str:
	if source_face_name == NO_SOURCE_FACE_NAME or source_face_name in source_face_names:
		return translator.get('uis.webcam_source_face_gallery') + ': ' + source_face_name
	return translator.get('uis.webcam_source_face_gallery')


# Rebuilt after every upload, clear or selection, so the marker, the label and the highlight all
# describe the same face
def update_source_face_gallery() -> gradio.Gallery:
	source_face_name, source_face_names = resolve_source_face_selection()

	return gradio.Gallery(
		label = resolve_source_face_label(source_face_name, source_face_names),
		value = create_source_face_items(source_face_name),
		selected_index = source_face_names.index(source_face_name) if source_face_name in source_face_names else None
	)


# The streamer reads the selected name on its next frame and rebinds inputs prepared at startup, so
# the swap needs no inference and no restart
def select_source_face(event : gradio.SelectData) -> gradio.Gallery:
	source_face_entries = create_source_face_entries()

	if 0 <= event.index < len(source_face_entries):
		state_manager.set_item('source_face_name', source_face_entries[event.index][0])
	return update_source_face_gallery()


def update_camera_backend(camera_backend : str) -> gradio.Dropdown:
	local_camera_ids = detect_local_camera_ids(0, 10, camera_backend) or [ 'none' ] #type:ignore[list-item]
	return gradio.Dropdown(value = get_first(local_camera_ids), choices = local_camera_ids)
