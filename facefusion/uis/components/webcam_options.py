from typing import Optional

import gradio

from facefusion import state_manager, translator
from facefusion.camera_manager import detect_local_camera_ids
from facefusion.common_helper import get_first
from facefusion.uis import choices as uis_choices
from facefusion.uis.core import register_ui_component
from facefusion.webcam_config import WEBCAM_FRAME_SKIPPING_MODES, WEBCAM_INLINE_PREVIEW_RESOLUTIONS, WEBCAM_PERFORMANCE_OVERLAYS, load_webcam_config

WEBCAM_DEVICE_ID_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_MODE_RADIO : Optional[gradio.Radio] = None
WEBCAM_RESOLUTION_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_FPS_SLIDER : Optional[gradio.Slider] = None
WEBCAM_PERFORMANCE_OVERLAY_RADIO : Optional[gradio.Radio] = None
WEBCAM_FRAME_SKIPPING_RADIO : Optional[gradio.Radio] = None
WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_EXECUTION_THREAD_COUNT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global WEBCAM_DEVICE_ID_DROPDOWN
	global WEBCAM_MODE_RADIO
	global WEBCAM_RESOLUTION_DROPDOWN
	global WEBCAM_FPS_SLIDER
	global WEBCAM_PERFORMANCE_OVERLAY_RADIO
	global WEBCAM_FRAME_SKIPPING_RADIO
	global WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN
	global WEBCAM_EXECUTION_THREAD_COUNT_SLIDER

	webcam_config = load_webcam_config(state_manager.get_item('webcam_config'))
	local_camera_ids = detect_local_camera_ids(0, 10) or [ 'none' ] #type:ignore[list-item]
	webcam_device_id = webcam_config.get('device_id')
	if webcam_device_id not in local_camera_ids:
		webcam_device_id = get_first(local_camera_ids)
	state_manager.init_item('webcam_frame_skipping', webcam_config.get('frame_skipping'))
	state_manager.init_item('webcam_performance_overlay', webcam_config.get('performance_overlay'))
	WEBCAM_DEVICE_ID_DROPDOWN = gradio.Dropdown(
		value = webcam_device_id,
		label = translator.get('uis.webcam_device_id_dropdown'),
		choices = local_camera_ids
	)
	WEBCAM_MODE_RADIO = gradio.Radio(
		label = translator.get('uis.webcam_mode_radio'),
		choices = uis_choices.webcam_modes,
		value = webcam_config.get('mode')
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
	WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.webcam_inline_preview_resolution_dropdown'),
		choices = WEBCAM_INLINE_PREVIEW_RESOLUTIONS,
		value = webcam_config.get('inline_preview_resolution')
	)
	WEBCAM_EXECUTION_THREAD_COUNT_SLIDER = gradio.Slider(
		label = translator.get('uis.webcam_execution_thread_count_slider'),
		value = webcam_config.get('execution_thread_count'),
		step = 1,
		minimum = 1,
		maximum = 8
	)
	register_ui_component('webcam_device_id_dropdown', WEBCAM_DEVICE_ID_DROPDOWN)
	register_ui_component('webcam_mode_radio', WEBCAM_MODE_RADIO)
	register_ui_component('webcam_resolution_dropdown', WEBCAM_RESOLUTION_DROPDOWN)
	register_ui_component('webcam_fps_slider', WEBCAM_FPS_SLIDER)
	register_ui_component('webcam_performance_overlay_radio', WEBCAM_PERFORMANCE_OVERLAY_RADIO)
	register_ui_component('webcam_frame_skipping_radio', WEBCAM_FRAME_SKIPPING_RADIO)
	register_ui_component('webcam_inline_preview_resolution_dropdown', WEBCAM_INLINE_PREVIEW_RESOLUTION_DROPDOWN)
	register_ui_component('webcam_execution_thread_count_slider', WEBCAM_EXECUTION_THREAD_COUNT_SLIDER)


def listen() -> None:
	if WEBCAM_PERFORMANCE_OVERLAY_RADIO:
		WEBCAM_PERFORMANCE_OVERLAY_RADIO.change(update_performance_overlay, inputs = WEBCAM_PERFORMANCE_OVERLAY_RADIO)
	if WEBCAM_FRAME_SKIPPING_RADIO:
		WEBCAM_FRAME_SKIPPING_RADIO.change(update_frame_skipping, inputs = WEBCAM_FRAME_SKIPPING_RADIO)


def update_performance_overlay(performance_overlay : str) -> None:
	state_manager.set_item('webcam_performance_overlay', performance_overlay)


def update_frame_skipping(frame_skipping : str) -> None:
	state_manager.set_item('webcam_frame_skipping', frame_skipping)
