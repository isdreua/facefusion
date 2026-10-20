from typing import List, Optional, Tuple

import gradio

from facefusion import nvidia_vfx, state_manager, translator
from facefusion.common_helper import calculate_int_step
from facefusion.processors.modules.nvidia_super_res import choices as nvidia_super_res_choices
from facefusion.processors.modules.nvidia_super_res.types import NvidiaSuperResLevel, NvidiaSuperResMode
from facefusion.uis.core import get_ui_component, register_ui_component

NVIDIA_SUPER_RES_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
NVIDIA_SUPER_RES_LEVEL_DROPDOWN : Optional[gradio.Dropdown] = None
NVIDIA_SUPER_RES_SCALE_SLIDER : Optional[gradio.Slider] = None
NVIDIA_SUPER_RES_STRENGTH_SLIDER : Optional[gradio.Slider] = None
NVIDIA_SUPER_RES_BLEND_SLIDER : Optional[gradio.Slider] = None

LOCALE_PACKAGE = 'facefusion.processors.modules.nvidia_super_res'


def render() -> None:
	global NVIDIA_SUPER_RES_MODE_DROPDOWN
	global NVIDIA_SUPER_RES_LEVEL_DROPDOWN
	global NVIDIA_SUPER_RES_SCALE_SLIDER
	global NVIDIA_SUPER_RES_STRENGTH_SLIDER
	global NVIDIA_SUPER_RES_BLEND_SLIDER

	has_nvidia_super_res = 'nvidia_super_res' in state_manager.get_item('processors')
	NVIDIA_SUPER_RES_MODE_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.mode_dropdown', LOCALE_PACKAGE),
		choices = nvidia_super_res_choices.nvidia_super_res_modes,
		value = state_manager.get_item('nvidia_super_res_mode'),
		visible = has_nvidia_super_res
	)
	NVIDIA_SUPER_RES_LEVEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.level_dropdown', LOCALE_PACKAGE),
		choices = nvidia_super_res_choices.nvidia_super_res_levels,
		value = state_manager.get_item('nvidia_super_res_level'),
		visible = has_nvidia_super_res
	)
	NVIDIA_SUPER_RES_SCALE_SLIDER = gradio.Slider(
		label = translator.get('uis.scale_slider', LOCALE_PACKAGE),
		value = state_manager.get_item('nvidia_super_res_scale'),
		step = calculate_int_step(nvidia_super_res_choices.nvidia_super_res_scale_range),
		minimum = nvidia_super_res_choices.nvidia_super_res_scale_range[0],
		maximum = nvidia_super_res_choices.nvidia_super_res_scale_range[-1],
		visible = has_nvidia_super_res
	)
	NVIDIA_SUPER_RES_STRENGTH_SLIDER = gradio.Slider(
		label = translator.get('uis.strength_slider', LOCALE_PACKAGE),
		value = state_manager.get_item('nvidia_super_res_strength'),
		step = calculate_int_step(nvidia_super_res_choices.nvidia_super_res_strength_range),
		minimum = nvidia_super_res_choices.nvidia_super_res_strength_range[0],
		maximum = nvidia_super_res_choices.nvidia_super_res_strength_range[-1],
		visible = has_nvidia_super_res
	)
	NVIDIA_SUPER_RES_BLEND_SLIDER = gradio.Slider(
		label = translator.get('uis.blend_slider', LOCALE_PACKAGE),
		value = state_manager.get_item('nvidia_super_res_blend'),
		step = calculate_int_step(nvidia_super_res_choices.nvidia_super_res_blend_range),
		minimum = nvidia_super_res_choices.nvidia_super_res_blend_range[0],
		maximum = nvidia_super_res_choices.nvidia_super_res_blend_range[-1],
		visible = has_nvidia_super_res
	)
	register_ui_component('nvidia_super_res_mode_dropdown', NVIDIA_SUPER_RES_MODE_DROPDOWN)
	register_ui_component('nvidia_super_res_level_dropdown', NVIDIA_SUPER_RES_LEVEL_DROPDOWN)
	register_ui_component('nvidia_super_res_scale_slider', NVIDIA_SUPER_RES_SCALE_SLIDER)
	register_ui_component('nvidia_super_res_strength_slider', NVIDIA_SUPER_RES_STRENGTH_SLIDER)
	register_ui_component('nvidia_super_res_blend_slider', NVIDIA_SUPER_RES_BLEND_SLIDER)


def listen() -> None:
	NVIDIA_SUPER_RES_MODE_DROPDOWN.change(update_mode, inputs = NVIDIA_SUPER_RES_MODE_DROPDOWN)
	NVIDIA_SUPER_RES_LEVEL_DROPDOWN.change(update_level, inputs = NVIDIA_SUPER_RES_LEVEL_DROPDOWN)
	NVIDIA_SUPER_RES_SCALE_SLIDER.release(update_scale, inputs = NVIDIA_SUPER_RES_SCALE_SLIDER)
	NVIDIA_SUPER_RES_STRENGTH_SLIDER.release(update_strength, inputs = NVIDIA_SUPER_RES_STRENGTH_SLIDER)
	NVIDIA_SUPER_RES_BLEND_SLIDER.release(update_blend, inputs = NVIDIA_SUPER_RES_BLEND_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ NVIDIA_SUPER_RES_MODE_DROPDOWN, NVIDIA_SUPER_RES_LEVEL_DROPDOWN, NVIDIA_SUPER_RES_SCALE_SLIDER, NVIDIA_SUPER_RES_STRENGTH_SLIDER, NVIDIA_SUPER_RES_BLEND_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider]:
	has_nvidia_super_res = 'nvidia_super_res' in processors
	return gradio.Dropdown(visible = has_nvidia_super_res), gradio.Dropdown(visible = has_nvidia_super_res), gradio.Slider(visible = has_nvidia_super_res), gradio.Slider(visible = has_nvidia_super_res), gradio.Slider(visible = has_nvidia_super_res)


# Every option change rebuilds the effect, because quality, strength and output dimensions are all
# fixed at load() time rather than per run
def update_mode(nvidia_super_res_mode : NvidiaSuperResMode) -> None:
	state_manager.set_item('nvidia_super_res_mode', nvidia_super_res_mode)
	nvidia_vfx.invalidate_nvidia_vfx_effects()


def update_level(nvidia_super_res_level : NvidiaSuperResLevel) -> None:
	state_manager.set_item('nvidia_super_res_level', nvidia_super_res_level)
	nvidia_vfx.invalidate_nvidia_vfx_effects()


def update_scale(nvidia_super_res_scale : float) -> None:
	state_manager.set_item('nvidia_super_res_scale', int(nvidia_super_res_scale))
	nvidia_vfx.invalidate_nvidia_vfx_effects()


def update_strength(nvidia_super_res_strength : float) -> None:
	state_manager.set_item('nvidia_super_res_strength', int(nvidia_super_res_strength))
	nvidia_vfx.invalidate_nvidia_vfx_effects()


def update_blend(nvidia_super_res_blend : float) -> None:
	state_manager.set_item('nvidia_super_res_blend', int(nvidia_super_res_blend))
