from argparse import ArgumentParser
from types import ModuleType
from typing import List

import cv2

import facefusion.jobs.job_store
from facefusion import config, content_analyser, logger, nvidia_vfx, state_manager, translator
from facefusion.common_helper import create_int_metavar
from facefusion.filesystem import in_directory, is_image, is_video, same_file_extension
from facefusion.processors.modules.nvidia_super_res import choices as nvidia_super_res_choices
from facefusion.processors.modules.nvidia_super_res.types import NvidiaSuperResInputs
from facefusion.processors.types import ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.types import ApplyStateItem, Args, InferencePool, ProcessMode, VisionFrame
from facefusion.vision import blend_frame


# This processor is backed by the nvidia video effects sdk rather than onnx runtime, so it owns no
# inference pool. The two hooks exist because the processor interface requires them
def get_inference_pool() -> InferencePool:
	return {}


def clear_inference_pool() -> None:
	nvidia_vfx.clear_nvidia_vfx_effects()


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--nvidia-super-res-mode', help = translator.get('help.mode', __package__), default = config.get_str_value('processors', 'nvidia_super_res_mode', 'denoise'), choices = nvidia_super_res_choices.nvidia_super_res_modes)
		group_processors.add_argument('--nvidia-super-res-level', help = translator.get('help.level', __package__), default = config.get_str_value('processors', 'nvidia_super_res_level', 'medium'), choices = nvidia_super_res_choices.nvidia_super_res_levels)
		group_processors.add_argument('--nvidia-super-res-scale', help = translator.get('help.scale', __package__), type = int, default = config.get_int_value('processors', 'nvidia_super_res_scale', '1'), choices = nvidia_super_res_choices.nvidia_super_res_scale_range, metavar = create_int_metavar(nvidia_super_res_choices.nvidia_super_res_scale_range))
		group_processors.add_argument('--nvidia-super-res-strength', help = translator.get('help.strength', __package__), type = int, default = config.get_int_value('processors', 'nvidia_super_res_strength', '100'), choices = nvidia_super_res_choices.nvidia_super_res_strength_range, metavar = create_int_metavar(nvidia_super_res_choices.nvidia_super_res_strength_range))
		group_processors.add_argument('--nvidia-super-res-blend', help = translator.get('help.blend', __package__), type = int, default = config.get_int_value('processors', 'nvidia_super_res_blend', '100'), choices = nvidia_super_res_choices.nvidia_super_res_blend_range, metavar = create_int_metavar(nvidia_super_res_choices.nvidia_super_res_blend_range))
		facefusion.jobs.job_store.register_step_keys([ 'nvidia_super_res_mode', 'nvidia_super_res_level', 'nvidia_super_res_scale', 'nvidia_super_res_strength', 'nvidia_super_res_blend' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('nvidia_super_res_mode', args.get('nvidia_super_res_mode'))
	apply_state_item('nvidia_super_res_level', args.get('nvidia_super_res_level'))
	apply_state_item('nvidia_super_res_scale', args.get('nvidia_super_res_scale'))
	apply_state_item('nvidia_super_res_strength', args.get('nvidia_super_res_strength'))
	apply_state_item('nvidia_super_res_blend', args.get('nvidia_super_res_blend'))


def get_common_modules() -> List[ModuleType]:
	return [ content_analyser ]


def pre_check() -> bool:
	for common_module in get_common_modules():
		if not common_module.pre_check():
			return False

	if not nvidia_vfx.resolve_nvidia_vfx():
		nvidia_vfx.log_nvidia_vfx_requirement()
		return False
	return True


def pre_process(mode : ProcessMode) -> bool:
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False

	if not nvidia_vfx.has_nvidia_vfx():
		nvidia_vfx.log_nvidia_vfx_requirement()
		return False
	# Scaling above 1 only helps where the larger frame survives; a webcam frame is resized back to
	# the configured resolution immediately after processing, so the extra work is discarded
	if mode == 'stream' and state_manager.get_item('nvidia_super_res_scale') > 1:
		logger.warn('nvidia super res scale above 1 is discarded in stream mode, because the frame is resized back to the webcam resolution; use scale 1 with denoise or deblur instead', __name__)
	return True


def post_process() -> None:
	nvidia_vfx.clear_nvidia_vfx_effects()


def enhance_frame(temp_vision_frame : VisionFrame) -> VisionFrame:
	result_vision_frame = nvidia_vfx.apply_super_res(temp_vision_frame,
		state_manager.get_item('nvidia_super_res_mode'),
		state_manager.get_item('nvidia_super_res_level'),
		state_manager.get_item('nvidia_super_res_scale'),
		state_manager.get_item('nvidia_super_res_strength') / 100)

	if result_vision_frame is None:
		return temp_vision_frame
	return blend_merge_frame(temp_vision_frame, result_vision_frame)


def blend_merge_frame(temp_vision_frame : VisionFrame, merge_vision_frame : VisionFrame) -> VisionFrame:
	nvidia_super_res_blend = state_manager.get_item('nvidia_super_res_blend') / 100

	if nvidia_super_res_blend >= 1:
		return merge_vision_frame
	if temp_vision_frame.shape[:2] != merge_vision_frame.shape[:2]:
		temp_vision_frame = cv2.resize(temp_vision_frame, (merge_vision_frame.shape[1], merge_vision_frame.shape[0]))
	return blend_frame(temp_vision_frame, merge_vision_frame, nvidia_super_res_blend)


# The sdk keeps one effect per thread and its load() takes seconds, so the first frame each stream
# worker sees would otherwise stall the whole ordered output. The warmup runs the real path on a
# real frame: a constant frame would trip the blank result guard and disable the effect for the
# session
def warm_stream_worker(vision_frame : VisionFrame) -> None:
	enhance_frame(vision_frame)


def process_frame(inputs : NvidiaSuperResInputs) -> ProcessorOutputs:
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	temp_vision_frame = enhance_frame(temp_vision_frame)

	# Scaling changes the frame geometry, so the mask has to follow it
	if temp_vision_mask.shape[:2] != temp_vision_frame.shape[:2]:
		temp_vision_mask = cv2.resize(temp_vision_mask, temp_vision_frame.shape[:2][::-1])
	return temp_vision_frame, temp_vision_mask
