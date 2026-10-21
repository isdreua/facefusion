import threading
from functools import lru_cache
from typing import Any, Dict, Optional, Set, Tuple

import numpy

from facefusion import logger, state_manager
from facefusion.inference_timer import measure_inference
from facefusion.types import VisionFrame

NVIDIA_VFX_RESOLVED : bool = False
NVIDIA_VFX_WARNED : Set[str] = set()
# Effects are cached per worker thread, so a clear from the ui thread cannot reach them. Bumping a
# generation counter instead lets every worker notice and rebuild on its next frame
NVIDIA_VFX_GENERATION : int = 0

# QualityLevel exposes BICUBIC, a bare LOW/MEDIUM/HIGH/ULTRA ladder for plain upscaling, and
# prefixed ladders for the other modes. STREAMING only ships MEDIUM and ULTRA
NVIDIA_SUPER_RES_MODE_PREFIXES : Dict[str, str] =\
{
	'standard': '',
	'denoise': 'DENOISE_',
	'deblur': 'DEBLUR_',
	'high-bitrate': 'HIGHBITRATE_',
	'streaming': 'STREAMING_'
}
NVIDIA_FRAME_GENERATION_MODE_VALUES : Dict[str, int] =\
{
	'low': 0,
	'medium': 1,
	'high': 2
}


class NvidiaVfxContext(threading.local):
	pass


NVIDIA_VFX_CONTEXT = NvidiaVfxContext()


def warn_once(warning_key : str, message : str) -> None:
	if warning_key in NVIDIA_VFX_WARNED:
		return

	NVIDIA_VFX_WARNED.add(warning_key)
	logger.warn(message, __name__)


# The pip distribution is nvidia-vfx but the import name is nvvfx, which is not derivable from the
# package name, so both are tried
@lru_cache()
def resolve_vfx_module() -> Any:
	for module_name in [ 'nvvfx', 'nvidia_vfx' ]:
		try:
			return __import__(module_name, fromlist = [ '*' ])
		except Exception:
			continue
	return None


# nvidia-vfx accepts only CUDA DLPack arrays and rejects host numpy, so an array library that can
# put a frame on the device is mandatory rather than optional
@lru_cache()
def resolve_array_module() -> Any:
	for module_name in [ 'cupy', 'torch' ]:
		try:
			return __import__(module_name)
		except Exception:
			continue
	return None


def has_nvidia_vfx() -> bool:
	return NVIDIA_VFX_RESOLVED


# Resolved once at startup so the hot path never re-checks and an unavailable warning cannot repeat
# per frame
def resolve_nvidia_vfx() -> bool:
	global NVIDIA_VFX_RESOLVED

	vfx_module = resolve_vfx_module()
	if not vfx_module:
		NVIDIA_VFX_RESOLVED = False
		return NVIDIA_VFX_RESOLVED

	array_module = resolve_array_module()
	if not array_module:
		logger.warn('nvidia-vfx is installed but no cuda array library was found; it accepts only cuda dlpack arrays, so install cupy (pip install cupy-cuda12x) or pytorch', __name__)
		NVIDIA_VFX_RESOLVED = False
		return NVIDIA_VFX_RESOLVED

	NVIDIA_VFX_RESOLVED = True
	logger.info('nvidia-vfx ' + str(getattr(vfx_module, '__version__', 'unknown')) + ' available through ' + array_module.__name__, __name__)
	return NVIDIA_VFX_RESOLVED


def log_nvidia_vfx_requirement() -> None:
	logger.warn('this processor needs the nvidia video effects bindings, which are not installed; run "pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com" plus "pip install cupy-cuda12x", on an nvidia gpu with tensor cores (turing or newer) and driver 570.190+/580.82+/590.44+ on linux', __name__)


def resolve_quality_level(super_res_mode : str, super_res_level : str) -> Any:
	vfx_module = resolve_vfx_module()

	if not vfx_module:
		return None
	if super_res_mode == 'bicubic':
		return getattr(vfx_module.QualityLevel, 'BICUBIC', None)

	mode_prefix = NVIDIA_SUPER_RES_MODE_PREFIXES.get(super_res_mode)
	if mode_prefix is None:
		return None
	return getattr(vfx_module.QualityLevel, mode_prefix + super_res_level.upper(), None)


# Mode is not a top-level export of nvvfx 0.2.0.0, so it is looked up where the binding actually
# keeps it and finally falls back to the documented integer values
def resolve_frame_generation_mode(frame_generation_mode : str) -> Any:
	vfx_module = resolve_vfx_module()
	mode_name = frame_generation_mode.upper()

	for mode_holder in [ vfx_module, getattr(vfx_module, 'effects', None), getattr(vfx_module, 'VideoFrameGeneration', None) ]:
		mode_enum = getattr(mode_holder, 'Mode', None) if mode_holder else None
		if mode_enum is not None and hasattr(mode_enum, mode_name):
			return getattr(mode_enum, mode_name)
	return NVIDIA_FRAME_GENERATION_MODE_VALUES.get(frame_generation_mode, 1)


def resolve_effects() -> Dict[Any, Any]:
	effects = getattr(NVIDIA_VFX_CONTEXT, 'effects', None)

	if effects is None or getattr(NVIDIA_VFX_CONTEXT, 'generation', None) != NVIDIA_VFX_GENERATION:
		close_effects(effects)
		effects = {}
		NVIDIA_VFX_CONTEXT.effects = effects
		NVIDIA_VFX_CONTEXT.generation = NVIDIA_VFX_GENERATION
	return effects


def close_effects(effects : Optional[Dict[Any, Any]]) -> None:
	if effects:
		# close() may release the sdk buffer without stream ordering, so nothing may still be in flight
		synchronize_device()

	for effect in (effects or {}).values():
		try:
			effect.close()
		except Exception:
			pass

	if effects:
		release_device_memory()


# cupy holds freed blocks in its own pool rather than returning them to the driver, which starves
# the onnx runtime arenas on the same device. Switching scale or quality has to hand that back
def release_device_memory() -> None:
	array_module = resolve_array_module()

	try:
		if array_module.__name__ == 'cupy':
			array_module.get_default_memory_pool().free_all_blocks()
		else:
			array_module.cuda.empty_cache()
	except Exception:
		pass


# Called when an option that is fixed at load() time changes, so every worker rebuilds its effects
def invalidate_nvidia_vfx_effects() -> None:
	global NVIDIA_VFX_GENERATION

	NVIDIA_VFX_GENERATION += 1


def resolve_device_memory() -> Tuple[int, int]:
	array_module = resolve_array_module()

	try:
		if array_module.__name__ == 'cupy':
			return array_module.cuda.runtime.memGetInfo()
		return array_module.cuda.mem_get_info()
	except Exception:
		return 0, 0


# A super resolution pass holds the sdk output, our copy of it and the conversion intermediates at
# once, and every worker thread keeps its own effect. At scale 3 on a 1080p frame that is roughly
# 600 MB per worker before the sdk's own working buffers, against onnx runtime arenas on the same
# device. An unchecked allocation failure inside the sdk surfaces as an illegal memory access rather
# than an out of memory error, taking the cuda context and every onnx session with it, so the cost
# is estimated up front and declined when it clearly does not fit
def estimate_super_res_bytes(width : int, height : int, super_res_scale : int) -> int:
	input_bytes = 3 * width * height * 4
	output_bytes = 3 * (width * super_res_scale) * (height * super_res_scale) * 4

	return input_bytes * 2 + output_bytes * 3


def has_device_headroom(required_bytes : int) -> bool:
	free_bytes, total_bytes = resolve_device_memory()

	if not free_bytes or not total_bytes:
		return True
	if required_bytes < free_bytes * 0.5:
		return True

	warn_once('headroom_' + str(required_bytes), 'nvidia-vfx needs about ' + str(required_bytes // (1024 * 1024)) + ' MB for this configuration but only ' + str(free_bytes // (1024 * 1024)) + ' MB of ' + str(total_bytes // (1024 * 1024)) + ' MB is free, and the onnx sessions need the rest; skipping the effect. Lower the scale or the resolution')
	return False


# Effect handles carry no documented thread safety guarantee and the streaming executor runs
# webcam_execution_thread_count workers, so every worker builds and keeps its own
def resolve_super_res_effect(quality_level : Any, strength : float, output_width : int, output_height : int) -> Any:
	effects = resolve_effects()
	effect_key = ('super_res', str(quality_level), round(strength, 3), output_width, output_height)

	if effect_key not in effects:
		vfx_module = resolve_vfx_module()
		execution_device_ids = state_manager.get_item('execution_device_ids') or [ 0 ]
		effect = vfx_module.VideoSuperRes(quality_level, strength, execution_device_ids[0])
		# Scale is expressed through the output dimensions, which must be set before load()
		effect.output_width = output_width
		effect.output_height = output_height
		effect.load()
		effects[effect_key] = effect
	return effects.get(effect_key)


def resolve_frame_generation_effect(mode : Any, shot_change_detection : bool) -> Any:
	effects = resolve_effects()
	effect_key = ('frame_generation', str(mode), shot_change_detection)

	if effect_key not in effects:
		vfx_module = resolve_vfx_module()
		execution_device_ids = state_manager.get_item('execution_device_ids') or [ 0 ]
		effect = vfx_module.VideoFrameGeneration(mode, shot_change_detection, execution_device_ids[0])
		effect.load()
		effects[effect_key] = effect
	return effects.get(effect_key)


def clear_nvidia_vfx_effects() -> None:
	close_effects(getattr(NVIDIA_VFX_CONTEXT, 'effects', None))
	NVIDIA_VFX_CONTEXT.effects = {}
	invalidate_nvidia_vfx_effects()


# The sdk documents that its buffers are released and replaced by host operations WITHOUT cuda
# stream ordering, and that the producer stream must be synchronized before run() when the input was
# written on a different stream than the sdk's stream_ptr. We build inputs with cupy or torch on
# their own stream and run with the default stream_ptr, so every boundary needs an explicit
# synchronize. Skipping any of them corrupts the cuda context, which then surfaces as an illegal
# memory access inside whatever touches the gpu next, usually an unrelated onnx runtime session
def synchronize_device() -> None:
	array_module = resolve_array_module()

	try:
		if array_module.__name__ == 'cupy':
			array_module.cuda.runtime.deviceSynchronize()
		else:
			array_module.cuda.synchronize()
	except Exception as exception:
		warn_once('synchronize', 'nvidia-vfx device synchronize failed: ' + str(exception))


# The frame is uploaded as-is and every conversion runs on the device, so a frame costs one host to
# device copy rather than a full frame transpose and scale on the cpu first
def upload_vision_frame(vision_frame : VisionFrame) -> Any:
	array_module = resolve_array_module()

	if array_module.__name__ == 'cupy':
		device_vision_frame = array_module.asarray(numpy.ascontiguousarray(vision_frame))
	else:
		device_vision_frame = array_module.from_numpy(numpy.ascontiguousarray(vision_frame)).cuda()

	# RGB8 wants (3, H, W) float32 in [0, 1], while facefusion frames are (H, W, 3) uint8 bgr
	device_vision_frame = device_vision_frame[:, :, ::-1].transpose(2, 0, 1)
	if array_module.__name__ == 'cupy':
		device_vision_frame = array_module.ascontiguousarray(device_vision_frame).astype(array_module.float32) / 255.0
	else:
		device_vision_frame = device_vision_frame.contiguous().float() / 255.0
	return device_vision_frame


# Takes the whole output object rather than a bare capsule, so the wrapper that owns the sdk buffer
# stays referenced while the capsule is imported. The capsule is valid only while the effect and its
# current output buffer exist, and must not be retained across another run(), so it is copied here
# and the copy is awaited before the caller lets the effect move on
def download_vision_frame(effect_output : Any) -> Tuple[VisionFrame, float]:
	array_module = resolve_array_module()
	dlpack_capsule = effect_output.image if hasattr(effect_output, 'image') else effect_output

	# The sdk may still be writing the output buffer, so wait before importing it
	synchronize_device()

	if array_module.__name__ == 'cupy':
		device_vision_frame = array_module.from_dlpack(dlpack_capsule).copy()
		synchronize_device()
		deviation = float(device_vision_frame.std())
		device_vision_frame = (device_vision_frame * 255.0).clip(0, 255).astype(array_module.uint8)
		device_vision_frame = array_module.ascontiguousarray(device_vision_frame.transpose(1, 2, 0)[:, :, ::-1])
		return array_module.asnumpy(device_vision_frame), deviation

	device_vision_frame = array_module.from_dlpack(dlpack_capsule).clone()
	synchronize_device()
	deviation = float(device_vision_frame.std())
	device_vision_frame = (device_vision_frame * 255.0).clamp(0, 255).to(array_module.uint8)
	device_vision_frame = device_vision_frame.permute(1, 2, 0).flip(-1).contiguous()
	return device_vision_frame.cpu().numpy(), deviation


# The known failure on this sdk is silent rather than loud: on linux the bundled cuda kernels have
# no image for some architectures, notably blackwell sm_120, and super resolution then returns a
# constant image without raising. A constant result is therefore treated as a failure
def is_blank_result(deviation : float) -> bool:
	return deviation < 1e-6


def apply_super_res(vision_frame : VisionFrame, super_res_mode : str, super_res_level : str, super_res_scale : int, super_res_strength : float) -> Optional[VisionFrame]:
	if not has_nvidia_vfx():
		return None

	quality_level = resolve_quality_level(super_res_mode, super_res_level)
	if quality_level is None:
		warn_once('quality_' + super_res_mode + super_res_level, 'nvidia-vfx has no quality level for mode ' + super_res_mode + ' at level ' + super_res_level + '; leaving the frame untouched')
		return None

	height, width = vision_frame.shape[:2]
	if not has_device_headroom(estimate_super_res_bytes(width, height, super_res_scale)):
		return None

	try:
		effect = resolve_super_res_effect(quality_level, super_res_strength, width * super_res_scale, height * super_res_scale)
		device_vision_frame = upload_vision_frame(vision_frame)
		# The input was written on the array library's stream, so it has to land before run() reads it
		synchronize_device()
		# RGB8 stays in blocking mode on purpose: the sdk documents that its RGB8 conversion can run
		# on stream 0 while inference runs on the given stream, so non_blocking is unsafe here
		with measure_inference():
			effect_output = effect.run(device_vision_frame)
		result_vision_frame, deviation = download_vision_frame(effect_output)
		# Keeps the input and the output wrapper referenced until the copy above has completed
		del effect_output, device_vision_frame
	except Exception as exception:
		warn_once('super_res', 'nvidia-vfx super resolution failed, leaving the frame untouched: ' + str(exception))
		return None

	if is_blank_result(deviation):
		warn_once('super_res_blank', 'nvidia-vfx super resolution returned a constant image, which is the known silent failure where the bundled cuda kernels have no build for this gpu architecture; disabling it for this session')
		disable_nvidia_vfx()
		return None
	return result_vision_frame


def apply_frame_generation(previous_vision_frame : VisionFrame, vision_frame : VisionFrame, frame_generation_mode : str, shot_change_detection : bool = True) -> Optional[VisionFrame]:
	if not has_nvidia_vfx():
		return None

	try:
		effect = resolve_frame_generation_effect(resolve_frame_generation_mode(frame_generation_mode), shot_change_detection)
		device_previous_vision_frame = upload_vision_frame(previous_vision_frame)
		device_vision_frame = upload_vision_frame(vision_frame)
		synchronize_device()
		with measure_inference():
			effect_output = effect.run(device_previous_vision_frame, device_vision_frame)
		result_vision_frame, deviation = download_vision_frame(effect_output)
		del effect_output, device_vision_frame, device_previous_vision_frame
	except Exception as exception:
		warn_once('frame_generation', 'nvidia-vfx frame generation failed, no frame was interpolated: ' + str(exception) + '. It requires an ada or blackwell gpu')
		return None

	if is_blank_result(deviation):
		warn_once('frame_generation_blank', 'nvidia-vfx frame generation returned a constant image, which is the known silent failure where the bundled cuda kernels have no build for this gpu architecture; disabling it for this session')
		disable_nvidia_vfx()
		return None
	return result_vision_frame


def disable_nvidia_vfx() -> None:
	global NVIDIA_VFX_RESOLVED

	NVIDIA_VFX_RESOLVED = False
	clear_nvidia_vfx_effects()
