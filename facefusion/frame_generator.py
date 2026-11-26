import time
from typing import Any, Dict, Optional, Tuple

from facefusion import logger, nvidia_vfx, state_manager
from facefusion.types import VisionFrame
from facefusion.webcam_config import WEBCAM_FRAME_GENERATION_MODES

# apply_frame_generation swallows its own exceptions and warns once, so an effect that cannot be built
# on this gpu would otherwise be rebuilt and fail again on every frame for the rest of the stream
FRAME_GENERATION_FAILURE_LIMIT = 3


# Owned by the one loop that consumes multi_process_capture and hands frames to the writer, never by
# the processing workers: those finish out of order and in parallel, while an interpolated frame is
# only meaningful between two frames that are adjacent in delivery order. One instance per stream,
# called from that one thread, so it carries no lock of its own; the sdk call underneath is still
# serialized against the super resolution workers by NVIDIA_VFX_LOCK
class StreamFrameGenerator:
	def __init__(self) -> None:
		# The last frame as already uploaded to the device, with its shape; the next interpolation
		# starts from it, so each frame is uploaded once rather than twice
		self.previous_device_vision_frame : Optional[Any] = None
		self.previous_frame_shape : Optional[Tuple[int, ...]] = None
		self.is_resolved = False
		self.failure_count = 0
		self.stats : Dict[str, Any] = { 'mode': 'none', 'generated': 0, 'skipped': 0, 'last_ms': 0.0 }

	# Read per frame like frame skipping, so switching between the quality modes applies to the running
	# stream. The output rate is still fixed when the stream starts, see resolve_output_fps
	def generate(self, vision_frame : VisionFrame, is_duplicate : bool = False) -> Optional[VisionFrame]:
		frame_generation_mode = state_manager.get_item('webcam_frame_generation')
		self.stats['mode'] = frame_generation_mode if frame_generation_mode in WEBCAM_FRAME_GENERATION_MODES else 'none'

		# Reported as failed rather than as the mode asked for, so the overlay does not claim it is running
		if self.stats.get('mode') != 'none' and self.failure_count >= FRAME_GENERATION_FAILURE_LIMIT:
			self.stats['mode'] = 'failed'
		if self.stats.get('mode') not in [ 'none', 'failed' ] and not self.resolve_available():
			self.stats['mode'] = 'unavailable'
		if self.stats.get('mode') in [ 'none', 'failed', 'unavailable' ]:
			self.previous_device_vision_frame = None
			self.previous_frame_shape = None
			return None
		has_previous_frame = self.previous_device_vision_frame is not None and self.previous_frame_shape == vision_frame.shape
		# A re-delivered frame is identical to the one before it, so there is no motion to interpolate,
		# and the frame already on the device stands for it
		if is_duplicate and has_previous_frame:
			self.stats['skipped'] += 1
			return None
		# Nothing to interpolate from yet (the first frame, or a camera switched to another size): this
		# frame goes on the device for the next one. The device copy is taken before the overlay is
		# drawn into the frame in place, so no host copy of the frame is needed to keep the hud out
		if not has_previous_frame:
			device_vision_frame = nvidia_vfx.upload_frame_generation_input(vision_frame)
			self.keep_previous_frame(device_vision_frame, vision_frame)
			self.stats['skipped'] += 1
			# A failed upload counts like a failed generation, so a gpu that cannot take the frame
			# still reaches the limit instead of being retried on every frame
			if device_vision_frame is None:
				self.count_failure()
			return None

		generate_started = time.perf_counter()
		generated_vision_frame, device_vision_frame = nvidia_vfx.generate_frame(self.previous_device_vision_frame, vision_frame, self.stats.get('mode'))
		self.stats['last_ms'] = (time.perf_counter() - generate_started) * 1000
		self.keep_previous_frame(device_vision_frame, vision_frame)

		if generated_vision_frame is None or generated_vision_frame.shape != vision_frame.shape:
			self.stats['skipped'] += 1
			self.count_failure()
			return None
		self.failure_count = 0
		self.stats['generated'] += 1
		return generated_vision_frame

	def count_failure(self) -> None:
		self.failure_count += 1
		if self.failure_count == FRAME_GENERATION_FAILURE_LIMIT:
			logger.warn('frame generation failed ' + str(FRAME_GENERATION_FAILURE_LIMIT) + ' times in a row and is off for the rest of this stream; real frames are still delivered', __name__)

	# A failed upload or generation keeps nothing, so the next frame starts over from its own upload
	def keep_previous_frame(self, device_vision_frame : Optional[Any], vision_frame : VisionFrame) -> None:
		self.previous_device_vision_frame = device_vision_frame
		self.previous_frame_shape = vision_frame.shape if device_vision_frame is not None else None

	# Resolved once: a later failure or blank result disables the sdk for the session, and that must
	# not be undone by resolving it again on the next frame
	def resolve_available(self) -> bool:
		if not self.is_resolved:
			self.is_resolved = True
			if not nvidia_vfx.has_nvidia_vfx() and not nvidia_vfx.resolve_nvidia_vfx():
				nvidia_vfx.log_nvidia_vfx_requirement()
		return nvidia_vfx.has_nvidia_vfx()

	def get_stats(self) -> Dict[str, Any]:
		return dict(self.stats)


# Every real frame is followed by one generated frame, so the transport is opened at twice the
# camera rate. Fixed at start because ffmpeg and the virtual camera take their rate once
def resolve_output_fps(webcam_fps : float) -> float:
	if state_manager.get_item('webcam_frame_generation') in WEBCAM_FRAME_GENERATION_MODES[1:]:
		return webcam_fps * 2
	return webcam_fps
