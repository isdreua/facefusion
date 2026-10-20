from typing import List, Literal, TypedDict

from facefusion.types import Mask, VisionFrame

NvidiaSuperResInputs = TypedDict('NvidiaSuperResInputs',
{
	'target_vision_frames' : List[VisionFrame],
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

# bicubic ignores the level, standard is the plain upscaling ladder, and denoise/deblur are the
# same-resolution restoration modes that make sense at scale 1
NvidiaSuperResMode = Literal['standard', 'denoise', 'deblur', 'high-bitrate', 'streaming', 'bicubic']
NvidiaSuperResLevel = Literal['low', 'medium', 'high', 'ultra']
