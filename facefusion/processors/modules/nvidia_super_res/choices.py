from typing import List, Sequence, get_args

from facefusion.common_helper import create_int_range
from facefusion.processors.modules.nvidia_super_res.types import NvidiaSuperResLevel, NvidiaSuperResMode

nvidia_super_res_modes : List[NvidiaSuperResMode] = list(get_args(NvidiaSuperResMode))
nvidia_super_res_levels : List[NvidiaSuperResLevel] = list(get_args(NvidiaSuperResLevel))

# 1 keeps the resolution and only restores, 2 through 4 upscale. The upper end is memory bound
# rather than capability bound, so has_device_headroom() declines a scale that will not fit instead
# of the range being clipped here
nvidia_super_res_scale_range : Sequence[int] = create_int_range(1, 4, 1)
nvidia_super_res_strength_range : Sequence[int] = create_int_range(0, 100, 1)
nvidia_super_res_blend_range : Sequence[int] = create_int_range(0, 100, 1)
