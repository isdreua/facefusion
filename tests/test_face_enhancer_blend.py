import numpy
import pytest

from facefusion.face_helper import paste_back
from facefusion.vision import blend_frame


@pytest.mark.parametrize('blend', [ 0.0, 0.4, 1.0 ])
def test_mask_blend_matches_full_frame_blend_with_uint8_tolerance(blend):
	random = numpy.random.RandomState(7)
	temp_frame = random.randint(0, 256, (8, 8, 3), dtype = numpy.uint8)
	crop_frame = random.randint(0, 256, (8, 8, 3), dtype = numpy.uint8)
	crop_mask = random.uniform(0, 1, (8, 8)).astype(numpy.float32)
	affine_matrix = numpy.array([ [ 1, 0, 0 ], [ 0, 1, 0 ] ], dtype = numpy.float32)

	old_paste_frame = paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)
	old_blend_frame = blend_frame(temp_frame, old_paste_frame, blend)
	new_blend_frame = paste_back(temp_frame, crop_frame, crop_mask * blend, affine_matrix)

	numpy.testing.assert_allclose(new_blend_frame, old_blend_frame, rtol = 0, atol = 1)
