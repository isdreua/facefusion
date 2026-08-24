import threading

import numpy

from facefusion.face_helper import get_paste_in_place, paste_back, set_paste_in_place


def create_paste_inputs():
	temp_frame = numpy.zeros((4, 4, 3), dtype = numpy.uint8)
	crop_frame = numpy.full((4, 4, 3), 255, dtype = numpy.uint8)
	crop_mask = numpy.ones((4, 4), dtype = numpy.float32)
	affine_matrix = numpy.array([ [ 1, 0, 0 ], [ 0, 1, 0 ] ], dtype = numpy.float32)
	return temp_frame, crop_frame, crop_mask, affine_matrix


def test_paste_back_copies_by_default():
	temp_frame, crop_frame, crop_mask, affine_matrix = create_paste_inputs()
	original_frame = temp_frame.copy()

	paste_frame = paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)

	assert paste_frame is not temp_frame
	numpy.testing.assert_array_equal(temp_frame, original_frame)
	assert numpy.any(paste_frame)


def test_paste_back_can_mutate_owned_frame_in_place():
	temp_frame, crop_frame, crop_mask, affine_matrix = create_paste_inputs()
	set_paste_in_place(True)
	try:
		paste_frame = paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)
	finally:
		set_paste_in_place(False)

	assert paste_frame is temp_frame
	assert numpy.any(temp_frame)


def test_paste_mode_is_thread_local():
	set_paste_in_place(True)
	try:
		worker_values = []
		worker = threading.Thread(target = lambda: worker_values.append(get_paste_in_place()))
		worker.start()
		worker.join()

		assert worker_values == [ False ]
		assert get_paste_in_place() is True
	finally:
		set_paste_in_place(False)
