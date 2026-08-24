import numpy

from facefusion import face_recognizer
from facefusion.processors.modules.face_enhancer import core as face_enhancer


def test_face_enhancer_preprocessing_is_float32_contiguous_and_equivalent():
	crop_frame = numpy.arange(60, dtype = numpy.uint8).reshape(4, 5, 3)
	expected = ((crop_frame[:, :, ::-1] / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(numpy.float32)

	prepared_frame = face_enhancer.prepare_crop_frame(crop_frame)

	assert prepared_frame.dtype == numpy.float32
	assert prepared_frame.flags.c_contiguous
	numpy.testing.assert_allclose(prepared_frame, expected, rtol = 0, atol = 1e-7)


def test_face_recognizer_preprocessing_is_float32_contiguous_and_equivalent(monkeypatch):
	crop_frame = numpy.arange(60, dtype = numpy.uint8).reshape(4, 5, 3)
	expected = (crop_frame / 127.5 - 1)[:, :, ::-1].transpose(2, 0, 1)[None].astype(numpy.float32)
	prepared_frames = []

	monkeypatch.setattr(face_recognizer, 'get_model_options', lambda: { 'template': 'arcface', 'size': (5, 4) })
	monkeypatch.setattr(face_recognizer, 'warp_face_by_face_landmark_5', lambda frame, landmarks, template, size: (crop_frame, None))
	monkeypatch.setattr(face_recognizer, 'forward', lambda frame: prepared_frames.append(frame) or numpy.ones((1, 4), dtype = numpy.float32))

	face_recognizer.calculate_face_embedding(crop_frame, numpy.zeros((5, 2), dtype = numpy.float32))
	prepared_frame = prepared_frames[0]
	assert prepared_frame.dtype == numpy.float32
	assert prepared_frame.flags.c_contiguous
	numpy.testing.assert_allclose(prepared_frame, expected, rtol = 0, atol = 1e-7)
