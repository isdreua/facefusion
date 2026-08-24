import numpy

from facefusion import face_detector


def test_detect_faces_skips_zero_margin_padding_and_preserves_input(monkeypatch):
	vision_frame = numpy.arange(48, dtype = numpy.uint8).reshape(4, 4, 3)
	original_frame = vision_frame.copy()
	bounding_box = numpy.array([ 3, 2, 1, 0 ], dtype = numpy.float32)
	landmarks = numpy.array([ [ 1, 1 ] ] * 5, dtype = numpy.float32)
	seen_frames = []

	monkeypatch.setattr(face_detector, 'prepare_margin', lambda frame: (0, 0, 0, 0))
	monkeypatch.setattr(face_detector.state_manager, 'get_item', lambda key: 'retinaface' if key == 'face_detector_model' else '640x640')
	monkeypatch.setattr(face_detector, 'detect_with_retinaface', lambda frame, size: (seen_frames.append(frame) or [ bounding_box ], [ 0.9 ], [ landmarks ]))

	bounding_boxes, _, detected_landmarks = face_detector.detect_faces(vision_frame)

	assert len(seen_frames) == 1
	assert seen_frames[0] is vision_frame
	numpy.testing.assert_array_equal(bounding_boxes[0], [ 1, 0, 3, 2 ])
	assert detected_landmarks[0] is landmarks
	numpy.testing.assert_array_equal(vision_frame, original_frame)


def test_detect_faces_preserves_nonzero_margin_coordinates(monkeypatch):
	vision_frame = numpy.zeros((4, 4, 3), dtype = numpy.uint8)
	bounding_box = numpy.array([ 5, 6, 3, 4 ], dtype = numpy.float32)
	landmarks = numpy.array([ [ 3, 4 ] ] * 5, dtype = numpy.float32)

	monkeypatch.setattr(face_detector, 'prepare_margin', lambda frame: (2, 0, 0, 1))
	monkeypatch.setattr(face_detector.state_manager, 'get_item', lambda key: 'retinaface' if key == 'face_detector_model' else '640x640')
	monkeypatch.setattr(face_detector, 'detect_with_retinaface', lambda frame, size: ([ bounding_box ], [ 0.9 ], [ landmarks ]))

	bounding_boxes, _, detected_landmarks = face_detector.detect_faces(vision_frame)

	numpy.testing.assert_array_equal(bounding_boxes[0], [ 2, 2, 4, 4 ])
	numpy.testing.assert_array_equal(detected_landmarks[0], [ [ 2, 2 ] ] * 5)
