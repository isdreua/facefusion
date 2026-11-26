import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Set, Tuple

import numpy

from facefusion import face_store, state_manager
from facefusion.app_context import detect_app_context, set_app_context_override
from facefusion.common_helper import get_first, get_middle
from facefusion.face_classifier import classify_face
from facefusion.face_detector import detect_faces, detect_faces_by_angle, get_stream_worker_count
from facefusion.face_helper import apply_nms, average_points, convert_to_face_landmark_5, estimate_face_angle, get_nms_threshold
from facefusion.face_landmarker import detect_face_landmark, estimate_face_landmark_68_5
from facefusion.face_recognizer import calculate_face_embedding
from facefusion.inference_timer import add_inference_timer, get_inference_timer, reset_inference_timer
from facefusion.types import Angle, AppContext, BoundingBox, Face, FaceLandmark5, FaceLandmarkSet, FaceScoreSet, Score, VisionFrame
from facefusion.vision import is_vision_frame


class FaceAnalysisContext(threading.local):
	features : Optional[Set[str]] = None
	cache_bypass = False
	single_face = False


FACE_ANALYSIS_CONTEXT = FaceAnalysisContext()
# A ceiling rather than a size: the executor only starts a thread when no idle one is left, so it
# grows to the stream's workers times its extra angles and no further
FACE_DETECTOR_ANGLE_EXECUTOR = ThreadPoolExecutor(max_workers = 32, thread_name_prefix = 'face_detector_angle')


def set_face_analysis_features(face_analysis_features : Optional[Set[str]]) -> None:
	FACE_ANALYSIS_CONTEXT.features = face_analysis_features


def get_face_analysis_features() -> Optional[Set[str]]:
	return getattr(FACE_ANALYSIS_CONTEXT, 'features', None)


def set_face_cache_bypass(cache_bypass : bool) -> None:
	FACE_ANALYSIS_CONTEXT.cache_bypass = cache_bypass


def get_face_cache_bypass() -> bool:
	return getattr(FACE_ANALYSIS_CONTEXT, 'cache_bypass', False)


# Analyse only the face select_faces would pick in 'one' mode. Set by select_faces around a stream's
# own uncached target detection and nowhere else, so a list cut down to one face is never stored in
# the face store for a caller that expects every face
def set_single_face_analysis(single_face : bool) -> None:
	FACE_ANALYSIS_CONTEXT.single_face = single_face


def get_single_face_analysis() -> bool:
	return getattr(FACE_ANALYSIS_CONTEXT, 'single_face', False)


# The face that comes first in face_selector.sort_faces_by_order, found from the detector's own boxes
# and scores before any face is analysed. The same keys and the same stable sort, so it is the face
# that ordering the analysed faces would have put first
def resolve_first_face_index(keep_indices : List[int], bounding_boxes : List[BoundingBox], face_scores : List[Score], face_selector_order : Optional[str]) -> int:
	face_order_keys =\
	{
		'left-right': (lambda index: bounding_boxes[index][0], False),
		'right-left': (lambda index: bounding_boxes[index][0], True),
		'top-bottom': (lambda index: bounding_boxes[index][1], False),
		'bottom-top': (lambda index: bounding_boxes[index][1], True),
		'small-large': (lambda index: (bounding_boxes[index][2] - bounding_boxes[index][0]) * (bounding_boxes[index][3] - bounding_boxes[index][1]), False),
		'large-small': (lambda index: (bounding_boxes[index][2] - bounding_boxes[index][0]) * (bounding_boxes[index][3] - bounding_boxes[index][1]), True),
		'best-worst': (lambda index: face_scores[index], True),
		'worst-best': (lambda index: face_scores[index], False)
	}

	if face_selector_order in face_order_keys:
		face_order_key, is_reverse = face_order_keys[face_selector_order]
		return sorted(keep_indices, key = face_order_key, reverse = is_reverse)[0]
	return keep_indices[0]


def create_faces(vision_frame : VisionFrame, bounding_boxes : List[BoundingBox], face_scores : List[Score], face_landmarks_5 : List[FaceLandmark5]) -> List[Face]:
	faces = []
	nms_threshold = get_nms_threshold(state_manager.get_item('face_detector_model'), state_manager.get_item('face_detector_angles'))
	keep_indices = apply_nms(bounding_boxes, face_scores, state_manager.get_item('face_detector_score'), nms_threshold)
	# The faces 'one' would drop are not landmarked, embedded or classified: bystanders, a poster or a
	# television in view each cost those model passes on every frame only to be thrown away
	if get_single_face_analysis() and len(keep_indices) > 1:
		keep_indices = [ resolve_first_face_index(list(keep_indices), bounding_boxes, face_scores, state_manager.get_item('face_selector_order')) ]

	for index in keep_indices:
		bounding_box = bounding_boxes[index]
		face_score = face_scores[index]
		face_landmark_5 = face_landmarks_5[index]
		face_landmark_5_68 = face_landmark_5
		face_landmark_68_5 = estimate_face_landmark_68_5(face_landmark_5_68)
		face_landmark_68 = face_landmark_68_5
		face_landmark_score_68 = 0.0
		face_angle = estimate_face_angle(face_landmark_68_5)

		if state_manager.get_item('face_landmarker_score') > 0:
			face_landmark_68, face_landmark_score_68 = detect_face_landmark(vision_frame, bounding_box, face_angle)
		if face_landmark_score_68 > state_manager.get_item('face_landmarker_score'):
			face_landmark_5_68 = convert_to_face_landmark_5(face_landmark_68)

		face_landmark_set : FaceLandmarkSet =\
		{
			'5': face_landmark_5,
			'5/68': face_landmark_5_68,
			'68': face_landmark_68,
			'68/5': face_landmark_68_5
		}
		face_score_set : FaceScoreSet =\
		{
			'detector': face_score,
			'landmarker': face_landmark_score_68
		}
		face_analysis_features = get_face_analysis_features()
		if face_analysis_features is None or 'embedding' in face_analysis_features:
			face_embedding, face_embedding_norm = calculate_face_embedding(vision_frame, face_landmark_set.get('5/68'))
		else:
			face_embedding = numpy.zeros(512, dtype = numpy.float32)
			face_embedding_norm = numpy.zeros(512, dtype = numpy.float32)

		if face_analysis_features is None or 'demographics' in face_analysis_features:
			gender, age, race = classify_face(vision_frame, face_landmark_set.get('5/68'))
		else:
			gender, age, race = None, range(0), None

		faces.append(Face(
			origin = 'detect',
			bounding_box = bounding_box,
			score_set = face_score_set,
			landmark_set = face_landmark_set,
			angle = face_angle,
			embedding = face_embedding,
			embedding_norm = face_embedding_norm,
			gender = gender,
			age = age,
			race = race
		))
	return faces


def get_one_face(faces : List[Face], position : int = 0) -> Optional[Face]:
	if faces:
		position = min(position, len(faces) - 1)
		return faces[position]
	return None


def get_many_faces(vision_frames : List[VisionFrame]) -> List[Face]:
	many_faces : List[Face] = []

	for vision_frame in vision_frames:
		if is_vision_frame(vision_frame):
			all_bounding_boxes = []
			all_face_scores = []
			all_face_landmarks_5 = []

			for bounding_boxes, face_scores, face_landmarks_5 in detect_faces_by_angles(vision_frame, state_manager.get_item('face_detector_angles')):
				all_bounding_boxes.extend(bounding_boxes)
				all_face_scores.extend(face_scores)
				all_face_landmarks_5.extend(face_landmarks_5)

			if all_bounding_boxes and all_face_scores and all_face_landmarks_5 and state_manager.get_item('face_detector_score') > 0:
				faces = create_faces(vision_frame, all_bounding_boxes, all_face_scores, all_face_landmarks_5)

				if faces:
					many_faces.extend(faces)

	return many_faces


def detect_faces_at_angle(vision_frame : VisionFrame, face_detector_angle : Angle) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
	if face_detector_angle == 0:
		return detect_faces(vision_frame)
	return detect_faces_by_angle(vision_frame, face_detector_angle)


# In a stream every angle after the first runs on a helper thread while the worker runs the first
# itself, so a frame waits for its slowest angle instead of their sum. The detector hands each caller
# a session of its own, so this adds no sharing. Results keep the angle order, so the nms after it
# sees exactly what the sequential loop produced. Outside a stream the loop stays sequential
def detect_faces_by_angles(vision_frame : VisionFrame, face_detector_angles : List[Angle]) -> List[Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]]:
	if get_stream_worker_count() < 1 or len(face_detector_angles) < 2:
		return [ detect_faces_at_angle(vision_frame, face_detector_angle) for face_detector_angle in face_detector_angles ]

	app_context = detect_app_context()
	angle_futures = [ FACE_DETECTOR_ANGLE_EXECUTOR.submit(detect_faces_at_angle_task, vision_frame, face_detector_angle, app_context) for face_detector_angle in face_detector_angles[1:] ]
	angle_detections = [ detect_faces_at_angle(vision_frame, face_detector_angles[0]) ]

	for angle_future in angle_futures:
		angle_detection, (total_ms, run_count) = angle_future.result()
		add_inference_timer(total_ms, run_count)
		angle_detections.append(angle_detection)
	return angle_detections


# The app context is thread local and decides which inference pool set the detector reads, so a
# helper without the worker's context would load a second set of sessions
def detect_faces_at_angle_task(vision_frame : VisionFrame, face_detector_angle : Angle, app_context : AppContext) -> Tuple[Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]], Tuple[float, int]]:
	set_app_context_override(app_context)
	reset_inference_timer()

	try:
		return detect_faces_at_angle(vision_frame, face_detector_angle), get_inference_timer()
	finally:
		set_app_context_override(None)


def get_static_faces(vision_frames : List[VisionFrame]) -> List[Face]:
	if get_face_cache_bypass():
		return get_many_faces(vision_frames)

	many_faces : List[Face] = []

	for vision_frame in vision_frames:
		face_analysis_features = get_face_analysis_features()
		cache_scope = None
		if face_analysis_features is not None:
			cache_scope = ','.join(sorted(face_analysis_features)) or 'minimal'
		vision_hash = face_store.create_vision_hash(vision_frame, cache_scope)
		faces = face_store.get_faces(vision_hash)

		if not faces:
			with face_store.resolve_lock(vision_hash):
				faces = face_store.get_faces(vision_hash)

				if not faces:
					faces = get_many_faces([ vision_frame ])

					if faces:
						face_store.set_faces(vision_hash, faces)

		many_faces.extend(faces)

	return many_faces


def refill_faces(faces : List[Optional[Face]]) -> List[Face]:
	fill_faces = []
	anchor_index_previous = -1

	for index, face in enumerate(faces):
		if face:
			for gap_index in range(anchor_index_previous + 1, index):
				average_factor = (gap_index - anchor_index_previous) / (index - anchor_index_previous)
				average_face = average_face_geometry([faces[anchor_index_previous], face], average_factor)
				fill_faces.append(average_face)

			fill_faces.append(face)
			anchor_index_previous = index

	return fill_faces


def average_face_geometry(faces : List[Face], average_factor : float) -> Face:
	face_first = get_first(faces)
	face_middle = get_middle(faces)
	face_anchor = face_middle

	if average_factor < 0.5:
		face_anchor = face_first

	landmark_set : FaceLandmarkSet =\
	{
		'5': average_points(face_first.landmark_set.get('5'), face_middle.landmark_set.get('5'), average_factor),
		'5/68': average_points(face_first.landmark_set.get('5/68'), face_middle.landmark_set.get('5/68'), average_factor),
		'68': average_points(face_first.landmark_set.get('68'), face_middle.landmark_set.get('68'), average_factor),
		'68/5': average_points(face_first.landmark_set.get('68/5'), face_middle.landmark_set.get('68/5'), average_factor)
	}

	return Face(
		origin = 'refill',
		bounding_box = average_points(face_first.bounding_box, face_middle.bounding_box, average_factor),
		score_set = face_anchor.score_set,
		landmark_set = landmark_set,
		angle = estimate_face_angle(landmark_set.get('68/5')),
		embedding = face_anchor.embedding,
		embedding_norm = face_anchor.embedding_norm,
		gender = face_anchor.gender,
		age = face_anchor.age,
		race = face_anchor.race
	)


def average_face_identity(faces : List[Face]) -> Optional[Face]:
	face_embeddings = []
	face_embeddings_norm = []

	if faces:
		first_face = get_first(faces)

		for face in faces:
			face_embeddings.append(face.embedding)
			face_embeddings_norm.append(face.embedding_norm)

		return Face(
			origin = first_face.origin,
			bounding_box = first_face.bounding_box,
			score_set = first_face.score_set,
			landmark_set = first_face.landmark_set,
			angle = first_face.angle,
			embedding = numpy.mean(face_embeddings, axis = 0),
			embedding_norm = numpy.mean(face_embeddings_norm, axis = 0),
			gender = first_face.gender,
			age = first_face.age,
			race = first_face.race
		)
	return None


def scale_face(target_face : Face, target_vision_frame : VisionFrame, temp_vision_frame : VisionFrame) -> Face:
	scale_x = temp_vision_frame.shape[1] / target_vision_frame.shape[1]
	scale_y = temp_vision_frame.shape[0] / target_vision_frame.shape[0]

	bounding_box = target_face.bounding_box * [ scale_x, scale_y, scale_x, scale_y ]
	landmark_set =\
	{
		'5': target_face.landmark_set.get('5') * numpy.array([ scale_x, scale_y ]),
		'5/68': target_face.landmark_set.get('5/68') * numpy.array([ scale_x, scale_y ]),
		'68': target_face.landmark_set.get('68') * numpy.array([ scale_x, scale_y ]),
		'68/5': target_face.landmark_set.get('68/5') * numpy.array([ scale_x, scale_y ])
	}

	return target_face._replace(
		bounding_box = bounding_box,
		landmark_set = landmark_set
	)
