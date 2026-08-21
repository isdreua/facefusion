import threading
from typing import List, Optional

from facefusion.hash_helper import create_hash
from facefusion.types import Face, FaceStore, VisionFrame
from facefusion.vision import is_vision_frame

FACE_STORE : FaceStore = {}
CACHE_MAX_SIZE = 200


def _limit_cache_size() -> None:
	if len(FACE_STORE) >= CACHE_MAX_SIZE:
		try:
			oldest_key = next(iter(FACE_STORE))
			del FACE_STORE[oldest_key]
		except (StopIteration, KeyError):
			pass


def create_vision_hash(vision_frame : VisionFrame, cache_scope : Optional[str] = None) -> Optional[str]:
	if is_vision_frame(vision_frame):
		vision_hash = create_hash(vision_frame.tobytes())
		if cache_scope:
			return vision_hash + ':' + cache_scope
		return vision_hash
	return None


def get_faces(vision_hash : Optional[str]) -> Optional[List[Face]]:
	if vision_hash and FACE_STORE.get(vision_hash):
		return FACE_STORE.get(vision_hash).get('faces')
	return None


def set_faces(vision_hash : Optional[str], faces : List[Face]) -> None:
	if vision_hash:
		if vision_hash not in FACE_STORE:
			_limit_cache_size()
		FACE_STORE.setdefault(vision_hash,
		{
			'lock': threading.Lock()
		})['faces'] = faces


def resolve_lock(vision_hash : Optional[str]) -> threading.Lock:
	if vision_hash:
		if vision_hash not in FACE_STORE:
			_limit_cache_size()
		return FACE_STORE.setdefault(vision_hash,
		{
			'lock': threading.Lock()
		}).get('lock')
	return threading.Lock()


def clear_faces() -> None:
	FACE_STORE.clear()
