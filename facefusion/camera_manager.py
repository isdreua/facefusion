import glob
from typing import List, Optional

import cv2

from facefusion import logger
from facefusion.common_helper import is_windows
from facefusion.types import CameraPoolSet

CAMERA_POOL_SET : CameraPoolSet =\
{
	'capture': {}
}


def configure_camera_capture(camera_capture : cv2.VideoCapture, width : Optional[int] = None, height : Optional[int] = None, fps : Optional[int] = None) -> None:
	property_results = []
	if width is not None:
		property_results.append(('width', width, camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)))
	if height is not None:
		property_results.append(('height', height, camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)))
	if fps is not None:
		property_results.append(('fps', fps, camera_capture.set(cv2.CAP_PROP_FPS, fps)))
	property_results.append(('buffer', 1, camera_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)))
	backend_name = camera_capture.getBackendName() if hasattr(camera_capture, 'getBackendName') else 'unknown'
	negotiated = None
	if hasattr(camera_capture, 'get'):
		negotiated = (camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH), camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT), camera_capture.get(cv2.CAP_PROP_FPS))
	logger.debug('camera configuration backend=' + str(backend_name) + ' results=' + str(property_results) + ' negotiated=' + str(negotiated), __name__)


def get_camera_backend_flag(backend : str) -> int:
	if is_windows() and backend == 'msmf':
		return cv2.CAP_MSMF
	if is_windows() and backend == 'dshow':
		return cv2.CAP_DSHOW
	return cv2.CAP_ANY


def get_local_camera_capture(camera_id : int, width : Optional[int] = None, height : Optional[int] = None, fps : Optional[int] = None, backend : str = 'auto') -> cv2.VideoCapture:
	camera_key = ':'.join(map(str, [ 'local', camera_id, backend, width, height, fps ]))

	if camera_key not in CAMERA_POOL_SET.get('capture'):
		for pooled_key, pooled_capture in list(CAMERA_POOL_SET.get('capture').items()):
			if pooled_key.startswith('local:' + str(camera_id) + ':'):
				pooled_capture.release()
				del CAMERA_POOL_SET['capture'][pooled_key]
		camera_capture = cv2.VideoCapture(camera_id, get_camera_backend_flag(backend))
		if not camera_capture.isOpened() and backend != 'auto':
			camera_capture.release()
			logger.warn('camera backend ' + backend + ' failed; falling back to auto', __name__)
			camera_capture = cv2.VideoCapture(camera_id, cv2.CAP_ANY)

		if camera_capture.isOpened():
			configure_camera_capture(camera_capture, width, height, fps)
			CAMERA_POOL_SET['capture'][camera_key] = camera_capture

	return CAMERA_POOL_SET.get('capture').get(camera_key)


def get_remote_camera_capture(camera_url : str) -> cv2.VideoCapture:
	if camera_url not in CAMERA_POOL_SET.get('capture'):
		camera_capture = cv2.VideoCapture(camera_url)

		if camera_capture.isOpened():
			configure_camera_capture(camera_capture)
			CAMERA_POOL_SET['capture'][camera_url] = camera_capture

	return CAMERA_POOL_SET.get('capture').get(camera_url)


def clear_camera_pool() -> None:
	for camera_capture in CAMERA_POOL_SET.get('capture').values():
		camera_capture.release()

	CAMERA_POOL_SET['capture'].clear()


# Probing absent camera indices makes opencv log an error per index, so discovery silences it. The
# cv2.utils.logging submodule is not present in every opencv build, so this is best effort and
# never a reason to fail discovery
def suppress_camera_log_level() -> Optional[int]:
	try:
		previous_log_level = cv2.utils.logging.getLogLevel()
		cv2.utils.logging.setLogLevel(0)
		return previous_log_level
	except Exception:
		return None


def restore_camera_log_level(previous_log_level : Optional[int]) -> None:
	if previous_log_level is None:
		return

	try:
		cv2.utils.logging.setLogLevel(previous_log_level)
	except Exception:
		pass


# Opening an absent index makes the v4l2 backend log a warning and an error, and this build has no
# cv2.utils.logging to silence them. On linux the real devices are enumerable, so probe only those
# instead of brute forcing the whole range
def resolve_camera_id_range(id_start : int, id_end : int) -> List[int]:
	candidate_camera_ids = list(range(id_start, id_end))
	device_camera_ids = []

	for device_path in glob.glob('/dev/video*'):
		device_suffix = device_path.rsplit('video', 1)[-1]
		if device_suffix.isdigit():
			device_camera_ids.append(int(device_suffix))

	if not device_camera_ids:
		return candidate_camera_ids
	return sorted(set(device_camera_ids).intersection(candidate_camera_ids))


def detect_local_camera_ids(id_start : int, id_end : int, backend : str = 'auto') -> List[int]:
	local_camera_ids = []
	previous_log_level = suppress_camera_log_level()

	try:
		for camera_id in resolve_camera_id_range(id_start, id_end):
			camera_capture = cv2.VideoCapture(camera_id, get_camera_backend_flag(backend))
			if not camera_capture.isOpened() and backend != 'auto':
				camera_capture.release()
				camera_capture = cv2.VideoCapture(camera_id, cv2.CAP_ANY)

			if camera_capture.isOpened():
				local_camera_ids.append(camera_id)
			camera_capture.release()
	finally:
		restore_camera_log_level(previous_log_level)

	return local_camera_ids
