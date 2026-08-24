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


def detect_local_camera_ids(id_start : int, id_end : int, backend : str = 'auto') -> List[int]:
	local_camera_ids = []

	for camera_id in range(id_start, id_end):
		cv2.utils.logging.setLogLevel(0)
		camera_capture = cv2.VideoCapture(camera_id, get_camera_backend_flag(backend))
		cv2.utils.logging.setLogLevel(3)
		if not camera_capture.isOpened() and backend != 'auto':
			camera_capture.release()
			camera_capture = cv2.VideoCapture(camera_id, cv2.CAP_ANY)

		if camera_capture.isOpened():
			local_camera_ids.append(camera_id)
		camera_capture.release()

	return local_camera_ids
