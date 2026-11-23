import glob
import threading
import time
from typing import Callable, List, Optional

import cv2

from facefusion import logger, state_manager
from facefusion.common_helper import is_windows
from facefusion.types import CameraPoolSet

CAMERA_POOL_SET : CameraPoolSet =\
{
	'capture': {}
}
# A camera switched mid stream is opened and released on the capture thread while the stream's start
# and stop run on another, so the pool is only ever changed under this lock
CAMERA_POOL_LOCK = threading.RLock()
# A camera that will not open is often only busy (another program releasing it, a driver waking up,
# a cable being plugged back in), so it is tried again every 100 ms until it opens. The log says so
# once and then every five seconds instead of ten times a second
CAMERA_RETRY_DELAY = 0.1
CAMERA_RETRY_LOG_INTERVAL = 50
# Set by the browser's stop button, so a start still waiting for its camera gives up
CAMERA_OPEN_CANCEL = threading.Event()


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


# Tries until the camera opens. Only is_cancelled ends it early (the browser's stop button); headless
# has none and is ended with Ctrl+C. The wait is spent outside the pool lock
def get_local_camera_capture(camera_id : int, width : Optional[int] = None, height : Optional[int] = None, fps : Optional[int] = None, backend : str = 'auto', is_cancelled : Optional[Callable[[], bool]] = None) -> Optional[cv2.VideoCapture]:
	camera_open_attempt = 0

	while True:
		camera_open_attempt += 1
		camera_capture = open_camera_capture_once(camera_id, width, height, fps, backend, camera_open_attempt > 1)
		if camera_capture:
			if camera_open_attempt > 1:
				logger.info('camera ' + str(camera_id) + ' opened after ' + str(camera_open_attempt) + ' attempts', __name__)
			return camera_capture
		if is_cancelled and is_cancelled():
			return None
		log_camera_retry(camera_id, camera_open_attempt)
		time.sleep(CAMERA_RETRY_DELAY)


# One attempt. Repeated attempts on an absent device would make opencv log an error each time, ten
# times a second, so those are silenced where the build allows it
def open_camera_capture_once(camera_id : int, width : Optional[int] = None, height : Optional[int] = None, fps : Optional[int] = None, backend : str = 'auto', is_quiet : bool = False) -> Optional[cv2.VideoCapture]:
	previous_log_level = suppress_camera_log_level() if is_quiet else None

	try:
		with CAMERA_POOL_LOCK:
			camera_capture = open_local_camera_capture(camera_id, width, height, fps, backend)
	finally:
		restore_camera_log_level(previous_log_level)
	if camera_capture and camera_capture.isOpened():
		return camera_capture
	return None


def log_camera_retry(camera_id : int, camera_open_attempt : int) -> None:
	if camera_open_attempt == 1:
		logger.warn('camera ' + str(camera_id) + ' could not be opened, trying again every ' + str(int(CAMERA_RETRY_DELAY * 1000)) + ' ms until it opens', __name__)
	elif camera_open_attempt % CAMERA_RETRY_LOG_INTERVAL == 0:
		logger.warn('camera ' + str(camera_id) + ' still could not be opened after ' + str(camera_open_attempt) + ' attempts, still trying', __name__)


def cancel_camera_open() -> None:
	CAMERA_OPEN_CANCEL.set()


def reset_camera_open() -> None:
	CAMERA_OPEN_CANCEL.clear()


def is_camera_open_cancelled() -> bool:
	return CAMERA_OPEN_CANCEL.is_set()


def open_local_camera_capture(camera_id : int, width : Optional[int] = None, height : Optional[int] = None, fps : Optional[int] = None, backend : str = 'auto') -> cv2.VideoCapture:
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
		else:
			# Released so the next attempt does not find the device still held by this one
			camera_capture.release()

	return CAMERA_POOL_SET.get('capture').get(camera_key)


def get_remote_camera_capture(camera_url : str) -> cv2.VideoCapture:
	if camera_url not in CAMERA_POOL_SET.get('capture'):
		camera_capture = cv2.VideoCapture(camera_url)

		if camera_capture.isOpened():
			configure_camera_capture(camera_capture)
			CAMERA_POOL_SET['capture'][camera_url] = camera_capture

	return CAMERA_POOL_SET.get('capture').get(camera_url)


def clear_camera_pool() -> None:
	with CAMERA_POOL_LOCK:
		for camera_capture in CAMERA_POOL_SET.get('capture').values():
			camera_capture.release()

		CAMERA_POOL_SET['capture'].clear()


# Dropped from the pool as well, so switching back to this camera opens it again instead of being
# handed the released capture
def release_camera_capture(camera_capture : cv2.VideoCapture) -> None:
	with CAMERA_POOL_LOCK:
		for pooled_key, pooled_capture in list(CAMERA_POOL_SET.get('capture').items()):
			if pooled_capture is camera_capture:
				del CAMERA_POOL_SET['capture'][pooled_key]
		camera_capture.release()


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


# Only linux names its devices somewhere readable without another dependency: the v4l2 driver puts the
# card name next to the device node. Index N under the auto backend is /dev/videoN there, so the name
# read is the device the capture opened. Elsewhere, or when the file is missing, there is no name
def resolve_camera_name(camera_id : int) -> Optional[str]:
	try:
		with open('/sys/class/video4linux/video' + str(int(camera_id)) + '/name') as name_file:
			camera_name = name_file.read().strip()
	except (OSError, ValueError):
		return None
	return camera_name or None


def resolve_camera_window_title(camera_id : int) -> str:
	return 'CAMERA (' + (resolve_camera_name(camera_id) or str(camera_id)) + ')'


def resolve_camera_label(camera_id : int) -> str:
	camera_name = resolve_camera_name(camera_id)

	if camera_name:
		return camera_name + ' (' + str(camera_id) + ')'
	return str(camera_id)


# Carries the camera a stream should be on to its capture thread: a switch asked for with Ctrl+0-9 or
# the browser's camera dropdown, or the same camera again after it stopped delivering. The capture
# thread makes one attempt every 100 ms and keeps reading the current camera in between, so a switch
# to a camera that is not there yet leaves the stream running until it appears, and asking for
# another camera replaces the one being waited for. New cameras open at the first one's resolution,
# rate and backend. What the stream is on, and waits for, is published to state for the overlay
class CameraSwitch:
	def __init__(self, camera_id : int, width : Optional[int] = None, height : Optional[int] = None, fps : Optional[int] = None, backend : str = 'auto'):
		self.camera_id = camera_id
		self.width = width
		self.height = height
		self.fps = fps
		self.backend = backend
		self.target_camera_id : Optional[int] = None
		self.is_reconnecting = False
		self.open_attempt = 0
		self.next_attempt_time = 0.0
		self.lock = threading.Lock()
		self.listeners : List[Callable[[int, str], None]] = []
		self.publish_label()

	# Asking for the camera already running drops a switch still waiting, except while that camera is
	# itself being reopened
	def request(self, camera_id : int) -> None:
		with self.lock:
			if camera_id == self.camera_id and not self.is_reconnecting:
				self.target_camera_id = None
			elif camera_id != self.target_camera_id:
				self.target_camera_id = camera_id
				self.open_attempt = 0
				self.next_attempt_time = 0.0
		self.publish_label()

	# The camera the browser should show as selected: the one waited for, else the one running
	def get_selected_camera_id(self) -> int:
		with self.lock:
			if self.target_camera_id is not None:
				return self.target_camera_id
			return self.camera_id

	def take_due_target(self) -> Optional[int]:
		with self.lock:
			if self.target_camera_id is not None and time.perf_counter() >= self.next_attempt_time:
				return self.target_camera_id
			return None

	# The stream's first open, tried every 100 ms until it works. The camera is read again before
	# every attempt, so a camera picked meanwhile (dropdown or Ctrl+0-9, which reach this switch
	# while it is active) is the one tried next rather than the number the start began with
	def open_selected_capture(self, is_cancelled : Optional[Callable[[], bool]] = None) -> Optional[cv2.VideoCapture]:
		attempted_camera_id : Optional[int] = None
		camera_open_attempt = 0

		while True:
			camera_id = self.get_selected_camera_id()
			if camera_id != attempted_camera_id:
				attempted_camera_id = camera_id
				camera_open_attempt = 0
			camera_open_attempt += 1
			camera_capture = open_camera_capture_once(camera_id, self.width, self.height, self.fps, self.backend, camera_open_attempt > 1)
			if camera_capture:
				if camera_open_attempt > 1:
					logger.info('camera ' + str(camera_id) + ' opened after ' + str(camera_open_attempt) + ' attempts', __name__)
				self.settle_first(camera_id)
				return camera_capture
			if is_cancelled and is_cancelled():
				return None
			log_camera_retry(camera_id, camera_open_attempt)
			time.sleep(CAMERA_RETRY_DELAY)

	# The camera the first open landed on becomes the running one. A camera asked for after that
	# attempt began stays the target, and the capture thread switches to it
	def settle_first(self, camera_id : int) -> None:
		with self.lock:
			self.camera_id = camera_id
			if self.target_camera_id == camera_id:
				self.target_camera_id = None
			self.open_attempt = 0
			self.next_attempt_time = 0.0
		self.publish_label()

	def try_open(self, camera_id : int) -> Optional[cv2.VideoCapture]:
		try:
			camera_capture = open_camera_capture_once(camera_id, self.width, self.height, self.fps, self.backend, self.open_attempt > 0)
		except Exception:
			camera_capture = None
		if camera_capture:
			return camera_capture
		with self.lock:
			self.open_attempt += 1
			self.next_attempt_time = time.perf_counter() + CAMERA_RETRY_DELAY
			camera_open_attempt = self.open_attempt
		log_camera_retry(camera_id, camera_open_attempt)
		return None

	def release_capture(self, camera_capture : cv2.VideoCapture) -> None:
		release_camera_capture(camera_capture)

	# The current camera stopped delivering, so it becomes the camera to open again, unless a switch
	# to another one is already waiting
	def mark_lost(self) -> None:
		with self.lock:
			if self.is_reconnecting:
				return
			self.is_reconnecting = True
			if self.target_camera_id is None:
				self.target_camera_id = self.camera_id
				self.open_attempt = 0
				self.next_attempt_time = 0.0
		logger.warn('camera ' + resolve_camera_label(self.camera_id) + ' stopped delivering, reopening it', __name__)
		self.publish_label()

	def commit(self, camera_id : int) -> None:
		with self.lock:
			is_reopened = self.is_reconnecting and camera_id == self.camera_id
			self.camera_id = camera_id
			if self.target_camera_id == camera_id:
				self.target_camera_id = None
			self.is_reconnecting = False
			self.open_attempt = 0
		self.publish_label()
		if is_reopened:
			logger.info('camera ' + resolve_camera_label(camera_id) + ' reopened', __name__)
		else:
			logger.info('switched to camera ' + resolve_camera_label(camera_id), __name__)
		for listener in list(self.listeners):
			try:
				listener(camera_id, resolve_camera_window_title(camera_id))
			except Exception:
				pass

	def publish_label(self) -> None:
		with self.lock:
			camera_id = self.camera_id
			target_camera_id = self.target_camera_id
			is_reconnecting = self.is_reconnecting
		camera_label = resolve_camera_label(camera_id)
		if is_reconnecting and target_camera_id in [ None, camera_id ]:
			camera_label += ' (reopening)'
		elif target_camera_id is not None and is_reconnecting:
			camera_label = resolve_camera_label(target_camera_id) + ' (opening)'
		elif target_camera_id is not None:
			camera_label += ' > waiting for ' + str(target_camera_id)
		state_manager.set_item('webcam_camera_name', camera_label)

	# Made the target of request_camera_switch for as long as its stream runs, so the browser's camera
	# dropdown reaches a running stream the way Ctrl+0-9 does
	def activate(self) -> None:
		global ACTIVE_CAMERA_SWITCH

		ACTIVE_CAMERA_SWITCH = self

	def deactivate(self) -> None:
		global ACTIVE_CAMERA_SWITCH

		if ACTIVE_CAMERA_SWITCH is self:
			ACTIVE_CAMERA_SWITCH = None


ACTIVE_CAMERA_SWITCH : Optional[CameraSwitch] = None


# True when a running stream took the request; the switch itself happens on its capture thread
def request_camera_switch(camera_id : int) -> bool:
	camera_switch = ACTIVE_CAMERA_SWITCH

	if camera_switch:
		camera_switch.request(camera_id)
		return True
	return False


def get_active_camera_id() -> Optional[int]:
	camera_switch = ACTIVE_CAMERA_SWITCH

	if camera_switch:
		return camera_switch.get_selected_camera_id()
	return None


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
