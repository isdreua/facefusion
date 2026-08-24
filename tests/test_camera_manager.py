from facefusion import camera_manager


class FakeCapture:
	def __init__(self, source):
		self.source = source
		self.opened = True
		self.released = False
		self.properties = []

	def isOpened(self):
		return self.opened

	def set(self, key, value):
		self.properties.append((key, value))
		return True

	def getBackendName(self):
		return 'fake'

	def get(self, key):
		return 0

	def release(self):
		self.released = True


def test_local_camera_configuration_and_cache(monkeypatch):
	created = []
	monkeypatch.setattr(camera_manager.cv2, 'VideoCapture', lambda source, *args: created.append(FakeCapture(source)) or created[-1])
	camera_manager.clear_camera_pool()

	first = camera_manager.get_local_camera_capture(0, 640, 480, 30)
	assert camera_manager.get_local_camera_capture(0, 640, 480, 30) is first
	assert [ value for _, value in first.properties ] == [ 640, 480, 30, 1 ]

	second = camera_manager.get_local_camera_capture(0, 1280, 720, 30)
	assert first.released is True
	assert second is not first
	camera_manager.clear_camera_pool()


def test_camera_detection_does_not_clear_pool(monkeypatch):
	created = []
	monkeypatch.setattr(camera_manager.cv2, 'VideoCapture', lambda source, *args: created.append(FakeCapture(source)) or created[-1])
	camera_manager.clear_camera_pool()
	pooled = camera_manager.get_local_camera_capture(9, 640, 480, 30)

	assert camera_manager.detect_local_camera_ids(0, 2) == [ 0, 1 ]
	assert pooled.released is False
	assert all(capture.released for capture in created[1:])
	camera_manager.clear_camera_pool()


def test_camera_backend_mapping(monkeypatch):
	monkeypatch.setattr(camera_manager, 'is_windows', lambda: True)
	assert camera_manager.get_camera_backend_flag('msmf') == camera_manager.cv2.CAP_MSMF
	assert camera_manager.get_camera_backend_flag('dshow') == camera_manager.cv2.CAP_DSHOW
	assert camera_manager.get_camera_backend_flag('auto') == camera_manager.cv2.CAP_ANY
