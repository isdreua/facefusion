import queue
from concurrent.futures import Future

import numpy
import pytest

from facefusion import streamer
from facefusion.streamer import CameraCaptureThread


class FakeCameraCapture:
	def __init__(self, frames):
		self.frames = iter(frames)
		self.opened = True

	def isOpened(self):
		return self.opened

	def read(self):
		try:
			return True, next(self.frames)
		except StopIteration:
			self.opened = False
			return False, None


def test_camera_capture_thread_drops_oldest_frame_without_blocking():
	first_frame = numpy.zeros((2, 2, 3), dtype = numpy.uint8)
	latest_frame = numpy.ones((2, 2, 3), dtype = numpy.uint8)
	capture_thread = CameraCaptureThread(FakeCameraCapture([ first_frame, latest_frame ]))

	capture_thread.run()

	_, queued_frame = capture_thread.frame_queue.get_nowait()
	assert queued_frame is latest_frame
	with pytest.raises(queue.Empty):
		capture_thread.frame_queue.get_nowait()


class FakeProgress:
	def __enter__(self):
		return self

	def __exit__(self, *args):
		pass

	def update(self):
		pass


class FakeExecutor:
	def __init__(self, *args, **kwargs):
		pass

	def submit(self, function, *args):
		future = Future()
		future.set_result(function(*args))
		return future

	def shutdown(self, wait = False):
		pass


class FakeCaptureThread:
	def __init__(self, camera_capture):
		self.running = camera_capture.running
		self.frame_queue = camera_capture

	def start(self):
		pass

	def stop(self):
		self.running = False

	def join(self, timeout):
		pass

	def is_alive(self):
		return False


class FakeFrameQueue:
	def __init__(self, frame = None, error = None):
		self.frame = frame
		self.error = error
		self.running = frame is not None or error is not None
		self.reads = 0

	def get(self, timeout):
		self.reads += 1
		if self.error:
			raise self.error
		if self.reads > 1:
			raise queue.Empty
		return 1.0, self.frame


def prepare_capture_test(monkeypatch, capture, context_calls):
	monkeypatch.setattr(streamer, 'read_static_images', lambda paths: [])
	monkeypatch.setattr(streamer, 'get_processors_modules', lambda processors: [])
	monkeypatch.setattr(streamer, 'create_empty_audio_frame', lambda: numpy.zeros(1))
	monkeypatch.setattr(streamer, 'tqdm', lambda **kwargs: FakeProgress())
	monkeypatch.setattr(streamer, 'ThreadPoolExecutor', FakeExecutor)
	monkeypatch.setattr(streamer, 'CameraCaptureThread', FakeCaptureThread)
	monkeypatch.setattr(streamer, 'detect_app_context', lambda: 'ui')
	monkeypatch.setattr(streamer, 'set_app_context_override', context_calls.append)
	monkeypatch.setattr(streamer.state_manager, 'get_item', lambda key: {
		'source_paths': [],
		'execution_thread_count': 1,
		'processors': [],
		'face_swapper_model': 'model',
		'log_level': 'info',
		'webcam_frame_skipping': 'disabled',
		'face_selector_mode': 'one',
		'face_selector_age_start': 0,
		'face_selector_age_end': 100
	}.get(key))
	return streamer.multi_process_capture(capture, 30)


def test_capture_context_is_cleared_after_normal_completion(monkeypatch):
	context_calls = []
	generator = prepare_capture_test(monkeypatch, FakeFrameQueue(), context_calls)

	assert list(generator) == []
	assert context_calls == [ 'ui', None ]


def test_capture_context_is_cleared_when_generator_closes(monkeypatch):
	context_calls = []
	frame = numpy.zeros((2, 2, 3), dtype = numpy.uint8)
	generator = prepare_capture_test(monkeypatch, FakeFrameQueue(frame), context_calls)

	next(generator)
	generator.close()
	assert context_calls == [ 'ui', None ]


def test_capture_context_is_cleared_after_exception(monkeypatch):
	context_calls = []
	generator = prepare_capture_test(monkeypatch, FakeFrameQueue(error = RuntimeError('capture failed')), context_calls)

	with pytest.raises(RuntimeError, match = 'capture failed'):
		next(generator)
	assert context_calls == [ 'ui', None ]
