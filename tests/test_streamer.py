import queue

import numpy
import pytest

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
