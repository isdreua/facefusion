import queue
import threading
from typing import Any, Callable, Optional

import numpy

from facefusion.types import VisionFrame


class LatestFrameWriter:
	def __init__(self, transport_factory : Callable[[], Any], width : int, height : int, repeat_latest : bool = False):
		self.transport_factory = transport_factory
		self.width = width
		self.height = height
		self.repeat_latest = repeat_latest
		self.frame_queue : queue.Queue = queue.Queue(maxsize = 1)
		self.stop_event = threading.Event()
		self.ready_event = threading.Event()
		self.thread = threading.Thread(target = self._run, daemon = True)
		self.transport = None
		self.error : Optional[BaseException] = None

	def start(self, timeout : float = 5.0) -> None:
		self.thread.start()
		if not self.ready_event.wait(timeout):
			self.close()
			raise TimeoutError('webcam output transport startup timed out')
		if self.error:
			raise RuntimeError('webcam output transport failed to start') from self.error

	def submit(self, frame : VisionFrame) -> None:
		if self.error or self.stop_event.is_set():
			return
		if frame.dtype != numpy.uint8 or frame.shape != (self.height, self.width, 3):
			raise ValueError('webcam output frame must be RGB uint8 with configured dimensions')
		if not frame.flags.c_contiguous:
			frame = numpy.ascontiguousarray(frame)
		try:
			self.frame_queue.put_nowait(frame)
		except queue.Full:
			try:
				self.frame_queue.get_nowait()
			except queue.Empty:
				pass
			self.frame_queue.put_nowait(frame)

	def close(self, timeout : float = 2.0) -> None:
		self.stop_event.set()
		if self.thread.is_alive():
			self.thread.join(timeout)
		if self.thread.is_alive() and self.transport and hasattr(self.transport, 'abort'):
			self.transport.abort()
			self.thread.join(timeout)

	def _run(self) -> None:
		latest_frame = None
		try:
			self.transport = self.transport_factory()
		except BaseException as exception:
			self.error = exception
			self.ready_event.set()
			return
		self.ready_event.set()
		try:
			while not self.stop_event.is_set():
				try:
					latest_frame = self.frame_queue.get(timeout = 0.01)
				except queue.Empty:
					if not self.repeat_latest:
						continue
				if latest_frame is not None:
					self.transport.stdin.write(latest_frame.tobytes())
		except BaseException as exception:
			self.error = exception
		finally:
			if self.transport and hasattr(self.transport.stdin, 'close'):
				self.transport.stdin.close()
