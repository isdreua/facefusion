import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

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
		self.stats_lock = threading.Lock()
		self.stats = { 'submitted': 0, 'replaced': 0, 'written': 0, 'repeated': 0, 'write_failures': 0, 'last_write_ms': 0.0 }

	def start(self, timeout : float = 5.0) -> None:
		self.thread.start()
		if not self.ready_event.wait(timeout):
			self.close()
			raise TimeoutError('webcam output transport startup timed out')
		if self.error:
			raise RuntimeError('webcam output transport failed to start') from self.error

	def submit(self, frame : VisionFrame, timing : Optional[Dict[str, float]] = None) -> None:
		if self.error or self.stop_event.is_set():
			return
		if frame.dtype != numpy.uint8 or frame.shape != (self.height, self.width, 3):
			raise ValueError('webcam output frame must be RGB uint8 with configured dimensions')
		if not frame.flags.c_contiguous:
			frame = numpy.ascontiguousarray(frame)
		if timing is not None:
			timing['output_enqueued'] = time.perf_counter()
		with self.stats_lock:
			self.stats['submitted'] += 1
		try:
			self.frame_queue.put_nowait((frame, timing))
		except queue.Full:
			try:
				self.frame_queue.get_nowait()
			except queue.Empty:
				pass
			with self.stats_lock:
				self.stats['replaced'] += 1
			self.frame_queue.put_nowait((frame, timing))

	def get_stats(self) -> Dict[str, float]:
		with self.stats_lock:
			return dict(self.stats)

	def close(self, timeout : float = 2.0) -> None:
		self.stop_event.set()
		if self.thread.is_alive():
			self.thread.join(timeout)
		if self.thread.is_alive() and self.transport and hasattr(self.transport, 'abort'):
			self.transport.abort()
			self.thread.join(timeout)

	def _run(self) -> None:
		latest_frame = None
		latest_timing = None
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
					latest_frame, latest_timing = self.frame_queue.get(timeout = 0.01)
					is_repeat = False
				except queue.Empty:
					if not self.repeat_latest:
						continue
					is_repeat = True
				if latest_frame is not None:
					write_started = time.perf_counter()
					self.transport.stdin.write(latest_frame.tobytes())
					write_finished = time.perf_counter()
					if latest_timing is not None and not is_repeat:
						latest_timing['output_written'] = write_finished
					with self.stats_lock:
						self.stats['written'] += 1
						self.stats['repeated'] += int(is_repeat)
						self.stats['last_write_ms'] = (write_finished - write_started) * 1000
		except BaseException as exception:
			self.error = exception
			with self.stats_lock:
				self.stats['write_failures'] += 1
		finally:
			if self.transport and hasattr(self.transport.stdin, 'close'):
				self.transport.stdin.close()
