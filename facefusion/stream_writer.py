import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import numpy

from facefusion.types import VisionFrame

# How quickly the measured spacing between real frames follows a change in the processing rate
SUBMIT_INTERVAL_SMOOTHING = 0.1
WRITTEN_RATE_WINDOW = 60
# Paced presentation holds each frame until its capture time plus a delay that covers nearly every
# frame's processing latency, so the camera's even cadence reaches the screen instead of the uneven
# one the workers finish in. The 95th percentile plus a margin leaves the odd outlier late rather
# than adding its latency to every frame
PACING_LATENCY_WINDOW = 60
PACING_LATENCY_PERCENTILE = 0.95
PACING_MARGIN = 0.003
# The delay rises at once so frames are not late, but falls slowly so a lull does not pull the
# schedule forward in a visible jump
PACING_DELAY_DECAY = 0.02


class LatestFrameWriter:
	def __init__(self, transport_factory : Callable[[], Any], width : int, height : int, repeat_latest : bool = False, fps : float = 30.0, pace_by_capture : bool = False):
		self.transport_factory = transport_factory
		self.width = width
		self.height = height
		self.repeat_latest = repeat_latest
		# A fixed rate transport already has its own cadence, so pacing only applies without it
		self.pace_by_capture = pace_by_capture and not repeat_latest
		self.frame_interval = 1.0 / max(1.0, fps)
		self.frame_queue : queue.Queue = queue.Queue(maxsize = 1)
		self.stop_event = threading.Event()
		self.ready_event = threading.Event()
		self.thread = threading.Thread(target = self._run_paced if self.pace_by_capture else self._run, daemon = True)
		self.transport = None
		self.error : Optional[BaseException] = None
		self.stats_lock = threading.Lock()
		self.stats = { 'submitted': 0, 'replaced': 0, 'written': 0, 'repeated': 0, 'generated': 0, 'late': 0, 'write_failures': 0, 'last_write_ms': 0.0, 'written_fps': 0.0, 'pacing_delay_ms': 0.0 }
		# Starts from the configured rate, which for a generated stream is twice the real one
		self.submit_interval = self.frame_interval * 2
		self.last_submit_time : Optional[float] = None
		self.written_timestamps : Deque[float] = deque(maxlen = WRITTEN_RATE_WINDOW)
		self.pacing_latencies : Deque[float] = deque(maxlen = PACING_LATENCY_WINDOW)
		self.pacing_delay = 0.0

	def start(self, timeout : float = 5.0) -> None:
		self.thread.start()
		if not self.ready_event.wait(timeout):
			self.close()
			raise TimeoutError('webcam output transport startup timed out')
		if self.error:
			raise RuntimeError('webcam output transport failed to start') from self.error

	# A generated frame belongs between the last real frame and this one, so it travels with this one
	# and is presented first; the real frame then waits half a real frame interval for its own slot
	def submit(self, frame : VisionFrame, timing : Optional[Dict[str, Any]] = None, generated_frame : Optional[VisionFrame] = None) -> None:
		if self.error:
			raise RuntimeError('webcam output transport failed') from self.error
		if self.stop_event.is_set():
			return
		frame = self.validate_frame(frame)
		if generated_frame is not None:
			generated_frame = self.validate_frame(generated_frame)
		submit_time = time.perf_counter()
		if timing is not None:
			timing['output_enqueued'] = submit_time
		with self.stats_lock:
			self.stats['submitted'] += 1
			if self.last_submit_time is not None:
				self.submit_interval += (submit_time - self.last_submit_time - self.submit_interval) * SUBMIT_INTERVAL_SMOOTHING
			self.last_submit_time = submit_time
		try:
			self.frame_queue.put_nowait((frame, timing, generated_frame))
		except queue.Full:
			try:
				self.frame_queue.get_nowait()
			except queue.Empty:
				pass
			with self.stats_lock:
				self.stats['replaced'] += 1
			self.frame_queue.put_nowait((frame, timing, generated_frame))

	def validate_frame(self, frame : VisionFrame) -> VisionFrame:
		if frame.dtype != numpy.uint8 or frame.shape != (self.height, self.width, 3):
			raise ValueError('webcam output frame must be RGB uint8 with configured dimensions')
		if not frame.flags.c_contiguous:
			frame = numpy.ascontiguousarray(frame)
		return frame

	def get_stats(self) -> Dict[str, float]:
		with self.stats_lock:
			stats = dict(self.stats)
			written_timestamps = list(self.written_timestamps)
		# Measured against now, so a stalled transport decays towards zero instead of freezing
		if len(written_timestamps) > 1:
			stats['written_fps'] = (len(written_timestamps) - 1) / max(1e-6, time.perf_counter() - written_timestamps[0])
		return stats

	def close(self, timeout : float = 2.0) -> None:
		self.stop_event.set()
		if self.thread.is_alive():
			self.thread.join(timeout)
		if self.thread.is_alive() and self.transport and hasattr(self.transport, 'abort'):
			self.transport.abort()
			self.thread.join(timeout)
		if self.thread.is_alive() and self.transport and hasattr(self.transport, 'terminate'):
			self.transport.terminate()
			self.thread.join(timeout)

	# A newer submission supersedes whatever of the previous one is still waiting for its slot, the
	# same latest-wins rule the queue applies to submissions the writer never got to
	def _take_submission(self, presentation_frames : Deque[Tuple[VisionFrame, Optional[Dict[str, Any]], bool]], timeout : float) -> bool:
		try:
			frame, timing, generated_frame = self.frame_queue.get(timeout = max(0.0, timeout))
		except queue.Empty:
			return False
		if presentation_frames:
			with self.stats_lock:
				self.stats['replaced'] += len(presentation_frames)
			presentation_frames.clear()
		if generated_frame is not None:
			presentation_frames.append((generated_frame, None, True))
		presentation_frames.append((frame, timing, False))
		return True

	def _write(self, frame : VisionFrame, timing : Optional[Dict[str, Any]], is_repeat : bool = False, is_generated : bool = False) -> Tuple[float, float]:
		write_started = time.perf_counter()
		self.transport.stdin.write(frame.tobytes())
		write_finished = time.perf_counter()
		if timing is not None:
			timing['output_written'] = write_finished
		with self.stats_lock:
			self.stats['written'] += 1
			self.stats['repeated'] += int(is_repeat)
			self.stats['generated'] += int(is_generated)
			self.stats['last_write_ms'] = (write_finished - write_started) * 1000
			if not is_repeat:
				self.written_timestamps.append(write_finished)
		return write_started, write_finished

	def _open_transport(self) -> bool:
		try:
			self.transport = self.transport_factory()
		except BaseException as exception:
			self.error = exception
			self.ready_event.set()
			return False
		self.ready_event.set()
		return True

	# When the camera delivered the frame, which is the cadence worth reproducing. Read end rather than
	# read start: the read blocks until the camera has a frame, so its start is only when the capture
	# thread began waiting
	def _resolve_capture_time(self, timing : Optional[Dict[str, Any]]) -> Optional[float]:
		if not timing:
			return None
		return timing.get('capture_read_end') or timing.get('capture_read_start')

	def _resolve_pacing_delay(self, latency : float) -> float:
		self.pacing_latencies.append(latency)
		sorted_latencies = sorted(self.pacing_latencies)
		target_delay = sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * PACING_LATENCY_PERCENTILE))] + PACING_MARGIN

		if target_delay > self.pacing_delay:
			self.pacing_delay = target_delay
		else:
			self.pacing_delay += (target_delay - self.pacing_delay) * PACING_DELAY_DECAY
		with self.stats_lock:
			self.stats['pacing_delay_ms'] = self.pacing_delay * 1000
		return self.pacing_delay

	# Each real frame is given the slot its capture time plus the pacing delay names, and a generated
	# frame the slot half way between its two real frames, so the spacing follows the camera even when
	# the frames before it were abandoned. Slots stay strictly increasing, since a falling delay could
	# otherwise schedule a frame ahead of one already waiting
	def _schedule_submission(self, schedule : Deque[Tuple[float, VisionFrame, Optional[Dict[str, Any]], bool]], frame : VisionFrame, timing : Optional[Dict[str, Any]], generated_frame : Optional[VisionFrame], previous_capture_time : Optional[float]) -> Optional[float]:
		now = time.perf_counter()
		capture_time = self._resolve_capture_time(timing)
		last_present_time = schedule[-1][0] if schedule else 0.0

		if capture_time is None:
			schedule.append((max(now, last_present_time), frame, timing, False))
			return previous_capture_time
		# The generated frame exists only once its later real frame has arrived, yet its slot is half
		# a camera interval before that frame's own, so it is the earlier slot the delay must reach
		generated_capture_time = (previous_capture_time + capture_time) * 0.5 if generated_frame is not None and previous_capture_time is not None else None
		pacing_delay = self._resolve_pacing_delay(now - (generated_capture_time or capture_time))
		if generated_frame is not None and generated_capture_time is not None:
			generated_present_time = max(generated_capture_time + pacing_delay, last_present_time + 0.001)
			# A generated frame is only worth showing ahead of its real one; once its slot has gone by
			# it would arrive bunched up against it, so it is let go
			if generated_present_time >= now:
				schedule.append((generated_present_time, generated_frame, None, True))
				last_present_time = generated_present_time
			else:
				with self.stats_lock:
					self.stats['late'] += 1
		schedule.append((max(capture_time + pacing_delay, last_present_time + 0.001), frame, timing, False))
		return capture_time

	def _run_paced(self) -> None:
		schedule : Deque[Tuple[float, VisionFrame, Optional[Dict[str, Any]], bool]] = deque()
		previous_capture_time = None

		if not self._open_transport():
			return
		try:
			while not self.stop_event.is_set():
				# The queue is the wait, so a submission that arrives before the next slot is scheduled
				# at once rather than after it
				waiting_time = schedule[0][0] - time.perf_counter() if schedule else 0.01
				if waiting_time > 0.0005:
					try:
						frame, timing, generated_frame = self.frame_queue.get(timeout = waiting_time)
						previous_capture_time = self._schedule_submission(schedule, frame, timing, generated_frame, previous_capture_time)
					except queue.Empty:
						pass
					continue
				if not schedule:
					continue
				present_time, frame, timing, is_generated = schedule.popleft()
				# Already overtaken by the next slot, so showing it now would only push that one back
				if schedule and schedule[0][0] <= time.perf_counter():
					with self.stats_lock:
						self.stats['late'] += 1
					continue
				self._write(frame, timing, False, is_generated)
		except BaseException as exception:
			self.error = exception
			with self.stats_lock:
				self.stats['write_failures'] += 1
		finally:
			if self.transport and hasattr(self.transport.stdin, 'close'):
				self.transport.stdin.close()

	def _run(self) -> None:
		latest_frame = None
		presentation_frames : Deque[Tuple[VisionFrame, Optional[Dict[str, Any]], bool]] = deque()

		if not self._open_transport():
			return
		try:
			next_write_time = time.perf_counter()
			while not self.stop_event.is_set():
				# Waiting for a slot: either the fixed cadence or the held back real frame. The queue
				# is the wait, so a submission arriving meanwhile is taken in rather than left to age
				waiting_time = next_write_time - time.perf_counter()
				if waiting_time > 0 and (self.repeat_latest or presentation_frames):
					self._take_submission(presentation_frames, waiting_time)
					continue
				if not presentation_frames and not self._take_submission(presentation_frames, 0.0 if self.repeat_latest else 0.01):
					if not self.repeat_latest or latest_frame is None:
						continue
				is_repeat = not presentation_frames
				is_generated = False
				latest_timing = None
				if presentation_frames:
					latest_frame, latest_timing, is_generated = presentation_frames.popleft()
				write_started, write_finished = self._write(latest_frame, latest_timing, is_repeat, is_generated)
				with self.stats_lock:
					submit_interval = self.submit_interval
				# On its own grid rather than from the end of this write: counting from the write took
				# the wake up delay into every tick, so two output ticks outlasted one camera frame and
				# the held back real frame was superseded before its slot. A stall still never bursts
				if self.repeat_latest:
					next_write_time = max(next_write_time + self.frame_interval, write_finished)
				elif presentation_frames:
					# Half way to the next real frame, the slot the generated frame was interpolated for
					next_write_time = write_started + submit_interval * 0.5
		except BaseException as exception:
			self.error = exception
			with self.stats_lock:
				self.stats['write_failures'] += 1
		finally:
			if self.transport and hasattr(self.transport.stdin, 'close'):
				self.transport.stdin.close()
