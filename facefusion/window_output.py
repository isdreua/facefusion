import multiprocessing
from functools import lru_cache
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

from facefusion import logger, state_manager
from facefusion.common_helper import get_first
from facefusion.filesystem import is_image
from facefusion.types import Fps
from facefusion.webcam_config import NO_SOURCE_FACE_NAME
from facefusion.window_process import FRAME_SLOT_COUNT, run_window


# The pip distribution is pygame-ce but it imports as pygame, which it shares with classic pygame;
# either one satisfies the window, so the import name is what gets probed
@lru_cache()
def resolve_pygame_module() -> Any:
	try:
		return __import__('pygame')
	except Exception:
		return None


def has_window_output() -> bool:
	return resolve_pygame_module() is not None


def log_window_output_requirement() -> None:
	logger.warn('the native output window needs pygame, which is not installed; run "pip install pygame-ce"', __name__)


# A LatestFrameWriter transport that hands finished frames to a native window instead of an encoder.
# The window lives in its own spawned process for two reasons: macOS requires a window event loop on
# the process main thread, which gradio already owns, and forking a process that holds live
# onnxruntime or cuda contexts deadlocks
class WindowStream:
	def __init__(self, width : int, height : int, fps : Fps, title : str = 'FaceFusion', is_fullscreen : bool = False):
		self.width = width
		self.height = height
		self.frame_size = width * height * 3
		# LatestFrameWriter writes to transport.stdin, the same shim WindowsVirtualCameraStream uses
		self.stdin = self
		self.is_window_closed = False
		self.is_closed = False
		self.last_status : Optional[Dict[str, Any]] = None
		context = multiprocessing.get_context('spawn')
		self.shared_frames = shared_memory.SharedMemory(create = True, size = self.frame_size * FRAME_SLOT_COUNT)
		self.sequence = context.Value('q', 0)
		self.frame_event = context.Event()
		self.stop_event = context.Event()
		self.intent_queue = context.Queue()
		self.status_queue = context.Queue()
		self.process = context.Process(target = run_window, args = (self.shared_frames.name, width, height, float(fps), title, bool(is_fullscreen), self.sequence, self.frame_event, self.stop_event, self.intent_queue, self.status_queue), daemon = True)
		self.process.start()

	# One writer, one reader, and the reader only ever draws the slot of an already bumped sequence,
	# so the frame on screen is never the one being written and no lock is needed on the hot path
	def write(self, frame_bytes : bytes) -> None:
		if len(frame_bytes) != self.frame_size:
			raise ValueError('window output frame must be ' + str(self.frame_size) + ' bytes of RGB')
		if self.is_closed or self.is_window_closed:
			return
		frame_slot = int(self.sequence.value) % FRAME_SLOT_COUNT
		frame_offset = frame_slot * self.frame_size
		self.shared_frames.buf[frame_offset:frame_offset + self.frame_size] = frame_bytes
		with self.sequence.get_lock():
			self.sequence.value += 1
		self.frame_event.set()

	def poll_events(self) -> List[Tuple[str, Any]]:
		window_intents : List[Tuple[str, Any]] = []

		while True:
			try:
				window_intent = self.intent_queue.get_nowait()
			except Exception:
				break
			if window_intent[0] == 'closed':
				self.is_window_closed = True
			window_intents.append(window_intent)

		if not self.process.is_alive():
			self.is_window_closed = True
		return window_intents

	# Only sent when something actually changed, so the window's on screen display appears on a
	# switch rather than being redrawn for every frame
	def push_status(self, source_face_names : List[str], source_face_name : Optional[str], is_bypassed : bool) -> None:
		status = { 'source_face_names': list(source_face_names), 'source_face_name': source_face_name, 'is_bypassed': is_bypassed }

		if status == self.last_status or self.is_closed or self.is_window_closed:
			return
		self.last_status = status
		try:
			self.status_queue.put_nowait(status)
		except Exception:
			pass

	def has_window_closed(self) -> bool:
		return self.is_window_closed

	# Reached from LatestFrameWriter._run's finally through transport.stdin.close(), and possibly
	# again from the caller, so it has to tolerate being called twice
	def close(self) -> None:
		if self.is_closed:
			return
		self.is_closed = True
		self.stop_event.set()
		self.frame_event.set()
		self.process.join(2.0)
		if self.process.is_alive():
			self.process.terminate()
			self.process.join(1.0)
		for window_queue in [ self.intent_queue, self.status_queue ]:
			window_queue.close()
			window_queue.cancel_join_thread()
		self.shared_frames.close()
		try:
			self.shared_frames.unlink()
		except FileNotFoundError:
			pass

	def terminate(self) -> None:
		if self.process.is_alive():
			self.process.terminate()
		self.close()


# The window reports a key, this resolves what it means. Both entry points call it so that neither
# can drift into its own control semantics, and the names come out in the same order and with the
# same is_image filtering the browser gallery uses, so key N and tile N are always the same face
def resolve_window_face_names() -> List[str]:
	# Imported late: streamer pulls in cv2 and every processor, and nothing else in this module
	# needs them
	from facefusion.streamer import resolve_source_face_groups

	source_face_names = []

	for face_name, source_paths in resolve_source_face_groups().items():
		source_path = get_first(source_paths)
		if source_path and is_image(source_path):
			source_face_names.append(face_name)

	return source_face_names


def apply_window_intents(window_stream : WindowStream) -> bool:
	is_running = True

	for intent_name, intent_value in window_stream.poll_events():
		if intent_name == 'closed':
			is_running = False
		# The three key groups are one exclusive choice, not a toggle and a selection: 1-9 pick a
		# face, 0 picks none, '-' picks the raw camera. Pressing one twice changes nothing, and
		# picking a face is what leaves the raw camera again
		elif intent_name == 'face_index':
			source_face_names = resolve_window_face_names()
			if 0 <= intent_value < len(source_face_names):
				state_manager.set_item('source_face_name', source_face_names[intent_value])
				state_manager.set_item('webcam_stream_bypass', False)
			else:
				logger.debug('no source face is bound to key ' + str(intent_value + 1), __name__)
		elif intent_name == 'face_none':
			state_manager.set_item('source_face_name', NO_SOURCE_FACE_NAME)
			state_manager.set_item('webcam_stream_bypass', False)
		elif intent_name == 'bypass_on':
			state_manager.set_item('webcam_stream_bypass', True)

	# The on screen display is part of the overlay, so a run with the overlay off keeps a clean frame
	# and never names the face it switched to
	if state_manager.get_item('webcam_performance_overlay') in [ 'simple', 'advanced' ]:
		window_stream.push_status(resolve_window_face_names(), state_manager.get_item('source_face_name'), bool(state_manager.get_item('webcam_stream_bypass')))
	return is_running and not window_stream.has_window_closed()
