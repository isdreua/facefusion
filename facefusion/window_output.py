import atexit
import multiprocessing
import threading
from functools import lru_cache
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

from facefusion import logger, state_manager
from facefusion.common_helper import get_first
from facefusion.filesystem import is_image
from facefusion.types import Fps
from facefusion.webcam_config import NO_SOURCE_FACE_NAME, WEBCAM_PERFORMANCE_OVERLAYS
from facefusion.window_process import FRAME_SLOT_COUNT, probe_displays, run_window

# How many faces the window can reach. The digit keys stop at 9, so the arrows walk the same nine
# rather than reaching faces no other key can name
WINDOW_FACE_KEY_COUNT = 9


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


# The displays attached right now, as one desktop size each in SDL's order, so the display dropdowns
# offer exactly those. None when pygame is missing or the probe found no display to ask
def detect_window_displays() -> Optional[List[Tuple[int, int]]]:
	if not has_window_output():
		return None
	context = multiprocessing.get_context('spawn')
	result_queue = context.Queue()
	probe_process = context.Process(target = probe_displays, args = (result_queue,), daemon = True)
	display_sizes = None

	try:
		probe_process.start()
		display_sizes = result_queue.get(timeout = 10)
	except Exception:
		logger.debug('the displays could not be detected', __name__)
	finally:
		probe_process.join(1.0)
		if probe_process.is_alive():
			probe_process.terminate()
	if not display_sizes:
		return None
	return [ (int(display_width), int(display_height)) for display_width, display_height in display_sizes ]


def log_window_output_requirement() -> None:
	logger.warn('the native output window needs pygame, which is not installed; run "pip install pygame-ce"', __name__)


# A LatestFrameWriter transport that hands finished frames to a native window instead of an encoder.
# The window lives in its own spawned process for two reasons: macOS requires a window event loop on
# the process main thread, which gradio already owns, and forking a process that holds live
# onnxruntime or cuda contexts deadlocks
class WindowStream:
	def __init__(self, width : int, height : int, fps : Fps, title : str = 'FaceFusion', is_fullscreen : bool = False, window_display : int = 0):
		self.width = width
		self.window_display = int(window_display or 0)
		self.is_fullscreen = bool(is_fullscreen)
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
		self.process = context.Process(target = run_window, args = (self.shared_frames.name, width, height, float(fps), title, bool(is_fullscreen), self.window_display, self.sequence, self.frame_event, self.stop_event, self.intent_queue, self.status_queue), daemon = True)
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
	def push_status(self, source_face_names : List[str], source_face_name : Optional[str], source_face_path : Optional[str], is_bypassed : bool) -> None:
		status = { 'source_face_names': list(source_face_names), 'source_face_name': source_face_name, 'source_face_path': source_face_path, 'is_bypassed': is_bypassed }

		if status == self.last_status or self.is_closed or self.is_window_closed:
			return
		self.last_status = status
		try:
			self.status_queue.put_nowait(status)
		except Exception:
			pass

	def has_window_closed(self) -> bool:
		return self.is_window_closed

	# Moves the open window, fullscreen or not, instead of reopening it
	def set_display(self, window_display : int) -> None:
		self.window_display = int(window_display or 0)
		if self.is_closed or self.is_window_closed:
			return
		try:
			self.status_queue.put_nowait({ 'window_display': self.window_display })
		except Exception:
			pass

	# Switches the open window in place: a new window would lose a capture bound to this one
	def set_fullscreen(self, is_fullscreen : bool) -> None:
		self.is_fullscreen = bool(is_fullscreen)
		if self.is_closed or self.is_window_closed:
			return
		try:
			self.status_queue.put_nowait({ 'window_fullscreen': self.is_fullscreen })
		except Exception:
			pass

	def set_title(self, title : str) -> None:
		if self.is_closed or self.is_window_closed:
			return
		try:
			self.status_queue.put_nowait({ 'window_title': title })
		except Exception:
			pass

	# Black rather than the last frame, so a stopped stream does not leave a swapped face standing on
	# screen as if it were live
	def clear(self) -> None:
		self.write(bytes(self.frame_size))

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


# The title of the processed output window, so it can be told apart from the raw camera window
# beside it and picked by title in capture software
OUTPUT_WINDOW_TITLE = 'FaceFusion OUTPUT'


# A second, plain window showing what the camera delivers before any processor touches it. It is
# fed from the capture thread through submit, which converts and stores only the newest frame, and
# draws on its own thread so neither the capture nor the processing waits on the window. The window
# matches the first frame's size, because the camera may not have granted the resolution that was
# asked for. Only F11 and Esc (fullscreen) work in it, the face keys are ignored,
# and closing it ends only this view, never the stream
class RawCameraView:
	def __init__(self, title : str, fps : Fps, is_fullscreen : bool = False, window_display : int = 0, camera_switch : Any = None, is_shared : bool = False):
		self.title = title
		self.fps = fps
		self.is_fullscreen = is_fullscreen
		self.window_display = window_display
		self.camera_switch = camera_switch
		self.is_shared = is_shared
		self.latest_frame : Any = None
		self.frame_lock = threading.Lock()
		self.frame_event = threading.Event()
		self.is_running = True
		self.window_stream : Optional[WindowStream] = None
		self.thread = threading.Thread(target = self._run, daemon = True)
		self.thread.start()
		if camera_switch:
			camera_switch.listeners.append(self.on_camera_switched)

	# Called from the display dropdown. A window not open yet opens on the new display
	def set_display(self, window_display : int) -> None:
		self.window_display = int(window_display or 0)
		window_stream = self.window_stream
		if window_stream:
			window_stream.set_display(self.window_display)

	# Called on the capture thread. A window not open yet takes the new title when it opens
	def on_camera_switched(self, camera_id : int, camera_title : str) -> None:
		self.title = camera_title
		window_stream = self.window_stream
		if window_stream:
			window_stream.set_title(camera_title)

	# Converted here on the capture thread rather than later on the view's own: the conversion is the
	# copy, and the camera frame it reads goes on to the processors, which may write into it
	def submit(self, vision_frame : Any) -> None:
		import cv2

		if not self.is_running:
			return
		vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2RGB)
		with self.frame_lock:
			self.latest_frame = vision_frame
		self.frame_event.set()

	# The browser layout's camera window is the shared one, already open while the option is on, and
	# reopened here only if the camera delivers another size than it was opened at. Headless owns its
	# window. Either way a window always matches the frame, so the picture is never stretched
	def acquire_window(self, frame_width : int, frame_height : int) -> WindowStream:
		if self.is_shared:
			return acquire_shared_camera_window(frame_width, frame_height, self.fps, self.title, self.is_fullscreen, self.window_display)
		if self.window_stream:
			self.window_stream.close()
		return WindowStream(frame_width, frame_height, self.fps, self.title, self.is_fullscreen, self.window_display)

	def _run(self) -> None:
		try:
			while self.is_running:
				if not self.frame_event.wait(0.1):
					# No camera frames (one being reopened or switched to): its keys are still read,
					# so Ctrl+0-9 can leave a camera that is gone and closing the window still works
					if self.window_stream:
						self.poll_window_intents()
					continue
				self.frame_event.clear()
				with self.frame_lock:
					vision_frame = self.latest_frame
					self.latest_frame = None
				if vision_frame is None:
					continue
				frame_height, frame_width = vision_frame.shape[:2]
				if not self.window_stream or (frame_width, frame_height) != (self.window_stream.width, self.window_stream.height):
					self.window_stream = self.acquire_window(frame_width, frame_height)
				self.window_stream.write(vision_frame.tobytes())
				self.poll_window_intents()
		except Exception as exception:
			logger.warn('the raw camera window stopped: ' + str(exception), __name__)
			self.is_running = False

	# Ctrl+0-9 works here too, since with any output mode but window this is the only window
	def poll_window_intents(self) -> None:
		for intent_name, intent_value in self.window_stream.poll_events():
			if intent_name == 'camera_id' and self.camera_switch:
				self.camera_switch.request(intent_value)
		if self.window_stream.has_window_closed():
			self.is_running = False

	# The shared window goes black and stays while the option is on; an owned one closes
	def close(self) -> None:
		global ACTIVE_RAW_CAMERA_VIEW

		if ACTIVE_RAW_CAMERA_VIEW is self:
			ACTIVE_RAW_CAMERA_VIEW = None
		self.is_running = False
		if self.camera_switch and self.on_camera_switched in self.camera_switch.listeners:
			self.camera_switch.listeners.remove(self.on_camera_switched)
		self.frame_event.set()
		self.thread.join(1.0)
		if self.window_stream and self.is_shared:
			if not self.window_stream.is_closed and not self.window_stream.has_window_closed():
				self.window_stream.clear()
		elif self.window_stream:
			self.window_stream.close()
		self.window_stream = None


# The one camera window of the running stream, so the display dropdown can move it while it is open
ACTIVE_RAW_CAMERA_VIEW : Optional[RawCameraView] = None


# Started next to any output mode, the native window or not. Without pygame it is skipped with a
# warning rather than refusing the stream, since the processed output does not depend on it
def open_raw_camera_view(camera_title : str, fps : Fps, is_fullscreen : bool = False, window_display : int = 0, camera_switch : Any = None, is_shared : bool = False) -> Optional[RawCameraView]:
	if not has_window_output():
		log_window_output_requirement()
		return None
	global ACTIVE_RAW_CAMERA_VIEW

	ACTIVE_RAW_CAMERA_VIEW = RawCameraView(camera_title, fps, is_fullscreen, window_display, camera_switch, is_shared)
	return ACTIVE_RAW_CAMERA_VIEW


def move_raw_camera_view(window_display : int) -> None:
	raw_camera_view = ACTIVE_RAW_CAMERA_VIEW
	if raw_camera_view:
		raw_camera_view.set_display(window_display)
	# The shared window is moved once: through the view when it draws into it, directly when idle
	if not raw_camera_view or raw_camera_view.window_stream is not SHARED_CAMERA_WINDOW:
		move_shared_camera_window(window_display)


# The browser layout keeps the camera window open for as long as its option is on, like the output
# window, black while no stream feeds it. Opened at the capture resolution that was asked for; the
# first frame of a stream reopens it if the camera granted another size
SHARED_CAMERA_WINDOW : Optional[WindowStream] = None
SHARED_CAMERA_LOCK = threading.RLock()


# Reused while open at the same size; retitled, moved and switched to or from fullscreen in place, so
# a capture bound to it survives. Keys pressed while no stream was reading them are dropped
def acquire_shared_camera_window(width : int, height : int, fps : Fps, title : str, is_fullscreen : bool, window_display : int = 0) -> WindowStream:
	global SHARED_CAMERA_WINDOW

	with SHARED_CAMERA_LOCK:
		if SHARED_CAMERA_WINDOW:
			SHARED_CAMERA_WINDOW.poll_events()
			is_resized = (SHARED_CAMERA_WINDOW.width, SHARED_CAMERA_WINDOW.height) != (width, height)
			if SHARED_CAMERA_WINDOW.has_window_closed() or SHARED_CAMERA_WINDOW.is_closed or is_resized:
				SHARED_CAMERA_WINDOW.close()
				SHARED_CAMERA_WINDOW = None
		if SHARED_CAMERA_WINDOW:
			SHARED_CAMERA_WINDOW.set_title(title)
			if SHARED_CAMERA_WINDOW.window_display != int(window_display or 0):
				SHARED_CAMERA_WINDOW.set_display(window_display)
			if SHARED_CAMERA_WINDOW.is_fullscreen != bool(is_fullscreen):
				SHARED_CAMERA_WINDOW.set_fullscreen(is_fullscreen)
		else:
			SHARED_CAMERA_WINDOW = WindowStream(width, height, fps, title, is_fullscreen, window_display)
			SHARED_CAMERA_WINDOW.clear()
		return SHARED_CAMERA_WINDOW


# For the option and the settings changing while no stream runs. A running stream's view owns the
# window's size and title until it stops, so it is left alone then
def acquire_idle_camera_window(width : int, height : int, fps : Fps, title : str, is_fullscreen : bool, window_display : int = 0) -> None:
	if ACTIVE_RAW_CAMERA_VIEW:
		return
	acquire_shared_camera_window(width, height, fps, title, is_fullscreen, window_display)


# The camera fullscreen checkbox: the running view keeps the choice for a window it reopens at
# another size, and the open window switches in place
def set_camera_window_fullscreen(is_fullscreen : bool) -> None:
	raw_camera_view = ACTIVE_RAW_CAMERA_VIEW
	if raw_camera_view:
		raw_camera_view.is_fullscreen = bool(is_fullscreen)
	with SHARED_CAMERA_LOCK:
		if SHARED_CAMERA_WINDOW and not SHARED_CAMERA_WINDOW.has_window_closed():
			SHARED_CAMERA_WINDOW.set_fullscreen(is_fullscreen)
	if raw_camera_view and raw_camera_view.window_stream and raw_camera_view.window_stream is not SHARED_CAMERA_WINDOW:
		raw_camera_view.window_stream.set_fullscreen(is_fullscreen)


def move_shared_camera_window(window_display : int) -> None:
	with SHARED_CAMERA_LOCK:
		if SHARED_CAMERA_WINDOW and not SHARED_CAMERA_WINDOW.has_window_closed():
			SHARED_CAMERA_WINDOW.set_display(window_display)


# Turning the option off closes it at once, also under a running stream, whose view then stops
# drawing: it only writes into a closed window
def release_shared_camera_window() -> None:
	global SHARED_CAMERA_WINDOW

	with SHARED_CAMERA_LOCK:
		if SHARED_CAMERA_WINDOW:
			SHARED_CAMERA_WINDOW.close()
			SHARED_CAMERA_WINDOW = None


@atexit.register
def close_shared_camera_window() -> None:
	release_shared_camera_window()


# The browser layout keeps one window open across streams while the window mode is selected, so
# starting and stopping does not make it flash away and come back at another place. Handlers run on
# gradio's worker threads, hence the lock
SHARED_WINDOW_STREAM : Optional[WindowStream] = None
SHARED_WINDOW_LOCK = threading.Lock()
SHARED_WINDOW_STATE : Dict[str, bool] = { 'in_use': False, 'close_pending': False }


# Reused when it is still open at the size asked for. The shared memory is sized at creation, so
# another output size needs another window. Keys pressed while no stream was reading them are
# dropped, or the next start would replay them as face switches
def acquire_shared_window_stream(width : int, height : int, fps : Fps, is_fullscreen : bool, window_display : int = 0) -> WindowStream:
	global SHARED_WINDOW_STREAM

	with SHARED_WINDOW_LOCK:
		if SHARED_WINDOW_STREAM:
			SHARED_WINDOW_STREAM.poll_events()
			# A stream still writing keeps its window; the new size is picked up by the next start
			is_resized = (SHARED_WINDOW_STREAM.width, SHARED_WINDOW_STREAM.height) != (width, height) and not SHARED_WINDOW_STATE.get('in_use')
			if SHARED_WINDOW_STREAM.has_window_closed() or is_resized:
				SHARED_WINDOW_STREAM.close()
				SHARED_WINDOW_STREAM = None
		# Another display or fullscreen choice is applied to the open window, never by reopening it
		if SHARED_WINDOW_STREAM:
			if SHARED_WINDOW_STREAM.window_display != int(window_display or 0):
				SHARED_WINDOW_STREAM.set_display(window_display)
			if SHARED_WINDOW_STREAM.is_fullscreen != bool(is_fullscreen):
				SHARED_WINDOW_STREAM.set_fullscreen(is_fullscreen)
		if not SHARED_WINDOW_STREAM:
			SHARED_WINDOW_STREAM = WindowStream(width, height, fps, OUTPUT_WINDOW_TITLE, is_fullscreen, window_display)
			SHARED_WINDOW_STREAM.clear()
		SHARED_WINDOW_STATE['close_pending'] = False
		return SHARED_WINDOW_STREAM


# The fullscreen checkbox switches the open shared window right away, whether or not a stream runs
def set_shared_window_fullscreen(is_fullscreen : bool) -> None:
	with SHARED_WINDOW_LOCK:
		if SHARED_WINDOW_STREAM and not SHARED_WINDOW_STREAM.has_window_closed():
			SHARED_WINDOW_STREAM.set_fullscreen(is_fullscreen)


# The display dropdown moves the open shared window right away, whether or not a stream is running
def move_shared_window_stream(window_display : int) -> None:
	with SHARED_WINDOW_LOCK:
		if SHARED_WINDOW_STREAM and not SHARED_WINDOW_STREAM.has_window_closed():
			SHARED_WINDOW_STREAM.set_display(window_display)


# A running stream keeps its window even when the mode is switched away underneath it; the close is
# then left to the end of that stream
def release_shared_window_stream() -> None:
	global SHARED_WINDOW_STREAM

	with SHARED_WINDOW_LOCK:
		if SHARED_WINDOW_STATE.get('in_use'):
			SHARED_WINDOW_STATE['close_pending'] = True
			return
		if SHARED_WINDOW_STREAM:
			SHARED_WINDOW_STREAM.close()
			SHARED_WINDOW_STREAM = None


# The window now outlives every stream, so nothing else would close it on exit and its shared memory
# would be left behind; at exit there is no stream to wait for
@atexit.register
def close_shared_window_stream() -> None:
	global SHARED_WINDOW_STREAM

	with SHARED_WINDOW_LOCK:
		if SHARED_WINDOW_STREAM:
			SHARED_WINDOW_STREAM.close()
			SHARED_WINDOW_STREAM = None


# What a stream hands to LatestFrameWriter in place of the window itself. The writer closes its
# transport when it finishes, which for the shared window has to mean going black, not going away
class SharedWindowTransport:
	def __init__(self, window_stream : WindowStream):
		self.window_stream = window_stream
		self.stdin = self
		with SHARED_WINDOW_LOCK:
			SHARED_WINDOW_STATE['in_use'] = True

	def write(self, frame_bytes : bytes) -> None:
		self.window_stream.write(frame_bytes)

	# Reached from the writer's finally and again from the stream's own, so it tolerates both
	def close(self) -> None:
		with SHARED_WINDOW_LOCK:
			if not SHARED_WINDOW_STATE.get('in_use'):
				return
			SHARED_WINDOW_STATE['in_use'] = False
			is_close_pending = SHARED_WINDOW_STATE.get('close_pending')
		if is_close_pending:
			release_shared_window_stream()
		else:
			self.window_stream.clear()


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


# The window process has no cv2 and no access to the face groups, so the preview travels to it as a
# path and pygame loads it there. The same first-source-path and is_image filtering as the name list
# above, so the preview is always the tile the browser gallery would show for that name
def resolve_window_face_path(source_face_name : Optional[str]) -> Optional[str]:
	from facefusion.streamer import resolve_source_face_groups

	if not source_face_name:
		return None

	source_path = get_first(resolve_source_face_groups().get(source_face_name) or [])
	if source_path and is_image(source_path):
		return source_path
	return None


def apply_window_intents(window_stream : WindowStream, camera_switch : Any = None) -> bool:
	is_running = True

	for intent_name, intent_value in window_stream.poll_events():
		if intent_name == 'closed':
			is_running = False
		# The face key groups are one exclusive choice, not a toggle and a selection: 1-9 pick a
		# face, left and right step through those same nine, 0 picks none, '-' picks the raw
		# camera. Pressing one twice changes nothing, and picking a face is what leaves the raw
		# camera again
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
		# Wrapping at both ends. The direction is all the window reports; the step is resolved here
		# because the parent owns the index to name mapping
		elif intent_name == 'face_step':
			source_face_names = resolve_window_face_names()[:WINDOW_FACE_KEY_COUNT]
			if source_face_names:
				source_face_name = state_manager.get_item('source_face_name')
				if source_face_name in source_face_names:
					face_index = (source_face_names.index(source_face_name) + intent_value) % len(source_face_names)
				else:
					# Coming from no face at all, so the list is entered at the end it is being
					# walked towards rather than at whatever was selected before
					face_index = 0 if intent_value > 0 else len(source_face_names) - 1
				state_manager.set_item('source_face_name', source_face_names[face_index])
				state_manager.set_item('webcam_stream_bypass', False)
			else:
				logger.debug('no source faces are bound to the arrow keys', __name__)
		elif intent_name == 'bypass_on':
			state_manager.set_item('webcam_stream_bypass', True)
		# Ctrl+0-9. Only a request: the capture thread opens the camera and stays on the old one if
		# it cannot
		elif intent_name == 'camera_id':
			if camera_switch:
				camera_switch.request(intent_value)
			else:
				logger.debug('camera switching is not available for this stream', __name__)
		# A, S and D switch the overlay in place. The window is the one output with no browser
		# beside it, so without these a headless run cannot be inspected and then cleaned up again.
		# Checked against the list even though the child is ours, because the value crossed a
		# process boundary to get here
		elif intent_name == 'overlay_mode':
			if intent_value in WEBCAM_PERFORMANCE_OVERLAYS:
				state_manager.set_item('webcam_performance_overlay', intent_value)
			else:
				logger.debug('unknown overlay mode ' + str(intent_value) + ' was ignored', __name__)

	# The on screen display is part of the overlay, so a run with the overlay off keeps a clean frame
	# and never names the face it switched to
	if state_manager.get_item('webcam_performance_overlay') in [ 'simple', 'advanced' ]:
		source_face_name = state_manager.get_item('source_face_name')
		is_bypassed = bool(state_manager.get_item('webcam_stream_bypass'))
		# No preview on the raw camera: the selected face is still remembered, but none of it is
		# reaching the frame, so showing it would claim a swap that is not happening
		source_face_path = None if is_bypassed else resolve_window_face_path(source_face_name)
		window_stream.push_status(resolve_window_face_names(), source_face_name, source_face_path, is_bypassed)
	return is_running and not window_stream.has_window_closed()
