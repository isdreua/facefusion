from typing import Any, Dict, Optional, Tuple

# This module is the target of a spawned process, so spawn re-imports it in the child. Every heavy
# import therefore lives inside run_window: the parent imports this module only to name the target,
# and must not drag pygame in, while the child must not drag onnxruntime in
FRAME_SLOT_COUNT = 2
OSD_DURATION = 1.5
OSD_MARGIN = 12
# The preview is sized against the frame's shorter side - the height of a landscape window, the
# width of a portrait one - so it keeps the same presence whatever the output resolution is
OSD_PREVIEW_DIVISOR = 3
OSD_PREVIEW_GAP = 6


def resolve_face_index(pygame : Any, key : int) -> Optional[int]:
	digit_keys =\
	[
		(pygame.K_1, pygame.K_KP1), (pygame.K_2, pygame.K_KP2), (pygame.K_3, pygame.K_KP3),
		(pygame.K_4, pygame.K_KP4), (pygame.K_5, pygame.K_KP5), (pygame.K_6, pygame.K_KP6),
		(pygame.K_7, pygame.K_KP7), (pygame.K_8, pygame.K_KP8), (pygame.K_9, pygame.K_KP9)
	]

	for face_index, keys in enumerate(digit_keys):
		if key in keys:
			return face_index

	return None


# Ctrl with a digit names a camera by its device number, 0 to 9, the numbers the camera dropdown lists
def resolve_camera_id(pygame : Any, key : int) -> Optional[int]:
	digit_keys =\
	[
		(pygame.K_0, pygame.K_KP0), (pygame.K_1, pygame.K_KP1), (pygame.K_2, pygame.K_KP2),
		(pygame.K_3, pygame.K_KP3), (pygame.K_4, pygame.K_KP4), (pygame.K_5, pygame.K_KP5),
		(pygame.K_6, pygame.K_KP6), (pygame.K_7, pygame.K_KP7), (pygame.K_8, pygame.K_KP8),
		(pygame.K_9, pygame.K_KP9)
	]

	for camera_id, keys in enumerate(digit_keys):
		if key in keys:
			return camera_id

	return None


# The overlay modes the parent accepts, keyed the way they sit under the left hand: A is the fullest
# one, S steps down to the summary, D turns it off. The value is a plain constant per key rather
# than anything the child resolves against state, so the parent stays the only owner of the mode
def resolve_overlay_mode(pygame : Any, key : int) -> Optional[str]:
	overlay_mode_keys =\
	{
		pygame.K_a: 'advanced',
		pygame.K_s: 'simple',
		pygame.K_d: 'none'
	}

	return overlay_mode_keys.get(key)


def resolve_osd_text(status : Dict[str, Any]) -> str:
	if status.get('is_bypassed'):
		return 'RAW CAMERA'
	source_face_name = status.get('source_face_name')
	if source_face_name:
		return 'FACE: ' + str(source_face_name)
	return ''


# A third of the frame's shorter side, which is the height of a landscape window and the width of a
# portrait one
def resolve_osd_preview_size(width : int, height : int) -> int:
	return max(1, round(min(width, height) / OSD_PREVIEW_DIVISOR))


# The source picture's biggest side is matched to that box, so a portrait and a landscape source
# take up the same amount of the frame rather than one of them running off it.
#
# Scaled once per path and kept, because the osd redraws on every frame it is visible for and the
# box never changes: it is sized from the frame, which is fixed for the life of the window.
# A path that fails to load caches None so a broken or unreadable image is not retried sixty times
# a second; the parent has already filtered the path by extension, but it may still be gone by now
def resolve_osd_preview(pygame : Any, osd_previews : Dict[str, Any], source_face_path : Optional[str], preview_size : int) -> Any:
	if not source_face_path:
		return None
	if source_face_path not in osd_previews:
		try:
			preview_surface = pygame.image.load(source_face_path)
			# convert_alpha needs a display mode, which the sdl2 window output never sets
			if pygame.display.get_surface():
				preview_surface = preview_surface.convert_alpha()
			preview_width, preview_height = preview_surface.get_size()
			preview_scale = preview_size / max(1, preview_width, preview_height)
			osd_previews[source_face_path] = pygame.transform.smoothscale(preview_surface, (max(1, round(preview_width * preview_scale)), max(1, round(preview_height * preview_scale))))
		except Exception:
			osd_previews[source_face_path] = None

	return osd_previews.get(source_face_path)


# The setting counts displays from 1 and keeps 0 for no pin; SDL counts from 0. A display that is
# not attached (unplugged since the config was written) drops the pin rather than failing the window
def resolve_display_index(pygame : Any, window_display : int) -> Optional[int]:
	if window_display < 1:
		return None
	try:
		display_count = pygame.display.get_num_displays()
	except Exception:
		return None
	if window_display > display_count:
		return None
	return window_display - 1


# SDL's "centred on display N" position. SDL resolves it against that display's bounds, so the window
# lands inside it whatever the window manager would otherwise choose
WINDOW_POSITION_CENTERED_DISPLAY = 0x2FFF0000
# How long a window move or fullscreen change is given to land (see SdlWindowOutput.wait_for_window)
WINDOW_SETTLE_TIMEOUT = 1.0
WINDOW_SETTLE_MINIMUM = 0.05
WINDOW_SETTLE_STABLE = 0.1


def resolve_centered_position(display_index : int) -> Tuple[int, int]:
	return WINDOW_POSITION_CENTERED_DISPLAY | display_index, WINDOW_POSITION_CENTERED_DISPLAY | display_index


def resolve_desktop_size(pygame : Any, display_index : int) -> Optional[Tuple[int, int]]:
	try:
		desktop_sizes = pygame.display.get_desktop_sizes()
	except Exception:
		return None
	if 0 <= display_index < len(desktop_sizes):
		desktop_width, desktop_height = desktop_sizes[display_index]
		if desktop_width > 0 and desktop_height > 0:
			return int(desktop_width), int(desktop_height)
	return None


# A frame larger than the display opens in a window that fits on it rather than running off the edges
def resolve_windowed_size(pygame : Any, width : int, height : int, display_index : Optional[int]) -> Tuple[int, int]:
	desktop_size = resolve_desktop_size(pygame, display_index or 0)

	if desktop_size:
		window_scale = min(1.0, desktop_size[0] * 0.9 / width, desktop_size[1] * 0.9 / height)
		return max(1, round(width * window_scale)), max(1, round(height * window_scale))
	return width, height


# The frame's box inside the target, scaled to fit with its aspect kept and centred
def resolve_fitted_rect(target_width : int, target_height : int, frame_width : int, frame_height : int) -> Tuple[int, int, int, int]:
	frame_scale = min(target_width / frame_width, target_height / frame_height)
	fitted_width = max(1, round(frame_width * frame_scale))
	fitted_height = max(1, round(frame_height * frame_scale))
	return (target_width - fitted_width) // 2, (target_height - fitted_height) // 2, fitted_width, fitted_height


# One window for the whole life of the process. Fullscreen, windowed and moving to another display
# all act on that same SDL window and never create a new one, so a capture bound to it (OBS window
# capture, which on X11 and PipeWire follows the window itself, not its title) keeps working. The
# renderer draws the frame fitted into whatever size the window has, so fullscreen fills the display
# without cropping, the scaling runs on the gpu, and a resized window keeps the picture whole.
# Synced to the display so a frame lands on a refresh instead of tearing across one
class SdlWindowOutput:
	def __init__(self, pygame : Any, width : int, height : int, title : str, is_fullscreen : bool, display_index : Optional[int]):
		from pygame._sdl2.video import Renderer, Texture, Window

		self.pygame = pygame
		self.width = width
		self.height = height
		self.is_fullscreen = bool(is_fullscreen)
		window_options : Dict[str, Any] = { 'resizable': True }
		if display_index is not None:
			window_options['position'] = resolve_centered_position(display_index)
		# Created fullscreen rather than switched after, so it never shows a windowed frame first; SDL
		# takes the display from the position, and the windowed size is kept for F11
		self.window = Window(title, size = resolve_windowed_size(pygame, width, height, display_index), fullscreen_desktop = self.is_fullscreen, **window_options)
		try:
			self.renderer = Renderer(self.window, vsync = True)
		except Exception:
			self.renderer = Renderer(self.window)
		self.texture = Texture(self.renderer, (width, height), streaming = True)
		self.texture_class = Texture
		self.window.show()

	def set_title(self, title : str) -> None:
		self.window.title = title

	def set_fullscreen(self, is_fullscreen : bool) -> None:
		if is_fullscreen == self.is_fullscreen:
			return
		if is_fullscreen:
			self.window.set_fullscreen(desktop = True)
		else:
			self.window.set_windowed()
		self.is_fullscreen = is_fullscreen
		self.wait_for_window()

	# Fullscreen goes to the display the window manager thinks the window is on, and on X11 a move or
	# a fullscreen change only takes effect once the window manager has handled it. Going fullscreen
	# straight after the move therefore took the old display's size and put it on the new one, where
	# it stuck out across both. So every step waits for the window to settle before the next: leave
	# fullscreen, size the window for the target display, centre it there, then fullscreen again
	def move_to_display(self, display_index : int) -> None:
		is_fullscreen = self.is_fullscreen

		if is_fullscreen:
			self.set_fullscreen(False)
		self.window.size = resolve_windowed_size(self.pygame, self.width, self.height, display_index)
		self.wait_for_window()
		self.window.position = resolve_centered_position(display_index)
		self.wait_for_window()
		if is_fullscreen:
			self.set_fullscreen(True)

	# Pumps events (they stay queued for the main loop) until the window's position and size have
	# stopped changing for a moment, or a second has passed. The first moment is always waited, since
	# the window manager may not have started on the change yet
	def wait_for_window(self) -> None:
		import time

		started_at = time.perf_counter()
		window_state = None
		stable_since = started_at

		while time.perf_counter() - started_at < WINDOW_SETTLE_TIMEOUT:
			self.pygame.event.pump()
			current_window_state = (tuple(self.window.position), tuple(self.window.size))
			if current_window_state != window_state:
				window_state = current_window_state
				stable_since = time.perf_counter()
			elif time.perf_counter() - started_at > WINDOW_SETTLE_MINIMUM and time.perf_counter() - stable_since > WINDOW_SETTLE_STABLE:
				return
			time.sleep(0.01)

	def present(self, canvas_surface : Any) -> None:
		try:
			self.texture.update(canvas_surface)
			frame_texture = self.texture
		except Exception:
			frame_texture = self.texture_class.from_surface(self.renderer, canvas_surface)
		viewport = self.renderer.get_viewport()
		self.renderer.draw_color = (0, 0, 0, 255)
		self.renderer.clear()
		frame_texture.draw(dstrect = resolve_fitted_rect(viewport.width, viewport.height, self.width, self.height))
		self.renderer.present()


# For a pygame without the sdl2 window classes: the old single set_mode with SCALED, which already
# keeps one window. F11 uses toggle_fullscreen on that window; moving to another display is not
# possible there and is ignored
class LegacyWindowOutput:
	def __init__(self, pygame : Any, width : int, height : int, title : str, is_fullscreen : bool, display_index : Optional[int]):
		import os

		self.pygame = pygame
		self.is_fullscreen = bool(is_fullscreen)
		display_flags = pygame.RESIZABLE | pygame.SCALED
		display_options : Dict[str, Any] = {}
		if is_fullscreen:
			display_flags = display_flags | pygame.FULLSCREEN
		if display_index is not None:
			os.environ['SDL_VIDEO_CENTERED'] = '1'
			display_options['display'] = display_index
		pygame.display.set_caption(title)
		try:
			self.display_surface = pygame.display.set_mode((width, height), display_flags, vsync = 1, **display_options)
		except pygame.error:
			self.display_surface = pygame.display.set_mode((width, height), display_flags, **display_options)

	def set_title(self, title : str) -> None:
		self.pygame.display.set_caption(title)

	def set_fullscreen(self, is_fullscreen : bool) -> None:
		if is_fullscreen != self.is_fullscreen:
			self.pygame.display.toggle_fullscreen()
			self.is_fullscreen = is_fullscreen

	def move_to_display(self, display_index : int) -> None:
		pass

	def present(self, canvas_surface : Any) -> None:
		self.display_surface.blit(canvas_surface, (0, 0))
		self.pygame.display.flip()


def create_window_output(pygame : Any, width : int, height : int, title : str, is_fullscreen : bool, display_index : Optional[int]) -> Any:
	try:
		return SdlWindowOutput(pygame, width, height, title, is_fullscreen, display_index)
	except ImportError:
		return LegacyWindowOutput(pygame, width, height, title, is_fullscreen, display_index)


# Run in its own short lived spawned process for the same reason the window is: the browser process
# must not start a video subsystem (macOS allows that only on the main thread, which gradio owns).
# Reports one desktop size per display in SDL's order, or None when there is no display to ask
def probe_displays(result_queue : Any) -> None:
	import os

	os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
	display_sizes = None

	try:
		import pygame

		pygame.display.init()
		if hasattr(pygame.display, 'get_desktop_sizes'):
			display_sizes = [ (int(display_width), int(display_height)) for display_width, display_height in pygame.display.get_desktop_sizes() ]
		else:
			display_sizes = [ (0, 0) ] * pygame.display.get_num_displays()
		pygame.quit()
	except BaseException:
		display_sizes = None
	result_queue.put(display_sizes)


# The child reports raw intents and resolves nothing: the parent owns the index to face name mapping
# so that the window and the browser gallery cannot disagree about what is selected
def run_window(shared_name : str, width : int, height : int, fps : float, title : str, is_fullscreen : bool, window_display : int, sequence : Any, frame_event : Any, stop_event : Any, intent_queue : Any, status_queue : Any) -> None:
	import os
	import time
	from multiprocessing import shared_memory

	os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

	import pygame

	frame_size = width * height * 3
	shared_frames = None
	status : Dict[str, Any] = { 'source_face_names': [], 'source_face_name': None, 'source_face_path': None, 'is_bypassed': False }
	osd_previews : Dict[str, Any] = {}
	osd_preview_size = resolve_osd_preview_size(width, height)
	osd_expires_at = 0.0
	is_closing = False

	try:
		shared_frames = shared_memory.SharedMemory(name = shared_name)
		pygame.init()
		display_index = resolve_display_index(pygame, window_display)
		window_output = create_window_output(pygame, width, height, title, bool(is_fullscreen), display_index)
		# The frame and the osd are composed here at the frame's own size, and the window output
		# scales the result to the window
		canvas_surface = pygame.Surface((width, height), 0, 32)
		osd_font = pygame.font.SysFont(None, 24)
		last_sequence = -1

		while not is_closing and not stop_event.is_set():
			for event in pygame.event.get():
				if event.type in [ pygame.QUIT, getattr(pygame, 'WINDOWCLOSE', pygame.QUIT) ]:
					is_closing = True
				# The window manager may resize the window after the call that asked for it returned
				# (fullscreen on X11 among them), so the frame is drawn again at the size it got
				elif event.type in [ getattr(pygame, 'WINDOWSIZECHANGED', -1), getattr(pygame, 'WINDOWRESIZED', -1), pygame.VIDEORESIZE ]:
					last_sequence = -1
				elif event.type == pygame.KEYDOWN:
					camera_id = resolve_camera_id(pygame, event.key) if event.mod & pygame.KMOD_CTRL else None
					face_index = resolve_face_index(pygame, event.key)
					overlay_mode = resolve_overlay_mode(pygame, event.key)
					if event.key == pygame.K_F11:
						# A pinned window that was dragged away goes back to its display first
						if display_index is not None and not window_output.is_fullscreen:
							window_output.move_to_display(display_index)
						window_output.set_fullscreen(not window_output.is_fullscreen)
						last_sequence = -1
					elif event.key == pygame.K_q and event.mod & pygame.KMOD_CTRL:
						is_closing = True
					# Before the face keys, which share the digits
					elif camera_id is not None:
						intent_queue.put(('camera_id', camera_id))
					elif event.key == pygame.K_ESCAPE:
						# Escape leaves fullscreen first, so it cannot close the window by surprise
						if window_output.is_fullscreen:
							window_output.set_fullscreen(False)
							last_sequence = -1
						else:
							is_closing = True
					elif face_index is not None:
						intent_queue.put(('face_index', face_index))
					elif event.key in [ pygame.K_0, pygame.K_KP0 ]:
						intent_queue.put(('face_none', None))
					elif event.key in [ pygame.K_MINUS, pygame.K_KP_MINUS ]:
						intent_queue.put(('bypass_on', None))
					elif overlay_mode is not None:
						intent_queue.put(('overlay_mode', overlay_mode))
					elif event.key == pygame.K_LEFT:
						intent_queue.put(('face_step', -1))
					elif event.key == pygame.K_RIGHT:
						intent_queue.put(('face_step', 1))

			while True:
				try:
					received_status = status_queue.get_nowait()
				except Exception:
					break
				# A new title (the camera window after a camera switch) is not a status and shows no osd
				if 'window_title' in received_status:
					window_output.set_title(str(received_status.get('window_title')))
					continue
				# The display dropdown changed while the window is open. Auto drops the pin and leaves
				# the window where it is; a display that is not attached is ignored the same way
				if 'window_display' in received_status:
					display_index = resolve_display_index(pygame, int(received_status.get('window_display') or 0))
					if display_index is not None:
						window_output.move_to_display(display_index)
						last_sequence = -1
					continue
				# The fullscreen checkbox changed while the window is open. Entering goes to the pinned
				# display, in case the window was dragged off it
				if 'window_fullscreen' in received_status:
					is_fullscreen_wanted = bool(received_status.get('window_fullscreen'))
					if is_fullscreen_wanted and not window_output.is_fullscreen and display_index is not None:
						window_output.move_to_display(display_index)
					window_output.set_fullscreen(is_fullscreen_wanted)
					last_sequence = -1
					continue
				status = received_status
				osd_expires_at = time.monotonic() + OSD_DURATION

			# A mode change above (F11, Esc, a display move) reset last_sequence, so the current frame
			# is drawn again at the window's new size rather than waiting for the next one
			current_sequence = int(sequence.value)
			if current_sequence != last_sequence and current_sequence > 0:
				last_sequence = current_sequence
				frame_slot = (current_sequence - 1) % FRAME_SLOT_COUNT
				frame_offset = frame_slot * frame_size
				# Copied out of the shared slot before it is drawn, so a frame the parent writes
				# meanwhile cannot tear the one on screen
				frame_bytes = bytes(shared_frames.buf[frame_offset:frame_offset + frame_size])
				frame_surface = pygame.image.frombuffer(frame_bytes, (width, height), 'RGB')
				canvas_surface.blit(frame_surface, (0, 0))
				osd_text = resolve_osd_text(status)
				if osd_text and time.monotonic() < osd_expires_at:
					text_surface = osd_font.render(osd_text, True, (255, 255, 255), (0, 0, 0))
					canvas_surface.blit(text_surface, (OSD_MARGIN, OSD_MARGIN))
					# Under the text, and only while the text is up, so switching faces shows what
					# was switched to rather than leaving a thumbnail sitting on the frame
					preview_surface = resolve_osd_preview(pygame, osd_previews, status.get('source_face_path'), osd_preview_size)
					if preview_surface:
						canvas_surface.blit(preview_surface, (OSD_MARGIN, OSD_MARGIN + text_surface.get_height() + OSD_PREVIEW_GAP))
				window_output.present(canvas_surface)
			else:
				# Nothing new to draw, so wait on the parent instead of spinning; the sequence above
				# is the real source of truth, which makes a missed event harmless
				frame_event.wait(0.005)
				frame_event.clear()
			# No frame rate cap here any more: it slept after every flip, so a frame arriving meanwhile
			# waited up to half a frame and the generated stream reached the screen unevenly. The loop
			# only draws on a new frame, idles on the event otherwise, and vsync paces the presents
	except BaseException:
		pass
	finally:
		try:
			intent_queue.put(('closed', None))
		except Exception:
			pass
		try:
			pygame.quit()
		except Exception:
			pass
		if shared_frames:
			shared_frames.close()
