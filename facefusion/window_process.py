from typing import Any, Dict, List, Optional, Tuple

# This module is the target of a spawned process, so spawn re-imports it in the child. Every heavy
# import therefore lives inside run_window: the parent imports this module only to name the target,
# and must not drag pygame in, while the child must not drag onnxruntime in
# Three, since the window draws straight from its slot without copying it first: the parent would
# have to write three frames during one draw to reach the slot being read
FRAME_SLOT_COUNT = 3
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


# pygame reads BGR directly from 2.1.3 on. An older one is given the channels swapped (the slow path,
# taken only there and only for the camera window)
def resolve_frame_format(pygame : Any, pixel_format : str) -> Tuple[str, bool]:
	if pixel_format != 'BGR':
		return 'RGB', False
	try:
		pygame.image.frombuffer(bytes(3), (1, 1), 'BGR')
		return 'BGR', False
	except Exception:
		return 'RGB', True


def resolve_frame_surface(pygame : Any, frame_view : Any, width : int, height : int, frame_format : str, is_frame_swapped : bool) -> Any:
	if is_frame_swapped:
		frame_bytes = bytes(frame_view)
		swapped_bytes = bytearray(len(frame_bytes))
		swapped_bytes[0::3] = frame_bytes[2::3]
		swapped_bytes[1::3] = frame_bytes[1::3]
		swapped_bytes[2::3] = frame_bytes[0::3]
		return pygame.image.frombuffer(swapped_bytes, (width, height), 'RGB')
	return pygame.image.frombuffer(frame_view, (width, height), frame_format)


# Each patch arrives as (x, y, width, height, RGBA bytes) in frame coordinates. A malformed one is
# skipped rather than ending the window
def resolve_overlay_patches(pygame : Any, received_patches : List[Any]) -> List[Tuple[int, int, Any]]:
	overlay_patches = []

	for received_patch in received_patches:
		try:
			patch_x, patch_y, patch_width, patch_height, patch_bytes = received_patch
			patch_surface = pygame.image.frombuffer(bytes(patch_bytes), (int(patch_width), int(patch_height)), 'RGBA')
			overlay_patches.append((int(patch_x), int(patch_y), patch_surface))
		except Exception:
			continue
	return overlay_patches


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
# The most of a display a window opens at. The control window opens smaller, since it sits
# beside other things on a working display rather than being the thing on it
WINDOW_FRACTION = 0.9
CONTROL_WINDOW_FRACTION = 0.5


# How the output window's drawing surface is sized. native draws at the display's real pixels, the
# frame scaled to them on the gpu; output forces the surface to the output resolution itself, one
# frame pixel per screen pixel: windowed at exactly that pixel size, fullscreen in a display mode of
# that size, so the display itself switches to it
WINDOW_RESOLUTIONS = [ 'native', 'output' ]


# Asks for the display's real pixels instead of the scaled ones a desktop at 150% or 200% hands out.
# Without it Windows reports a 4K display scaled 300% as 1280x720 and stretches a 1280x720 window
# over it, which is why the window looked low resolution whatever the frame was. Called in each
# window and probe process before SDL starts its video subsystem, and only ever there: it is a per
# process setting, and the browser process is left alone
def prepare_native_resolution() -> None:
	import os
	import sys

	os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS', 'permonitorv2')
	os.environ.setdefault('SDL_WINDOWS_DPI_SCALING', '0')
	os.environ.setdefault('SDL_VIDEO_HIGHDPI_DISABLED', '0')
	if sys.platform == 'win32':
		import ctypes

		try:
			# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, Windows 10 1703 and later
			if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
				return
		except Exception:
			pass
		try:
			# PROCESS_PER_MONITOR_DPI_AWARE, Windows 8.1 and later
			ctypes.windll.shcore.SetProcessDpiAwareness(2)
			return
		except Exception:
			pass
		try:
			ctypes.windll.user32.SetProcessDPIAware()
		except Exception:
			pass


# The display's largest mode, which is its panel's own resolution. Where the desktop is still handed
# out in scaled units (macOS, Wayland through SDL), this is what shows the display's real size
def resolve_native_size(pygame : Any, display_index : int, desktop_size : Tuple[int, int]) -> Tuple[int, int]:
	native_width, native_height = desktop_size

	try:
		display_modes = pygame.display.list_modes(0, pygame.FULLSCREEN, display_index)
	except Exception:
		display_modes = None
	if isinstance(display_modes, list):
		for mode_width, mode_height in display_modes:
			if int(mode_width) * int(mode_height) > native_width * native_height:
				native_width, native_height = int(mode_width), int(mode_height)
	return native_width, native_height


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


# A frame larger than the display opens in a window that fits on it rather than running off the edges.
# The fraction is how much of the display the window may take at most
def resolve_windowed_size(pygame : Any, width : int, height : int, display_index : Optional[int], window_fraction : float = WINDOW_FRACTION) -> Tuple[int, int]:
	desktop_size = resolve_desktop_size(pygame, display_index or 0)

	if desktop_size:
		window_scale = min(1.0, desktop_size[0] * window_fraction / width, desktop_size[1] * window_fraction / height)
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
# Synced to the display so a frame lands on a refresh instead of tearing across one.
#
# High dpi is allowed so the drawing surface has the display's real pixels where the desktop is
# scaled; window sizes and positions stay in the desktop's own units, which the surface's pixel
# scale (pixels per desktop unit) converts between
class SdlWindowOutput:
	def __init__(self, pygame : Any, width : int, height : int, title : str, is_fullscreen : bool, display_index : Optional[int], window_fraction : float = WINDOW_FRACTION, window_resolution : str = 'native'):
		from pygame._sdl2.video import Renderer, Texture, Window

		self.pygame = pygame
		self.width = width
		self.height = height
		self.window_fraction = window_fraction
		self.window_resolution = window_resolution if window_resolution in WINDOW_RESOLUTIONS else 'native'
		self.is_fullscreen = bool(is_fullscreen)
		self.display_index = display_index
		window_options : Dict[str, Any] = { 'resizable': True, 'allow_high_dpi': True }
		if display_index is not None:
			window_options['position'] = resolve_centered_position(display_index)
		# Created fullscreen rather than switched after, so it never shows a windowed frame first; SDL
		# takes the display from the position, and the windowed size is kept for F11. Forced to the
		# output resolution, fullscreen is the display mode of the frame's size instead of the desktop
		if self.is_fullscreen and self.window_resolution == 'output':
			window_options['fullscreen'] = True
			window_size = (width, height)
		else:
			window_options['fullscreen_desktop'] = self.is_fullscreen
			window_size = resolve_windowed_size(pygame, width, height, display_index, window_fraction)
		try:
			self.window = Window(title, size = window_size, **window_options)
		except TypeError:
			# A pygame that does not know the high dpi flag
			window_options.pop('allow_high_dpi', None)
			self.window = Window(title, size = window_size, **window_options)
		try:
			self.renderer = Renderer(self.window, vsync = True)
		except Exception:
			self.renderer = Renderer(self.window)
		self.texture = Texture(self.renderer, (width, height), streaming = True)
		self.texture_class = Texture
		self.window.show()
		# The pixel scale is only known once the window has its surface, so a window forced to the
		# output resolution is given its exact pixel size right after
		if not self.is_fullscreen and self.window_resolution == 'output':
			self.window.size = self.resolve_windowed_size(display_index)
			self.wait_for_window()
			if display_index is not None:
				self.window.position = resolve_centered_position(display_index)
				self.wait_for_window()

	# Pixels of the drawing surface per desktop unit of the window: 1 where the desktop is not scaled
	# or the process is dpi aware (Windows), 2 or 3 on a scaled macOS or Wayland desktop
	def resolve_pixel_scale(self) -> float:
		try:
			window_width = self.window.size[0]
			viewport = self.renderer.get_viewport()
			if window_width > 0 and viewport.width > 0:
				return viewport.width / window_width
		except Exception:
			pass
		return 1.0

	# native takes the usual share of the display; output is exactly the frame's pixels, shrunk only
	# when the display cannot hold that many
	def resolve_windowed_size(self, display_index : Optional[int]) -> Tuple[int, int]:
		if self.window_resolution != 'output':
			return resolve_windowed_size(self.pygame, self.width, self.height, display_index, self.window_fraction)
		pixel_scale = self.resolve_pixel_scale()
		return resolve_windowed_size(self.pygame, max(1, round(self.width / pixel_scale)), max(1, round(self.height / pixel_scale)), display_index, 1.0)

	def set_title(self, title : str) -> None:
		self.window.title = title

	# Fullscreen fills the desktop as it is, or, forced to the output resolution, switches the display
	# to the mode closest to the frame's size. Leaving it gives the window its windowed size back,
	# which a mode switch had replaced with the frame's
	def set_fullscreen(self, is_fullscreen : bool) -> None:
		if is_fullscreen == self.is_fullscreen:
			return
		if is_fullscreen and self.window_resolution == 'output':
			self.window.size = (self.width, self.height)
			self.window.set_fullscreen(desktop = False)
		elif is_fullscreen:
			self.window.set_fullscreen(desktop = True)
		else:
			self.window.set_windowed()
		self.is_fullscreen = is_fullscreen
		self.wait_for_window()
		if not is_fullscreen and self.window_resolution == 'output':
			self.window.size = self.resolve_windowed_size(self.display_index)
			self.wait_for_window()

	# Switched in place like everything else: a fullscreen window leaves fullscreen and enters it again
	# the new way, a windowed one is resized
	def set_window_resolution(self, window_resolution : str) -> None:
		if window_resolution not in WINDOW_RESOLUTIONS or window_resolution == self.window_resolution:
			return
		is_fullscreen = self.is_fullscreen

		if is_fullscreen:
			self.set_fullscreen(False)
		self.window_resolution = window_resolution
		self.window.size = self.resolve_windowed_size(self.display_index)
		self.wait_for_window()
		if is_fullscreen:
			self.set_fullscreen(True)

	# Fullscreen goes to the display the window manager thinks the window is on, and on X11 a move or
	# a fullscreen change only takes effect once the window manager has handled it. Going fullscreen
	# straight after the move therefore took the old display's size and put it on the new one, where
	# it stuck out across both. So every step waits for the window to settle before the next: leave
	# fullscreen, size the window for the target display, centre it there, then fullscreen again
	def move_to_display(self, display_index : int) -> None:
		is_fullscreen = self.is_fullscreen

		if is_fullscreen:
			self.set_fullscreen(False)
		self.display_index = display_index
		self.window.size = self.resolve_windowed_size(display_index)
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

	# The viewport is in the surface's pixels, so with high dpi the frame is scaled to the display's
	# real resolution rather than to the scaled desktop's
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

	def set_window_resolution(self, window_resolution : str) -> None:
		pass

	def present(self, canvas_surface : Any) -> None:
		self.display_surface.blit(canvas_surface, (0, 0))
		self.pygame.display.flip()


def create_window_output(pygame : Any, width : int, height : int, title : str, is_fullscreen : bool, display_index : Optional[int], window_fraction : float = WINDOW_FRACTION, window_resolution : str = 'native') -> Any:
	try:
		return SdlWindowOutput(pygame, width, height, title, is_fullscreen, display_index, window_fraction, window_resolution)
	except ImportError:
		return LegacyWindowOutput(pygame, width, height, title, is_fullscreen, display_index)


# Run in its own short lived spawned process for the same reason the window is: the browser process
# must not start a video subsystem (macOS allows that only on the main thread, which gradio owns).
# Being a fresh process every time is also what keeps the answer current: SDL reads the displays
# once at video init, so nothing here is remembered between probes.
# Reports per display, in SDL's order, (desktop width, desktop height, native width, native height):
# the desktop as it is set up now, in real pixels where the process can be made dpi aware, and the
# panel's own resolution. None when there is no display to ask
def probe_displays(result_queue : Any) -> None:
	import os

	os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
	prepare_native_resolution()
	display_sizes = None

	try:
		import pygame

		pygame.display.init()
		if hasattr(pygame.display, 'get_desktop_sizes'):
			display_sizes = []
			for display_index, (desktop_width, desktop_height) in enumerate(pygame.display.get_desktop_sizes()):
				desktop_size = (int(desktop_width), int(desktop_height))
				display_sizes.append(desktop_size + resolve_native_size(pygame, display_index, desktop_size))
		pygame.quit()
	except BaseException:
		display_sizes = None
	result_queue.put(display_sizes)


# The child reports raw intents and resolves nothing: the parent owns the index to face name mapping
# so that the window and the browser gallery cannot disagree about what is selected
# A control window (is_control) is never fullscreen and never pinned: it ignores F11 and the display and fullscreen settings, and Esc simply closes it.
# Every other key works the same, so the stream can be controlled from either window
# The frames are RGB, or BGR (pixel_format) for the camera window, which is handed the camera's own
# frame as captured rather than a converted copy of it
def run_window(shared_name : str, width : int, height : int, fps : float, title : str, is_fullscreen : bool, window_display : int, sequence : Any, frame_event : Any, stop_event : Any, intent_queue : Any, status_queue : Any, is_control : bool = False, pixel_format : str = 'RGB', window_resolution : str = 'native') -> None:
	import os
	import time
	from multiprocessing import shared_memory

	os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
	prepare_native_resolution()

	import pygame

	frame_size = width * height * 3
	shared_frames = None
	status : Dict[str, Any] = { 'source_face_names': [], 'source_face_name': None, 'source_face_path': None, 'is_bypassed': False }
	osd_previews : Dict[str, Any] = {}
	osd_preview_size = resolve_osd_preview_size(width, height)
	osd_expires_at = 0.0
	overlay_patches : List[Tuple[int, int, Any]] = []
	is_closing = False

	try:
		shared_frames = shared_memory.SharedMemory(name = shared_name)
		pygame.init()
		display_index = None if is_control else resolve_display_index(pygame, window_display)
		window_output = create_window_output(pygame, width, height, title, bool(is_fullscreen) and not is_control, display_index, CONTROL_WINDOW_FRACTION if is_control else WINDOW_FRACTION, 'native' if is_control else window_resolution)
		# The frame and the osd are composed here at the frame's own size, and the window output
		# scales the result to the window
		canvas_surface = pygame.Surface((width, height), 0, 32)
		osd_font = pygame.font.SysFont(None, 24)
		frame_format, is_frame_swapped = resolve_frame_format(pygame, pixel_format)
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
					if event.key == pygame.K_F11 and is_control:
						pass
					elif event.key == pygame.K_F11:
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
				if is_control and ('window_display' in received_status or 'window_fullscreen' in received_status or 'window_resolution' in received_status):
					continue
				# The overlay boxes the output does not show, drawn by the parent and laid over the
				# shared frame here. They stay until the next set replaces them; an empty set clears
				if 'overlay_patches' in received_status:
					overlay_patches = resolve_overlay_patches(pygame, received_status.get('overlay_patches') or [])
					last_sequence = -1
					continue
				# The window resolution option changed while the window is open
				if 'window_resolution' in received_status:
					window_output.set_window_resolution(str(received_status.get('window_resolution')))
					last_sequence = -1
					continue
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
				# Drawn straight from the shared slot, with no copy taken first: the parent writes the
				# other slots meanwhile (see FRAME_SLOT_COUNT)
				frame_view = shared_frames.buf[frame_offset:frame_offset + frame_size]
				try:
					frame_surface = resolve_frame_surface(pygame, frame_view, width, height, frame_format, is_frame_swapped)
					# The blit is also the conversion to the 32-bit canvas the texture takes, so it is
					# needed even with nothing to draw over the frame
					canvas_surface.blit(frame_surface, (0, 0))
					for patch_x, patch_y, patch_surface in overlay_patches:
						canvas_surface.blit(patch_surface, (patch_x, patch_y))
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
				finally:
					# The surface holds the shared buffer, which cannot be closed while it is exported
					frame_surface = None
					try:
						frame_view.release()
					except BufferError:
						pass
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
