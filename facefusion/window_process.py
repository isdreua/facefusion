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
			preview_surface = pygame.image.load(source_face_path).convert_alpha()
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


# Synced to the display so a frame lands on a refresh instead of tearing across one; SCALED is what
# makes vsync available without opengl. A driver that refuses it still gets a window.
#
# The display argument alone did not pin anything: without SDL_VIDEO_CENTERED pygame asks for an
# undefined position on that display, which X11 window managers and Windows treat as "place it
# anywhere", and fullscreen then follows wherever the window landed. Centred, SDL works out real
# coordinates inside that display's bounds, and fullscreen takes the display the window is on
def set_display_mode(pygame : Any, width : int, height : int, display_flags : int, display_index : Optional[int]) -> Any:
	import os

	display_options : Dict[str, Any] = {}

	if display_index is not None:
		os.environ['SDL_VIDEO_CENTERED'] = '1'
		display_options['display'] = display_index
	try:
		return pygame.display.set_mode((width, height), display_flags, vsync = 1, **display_options)
	except pygame.error:
		return pygame.display.set_mode((width, height), display_flags, **display_options)


def resolve_display_flags(pygame : Any, is_fullscreen : bool) -> int:
	display_flags = pygame.RESIZABLE | pygame.SCALED

	if is_fullscreen:
		display_flags = display_flags | pygame.FULLSCREEN
	return display_flags


# Windowed, the surface is the frame size and SCALED scales the window. Fullscreen it is the size of
# the display itself: with the frame as the logical size, SDL's renderer could draw a frame larger
# than the display (a bigger output resolution, or a desktop shrunk by Windows display scaling) at
# 1:1 and crop it. So fullscreen is opened once to learn the window's real size, then again at that
# size, and the frame is fitted into it by draw_frame. Without a display index pygame keeps the
# display the window is on, so an unpinned window stays where it is
def open_display(pygame : Any, width : int, height : int, is_fullscreen : bool, display_index : Optional[int]) -> Any:
	display_flags = resolve_display_flags(pygame, is_fullscreen)
	display_surface = set_display_mode(pygame, width, height, display_flags, display_index)

	if is_fullscreen:
		window_size = resolve_window_size(pygame)
		if window_size and window_size != display_surface.get_size():
			display_surface = set_display_mode(pygame, window_size[0], window_size[1], display_flags, display_index)
	return display_surface


def resolve_window_size(pygame : Any) -> Optional[Tuple[int, int]]:
	try:
		window_width, window_height = pygame.display.get_window_size()
	except Exception:
		return None
	if window_width > 0 and window_height > 0:
		return int(window_width), int(window_height)
	return None


def is_display_fullscreen(pygame : Any, display_surface : Any) -> bool:
	return bool(display_surface.get_flags() & pygame.FULLSCREEN)


# Fullscreen is left first and entered again from a window centred on the target display: a
# fullscreen window keeps its display through set_mode, so going there directly would not move it
def move_to_display(pygame : Any, display_surface : Any, width : int, height : int, display_index : int) -> Any:
	is_fullscreen = is_display_fullscreen(pygame, display_surface)

	try:
		display_surface = open_display(pygame, width, height, False, display_index)
		if is_fullscreen:
			display_surface = open_display(pygame, width, height, True, display_index)
	except Exception:
		pass
	return pygame.display.get_surface() or display_surface


# Done with set_mode rather than toggle_fullscreen, which keeps the frame as the logical size and so
# brings back the crop. A pinned window is moved back to its display first, since it may have been
# dragged away. The plain toggle is only the fallback when set_mode is refused
def switch_fullscreen(pygame : Any, display_surface : Any, width : int, height : int, display_index : Optional[int]) -> Any:
	is_fullscreen = is_display_fullscreen(pygame, display_surface)

	try:
		if display_index is not None:
			display_surface = open_display(pygame, width, height, False, display_index)
		return open_display(pygame, width, height, not is_fullscreen, display_index)
	except Exception:
		pass
	pygame.display.toggle_fullscreen()
	return pygame.display.get_surface() or display_surface


# The frame scaled to fit the surface with its aspect kept, centred on black. A surface of the
# frame's own size (always the case windowed) is a plain blit with no scaling cost
def draw_frame(pygame : Any, display_surface : Any, frame_surface : Any) -> None:
	surface_width, surface_height = display_surface.get_size()
	frame_width, frame_height = frame_surface.get_size()

	if (surface_width, surface_height) == (frame_width, frame_height):
		display_surface.blit(frame_surface, (0, 0))
		return
	frame_scale = min(surface_width / frame_width, surface_height / frame_height)
	fitted_size = (max(1, round(frame_width * frame_scale)), max(1, round(frame_height * frame_scale)))
	try:
		fitted_surface = pygame.transform.smoothscale(frame_surface, fitted_size)
	except Exception:
		fitted_surface = pygame.transform.scale(frame_surface, fitted_size)
	display_surface.fill((0, 0, 0))
	display_surface.blit(fitted_surface, ((surface_width - fitted_size[0]) // 2, (surface_height - fitted_size[1]) // 2))


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
		pygame.display.set_caption(title)
		# Asked for at set_mode rather than toggled after, so the window never shows a windowed
		# frame first
		display_index = resolve_display_index(pygame, window_display)
		display_surface = open_display(pygame, width, height, is_fullscreen, display_index)
		osd_font = pygame.font.SysFont(None, 24)
		last_sequence = -1
		drawn_surface_size = display_surface.get_size()
		drawn_fullscreen = is_display_fullscreen(pygame, display_surface)

		while not is_closing and not stop_event.is_set():
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					is_closing = True
				elif event.type == pygame.KEYDOWN:
					camera_id = resolve_camera_id(pygame, event.key) if event.mod & pygame.KMOD_CTRL else None
					face_index = resolve_face_index(pygame, event.key)
					overlay_mode = resolve_overlay_mode(pygame, event.key)
					if event.key == pygame.K_F11:
						display_surface = switch_fullscreen(pygame, display_surface, width, height, display_index)
					elif event.key == pygame.K_q and event.mod & pygame.KMOD_CTRL:
						is_closing = True
					# Before the face keys, which share the digits
					elif camera_id is not None:
						intent_queue.put(('camera_id', camera_id))
					elif event.key == pygame.K_ESCAPE:
						# Escape leaves fullscreen first, so it cannot close the window by surprise
						if is_display_fullscreen(pygame, display_surface):
							display_surface = switch_fullscreen(pygame, display_surface, width, height, display_index)
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
					pygame.display.set_caption(str(received_status.get('window_title')))
					continue
				# The display dropdown changed while the window is open. Auto drops the pin and leaves
				# the window where it is; a display that is not attached is ignored the same way
				if 'window_display' in received_status:
					display_index = resolve_display_index(pygame, int(received_status.get('window_display') or 0))
					if display_index is not None:
						display_surface = move_to_display(pygame, display_surface, width, height, display_index)
					continue
				status = received_status
				osd_expires_at = time.monotonic() + OSD_DURATION

			# A mode change (F11, Esc, a display move) leaves a new surface with nothing on it, so the
			# current frame is drawn again rather than waiting for the next one
			if display_surface.get_size() != drawn_surface_size or is_display_fullscreen(pygame, display_surface) != drawn_fullscreen:
				drawn_surface_size = display_surface.get_size()
				drawn_fullscreen = is_display_fullscreen(pygame, display_surface)
				last_sequence = -1
			current_sequence = int(sequence.value)
			if current_sequence != last_sequence and current_sequence > 0:
				last_sequence = current_sequence
				frame_slot = (current_sequence - 1) % FRAME_SLOT_COUNT
				frame_offset = frame_slot * frame_size
				# Copied out of the shared slot before it is drawn, so a frame the parent writes
				# meanwhile cannot tear the one on screen
				frame_bytes = bytes(shared_frames.buf[frame_offset:frame_offset + frame_size])
				frame_surface = pygame.image.frombuffer(frame_bytes, (width, height), 'RGB')
				draw_frame(pygame, display_surface, frame_surface)
				osd_text = resolve_osd_text(status)
				if osd_text and time.monotonic() < osd_expires_at:
					text_surface = osd_font.render(osd_text, True, (255, 255, 255), (0, 0, 0))
					display_surface.blit(text_surface, (OSD_MARGIN, OSD_MARGIN))
					# Under the text, and only while the text is up, so switching faces shows what
					# was switched to rather than leaving a thumbnail sitting on the frame
					preview_surface = resolve_osd_preview(pygame, osd_previews, status.get('source_face_path'), osd_preview_size)
					if preview_surface:
						display_surface.blit(preview_surface, (OSD_MARGIN, OSD_MARGIN + text_surface.get_height() + OSD_PREVIEW_GAP))
				pygame.display.flip()
			else:
				# Nothing new to draw, so wait on the parent instead of spinning; the sequence above
				# is the real source of truth, which makes a missed event harmless
				frame_event.wait(0.005)
				frame_event.clear()
			# No frame rate cap here any more: it slept after every flip, so a frame arriving meanwhile
			# waited up to half a frame and the generated stream reached the screen unevenly. The loop
			# only draws on a new frame, idles on the event otherwise, and vsync paces the flips
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
