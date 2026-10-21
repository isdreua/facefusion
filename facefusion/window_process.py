from typing import Any, Dict, Optional

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
# box never changes: the window is created at a fixed logical size and SCALED fullscreen keeps it.
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


# The child reports raw intents and resolves nothing: the parent owns the index to face name mapping
# so that the window and the browser gallery cannot disagree about what is selected
def run_window(shared_name : str, width : int, height : int, fps : float, title : str, is_fullscreen : bool, sequence : Any, frame_event : Any, stop_event : Any, intent_queue : Any, status_queue : Any) -> None:
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
		# frame first; SCALED keeps the logical size so F11 still switches back
		display_flags = pygame.RESIZABLE | pygame.SCALED
		if is_fullscreen:
			display_flags = display_flags | pygame.FULLSCREEN
		display_surface = pygame.display.set_mode((width, height), display_flags)
		osd_font = pygame.font.SysFont(None, 24)
		clock = pygame.time.Clock()
		last_sequence = -1

		while not is_closing and not stop_event.is_set():
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					is_closing = True
				elif event.type == pygame.KEYDOWN:
					face_index = resolve_face_index(pygame, event.key)
					overlay_mode = resolve_overlay_mode(pygame, event.key)
					if event.key == pygame.K_F11:
						pygame.display.toggle_fullscreen()
					elif event.key == pygame.K_q and event.mod & pygame.KMOD_CTRL:
						is_closing = True
					elif event.key == pygame.K_ESCAPE:
						# Escape leaves fullscreen first, so it cannot close the window by surprise
						if display_surface.get_flags() & pygame.FULLSCREEN:
							pygame.display.toggle_fullscreen()
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
					status = status_queue.get_nowait()
					osd_expires_at = time.monotonic() + OSD_DURATION
				except Exception:
					break

			current_sequence = int(sequence.value)
			if current_sequence != last_sequence and current_sequence > 0:
				last_sequence = current_sequence
				frame_slot = (current_sequence - 1) % FRAME_SLOT_COUNT
				frame_offset = frame_slot * frame_size
				# Copied out of the shared slot before it is drawn, so a frame the parent writes
				# meanwhile cannot tear the one on screen
				frame_bytes = bytes(shared_frames.buf[frame_offset:frame_offset + frame_size])
				frame_surface = pygame.image.frombuffer(frame_bytes, (width, height), 'RGB')
				display_surface.blit(frame_surface, (0, 0))
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
			clock.tick(max(1.0, fps) * 2)
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
