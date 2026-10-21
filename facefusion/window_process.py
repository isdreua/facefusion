from typing import Any, Dict, Optional

# This module is the target of a spawned process, so spawn re-imports it in the child. Every heavy
# import therefore lives inside run_window: the parent imports this module only to name the target,
# and must not drag pygame in, while the child must not drag onnxruntime in
FRAME_SLOT_COUNT = 2
OSD_DURATION = 1.5
OSD_MARGIN = 12


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


def resolve_osd_text(status : Dict[str, Any]) -> str:
	if status.get('is_bypassed'):
		return 'RAW CAMERA'
	source_face_name = status.get('source_face_name')
	if source_face_name:
		return 'FACE: ' + str(source_face_name)
	return ''


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
	status : Dict[str, Any] = { 'source_face_names': [], 'source_face_name': None, 'is_bypassed': False }
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
