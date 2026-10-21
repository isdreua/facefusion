import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy

from facefusion import state_manager
from facefusion.types import VisionFrame


class PerformanceOverlay:
	def __init__(self, window_size : int = 30, history_size : int = 60) -> None:
		self.window_size = window_size
		self.history_size = history_size
		self.timestamps : Deque[float] = deque(maxlen = window_size)
		self.processed_timestamps : Deque[float] = deque(maxlen = window_size)
		self.latencies : Deque[float] = deque(maxlen = window_size)
		self.fps_history : Deque[float] = deque(maxlen = history_size)
		self.latency_history : Deque[float] = deque(maxlen = history_size)
		self.last_processed_time = time.perf_counter()
		self.last_instant_fps = 0.0
		self.last_latency = 0.0

	# Frames re-delivered by frame skipping carry an already processed frame, so counting them
	# would report the delivery rate as if it were the rate at which frames actually get processed
	def update(self, capture_time : float, is_duplicate : bool = False) -> Tuple[float, float, float, float, float]:
		current_time = time.perf_counter()
		self.timestamps.append(current_time)

		if is_duplicate:
			instant_fps = self.last_instant_fps
			latency_ms = self.last_latency
		else:
			processed_delta = current_time - self.last_processed_time
			self.last_processed_time = current_time
			instant_fps = 1.0 / processed_delta if processed_delta > 0 else 0.0
			latency_ms = (current_time - capture_time) * 1000.0 if capture_time > 0 else 0.0
			self.last_instant_fps = instant_fps
			self.last_latency = latency_ms
			self.processed_timestamps.append(current_time)
			self.latencies.append(latency_ms)
			self.fps_history.append(instant_fps)
			self.latency_history.append(latency_ms)

		average_fps = self.calculate_rate(self.processed_timestamps, current_time, instant_fps)
		output_fps = self.calculate_rate(self.timestamps, current_time, instant_fps)
		average_latency = sum(self.latencies) / len(self.latencies) if self.latencies else latency_ms

		return instant_fps, average_fps, latency_ms, average_latency, output_fps

	# Measures against the current time so a stalled pipeline decays towards zero instead of freezing
	def calculate_rate(self, timestamps : Deque[float], current_time : float, fallback_rate : float) -> float:
		if len(timestamps) > 1:
			total_elapsed = current_time - timestamps[0]
			if total_elapsed > 0:
				return (len(timestamps) - 1) / total_elapsed
		return fallback_rate

	def render(self, vision_frame : VisionFrame, capture_time : float, mode : str = 'simple', is_duplicate : bool = False, timing : Optional[Dict[str, Any]] = None) -> VisionFrame:
		# The caller must pass an exclusively owned frame because overlays are rendered in place.

		# 1. Render Top-Left Performance HUD
		vision_frame = self._render_performance_hud(vision_frame, capture_time, is_duplicate)

		# 2. Render Top-Right Pipeline Inspector HUD if advanced mode is enabled
		if mode == 'advanced':
			vision_frame = self._render_pipeline_inspector(vision_frame, timing)

		return vision_frame

	def _render_performance_hud(self, vision_frame : VisionFrame, capture_time : float, is_duplicate : bool = False) -> VisionFrame:
		instant_fps, avg_fps, instant_lat, avg_lat, output_fps = self.update(capture_time, is_duplicate)
		frame_height, frame_width = vision_frame.shape[:2]

		box_x, box_y = 12, 12
		box_w, box_h = 320, 174

		if frame_width < box_w + 24 or frame_height < box_h + 24:
			return vision_frame

		overlay = vision_frame
		sub_img = overlay[box_y:box_y + box_h, box_x:box_x + box_w]
		dark_rect = numpy.zeros(sub_img.shape, dtype = numpy.uint8)
		dark_rect[:] = (15, 20, 28)
		res = cv2.addWeighted(sub_img, 0.35, dark_rect, 0.65, 0)
		overlay[box_y:box_y + box_h, box_x:box_x + box_w] = res

		cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 80, 110), 1, cv2.LINE_AA)

		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.42

		# Header
		cv2.putText(overlay, 'PERFORMANCE DEBUG', (box_x + 10, box_y + 18), font, 0.45, (0, 210, 255), 1, cv2.LINE_AA)
		cv2.putText(overlay, f'OUT {output_fps:4.1f}', (box_x + 240, box_y + 18), font, 0.4, (140, 160, 180), 1, cv2.LINE_AA)

		# Metrics Readout, reporting the rate at which frames get processed rather than delivered
		fps_text = f'FPS: {instant_fps:5.1f}  (Avg: {avg_fps:4.1f})'
		p95_latency = float(numpy.percentile(self.latencies, 95)) if self.latencies else instant_lat
		lat_text = f'PIPE: {instant_lat:5.1f}ms (P95: {p95_latency:4.1f}ms)'
		# One per line: side by side they were placed at a fixed offset that assumed a shorter
		# reading, so a three digit fps or latency ran the first string through the second
		cv2.putText(overlay, fps_text, (box_x + 10, box_y + 36), font, font_scale, (0, 255, 120), 1, cv2.LINE_AA)
		cv2.putText(overlay, lat_text, (box_x + 10, box_y + 52), font, font_scale, (255, 170, 0), 1, cv2.LINE_AA)

		# Dual Graphs
		graph_y = box_y + 62
		graph_w = 140
		graph_h = 80
		graph1_x = box_x + 10
		graph2_x = box_x + 165

		self._draw_graph(overlay, graph1_x, graph_y, graph_w, graph_h, list(self.fps_history), max_val = 60.0, color = (0, 255, 120), label = 'FPS History (0-60)')
		latency_scale = max(100.0, max(self.latency_history, default = 0.0) * 1.2)
		self._draw_graph(overlay, graph2_x, graph_y, graph_w, graph_h, list(self.latency_history), max_val = latency_scale, color = (255, 170, 0), label = f'Latency (0-{latency_scale:.0f}ms)')

		# Which identity is being swapped in, so a face switched mid-stream is confirmable without
		# looking away from the stream
		cv2.putText(overlay, self._resolve_source_face_text(), (box_x + 10, box_y + box_h - 9), font, font_scale, (120, 220, 255), 1, cv2.LINE_AA)

		return overlay

	def _resolve_source_face_text(self) -> str:
		source_face_name = state_manager.get_item('source_face_name') or '-'

		if len(source_face_name) > 36:
			source_face_name = source_face_name[:33] + '...'
		return 'FACE: ' + source_face_name

	def _truncate_text(self, text : str, font : int, font_scale : float, max_width : int) -> str:
		(text_width, _), _ = cv2.getTextSize(text, font, font_scale, 1)

		if text_width <= max_width:
			return text
		while text and text_width > max_width:
			text = text[:-1]
			(text_width, _), _ = cv2.getTextSize(text + '...', font, font_scale, 1)

		return text + '...'


	# The slowest processor arrives as a full module path, which is wider than the panel; the module
	# name alone is what identifies it
	def _resolve_processor_label(self, processor_name : str) -> str:
		processor_parts = processor_name.split('.')

		if len(processor_parts) > 1 and processor_parts[-1] == 'core':
			return processor_parts[-2]
		return processor_parts[-1]


	# Collected before anything is drawn so the panel can be sized to what it will actually hold. A
	# fixed height pushed the last line below the border as soon as an optional metric appeared
	def _collect_inspector_lines(self, frame_width : int, frame_height : int, timing : Optional[Dict[str, Any]] = None) -> List[Tuple[str, Tuple[int, int, int]]]:
		inspector_lines : List[Tuple[str, Tuple[int, int, int]]] = []

		threads = state_manager.get_item('execution_thread_count') or 1
		providers = state_manager.get_item('execution_providers') or [ 'cpu' ]
		prov_str = ', '.join([ p.replace('ExecutionProvider', '').lower() for p in providers ])
		inspector_lines.append((f'Res: {frame_width}x{frame_height} | Threads: {threads} ({prov_str})', (220, 230, 240)))

		if timing:
			capture_ms = (timing.get('capture_read_end', 0) - timing.get('capture_read_start', 0)) * 1000
			queue_ms = (timing.get('processing_started', 0) - timing.get('scheduler_admitted', 0)) * 1000
			process_ms = (timing.get('processing_finished', 0) - timing.get('processing_started', 0)) * 1000
			inspector_lines.append((f'Capture {capture_ms:.1f} | Queue {queue_ms:.1f} | Process {process_ms:.1f} ms', (255, 190, 80)))
			stages = { 'capture': capture_ms, 'queue': queue_ms, 'process': process_ms }
			bottleneck = max(stages, key = stages.get)
			p95_latency = float(numpy.percentile(self.latencies, 95)) if self.latencies else 0.0
			frame_interval_ms = timing.get('frame_interval_ms', 0.0)
			budget_status = 'OVER BUDGET' if frame_interval_ms and p95_latency > frame_interval_ms else 'within budget'
			inspector_lines.append((f'Bottleneck: {bottleneck} | {budget_status}', (80, 120, 255) if budget_status == 'OVER BUDGET' else (120, 220, 140)))
			# What the models cost against what the python around them costs; a large remainder means
			# the frame is spent on preparation, pasting and copies rather than on the device
			model_ms = timing.get('model_ms')
			if model_ms is not None:
				inspector_lines.append((f'Models {model_ms:.1f} ms x{timing.get("model_runs", 0)} | Overhead {max(0.0, process_ms - model_ms):.1f} ms', (120, 220, 255)))
			processor_times = timing.get('processor_times') or {}
			if processor_times:
				slowest_processor = max(processor_times, key = processor_times.get)
				inspector_lines.append((f'{self._resolve_processor_label(slowest_processor)}: {processor_times[slowest_processor]:.1f} ms', (80, 210, 255)))
			from facefusion.streamer import get_content_analysis_metrics
			analysis_metrics = get_content_analysis_metrics()
			if analysis_metrics.get('runs'):
				inspector_lines.append((f'Safety sample {analysis_metrics.get("last_ms", 0):.1f} ms', (180, 160, 255)))

		# Face Selector & Frame Skipping
		selector_mode = state_manager.get_item('face_selector_mode') or 'one'
		skipping_mode = state_manager.get_item('webcam_frame_skipping') or 'adaptive'
		inspector_lines.append((f'Selector: {selector_mode} | Skipping: {skipping_mode}', (220, 230, 240)))

		# Active Processors and Models
		processors : List[str] = state_manager.get_item('processors') or []
		if not processors:
			inspector_lines.append(('Active Processors: None (Pass-through)', (140, 160, 180)))
		else:
			for proc in processors[:3]:
				model_key = f'{proc}_model'
				model_val = state_manager.get_item(model_key)
				proc_name = proc.replace('_', ' ').title()
				if model_val:
					# Infer model resolution if in name (e.g. 128, 256, 512)
					res_tag = ''
					if '128' in str(model_val):
						res_tag = ' [128px]'
					elif '256' in str(model_val):
						res_tag = ' [256px]'
					elif '512' in str(model_val):
						res_tag = ' [512px]'
					text = f'{proc_name}: {model_val}{res_tag}'
				else:
					text = f'{proc_name}: Active'
				inspector_lines.append((text, (0, 255, 200)))

		return inspector_lines


	def _render_pipeline_inspector(self, vision_frame : VisionFrame, timing : Optional[Dict[str, Any]] = None) -> VisionFrame:
		frame_height, frame_width = vision_frame.shape[:2]

		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.36
		line_spacing = 16
		inspector_lines = self._collect_inspector_lines(frame_width, frame_height, timing)
		box_w = 310
		box_h = 30 + len(inspector_lines) * line_spacing
		box_x = frame_width - box_w - 12
		box_y = 12

		if box_x < 340 or frame_height < box_h + 24:
			return vision_frame

		overlay = vision_frame
		sub_img = overlay[box_y:box_y + box_h, box_x:box_x + box_w]
		dark_rect = numpy.zeros(sub_img.shape, dtype = numpy.uint8)
		dark_rect[:] = (15, 20, 28)
		res = cv2.addWeighted(sub_img, 0.35, dark_rect, 0.65, 0)
		overlay[box_y:box_y + box_h, box_x:box_x + box_w] = res

		cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 80, 110), 1, cv2.LINE_AA)

		# Header
		cv2.putText(overlay, 'PIPELINE & MODEL INSPECTOR', (box_x + 10, box_y + 18), font, 0.42, (0, 210, 255), 1, cv2.LINE_AA)
		cur_y = box_y + 36

		# putText does not clip to the rectangle, so an over long line used to run off the frame edge
		for text, color in inspector_lines:
			cv2.putText(overlay, self._truncate_text(text, font, font_scale, box_w - 20), (box_x + 10, cur_y), font, font_scale, color, 1, cv2.LINE_AA)
			cur_y += line_spacing

		return overlay


	def _draw_graph(self, frame : VisionFrame, x : int, y : int, w : int, h : int, data : list, max_val : float, color : Tuple[int, int, int], label : str) -> None:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 55, 75), 1, cv2.LINE_AA)
		mid_y = y + h // 2
		cv2.line(frame, (x + 1, mid_y), (x + w - 1, mid_y), (30, 42, 58), 1, cv2.LINE_AA)

		cv2.putText(frame, label, (x + 4, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 180, 200), 1, cv2.LINE_AA)

		if len(data) < 2:
			return

		points = []
		step = (w - 4) / max(1, self.history_size - 1)
		start_offset = (self.history_size - len(data)) * step

		for i, val in enumerate(data):
			clamped_val = min(max_val, max(0.0, val))
			px = int(x + 2 + start_offset + i * step)
			py = int(y + h - 2 - (clamped_val / max_val) * (h - 16))
			points.append((px, py))

		for i in range(len(points) - 1):
			cv2.line(frame, points[i], points[i + 1], color, 1, cv2.LINE_AA)


PERFORMANCE_OVERLAY = PerformanceOverlay()
