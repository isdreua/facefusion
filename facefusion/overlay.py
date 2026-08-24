import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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

	def render(self, vision_frame : VisionFrame, capture_time : float, mode : str = 'simple', is_duplicate : bool = False, timing : Optional[Dict[str, float]] = None) -> VisionFrame:
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
		box_w, box_h = 320, 140

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
		cv2.putText(overlay, fps_text, (box_x + 10, box_y + 36), font, font_scale, (0, 255, 120), 1, cv2.LINE_AA)
		cv2.putText(overlay, lat_text, (box_x + 160, box_y + 36), font, font_scale, (255, 170, 0), 1, cv2.LINE_AA)

		# Dual Graphs
		graph_y = box_y + 46
		graph_w = 140
		graph_h = 80
		graph1_x = box_x + 10
		graph2_x = box_x + 165

		self._draw_graph(overlay, graph1_x, graph_y, graph_w, graph_h, list(self.fps_history), max_val = 60.0, color = (0, 255, 120), label = 'FPS History (0-60)')
		latency_scale = max(100.0, max(self.latency_history, default = 0.0) * 1.2)
		self._draw_graph(overlay, graph2_x, graph_y, graph_w, graph_h, list(self.latency_history), max_val = latency_scale, color = (255, 170, 0), label = f'Latency (0-{latency_scale:.0f}ms)')

		return overlay

	def _render_pipeline_inspector(self, vision_frame : VisionFrame, timing : Optional[Dict[str, float]] = None) -> VisionFrame:
		frame_height, frame_width = vision_frame.shape[:2]

		box_w = 310
		box_h = 140
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

		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.36
		line_spacing = 16
		cur_y = box_y + 18

		# Header
		cv2.putText(overlay, 'PIPELINE & MODEL INSPECTOR', (box_x + 10, cur_y), font, 0.42, (0, 210, 255), 1, cv2.LINE_AA)
		cur_y += line_spacing + 2

		# Stream & Resolution Info
		threads = state_manager.get_item('execution_thread_count') or 1
		providers = state_manager.get_item('execution_providers') or [ 'cpu' ]
		prov_str = ', '.join([ p.replace('ExecutionProvider', '').lower() for p in providers ])
		cv2.putText(overlay, f'Res: {frame_width}x{frame_height} | Threads: {threads} ({prov_str})', (box_x + 10, cur_y), font, font_scale, (220, 230, 240), 1, cv2.LINE_AA)
		cur_y += line_spacing
		if timing:
			capture_ms = (timing.get('capture_read_end', 0) - timing.get('capture_read_start', 0)) * 1000
			queue_ms = (timing.get('processing_started', 0) - timing.get('scheduler_admitted', 0)) * 1000
			process_ms = (timing.get('processing_finished', 0) - timing.get('processing_started', 0)) * 1000
			cv2.putText(overlay, f'Capture {capture_ms:.1f} | Queue {queue_ms:.1f} | Process {process_ms:.1f} ms', (box_x + 10, cur_y), font, font_scale, (255, 190, 80), 1, cv2.LINE_AA)
			cur_y += line_spacing
			stages = { 'capture': capture_ms, 'queue': queue_ms, 'process': process_ms }
			bottleneck = max(stages, key = stages.get)
			p95_latency = float(numpy.percentile(self.latencies, 95)) if self.latencies else 0.0
			frame_interval_ms = timing.get('frame_interval_ms', 0.0)
			budget_status = 'OVER BUDGET' if frame_interval_ms and p95_latency > frame_interval_ms else 'within budget'
			cv2.putText(overlay, f'Bottleneck: {bottleneck} | {budget_status}', (box_x + 10, cur_y), font, font_scale, (80, 120, 255) if budget_status == 'OVER BUDGET' else (120, 220, 140), 1, cv2.LINE_AA)
			cur_y += line_spacing
			from facefusion.streamer import get_content_analysis_metrics
			analysis_metrics = get_content_analysis_metrics()
			if analysis_metrics.get('runs'):
				cv2.putText(overlay, f'Safety sample {analysis_metrics.get("last_ms", 0):.1f} ms', (box_x + 10, cur_y), font, font_scale, (180, 160, 255), 1, cv2.LINE_AA)
				cur_y += line_spacing

		# Face Selector & Frame Skipping
		selector_mode = state_manager.get_item('face_selector_mode') or 'one'
		skipping_mode = state_manager.get_item('webcam_frame_skipping') or 'adaptive'
		cv2.putText(overlay, f'Selector: {selector_mode} | Skipping: {skipping_mode}', (box_x + 10, cur_y), font, font_scale, (220, 230, 240), 1, cv2.LINE_AA)
		cur_y += line_spacing

		# Active Processors and Models
		processors : List[str] = state_manager.get_item('processors') or []
		if not processors:
			cv2.putText(overlay, 'Active Processors: None (Pass-through)', (box_x + 10, cur_y), font, font_scale, (140, 160, 180), 1, cv2.LINE_AA)
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
				cv2.putText(overlay, text, (box_x + 10, cur_y), font, font_scale, (0, 255, 200), 1, cv2.LINE_AA)
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
