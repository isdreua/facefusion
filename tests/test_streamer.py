import queue
from concurrent.futures import Future
from types import ModuleType

import numpy
import pytest

from facefusion import streamer
from facefusion.face_helper import get_paste_in_place
from facefusion.streamer import CameraCaptureThread


class FakeCameraCapture:
	def __init__(self, frames):
		self.frames = iter(frames)
		self.opened = True

	def isOpened(self):
		return self.opened

	def read(self):
		try:
			return True, next(self.frames)
		except StopIteration:
			self.opened = False
			return False, None


class IntermittentFakeCameraCapture(FakeCameraCapture):
	def read(self):
		try:
			frame = next(self.frames)
		except StopIteration:
			self.opened = False
			return False, None
		if frame is None:
			return False, None
		return True, frame


def test_camera_capture_thread_drops_oldest_frame_without_blocking():
	first_frame = numpy.zeros((2, 2, 3), dtype = numpy.uint8)
	latest_frame = numpy.ones((2, 2, 3), dtype = numpy.uint8)
	capture_thread = CameraCaptureThread(FakeCameraCapture([ first_frame, latest_frame ]))

	capture_thread.run()

	_, _, queued_frame = capture_thread.frame_queue.get_nowait()
	assert queued_frame is latest_frame
	assert capture_thread.running is False
	with pytest.raises(queue.Empty):
		capture_thread.frame_queue.get_nowait()


def test_camera_capture_thread_recovers_from_transient_read_failure(monkeypatch):
	frame = numpy.ones((2, 2, 3), dtype = numpy.uint8)
	camera_capture = IntermittentFakeCameraCapture([ None, frame ])
	capture_thread = CameraCaptureThread(camera_capture)
	monkeypatch.setattr(capture_thread, 'READ_FAILURE_RETRY_DELAY', 0)

	capture_thread.run()

	_, _, queued_frame = capture_thread.frame_queue.get_nowait()
	assert queued_frame is frame


def test_content_analysis_metrics_are_recorded(monkeypatch):
	monkeypatch.setattr(streamer, 'analyse_frame', lambda frame: False)
	with streamer.CONTENT_ANALYSIS_METRICS_LOCK:
		streamer.CONTENT_ANALYSIS_METRICS.update({ 'runs': 0, 'last_ms': 0.0, 'max_ms': 0.0 })
	streamer.analyse_frame_background(numpy.zeros((2, 2, 3), dtype = numpy.uint8), streamer.threading.Event())
	metrics = streamer.get_content_analysis_metrics()
	assert metrics['runs'] == 1
	assert metrics['last_ms'] >= 0


class FakeProgress:
	def __enter__(self):
		return self

	def __exit__(self, *args):
		pass

	def update(self):
		pass


class FakeExecutor:
	def __init__(self, *args, **kwargs):
		pass

	def submit(self, function, *args):
		future = Future()
		future.set_result(function(*args))
		return future

	def shutdown(self, wait = False):
		pass


class FakeCaptureThread:
	def __init__(self, camera_capture):
		self.running = camera_capture.running
		self.frame_queue = camera_capture

	def start(self):
		pass

	def stop(self):
		self.running = False

	def join(self, timeout):
		pass

	def is_alive(self):
		return False


class FakeFrameQueue:
	def __init__(self, frame = None, error = None):
		self.frame = frame
		self.error = error
		self.running = frame is not None or error is not None
		self.reads = 0

	def get(self, timeout):
		self.reads += 1
		if self.error:
			raise self.error
		if self.reads > 1:
			raise queue.Empty
		return 1.0, self.frame


def prepare_capture_test(monkeypatch, capture, context_calls, processor_modules = None, get_state_item = None):
	processor_modules = processor_modules or []
	monkeypatch.setattr(streamer, 'read_static_images', lambda paths: [])
	monkeypatch.setattr(streamer, 'get_processors_modules', lambda processors: processor_modules)
	monkeypatch.setattr(streamer, 'create_empty_audio_frame', lambda: numpy.zeros(1))
	monkeypatch.setattr(streamer, 'tqdm', lambda **kwargs: FakeProgress())
	monkeypatch.setattr(streamer, 'ThreadPoolExecutor', FakeExecutor)
	monkeypatch.setattr(streamer, 'CameraCaptureThread', FakeCaptureThread)
	monkeypatch.setattr(streamer, 'detect_app_context', lambda: 'ui')
	monkeypatch.setattr(streamer, 'set_app_context_override', context_calls.append)
	default_state = {
		'source_paths': [],
		'source_face_groups': {},
		'source_face_uploads': {},
		'source_face_groups_version': 0,
		'execution_thread_count': 1,
		'processors': [],
		'face_swapper_model': 'model',
		'face_swapper_weight': 0.5,
		'log_level': 'info',
		'webcam_frame_skipping': 'disabled',
		'face_selector_mode': 'one',
		'face_selector_age_start': 0,
		'face_selector_age_end': 100
	}
	monkeypatch.setattr(streamer.state_manager, 'get_item', get_state_item or default_state.get)
	return streamer.multi_process_capture(capture, 30)


def test_prepare_stream_processors_includes_newly_enabled_processor(monkeypatch):
	prepared_frames = []
	processor_module = ModuleType('face_enhancer_test')
	processor_module.pre_process = lambda mode: mode == 'stream'
	processor_module.prepare_stream_inputs = lambda frames: prepared_frames.extend(frames) or { 'ready': True }
	monkeypatch.setattr(streamer, 'get_processors_modules', lambda processors: [ processor_module ])
	source_frame = numpy.zeros((2, 2, 3), dtype = numpy.uint8)

	processor_modules, processor_stream_inputs = streamer.prepare_stream_processors([ 'face_enhancer' ], [ source_frame ])

	assert processor_modules == [ processor_module ]
	assert processor_stream_inputs == { 'face_enhancer_test': { 'ready': True } }
	assert prepared_frames == [ source_frame ]


def test_prepare_stream_face_sets_prepares_every_group(monkeypatch):
	prepared_calls = []
	processor_module = ModuleType('face_swapper_test')
	processor_module.prepare_stream_inputs = lambda frames: prepared_calls.append(frames) or { 'source_embedding': frames[0] }
	monkeypatch.setattr(streamer, 'read_static_images', lambda paths: list(paths))

	stream_face_sets = streamer.prepare_stream_face_sets([ processor_module ], { 'alice': [ 'a1', 'a2' ], 'bob': [ 'b1' ] })

	assert list(stream_face_sets) == [ 'alice', 'bob' ]
	assert stream_face_sets['alice'].get('source_paths') == [ 'a1', 'a2' ]
	assert stream_face_sets['alice'].get('source_vision_frames') == [ 'a1', 'a2' ]
	assert stream_face_sets['alice'].get('processor_stream_inputs') == { 'face_swapper_test': { 'source_embedding': 'a1' } }
	assert stream_face_sets['bob'].get('processor_stream_inputs') == { 'face_swapper_test': { 'source_embedding': 'b1' } }
	assert prepared_calls == [ [ 'a1', 'a2' ], [ 'b1' ] ]


def test_prepare_stream_face_sets_warns_about_group_without_image(monkeypatch):
	warnings = []
	monkeypatch.setattr(streamer, 'read_static_images', lambda paths: [])
	monkeypatch.setattr(streamer.logger, 'warn', lambda message, module_name: warnings.append(message))

	stream_face_sets = streamer.prepare_stream_face_sets([], { 'alice': [ 'missing.jpg' ] })

	assert list(stream_face_sets) == [ 'alice' ]
	assert len(warnings) == 1


@pytest.mark.parametrize(('source_face_groups', 'source_face_uploads', 'source_paths', 'resolved'), [
	({ 'alice': [ 'a1' ] }, {}, [], { 'alice': [ 'a1' ] }),
	({ 'alice': [ 'a1' ] }, { 'bob': [ 'b1' ] }, [], { 'alice': [ 'a1' ], 'bob': [ 'b1' ] }),
	({}, { 'bob': [ 'b1' ] }, [], { 'bob': [ 'b1' ] }),
	(None, None, [ 'u1' ], { 'default': [ 'u1' ] }),
	(None, None, [], { 'default': [] })
])
def test_resolve_source_face_groups_merges_config_and_uploads(monkeypatch, source_face_groups, source_face_uploads, source_paths, resolved):
	state = { 'source_face_groups': source_face_groups, 'source_face_uploads': source_face_uploads, 'source_paths': source_paths }
	monkeypatch.setattr(streamer.state_manager, 'get_item', state.get)

	assert streamer.resolve_source_face_groups() == resolved


def test_resolve_source_face_groups_copies_lists_out_of_state(monkeypatch):
	state = { 'source_face_groups': { 'alice': [ 'a1' ] }, 'source_face_uploads': {}, 'source_paths': [] }
	monkeypatch.setattr(streamer.state_manager, 'get_item', state.get)

	streamer.resolve_source_face_groups()['alice'].append('mutated')

	assert state['source_face_groups'] == { 'alice': [ 'a1' ] }


def test_update_stream_face_sets_only_prepares_what_changed(monkeypatch):
	prepared_calls = []
	processor_module = ModuleType('face_swapper_test')
	processor_module.prepare_stream_inputs = lambda frames: prepared_calls.append(list(frames)) or { 'source_embedding': frames[0] }
	monkeypatch.setattr(streamer, 'read_static_images', lambda paths: list(paths))
	stream_face_sets = streamer.prepare_stream_face_sets([ processor_module ], { 'alice': [ 'a1' ] })
	prepared_calls.clear()

	updated_stream_face_sets = streamer.update_stream_face_sets(stream_face_sets, [ processor_module ], { 'alice': [ 'a1' ], 'bob': [ 'b1' ] })

	assert list(updated_stream_face_sets) == [ 'alice', 'bob' ]
	assert updated_stream_face_sets['alice'] is stream_face_sets['alice']
	assert prepared_calls == [ [ 'b1' ] ]


def test_update_stream_face_sets_reprepares_a_changed_group(monkeypatch):
	prepared_calls = []
	processor_module = ModuleType('face_swapper_test')
	processor_module.prepare_stream_inputs = lambda frames: prepared_calls.append(list(frames)) or { 'source_embedding': frames[0] }
	monkeypatch.setattr(streamer, 'read_static_images', lambda paths: list(paths))
	stream_face_sets = streamer.prepare_stream_face_sets([ processor_module ], { 'alice': [ 'a1' ] })
	prepared_calls.clear()

	updated_stream_face_sets = streamer.update_stream_face_sets(stream_face_sets, [ processor_module ], { 'alice': [ 'a1', 'a2' ] })

	assert prepared_calls == [ [ 'a1', 'a2' ] ]
	assert updated_stream_face_sets['alice'] is not stream_face_sets['alice']


def test_update_stream_face_sets_drops_removed_groups(monkeypatch):
	processor_module = ModuleType('face_swapper_test')
	processor_module.prepare_stream_inputs = lambda frames: { 'source_embedding': frames[0] if frames else None }
	monkeypatch.setattr(streamer, 'read_static_images', lambda paths: list(paths))
	stream_face_sets = streamer.prepare_stream_face_sets([ processor_module ], { 'alice': [ 'a1' ], 'bob': [ 'b1' ] })

	updated_stream_face_sets = streamer.update_stream_face_sets(stream_face_sets, [ processor_module ], { 'bob': [ 'b1' ] })

	assert list(updated_stream_face_sets) == [ 'bob' ]


def test_resolve_stream_face_set_selects_by_name_and_falls_back(monkeypatch):
	stream_face_sets =\
	{
		'alice': { 'source_vision_frames': [ 'a1' ], 'processor_stream_inputs': { 'face_swapper_test': { 'source_embedding': 'a1' } } },
		'bob': { 'source_vision_frames': [ 'b1' ], 'processor_stream_inputs': { 'face_swapper_test': { 'source_embedding': 'b1' } } }
	}

	assert streamer.resolve_stream_face_set(stream_face_sets, 'bob')[0] == 'bob'
	assert streamer.resolve_stream_face_set(stream_face_sets, 'bob')[1] == [ 'b1' ]
	assert streamer.resolve_stream_face_set(stream_face_sets, 'carol')[0] == 'alice'
	assert streamer.resolve_stream_face_set(stream_face_sets, None)[0] == 'alice'
	assert streamer.resolve_stream_face_set({}, 'alice') == (None, [], {})


@pytest.mark.parametrize(('pending_total', 'buffered_total', 'max_queue_size', 'has_capacity'), [
	(0, 0, 8, True),
	(7, 7, 8, True),
	(8, 7, 8, False),
	(1, 8, 8, False)
])
def test_stream_capacity_caps_ordered_frame_buffer(pending_total, buffered_total, max_queue_size, has_capacity):
	assert streamer.has_stream_capacity(pending_total, buffered_total, max_queue_size) is has_capacity


def test_capture_context_is_cleared_after_normal_completion(monkeypatch):
	context_calls = []
	generator = prepare_capture_test(monkeypatch, FakeFrameQueue(), context_calls)

	assert list(generator) == []
	assert context_calls == [ 'ui', None ]


def test_capture_context_is_cleared_when_generator_closes(monkeypatch):
	context_calls = []
	frame = numpy.zeros((2, 2, 3), dtype = numpy.uint8)
	generator = prepare_capture_test(monkeypatch, FakeFrameQueue(frame), context_calls)

	next(generator)
	generator.close()
	assert context_calls == [ 'ui', None ]


def test_capture_context_is_cleared_after_exception(monkeypatch):
	context_calls = []
	generator = prepare_capture_test(monkeypatch, FakeFrameQueue(error = RuntimeError('capture failed')), context_calls)

	with pytest.raises(RuntimeError, match = 'capture failed'):
		next(generator)
	assert context_calls == [ 'ui', None ]


def test_weight_change_refreshes_features_without_rebuilding_inputs(monkeypatch):
	context_calls = []
	prepared_weights = []
	feature_weights = []
	weight_reads = 0
	default_state = {
		'source_paths': [], 'execution_thread_count': 1, 'processors': [ 'face_swapper' ],
		'face_swapper_model': 'model', 'log_level': 'info', 'webcam_frame_skipping': 'disabled',
		'face_selector_mode': 'one', 'face_selector_age_start': 0, 'face_selector_age_end': 100
	}

	def get_state_item(key):
		nonlocal weight_reads
		if key == 'face_swapper_weight':
			weight_reads += 1
			return 0.5 if weight_reads <= 3 else 0.75
		return default_state.get(key)

	processor_module = ModuleType('face_swapper_test')
	processor_module.pre_process = lambda mode: True
	processor_module.prepare_stream_inputs = lambda frames: prepared_weights.append(get_state_item('face_swapper_weight')) or {}
	processor_module.get_stream_face_analysis_features = lambda: feature_weights.append(get_state_item('face_swapper_weight')) or []
	processor_module.process_frame = lambda inputs: (inputs['temp_vision_frame'], inputs['temp_vision_mask'])
	frame = numpy.zeros((2, 2, 3), dtype = numpy.uint8)
	generator = prepare_capture_test(monkeypatch, FakeFrameQueue(frame), context_calls, [ processor_module ], get_state_item)

	next(generator)
	generator.close()
	assert len(prepared_weights) == 1
	assert feature_weights[-1] == 0.75
	assert len(feature_weights) == 2


def test_stream_paste_mode_is_cleared_after_processor_exception(monkeypatch):
	processor_module = ModuleType('failing_processor')
	processor_module.process_frame = lambda inputs: (_ for _ in ()).throw(RuntimeError('processor failed'))
	frame = numpy.zeros((2, 2, 3), dtype = numpy.uint8)

	with pytest.raises(RuntimeError, match = 'processor failed'):
		streamer.process_stream_frame([], frame, 1.0, [ processor_module ])
	assert get_paste_in_place() is False
