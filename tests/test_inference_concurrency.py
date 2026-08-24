from contextlib import contextmanager

import numpy
import pytest

from facefusion import face_detector, thread_helper
from facefusion.processors.modules.face_enhancer import core as face_enhancer


class FakeSession:
	def __init__(self, output):
		self.output = output
		self.calls = 0

	def run(self, outputs, inputs):
		self.calls += 1
		return self.output

	def get_inputs(self):
		return []


@pytest.mark.parametrize(('forward_name', 'model_name'), [
	('forward_with_retinaface', 'retinaface'),
	('forward_with_scrfd', 'scrfd'),
	('forward_with_yolo_face', 'yolo_face'),
	('forward_with_yunet', 'yunet')
])
def test_face_detector_forwards_use_conditional_semaphore(monkeypatch, forward_name, model_name):
	session = FakeSession([ numpy.zeros(1) ])
	semaphore_entries = []

	@contextmanager
	def conditional_semaphore():
		semaphore_entries.append(model_name)
		yield

	monkeypatch.setattr(face_detector, 'get_inference_pool', lambda: { model_name: session })
	monkeypatch.setattr(face_detector, 'conditional_thread_semaphore', conditional_semaphore)

	getattr(face_detector, forward_name)(numpy.zeros((1, 3, 1, 1), dtype = numpy.float32))
	assert semaphore_entries == [ model_name ]
	assert session.calls == 1


def test_face_enhancer_forward_uses_conditional_semaphore(monkeypatch):
	session = FakeSession(numpy.zeros((1, 1, 3, 2, 2), dtype = numpy.float32))
	semaphore_entries = []

	@contextmanager
	def conditional_semaphore():
		semaphore_entries.append('face_enhancer')
		yield

	monkeypatch.setattr(face_enhancer, 'get_inference_pool', lambda: { 'face_enhancer': session })
	monkeypatch.setattr(face_enhancer, 'conditional_thread_semaphore', conditional_semaphore)

	face_enhancer.forward(numpy.zeros((1, 3, 2, 2), dtype = numpy.float32), numpy.ones(1))
	assert semaphore_entries == [ 'face_enhancer' ]
	assert session.calls == 1


@pytest.mark.parametrize(('is_windows', 'is_linux', 'provider', 'is_serialized'), [
	(True, False, 'directml', True),
	(False, True, 'migraphx', True),
	(False, True, 'rocm', True),
	(False, True, 'cuda', False),
	(False, True, 'cpu', False)
])
def test_conditional_semaphore_provider_policy(monkeypatch, is_windows, is_linux, provider, is_serialized):
	monkeypatch.setattr(thread_helper, 'is_windows', lambda: is_windows)
	monkeypatch.setattr(thread_helper, 'is_linux', lambda: is_linux)
	monkeypatch.setattr(thread_helper, 'has_execution_provider', lambda name: name == provider)

	semaphore = thread_helper.conditional_thread_semaphore()
	assert (semaphore is thread_helper.THREAD_SEMAPHORE) is is_serialized
