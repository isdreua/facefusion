import numpy

from facefusion import streamer
from facefusion.processors.modules.face_swapper import core as face_swapper


def test_neutral_weight_skips_target_embedding_feature(monkeypatch):
	monkeypatch.setattr(face_swapper, 'get_model_options', lambda: { 'type': 'hyperswap' })
	monkeypatch.setattr(face_swapper.state_manager, 'get_item', lambda key: 0.5)

	assert face_swapper.get_stream_face_analysis_features() == []


def test_non_neutral_weight_requests_target_embedding_feature(monkeypatch):
	monkeypatch.setattr(face_swapper, 'get_model_options', lambda: { 'type': 'hyperswap' })
	monkeypatch.setattr(face_swapper.state_manager, 'get_item', lambda key: 0.75)

	assert face_swapper.get_stream_face_analysis_features() == [ 'embedding' ]


def test_neutral_weight_does_not_normalize_zero_target_embedding(monkeypatch):
	monkeypatch.setattr(face_swapper, 'get_model_options', lambda: { 'type': 'hyperswap' })
	monkeypatch.setattr(face_swapper.state_manager, 'get_item', lambda key: 0.5)
	source_embedding = numpy.arange(4, dtype = numpy.float32)
	target_embedding = numpy.zeros(4, dtype = numpy.float32)

	balanced_embedding = face_swapper.balance_source_embedding(source_embedding, target_embedding)

	numpy.testing.assert_array_equal(balanced_embedding, source_embedding.reshape(1, -1))
	assert numpy.isfinite(balanced_embedding).all()


def test_reference_mode_preserves_embedding_requirement(monkeypatch):
	monkeypatch.setattr(streamer.state_manager, 'get_item', lambda key: 'reference' if key == 'face_selector_mode' else None)

	assert 'embedding' in streamer.collect_stream_face_analysis_features([])
