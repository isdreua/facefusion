import numpy
import pytest

from facefusion import state_manager
from facefusion.overlay import PerformanceOverlay


@pytest.mark.parametrize('mode', [ 'simple', 'advanced' ])
def test_performance_overlay_renders_in_place(monkeypatch, mode):
	monkeypatch.setattr(state_manager, 'get_item', lambda key: {
		'execution_thread_count': 1,
		'execution_providers': [ 'cpu' ],
		'face_selector_mode': 'one',
		'webcam_frame_skipping': 'disabled',
		'processors': []
	}.get(key))
	vision_frame = numpy.zeros((200, 800, 3), dtype = numpy.uint8)
	overlay = PerformanceOverlay()

	rendered_frame = overlay.render(vision_frame, 0.0, mode)

	assert rendered_frame is vision_frame
	assert numpy.any(vision_frame)
