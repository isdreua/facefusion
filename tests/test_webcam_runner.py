from facefusion import state_manager, webcam_runner


def test_headless_webcam_rejects_inline_mode(monkeypatch):
	monkeypatch.setattr(webcam_runner, 'load_webcam_config', lambda path:
	{
		'device_id': 0,
		'headless_mode': 'inline',
		'resolution': '640x480',
		'fps': 30
	})
	monkeypatch.setattr(state_manager, 'get_item', lambda key: 'webcam.json' if key == 'webcam_config' else None)

	assert webcam_runner.run() == 2
