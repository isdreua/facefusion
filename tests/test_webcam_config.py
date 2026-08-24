import json

from facefusion.webcam_config import DEFAULT_WEBCAM_CONFIG, apply_webcam_state, load_webcam_config


def test_load_webcam_config(tmp_path):
	config_path = tmp_path / 'webcam.json'
	config_path.write_text(json.dumps(
	{
		'auto_start': True,
		'device_id': 2,
		'mode': 'inline',
		'resolution': '1280x720',
		'fps': 25,
		'fps_options': [ 1, 30 ],
		'frame_skipping': 'adaptive',
		'frame_skipping_options': [ 'adaptive', 'disabled' ],
		'performance_overlay': 'simple',
		'webcam_inline_preview_resolution': '960x540',
		'webcam_execution_thread_count': 1
	}), encoding = 'utf-8')

	webcam_config = load_webcam_config(str(config_path))

	assert webcam_config['auto_start'] is True
	assert webcam_config['device_id'] == 2
	assert webcam_config['resolution'] == '1280x720'
	assert webcam_config['fps'] == 25
	assert webcam_config['frame_skipping'] == 'adaptive'
	assert webcam_config['performance_overlay'] == 'simple'
	assert webcam_config['webcam_inline_preview_resolution'] == '960x540'
	assert webcam_config['webcam_execution_thread_count'] == 1
	assert 'fps_options' not in webcam_config
	assert 'frame_skipping_options' not in webcam_config


def test_load_webcam_config_uses_safe_defaults_for_invalid_values(tmp_path):
	config_path = tmp_path / 'invalid-webcam.json'
	config_path.write_text('{"auto_start": "yes", "fps": 120, "mode": "invalid"}', encoding = 'utf-8')

	webcam_config = load_webcam_config(str(config_path))

	assert webcam_config == DEFAULT_WEBCAM_CONFIG


def test_apply_webcam_state_uses_settings_and_ignores_documentation(tmp_path):
	config_path = tmp_path / 'webcam-state.json'
	config_path.write_text(json.dumps(
	{
		'settings':
		{
			'processors': [ 'face_swapper', 'face_enhancer' ],
			'face_swapper_model': 'inswapper_128_fp16',
			'face_enhancer_model': 'gfpgan_1.4'
		},
		'face_enhancer_model_options': [ 'codeformer', 'gfpgan_1.4' ]
	}), encoding = 'utf-8')
	state = {}

	apply_webcam_state(str(config_path), state.__setitem__)

	assert state ==\
	{
		'processors': [ 'face_swapper', 'face_enhancer' ],
		'face_swapper_model': 'inswapper_128_fp16',
		'face_enhancer_model': 'gfpgan_1.4'
	}
