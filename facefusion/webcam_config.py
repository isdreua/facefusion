from functools import lru_cache
from typing import Any, Dict

from facefusion.json import read_json


DEFAULT_WEBCAM_CONFIG : Dict[str, Any] =\
{
	'auto_start': False,
	'device_id': 0,
	'mode': 'inline',
	'headless_mode': 'v4l2',
	'resolution': '320x240',
	'fps': 30,
	'frame_skipping': 'adaptive',
	'performance_overlay': 'none',
	'inline_preview_resolution': '640x480'
}
WEBCAM_MODES = [ 'inline', 'udp', 'v4l2' ]
WEBCAM_RESOLUTIONS = [ '320x240', '640x480', '800x600', '1024x768', '1280x720', '1280x960', '1920x1080' ]
WEBCAM_FRAME_SKIPPING_MODES = [ 'adaptive', 'disabled', '1-in-2', '1-in-3' ]
WEBCAM_PERFORMANCE_OVERLAYS = [ 'none', 'simple', 'advanced' ]
WEBCAM_INLINE_PREVIEW_RESOLUTIONS = [ 'native', '320x240', '640x480', '960x540' ]
WEBCAM_STATE_KEYS =\
[
	'source_paths', 'processors', 'execution_device_ids', 'execution_providers', 'execution_thread_count', 'video_memory_strategy',
	'face_detector_model', 'face_detector_size', 'face_detector_angles', 'face_detector_score', 'face_landmarker_model', 'face_landmarker_score',
	'face_selector_mode', 'face_selector_order', 'face_selector_age_start', 'face_selector_age_end', 'face_selector_gender', 'face_selector_race',
	'face_mask_types', 'face_mask_areas', 'face_mask_regions', 'face_mask_blur', 'face_mask_padding',
	'age_modifier_model', 'age_modifier_direction', 'background_remover_model', 'background_remover_fill_color', 'background_remover_despill_color',
	'deep_swapper_model', 'deep_swapper_morph', 'expression_restorer_model', 'expression_restorer_factor', 'expression_restorer_areas',
	'face_debugger_items', 'face_editor_model', 'face_editor_eyebrow_direction', 'face_editor_eye_gaze_horizontal', 'face_editor_eye_gaze_vertical',
	'face_editor_eye_open_ratio', 'face_editor_head_pitch', 'face_editor_head_roll', 'face_editor_head_yaw', 'face_editor_lip_open_ratio',
	'face_editor_mouth_grim', 'face_editor_mouth_position_horizontal', 'face_editor_mouth_position_vertical', 'face_editor_mouth_pout',
	'face_editor_mouth_purse', 'face_editor_mouth_smile', 'face_enhancer_model', 'face_enhancer_blend', 'face_enhancer_weight',
	'face_swapper_model', 'face_swapper_pixel_boost', 'face_swapper_weight', 'frame_colorizer_model', 'frame_colorizer_blend',
	'frame_colorizer_size', 'frame_enhancer_model', 'frame_enhancer_blend', 'lip_syncer_model', 'lip_syncer_weight'
]


@lru_cache()
def load_webcam_config(config_path : str) -> Dict[str, Any]:
	webcam_config = DEFAULT_WEBCAM_CONFIG.copy()
	content = read_json(config_path) if config_path else None

	if not isinstance(content, dict):
		return webcam_config
	if isinstance(content.get('auto_start'), bool):
		webcam_config['auto_start'] = content.get('auto_start')
	if isinstance(content.get('device_id'), int) and content.get('device_id') >= 0:
		webcam_config['device_id'] = content.get('device_id')
	if content.get('mode') in WEBCAM_MODES:
		webcam_config['mode'] = content.get('mode')
	if content.get('headless_mode') in [ 'udp', 'v4l2' ]:
		webcam_config['headless_mode'] = content.get('headless_mode')
	if content.get('resolution') in WEBCAM_RESOLUTIONS:
		webcam_config['resolution'] = content.get('resolution')
	if isinstance(content.get('fps'), int) and 1 <= content.get('fps') <= 30:
		webcam_config['fps'] = content.get('fps')
	if content.get('frame_skipping') in WEBCAM_FRAME_SKIPPING_MODES:
		webcam_config['frame_skipping'] = content.get('frame_skipping')
	if content.get('performance_overlay') in WEBCAM_PERFORMANCE_OVERLAYS:
		webcam_config['performance_overlay'] = content.get('performance_overlay')
	if content.get('inline_preview_resolution') in WEBCAM_INLINE_PREVIEW_RESOLUTIONS:
		webcam_config['inline_preview_resolution'] = content.get('inline_preview_resolution')

	return webcam_config


def apply_webcam_state(config_path : str, apply_state_item) -> None:
	content = read_json(config_path) if config_path else None
	settings = content.get('settings') if isinstance(content, dict) else None

	if isinstance(settings, dict):
		for key in WEBCAM_STATE_KEYS:
			if key in settings:
				apply_state_item(key, settings.get(key))
