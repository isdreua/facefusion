import os
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Optional

from facefusion import logger
from facefusion.json import format_json, read_json
from facefusion.common_helper import is_windows
from facefusion.filesystem import is_image, resolve_file_paths
from facefusion.normalizer import normalize_color, normalize_space


DEFAULT_WEBCAM_CONFIG : Dict[str, Any] =\
{
	'auto_start': False,
	'device_id': 0,
	'mode': 'inline',
	'headless_mode': 'v4l2',
	'resolution': '320x240',
	'fps': 30,
	'frame_skipping': 'adaptive',
	'content_analysis_interval': 1,
	'performance_overlay': 'none',
	'webcam_inline_preview_resolution': '640x480',
	'webcam_execution_thread_count': 2,
	'webcam_camera_backend': 'auto',
	'webcam_output_resolution': 'copy',
	'webcam_window_fullscreen': False,
	'webcam_source_faces': {},
	'webcam_source_faces_directory': None,
	'webcam_source_face': None
}
DEFAULT_SOURCE_FACE_NAME = 'default'
# The sentinel that selects no source face at all. sanitize_source_face_name keeps only
# alphanumerics, '-' and '_', so no face file can ever sanitize into this name and claim it
NO_SOURCE_FACE_NAME = '(none)'
WEBCAM_MODES = [ 'inline', 'udp', 'v4l2', 'window' ]
WEBCAM_RESOLUTIONS = [ '320x240', '640x480', '800x600', '1024x768', '1280x720', '1280x960', '1920x1080', '240x320', '480x640', '600x800', '768x1024', '720x1280', '960x1280', '1080x1920' ]
WEBCAM_FRAME_SKIPPING_MODES = [ 'adaptive', 'disabled', '1-in-2', '1-in-3' ]
# How many seconds apart the stream samples a frame for content analysis. The check runs either
# way; this only sets how much of the device it is allowed to take while it does
WEBCAM_CONTENT_ANALYSIS_INTERVALS = list(range(1, 31))
WEBCAM_PERFORMANCE_OVERLAYS = [ 'none', 'simple', 'advanced' ]
WEBCAM_INLINE_PREVIEW_RESOLUTIONS = [ 'native', '320x240', '640x480', '960x540', '240x320', '480x640', '540x960' ]
WEBCAM_CAMERA_BACKENDS = [ 'auto', 'msmf', 'dshow' ] if is_windows() else [ 'auto' ]
WEBCAM_OUTPUT_SCALES = [ 'x2', 'x3', 'x4' ]
WEBCAM_OUTPUT_RESOLUTIONS = [ 'copy' ] + WEBCAM_OUTPUT_SCALES + WEBCAM_RESOLUTIONS
# Headless has no browser to stream into, so inline is the one mode it cannot serve
WEBCAM_HEADLESS_MODES = [ webcam_mode for webcam_mode in WEBCAM_MODES if webcam_mode != 'inline' ]
WEBCAM_CONFIG_OPTIONS =\
{
	'auto_start': [ True, False ],
	'device_id': list(range(0, 10)),
	'mode': WEBCAM_MODES,
	'headless_mode': WEBCAM_HEADLESS_MODES,
	'resolution': WEBCAM_RESOLUTIONS,
	'fps': list(range(1, 31)),
	'frame_skipping': WEBCAM_FRAME_SKIPPING_MODES,
	'content_analysis_interval': WEBCAM_CONTENT_ANALYSIS_INTERVALS,
	'performance_overlay': WEBCAM_PERFORMANCE_OVERLAYS,
	'webcam_inline_preview_resolution': WEBCAM_INLINE_PREVIEW_RESOLUTIONS,
	'webcam_execution_thread_count': list(range(1, 9)),
	'webcam_camera_backend': WEBCAM_CAMERA_BACKENDS,
	'webcam_output_resolution': WEBCAM_OUTPUT_RESOLUTIONS,
	'webcam_window_fullscreen': [ True, False ]
}
# These settings reach code that needs a fixed length tuple, and create_box_mask is lru_cached so a
# list is not even hashable; args.py normalizes them on the cli path and this path must match
WEBCAM_STATE_NORMALIZERS =\
{
	'face_mask_padding': normalize_space,
	'background_remover_fill_color': normalize_color,
	'background_remover_despill_color': normalize_color
}
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
	# Deep rather than shallow: the defaults hold a mutable face groups dict, and a shallow copy
	# would hand every caller a reference to the module level default
	webcam_config = deepcopy(DEFAULT_WEBCAM_CONFIG)
	content = read_json(config_path) if config_path else None

	if not isinstance(content, dict):
		return webcam_config
	if isinstance(content.get('auto_start'), bool):
		webcam_config['auto_start'] = content.get('auto_start')
	if isinstance(content.get('device_id'), int) and content.get('device_id') >= 0:
		webcam_config['device_id'] = content.get('device_id')
	if content.get('mode') in WEBCAM_MODES:
		webcam_config['mode'] = content.get('mode')
	if content.get('headless_mode') in WEBCAM_HEADLESS_MODES:
		webcam_config['headless_mode'] = content.get('headless_mode')
	if content.get('resolution') in WEBCAM_RESOLUTIONS:
		webcam_config['resolution'] = content.get('resolution')
	if isinstance(content.get('fps'), int) and 1 <= content.get('fps') <= 30:
		webcam_config['fps'] = content.get('fps')
	if content.get('frame_skipping') in WEBCAM_FRAME_SKIPPING_MODES:
		webcam_config['frame_skipping'] = content.get('frame_skipping')
	if content.get('content_analysis_interval') in WEBCAM_CONTENT_ANALYSIS_INTERVALS:
		webcam_config['content_analysis_interval'] = content.get('content_analysis_interval')
	if content.get('performance_overlay') in WEBCAM_PERFORMANCE_OVERLAYS:
		webcam_config['performance_overlay'] = content.get('performance_overlay')
	if content.get('webcam_inline_preview_resolution') in WEBCAM_INLINE_PREVIEW_RESOLUTIONS:
		webcam_config['webcam_inline_preview_resolution'] = content.get('webcam_inline_preview_resolution')
	if isinstance(content.get('webcam_execution_thread_count'), int) and 1 <= content.get('webcam_execution_thread_count') <= 8:
		webcam_config['webcam_execution_thread_count'] = content.get('webcam_execution_thread_count')
	if content.get('webcam_camera_backend') in WEBCAM_CAMERA_BACKENDS:
		webcam_config['webcam_camera_backend'] = content.get('webcam_camera_backend')
	if content.get('webcam_output_resolution') in WEBCAM_OUTPUT_RESOLUTIONS:
		webcam_config['webcam_output_resolution'] = content.get('webcam_output_resolution')
	if isinstance(content.get('webcam_window_fullscreen'), bool):
		webcam_config['webcam_window_fullscreen'] = content.get('webcam_window_fullscreen')
	if isinstance(content.get('webcam_source_faces_directory'), str):
		webcam_config['webcam_source_faces_directory'] = content.get('webcam_source_faces_directory')
	source_faces = resolve_source_face_directory(webcam_config.get('webcam_source_faces_directory'))
	if isinstance(content.get('webcam_source_faces'), dict):
		source_faces.update(normalize_source_faces(content.get('webcam_source_faces')))
	webcam_config['webcam_source_faces'] = source_faces
	if content.get('webcam_source_face') in webcam_config.get('webcam_source_faces'):
		webcam_config['webcam_source_face'] = content.get('webcam_source_face')

	return webcam_config


# A face group holds every photo of one identity, so a group with no usable path is dropped rather
# than kept as a selectable entry that would swap nothing
def normalize_source_faces(content : Dict[str, Any]) -> Dict[str, List[str]]:
	source_faces = {}

	for face_name, source_paths in content.items():
		if isinstance(face_name, str) and face_name and isinstance(source_paths, list):
			source_paths = [ source_path for source_path in source_paths if isinstance(source_path, str) and source_path ]
			if source_paths:
				source_faces[face_name] = source_paths

	return source_faces


def apply_webcam_state(config_path : str, apply_state_item) -> None:
	content = read_json(config_path) if config_path else None
	settings = content.get('settings') if isinstance(content, dict) else None

	if isinstance(settings, dict):
		for key in WEBCAM_STATE_KEYS:
			if key in settings:
				state_value = settings.get(key)
				if key in WEBCAM_STATE_NORMALIZERS:
					state_value = WEBCAM_STATE_NORMALIZERS.get(key)(state_value) if isinstance(state_value, list) else None
					if state_value is None:
						logger.warn('webcam setting ' + key + ' is malformed and was ignored', __name__)
						continue
				apply_state_item(key, state_value)


def create_webcam_config(webcam_options : Dict[str, Any], settings : Dict[str, Any]) -> Dict[str, Any]:
	content : Dict[str, Any] = { 'settings': settings }

	for config_key in DEFAULT_WEBCAM_CONFIG:
		content[config_key] = webcam_options.get(config_key, DEFAULT_WEBCAM_CONFIG.get(config_key))
		if config_key in WEBCAM_CONFIG_OPTIONS:
			content[config_key + '_options'] = WEBCAM_CONFIG_OPTIONS.get(config_key)

	return content


# A face name reaches the config file, the dropdown and the state key, so it is reduced to the
# characters all three handle rather than carrying whatever the file system allowed
def sanitize_source_face_name(file_name : str) -> str:
	face_name = ''.join(character if character.isalnum() or character in [ '-', '_' ] else '-' for character in file_name)

	while '--' in face_name:
		face_name = face_name.replace('--', '-')

	return face_name.strip('-')


# Two files can sanitize to the same name, and a name is the key a face is selected by, so a clash
# is suffixed rather than allowed to overwrite the earlier face
def resolve_source_face_name(file_path : str, source_faces : Dict[str, List[str]]) -> Optional[str]:
	face_name = sanitize_source_face_name(os.path.splitext(os.path.basename(file_path))[0])

	if not face_name:
		return None
	unique_face_name = face_name
	name_index = 2

	while unique_face_name in source_faces:
		unique_face_name = face_name + '-' + str(name_index)
		name_index += 1

	return unique_face_name


# Each image in the directory becomes its own selectable face named after the file, so dropping a
# photo in is the whole workflow; is_image applies the one image whitelist this project already has
def resolve_source_face_directory(directory_path : Optional[str]) -> Dict[str, List[str]]:
	source_faces : Dict[str, List[str]] = {}

	if not directory_path:
		return source_faces

	for file_path in resolve_file_paths(directory_path):
		if is_image(file_path):
			face_name = resolve_source_face_name(file_path, source_faces)
			if face_name:
				source_faces[face_name] = [ file_path ]

	if not source_faces:
		logger.warn('source face directory ' + directory_path + ' holds no usable image', __name__)

	return source_faces


def format_webcam_config(webcam_options : Dict[str, Any], settings : Dict[str, Any]) -> Optional[str]:
	try:
		return format_json(create_webcam_config(webcam_options, settings))
	except (TypeError, ValueError) as exception:
		logger.error('webcam config could not be generated: ' + str(exception), __name__)

	return None


# load_webcam_config is cached, so its dict is shared by every caller; anything that stores the face
# groups elsewhere takes a copy instead of a reference into that cache
def get_source_faces(webcam_config : Dict[str, Any]) -> Dict[str, List[str]]:
	return deepcopy(webcam_config.get('webcam_source_faces') or {})


# Uploads accumulate rather than replace: the browser widget hands over whatever it currently holds,
# and a face already accumulated under the same path keeps its name instead of being added twice
def append_source_face_uploads(source_face_uploads : Dict[str, List[str]], file_paths : List[str]) -> Dict[str, List[str]]:
	appended_source_face_uploads = deepcopy(source_face_uploads or {})
	uploaded_file_paths = [ source_path for source_paths in appended_source_face_uploads.values() for source_path in source_paths ]

	for file_path in file_paths:
		if is_image(file_path) and file_path not in uploaded_file_paths:
			face_name = resolve_source_face_name(file_path, appended_source_face_uploads)
			if face_name:
				appended_source_face_uploads[face_name] = [ file_path ]

	return appended_source_face_uploads
