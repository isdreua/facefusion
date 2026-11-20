import threading
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import lru_cache
from types import ModuleType
from typing import Iterator, List

import numpy

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, state_manager, translator, video_manager, voice_extractor
from facefusion.common_helper import create_float_metavar, create_int_metavar, get_middle
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_creator import scale_face
from facefusion.face_detector import get_stream_worker_count
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask, create_occlusion_mask
from facefusion.face_selector import select_faces
from facefusion.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.processors.modules.face_enhancer import choices as face_enhancer_choices
from facefusion.processors.modules.face_enhancer.types import FaceEnhancerInputs, FaceEnhancerWeight
from facefusion.processors.types import ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import ApplyStateItem, Args, DownloadScope, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_video_frame

# A stream gives the enhancer workers of its own, decoupled from the webcam workers: each is a session
# of its own, checked out by one thread at a time, so the count bounds how many frames enhance at once.
# Zero copies the webcam worker count. Outside a stream the plain shared session is used unchecked,
# which is the upstream behaviour
ENHANCER_SLOT_CONDITION = threading.Condition()
ENHANCER_SLOT_INDICES : List[int] = []
ENHANCER_SLOT_COUNTS = { 'created': 0 }


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'codeformer':
		{
			'__metadata__':
			{
				'vendor': 'sczhou',
				'license': 'S-Lab-1.0',
				'year': 2022
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'codeformer.hash'),
					'path': resolve_relative_path('../.assets/models/codeformer.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'codeformer.onnx'),
					'path': resolve_relative_path('../.assets/models/codeformer.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gfpgan_1.2':
		{
			'__metadata__':
			{
				'vendor': 'TencentARC',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.2.hash'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.2.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.2.onnx'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.2.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gfpgan_1.3':
		{
			'__metadata__':
			{
				'vendor': 'TencentARC',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.3.hash'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.3.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.3.onnx'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.3.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gfpgan_1.4':
		{
			'__metadata__':
			{
				'vendor': 'TencentARC',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.4.hash'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.4.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.4.onnx'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.4.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gpen_bfr_256':
		{
			'__metadata__':
			{
				'vendor': 'yangxy',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_256.hash'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_256.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_256.onnx'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_256.onnx')
				}
			},
			'template': 'arcface_128',
			'size': (256, 256)
		},
		'gpen_bfr_512':
		{
			'__metadata__':
			{
				'vendor': 'yangxy',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_512.hash'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_512.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_512.onnx'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_512.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gpen_bfr_1024':
		{
			'__metadata__':
			{
				'vendor': 'yangxy',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_1024.hash'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_1024.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_1024.onnx'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_1024.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (1024, 1024)
		},
		'gpen_bfr_2048':
		{
			'__metadata__':
			{
				'vendor': 'yangxy',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_2048.hash'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_2048.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_2048.onnx'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_2048.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (2048, 2048)
		},
		'restoreformer_plus_plus':
		{
			'__metadata__':
			{
				'vendor': 'wzhouxiff',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'restoreformer_plus_plus.hash'),
					'path': resolve_relative_path('../.assets/models/restoreformer_plus_plus.hash')
				}
			},
			'sources':
			{
				'face_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'restoreformer_plus_plus.onnx'),
					'path': resolve_relative_path('../.assets/models/restoreformer_plus_plus.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		}
	}


def get_inference_pool(slot_index : int = 0) -> InferencePool:
	model_names = resolve_slot_model_names(slot_index)
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	with ENHANCER_SLOT_CONDITION:
		slot_count = max(1, ENHANCER_SLOT_COUNTS.get('created'))

	for slot_index in range(slot_count):
		inference_manager.clear_inference_pool(__name__, resolve_slot_model_names(slot_index))


# Slot zero keeps the plain context, so it is the very session everything outside a stream and the
# tensorrt precache use. Every further slot is a context of its own and therefore a session of its own
def resolve_slot_model_names(slot_index : int) -> List[str]:
	model_names = [ state_manager.get_item('face_enhancer_model') ]

	if slot_index > 0:
		model_names.append('slot_' + str(slot_index))
	return model_names


# Read on every checkout rather than fixed at stream start, so changing either count mid stream takes
# effect on the next face. Zero means no stream is running
def resolve_slot_limit() -> int:
	stream_worker_count = get_stream_worker_count()

	if stream_worker_count > 0:
		return state_manager.get_item('webcam_face_enhancer_worker_count') or stream_worker_count
	return 0


def acquire_slot(slot_limit : int) -> int:
	with ENHANCER_SLOT_CONDITION:
		while True:
			free_slot_indices = [ slot_index for slot_index in ENHANCER_SLOT_INDICES if slot_index < slot_limit ]

			if free_slot_indices:
				slot_index = min(free_slot_indices)
				ENHANCER_SLOT_INDICES.remove(slot_index)
				return slot_index
			if ENHANCER_SLOT_COUNTS.get('created') < slot_limit:
				ENHANCER_SLOT_COUNTS['created'] += 1
				return ENHANCER_SLOT_COUNTS.get('created') - 1
			ENHANCER_SLOT_CONDITION.wait()
			slot_limit = resolve_slot_limit() or slot_limit


def release_slot(slot_index : int) -> None:
	with ENHANCER_SLOT_CONDITION:
		ENHANCER_SLOT_INDICES.append(slot_index)
		ENHANCER_SLOT_CONDITION.notify_all()


@contextmanager
def checkout_inference_pool() -> Iterator[InferencePool]:
	slot_limit = resolve_slot_limit()

	if slot_limit < 1:
		yield get_inference_pool()
		return
	slot_index = acquire_slot(slot_limit)

	try:
		yield get_inference_pool(slot_index)
	finally:
		release_slot(slot_index)


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('face_enhancer_model')
	return create_static_model_set('full').get(model_name)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--face-enhancer-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'face_enhancer_model', 'gfpgan_1.4'), choices = face_enhancer_choices.face_enhancer_models)
		group_processors.add_argument('--face-enhancer-blend', help = translator.get('help.blend', __package__), type = int, default = config.get_int_value('processors', 'face_enhancer_blend', '80'), choices = face_enhancer_choices.face_enhancer_blend_range, metavar = create_int_metavar(face_enhancer_choices.face_enhancer_blend_range))
		group_processors.add_argument('--face-enhancer-weight', help = translator.get('help.weight', __package__), type = float, default = config.get_float_value('processors', 'face_enhancer_weight', '0.5'), choices = face_enhancer_choices.face_enhancer_weight_range, metavar = create_float_metavar(face_enhancer_choices.face_enhancer_weight_range))
		facefusion.jobs.job_store.register_step_keys([ 'face_enhancer_model', 'face_enhancer_blend', 'face_enhancer_weight' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('face_enhancer_model', args.get('face_enhancer_model'))
	apply_state_item('face_enhancer_blend', args.get('face_enhancer_blend'))
	apply_state_item('face_enhancer_weight', args.get('face_enhancer_weight'))


def get_common_modules() -> List[ModuleType]:
	return [ content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, voice_extractor ]


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	for common_module in get_common_modules():
		if not common_module.pre_check():
			return False

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def pre_process(mode : ProcessMode) -> bool:
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()

	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()

	if state_manager.get_item('video_memory_strategy') == 'strict':
		for common_module in get_common_modules():
			common_module.clear_inference_pool()


def enhance_face(target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, model_size)
	box_mask = create_box_mask(crop_vision_frame.shape[:2][::-1], state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
	crop_masks =\
	[
		box_mask
	]

	if 'occlusion' in state_manager.get_item('face_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	crop_vision_frame = prepare_crop_frame(crop_vision_frame)
	face_enhancer_weight = numpy.array([ state_manager.get_item('face_enhancer_weight') ]).astype(numpy.double)
	crop_vision_frame = forward(crop_vision_frame, face_enhancer_weight)
	crop_vision_frame = normalize_crop_frame(crop_vision_frame)
	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	face_enhancer_blend = state_manager.get_item('face_enhancer_blend') / 100
	crop_mask = crop_mask * face_enhancer_blend
	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	return paste_vision_frame


def forward(crop_vision_frame : VisionFrame, face_enhancer_weight : FaceEnhancerWeight) -> VisionFrame:
	with checkout_inference_pool() as inference_pool:
		face_enhancer = inference_pool.get('face_enhancer')
		face_enhancer_inputs = {}

		for face_enhancer_input in face_enhancer.get_inputs():
			if face_enhancer_input.name == 'input':
				face_enhancer_inputs[face_enhancer_input.name] = crop_vision_frame
			if face_enhancer_input.name == 'weight':
				face_enhancer_inputs[face_enhancer_input.name] = face_enhancer_weight

		with conditional_thread_semaphore():
			crop_vision_frame = face_enhancer.run(None, face_enhancer_inputs)[0][0]

	return crop_vision_frame


def has_weight_input() -> bool:
	face_enhancer = get_inference_pool().get('face_enhancer')

	for deep_swapper_input in face_enhancer.get_inputs():
		if deep_swapper_input.name == 'weight':
			return True

	return False


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = crop_vision_frame[:, :, ::-1].astype(numpy.float32)
	numpy.divide(crop_vision_frame, 127.5, out = crop_vision_frame)
	numpy.subtract(crop_vision_frame, 1.0, out = crop_vision_frame)
	crop_vision_frame = numpy.ascontiguousarray(crop_vision_frame.transpose(2, 0, 1))[None]
	return crop_vision_frame


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = numpy.clip(crop_vision_frame, -1, 1)
	crop_vision_frame = (crop_vision_frame + 1) / 2
	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
	crop_vision_frame = (crop_vision_frame * 255.0).round()
	crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
	return crop_vision_frame


def process_frame(inputs : FaceEnhancerInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	source_vision_frames = inputs.get('source_vision_frames')
	target_vision_frames = inputs.get('target_vision_frames')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')

	target_vision_frame = get_middle(target_vision_frames)
	target_faces = select_faces(reference_vision_frame, source_vision_frames, target_vision_frames)

	if target_faces:
		for target_face in target_faces:
			target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
			temp_vision_frame = enhance_face(target_face, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
