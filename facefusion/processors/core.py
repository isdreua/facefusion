import importlib
from types import ModuleType
from typing import Any, List

from facefusion import logger, translator
from facefusion.exit_helper import hard_exit


PROCESSORS_METHODS =\
[
	'get_inference_pool',
	'clear_inference_pool',
	'register_args',
	'apply_args',
	'get_common_modules',
	'pre_check',
	'pre_process',
	'post_process',
	'process_frame'
]


def load_processor_module(processor : str) -> Any:
	try:
		processor_module = importlib.import_module('facefusion.processors.modules.' + processor + '.core')
		for method_name in PROCESSORS_METHODS:
			if not hasattr(processor_module, method_name):
				raise NotImplementedError
	except ModuleNotFoundError as exception:
		logger.error(translator.get('processor_not_loaded').format(processor = processor), __name__)
		logger.debug(exception.msg, __name__)
		hard_exit(1)
	except NotImplementedError:
		logger.error(translator.get('processor_not_implemented').format(processor = processor), __name__)
		hard_exit(1)
	return processor_module


# The chain carries loaded modules, whose path is 'facefusion.processors.modules.<name>.core'. The
# name is the segment before core, and is what the state keys, the config and the ui all use
def resolve_processor_name(processor_module : ModuleType) -> str:
	processor_parts = processor_module.__name__.split('.')

	if len(processor_parts) > 1 and processor_parts[-1] == 'core':
		return processor_parts[-2]
	return processor_parts[-1]


def get_processors_modules(processors : List[str]) -> List[ModuleType]:
	processor_modules = []

	for processor in processors:
		processor_module = load_processor_module(processor)
		processor_modules.append(processor_module)
	return processor_modules
