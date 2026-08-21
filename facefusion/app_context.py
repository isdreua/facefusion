import os
import sys
import threading
from typing import Dict, Optional

from facefusion.types import AppContext

JOBS_PATH = os.path.join('facefusion', 'jobs')
UIS_PATH = os.path.join('facefusion', 'uis')
FILE_CONTEXT_CACHE : Dict[str, Optional[AppContext]] = {}


class AppContextOverride(threading.local):
	context : Optional[AppContext] = None


APP_CONTEXT_OVERRIDE = AppContextOverride()


def set_app_context_override(app_context : Optional[AppContext]) -> None:
	APP_CONTEXT_OVERRIDE.context = app_context


def classify_file_context(file_name : str) -> Optional[AppContext]:
	if JOBS_PATH in file_name:
		return 'cli'
	if UIS_PATH in file_name:
		return 'ui'
	return None


def detect_app_context() -> AppContext:
	app_context = getattr(APP_CONTEXT_OVERRIDE, 'context', None)

	if app_context:
		return app_context

	frame = sys._getframe(1)

	while frame:
		file_name = frame.f_code.co_filename

		if file_name in FILE_CONTEXT_CACHE:
			file_context = FILE_CONTEXT_CACHE.get(file_name)
		else:
			file_context = classify_file_context(file_name)
			FILE_CONTEXT_CACHE[file_name] = file_context

		if file_context:
			return file_context
		frame = frame.f_back
	return 'cli'
