import threading
from typing import Callable, Optional


# Raised from deep inside a processor, so it derives from BaseException rather than Exception: the
# processors and nvidia_vfx catch Exception broadly to keep one failed step from taking the frame
# down with it, and an abort has to pass straight through those handlers to the one place that knows
# what it means
class StreamAborted(BaseException):
	pass


# Thread local like the app context override and the paste in place flag, because a stream worker
# owns its frame for the whole chain and nothing else may answer for it
class StreamAbortContext(threading.local):
	resolve_abort : Optional[Callable[[], bool]] = None


STREAM_ABORT_CONTEXT = StreamAbortContext()


def set_stream_abort(resolve_abort : Optional[Callable[[], bool]]) -> None:
	STREAM_ABORT_CONTEXT.resolve_abort = resolve_abort


def has_stream_aborted() -> bool:
	resolve_abort = getattr(STREAM_ABORT_CONTEXT, 'resolve_abort', None)
	return bool(resolve_abort and resolve_abort())


# Called wherever the next step is expensive enough to be worth not taking. Outside a stream worker
# nothing sets a predicate, so this is one thread local attribute read
def abort_stale_stream_frame() -> None:
	if has_stream_aborted():
		raise StreamAborted()
