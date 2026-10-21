import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Tuple


# Per thread rather than global: the stream processes several frames at once in a worker pool, and a
# shared counter would bill one worker's inference to another worker's frame. The background safety
# analysis runs in its own thread too, so it stays out of the figure by construction
class InferenceTimerContext(threading.local):
	def __init__(self) -> None:
		self.total_ms = 0.0
		self.run_count = 0


INFERENCE_TIMER_CONTEXT = InferenceTimerContext()


def reset_inference_timer() -> None:
	INFERENCE_TIMER_CONTEXT.total_ms = 0.0
	INFERENCE_TIMER_CONTEXT.run_count = 0


def get_inference_timer() -> Tuple[float, int]:
	return INFERENCE_TIMER_CONTEXT.total_ms, INFERENCE_TIMER_CONTEXT.run_count


# Wall clock around the call, which is what the frame actually waits for. On cuda the runtime
# synchronizes on output, so this is the real device time rather than a queue submission
@contextmanager
def measure_inference() -> Iterator[None]:
	start_time = time.perf_counter()

	try:
		yield
	finally:
		INFERENCE_TIMER_CONTEXT.total_ms += (time.perf_counter() - start_time) * 1000
		INFERENCE_TIMER_CONTEXT.run_count += 1


# Every onnx model in the project reaches its device through a session built in one place, so
# wrapping the session measures all of them without touching the 33 call sites that run one
class TimedInferenceSession:
	def __init__(self, inference_session : Any) -> None:
		self.inference_session = inference_session

	def run(self, *args : Any, **kwargs : Any) -> Any:
		with measure_inference():
			return self.inference_session.run(*args, **kwargs)

	# Everything else - get_inputs, get_outputs, get_providers - is forwarded untouched.
	# object.__getattribute__ rather than self., so a missing attribute raises instead of recursing
	def __getattr__(self, attribute_name : str) -> Any:
		return getattr(object.__getattribute__(self, 'inference_session'), attribute_name)
