import time

import numpy

from facefusion.stream_writer import LatestFrameWriter


class FakeStdin:
	def __init__(self):
		self.writes = []
		self.closed = False

	def write(self, data):
		self.writes.append(data)

	def close(self):
		self.closed = True


class FakeTransport:
	def __init__(self):
		self.stdin = FakeStdin()


def test_latest_frame_writer_writes_and_closes():
	transport = FakeTransport()
	writer = LatestFrameWriter(lambda: transport, 2, 2)
	writer.start()
	writer.submit(numpy.zeros((2, 2, 3), dtype = numpy.uint8))
	time.sleep(0.03)
	writer.close()
	assert len(transport.stdin.writes) == 1
	assert transport.stdin.closed is True


def test_latest_frame_writer_rejects_bad_shape():
	writer = LatestFrameWriter(FakeTransport, 2, 2)
	writer.start()
	try:
		writer.submit(numpy.zeros((1, 2, 3), dtype = numpy.uint8))
		assert False
	except ValueError:
		pass
	finally:
		writer.close()
