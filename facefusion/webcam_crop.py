from typing import Any, Optional

import cv2
import numpy

from facefusion import logger, state_manager
from facefusion.types import Resolution, VisionFrame, WebcamCropBox
from facefusion.webcam_config import WEBCAM_RESOLUTION_RANGE, normalize_webcam_crop_size

# 1 is the whole picture. At the top a 320x240 capture still leaves an 80x60 crop, above the 16 pixel
# side every crop is held to
WEBCAM_CROP_ZOOM_RANGE = (1.0, 4.0)
WEBCAM_CROP_ZOOM_STEP = 0.1
# How far a crop may stray from the capture's aspect ratio before an upscale counts as a stretch. Zoom
# rounds both sides to even pixels, which alone moves the ratio by a fraction of a percent
WEBCAM_CROP_ASPECT_TOLERANCE = 0.01


# None when nothing is cropped. Offsets arrive as ints from a config and as floats or None from a
# number field; a negative one is kept as it is, so the bounds check can name it rather than hide it
def resolve_webcam_crop_box(crop_size : Any, crop_offset_x : Any, crop_offset_y : Any) -> Optional[WebcamCropBox]:
	crop_size = normalize_webcam_crop_size(crop_size)

	if not crop_size or crop_size == 'none':
		return None
	crop_width, crop_height = map(int, crop_size.split('x'))
	return coerce_webcam_crop_offset(crop_offset_x), coerce_webcam_crop_offset(crop_offset_y), crop_width, crop_height


def coerce_webcam_crop_offset(crop_offset : Any) -> int:
	if isinstance(crop_offset, (int, float)) and not isinstance(crop_offset, bool):
		try:
			return int(round(crop_offset))
		except (OverflowError, ValueError):
			return 0
	return 0


def format_webcam_crop_size(crop_box : Optional[WebcamCropBox]) -> str:
	if crop_box:
		return str(crop_box[2]) + 'x' + str(crop_box[3])
	return 'none'


# Why the box cannot be taken from a frame of the capture resolution, or None when it can
def describe_webcam_crop_error(capture_resolution : Resolution, crop_box : WebcamCropBox) -> Optional[str]:
	capture_width, capture_height = capture_resolution
	crop_x, crop_y, crop_width, crop_height = crop_box
	capture_size = str(capture_width) + 'x' + str(capture_height)

	if crop_width > capture_width or crop_height > capture_height:
		return 'webcam crop ' + format_webcam_crop_size(crop_box) + ' is larger than the capture resolution ' + capture_size
	if crop_x < 0 or crop_y < 0 or crop_x + crop_width > capture_width or crop_y + crop_height > capture_height:
		return 'webcam crop ' + format_webcam_crop_size(crop_box) + ' at ' + str(crop_x) + ',' + str(crop_y) + ' reaches past the edge of the capture resolution ' + capture_size + ', its offset can be 0,0 to ' + str(capture_width - crop_width) + ',' + str(capture_height - crop_height)
	return None


# The nearest box that lies inside the capture: a side larger than the capture's is cut to it, then the
# box is moved back in from whichever edge it crossed. Capture sides are even, so a cut side stays even
def fit_webcam_crop_box(capture_resolution : Resolution, crop_box : WebcamCropBox) -> WebcamCropBox:
	capture_width, capture_height = capture_resolution
	crop_x, crop_y, crop_width, crop_height = crop_box
	crop_width = min(crop_width, capture_width)
	crop_height = min(crop_height, capture_height)
	crop_x = min(max(0, crop_x), capture_width - crop_width)
	crop_y = min(max(0, crop_y), capture_height - crop_height)
	return crop_x, crop_y, crop_width, crop_height


# The zoom sets a crop of the capture's own aspect ratio, so an upscaled zoom never stretches, and keeps
# the centre of the crop it replaces, so zooming in on a panned crop stays on what it was panned to.
# Zoom 1 is no crop at all rather than a crop of the whole picture, which would only cost a resize
def zoom_webcam_crop_box(capture_resolution : Resolution, crop_zoom : float, crop_box : Optional[WebcamCropBox]) -> Optional[WebcamCropBox]:
	capture_width, capture_height = capture_resolution
	crop_zoom = min(float(crop_zoom or WEBCAM_CROP_ZOOM_RANGE[0]), WEBCAM_CROP_ZOOM_RANGE[1])

	if round(crop_zoom, 2) <= WEBCAM_CROP_ZOOM_RANGE[0]:
		return None
	crop_width = min(capture_width, max(WEBCAM_RESOLUTION_RANGE[0], int(capture_width / crop_zoom) // 2 * 2))
	crop_height = min(capture_height, max(WEBCAM_RESOLUTION_RANGE[0], int(capture_height / crop_zoom) // 2 * 2))
	center_x, center_y = capture_width / 2, capture_height / 2
	if crop_box:
		center_x, center_y = crop_box[0] + crop_box[2] / 2, crop_box[1] + crop_box[3] / 2
	return fit_webcam_crop_box(capture_resolution, (int(round(center_x - crop_width / 2)), int(round(center_y - crop_height / 2)), crop_width, crop_height))


# The zoom a crop amounts to, on the slider's steps. A crop typed in another aspect ratio shows the
# smaller of its two magnifications, the one the whole crop is seen at
def resolve_webcam_crop_zoom(capture_resolution : Resolution, crop_box : Optional[WebcamCropBox]) -> float:
	if not crop_box:
		return WEBCAM_CROP_ZOOM_RANGE[0]
	crop_zoom = min(capture_resolution[0] / crop_box[2], capture_resolution[1] / crop_box[3])
	crop_zoom = round(round(crop_zoom / WEBCAM_CROP_ZOOM_STEP) * WEBCAM_CROP_ZOOM_STEP, 2)
	return min(max(crop_zoom, WEBCAM_CROP_ZOOM_RANGE[0]), WEBCAM_CROP_ZOOM_RANGE[1])


def has_webcam_crop_aspect_ratio(capture_resolution : Resolution, crop_box : WebcamCropBox) -> bool:
	capture_width, capture_height = capture_resolution
	crop_width, crop_height = crop_box[2], crop_box[3]
	return abs(crop_width * capture_height - capture_width * crop_height) <= WEBCAM_CROP_ASPECT_TOLERANCE * capture_width * crop_height


# The size frames reach the processors at, which every output size is derived from: the capture
# resolution, unless a crop is taken and left at its own size
def resolve_webcam_stream_resolution(webcam_resolution : str, crop_box : Optional[WebcamCropBox], is_upscale : bool) -> str:
	if crop_box and not is_upscale:
		return format_webcam_crop_size(crop_box)
	return webcam_resolution


# A box set in capture pixels, carried over to a frame of another size the way the picture itself
# would be, so a camera that grants a smaller mode keeps the same framing rather than a shifted one
def scale_webcam_crop_box(crop_box : WebcamCropBox, capture_resolution : Resolution, frame_resolution : Resolution) -> WebcamCropBox:
	capture_width, capture_height = capture_resolution
	frame_width, frame_height = frame_resolution
	crop_x, crop_y, crop_width, crop_height = crop_box
	crop_box = (crop_x * frame_width // capture_width, crop_y * frame_height // capture_height, max(1, crop_width * frame_width // capture_width), max(1, crop_height * frame_height // capture_height))
	return fit_webcam_crop_box(frame_resolution, crop_box)


# Applied by the scheduler to each camera frame before anything processes it. The box is read from the
# state on every frame, so the zoom and the offsets move a running stream. With the upscale, every crop
# leaves at the capture resolution and may change size freely. Without it, the crop size is the size the
# writer and window were opened at, so a box of another size waits for the next start
class StreamCrop:
	def __init__(self, capture_resolution : Resolution, is_upscale : bool, crop_box : Optional[WebcamCropBox]) -> None:
		self.capture_resolution = capture_resolution
		self.is_upscale = is_upscale
		self.crop_box = crop_box
		self.crop_state : Any = None
		self.has_logged_frame_scale = False

	def resolve_crop_box(self) -> Optional[WebcamCropBox]:
		crop_state = (state_manager.get_item('webcam_crop_size'), state_manager.get_item('webcam_crop_offset_x'), state_manager.get_item('webcam_crop_offset_y'))

		if crop_state != self.crop_state:
			self.crop_state = crop_state
			crop_box = resolve_webcam_crop_box(*crop_state)
			if crop_box:
				crop_box = fit_webcam_crop_box(self.capture_resolution, crop_box)
			if self.is_upscale or (crop_box and self.crop_box and crop_box[2:] == self.crop_box[2:]):
				self.crop_box = crop_box
			elif crop_box != self.crop_box:
				logger.info('webcam crop ' + format_webcam_crop_size(crop_box) + ' changes the output size while upscale is off, it applies on the next start', __name__)
		return self.crop_box

	def apply(self, vision_frame : VisionFrame) -> VisionFrame:
		crop_box = self.resolve_crop_box()

		if not crop_box:
			return vision_frame
		frame_height, frame_width = vision_frame.shape[:2]
		if (frame_width, frame_height) != self.capture_resolution:
			if not self.has_logged_frame_scale:
				self.has_logged_frame_scale = True
				logger.debug('camera delivered ' + str(frame_width) + 'x' + str(frame_height) + ' for the capture resolution ' + str(self.capture_resolution[0]) + 'x' + str(self.capture_resolution[1]) + ', scaling the crop to it', __name__)
			crop_box = scale_webcam_crop_box(crop_box, self.capture_resolution, (frame_width, frame_height))
		crop_x, crop_y, crop_width, crop_height = crop_box
		if (crop_width, crop_height) == (frame_width, frame_height) and (frame_width, frame_height) == self.capture_resolution:
			return vision_frame
		crop_vision_frame = vision_frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
		if self.is_upscale:
			return cv2.resize(crop_vision_frame, self.capture_resolution)
		# A slice is a view with the full frame's row stride; the processors get a frame of their own
		return numpy.ascontiguousarray(crop_vision_frame)
