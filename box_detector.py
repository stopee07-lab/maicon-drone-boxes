"""Pixel candidates for the practice field's printed static box panels.

These are upper-face observations. This module neither estimates ground position
nor decides whether an observed object is stationary; the caller owns those checks.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


_REFERENCE_SIZE = (1280, 720)
_PANEL_SIZE = 64
_PANEL_CORNERS = np.float32(((0, 0), (63, 0), (63, 63), (0, 63)))
_TEXT_KERNEL = np.ones((5, 5), dtype=np.uint8)
_MIN_SCORE = 0.42


@dataclass(frozen=True)
class BoxDetection:
    """Upper-face bounding box and center in original image pixels.

    The bounding box uses half-open (left, top, right, bottom) coordinates.
    Confidence is a heuristic ranking score in [0, 1], not a probability.
    """

    bbox_px: tuple[int, int, int, int]
    center_px: tuple[float, float]
    confidence: float


def _order_quad(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -int(ordered.sum(axis=1).argmin()), axis=0).astype(np.float32)


def _outline_candidates(gray: np.ndarray) -> Iterator[np.ndarray]:
    # At 720p, a visible upper face usually spans several dozen pixels per side.
    # Different edge thresholds and gap widths help with glare and printing gaps.
    for low, high in ((25, 65), (40, 90), (55, 130), (75, 175)):
        edges = cv2.Canny(gray, low, high)
        for gap in (1, 3, 5, 7):
            closed = edges if gap == 1 else cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE, np.ones((gap, gap), dtype=np.uint8)
            )
            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not 1700 <= area <= 23000:
                    continue
                perimeter = cv2.arcLength(contour, True)
                for tolerance in (0.015, 0.025, 0.035):
                    points = cv2.approxPolyDP(contour, tolerance * perimeter, True)
                    if len(points) != 4 or not cv2.isContourConvex(points):
                        continue
                    _, _, width, height = cv2.boundingRect(points)
                    if not (36 <= width <= 205 and 39 <= height <= 205):
                        continue
                    quad = _order_quad(points.reshape(4, 2))
                    yield quad
                    if area < 5600:
                        continue
                    # Adjacent boxes sometimes close as one outline. Both axes
                    # are tried and each half must pass the same perimeter test.
                    for offset in (1, 2):
                        a, b, c, d = np.roll(quad, offset, axis=0)
                        for split in (0.47, 0.50, 0.53):
                            upper = (1 - split) * a + split * b
                            lower = (1 - split) * d + split * c
                            yield np.float32([a, upper, lower, d])
                            yield np.float32([upper, b, c, lower])


def _perimeter_score(gray: np.ndarray, quad: np.ndarray) -> float:
    sides = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
    shortest, longest = sides.min(), sides.max()
    if shortest < 34 or longest > 122 or longest / shortest > 2.4:
        return 0.0
    transform = cv2.getPerspectiveTransform(quad, _PANEL_CORNERS)
    patch = cv2.warpPerspective(gray, transform, (_PANEL_SIZE, _PANEL_SIZE))
    detail = cv2.morphologyEx(patch, cv2.MORPH_TOPHAT, _TEXT_KERNEL)

    # Only the four narrow perimeter strips vote. A box fits when several
    # strips contain alternating bright details from the printed border; a
    # continuous painted floor line or a bright center cannot provide that.
    sides_with_print = 0
    scores = []
    for side in range(4):
        strip = np.rot90(detail, side)[1:8, 5:59]
        bright = strip.max(axis=0) > 18
        transitions = float(np.mean(bright[1:] != bright[:-1]))
        if transitions > 0.055:
            sides_with_print += 1
        coverage = min(float(bright.mean()), 0.65)
        scores.append(0.43 * coverage + 0.75 * transitions + float(strip.mean()) / 55)
    if sides_with_print < 2:
        return 0.0
    return min(1.0, float(np.mean(scores)))


def detect_boxes(
    image: np.ndarray, roi: tuple[int, int, int, int] | None = None
) -> list[BoxDetection]:
    """Return box-panel candidates in an 8-bit grayscale, BGR or BGRA image.

    ``roi`` is an optional half-open (left, top, right, bottom) rectangle in
    the input frame. Pixel outputs always refer to the full input frame.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or not image.size:
        raise ValueError("expected a nonempty uint8 image")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(image, conversion)
    else:
        raise ValueError("expected a grayscale, BGR or BGRA image")
    height, width = gray.shape

    left, top, right, bottom = (0, 0, width, height) if roi is None else roi
    if not all(isinstance(value, (int, np.integer)) for value in (left, top, right, bottom)):
        raise ValueError("ROI coordinates must be integers")
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("ROI must be a nonempty rectangle within the image")

    # Size gates depend on the full camera frame, not on the size of an ROI.
    scale = min(_REFERENCE_SIZE[0] / width, _REFERENCE_SIZE[1] / height)
    crop = gray[top:bottom, left:right]
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    if min(size) < 34:
        return []
    if size != (crop.shape[1], crop.shape[0]):
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        crop = cv2.resize(crop, size, interpolation=interpolation)

    candidates = []
    seen = set()
    for quad in _outline_candidates(crop):
        key = tuple(np.rint(quad).astype(np.int32).ravel())
        if key in seen:
            continue
        seen.add(key)
        score = _perimeter_score(crop, quad)
        if score >= _MIN_SCORE:
            candidates.append((score, quad, quad.mean(axis=0), cv2.contourArea(quad)))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected = []
    for score, quad, center, area in candidates:
        duplicate = any(
            np.linalg.norm(center - prior_center) < 21
            or cv2.intersectConvexConvex(quad, prior_quad)[0] > 0.42 * min(area, prior_area)
            for _, prior_quad, prior_center, prior_area in selected
        )
        if not duplicate:
            selected.append((score, quad, center, area))

    scale_xy = np.array([(right - left) / size[0], (bottom - top) / size[1]])
    origin = np.array([left, top])
    detections = []
    for score, quad, center, _ in selected:
        mapped = quad * scale_xy + origin
        x1, y1 = (int(value) for value in np.floor(mapped.min(axis=0)))
        x2, y2 = (int(value) for value in np.ceil(mapped.max(axis=0)) + 1)
        x, y = (float(value) for value in center * scale_xy + origin)
        detections.append(BoxDetection((x1, y1, min(x2, right), min(y2, bottom)), (x, y), score))
    return detections
