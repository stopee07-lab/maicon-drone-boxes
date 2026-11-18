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
_MIN_ENHANCED_SCORE = 0.345
_MIN_CENTER_DISTANCE = 36
_FACE_INTERIOR = np.ones((40, 40), dtype=bool)
_FACE_INTERIOR[11:29, 12:28] = False


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


def _outline_candidates(gray: np.ndarray, recovery: bool = False) -> Iterator[np.ndarray]:
    # At 720p, a visible upper face usually spans several dozen pixels per side.
    # Different edge thresholds and gap widths help with glare and printing gaps.
    thresholds = ((100, 230),) if recovery else ((25, 65), (40, 90), (55, 130), (75, 175))
    gaps = (5, 7, 9) if recovery else (1, 3, 5, 7)
    for low, high in thresholds:
        edges = cv2.Canny(gray, low, high)
        for gap in gaps:
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


def _panel_patch(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    transform = cv2.getPerspectiveTransform(quad, _PANEL_CORNERS)
    return cv2.warpPerspective(gray, transform, (_PANEL_SIZE, _PANEL_SIZE))


def _perimeter_score(gray: np.ndarray, quad: np.ndarray) -> float:
    sides = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
    shortest, longest = sides.min(), sides.max()
    if shortest < 34 or longest > 122 or longest / shortest > 2.4:
        return 0.0
    patch = _panel_patch(gray, quad)
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


def _has_glare_box_evidence(gray: np.ndarray, quad: np.ndarray) -> bool:
    """Require both a bright face and printed details on opposite sides."""
    area = cv2.contourArea(quad)
    lengths = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
    shortest = float(lengths.min())
    if not (3000 <= area <= 11000 and shortest >= 36 and lengths.max() / shortest <= 2.2):
        return False

    patch = _panel_patch(gray, quad)
    center = patch[12:52, 12:52]
    if np.mean(center >= 190) < 0.65 or np.percentile(center, 90) < 218:
        return False

    detail = cv2.morphologyEx(patch, cv2.MORPH_TOPHAT, _TEXT_KERNEL)
    transitions, peaks = [], []
    for side in range(4):
        line = np.rot90(detail, side)[1:8, 5:59].max(axis=0)
        bright = line > 18
        transitions.append(float(np.mean(bright[1:] != bright[:-1])))
        peaks.append(float(np.percentile(line, 90)))
    return any(min(transitions[i], transitions[i + 2]) >= 0.07 and max(peaks[i], peaks[i + 2]) >= 28
               for i in (0, 1))


def _has_printed_flat_face(gray: np.ndarray, quad: np.ndarray, minimum_sides: int) -> bool:
    patch = _panel_patch(gray, quad)
    detail = cv2.morphologyEx(patch, cv2.MORPH_TOPHAT, _TEXT_KERNEL)
    printed, continuous = 0, 0
    for side in range(4):
        bright = np.rot90(detail, side)[1:8, 5:59].max(axis=0) > 18
        coverage = float(bright.mean())
        transitions = float(np.mean(bright[1:] != bright[:-1]))
        printed += int(transitions >= 0.075 and 0.12 <= coverage <= 0.84)
        continuous += int(coverage >= 0.90)
    if printed < minimum_sides or continuous >= 2:
        return False
    return _is_flat_panel(patch)


def _is_flat_panel(patch: np.ndarray, max_texture: float = 20) -> bool:
    # Ignore the printed border and the small center label in the texture
    # check. A seam through the interior can indicate two touching boxes.
    residual = cv2.absdiff(patch, cv2.GaussianBlur(patch, (7, 7), 0))[12:52, 12:52]
    return float(np.percentile(residual[_FACE_INTERIOR], 95)) <= max_texture


def _sharpen(gray: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def _outline_views(gray: np.ndarray) -> Iterator[tuple[np.ndarray, str, float]]:
    """Yield outline images, their evidence rules, and the minimum normal score."""
    yield gray, "original", _MIN_SCORE
    for clip in (2.0, 3.0):
        yield cv2.createCLAHE(clip, (8, 8)).apply(gray), "contrast", _MIN_ENHANCED_SCORE
    yield _sharpen(gray, 2, 0.9), "contrast", _MIN_SCORE
    yield _sharpen(gray, 3.5, 1.0), "contrast", _MIN_ENHANCED_SCORE
    yield _sharpen(gray, 2, 0.6), "contrast", _MIN_SCORE
    for clip in (3.0, 2.0):
        yield cv2.createCLAHE(clip, (4, 4)).apply(gray), "contrast", _MIN_ENHANCED_SCORE
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 2)
    yield binary, "contrast", _MIN_SCORE
    for grid, clip in ((8, 2.5), (6, 3.0), (6, 2.0), (4, 4.0)):
        yield cv2.createCLAHE(clip, (grid, grid)).apply(gray), "contrast", _MIN_ENHANCED_SCORE
    yield _sharpen(gray, 2.5, 1.3), "contrast", _MIN_SCORE
    # These fine-detail views are checked after ordinary and adjacent panels.
    # Limiting them to the intended lighting extremes avoids floor-line matches.
    yield cv2.createCLAHE(4.0, (12, 12)).apply(gray), "dark", _MIN_ENHANCED_SCORE
    yield _sharpen(gray, 0.8, 0.4), "glare", _MIN_SCORE
    # Strong edges and a wider closing kernel join fragmented printed borders.
    yield gray, "recovery", 0.50
    yield cv2.createCLAHE(3.0, (8, 8)).apply(gray), "recovery", 0.50


_Candidate = tuple[float, np.ndarray, np.ndarray, float]


def _duplicates(candidate: _Candidate, selected: list[_Candidate], overlap: float) -> bool:
    _, quad, center, area = candidate
    return any(
        np.linalg.norm(center - prior_center) < _MIN_CENTER_DISTANCE
        or cv2.intersectConvexConvex(quad, prior_quad)[0] > overlap * min(area, prior_area)
        for _, prior_quad, prior_center, prior_area in selected
    )


def _select(candidates: list[_Candidate], selected: list[_Candidate], overlap: float) -> None:
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        if not _duplicates(candidate, selected, overlap):
            selected.append(candidate)


def _neighbor_quads(
    shape: tuple[int, int], selected: list[_Candidate]
) -> Iterator[np.ndarray]:
    height, width = shape
    for _, quad, _, _ in selected:
        across = ((quad[1] - quad[0]) + (quad[2] - quad[3])) * 0.5
        down = ((quad[3] - quad[0]) + (quad[2] - quad[1])) * 0.5
        for axis, perpendicular in ((across, down), (-across, down), (down, across), (-down, across)):
            for step in (0.84, 0.94, 1.04, 1.14):
                for offset in (-0.13, 0.0, 0.13):
                    proposed = quad + axis * step + perpendicular * offset
                    if proposed.min() >= 0 and proposed[:, 0].max() < width and proposed[:, 1].max() < height:
                        yield proposed


def _printed_side_count(patch: np.ndarray) -> int:
    detail = cv2.morphologyEx(patch, cv2.MORPH_TOPHAT, _TEXT_KERNEL)
    printed = 0
    for side in range(4):
        bright = np.rot90(detail, side)[1:8, 5:59].max(axis=0) > 18
        printed += int(bright.mean() >= 0.25 and np.mean(bright[1:] != bright[:-1]) >= 0.055)
    return printed


def _recover_adjacent(gray: np.ndarray, selected: list[_Candidate]) -> None:
    # A shared edge may leave one box without a closed contour. Use the nearby
    # panel's geometry, then require actual printing on all four proposed sides.
    candidates = []
    for quad in _neighbor_quads(gray.shape, selected):
        center = quad.mean(axis=0)
        area = cv2.contourArea(quad)
        if _duplicates((0.0, quad, center, area), selected, 0.32):
            continue
        score = _perimeter_score(gray, quad)
        if score < _MIN_SCORE:
            continue
        patch = _panel_patch(gray, quad)
        if _is_flat_panel(patch, max_texture=12) and _printed_side_count(patch) == 4:
            candidates.append((score, quad, center, area))
    _select(candidates, selected, 0.32)


def _dark_neighbor_quads(gray: np.ndarray, selected: list[_Candidate]) -> Iterator[np.ndarray]:
    height, width = gray.shape
    for _, anchor, origin, _ in selected:
        if np.median(_panel_patch(gray, anchor)[12:52, 12:52]) > 65:
            continue
        for side in range(4):
            a, b, c, d = np.roll(anchor, side, axis=0)
            edge = c - d
            outward = ((d - a) + (c - b)) * 0.5
            if any(np.linalg.norm(origin + outward - center) < 28 for _, _, center, _ in selected):
                continue
            for gap in (0.0, 0.13, 0.23):
                edge_center = (d + c) * 0.5 + gap * outward
                for breadth in (0.90, 1.0, 1.14):
                    left = edge_center - 0.5 * breadth * edge
                    right = edge_center + 0.5 * breadth * edge
                    for length in (0.90, 1.02, 1.13, 1.25):
                        for skew in (-0.15, 0.0, 0.15):
                            offset = length * outward + skew * edge
                            quad = np.float32((left, right, right + offset, left + offset))
                            if quad.min() >= 0 and quad[:, 0].max() < width and quad[:, 1].max() < height:
                                yield np.roll(quad, -int(quad.sum(axis=1).argmin()), axis=0)


def _recover_dark_neighbors(gray: np.ndarray, selected: list[_Candidate]) -> None:
    # The neighboring face's width, depth, and skew can differ under perspective.
    # Accept the adjusted outline only when its own printed border supports it.
    candidates = []
    for quad in _dark_neighbor_quads(gray, selected):
        center = quad.mean(axis=0)
        area = cv2.contourArea(quad)
        if area < 2200 or _duplicates((0.0, quad, center, area), selected, 0.32):
            continue
        score = _perimeter_score(gray, quad)
        if score < 0.62:
            continue
        patch = _panel_patch(gray, quad)
        if np.median(patch[12:52, 12:52]) > 65 or not _is_flat_panel(patch, max_texture=10):
            continue
        if _printed_side_count(patch) >= 3:
            candidates.append((score, quad, center, area))
    _select(candidates, selected, 0.32)


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

    original, enhanced, supplemental, refined = [], [], [], []
    seen = set()
    for view, mode, min_score in _outline_views(crop):
        for quad in _outline_candidates(view, recovery=mode == "recovery"):
            key = tuple(np.rint(quad).astype(np.int32).ravel())
            if key in seen:
                continue
            seen.add(key)
            # Enhancement proposes outlines only. All scoring uses the original pixels.
            score = _perimeter_score(crop, quad)
            area = cv2.contourArea(quad)
            if mode == "original":
                if score >= _MIN_SCORE:
                    if not _is_flat_panel(_panel_patch(crop, quad)):
                        continue
                    group = original
                elif score >= 0.12 and _has_glare_box_evidence(crop, quad):
                    group = supplemental
                else:
                    continue
            else:
                if score < 0.26:
                    continue
                dark_outline = mode != "recovery" and score < _MIN_SCORE and area >= 2200
                dark_face = (mode == "dark" or dark_outline) and (
                    np.median(_panel_patch(crop, quad)[12:52, 12:52]) <= 65
                )
                dark = dark_outline and dark_face
                if score < min_score and not dark:
                    continue
                minimum_sides = 2 if score >= _MIN_SCORE or dark else 3
                if not _has_printed_flat_face(crop, quad, minimum_sides):
                    continue
                if mode == "dark" and not dark_face:
                    continue
                if mode == "glare" and not _has_glare_box_evidence(crop, quad):
                    continue
                group = enhanced if score >= _MIN_SCORE else supplemental
                if mode in ("dark", "glare", "recovery"):
                    group = refined

            # Large floor shadows and painted bays sometimes pass the print
            # test. Their proportions differ from the printed box faces.
            if area > 5900:
                lengths = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
                if np.max(lengths / np.roll(lengths, 2)) > 1.22 or lengths.max() > 2.2 * lengths.min():
                    continue
            group.append((score, quad, quad.mean(axis=0), area))

    selected = []
    # Original detections are evaluated before the additional candidates.
    for group in (original, enhanced, supplemental):
        _select(group, selected, 0.42)
    _recover_adjacent(crop, selected)
    _select(refined, selected, 0.42)
    _recover_dark_neighbors(crop, selected)

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
