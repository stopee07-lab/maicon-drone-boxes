"""Inspect aerial UGV body candidates in original image pixels.

This OpenCV heuristic looks for compact dark chassis and bright receiver-like
hardware. It is not a trained general-purpose detector or an exhaustive census.
Coordinates describe image positions, not ground-plane metres.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

VERSION = 'body-ugv-20261119-r4'


def _roi(image, roi):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected a uint8 BGR image')
    h, w = image.shape[:2]
    if h < 24 or w < 24 or h*w > 12_000_000:
        raise ValueError('Unsupported image dimensions')
    bounds = (0, 0, w, h) if roi is None else tuple(roi)
    if (len(bounds) != 4 or any(type(v) is not int for v in bounds)
            or not 0 <= bounds[0] < bounds[2] <= w or not 0 <= bounds[1] < bounds[3] <= h):
        raise ValueError('ROI must be an in-image nonempty integer rectangle')
    return bounds


def _hardware_blobs(gray, bounds, scale):
    """Compact bright receiver-like shapes with dark hardware around them.

    Brightness and shape provide chassis evidence. Several thresholds tolerate
    dome glare boundaries.
    """
    h, w = gray.shape
    xlo, ylo, xhi, yhi = bounds
    found = []
    for threshold in (145, 170, 190):
        mask = np.uint8(gray > threshold)*255
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if not (7*scale*scale <= area <= 140*scale*scale and .45 <= bw/bh <= 1.8
                    and min(bw,bh) >= max(2,3*scale) and max(bw,bh) <= 19*scale
                    and area/(bw*bh) >= .42 and 4*math.pi*area/max(1.,perimeter*perimeter) >= .55):
                continue
            cx, cy, radius = x+bw/2., y+bh/2., max(bw,bh)/2.
            if not xlo < cx < xhi or not max(ylo,10*scale) < cy < min(yhi,h-10*scale):
                continue
            margin = 5*scale
            ax, ay = max(0,int(cx-radius-margin)), max(0,int(cy-radius-margin))
            bx, by = min(w,int(cx+radius+margin+1)), min(h,int(cy+radius+margin+1))
            patch = gray[ay:by,ax:bx]
            yy, xx = np.indices(patch.shape)
            distance = np.hypot(xx+ax-cx,yy+ay-cy)
            ring = (distance > radius+scale) & (distance < radius+margin)
            # Glare can lift an otherwise black body above 85. Require a
            # brightness contrast as well as a relatively dark surround.
            darkness = float(np.mean(patch[ring] < 120)) if ring.any() else 0.
            core = gray[y:y+bh,x:x+bw]
            contrast = float(np.percentile(core,80)-np.median(patch[ring])) if ring.any() else 0.
            if (darkness < (.50 if contrast >= 100 else .60) or contrast < 55
                    or any(math.hypot(cx-a['x'],cy-a['y']) <= 6*scale for a in found)):
                continue
            found.append({'x':cx,'y':cy,'radius':radius,'area_px':float(area),
                          'dark_ring_fraction':round(darkness,3)})
    return found


def _overlap(a, b):
    x0,y0,x1,y1 = a
    u0,v0,u1,v1 = b
    area = max(0,min(x1,u1)-max(x0,u0))*max(0,min(y1,v1)-max(y0,v0))
    return area/max(1,min((x1-x0)*(y1-y0),(u1-u0)*(v1-v0)))


def _round_edge_blobs(gray, bounds, scale):
    """Find solid round highlights that may be surrounded by bright floor."""
    left, top, right, bottom = bounds
    found = []
    for threshold in (170, 190, 210, 145):
        bright = np.uint8(gray > threshold)
        contours = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            small = area < 16 * scale**2
            if threshold == 145 and not small:
                continue
            if not (8 * scale**2 <= area <= 100 * scale**2
                    and (3 if small else 5) * scale <= min(width, height)
                    and max(width, height) <= 14 * scale
                    and .55 <= width / height <= 1.8
                    and 4 * math.pi * area / max(1., perimeter**2) >= (.60 if small else .62)):
                continue
            major, minor = cv2.minAreaRect(contour)[1]
            if (len(cv2.approxPolyDP(contour, .025 * perimeter, True)) < (5 if small else 6)
                    or area / max(1., major * minor) > .9):
                continue
            cx, cy = x + width / 2, y + height / 2
            if not (left < cx < right and top < cy < bottom):
                continue
            if any(math.hypot(cx - b['x'], cy - b['y']) < 5 * scale for b in found):
                continue
            filled = np.zeros((height, width), np.uint8)
            cv2.drawContours(filled, [contour - np.array([[[x, y]]])], -1, 1, -1)
            if bright[y:y+height, x:x+width][filled > 0].mean() < .85:
                continue
            radius = max(width, height) / 2
            extent = radius + 5 * scale
            x0, y0 = max(0, int(cx - extent)), max(0, int(cy - extent))
            x1, y1 = min(gray.shape[1], int(cx + extent + 1)), min(gray.shape[0], int(cy + extent + 1))
            yy, xx = np.ogrid[y0:y1, x0:x1]
            distance = np.hypot(xx - cx, yy - cy)
            ring = (distance > radius + scale) & (distance < extent)
            if not ring.any():
                continue
            darkness = float(np.mean(gray[y0:y1, x0:x1][ring] < 120))
            if darkness >= (.68 if small else .28):
                found.append({'x': cx, 'y': cy, 'radius': radius, 'area_px': float(area),
                              'dark_ring_fraction': round(darkness, 3)})
    return sorted(found, key=lambda blob: blob['x'])


def _anchor_matches(a, b, scale):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y']) <= 5 * scale


def _duplicate(candidate, previous, scale):
    if _overlap(candidate['box'], previous['box']) > .35:
        return True
    # A neighboring floor segment may borrow an anchor from a stronger body.
    # Keep it only if it has support of its own.
    return all(any(_anchor_matches(anchor, used, scale) for used in previous['blobs'])
               for anchor in candidate['blobs'])


def _paired_chassis(gray, bounds, scale, retained):
    """Recover bodies from two supported domes even if the contour hits a box."""
    points = _round_edge_blobs(gray, bounds, scale)
    candidates = []
    for i, a in enumerate(points):
        for b in points[i+1:]:
            if b['x'] - a['x'] > 44 * scale:
                break
            delta = np.array([b['x'] - a['x'], b['y'] - a['y']])
            span = float(np.linalg.norm(delta))
            if not 29 * scale <= span <= 44 * scale:
                continue
            smaller, larger = sorted((a['area_px'], b['area_px']))
            small = smaller < 16 * scale**2
            # A dim or partly hidden receiver must be supported by a stronger
            # receiver and a more consistently dark chassis.
            if (small and larger < 24 * scale**2) or smaller < (.26 if small else .38) * larger:
                continue
            axis = delta / span
            perpendicular = np.array([-axis[1], axis[0]])
            mid = np.array([(a['x'] + b['x']) / 2, (a['y'] + b['y']) / 2])
            along, depth = np.meshgrid(
                np.arange(-span / 2 - 3 * scale, span / 2 + 3 * scale, scale),
                np.arange(3 * scale, 29 * scale, scale),
            )
            for direction in (perpendicular, -perpendicular):
                sample_points = np.rint(mid + along[..., None] * axis + depth[..., None] * direction).astype(int)
                x, y = sample_points[..., 0], sample_points[..., 1]
                left, top, right, bottom = bounds
                if x.min() < left or x.max() >= right or y.min() < top or y.max() >= bottom:
                    continue
                values = gray[y, x]
                dark = values < 100
                darkness = float(dark.mean())
                end_support = float(min(dark[:, :7].mean(), dark[:, -7:].mean()))
                if (darkness < (.62 if small else .50)
                        or end_support < (.18 if small else .08)
                        or values[:, 7:-7].std() < 24):
                    continue
                corners = np.array([mid + u * axis + v * direction
                                    for u in (-span/2 - 8*scale, span/2 + 8*scale)
                                    for v in (-7*scale, 30*scale)])
                x0, y0 = np.floor(corners.min(axis=0)).astype(int)
                x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
                candidates.append({
                    'box': [int(x0), int(y0), int(x1), int(y1)], 'blobs': [a, b],
                    'score': .65, 'support': darkness + .3 * end_support,
                    'method': 'paired_round_domes_local_chassis',
                })
    for candidate in sorted(candidates, key=lambda item: -item['support']):
        if not any(_duplicate(candidate, previous, scale) for previous in retained):
            retained.append(candidate)


def detect_ugvs(image, roi=None):
    """Return UGV body candidates within an optional [x0,y0,x1,y1] ROI.

    Coordinates always refer to the original image. Clipped bodies, unusual
    scale, occlusion and nonstandard hardware can cause missed detections.
    """
    bounds = _roi(image,roi)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    scale = h/720.
    blobs = _hardware_blobs(gray,bounds,scale)
    candidates = []
    opening = max(1, int(round(3*scale)) | 1)
    closing = max(1, int(round(5*scale)) | 1)
    for threshold in (50,65,80,100,120):
        mask = np.uint8(gray < threshold)*255
        x0,y0,x1,y1 = bounds
        mask[:y0] = mask[y1:] = 0
        mask[:,:x0] = mask[:,x1:] = 0
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((opening,opening),np.uint8))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((closing,closing),np.uint8))
        count,_,stats,_ = cv2.connectedComponentsWithStats(mask)
        for label in range(1,count):
            x,y,bw,bh,area = map(int,stats[label])
            if not (20*scale <= bw <= 80*scale and 20*scale <= bh <= 80*scale
                    and area >= 220*scale*scale and .4 <= bw/bh <= 2.5):
                continue
            attached = [a for a in blobs if x-5*scale <= a['x'] <= x+bw+5*scale and y-5*scale <= a['y'] <= y+bh+5*scale]
            if not attached or not any(a['area_px'] >= 20*scale*scale for a in attached):
                continue
            quality = .4 + .11*min(2,len(attached)) + .15*min(1.,area/(bw*bh)) + .12*min(1.,area/(1000*scale*scale))
            # A higher brightness threshold can merge the body into a boundary
            # stripe. Prefer a compact supported body over that inflated region.
            quality -= min(.3, max(0.,bw*bh/(2500*scale*scale)-1.)*.15)
            candidates.append({'box':[x,y,x+bw,y+bh],'score':quality,'blobs':attached,
                               'method':'compact_dark_chassis_with_receiver_like_blobs'})
    retained = []
    for candidate in sorted(candidates,key=lambda item:-item['score']):
        if any(_duplicate(candidate, prior, scale) for prior in retained):
            continue
        retained.append(candidate)
    _paired_chassis(gray, bounds, scale, retained)
    output = []
    for candidate in retained:
        raw = candidate['box']
        pad = max(2,int(round(4*scale)))
        bbox = [max(bounds[0],raw[0]-pad),max(bounds[1],raw[1]-pad),min(bounds[2],raw[2]+pad),min(bounds[3],raw[3]+pad)]
        output.append({'bbox_px':bbox,'center_px':[(bbox[0]+bbox[2])/2.,(bbox[1]+bbox[3])/2.],
                       'confidence':round(min(.89,candidate['score']),3),
                       'detection_evidence':{'method':candidate['method'],
                                             'hardware_blob_count':len(candidate['blobs']),
                                             'hardware_blobs':candidate['blobs'],
                                             'complete_census':False,'version':VERSION}})
    return sorted(output,key=lambda d:(d['center_px'][1],d['center_px'][0]))
