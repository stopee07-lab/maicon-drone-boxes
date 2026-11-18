"""Capture one official aerial feed and record box and UGV observations."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import getpass
import json
import os
from pathlib import Path
import sys
import time
import uuid
import warnings

import cv2
import numpy as np

from box_detector import detect_boxes
from drone_feed import DroneFeed, FeedError
from ugv_detector import detect_ugvs
from ugv_tracking import UGVTracker


SEOUL = timezone(timedelta(hours=9))


def mission_code():
    code = os.environ.get("MAICON_MISSION_CODE", "").strip()
    if not code and sys.stdin.isatty():
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                code = getpass.getpass("MAICON_MISSION_CODE (숨김 입력): ").strip()
            except getpass.GetPassWarning:
                raise ValueError("이 터미널에서는 환경 변수로 임무코드를 설정하세요.") from None
    if not code:
        raise ValueError("MAICON_MISSION_CODE를 환경 변수 또는 터미널 숨김 입력으로 제공하세요.")
    return code


def write_atomic(path, data):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def output_report(value, path=None):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if path is None:
        print(raw, end="", flush=True)
    else:
        write_atomic(path, raw.encode("utf-8"))


def decode_image(jpeg):
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] * image.shape[1] > 12_000_000:
        raise ValueError("이미지를 읽을 수 없거나 입력 해상도가 너무 큽니다.")
    return image


def analyze_image(image, roi=None):
    boxes = detect_boxes(image, roi)
    observations = [
        {"bbox_px": list(box.bbox_px), "center_px": list(box.center_px), "score": round(box.confidence, 3)}
        for box in boxes
    ]
    ugvs = detect_ugvs(image, roi=roi)
    return {"image_size_px": [image.shape[1], image.shape[0]],
            "observed_count": len(observations), "observations": observations,
            "ugv_count": len(ugvs), "ugvs": ugvs}


def analyze_jpeg(jpeg, roi=None):
    """Decode a JPEG and return box and UGV counts and pixel coordinates."""
    return analyze_image(decode_image(jpeg), roi)


def render_ugv_overlay(image, ugvs, path, roi=None):
    """Draw UGV locations and their visual track IDs."""
    height, width = image.shape[:2]
    left, top, right, bottom = roi if roi is not None else (0, 0, width, height)
    canvas = image[top:bottom, left:right].copy()
    color = (0, 210, 255)
    for index, ugv in enumerate(ugvs, 1):
        x1, y1, x2, y2 = map(round, ugv['bbox_px'])
        cv2.rectangle(canvas, (x1-left, y1-top), (x2-left-1, y2-top-1), color, 2)
        title = ugv.get('track_id') or f'UGV {index}'
        label_position = (max(0, x1-left), max(14, y1-top-6))
        cv2.putText(canvas, title, label_position, cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 3)
        cv2.putText(canvas, title, label_position, cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
    ok, data = cv2.imencode('.png', canvas)
    if not ok:
        raise ValueError('Could not encode UGV overlay')
    write_atomic(path, data.tobytes())


def render_overlay(image, observations, path, roi):
    for number, box in enumerate(observations, 1):
        x1, y1, x2, y2 = box["bbox_px"]
        cx, cy = map(round, box["center_px"])
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (70, 240, 40), 2)
        cv2.circle(image, (cx, cy), 3, (0, 220, 255), -1)
        cv2.putText(image, str(number), (x1 + 3, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, .49, (70, 240, 40), 1)
    if roi is not None:
        x1, y1, x2, y2 = roi
        image = image[y1:y2, x1:x2]
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("검출 표시 이미지를 저장할 수 없습니다.")
    write_atomic(path, data.tobytes())


def frame_state(frame, received_at, last_capture, unknown_time_hashes):
    captured_at = frame.captured_at
    if captured_at is None:
        return "duplicate" if frame.content_sha256 in unknown_time_hashes else "unknown_capture_time"
    if last_capture is not None and captured_at <= last_capture:
        return "duplicate" if captured_at == last_capture else "out_of_order"
    age_s = (received_at - captured_at).total_seconds()
    maximum_age_s = max(2.5, min(12.0, 2.25 * frame.frame_interval_s))
    return "fresh" if -1 <= age_s <= maximum_age_s else "outdated"


def save_frame(directory, number, frame, received_at, state, roi, *, tracker=None):
    base = f"frame-{number:04d}"
    jpeg_path, overlay_path = directory / f"{base}.jpg", directory / f"{base}.png"
    write_atomic(jpeg_path, frame.jpeg)
    record = {
        "camera": frame.camera,
        "captured_at": frame.captured_at.isoformat() if frame.captured_at is not None else None,
        "received_at": received_at.isoformat(),
        "frame_interval_s": frame.frame_interval_s,
        "frame_state": state,
        "sha256": frame.content_sha256,
        "jpeg": jpeg_path.name,
        "overlay": None,
    }
    try:
        image = decode_image(frame.jpeg)
        analysis = analyze_image(image, roi)
        if tracker is not None and state == 'fresh' and frame.captured_at is not None:
            tracked = tracker.update(analysis['ugvs'],frame.captured_at.timestamp(),frame.content_sha256,
                                     camera=frame.camera)
            analysis['tracking'] = {k:v for k,v in tracked.items() if k!='ugvs'}
            if tracked['accepted']:
                analysis['ugvs'] = tracked['ugvs']
        ugv_overlay = directory / f'{base}-ugvs.png'
        render_ugv_overlay(image,analysis['ugvs'],ugv_overlay,roi)
        render_overlay(image, analysis["observations"], overlay_path, roi)
        record.update(analysis)
        record["overlay"] = overlay_path.name
        record['ugv_overlay'] = ugv_overlay.name
    except (ValueError, cv2.error) as error:
        record.update(observed_count=None, observations=[], ugv_count=None, ugvs=[],
                      analysis_error=str(error))
    return record


def run_capture(feed, directory, *, mode, camera, frames, seconds, roi=None):
    directory.mkdir(parents=True, exist_ok=False)
    stats = Counter()
    last_capture = None
    tracker = UGVTracker()
    unknown_time_hashes = set()
    started = time.monotonic()
    deadline = started + seconds
    exit_reason = "time_limit"
    summary = {"mode": mode, "camera": camera, "requested_frames": frames, "started_at": datetime.now(SEOUL).isoformat()}

    try:
        with (directory / "frames.jsonl").open("w", encoding="utf-8") as manifest:
            while stats["saved_frames"] < frames and time.monotonic() < deadline:
                interval = 1.0
                try:
                    frame = feed.fetch()
                except FeedError as error:
                    name = f"http_{error.status}" if error.status is not None else "feed_error"
                    stats[name] += 1
                    if error.status in (401, 403, 422) or (error.status == 409 and mode == "practice"):
                        exit_reason = name
                        break
                else:
                    received = datetime.now(SEOUL)
                    interval = max(1.0, min(5.0, frame.frame_interval_s))
                    state = frame_state(frame, received, last_capture, unknown_time_hashes)
                    if state in ("fresh", "unknown_capture_time"):
                        record = save_frame(directory, stats["saved_frames"] + 1, frame, received, state, roi,
                                            tracker=tracker)
                        manifest.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        manifest.flush()
                        stats["saved_frames"] += 1
                        stats["analyzed_frames"] += int(record["observed_count"] is not None)
                        stats[state] += 1
                        if frame.captured_at is not None:
                            last_capture = frame.captured_at
                        else:
                            unknown_time_hashes.add(frame.content_sha256)
                    else:
                        stats[state] += 1
                if stats["saved_frames"] < frames:
                    time.sleep(max(0.0, min(interval, deadline - time.monotonic())))
            if stats["saved_frames"] >= frames:
                exit_reason = "frame_limit"
    except KeyboardInterrupt:
        exit_reason = "interrupted"
    except OSError:
        exit_reason = "storage_error"
        raise
    finally:
        summary.update({"saved_frames": stats["saved_frames"], "analyzed_frames": stats["analyzed_frames"],
                        "stop_reason": exit_reason, "counters": dict(stats),
                        "elapsed_s": round(time.monotonic() - started, 2)})
        output_report(summary, directory / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="상공 JPEG 수집과 상자·UGV 검출")
    parser.add_argument("--mode", choices=("practice", "match"), default="practice")
    parser.add_argument("--camera", choices=("test1", "test2"), help="연습장 영상 채널")
    parser.add_argument("--origin", required=True, help="카메라 서버의 http(s)://호스트:포트")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--output", type=Path, default=Path("captures"))
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()
    if args.mode == "match" and args.camera is not None:
        parser.error("본경기에는 --camera를 생략하세요. main을 자동으로 요청합니다.")
    if not (1 <= args.frames <= 500 and 1 <= args.seconds <= 600):
        parser.error("--frames는 1~500, --seconds는 1~600입니다.")
    roi = tuple(args.roi) if args.roi else None
    if roi and not (0 <= roi[0] < roi[2] and 0 <= roi[1] < roi[3]):
        parser.error("ROI를 왼쪽 위와 오른쪽 아래 모서리의 픽셀 좌표로 지정하세요.")
    camera = (args.camera or "test1") if args.mode == "practice" else None
    label = camera or "main"
    folder = args.output / f"{datetime.now(SEOUL):%Y%m%d-%H%M%S}-{label}-{uuid.uuid4().hex[:6]}"
    try:
        feed = DroneFeed(args.origin, mission_code(), camera=camera)
        summary = run_capture(feed, folder, mode=args.mode, camera=label, frames=args.frames,
                              seconds=args.seconds, roi=roi)
        output_report({"directory": str(folder.resolve()), **summary}, None)
        return int(summary["analyzed_frames"] != args.frames)
    except (OSError, ValueError, KeyError, TypeError, FeedError, cv2.error) as error:
        print(f"상공 영상 수집 오류: {error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        return 130


if __name__ == "__main__":
    sys.exit(main())
