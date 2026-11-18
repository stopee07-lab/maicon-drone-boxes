from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from capture_boxes import frame_state, run_capture
from drone_feed import decode_frame, FeedError


class LocalFeed:
    def __init__(self, results):
        self.results = iter(results)

    def fetch(self):
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def jpeg():
    image = np.full((720, 1280, 3), 130, np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


class CaptureTest(unittest.TestCase):
    def test_frame_time_classification(self):
        now = datetime.now(timezone.utc)
        frame = decode_frame(jpeg(), {"X-Camera": "test1", "X-Captured-At": now.isoformat()}, "test1")
        self.assertEqual(frame_state(frame, now, None, set()), "fresh")
        self.assertEqual(frame_state(frame, now, now, set()), "duplicate")
        self.assertEqual(frame_state(frame, now, now + timedelta(seconds=1), set()), "out_of_order")
        self.assertEqual(frame_state(frame, now + timedelta(seconds=4), None, set()), "outdated")
        unknown = replace(frame, captured_at=None)
        self.assertEqual(frame_state(unknown, now, None, set()), "unknown_capture_time")
        self.assertEqual(frame_state(unknown, now, None, {frame.content_sha256}), "duplicate")

    def test_practice_capture_saves_original_and_skips_duplicate(self):
        stamp = datetime.now(timezone.utc)
        frame = decode_frame(jpeg(), {"X-Camera": "test1", "X-Captured-At": stamp.isoformat()}, "test1")
        next_capture = replace(frame, captured_at=stamp + timedelta(milliseconds=300))
        with TemporaryDirectory() as root, patch("capture_boxes.time.sleep"):
            out = Path(root) / "capture"
            summary = run_capture(LocalFeed([frame, frame, next_capture]), out, mode="practice",
                                  camera="test1", frames=2, seconds=6)
            rows = [json.loads(line) for line in (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["saved_frames"], 2)
            self.assertEqual(summary["counters"]["duplicate"], 1)
            self.assertEqual([row["observed_count"] for row in rows], [0, 0])
            self.assertTrue(all((out / row["jpeg"]).read_bytes() == frame.jpeg for row in rows))
            self.assertTrue(all((out / row["overlay"]).exists() for row in rows))
            self.assertTrue(all((out / row['ugv_overlay']).exists() for row in rows))
            self.assertTrue(all(row['ugv_count'] == 0 and row['ugvs'] == [] for row in rows))

    def test_main_capture_can_wait_for_match_phase(self):
        stamp = datetime.now(timezone.utc)
        frame = decode_frame(jpeg(), {"X-Camera": "main", "X-Captured-At": stamp.isoformat()}, "main")
        with TemporaryDirectory() as root, patch("capture_boxes.time.sleep"):
            out = Path(root) / "capture"
            summary = run_capture(LocalFeed([FeedError("Drone feed HTTP 409.", status=409), frame]), out,
                                  mode="match", camera="main", frames=1, seconds=5)
            self.assertEqual(summary["stop_reason"], "frame_limit")
            self.assertEqual(summary["counters"]["http_409"], 1)
            self.assertEqual(summary["saved_frames"], 1)


if __name__ == "__main__":
    unittest.main()
