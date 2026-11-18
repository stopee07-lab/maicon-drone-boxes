"""Read the latest JPEG from the official MAICON aerial camera endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from http.client import HTTPException
from math import isfinite
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MAX_JPEG_BYTES = 16 * 1024 * 1024
CAMERAS = ("main", "test1", "test2")


class FeedError(RuntimeError):
    """A frame could not be read or validated; messages exclude credentials."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class DroneFrame:
    jpeg: bytes
    camera: str
    captured_at: datetime | None
    frame_interval_s: float
    content_sha256: str


def _capture_time(value: str) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        captured = datetime.fromisoformat(value)
    except ValueError:
        return None
    return captured if captured.utcoffset() is not None else None


def _media_type(value: str) -> str:
    return value.partition(";")[0].strip().lower()


def decode_frame(jpeg: bytes, headers: Mapping[str, str], expected_camera: str | None = None) -> DroneFrame:
    """Validate the payload envelope and interpret frame metadata.

    Image decoding and dimension validation remain with the image detector.
    Request time is never substituted for a missing or ambiguous capture time.
    """
    if not isinstance(jpeg, bytes) or not 4 < len(jpeg) <= MAX_JPEG_BYTES:
        raise FeedError("Drone JPEG size is invalid.")
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise FeedError("Drone frame is not a complete JPEG.")

    metadata = {name.lower(): value.strip() for name, value in headers.items()}
    if "content-type" in metadata and _media_type(metadata["content-type"]) != "image/jpeg":
        raise FeedError("Drone feed did not return image/jpeg.")
    camera = metadata.get("x-camera") or "unknown"
    if expected_camera is not None and camera != expected_camera:
        raise FeedError("Drone camera header does not match the requested camera.")

    try:
        interval = float(metadata.get("x-frame-interval-s", "1"))
    except ValueError:
        interval = 1.0
    if not isfinite(interval) or interval <= 0.0:
        interval = 1.0

    return DroneFrame(
        jpeg=jpeg,
        camera=camera,
        captured_at=_capture_time(metadata.get("x-captured-at", "")),
        frame_interval_s=interval,
        content_sha256=sha256(jpeg).hexdigest(),
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DroneFeed:
    def __init__(self, origin: str, mission_code: str, camera: str | None = None, timeout: float = 3.0):
        try:
            url = urlsplit(origin)
            valid_origin = (
                url.scheme in ("http", "https")
                and bool(url.hostname)
                and url.username is None
                and url.password is None
                and url.path in ("", "/")
                and not url.query
                and not url.fragment
                and (url.port is None or 1 <= url.port <= 65535)
                and not any(ch.isspace() for ch in origin)
            )
        except (AttributeError, TypeError, ValueError):
            valid_origin = False
        if not valid_origin:
            raise FeedError("Drone feed origin must be an HTTP(S) host and optional port.")
        if not isinstance(mission_code, str) or not mission_code.strip():
            raise FeedError("A mission code is required.")
        if camera is not None and camera not in CAMERAS:
            raise FeedError("Camera must be main, test1 or test2, or omitted for a match.")
        if not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
            raise FeedError("Drone feed timeout must be positive and finite.")

        parameters = {"mission_code": mission_code.strip()}
        if camera is not None:
            parameters["camera"] = camera
        self._url = origin.rstrip("/") + "/api/report/drone?" + urlencode(parameters)
        self._expected_camera = camera or "main"
        self._timeout = timeout
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def fetch(self) -> DroneFrame:
        request = Request(self._url, headers={"Accept": "image/jpeg"}, method="GET")
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                if _media_type(response.headers.get("Content-Type", "")) != "image/jpeg":
                    raise FeedError("Drone feed did not return image/jpeg.")
                jpeg = response.read(MAX_JPEG_BYTES + 1)
                headers = response.headers
        except HTTPError as exc:
            exc.close()
            raise FeedError(f"Drone feed HTTP {exc.code}.", status=exc.code) from None
        except (OSError, HTTPException):
            raise FeedError("Could not receive a drone frame.") from None

        return decode_frame(jpeg, headers, self._expected_camera)
