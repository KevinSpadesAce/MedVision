# security_controls.py

import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

from PIL import Image

class RateLimitError(Exception):
    pass


class ResourceBusyError(Exception):
    pass


class FileSecurityError(Exception):
    pass


RATE_LIMIT_WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 20

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 8_000_000

MAX_CONCURRENT_INFERENCES = 3

_request_history = defaultdict(deque)
_rate_limit_lock = threading.Lock()

_inference_semaphore = threading.BoundedSemaphore(value=MAX_CONCURRENT_INFERENCES)

def get_client_identifier(req) -> str:
    forwarded_for = req.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return client_ip

    remote_addr = getattr(req, "remote_addr", None)
    if remote_addr:
        return remote_addr

    return "unknown"


def check_rate_limit(client_id: str) -> None:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        timestamps = _request_history[client_id]

        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= MAX_REQUESTS_PER_WINDOW:
            raise RateLimitError(
                f"Too many requests from this client. "
                f"Limit: {MAX_REQUESTS_PER_WINDOW} requests per "
                f"{RATE_LIMIT_WINDOW_SECONDS} seconds."
            )

        timestamps.append(now)


def check_file_size(file_bytes: bytes) -> None:
    if file_bytes is None:
        raise FileSecurityError("Uploaded file content is missing.")

    file_size = len(file_bytes)

    if file_size == 0:
        raise FileSecurityError("Uploaded file is empty.")

    if file_size > MAX_UPLOAD_BYTES:
        raise FileSecurityError(
            f"Uploaded file is too large. "
            f"Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )


def check_image_dimensions(image: Image.Image) -> None:
    if image is None:
        raise FileSecurityError("Image object is missing.")

    width, height = image.size
    total_pixels = width * height

    if width <= 0 or height <= 0:
        raise FileSecurityError("Invalid image dimensions.")

    if width > MAX_IMAGE_WIDTH:
        raise FileSecurityError(
            f"Image width exceeds limit ({width}px > {MAX_IMAGE_WIDTH}px)."
        )

    if height > MAX_IMAGE_HEIGHT:
        raise FileSecurityError(
            f"Image height exceeds limit ({height}px > {MAX_IMAGE_HEIGHT}px)."
        )

    if total_pixels > MAX_IMAGE_PIXELS:
        raise FileSecurityError(
            f"Image pixel count exceeds limit "
            f"({total_pixels} > {MAX_IMAGE_PIXELS})."
        )


def run_with_inference_limit(func: Callable[..., Any], *args, **kwargs) -> Any:
    acquired = _inference_semaphore.acquire(blocking=False)

    if not acquired:
        raise ResourceBusyError(
            "Inference service is currently busy. Please try again shortly."
        )

    try:
        return func(*args, **kwargs)
    finally:
        _inference_semaphore.release()