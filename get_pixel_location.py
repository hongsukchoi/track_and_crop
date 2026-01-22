#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import viser


def load_first_frame_rgb(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from: {video_path}")

    # Convert OpenCV BGR -> RGB for viser.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to a video file (mp4/mov/avi/...)")
    parser.add_argument("--port", type=int, default=8080, help="Port for viser server")
    args = parser.parse_args()

    video_path = str(Path(args.video).expanduser())
    frame_rgb = load_first_frame_rgb(video_path)
    H, W = frame_rgb.shape[:2]

    server = viser.ViserServer(port=args.port)

    # Show the first frame as a full-viewport background image.
    # Expects shape (H, W, 3). :contentReference[oaicite:1]{index=1}
    server.scene.set_background_image(frame_rgb)
    # Hide other scene nodes so you only see the background image. :contentReference[oaicite:2]{index=2}
    server.scene.set_global_visibility(False)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # A simple GUI readout (it's an editable text box, but works fine as a display).
        readout = client.gui.add_text("Pixel (x, y)", initial_value="Click on the image")

        # Scene pointer events: click gives screen_pos in normalized OpenCV image coords:
        # (0,0)=upper-left, (1,1)=bottom-right. :contentReference[oaicite:3]{index=3}
        @client.scene.on_pointer_event(event_type="click")
        def _(event: viser.ScenePointerEvent) -> None:
            # screen_pos is (x_norm, y_norm) in [0,1]. :contentReference[oaicite:4]{index=4}
            x_norm, y_norm = event.screen_pos[0]  # type: ignore[misc]
            x = int(round(x_norm * (W - 1)))
            y = int(round(y_norm * (H - 1)))
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))

            # Also fetch the pixel value for convenience.
            rgb = frame_rgb[y, x].tolist()
            readout.value = f"({x}, {y})   RGB={tuple(rgb)}"

            print(f"[client {event.client_id}] pixel=({x},{y}) rgb={tuple(rgb)}")

    print(f"Viser running on http://localhost:{args.port}")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
