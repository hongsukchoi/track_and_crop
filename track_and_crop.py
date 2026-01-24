#!/usr/bin/env python3
"""
Track an object in a video using SAM2 and create a cropped video around the tracked object.

Given a video and a bounding box of an object at the first frame, this script:
1. Uses SAM2 to track the object throughout the video
2. Computes a consistent crop region with fixed aspect ratio
3. Creates a new video with cropped frames centered on the tracked object
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Add thirdparty directory to path for sam2
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "thirdparty" / "sam2"))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


def has_audio_stream(video_path: str) -> bool:
    """Check if video has an audio stream using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        return "audio" in result.stdout
    except Exception:
        return False


def convert_to_h264(video_path: str, output_path: str) -> bool:
    """
    Convert video to H.264 codec for browser compatibility.

    Args:
        video_path: Path to input video file
        output_path: Path to output video

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i", video_path,
                "-c:v", "libx264",  # H.264 codec for Chrome compatibility
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",  # Required for browser compatibility
                "-movflags", "+faststart",  # Enable streaming
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Warning: Failed to convert to H.264: {e}")
        return False


def merge_audio(video_path: str, audio_source: str, output_path: str) -> bool:
    """
    Merge video with audio from another source using ffmpeg.
    Re-encodes video to H.264 for browser compatibility.

    Args:
        video_path: Path to video file (without audio)
        audio_source: Path to source video to copy audio from
        output_path: Path to output video with audio

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use ffmpeg to combine video with audio
        # -c:v libx264: encode to H.264 for browser compatibility
        # -c:a aac: encode audio as AAC for compatibility
        # -map 0:v: take video from first input
        # -map 1:a: take audio from second input
        # -shortest: end when shortest stream ends
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i", video_path,  # Input video (no audio)
                "-i", audio_source,  # Input for audio
                "-c:v", "libx264",  # H.264 codec for Chrome compatibility
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",  # Required for browser compatibility
                "-c:a", "aac",  # Encode audio as AAC
                "-map", "0:v:0",  # Take video from first input
                "-map", "1:a:0?",  # Take audio from second input (optional)
                "-shortest",  # End at shortest stream
                "-movflags", "+faststart",  # Enable streaming
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Warning: Failed to merge audio: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track an object and create a cropped video around it"
    )
    parser.add_argument(
        "--video", "-v", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--bbox",
        "-b",
        type=str,
        required=True,
        help="Bounding box as 'x1,y1,x2,y2' (top-left and bottom-right corners)",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to output video"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large",
        choices=["tiny", "small", "base_plus", "large"],
        help="SAM2 model size (default: large)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing SAM2 checkpoints",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.5,
        help="Scale factor for crop region relative to object size (default: 1.5)",
    )
    parser.add_argument(
        "--output-size",
        type=str,
        default=None,
        help="Output video size as 'WxH' (default: auto-determined from crop)",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default=None,
        help="Force aspect ratio as 'W:H' (e.g., '16:9', '1:1'). Default: use initial bbox aspect ratio",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS (default: same as input)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="EMA smoothing factor for camera motion (0.0=none, 0.9=very smooth). Default: 0.0",
    )
    return parser.parse_args()


def get_model_config(model_size: str) -> tuple[str, str]:
    """Get model config path and checkpoint filename for given model size."""
    configs = {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "base_plus": (
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_base_plus.pt",
        ),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    }
    return configs[model_size]


def parse_bbox(bbox_str: str) -> np.ndarray:
    """Parse bounding box string 'x1,y1,x2,y2' to numpy array."""
    coords = [int(x.strip()) for x in bbox_str.split(",")]
    if len(coords) != 4:
        raise ValueError("Bounding box must have exactly 4 values: x1,y1,x2,y2")
    return np.array(coords, dtype=np.float32)


def parse_aspect_ratio(aspect_str: str | None) -> float | None:
    """Parse aspect ratio string 'W:H' to float."""
    if aspect_str is None:
        return None
    parts = aspect_str.split(":")
    if len(parts) != 2:
        raise ValueError("Aspect ratio must be in format 'W:H' (e.g., '16:9')")
    return float(parts[0]) / float(parts[1])


def parse_output_size(size_str: str | None) -> tuple[int, int] | None:
    """Parse output size string 'WxH' to tuple."""
    if size_str is None:
        return None
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError("Output size must be in format 'WxH' (e.g., '640x480')")
    return int(parts[0]), int(parts[1])


def get_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Get bounding box from a binary mask. Returns (x1, y1, x2, y2) or None if empty."""
    if mask.sum() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2), int(y2)


def compute_crop_region(
    bbox: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    target_aspect_ratio: float,
    scale: float = 1.5,
) -> tuple[int, int, int, int]:
    """
    Compute crop region centered on bbox with target aspect ratio.

    Args:
        bbox: (x1, y1, x2, y2) bounding box of object
        frame_size: (width, height) of frame
        target_aspect_ratio: width/height ratio for crop
        scale: scale factor relative to object size

    Returns:
        (x1, y1, x2, y2) crop region
    """
    x1, y1, x2, y2 = bbox
    frame_w, frame_h = frame_size

    # Object center and size
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    obj_w = x2 - x1
    obj_h = y2 - y1

    # Scale up the object size
    crop_w = obj_w * scale
    crop_h = obj_h * scale

    # Adjust to match target aspect ratio
    current_aspect = crop_w / crop_h
    if current_aspect > target_aspect_ratio:
        # Too wide, increase height
        crop_h = crop_w / target_aspect_ratio
    else:
        # Too tall, increase width
        crop_w = crop_h * target_aspect_ratio

    # Compute crop bounds centered on object
    crop_x1 = cx - crop_w / 2
    crop_y1 = cy - crop_h / 2
    crop_x2 = cx + crop_w / 2
    crop_y2 = cy + crop_h / 2

    # Clamp to frame bounds while maintaining aspect ratio
    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > frame_w:
        crop_x1 -= crop_x2 - frame_w
        crop_x2 = frame_w
    if crop_y2 > frame_h:
        crop_y1 -= crop_y2 - frame_h
        crop_y2 = frame_h

    # Final clamp
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(frame_w, crop_x2)
    crop_y2 = min(frame_h, crop_y2)

    return int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)


def compute_unified_crop_params(
    bboxes: list[tuple[int, int, int, int] | None],
    target_aspect_ratio: float,
    scale: float = 1.5,
) -> tuple[int, int]:
    """
    Compute unified crop dimensions that work for all frames.

    Returns the maximum crop width and height needed across all frames.
    """
    max_w = 0
    max_h = 0

    for bbox in bboxes:
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        obj_w = x2 - x1
        obj_h = y2 - y1

        crop_w = obj_w * scale
        crop_h = obj_h * scale

        # Adjust to match target aspect ratio
        current_aspect = crop_w / crop_h
        if current_aspect > target_aspect_ratio:
            crop_h = crop_w / target_aspect_ratio
        else:
            crop_w = crop_h * target_aspect_ratio

        max_w = max(max_w, crop_w)
        max_h = max(max_h, crop_h)

    return int(max_w), int(max_h)


def smooth_bbox_centers(
    frame_bboxes: dict[int, tuple[int, int, int, int]],
    total_frames: int,
    alpha: float,
) -> dict[int, tuple[float, float]]:
    """
    Apply EMA smoothing to bounding box centers.

    Args:
        frame_bboxes: dict mapping frame_idx to (x1, y1, x2, y2)
        total_frames: total number of frames in video
        alpha: EMA smoothing factor (0 = no smoothing, higher = more smooth)

    Returns:
        dict mapping frame_idx to smoothed (cx, cy) center coordinates
    """
    if alpha <= 0:
        # No smoothing, return original centers
        return {
            idx: ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            for idx, bbox in frame_bboxes.items()
        }

    # Get sorted frame indices
    sorted_indices = sorted(frame_bboxes.keys())

    # Extract centers
    centers = {}
    for idx in sorted_indices:
        bbox = frame_bboxes[idx]
        centers[idx] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    # Forward pass EMA
    smoothed_forward = {}
    prev_cx, prev_cy = None, None
    for idx in range(total_frames):
        if idx in centers:
            cx, cy = centers[idx]
            if prev_cx is None:
                smoothed_forward[idx] = (cx, cy)
            else:
                smoothed_forward[idx] = (
                    alpha * prev_cx + (1 - alpha) * cx,
                    alpha * prev_cy + (1 - alpha) * cy,
                )
            prev_cx, prev_cy = smoothed_forward[idx]
        elif prev_cx is not None:
            # No bbox for this frame, keep previous smoothed value
            smoothed_forward[idx] = (prev_cx, prev_cy)

    # Backward pass EMA for bidirectional smoothing
    smoothed_backward = {}
    prev_cx, prev_cy = None, None
    for idx in range(total_frames - 1, -1, -1):
        if idx in centers:
            cx, cy = centers[idx]
            if prev_cx is None:
                smoothed_backward[idx] = (cx, cy)
            else:
                smoothed_backward[idx] = (
                    alpha * prev_cx + (1 - alpha) * cx,
                    alpha * prev_cy + (1 - alpha) * cy,
                )
            prev_cx, prev_cy = smoothed_backward[idx]
        elif prev_cx is not None:
            smoothed_backward[idx] = (prev_cx, prev_cy)

    # Average forward and backward passes
    smoothed = {}
    for idx in smoothed_forward:
        if idx in smoothed_backward:
            fx, fy = smoothed_forward[idx]
            bx, by = smoothed_backward[idx]
            smoothed[idx] = ((fx + bx) / 2, (fy + by) / 2)
        else:
            smoothed[idx] = smoothed_forward[idx]

    return smoothed


def crop_frame_at_center(
    frame: np.ndarray,
    center: tuple[float, float],
    crop_size: tuple[int, int],
) -> np.ndarray:
    """
    Crop frame centered at a specific point with fixed crop size.
    Pads with white if crop extends beyond frame boundaries.
    """
    cx, cy = center
    crop_w, crop_h = crop_size
    frame_h, frame_w = frame.shape[:2]

    # Crop bounds
    crop_x1 = int(cx - crop_w / 2)
    crop_y1 = int(cy - crop_h / 2)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    # Create output with white padding
    output = np.full((crop_h, crop_w, 3), 255, dtype=frame.dtype)

    # Compute valid regions
    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(frame_w, crop_x2)
    src_y2 = min(frame_h, crop_y2)

    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    output[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    return output


def crop_frame_centered(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    crop_size: tuple[int, int],
) -> np.ndarray:
    """
    Crop frame centered on bbox with fixed crop size.
    Pads with white if crop extends beyond frame boundaries.
    """
    x1, y1, x2, y2 = bbox
    crop_w, crop_h = crop_size
    frame_h, frame_w = frame.shape[:2]

    # Object center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Crop bounds
    crop_x1 = int(cx - crop_w / 2)
    crop_y1 = int(cy - crop_h / 2)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    # Create output with white padding
    output = np.full((crop_h, crop_w, 3), 255, dtype=frame.dtype)

    # Compute valid regions
    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(frame_w, crop_x2)
    src_y2 = min(frame_h, crop_y2)

    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    output[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    return output


def extract_frames_to_dir(video_path: str, output_dir: str) -> tuple[int, int, float]:
    """
    Extract frames from video to a directory as JPEG files.
    SAM2 requires frames as individual image files.

    Returns: (width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting {total_frames} frames...")
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        # SAM2 expects frames named as sequential numbers
        frame_path = os.path.join(output_dir, f"{i:06d}.jpg")
        cv2.imwrite(frame_path, frame)

    cap.release()
    return width, height, fps


def main():
    args = parse_args()

    # Parse inputs
    initial_bbox = parse_bbox(args.bbox)
    target_aspect_ratio = parse_aspect_ratio(args.aspect_ratio)
    output_size = parse_output_size(args.output_size)

    # If no aspect ratio specified, use initial bbox aspect ratio
    if target_aspect_ratio is None:
        bbox_w = initial_bbox[2] - initial_bbox[0]
        bbox_h = initial_bbox[3] - initial_bbox[1]
        target_aspect_ratio = bbox_w / bbox_h
        print(f"Using initial bbox aspect ratio: {target_aspect_ratio:.3f}")

    # Get model config
    model_cfg, checkpoint_name = get_model_config(args.model_size)
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please download SAM2 checkpoints first:")
        print(f"  cd {args.checkpoint_dir} && ./download_ckpts.sh")
        return 1

    print(f"Loading SAM2 model ({args.model_size})...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=args.device)

    # Extract frames to temporary directory (SAM2 requires frame directory)
    with tempfile.TemporaryDirectory() as temp_dir:
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)

        # Extract frames
        frame_w, frame_h, video_fps = extract_frames_to_dir(args.video, frames_dir)
        fps = args.fps if args.fps else video_fps

        print("Initializing SAM2 video predictor...")
        inference_state = predictor.init_state(video_path=frames_dir)

        # Add initial bounding box prompt at frame 0
        print("Adding initial bounding box prompt...")
        _, _, masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=initial_bbox,
        )

        # Propagate through video and collect masks
        print("Tracking object through video...")
        frame_masks = {}

        # Store initial frame mask
        if len(masks) > 0:
            frame_masks[0] = (masks[0] > 0).cpu().numpy().squeeze()

        # Propagate forward
        for frame_idx, _, masks in tqdm(
            predictor.propagate_in_video(inference_state),
            desc="Propagating",
        ):
            if len(masks) > 0:
                mask = (masks[0] > 0).cpu().numpy().squeeze()
                frame_masks[frame_idx] = mask

        # Get bounding boxes from masks
        print("Computing bounding boxes from masks...")
        frame_bboxes = {}
        for frame_idx, mask in frame_masks.items():
            bbox = get_mask_bbox(mask)
            if bbox is not None:
                frame_bboxes[frame_idx] = bbox
            else:
                # Use previous frame's bbox if mask is empty
                if frame_idx > 0 and (frame_idx - 1) in frame_bboxes:
                    frame_bboxes[frame_idx] = frame_bboxes[frame_idx - 1]

        if not frame_bboxes:
            print("Error: No valid masks found. Check your bounding box input.")
            return 1

        # Get total frames for smoothing
        cap_temp = cv2.VideoCapture(args.video)
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()

        # Apply EMA smoothing to bounding box centers
        if args.smoothing > 0:
            print(f"Applying EMA smoothing (alpha={args.smoothing})...")
        smoothed_centers = smooth_bbox_centers(frame_bboxes, total_frames, args.smoothing)

        # Compute unified crop dimensions
        print("Computing unified crop dimensions...")
        crop_w, crop_h = compute_unified_crop_params(
            list(frame_bboxes.values()),
            target_aspect_ratio,
            args.crop_scale,
        )

        # Ensure crop doesn't exceed frame size
        crop_w = min(crop_w, frame_w)
        crop_h = min(crop_h, frame_h)

        print(f"Crop size: {crop_w}x{crop_h}")

        # Determine output size
        if output_size:
            out_w, out_h = output_size
        else:
            out_w, out_h = crop_w, crop_h

        print(f"Output size: {out_w}x{out_h}")

        # Create output video
        print("Creating output video...")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then merge audio
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_w, out_h))

        # Read original video and create cropped frames
        cap = cv2.VideoCapture(args.video)

        last_valid_center = None
        for frame_idx in tqdm(range(total_frames), desc="Writing output"):
            ret, frame = cap.read()
            if not ret:
                break

            # Get smoothed center for this frame
            if frame_idx in smoothed_centers:
                center = smoothed_centers[frame_idx]
                last_valid_center = center
            elif last_valid_center is not None:
                center = last_valid_center
            else:
                # Skip frame if no center available
                continue

            # Crop frame using smoothed center
            cropped = crop_frame_at_center(frame, center, (crop_w, crop_h))

            # Resize if needed
            if (out_w, out_h) != (crop_w, crop_h):
                cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            out.write(cropped)

        cap.release()
        out.release()

        # Convert to H.264 and merge audio from original video
        if check_ffmpeg():
            if has_audio_stream(args.video):
                print("Converting to H.264 and merging audio from original video...")
                if merge_audio(temp_video_path, args.video, args.output):
                    print("Video converted and audio merged successfully.")
                else:
                    print("Warning: Failed to convert/merge, trying without audio...")
                    if not convert_to_h264(temp_video_path, args.output):
                        print("Warning: H.264 conversion failed, copying raw video.")
                        shutil.copy(temp_video_path, args.output)
            else:
                print("Converting to H.264 (no audio stream in original)...")
                if convert_to_h264(temp_video_path, args.output):
                    print("Video converted successfully.")
                else:
                    print("Warning: H.264 conversion failed, copying raw video.")
                    shutil.copy(temp_video_path, args.output)
        else:
            print("Warning: ffmpeg not found, saving video with mp4v codec (may not play in browser).")
            shutil.copy(temp_video_path, args.output)

    print(f"Output saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
