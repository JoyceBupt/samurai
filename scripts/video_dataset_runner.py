#!/usr/bin/env python3
"""Generate binary mask datasets from a video or frame directory with SAMURAI."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sam2"))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def determine_model_cfg(model_path: str) -> str:
    checkpoint_name = Path(model_path).name
    if "large" in checkpoint_name:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in checkpoint_name:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in checkpoint_name:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    if "tiny" in checkpoint_name:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError(f"Cannot infer SAMURAI config from checkpoint name: {model_path}")


def parse_box(text: str, box_format: str) -> tuple[float, float, float, float]:
    values = [float(part.strip()) for part in text.replace(" ", ",").split(",") if part.strip()]
    if len(values) != 4:
        raise ValueError(f"Expected 4 box values, got {len(values)} from: {text!r}")
    x0, y0, a, b = values
    if box_format == "xywh":
        return x0, y0, x0 + a, y0 + b
    if box_format == "xyxy":
        return x0, y0, a, b
    raise ValueError(f"Unsupported box format: {box_format}")


def load_box(args: argparse.Namespace) -> tuple[float, float, float, float]:
    if args.box:
        return parse_box(args.box, args.box_format)
    if not args.prompt_file:
        raise ValueError("Provide either --box or --prompt-file.")
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return parse_box(line, args.box_format)
    raise ValueError(f"No usable box prompt found in {args.prompt_file}")


def list_frame_files(frame_dir: Path) -> list[Path]:
    frames = [p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(frames, key=lambda p: p.name)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    dirs = {
        "frames": output_dir / "frames",
        "masks_binary": output_dir / "masks_binary",
        "masks_vis": output_dir / "masks_vis",
        "overlays": output_dir / "overlays",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_first_video_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Cannot read the first frame from video: {video_path}")
    return frame


def resize_if_needed(frame: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    if width is None and height is None:
        return frame
    if width is None or height is None:
        raise ValueError("Both --width and --height must be provided when resizing.")
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def materialize_frames(
    input_path: Path,
    frames_dir: Path,
    frame_stride: int,
    width: int | None,
    height: int | None,
    max_frames: int | None,
) -> tuple[list[dict[str, Any]], tuple[int, int], tuple[int, int], float | None]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1.")

    sample_map: list[dict[str, Any]] = []
    fps: float | None = None

    if input_path.is_dir():
        source_frames = list_frame_files(input_path)
        if not source_frames:
            raise ValueError(f"No image frames found in {input_path}")

        first = cv2.imread(str(source_frames[0]))
        if first is None:
            raise ValueError(f"Cannot read frame: {source_frames[0]}")
        original_h, original_w = first.shape[:2]

        sample_idx = 0
        for source_idx, frame_path in enumerate(source_frames):
            if source_idx % frame_stride != 0:
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Cannot read frame: {frame_path}")
            frame = resize_if_needed(frame, width, height)
            cv2.imwrite(str(frames_dir / f"{sample_idx:05d}.jpg"), frame)
            sample_map.append(
                {
                    "sample_index": sample_idx,
                    "source_frame_index": source_idx,
                    "source_name": frame_path.name,
                }
            )
            sample_idx += 1
            if max_frames is not None and sample_idx >= max_frames:
                break
    elif input_path.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or None

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise ValueError(f"Cannot read frames from video: {input_path}")
        original_h, original_w = frame.shape[:2]

        source_idx = 0
        sample_idx = 0
        while ok and frame is not None:
            if source_idx % frame_stride == 0:
                output_frame = resize_if_needed(frame, width, height)
                cv2.imwrite(str(frames_dir / f"{sample_idx:05d}.jpg"), output_frame)
                sample_map.append(
                    {
                        "sample_index": sample_idx,
                        "source_frame_index": source_idx,
                        "source_name": f"{source_idx:05d}",
                    }
                )
                sample_idx += 1
                if max_frames is not None and sample_idx >= max_frames:
                    break
            ok, frame = cap.read()
            source_idx += 1
        cap.release()
    else:
        raise ValueError(f"Input must be a video file or frame directory: {input_path}")

    if not sample_map:
        raise RuntimeError("No frames were materialized. Check --frame-stride and --max-frames.")

    first_output = cv2.imread(str(frames_dir / "00000.jpg"))
    output_h, output_w = first_output.shape[:2]
    return sample_map, (original_w, original_h), (output_w, output_h), fps


def scale_box(
    box_xyxy: tuple[float, float, float, float],
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    original_w, original_h = original_size
    output_w, output_h = output_size
    sx = output_w / original_w
    sy = output_h / original_h
    x1, y1, x2, y2 = box_xyxy
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def create_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    overlay[mask] = [0, 0, 255]
    output = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


def save_mask_outputs(
    frame_idx: int,
    mask_tensor: torch.Tensor,
    dirs: dict[str, Path],
    threshold: float,
    save_overlays: bool,
) -> dict[str, Any]:
    mask = mask_tensor[0].detach().cpu().numpy() > threshold
    binary = mask.astype(np.uint8)
    vis = (binary * 255).astype(np.uint8)

    stem = f"{frame_idx:05d}"
    cv2.imwrite(str(dirs["masks_binary"] / f"{stem}.png"), binary)
    cv2.imwrite(str(dirs["masks_vis"] / f"{stem}.png"), vis)

    if save_overlays:
        frame = cv2.imread(str(dirs["frames"] / f"{stem}.jpg"))
        if frame is None:
            raise ValueError(f"Cannot read materialized frame for overlay: {stem}.jpg")
        cv2.imwrite(str(dirs["overlays"] / f"{stem}.jpg"), create_overlay(frame, mask))

    area = int(binary.sum())
    h, w = binary.shape[:2]
    return {
        "frame_index": frame_idx,
        "area": area,
        "area_ratio": area / float(h * w),
        "bbox_xywh": mask_bbox(mask),
        "empty": area == 0,
    }


def check_cuda_memory(device: str, min_free_gb: float | None) -> None:
    if min_free_gb is None or not device.startswith("cuda"):
        return
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    if free_gb < min_free_gb:
        raise RuntimeError(f"GPU {device} has {free_gb:.2f} GiB free of {total_gb:.2f} GiB; need {min_free_gb:.2f} GiB.")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dirs = prepare_output_dir(output_dir, args.overwrite)

    prompt_box = load_box(args)
    sample_map, original_size, output_size, fps = materialize_frames(
        input_path=input_path,
        frames_dir=dirs["frames"],
        frame_stride=args.frame_stride,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
    )
    scaled_box = scale_box(prompt_box, original_size, output_size)
    box_array = np.array(scaled_box, dtype=np.float32)

    write_json(output_dir / "sample_map.json", {"frames": sample_map})

    check_cuda_memory(args.device, args.min_free_gb)
    model_cfg = args.model_cfg or determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=args.device)
    if hasattr(predictor, "to"):
        predictor.to(args.device)
    elif hasattr(predictor, "sam_model") and hasattr(predictor.sam_model, "to"):
        predictor.sam_model.to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    frame_stats: list[dict[str, Any]] = []
    device_type = args.device.split(":")[0]
    try:
        with torch.inference_mode(), torch.autocast(device_type, dtype=torch.float16, enabled=device_type == "cuda"):
            state = predictor.init_state(str(dirs["frames"]), offload_video_to_cpu=args.offload_video_to_cpu)
            _, _, masks = predictor.add_new_points_or_box(state, box=box_array, frame_idx=0, obj_id=0)
            frame_stats.append(save_mask_outputs(0, masks[0], dirs, args.threshold, not args.no_overlays))

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                if frame_idx == 0:
                    continue
                if not object_ids:
                    continue
                frame_stats.append(save_mask_outputs(frame_idx, masks[0], dirs, args.threshold, not args.no_overlays))
                if frame_idx % args.progress_interval == 0:
                    print(f"processed {frame_idx}/{len(sample_map) - 1}", flush=True)
    finally:
        del predictor
        if "state" in locals():
            del state
        torch.clear_autocast_cache()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    meta = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "model_cfg": model_cfg,
        "device": args.device,
        "frame_stride": args.frame_stride,
        "frame_count": len(sample_map),
        "fps": fps,
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "output_size": {"width": output_size[0], "height": output_size[1]},
        "prompt": {
            "box_format": args.box_format,
            "box_xyxy_original": [round(v, 3) for v in prompt_box],
            "box_xyxy_scaled": [round(float(v), 3) for v in scaled_box],
        },
        "threshold": args.threshold,
        "offload_video_to_cpu": args.offload_video_to_cpu,
        "mask_frames": len(frame_stats),
        "empty_mask_frames": sum(1 for item in frame_stats if item["empty"]),
        "frame_stats": frame_stats,
    }
    write_json(output_dir / "meta.json", meta)

    print(f"done: {output_dir}")
    print(f"frames: {len(sample_map)}, masks: {len(frame_stats)}, empty masks: {meta['empty_mask_frames']}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input video file or image frame directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for frames, masks, overlays, and metadata.")
    parser.add_argument("--prompt-file", default=None, help="Text file containing one bbox line.")
    parser.add_argument("--box", default=None, help="Inline bbox prompt, e.g. 'x,y,w,h'.")
    parser.add_argument("--box-format", choices=["xywh", "xyxy"], default="xywh", help="Format used by --box/--prompt-file.")
    parser.add_argument("--model-path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="SAMURAI checkpoint.")
    parser.add_argument("--model-cfg", default=None, help="Override SAMURAI config path.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Keep one frame every N source frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of sampled frames.")
    parser.add_argument("--width", type=int, default=None, help="Optional output frame width.")
    parser.add_argument("--height", type=int, default=None, help="Optional output frame height.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Mask logit threshold.")
    parser.add_argument("--min-free-gb", type=float, default=None, help="Fail if the selected CUDA device has less free memory.")
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-overlays", action="store_true", help="Skip overlay image generation.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    parser.add_argument("--progress-interval", type=int, default=50, help="Print progress every N sampled frames.")
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
