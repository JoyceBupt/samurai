#!/usr/bin/env python3
"""Merge single-object SAMURAI runs into a multiclass semantic dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PALETTE = [
    [0, 0, 0],
    [230, 57, 70],
    [42, 157, 143],
    [69, 123, 157],
    [244, 162, 97],
    [131, 56, 236],
    [255, 190, 11],
    [38, 70, 83],
]


@dataclass(frozen=True)
class SourceSpec:
    class_name: str
    class_id: int
    run_dir: Path


@dataclass
class Sample:
    source: SourceSpec
    frame_path: Path
    mask_path: Path
    overlay_path: Path | None
    sample_index: int
    source_frame_index: int | None
    source_name: str | None
    area: int
    area_ratio: float
    empty: bool


def parse_source(value: str) -> SourceSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--source must be class_name:class_id:run_dir")
    class_name, class_id_text, run_dir_text = parts
    class_name = class_name.strip()
    if not class_name:
        raise argparse.ArgumentTypeError("class_name cannot be empty")
    try:
        class_id = int(class_id_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"class_id must be an integer: {class_id_text}") from exc
    if class_id <= 0 or class_id > 255:
        raise argparse.ArgumentTypeError("class_id must be in [1, 255]")
    run_dir = Path(run_dir_text).expanduser().resolve()
    return SourceSpec(class_name=class_name, class_id=class_id, run_dir=run_dir)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def list_samples(source: SourceSpec) -> list[Sample]:
    frames_dir = source.run_dir / "frames"
    masks_dir = source.run_dir / "masks_binary"
    overlays_dir = source.run_dir / "overlays"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Missing frames directory: {frames_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks_binary directory: {masks_dir}")

    sample_map = read_json(source.run_dir / "sample_map.json").get("frames", [])
    sample_map_by_index = {
        int(item.get("sample_index")): item
        for item in sample_map
        if item.get("sample_index") is not None
    }

    samples: list[Sample] = []
    for frame_path in sorted(frames_dir.glob("*.jpg")):
        sample_index = int(frame_path.stem)
        mask_path = masks_dir / f"{frame_path.stem}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        if frame is None:
            raise ValueError(f"Cannot read frame: {frame_path}")
        if mask.shape[:2] != frame.shape[:2]:
            raise ValueError(f"Frame/mask size mismatch: {frame_path} vs {mask_path}")

        binary = mask > 0
        area = int(binary.sum())
        h, w = mask.shape[:2]
        map_item = sample_map_by_index.get(sample_index, {})
        overlay_path = overlays_dir / f"{frame_path.stem}.jpg"
        samples.append(
            Sample(
                source=source,
                frame_path=frame_path,
                mask_path=mask_path,
                overlay_path=overlay_path if overlay_path.exists() else None,
                sample_index=sample_index,
                source_frame_index=map_item.get("source_frame_index"),
                source_name=map_item.get("source_name"),
                area=area,
                area_ratio=area / float(h * w),
                empty=area == 0,
            )
        )
    if not samples:
        raise RuntimeError(f"No frame/mask pairs found in {source.run_dir}")
    return samples


def limit_empty_samples(samples: list[Sample], max_empty_to_positive_ratio: float) -> list[Sample]:
    if max_empty_to_positive_ratio < 0:
        return samples
    positives = [sample for sample in samples if not sample.empty]
    empties = [sample for sample in samples if sample.empty]
    if not positives:
        return samples
    max_empties = int(math.ceil(len(positives) * max_empty_to_positive_ratio))
    if len(empties) <= max_empties:
        return samples
    if max_empties <= 0:
        kept_empty: set[int] = set()
    elif max_empties == 1:
        kept_empty = {empties[len(empties) // 2].sample_index}
    else:
        positions = np.linspace(0, len(empties) - 1, max_empties).round().astype(int).tolist()
        kept_empty = {empties[pos].sample_index for pos in positions}
    return [sample for sample in samples if not sample.empty or sample.sample_index in kept_empty]


def build_split_assignments(samples: list[Sample], chunk_size: int, val_ratio: float) -> dict[int, str]:
    if val_ratio <= 0:
        return {sample.sample_index: "train" for sample in samples}
    if val_ratio >= 1:
        return {sample.sample_index: "val" for sample in samples}

    chunks: dict[int, list[Sample]] = {}
    for sample in samples:
        chunk_id = sample.sample_index // max(chunk_size, 1)
        chunks.setdefault(chunk_id, []).append(sample)

    chunk_ids = sorted(chunks)
    positive_chunk_ids = [chunk_id for chunk_id in chunk_ids if any(not sample.empty for sample in chunks[chunk_id])]
    candidates = positive_chunk_ids or chunk_ids
    val_chunk_count = max(1, int(round(len(chunk_ids) * val_ratio)))
    val_chunk_count = min(val_chunk_count, len(candidates))
    ranked_candidates = sorted(
        candidates,
        key=lambda chunk_id: stable_hash(f"{samples[0].source.class_name}:{samples[0].source.run_dir.name}:{chunk_id}"),
    )
    val_chunks = set(ranked_candidates[:val_chunk_count])
    return {
        sample.sample_index: "val" if (sample.sample_index // max(chunk_size, 1)) in val_chunks else "train"
        for sample in samples
    }


def split_sample(sample: Sample, split_assignments: dict[int, str]) -> str:
    return split_assignments.get(sample.sample_index, "train")


def colorize_label(label: np.ndarray, palette: list[list[int]]) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        out[label == class_id] = color
    return out


def prepare_output(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    dirs = {
        "images": output_dir / "images",
        "labels": output_dir / "labels",
        "labels_vis": output_dir / "labels_vis",
        "source_meta": output_dir / "source_meta",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def copy_source_meta(sources: list[SourceSpec], output_dir: Path) -> None:
    meta_dir = output_dir / "source_meta"
    for source in sources:
        prefix = f"{source.class_id}_{safe_name(source.class_name)}_{safe_name(source.run_dir.name)}"
        for name in ["meta.json", "sample_map.json"]:
            src = source.run_dir / name
            if src.exists():
                shutil.copy2(src, meta_dir / f"{prefix}_{name}")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    sources: list[SourceSpec] = args.source
    class_ids = [source.class_id for source in sources]
    class_names = [source.class_name for source in sources]
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("Duplicate class_id in --source")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Duplicate class_name in --source")

    output_dir = Path(args.output_dir).expanduser().resolve()
    dirs = prepare_output(output_dir, args.overwrite)
    copy_source_meta(sources, output_dir)

    palette = PALETTE[:]
    while len(palette) <= max(class_ids):
        idx = len(palette)
        palette.append([(37 * idx) % 255, (91 * idx) % 255, (163 * idx) % 255])

    manifest: list[dict[str, Any]] = []
    split_lines = {"train": [], "val": []}
    source_reports: list[dict[str, Any]] = []

    for source in sources:
        samples = list_samples(source)
        original_count = len(samples)
        original_empty = sum(1 for sample in samples if sample.empty)
        samples = limit_empty_samples(samples, args.max_empty_to_positive_ratio)
        split_assignments = build_split_assignments(samples, args.chunk_size, args.val_ratio)

        for sample in samples:
            frame = cv2.imread(str(sample.frame_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None:
                raise ValueError(f"Cannot read sample {sample.frame_path}")

            label = np.zeros(mask.shape[:2], dtype=np.uint8)
            label[mask > 0] = source.class_id
            stem = f"{source.class_id:02d}_{safe_name(source.class_name)}_{safe_name(source.run_dir.name)}_{sample.sample_index:05d}"
            image_rel = f"images/{stem}.jpg"
            label_rel = f"labels/{stem}.png"
            vis_rel = f"labels_vis/{stem}.png"

            cv2.imwrite(str(output_dir / image_rel), frame)
            cv2.imwrite(str(output_dir / label_rel), label)
            cv2.imwrite(str(output_dir / vis_rel), colorize_label(label, palette))

            split = split_sample(sample, split_assignments)
            split_lines[split].append(f"{image_rel} {label_rel}")
            manifest.append(
                {
                    "image": image_rel,
                    "label": label_rel,
                    "label_vis": vis_rel,
                    "split": split,
                    "class_name": source.class_name,
                    "class_id": source.class_id,
                    "run_dir": str(source.run_dir),
                    "sample_index": sample.sample_index,
                    "source_frame_index": sample.source_frame_index,
                    "source_name": sample.source_name,
                    "empty": sample.empty,
                    "area": sample.area,
                    "area_ratio": sample.area_ratio,
                }
            )

        kept_empty = sum(1 for sample in samples if sample.empty)
        kept_positive = len(samples) - kept_empty
        area_ratios = [sample.area_ratio for sample in samples if not sample.empty]
        source_reports.append(
            {
                "class_name": source.class_name,
                "class_id": source.class_id,
                "run_dir": str(source.run_dir),
                "original_count": original_count,
                "original_empty": original_empty,
                "kept_count": len(samples),
                "kept_positive": kept_positive,
                "kept_empty": kept_empty,
                "positive_area_ratio_min": min(area_ratios) if area_ratios else None,
                "positive_area_ratio_mean": float(np.mean(area_ratios)) if area_ratios else None,
                "positive_area_ratio_max": max(area_ratios) if area_ratios else None,
            }
        )

    for split, lines in split_lines.items():
        (output_dir / f"{split}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    classes = {"0": "background", **{str(source.class_id): source.class_name for source in sources}}
    write_json(
        output_dir / "classes.json",
        {
            "classes": classes,
            "palette": {str(idx): color for idx, color in enumerate(palette[: max(class_ids) + 1])},
            "label_format": "uint8 single-channel semantic class id",
            "background_id": 0,
        },
    )

    report = {
        "output_dir": str(output_dir),
        "sources": source_reports,
        "total_samples": len(manifest),
        "split_counts": {split: len(lines) for split, lines in split_lines.items()},
        "val_ratio": args.val_ratio,
        "chunk_size": args.chunk_size,
        "max_empty_to_positive_ratio": args.max_empty_to_positive_ratio,
        "notes": [
            "Labels are semantic class-id masks, not instance masks.",
            "Validation split is chunk-based to reduce adjacent-frame leakage.",
            "Empty masks are kept as AR hard negatives, capped per source by max_empty_to_positive_ratio.",
        ],
    }
    write_json(output_dir / "dataset_report.json", report)
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=parse_source, help="class_name:class_id:run_dir")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Chunk-level validation ratio.")
    parser.add_argument("--chunk-size", type=int, default=20, help="Contiguous sampled frames per split chunk.")
    parser.add_argument(
        "--max-empty-to-positive-ratio",
        type=float,
        default=0.5,
        help="Keep at most this many empty masks per positive sample per source. Use -1 to keep all.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    report = build_dataset(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
