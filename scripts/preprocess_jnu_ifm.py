#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess JNU-IFM us_data into a train/test dataset.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/222010008/yuncheng/Data/JNU-IFM/us_data"),
        help="Path to the original JNU-IFM us_data directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/222010008/yuncheng/Data/JNU-IFM/preprocessed"),
        help="Path to the output preprocessed dataset directory.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting.")
    return parser.parse_args()


def sorted_dirs(root: Path) -> List[Path]:
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)


def sorted_files(root: Path) -> List[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )


def normalize_mask_stem(stem: str) -> str:
    return stem[:-5] if stem.endswith("_mask") else stem


def label_sort_key(label: int | str) -> Tuple[int, str]:
    return (0, f"{int(label):08d}") if isinstance(label, int) else (1, str(label))


def read_frame_label_csv(csv_path: Path) -> Tuple[Counter, int, bool, int]:
    counter: Counter = Counter()
    row_count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "frame_label" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing the 'frame_label' column.")
        for row in reader:
            raw_label = str(row["frame_label"]).strip()
            if raw_label == "":
                continue
            label = int(raw_label)
            counter[label] += 1
            row_count += 1

    if not counter:
        raise ValueError(f"{csv_path} does not contain any valid frame_label values.")

    max_count = max(counter.values())
    majority_candidates = sorted(
        [label for label, count in counter.items() if count == max_count],
        key=label_sort_key,
    )
    majority_label = int(majority_candidates[0])
    tie = len(majority_candidates) > 1
    return counter, majority_label, tie, row_count


def build_mask_map(mask_dir: Path) -> Dict[str, Path]:
    mask_map: Dict[str, Path] = {}
    for mask_path in sorted_files(mask_dir):
        normalized_stem = normalize_mask_stem(mask_path.stem)
        if normalized_stem in mask_map:
            raise ValueError(f"Duplicate mask stem '{normalized_stem}' found in {mask_dir}.")
        mask_map[normalized_stem] = mask_path
    return mask_map


def ensure_video_layout(video_dir: Path) -> None:
    required_paths = [
        video_dir / "image",
        video_dir / "mask",
        video_dir / "frame_label.csv",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Video directory {video_dir} is missing required items: {missing}")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_single_video_label(output_path: Path, video_id: str, video_label: int) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "video_label"], delimiter=";")
        writer.writeheader()
        writer.writerow({"video_id": video_id, "video_label": video_label})


def write_video_labels_table(output_path: Path, records: Iterable[Dict[str, int | str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "video_label"], delimiter=";")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_split_manifest(output_path: Path, records: Iterable[Dict[str, int | str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "split", "video_label", "frame_count", "tie_resolved_by_smallest_label"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def split_videos(video_dirs: List[Path], train_ratio: float, seed: int) -> Dict[str, List[Path]]:
    ordered = list(video_dirs)
    random.Random(seed).shuffle(ordered)
    train_count = int(len(ordered) * train_ratio)
    train_count = max(1, min(train_count, len(ordered) - 1))
    return {
        "train": sorted(ordered[:train_count], key=lambda path: path.name),
        "test": sorted(ordered[train_count:], key=lambda path: path.name),
    }


def preprocess_dataset(input_root: Path, output_root: Path, train_ratio: float, seed: int) -> Dict[str, object]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if output_root.exists():
        raise FileExistsError(f"Output root already exists, refusing to overwrite: {output_root}")

    video_dirs = sorted_dirs(input_root)
    if len(video_dirs) < 2:
        raise ValueError("At least two video folders are required to produce train/test splits.")

    split_map = split_videos(video_dirs, train_ratio=train_ratio, seed=seed)

    safe_mkdir(output_root)
    overall_records: List[Dict[str, int | str]] = []
    manifest_records: List[Dict[str, int | str]] = []
    report: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": seed,
        "train_ratio": train_ratio,
        "total_videos": len(video_dirs),
        "splits": {},
        "label_distributions": {},
        "validation": {},
    }

    for split_name, split_videos_list in split_map.items():
        split_root = output_root / split_name
        image_root = split_root / "image"
        mask_root = split_root / "mask"
        # Keep label sidecars outside frame folders so downstream loaders do not
        # mistake them for image frames.
        meta_root = split_root / "meta"
        safe_mkdir(image_root)
        safe_mkdir(mask_root)
        safe_mkdir(meta_root)

        split_records: List[Dict[str, int | str]] = []
        split_frame_total = 0

        for video_dir in split_videos_list:
            ensure_video_layout(video_dir)
            video_id = video_dir.name
            frame_label_csv = video_dir / "frame_label.csv"
            image_src_dir = video_dir / "image"
            mask_src_dir = video_dir / "mask"

            label_counter, video_label, tie, labeled_row_count = read_frame_label_csv(frame_label_csv)
            image_files = sorted_files(image_src_dir)
            if not image_files:
                raise ValueError(f"No image files found in {image_src_dir}")
            if labeled_row_count != len(image_files):
                raise ValueError(
                    f"Frame label row count does not match image count for video {video_id}: "
                    f"{labeled_row_count} labels vs {len(image_files)} images"
                )

            mask_map = build_mask_map(mask_src_dir)
            missing_masks = [image_path.name for image_path in image_files if image_path.stem not in mask_map]
            if missing_masks:
                raise ValueError(f"Missing masks for video {video_id}: {missing_masks[:5]}")

            image_stems = {image_path.stem for image_path in image_files}
            extra_masks = sorted(stem for stem in mask_map if stem not in image_stems)
            if extra_masks:
                raise ValueError(f"Extra masks without matching image in video {video_id}: {extra_masks[:5]}")

            image_dst_dir = image_root / video_id
            mask_dst_dir = mask_root / video_id
            meta_dst_dir = meta_root / video_id
            safe_mkdir(image_dst_dir)
            safe_mkdir(mask_dst_dir)
            safe_mkdir(meta_dst_dir)

            for image_path in image_files:
                shutil.copy2(image_path, image_dst_dir / image_path.name)
                shutil.copy2(mask_map[image_path.stem], mask_dst_dir / image_path.name)

            shutil.copy2(frame_label_csv, meta_dst_dir / "frame_label.csv")
            write_single_video_label(meta_dst_dir / "video_label.txt", video_id=video_id, video_label=video_label)

            split_records.append({"video_id": video_id, "video_label": video_label})
            overall_records.append({"video_id": video_id, "video_label": video_label})
            manifest_records.append(
                {
                    "video_id": video_id,
                    "split": split_name,
                    "video_label": video_label,
                    "frame_count": len(image_files),
                    "tie_resolved_by_smallest_label": int(tie),
                }
            )
            split_frame_total += len(image_files)
            report["label_distributions"][video_id] = {
                "split": split_name,
                "frame_label_counts": dict(sorted(label_counter.items(), key=lambda item: item[0])),
                "video_label": video_label,
                "frame_count": len(image_files),
                "frame_label_rows": labeled_row_count,
                "tie_resolved_by_smallest_label": tie,
            }

        write_video_labels_table(split_root / "video_labels.txt", split_records)
        report["splits"][split_name] = {
            "video_count": len(split_videos_list),
            "frame_count": split_frame_total,
            "video_ids": [video_dir.name for video_dir in split_videos_list],
        }

    write_video_labels_table(output_root / "video_labels.txt", sorted(overall_records, key=lambda item: str(item["video_id"])))
    write_split_manifest(output_root / "split_manifest.csv", manifest_records)
    report["validation"] = validate_output(output_root=output_root, split_map=split_map, report=report)

    with (output_root / "preprocess_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)

    return report


def validate_output(output_root: Path, split_map: Dict[str, List[Path]], report: Dict[str, object]) -> Dict[str, object]:
    validation: Dict[str, object] = {
        "status": "passed",
        "checks": [],
    }

    expected_counts = {split_name: len(video_dirs) for split_name, video_dirs in split_map.items()}
    actual_counts = {}

    for split_name, video_dirs in split_map.items():
        split_root = output_root / split_name
        image_root = split_root / "image"
        mask_root = split_root / "mask"
        meta_root = split_root / "meta"

        for required_dir in (image_root, mask_root, meta_root):
            if not required_dir.is_dir():
                raise FileNotFoundError(f"Required directory is missing: {required_dir}")

        image_videos = sorted_dirs(image_root)
        mask_videos = sorted_dirs(mask_root)
        meta_videos = sorted_dirs(meta_root)
        expected_video_names = sorted(video_dir.name for video_dir in video_dirs)

        image_names = [path.name for path in image_videos]
        mask_names = [path.name for path in mask_videos]
        meta_names = [path.name for path in meta_videos]
        if image_names != expected_video_names or mask_names != expected_video_names or meta_names != expected_video_names:
            raise ValueError(
                f"Split {split_name} has inconsistent video folders. "
                f"Expected={expected_video_names}, image={image_names}, mask={mask_names}, meta={meta_names}"
            )

        actual_counts[split_name] = len(image_videos)
        for video_name in expected_video_names:
            image_files = sorted_files(image_root / video_name)
            mask_files = sorted_files(mask_root / video_name)
            image_names_set = [path.name for path in image_files]
            mask_names_set = [path.name for path in mask_files]
            if image_names_set != mask_names_set:
                raise ValueError(f"Image/mask file mismatch detected in split={split_name}, video={video_name}")

            video_label_path = meta_root / video_name / "video_label.txt"
            if not video_label_path.is_file():
                raise FileNotFoundError(f"Missing video label file: {video_label_path}")

            with video_label_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=";")
                rows = list(reader)
            if len(rows) != 1:
                raise ValueError(f"{video_label_path} should contain exactly one record.")
            row = rows[0]
            expected_label = report["label_distributions"][video_name]["video_label"]
            if row["video_id"] != video_name or int(row["video_label"]) != int(expected_label):
                raise ValueError(
                    f"Incorrect video label in {video_label_path}: got {row}, expected label {expected_label}"
                )

    total_expected = sum(expected_counts.values())
    total_actual = sum(actual_counts.values())
    if total_expected != total_actual:
        raise ValueError(f"Split count mismatch: expected {total_expected}, got {total_actual}")

    validation["checks"].append(
        {
            "name": "split_ratio",
            "expected": expected_counts,
            "actual": actual_counts,
            "passed": expected_counts == actual_counts,
        }
    )
    validation["checks"].append(
        {
            "name": "directory_structure",
            "required_subdirs": ["image", "mask", "meta"],
            "passed": True,
        }
    )
    validation["checks"].append(
        {
            "name": "video_label_files",
            "checked_videos": total_actual,
            "passed": True,
        }
    )
    return validation


def main() -> None:
    args = parse_args()
    report = preprocess_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    summary = {
        "total_videos": report["total_videos"],
        "train_videos": report["splits"]["train"]["video_count"],
        "test_videos": report["splits"]["test"]["video_count"],
        "seed": report["seed"],
        "output_root": report["output_root"],
        "validation_status": report["validation"]["status"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
