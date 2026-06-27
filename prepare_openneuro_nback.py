"""Prepare OpenNeuro n-back datasets for the graph working-memory model.

This handles the two public datasets used as an immediate real-data path:

- ds002424: children with/without ADHD
- ds002687: adult companion dataset

The script can:

1. Extract real 2-back accuracy targets from BIDS events.tsv files.
2. Download selected task fMRI BOLD files from OpenNeuro's public S3 bucket.
3. Convert downloaded BOLD files into coarse functional connectivity matrices.

The parcellation here is deliberately lightweight: it creates 10 data-driven
spatial parcels from active voxels so the project can run without Nilearn,
Nibabel, or an atlas install. For publication work, replace this with a named
atlas such as Schaefer, AAL, or Glasser/HCP-MMP.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


S3_BASE = "https://s3.amazonaws.com/openneuro.org"
TASKS = ["SLD", "SLI", "SSD", "SSI", "VLD", "VLI", "VSD", "VSI"]


def dataset_id(metadata_dir: Path) -> str:
    name = metadata_dir.name.lower()
    if "ds002424" in name:
        return "ds002424"
    if "ds002687" in name:
        return "ds002687"
    raise ValueError(f"Cannot infer dataset id from {metadata_dir}")


def subject_dirs(metadata_dir: Path) -> list[Path]:
    return sorted([p for p in metadata_dir.glob("sub-*") if p.is_dir()])


def extract_targets(metadata_dir: Path, output_csv: Path) -> pd.DataFrame:
    participants_path = metadata_dir / "participants.tsv"
    participants = pd.read_csv(participants_path, sep="\t") if participants_path.exists() else pd.DataFrame()
    rows = []

    for sub_dir in subject_dirs(metadata_dir):
        subject_id = sub_dir.name
        event_files = sorted(sub_dir.glob("**/*_events.tsv"))
        accuracies = []
        reaction_times = []
        task_count = 0
        for event_file in event_files:
            events = pd.read_csv(event_file, sep="\t")
            if "trial_type" not in events or "accuracy" not in events:
                continue
            wm = events[events["trial_type"].astype(str).str.lower().eq("2-back")].copy()
            if wm.empty:
                continue
            wm["accuracy"] = pd.to_numeric(wm["accuracy"], errors="coerce")
            accuracies.extend(wm["accuracy"].dropna().tolist())
            if "response_time" in wm:
                rt = pd.to_numeric(wm["response_time"], errors="coerce")
                reaction_times.extend(rt.dropna().tolist())
            task_count += 1

        if not accuracies:
            continue
        row = {
            "subject_id": subject_id,
            "working_memory_score": float(np.mean(accuracies)),
            "wm_2back_accuracy": float(np.mean(accuracies)),
            "wm_2back_rt": float(np.mean(reaction_times)) if reaction_times else np.nan,
            "n_2back_trials": int(len(accuracies)),
            "n_tasks_with_2back": int(task_count),
        }
        for task in TASKS:
            task_files = [p for p in event_files if f"_task-{task}_" in p.name]
            task_acc = []
            task_rt = []
            for event_file in task_files:
                events = pd.read_csv(event_file, sep="\t")
                wm = events[events["trial_type"].astype(str).str.lower().eq("2-back")].copy()
                if wm.empty:
                    continue
                wm["accuracy"] = pd.to_numeric(wm["accuracy"], errors="coerce")
                task_acc.extend(wm["accuracy"].dropna().tolist())
                if "response_time" in wm:
                    task_rt.extend(pd.to_numeric(wm["response_time"], errors="coerce").dropna().tolist())
            if task_acc:
                row[f"{task}_2back_accuracy"] = float(np.mean(task_acc))
                row[f"{task}_2back_rt"] = float(np.mean(task_rt)) if task_rt else np.nan
        if not participants.empty and "participant_id" in participants:
            match = participants[participants["participant_id"].eq(subject_id)]
            if not match.empty:
                for col in ["age", "age_ses-T1", "sex", "ADHD_diagnosis"]:
                    if col in match:
                        row[col] = match.iloc[0][col]
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("subject_id")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def is_annex_pointer(path: Path) -> bool:
    return path.exists() and path.stat().st_size < 1024


def find_bold_pointer(metadata_dir: Path, subject_id: str, task: str) -> Path | None:
    matches = sorted((metadata_dir / subject_id).glob(f"**/{subject_id}*_task-{task}_bold.nii.gz"))
    return matches[0] if matches else None


def relative_dataset_path(metadata_dir: Path, path: Path) -> str:
    return path.relative_to(metadata_dir).as_posix()


def download_bolds(metadata_dir: Path, output_dir: Path, task: str, max_subjects: int) -> list[Path]:
    ds_id = dataset_id(metadata_dir)
    downloaded = []
    for sub_dir in subject_dirs(metadata_dir)[:max_subjects]:
        subject_id = sub_dir.name
        pointer = find_bold_pointer(metadata_dir, subject_id, task)
        if pointer is None:
            continue
        rel = relative_dataset_path(metadata_dir, pointer)
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 1024:
            downloaded.append(target)
            continue
        if not is_annex_pointer(pointer):
            shutil.copyfile(pointer, target)
            downloaded.append(target)
            continue
        url = f"{S3_BASE}/{ds_id}/{rel}"
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
        downloaded.append(target)
    return downloaded


def nifti_dtype(datatype: int) -> np.dtype:
    mapping = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        256: np.int8,
        512: np.uint16,
        768: np.uint32,
    }
    if datatype not in mapping:
        raise ValueError(f"Unsupported NIfTI datatype: {datatype}")
    return np.dtype(mapping[datatype])


def load_nifti_4d(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        blob = f.read()
    endian = "<" if struct.unpack("<i", blob[:4])[0] == 348 else ">"
    dim = struct.unpack(endian + "8h", blob[40:56])
    datatype = struct.unpack(endian + "h", blob[70:72])[0]
    vox_offset = int(struct.unpack(endian + "f", blob[108:112])[0])
    shape = tuple(int(x) for x in dim[1 : dim[0] + 1])
    dtype = nifti_dtype(datatype).newbyteorder(endian)
    data = np.frombuffer(blob, dtype=dtype, offset=vox_offset)
    expected = int(np.prod(shape))
    if data.size < expected:
        raise ValueError(f"{path} is shorter than expected from NIfTI header")
    data = data[:expected].reshape(shape, order="F").astype(np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD NIfTI, got shape {data.shape}: {path}")
    return data


def bold_to_connectivity(path: Path, parcels: int = 10) -> tuple[np.ndarray, list[str]]:
    bold = load_nifti_4d(path)
    finite = np.isfinite(bold).all(axis=3)
    temporal_std = np.std(bold, axis=3)
    mask = finite & (temporal_std > np.percentile(temporal_std[finite], 60))
    coords = np.argwhere(mask)
    if len(coords) < parcels:
        raise ValueError(f"Not enough active voxels to parcel {path}")

    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    chunks = np.array_split(coords[order], parcels)
    series = []
    names = []
    for i, chunk in enumerate(chunks, start=1):
        values = bold[chunk[:, 0], chunk[:, 1], chunk[:, 2], :]
        ts = values.mean(axis=0)
        ts = (ts - ts.mean()) / (ts.std() + 1e-8)
        series.append(ts)
        names.append(f"parcel_{i:02d}")
    matrix = np.corrcoef(np.asarray(series))
    np.fill_diagonal(matrix, 0.0)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, names


def build_connectivity(raw_dir: Path, targets_csv: Path, output_dir: Path, task: str, parcels: int) -> int:
    targets = pd.read_csv(targets_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for subject_id in targets["subject_id"].astype(str):
        matches = sorted((raw_dir / subject_id).glob(f"**/{subject_id}*_task-{task}_bold.nii.gz"))
        if not matches:
            continue
        matrix, names = bold_to_connectivity(matches[0], parcels=parcels)
        out = output_dir / f"{subject_id}.csv"
        pd.DataFrame(matrix, index=names, columns=names).to_csv(out)
        made += 1
        print(f"Wrote {out}")
    return made


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenNeuro n-back data for the WM graph model.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="VLI")
    parser.add_argument("--max-subjects", type=int, default=8)
    parser.add_argument("--extract-targets", action="store_true")
    parser.add_argument("--download-bolds", action="store_true")
    parser.add_argument("--build-connectivity", action="store_true")
    parser.add_argument("--parcels", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets_csv = args.prepared_dir / "targets.csv"
    raw_dir = args.prepared_dir / "raw"
    connectivity_dir = args.prepared_dir / "connectivity"

    if args.extract_targets:
        targets = extract_targets(args.metadata_dir, targets_csv)
        print(f"Wrote {len(targets)} targets to {targets_csv}")
    if args.download_bolds:
        files = download_bolds(args.metadata_dir, raw_dir, args.task, args.max_subjects)
        print(f"Downloaded/available BOLD files: {len(files)}")
    if args.build_connectivity:
        if not targets_csv.exists():
            extract_targets(args.metadata_dir, targets_csv)
        count = build_connectivity(raw_dir, targets_csv, connectivity_dir, args.task, args.parcels)
        print(f"Wrote connectivity matrices: {count}")


if __name__ == "__main__":
    main()
