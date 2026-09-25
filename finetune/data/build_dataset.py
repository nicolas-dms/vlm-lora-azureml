"""Build the VLM fine-tuning dataset: telemetry -> chart PNGs + JSONL samples.

Source-agnostic by design. `build_from_frame` takes a pandas DataFrame with six
canonical columns, so the same code runs over the CSV shipped here, a parquet
snapshot in a datastore, or a live query against whatever time-series store you use:

    minute, device_id, temperature_c, humidity_pct, pressure_bar, vibration_mm_s

`minute` is a per-minute timestamp; the other four are float sensor readings.

A 24-hour sample of 4 machines is committed as `telemetry_sample.csv`, so the repo is
runnable end to end without any data plumbing. Nothing in this file touches Azure -
building the images is free and local.

Usage:

    python finetune/data/build_dataset.py \
        --telemetry finetune/data/telemetry_sample.csv --out ./out/vlm_ds_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd

from labeling_rules import label_to_text, label_window, window_features
from render_charts import render_window

WINDOW_MINUTES = 30
STRIDE_MINUTES = 5
TIME_COLUMN = "minute"

INSTRUCTION = (
    "Read the 4 sensor charts for this machine over the last 30 minutes and issue a "
    "maintenance work order. Reply with JSON only, no prose, exactly these keys: "
    '{"device_id": string, '
    '"severity": one of [info, warning, critical], '
    '"root_cause": one of [normal_operation, cooling_degradation, bearing_wear, pressure_loss], '
    '"action_code": one of [NONE, MONITOR, CHK_COOLING, CHK_BEARING, CHK_SEALS, STOP_MAINT], '
    '"eta_minutes": integer}'
)


def load_telemetry(path: str) -> pd.DataFrame:
    """Read the telemetry table. CSV or parquet, decided by the suffix.

    CSV is the committed format on purpose: a reader can open it, see what the six
    columns actually contain, and edit a value to watch the label change. Parquet is
    smaller and faster, and completely opaque to someone trying to understand the task.
    """
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=[TIME_COLUMN])


def iter_windows(frame: pd.DataFrame) -> Iterator[Tuple[str, pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """Yield (device_id, start, end, window) for every sliding window."""
    for device_id, group in frame.groupby("device_id"):
        group = group.sort_values(TIME_COLUMN).reset_index(drop=True)
        for start in range(0, len(group) - WINDOW_MINUTES + 1, STRIDE_MINUTES):
            window = group.iloc[start : start + WINDOW_MINUTES]
            yield (
                str(device_id),
                window[TIME_COLUMN].iloc[0],
                window[TIME_COLUMN].iloc[-1],
                window,
            )


def _split_boundaries(frame: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t_min, t_max = frame[TIME_COLUMN].min(), frame[TIME_COLUMN].max()
    span = t_max - t_min
    return t_min + span * train_frac, t_min + span * (train_frac + val_frac)


def _assign_split(start: pd.Timestamp, end: pd.Timestamp, b1: pd.Timestamp, b2: pd.Timestamp) -> str | None:
    """Chronological split. Windows straddling a boundary are dropped.

    Windows overlap (stride 5 min < window 30 min), so a random split would leak
    almost-identical charts between train and test and the eval would be worthless.
    """
    if start < b1 <= end or start < b2 <= end:
        return None
    if end < b1:
        return "train"
    if end < b2:
        return "val"
    return "test"


def _subsample(items: List[Dict[str, Any]], limit: int | None) -> List[Dict[str, Any]]:
    """Keep at most `limit` items, evenly spread over the timeline (no reshuffling)."""
    if limit is None or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def build_from_frame(
    frame: pd.DataFrame,
    out_dir: str | Path,
    max_train: int | None = 500,
    max_val: int | None = 60,
    max_test: int | None = 60,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Render every window, label it, and write images/ + {train,val,test}.jsonl.

    Rendering is idempotent: windowing, labelling and index assignment are fully
    deterministic, so an image that already exists on disk is the image this run
    would have produced. `skip_existing` therefore makes the build resumable after
    an interruption instead of restarting from scratch.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frame = frame.copy()
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN])

    b1, b2 = _split_boundaries(frame, train_frac=0.70, val_frac=0.10)

    # Pass 1: window, label, assign a split. No image is rendered yet - rendering is
    # by far the slow part and we are about to throw most windows away.
    staged: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    dropped = 0
    for device_id, start, end, window in iter_windows(frame):
        split = _assign_split(start, end, b1, b2)
        if split is None:
            dropped += 1
            continue
        features = window_features(window)
        label = label_window(device_id, features)
        staged[split].append(
            {"device_id": device_id, "start": start, "end": end, "window": window, "label": label}
        )

    limits = {"train": max_train, "val": max_val, "test": max_test}
    kept = {name: _subsample(items, limits[name]) for name, items in staged.items()}

    # Pass 2: render and write, only for the windows we actually keep.
    index = 0
    counts: Dict[str, Dict[str, int]] = {}
    for split, items in kept.items():
        lines: List[str] = []
        for item in items:
            image_rel = f"images/chart_{index:06d}.png"
            image_abs = out_dir / image_rel
            if not (skip_existing and image_abs.exists()):
                render_window(item["window"], item["device_id"], image_abs)
            lines.append(
                json.dumps(
                    {
                        "image": image_rel,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": label_to_text(item["label"])}],
                            },
                        ],
                        "meta": {
                            "device_id": item["device_id"],
                            "window_end": item["end"].isoformat(),
                        },
                    }
                )
            )
            index += 1
        (out_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

        causes = pd.Series([i["label"]["root_cause"] for i in items]).value_counts().to_dict()
        severities = pd.Series([i["label"]["severity"] for i in items]).value_counts().to_dict()
        counts[split] = {"n": len(items), "root_cause": causes, "severity": severities}

    manifest = {
        "window_minutes": WINDOW_MINUTES,
        "stride_minutes": STRIDE_MINUTES,
        "split": "chronological (70/10/20), straddling windows dropped",
        "dropped_at_boundaries": dropped,
        "time_range": [str(frame[TIME_COLUMN].min()), str(frame[TIME_COLUMN].max())],
        "devices": sorted(frame["device_id"].unique().tolist()),
        "counts": counts,
        "instruction": INSTRUCTION,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--telemetry",
        required=True,
        help="Telemetry table: .csv or .parquet, local path or URL.",
    )
    parser.add_argument("--out", required=True, help="Output folder for images/ and *.jsonl.")
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-val", type=int, default=60)
    parser.add_argument("--max-test", type=int, default=60)
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Re-render images that already exist instead of reusing them.",
    )
    args = parser.parse_args()

    frame = load_telemetry(args.telemetry)
    manifest = build_from_frame(
        frame,
        args.out,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        skip_existing=not args.force_render,
    )
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
