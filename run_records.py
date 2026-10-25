"""Evaluation identities and CSV records shared by both calibration benchmarks."""

import csv
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def new_run(root):
    run_id = uuid4().hex
    return run_id, Path(root) / f"run_{run_id}"


def require_fresh_log(path):
    """Prevent a new optimization from silently mixing with an earlier history."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        raise FileExistsError(
            f"Results already exist: {path}. Move or remove this log before "
            "starting a fresh optimization."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def read_records(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_records(path, fields, records):
    """Atomically replace a log while preserving the supplied row order/values."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent,
                                     delete=False, suffix=".csv.tmp") as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def append_record(path, record):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        fields, records = read_records(path)
        extra = [key for key in record if key not in fields]
        if extra:
            write_records(path, fields + extra, records + [record])
            return
    else:
        fields = list(record)
        write_records(path, fields, [record])
        return
    with path.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writerow(record)


def best_successful_record(path):
    fields, records = read_records(path)
    loss_key = "total_loss" if "total_loss" in fields else "loss"
    successful = [r for r in records if r["status"] == "ok"
                  and math.isfinite(float(r[loss_key]))]
    if not successful:
        raise RuntimeError(f"No successful evaluation in {path}")
    # Single-target R² is logged positively. Double-target already logs -R².
    maximize = loss_key == "loss" and successful[0]["metric"] == "r2"
    return (max if maximize else min)(successful, key=lambda r: float(r[loss_key]))


def append_best_record(path):
    record = dict(best_successful_record(path))
    record.update(status="best", timestamp=datetime.now().isoformat(),
                  note="Best successful logged evaluation; exact run path retained")
    append_record(path, record)
    return record


def recorded_run_dir(row, log_dir):
    """Resolve only the recorded path: never guess from rounded parameters."""
    value = row.get("run_dir", "")
    if not isinstance(value, str) or not value.strip():
        raise FileNotFoundError("Selected evaluation has no verified saved run_dir")
    path = Path(value)
    if not path.is_absolute():
        path = Path(log_dir) / path
    if not path.is_dir():
        raise FileNotFoundError(f"Selected evaluation output is missing: {path}")
    return path
