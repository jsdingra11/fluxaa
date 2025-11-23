from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)


def _now_utc_iso() -> str:
    return datetime.utcnow().isoformat()


def _generate_run_id() -> str:
    # e.g. 20251123T105121470510
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")


@dataclass
class RunRecord:
    run_id: str
    name: str
    script: str
    backend: str
    status: str
    attempts: int
    last_exit_code: Optional[int]
    checkpoint_dir: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_s: Optional[float]
    extra_args: List[str]
    metrics: Optional[Dict[str, Any]] = None

    @property
    def log_path(self) -> Path:
        return RUNS_DIR / f"run_{self.run_id}.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        # Backwards compatibility if metrics missing
        if "metrics" not in data:
            data["metrics"] = None
        return cls(**data)


def _save_run_record(record: RunRecord) -> None:
    record.log_path.parent.mkdir(exist_ok=True)
    with record.log_path.open("w") as f:
        json.dump(record.to_dict(), f, indent=2)


def _parse_fluxaa_metrics(output: str) -> Optional[Dict[str, Any]]:
    """
    Scan combined stdout/stderr for a JSON line like:
        {"fluxaa_metrics": {...}}
    and return the inner dict.
    """
    # Look from bottom up (most likely near the end)
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        if "fluxaa_metrics" not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "fluxaa_metrics" in obj:
            metrics = obj["fluxaa_metrics"]
            if isinstance(metrics, dict):
                return metrics
    return None


def _run_subprocess_with_output(cmd: List[str]) -> Tuple[int, str]:
    """
    Run a subprocess, stream output to our stdout, and capture it.
    Returns (returncode, combined_output).
    """
    print(f"[orchestrator][local] Running: {' '.join(cmd)}")
    sys.stdout.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        # Stream to user in real-time
        print(line, end="")
    proc.wait()

    return proc.returncode, "".join(lines)


def run_local_job(
    script: str,
    checkpoint_dir: str,
    max_retries: int,
    extra_args: Optional[List[str]] = None,
    name: str = "",
    python_executable: str = sys.executable,
) -> RunRecord:
    """
    Core orchestration loop for the 'local' backend.

    - Retries up to max_retries on non-zero exit code.
    - Always logs a RunRecord to runs/run_<id>.json
    - On SUCCESS, attempts to parse fluxaa_metrics from child output.
    """
    extra_args = extra_args or []
    run_id = _generate_run_id()
    attempts_allowed = max_retries + 1

    record = RunRecord(
        run_id=run_id,
        name=name,
        script=script,
        backend="local",
        status="RUNNING",
        attempts=0,
        last_exit_code=None,
        checkpoint_dir=checkpoint_dir,
        start_time=_now_utc_iso(),
        end_time=None,
        duration_s=None,
        extra_args=extra_args,
        metrics=None,
    )

    RUNS_DIR.mkdir(exist_ok=True)

    script_path = Path(script)
    if not script_path.exists():
        print(f"[orchestrator][local] ERROR: script not found: {script}", file=sys.stderr)
        record.status = "FAIL"
        record.end_time = _now_utc_iso()
        _save_run_record(record)
        return record

    cmd_base = [python_executable, str(script_path), "--checkpoint-dir", checkpoint_dir]

    last_output: str = ""

    print()
    for attempt in range(1, attempts_allowed + 1):
        print(f"[orchestrator][local] Attempt {attempt}/{attempts_allowed}")
        sys.stdout.flush()

        cmd = cmd_base.copy()
        if attempt > 1:
            # On retries, inject --resume
            cmd.append("--resume")

        # Add user extra args at the end
        cmd.extend(extra_args)

        code, combined_output = _run_subprocess_with_output(cmd)
        last_output = combined_output
        record.attempts = attempt
        record.last_exit_code = code

        if code == 0:
            print("[orchestrator][local] Job completed successfully ✅")
            record.status = "SUCCESS"
            break
        else:
            print(f"[orchestrator][local] Job failed with exit code {code} ❌")
            if attempt >= attempts_allowed:
                print("[orchestrator][local] Reached max retries. Giving up.")
                record.status = "FAIL"
                break
            else:
                print("[orchestrator][local] Retrying...")

    record.end_time = _now_utc_iso()
    try:
        start_dt = datetime.fromisoformat(record.start_time)
        end_dt = datetime.fromisoformat(record.end_time)
        record.duration_s = (end_dt - start_dt).total_seconds()
    except Exception:
        record.duration_s = None

    # If success, try to parse metrics from output
    if record.status == "SUCCESS":
        metrics = _parse_fluxaa_metrics(last_output)
        if metrics is not None:
            record.metrics = metrics
            print(f"[orchestrator][local] Parsed metrics from child process: {metrics}")

    _save_run_record(record)
    print(f"[orchestrator] Logged run -> {record.log_path.name}")
    return record


def load_all_runs() -> List[RunRecord]:
    records: List[RunRecord] = []
    if not RUNS_DIR.exists():
        return records
    for path in sorted(RUNS_DIR.glob("run_*.json")):
        try:
            with path.open("r") as f:
                data = json.load(f)
            records.append(RunRecord.from_dict(data))
        except Exception:
            continue
    # Sort by run_id (roughly chronological)
    records.sort(key=lambda r: r.run_id)
    return records


def get_run_by_index(idx: int) -> Optional[RunRecord]:
    runs = load_all_runs()
    if idx < 1 or idx > len(runs):
        return None
    return runs[idx - 1]
