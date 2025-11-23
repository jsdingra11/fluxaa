# orchestrator/runner.py

import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class JobConfig:
    def __init__(
        self,
        script_path: str,
        checkpoint_dir: str,
        max_retries: int = 3,
        python_executable: str = sys.executable,
        extra_args: Optional[List[str]] = None,
        backend: str = "local",
        slurm_partition: Optional[str] = None,
        slurm_gpus: int = 1,
        slurm_time: str = "01:00:00",
        slurm_job_name: str = "orc_job",
        run_name: Optional[str] = None,
    ):
        self.script_path = script_path
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = max_retries
        self.python_executable = python_executable
        self.extra_args = extra_args or []

        # backend can be "local" or "slurm"
        self.backend = backend

        # Slurm-specific options (used only if backend == "slurm")
        self.slurm_partition = slurm_partition
        self.slurm_gpus = slurm_gpus
        self.slurm_time = slurm_time
        self.slurm_job_name = slurm_job_name

        # Optional human-friendly name for this run
        self.run_name = run_name


def _log_run(
    config: JobConfig,
    success: bool,
    attempts: int,
    start_time_utc: str,
    end_time_utc: str,
    last_exit_code: int,
):
    """Log a summary of the run (including config) to runs/run_<id>.json."""
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    run_id = start_time_utc.replace(":", "").replace("-", "").replace(".", "")
    log_path = runs_dir / f"run_{run_id}.json"

    # duration in seconds
    try:
        t0 = datetime.fromisoformat(start_time_utc)
        t1 = datetime.fromisoformat(end_time_utc)
        duration_seconds = (t1 - t0).total_seconds()
    except Exception:
        duration_seconds = None

    data = {
        "run_id": run_id,
        "name": config.run_name,
        "script": str(config.script_path),
        "checkpoint_dir": str(config.checkpoint_dir),
        "success": success,
        "attempts": attempts,
        "timestamp_start_utc": start_time_utc,
        "timestamp_end_utc": end_time_utc,
        "duration_seconds": duration_seconds,
        "backend": config.backend,
        "python_executable": config.python_executable,
        "extra_args": config.extra_args,
        "max_retries": config.max_retries,
        "last_exit_code": last_exit_code,
        "slurm": {
            "partition": config.slurm_partition,
            "gpus": config.slurm_gpus,
            "time": config.slurm_time,
            "job_name": config.slurm_job_name,
        },
    }

    with log_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"[orchestrator] Logged run -> {log_path}")


def run_ml_job(config: JobConfig) -> bool:
    """
    Dispatch to the correct backend implementation.
    """
    if config.backend == "local":
        return _run_local(config)
    elif config.backend == "slurm":
        return _run_slurm(config)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")


def _run_local(config: JobConfig) -> bool:
    """
    Local execution using subprocess.
    """
    script = Path(config.script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.utcnow().isoformat()
    last_exit_code = -1
    attempt = 0

    while attempt <= config.max_retries:
        print(f"\n[orchestrator][local] Attempt {attempt + 1}/{config.max_retries + 1}")

        cmd = [
            config.python_executable,
            str(script),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]

        # On retry, tell the script to resume from checkpoint
        if attempt > 0:
            cmd.append("--resume")

        # Add any extra user arguments
        cmd.extend(config.extra_args)

        print(f"[orchestrator][local] Running: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd)
        last_exit_code = proc.wait()

        if last_exit_code == 0:
            print("[orchestrator][local] Job completed successfully ✅")
            end_dt = datetime.utcnow().isoformat()
            _log_run(
                config,
                success=True,
                attempts=attempt + 1,
                start_time_utc=start_dt,
                end_time_utc=end_dt,
                last_exit_code=last_exit_code,
            )
            return True

        print(f"[orchestrator][local] Job failed with exit code {last_exit_code} ❌")
        attempt += 1

        if attempt > config.max_retries:
            print("[orchestrator][local] Reached max retries. Giving up.")
            break

        time.sleep(5)

    end_dt = datetime.utcnow().isoformat()
    _log_run(
        config,
        success=False,
        attempts=attempt,
        start_time_utc=start_dt,
        end_time_utc=end_dt,
        last_exit_code=last_exit_code,
    )
    return False


# Slurm helpers unchanged; they still call _log_run at the end.

def _submit_slurm_job(job_script: Path) -> Optional[str]:
    """
    Submit a Slurm job script via sbatch and return the job id as string.
    """
    print(f"[orchestrator][slurm] Submitting job script: {job_script}")
    proc = subprocess.run(["sbatch", str(job_script)], capture_output=True, text=True)
    if proc.returncode != 0:
        print("[orchestrator][slurm] sbatch failed:")
        print(proc.stderr)
        return None

    stdout = proc.stdout.strip()
    print(f"[orchestrator][slurm] sbatch output: {stdout}")
    parts = stdout.split()
    job_id = parts[-1] if parts else None
    return job_id


def _wait_for_slurm_job(job_id: str) -> bool:
    """
    Poll Slurm until job finishes. Returns True if COMPLETED, False otherwise.
    Uses 'sacct'; may need adaption per cluster.
    """
    print(f"[orchestrator][slurm] Waiting for job {job_id} to finish ...")

    while True:
        proc = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "-P"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("[orchestrator][slurm] sacct not ready or job not yet visible, retrying ...")
            time.sleep(10)
            continue

        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if len(lines) < 2:
            time.sleep(10)
            continue

        state_line = lines[1]
        state = state_line.split("|")[0]
        print(f"[orchestrator][slurm] Job {job_id} state: {state}")

        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            return state == "COMPLETED"

        time.sleep(10)


def _run_slurm(config: JobConfig) -> bool:
    """
    Slurm-backed execution.
    Not testable on your Mac, but ready for a real cluster.
    """
    script = Path(config.script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path("slurm_jobs")
    runs_dir.mkdir(exist_ok=True)

    start_dt = datetime.utcnow().isoformat()
    last_exit_code = -1
    attempt = 0

    while attempt <= config.max_retries:
        print(f"\n[orchestrator][slurm] Attempt {attempt + 1}/{config.max_retries + 1}")

        py_cmd_parts = [
            config.python_executable,
            str(script),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
        if attempt > 0:
            py_cmd_parts.append("--resume")
        py_cmd_parts.extend(config.extra_args)
        py_cmd = " ".join(py_cmd_parts)

        ts = datetime.utcnow().isoformat().replace(":", "").replace("-", "").replace(".", "")
        job_script = runs_dir / f"job_{config.slurm_job_name}_{ts}_attempt{attempt+1}.sh"

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={config.slurm_job_name}",
            f"#SBATCH --time={config.slurm_time}",
            f"#SBATCH --gres=gpu:{config.slurm_gpus}",
            "#SBATCH --output=slurm-%j.out",
        ]
        if config.slurm_partition:
            lines.append(f"#SBATCH --partition={config.slurm_partition}")

        lines.append("")
        lines.append(f"echo 'Running command: {py_cmd}'")
        lines.append(py_cmd)

        job_script.write_text("\n".join(lines))
        print(f"[orchestrator][slurm] Wrote job script: {job_script}")

        job_id = _submit_slurm_job(job_script)
        if job_id is None:
            print("[orchestrator][slurm] Failed to submit job.")
            attempt += 1
            time.sleep(5)
            continue

        success = _wait_for_slurm_job(job_id)
        last_exit_code = 0 if success else 1
        if success:
            print("[orchestrator][slurm] Job completed successfully ✅")
            end_dt = datetime.utcnow().isoformat()
            _log_run(
                config,
                success=True,
                attempts=attempt + 1,
                start_time_utc=start_dt,
                end_time_utc=end_dt,
                last_exit_code=last_exit_code,
            )
            return True

        print("[orchestrator][slurm] Job failed according to Slurm state ❌")
        attempt += 1
        if attempt > config.max_retries:
            print("[orchestrator][slurm] Reached max retries. Giving up.")
            break
        time.sleep(5)

    end_dt = datetime.utcnow().isoformat()
    _log_run(
        config,
        success=False,
        attempts=attempt,
        start_time_utc=start_dt,
        end_time_utc=end_dt,
        last_exit_code=last_exit_code,
    )
    return False
