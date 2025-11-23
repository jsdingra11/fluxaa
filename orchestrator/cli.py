# orchestrator/cli.py

import sys
import json
from pathlib import Path
from typing import Optional, List

import click

from .runner import JobConfig, run_ml_job


@click.group()
def cli():
    """Simple ML job orchestrator."""
    pass


@cli.command()
@click.argument("script_path", type=click.Path(exists=True))
@click.option(
    "--checkpoint-dir",
    "-c",
    type=click.Path(),
    required=True,
    help="Directory where checkpoints are saved/loaded.",
)
@click.option(
    "--name",
    "-n",
    type=str,
    default=None,
    help="Optional human-friendly name/label for this run.",
)
@click.option(
    "--max-retries",
    "-r",
    type=int,
    default=3,
    show_default=True,
    help="Maximum number of retries on failure.",
)
@click.option(
    "--python-exe",
    "-p",
    type=click.Path(),
    default=sys.executable,
    show_default=True,
    help="Python executable to run the script with.",
)
@click.option(
    "--extra-args",
    "-e",
    type=str,
    default="",
    help="Extra args to pass to the script (single string, space-separated).",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["local", "slurm"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Execution backend.",
)
@click.option(
    "--slurm-partition",
    type=str,
    default=None,
    help="Slurm partition to use (if backend=slurm).",
)
@click.option(
    "--slurm-gpus",
    type=int,
    default=1,
    show_default=True,
    help="Number of GPUs to request via Slurm (if backend=slurm).",
)
@click.option(
    "--slurm-time",
    type=str,
    default="01:00:00",
    show_default=True,
    help="Time limit for the Slurm job (if backend=slurm).",
)
@click.option(
    "--slurm-job-name",
    type=str,
    default="orc_job",
    show_default=True,
    help="Slurm job name (if backend=slurm).",
)
def run(
    script_path: str,
    checkpoint_dir: str,
    name: Optional[str],
    max_retries: int,
    python_exe: str,
    extra_args: str,
    backend: str,
    slurm_partition: Optional[str],
    slurm_gpus: int,
    slurm_time: str,
    slurm_job_name: str,
):
    """
    Run an ML training script with auto-resume on failure.
    """
    extra_args_list: Optional[List[str]] = (
        extra_args.split() if extra_args.strip() else []
    )

    config = JobConfig(
        script_path=script_path,
        checkpoint_dir=checkpoint_dir,
        max_retries=max_retries,
        python_executable=python_exe,
        extra_args=extra_args_list,
        backend=backend.lower(),
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_time=slurm_time,
        slurm_job_name=slurm_job_name,
        run_name=name,
    )

    success = run_ml_job(config)
    sys.exit(0 if success else 1)


@cli.command("list-runs")
def list_runs():
    """
    List logged runs from the 'runs/' directory.
    """
    runs_dir = Path("runs")
    if not runs_dir.exists():
        click.echo("No runs directory found.")
        sys.exit(0)

    log_files = sorted(runs_dir.glob("run_*.json"))
    if not log_files:
        click.echo("No runs logged yet.")
        sys.exit(0)

    rows = []
    for idx, lf in enumerate(log_files, start=1):
        try:
            with lf.open("r") as f:
                data = json.load(f)
        except Exception:
            continue

        rows.append(
            {
                "idx": idx,
                "file": lf.name,
                "run_id": data.get("run_id", lf.stem),
                "name": data.get("name", "") or "",
                "script": Path(data.get("script", "")).name,
                "success": data.get("success"),
                "attempts": data.get("attempts"),
                "backend": data.get("backend", "local"),
                "timestamp_start": data.get("timestamp_start_utc", ""),
            }
        )

    click.echo(
        f"{'IDX':>3}  {'STATUS':>7}  {'ATT':>3}  {'BACKEND':>7}  {'NAME':<18}  {'SCRIPT':<28}  {'RUN_ID'}"
    )
    click.echo("-" * 110)
    for r in rows:
        status = "OK" if r["success"] else "FAIL"
        click.echo(
            f"{r['idx']:>3}  {status:>7}  {r['attempts']:>3}  {r['backend']:>7}  {r['name'][:18]:<18}  {r['script']:<28}  {r['run_id']}"
        )


@cli.command("status")
@click.argument("run_idx", type=int)
def status(run_idx: int):
    """
    Show detailed status for a single run by index.
    """
    runs_dir = Path("runs")
    log_files = sorted(runs_dir.glob("run_*.json"))
    if not log_files:
        click.echo("No runs logged yet.")
        sys.exit(1)

    if run_idx < 1 or run_idx > len(log_files):
        click.echo(f"Invalid index {run_idx}. Valid range is 1..{len(log_files)}")
        sys.exit(1)

    lf = log_files[run_idx - 1]
    with lf.open("r") as f:
        data = json.load(f)

    name = data.get("name") or ""
    script = data.get("script", "")
    backend = data.get("backend", "local")
    success = data.get("success")
    attempts = data.get("attempts")
    start_ts = data.get("timestamp_start_utc")
    end_ts = data.get("timestamp_end_utc")
    duration = data.get("duration_seconds")
    extra_args = data.get("extra_args", [])
    last_exit = data.get("last_exit_code")
    checkpoint_dir = data.get("checkpoint_dir", "")
    run_id = data.get("run_id", lf.stem)

    status_str = "SUCCESS" if success else "FAIL"
    click.echo(f"Run index:      {run_idx}")
    click.echo(f"Run ID:         {run_id}")
    click.echo(f"Name:           {name}")
    click.echo(f"Script:         {script}")
    click.echo(f"Backend:        {backend}")
    click.echo(f"Status:         {status_str}")
    click.echo(f"Attempts:       {attempts}")
    click.echo(f"Last exit code: {last_exit}")
    click.echo(f"Checkpoint dir: {checkpoint_dir}")
    click.echo(f"Start (UTC):    {start_ts}")
    click.echo(f"End   (UTC):    {end_ts}")
    click.echo(f"Duration (s):   {duration}")
    click.echo(f"Extra args:     {extra_args}")


@cli.command("rerun")
@click.argument("run_idx", type=int)
@click.option(
    "--max-retries",
    "-r",
    type=int,
    default=None,
    help="Override max retries for this rerun.",
)
@click.option(
    "--backend",
    "-b",
    type=str,
    default=None,
    help="Override backend for this rerun ('local' or 'slurm').",
)
@click.option(
    "--extra-args",
    "-e",
    type=str,
    default=None,
    help="Override extra args (space-separated) for this rerun.",
)
def rerun(run_idx: int, max_retries: Optional[int], backend: Optional[str], extra_args: Optional[str]):
    """
    Rerun a past job by its index from list-runs.
    """
    runs_dir = Path("runs")
    log_files = sorted(runs_dir.glob("run_*.json"))
    if not log_files:
        click.echo("No runs logged yet.")
        sys.exit(1)

    if run_idx < 1 or run_idx > len(log_files):
        click.echo(f"Invalid index {run_idx}. Valid range is 1..{len(log_files)}")
        sys.exit(1)

    log_file = log_files[run_idx - 1]
    with log_file.open("r") as f:
        data = json.load(f)

    script = data.get("script")
    checkpoint_dir = data.get("checkpoint_dir")
    if not script or not checkpoint_dir:
        click.echo(f"Log {log_file.name} missing script or checkpoint_dir.")
        sys.exit(1)

    stored_backend = data.get("backend", "local")
    stored_max_retries = data.get("max_retries", 3)
    stored_python_exe = data.get("python_executable", sys.executable)
    stored_extra_args = data.get("extra_args", [])
    if isinstance(stored_extra_args, str):
        stored_extra_args = stored_extra_args.split()

    slurm_cfg = data.get("slurm", {}) or {}
    stored_partition = slurm_cfg.get("partition")
    stored_gpus = slurm_cfg.get("gpus", 1)
    stored_time = slurm_cfg.get("time", "01:00:00")
    stored_job_name = slurm_cfg.get("job_name", "orc_job")

    # Apply overrides
    final_backend = backend.lower() if backend else stored_backend
    final_max_retries = max_retries if max_retries is not None else stored_max_retries
    final_extra_args = stored_extra_args
    if extra_args is not None:
        final_extra_args = extra_args.split() if extra_args.strip() else []

    click.echo(f"[orchestrator][rerun] Using log: {log_file.name}")
    click.echo(f"[orchestrator][rerun] Script: {script}")
    click.echo(f"[orchestrator][rerun] Backend: {final_backend}")
    click.echo(f"[orchestrator][rerun] Checkpoint dir: {checkpoint_dir}")
    click.echo(f"[orchestrator][rerun] Max retries: {final_max_retries}")
    click.echo(f"[orchestrator][rerun] Extra args: {final_extra_args}")

    config = JobConfig(
        script_path=script,
        checkpoint_dir=checkpoint_dir,
        max_retries=final_max_retries,
        python_executable=stored_python_exe,
        extra_args=final_extra_args,
        backend=final_backend,
        slurm_partition=stored_partition,
        slurm_gpus=stored_gpus,
        slurm_time=stored_time,
        slurm_job_name=stored_job_name,
        run_name=data.get("name"),
    )

    success = run_ml_job(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    cli()
