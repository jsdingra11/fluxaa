from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

import click
import yaml

from .runner import (
    run_local_job,
    load_all_runs,
    get_run_by_index,
    RUNS_DIR,
)


@click.group()
def cli():
    """Fluxaa / Orchestrator CLI."""
    pass


# -------------------------------
# Core "run" command
# -------------------------------
@cli.command("run")
@click.argument("script", type=str)
@click.option("--checkpoint-dir", required=True, type=str, help="Directory for checkpoints.")
@click.option("--backend", default="local", type=click.Choice(["local"]), show_default=True)
@click.option("--name", default="", type=str, help="Optional name/tag for this run.")
@click.option("--max-retries", default=0, type=int, show_default=True)
@click.option(
    "--extra-args",
    default="",
    type=str,
    help="Extra args to pass to the script, e.g. \"--epochs 10 --lr 0.001\"",
)
@click.option(
    "--python-executable",
    default=None,
    type=str,
    help="Python executable to use (defaults to current Python).",
)
def run_cmd(
    script: str,
    checkpoint_dir: str,
    backend: str,
    name: str,
    max_retries: int,
    extra_args: str,
    python_executable: Optional[str],
):
    """Run a training script under the orchestrator."""
    extra_args_list: List[str] = []
    if extra_args.strip():
        extra_args_list = extra_args.strip().split()

    if backend != "local":
        raise click.ClickException(f"Backend '{backend}' not implemented yet.")

    record = run_local_job(
        script=script,
        checkpoint_dir=checkpoint_dir,
        max_retries=max_retries,
        extra_args=extra_args_list,
        name=name,
        python_executable=python_executable or "python3",
    )

    if record.status != "SUCCESS":
        raise SystemExit(1)


# -------------------------------
# run-config command (YAML)
# -------------------------------
@cli.command("run-config")
@click.argument("config_path", type=str)
def run_config_cmd(config_path: str):
    """
    Run a job from a YAML config file.

    YAML structure:
      name: my_run
      script: examples/train_pytorch_example.py
      checkpoint_dir: /tmp/ckpts
      backend: local
      python_executable: python3
      max_retries: 2
      extra_args:
        - --epochs
        - "5"
        - --lr
        - "0.001"
    """
    path = Path(config_path)
    if not path.exists():
        raise click.ClickException(f"Config not found: {config_path}")

    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", "")
    script = cfg["script"]
    checkpoint_dir = cfg["checkpoint_dir"]
    backend = cfg.get("backend", "local")
    python_executable = cfg.get("python_executable", "python3")
    max_retries = int(cfg.get("max_retries", 0))
    extra_args = cfg.get("extra_args", []) or []

    if backend != "local":
        raise click.ClickException(f"Backend '{backend}' not implemented yet.")

    print()
    print(f"[orchestrator][run-config] Loaded config from {config_path}")
    print(f"[orchestrator][run-config] Name: {name}")
    print(f"[orchestrator][run-config] Script: {script}")
    print(f"[orchestrator][run-config] Backend: {backend}")
    print(f"[orchestrator][run-config] Checkpoint dir: {checkpoint_dir}")
    print(f"[orchestrator][run-config] Extra args: {extra_args}")
    print()

    record = run_local_job(
        script=script,
        checkpoint_dir=checkpoint_dir,
        max_retries=max_retries,
        extra_args=extra_args,
        name=name,
        python_executable=python_executable,
    )

    if record.status != "SUCCESS":
        raise SystemExit(1)


# -------------------------------
# list-runs
# -------------------------------
@cli.command("list-runs")
def list_runs_cmd():
    """List all recorded runs."""
    runs = load_all_runs()
    if not runs:
        print("No runs found.")
        return

    print(
        "IDX   STATUS  ATT  BACKEND  NAME                SCRIPT                        RUN_ID"
    )
    print("-" * 94)
    for idx, r in enumerate(runs, start=1):
        name = (r.name or "")[:18]
        script = Path(r.script).name[:28]
        print(
            f"{idx:3d}  {r.status:>6}  {r.attempts:3d}    {r.backend:<5}  {name:<18}  {script:<28}  {r.run_id}"
        )


# -------------------------------
# status
# -------------------------------
@cli.command("status")
@click.argument("run_idx", type=int)
def status_cmd(run_idx: int):
    """Show detailed status for a run by index."""
    run = get_run_by_index(run_idx)
    if run is None:
        raise click.ClickException(f"Run index out of range: {run_idx}")

    print(f"Run index:      {run_idx}")
    print(f"Run ID:         {run.run_id}")
    print(f"Name:           {run.name}")
    print(f"Script:         {run.script}")
    print(f"Backend:        {run.backend}")
    print(f"Status:         {run.status}")
    print(f"Attempts:       {run.attempts}")
    print(f"Last exit code: {run.last_exit_code}")
    print(f"Checkpoint dir: {run.checkpoint_dir}")
    print(f"Start (UTC):    {run.start_time}")
    print(f"End   (UTC):    {run.end_time}")
    print(f"Duration (s):   {run.duration_s}")
    print(f"Extra args:     {run.extra_args}")
    if run.metrics is not None:
        print(f"Metrics:        {json.dumps(run.metrics, indent=2)}")


# -------------------------------
# rerun
# -------------------------------
@cli.command("rerun")
@click.argument("run_idx", type=int)
def rerun_cmd(run_idx: int):
    """Rerun a previous run with the same settings."""
    run = get_run_by_index(run_idx)
    if run is None:
        raise click.ClickException(f"Run index out of range: {run_idx}")

    print(f"[orchestrator][rerun] Using log: {run.log_path.name}")
    print(f"[orchestrator][rerun] Script: {run.script}")
    print(f"[orchestrator][rerun] Backend: {run.backend}")
    print(f"[orchestrator][rerun] Checkpoint dir: {run.checkpoint_dir}")
    print(f"[orchestrator][rerun] Max retries: {run.attempts}")
    print(f"[orchestrator][rerun] Extra args: {run.extra_args}")
    print()

    if run.backend != "local":
        raise click.ClickException(f"Backend '{run.backend}' not implemented yet for rerun.")

    record = run_local_job(
        script=run.script,
        checkpoint_dir=run.checkpoint_dir,
        max_retries=run.attempts,
        extra_args=run.extra_args,
        name=run.name,
        python_executable="python3",
    )

    if record.status != "SUCCESS":
        raise SystemExit(1)


# -------------------------------
# top-runs (new)
# -------------------------------
@cli.command("top-runs")
@click.option(
    "--metric",
    required=True,
    type=str,
    help="Name of metric key to sort by (e.g. final_test_accuracy, best_val_accuracy).",
)
@click.option(
    "--limit",
    default=10,
    type=int,
    show_default=True,
    help="Maximum number of runs to show.",
)
def top_runs_cmd(metric: str, limit: int):
    """
    Show top runs sorted by a metric stored in run.metrics.

    Only runs with that metric present are considered.
    """
    runs = load_all_runs()
    runs_with_metrics = [
        r for r in runs if r.metrics is not None and metric in r.metrics
    ]

    if not runs_with_metrics:
        print(f"No runs found with metric '{metric}'.")
        return

    sorted_runs = sorted(
        runs_with_metrics,
        key=lambda r: r.metrics.get(metric, float("-inf")),
        reverse=True,
    )

    print(
        f"IDX   STATUS  {metric:>16}  NAME                SCRIPT                        RUN_ID"
    )
    print("-" * 110)

    for idx, r in enumerate(sorted_runs[:limit], start=1):
        val = r.metrics.get(metric)
        name = (r.name or "")[:18]
        script = Path(r.script).name[:28]
        print(
            f"{idx:3d}  {r.status:>6}  {val:16.6f}  {name:<18}  {script:<28}  {r.run_id}"
        )


if __name__ == "__main__":
    cli()
