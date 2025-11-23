# orc (working name)

Minimal ML job orchestrator:

- Runs training scripts with retries and auto-resume from checkpoints.
- Logs every run (config, status, duration, exit code).
- Lets you list runs, inspect status, and rerun previous jobs.

## Install (dev)

```bash
git clone <repo-url>
cd orchestrator-core
python -m venv venv
source venv/bin/activate
pip install -e .
````

## Usage

```bash
orc run examples/train_pytorch_example.py \
  --checkpoint-dir /tmp/exp1 \
  --name "baseline" \
  --extra-args "--epochs 5 --lr 0.001"

orc list-runs

orc status 1

orc rerun 1 --extra-args "--epochs 10 --lr 0.0005"
```
