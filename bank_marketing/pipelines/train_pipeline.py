from __future__ import annotations
import argparse
from pathlib import Path
from ..core.logging import get_logger
from .trainer import train_and_log

log = get_logger(__name__)

def cli():
    parser = argparse.ArgumentParser("Train pipeline")
    parser.add_argument("--input", default=str(Path.cwd() / "data" / "clean" / "bank.parquet"))
    parser.add_argument("--output", default=str(Path.cwd() / "artifacts"))
    parser.add_argument("--run-name", default="train_run")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    try:
        res = train_and_log(args.input, args.run_name, args.output)
        log.info(f"Run completed: {res['run_id']}")
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    cli()
