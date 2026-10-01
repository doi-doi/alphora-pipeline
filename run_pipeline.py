"""PyCharm entry point: edit CONFIG_PATH or add --config in Run Configuration."""

from pathlib import Path
import sys

from alphora_pipeline.cli import main

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "endpoint.json"

if __name__ == "__main__":
    main(["run", "--config", str(CONFIG_PATH), *sys.argv[1:]])
