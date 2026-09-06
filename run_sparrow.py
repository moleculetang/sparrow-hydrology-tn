"""Simple CLI entrypoint for running SPARROW from this workspace.

Usage examples:
  python run_sparrow.py --config 2_Code/config/puget_config.py
  python run_sparrow.py --config my_config.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "2_Code"
from runtime_environment import assert_sparrow_runtime  # noqa: E402

RUNTIME_IDENTITY = assert_sparrow_runtime()
PKG_NAME = "sparrow_py"
if PKG_NAME not in sys.modules:
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(CODE_DIR)]  # type: ignore[attr-defined]
    sys.modules[PKG_NAME] = pkg

from sparrow_py.cli.runner import run_main  # type: ignore  # noqa: E402


def _load_python_config(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("sparrow_user_config", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_config"):
        cfg = module.get_config()
        if isinstance(cfg, dict):
            return dict(cfg)
    if hasattr(module, "CONFIG") and isinstance(module.CONFIG, dict):
        return dict(module.CONFIG)
    raise RuntimeError("Config .py must define CONFIG dict or get_config()")


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _load_python_config(path)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required for YAML config files") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("YAML config root must be a mapping")
        return dict(data)
    raise RuntimeError(f"Unsupported config extension: {suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SPARROW workflow")
    parser.add_argument("--config", required=True, help="Path to config (.py/.json/.yml)")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = _load_config(cfg_path)
    result = run_main(cfg)

    for msg in result.warnings:
        print(f"WARNING: {msg}")
    for msg in result.errors:
        print(f"ERROR: {msg}", file=sys.stderr)

    if result.errors:
        return 1
    print("SPARROW run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
