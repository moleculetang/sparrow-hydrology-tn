"""Command-line entry points for SPARROW."""

from .runner import FileTableStore, RunResult, run_main

__all__ = ["FileTableStore", "RunResult", "run_main"]
