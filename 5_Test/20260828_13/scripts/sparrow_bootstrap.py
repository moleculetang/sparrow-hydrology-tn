from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


EXPECTED_PREFIX = Path(r"D:\ProgramData\anaconda3\envs\sparrow")


def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(path))


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: sparrow_bootstrap.py TARGET_SCRIPT.py [arguments...]")
    if _norm(sys.prefix) != _norm(EXPECTED_PREFIX):
        raise RuntimeError(f"wrong Python prefix: {sys.prefix}")
    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        raise RuntimeError(
            "KMP_DUPLICATE_LIB_OK is forbidden because it can hide an unsafe OpenMP conflict"
        )

    runtime_dirs = [
        EXPECTED_PREFIX,
        EXPECTED_PREFIX / "Library" / "bin",
        EXPECTED_PREFIX / "Library" / "mingw-w64" / "bin",
        EXPECTED_PREFIX / "Library" / "usr" / "bin",
        EXPECTED_PREFIX / "DLLs",
        EXPECTED_PREFIX / "Scripts",
        EXPECTED_PREFIX / "bin",
    ]
    runtime_path = os.pathsep.join(str(path) for path in runtime_dirs)
    inherited_path = os.environ.get("Path") or os.environ.get("PATH", "")
    merged_path = runtime_path + os.pathsep + inherited_path
    os.environ["Path"] = merged_path
    os.environ["PATH"] = merged_path
    os.putenv("Path", merged_path)
    os.putenv("PATH", merged_path)
    os.environ.setdefault("GDAL_DATA", str(EXPECTED_PREFIX / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(EXPECTED_PREFIX / "Library" / "share" / "proj"))
    dll_handles = []
    if hasattr(os, "add_dll_directory"):
        for path in runtime_dirs:
            if path.is_dir():
                dll_handles.append(os.add_dll_directory(str(path)))

    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
