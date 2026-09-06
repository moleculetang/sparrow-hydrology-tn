from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime


EXPECTED_PREFIX = Path(r"D:\ProgramData\anaconda3\envs\sparrow")
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "sparrow_environment_audit.json"

PACKAGE_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "pyarrow": "pyarrow",
    "torch": "torch",
    "rasterio": "rasterio",
    "netCDF4": "netCDF4",
    "seaborn": "seaborn",
}


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _probe(code: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    conda_paths = [
        EXPECTED_PREFIX,
        EXPECTED_PREFIX / "Library" / "mingw-w64" / "bin",
        EXPECTED_PREFIX / "Library" / "usr" / "bin",
        EXPECTED_PREFIX / "Library" / "bin",
        EXPECTED_PREFIX / "Scripts",
        EXPECTED_PREFIX / "bin",
    ]
    env["PATH"] = os.pathsep.join(str(path) for path in conda_paths) + os.pathsep + env.get("PATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
        "runtime_path_head": env["PATH"].split(os.pathsep)[:6],
    }


def _probe_current_process(modules: tuple[str, ...]) -> dict[str, object]:
    try:
        for module in modules:
            importlib.import_module(module)
    except Exception as exc:  # environment audit must preserve the concrete import error
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "passed": False,
            "runtime_path_head": os.environ.get("PATH", "").split(os.pathsep)[:6],
        }
    return {
        "returncode": 0,
        "stdout": "ok",
        "stderr": "",
        "passed": True,
        "runtime_path_head": os.environ.get("PATH", "").split(os.pathsep)[:6],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    prefix_ok = _norm(Path(sys.prefix)) == _norm(EXPECTED_PREFIX)
    unsafe_kmp = os.environ.get("KMP_DUPLICATE_LIB_OK")

    packages: dict[str, str | None] = {}
    for label, distribution in PACKAGE_NAMES.items():
        try:
            packages[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[label] = None

    dlls = []
    for path in EXPECTED_PREFIX.rglob("libiomp5md.dll"):
        dlls.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    probes = {
        "scientific_stack": _probe_current_process(
            ("numpy", "pandas", "scipy", "sklearn", "pyarrow")
        ),
        "spatial_stack": _probe_current_process(
            ("rasterio", "netCDF4", "seaborn")
        ),
        "torch_only": _probe(
            "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
        ),
        "numpy_then_torch": _probe("import numpy,torch; print('ok')"),
        "torch_then_numpy": _probe("import torch,numpy; print('ok')"),
    }

    all_packages_present = all(version is not None for version in packages.values())
    cpu_probes_passed = bool(probes["scientific_stack"]["passed"]) and bool(
        probes["spatial_stack"]["passed"]
    )
    torch_probes_passed = all(
        bool(probes[name]["passed"])
        for name in ("torch_only", "numpy_then_torch", "torch_then_numpy")
    )
    all_probes_passed = cpu_probes_passed and torch_probes_passed
    duplicate_openmp_runtime = len(dlls) > 1
    cpu_runtime_allowed = (
        prefix_ok
        and unsafe_kmp is None
        and all_packages_present
        and cpu_probes_passed
    )
    torch_runtime_allowed = (
        cpu_runtime_allowed
        and torch_probes_passed
    )
    passed = torch_runtime_allowed

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if passed else "FAIL",
        "expected_environment": "sparrow",
        "expected_prefix": str(EXPECTED_PREFIX),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_version": sys.version,
        "platform": platform.platform(),
        "prefix_ok": prefix_ok,
        "unsafe_KMP_DUPLICATE_LIB_OK_present": unsafe_kmp is not None,
        "packages": packages,
        "openmp_runtime_files": dlls,
        "duplicate_openmp_runtime": duplicate_openmp_runtime,
        "duplicate_openmp_file_status": (
            "FILES_PRESENT_BUT_ALL_IMPORT_ORDERS_PASS"
            if duplicate_openmp_runtime and torch_probes_passed
            else "UNRESOLVED_RUNTIME_CONFLICT"
            if duplicate_openmp_runtime
            else "SINGLE_RUNTIME_FILE"
        ),
        "import_probes": probes,
        "all_packages_present": all_packages_present,
        "cpu_probes_passed": cpu_probes_passed,
        "torch_probes_passed": torch_probes_passed,
        "all_probes_passed": all_probes_passed,
        "cpu_runtime_allowed": cpu_runtime_allowed,
        "torch_runtime_allowed": torch_runtime_allowed,
        "policy": {
            "formal_python": str(EXPECTED_PREFIX / "python.exe"),
            "base_or_system_python": "forbidden",
            "KMP_DUPLICATE_LIB_OK": "forbidden",
            "text_encoding": "UTF-8",
            "cpu_reservoir_preflight": "allowed only when cpu_runtime_allowed is true",
            "torch_or_gpu_fit": "allowed only when torch_runtime_allowed is true",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
