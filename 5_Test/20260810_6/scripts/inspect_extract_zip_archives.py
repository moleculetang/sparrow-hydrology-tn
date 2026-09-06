from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


def safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return not path.is_absolute() and ".." not in path.parts


def inspect_archive(path: Path) -> dict[str, object]:
    with ZipFile(path) as archive:
        members = archive.infolist()
        unsafe = [item.filename for item in members if not safe_member(item.filename)]
        suffixes: dict[str, int] = {}
        for item in members:
            if item.is_dir():
                continue
            suffix = Path(item.filename).suffix.lower() or "<none>"
            suffixes[suffix] = suffixes.get(suffix, 0) + 1
        return {
            "archive": str(path),
            "member_count": len(members),
            "file_count": sum(not item.is_dir() for item in members),
            "compressed_bytes": path.stat().st_size,
            "uncompressed_bytes": sum(item.file_size for item in members),
            "unsafe_members": unsafe,
            "suffix_counts": dict(sorted(suffixes.items())),
            "sample_members": [item.filename for item in members[:12]],
        }


def extract_archive(path: Path) -> Path:
    destination = path.with_suffix("")
    destination.mkdir(parents=False, exist_ok=True)
    destination_resolved = destination.resolve()
    with ZipFile(path) as archive:
        for item in archive.infolist():
            if not safe_member(item.filename):
                raise ValueError(f"Unsafe ZIP member in {path}: {item.filename}")
            target = (destination / item.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"ZIP member escapes destination in {path}: {item.filename}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise BadZipFile(f"CRC failure in {path}: {bad_member}")
        archive.extractall(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    archives = sorted(args.root.rglob("*.zip"))
    results: list[dict[str, object]] = []
    for archive_path in archives:
        result = inspect_archive(archive_path)
        if result["unsafe_members"]:
            raise ValueError(f"Unsafe members found in {archive_path}")
        if args.extract:
            result["extracted_to"] = str(extract_archive(archive_path))
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    print(
        json.dumps(
            {
                "archive_count": len(results),
                "total_compressed_bytes": sum(int(x["compressed_bytes"]) for x in results),
                "total_uncompressed_bytes": sum(int(x["uncompressed_bytes"]) for x in results),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
