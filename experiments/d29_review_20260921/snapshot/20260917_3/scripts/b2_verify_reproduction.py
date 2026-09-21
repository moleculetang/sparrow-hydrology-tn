"""B2: show that round B's product is the frozen screen's product, by reproducing it.

Round B re-runs round A's chain in its own tree (`b1`).  That is only worth anything if the result is
then held against the thing the screen was frozen on, so this compares every file round A's adopted arm
produced with the file round B produced in its place.

Three outcomes, reported separately because they mean different things:

  * **byte-identical** -- the strongest form, and the expected one: the chain is deterministic, its seed
    is fixed, and round B ran the same fork bytes on inputs whose masking signature was recounted in
    `b0`.  `20260828_43` was reproduced this way too (round A's V3, all columns 0.0).
  * **equal, bytes differ** -- not a failure.  A parquet file's bytes can differ while the table is
    identical (writer metadata, row-group padding), and a `.pt` checkpoint records the directory it was
    written in, so its bytes *must* differ between two trees even when it holds the same fit.  Parquet
    is compared column by column; checkpoints are compared by flattened tensor key with exact equality,
    and a recorded path is accepted only if, made relative to its own tree root, it names the same
    position in both trees -- an appearance in `path_only_differences`, not an exemption.
  * **differs** -- a real mismatch: the product round B built is not the product the screen was frozen
    on.
  * **uncomparable** -- a file this script has no reader for.  Counted as a failure, not skipped: a
    reproduction that quietly excludes the types it cannot read is not a reproduction.

Writes `reports/b2_reproduction.json`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


T = Path(r"E:\SPARROW") / "5_Test"
RUN = T / "20260917_3"
A = T / "20260917_2"

OURS = RUN / "work" / "screen" / "outputs"
THEIRS = A / "arms" / "R74" / "outputs"
REPORTS = RUN / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal_column(values_a, values_b, chunk: int = 200_000) -> tuple[bool, str]:
    """Whether two Arrow columns are equal, and how it was decided.

    `pc.equal` has no kernel for nested types (`list<...>`, which these products do carry), so those
    fall back to chunked Python comparison.  Chunked rather than wholesale so that a large nested
    column cannot pull the whole thing into Python objects at once, and recorded by name so the report
    says which columns were proved by which method -- a fallback that silently produced `True` would be
    a hole in the reproduction claim.
    """
    try:
        equal_or_null = pc.equal(values_a, values_b)
    except pa.ArrowNotImplementedError:
        for start in range(0, len(values_a), chunk):
            if (values_a.slice(start, chunk).to_pylist()
                    != values_b.slice(start, chunk).to_pylist()):
                return False, "python_chunked"
        return True, "python_chunked"

    # `equal` yields null wherever *either* side is null, and it does so for both the agree
    # (null, null) and disagree (null, 5) cases.  Folding those to False would call every masked cell a
    # mismatch -- and these products are mostly NaN by construction, so a file would fail to reproduce
    # itself.  Nulls are equal when both sides are null and unequal otherwise, which is what filling
    # with `both_null` says: (null, null) -> True, (null, 5) -> False.
    return bool(pc.all(pc.fill_null(equal_or_null,
                                    pc.and_(pc.is_null(values_a), pc.is_null(values_b)))).as_py()), "arrow"


def compare_parquet(left: Path, right: Path) -> dict:
    """Column-by-column equality, one column at a time, and only when the bytes already differ.

    Streaming by column rather than loading both tables: the largest artifact here is 1.15 GB, and two
    copies of it in memory at once is a cost this comparison does not need to pay to answer the
    question.
    """
    left_meta, right_meta = pq.ParquetFile(left), pq.ParquetFile(right)
    a_names = left_meta.schema_arrow.names
    b_names = right_meta.schema_arrow.names
    detail = {
        "shape_a": [left_meta.metadata.num_rows, len(a_names)],
        "shape_b": [right_meta.metadata.num_rows, len(b_names)],
        "columns_identical": a_names == b_names,
    }
    if not detail["columns_identical"]:
        detail["columns_only_in_a"] = [c for c in a_names if c not in b_names]
        detail["columns_only_in_b"] = [c for c in b_names if c not in a_names]
        detail["equal"] = False
        return detail

    differing, magnitudes, methods = [], {}, {}
    for column in a_names:
        values_a = pq.read_table(left, columns=[column]).column(column)
        values_b = pq.read_table(right, columns=[column]).column(column)
        if len(values_a) != len(values_b) or values_a.type != values_b.type:
            differing.append(column)
            methods[column] = "shape_or_dtype"
            magnitudes[column] = {"max_abs_difference": None,
                                  "nulls_a": values_a.null_count, "nulls_b": values_b.null_count}
            continue
        same, method = equal_column(values_a, values_b)
        methods[column] = method
        if same:
            continue
        differing.append(column)
        # How far apart -- a last-bit rounding and a wrong panel are both "not equal", and the report
        # should not leave the reader to guess which one happened.  Null counts travel with the
        # magnitude because the magnitude is taken over non-null pairs only: a column that differs
        # solely in *which* cells are masked would otherwise be reported as a maximum difference of 0.
        try:
            delta = pc.max(pc.abs(pc.subtract(pc.cast(values_a, "float64"),
                                              pc.cast(values_b, "float64")))).as_py()
            magnitude = None if delta is None else float(delta)
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid):   # non-numeric or nested column
            magnitude = None
        magnitudes[column] = {"max_abs_difference": magnitude,
                              "nulls_a": values_a.null_count, "nulls_b": values_b.null_count}
    detail["comparison_method"] = methods
    detail["differing_columns"] = differing
    detail["equal"] = not differing
    detail["max_abs_difference"] = magnitudes
    return detail


def _position_within_tree(value: str, root: Path | None) -> str | None:
    """The path's location relative to its own tree root, or None if it is not under that root."""
    if root is None:
        return None
    try:
        return Path(value).relative_to(root).as_posix()
    except (ValueError, TypeError):
        return None


def compare_pt(left: Path, right: Path, root_left: Path | None = None,
               root_right: Path | None = None) -> dict:
    """Tensors by key, so a checkpoint that only stores different paths is not read as a different fit.

    A checkpoint records *where* its inputs were read from, so the two trees' checkpoints necessarily
    differ there.  That is not waved through as "a path, ignore it": a differing string is accepted only
    when, made relative to its own tree root, it names the **same position in both trees** -- which is
    what proves each checkpoint points at its own tree's copy of the same file.  A string that names a
    different file, or escapes its root, stays a mismatch.
    """
    import torch

    a = torch.load(left, map_location="cpu", weights_only=False)
    b = torch.load(right, map_location="cpu", weights_only=False)

    def entries(obj, prefix=""):
        out = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                out.update(entries(value, f"{prefix}{key}."))
        elif isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                out.update(entries(value, f"{prefix}{index}."))
        else:
            out[prefix.rstrip(".")] = obj
        return out

    flat_a, flat_b = entries(a), entries(b)
    keys_equal = sorted(flat_a) == sorted(flat_b)
    differing, path_only = [], []
    for key in sorted(set(flat_a) & set(flat_b)):
        value_a, value_b = flat_a[key], flat_b[key]
        if isinstance(value_a, str) and isinstance(value_b, str):
            if value_a == value_b:
                continue
            position_a = _position_within_tree(value_a, root_left)
            position_b = _position_within_tree(value_b, root_right)
            if position_a is not None and position_a == position_b:
                path_only.append({"key": key, "position_in_tree": position_a,
                                  "a": value_a, "b": value_b})
            else:
                differing.append({"key": key, "kind": "string", "a": value_a, "b": value_b,
                                  "a_position": position_a, "b_position": position_b})
            continue
        try:
            if value_a == value_b:
                continue
        except Exception:                                    # noqa: BLE001 - tensors raise on `==`
            pass
        try:
            import torch as _torch
            if _torch.allclose(_torch.as_tensor(value_a, dtype=_torch.float64),
                               _torch.as_tensor(value_b, dtype=_torch.float64), rtol=0, atol=0):
                continue
        except Exception:                                    # noqa: BLE001
            pass
        differing.append({"key": key, "kind": type(value_a).__name__})
    return {"keys_identical": keys_equal,
            "keys_only_in_a": sorted(set(flat_a) - set(flat_b)),
            "keys_only_in_b": sorted(set(flat_b) - set(flat_a)),
            "differing_entries": differing,
            "path_only_differences": path_only}


def main() -> None:
    if not OURS.is_dir():
        raise RuntimeError(f"{OURS} 不存在；先跑 b1_run_product.py")
    if not THEIRS.is_dir():
        raise RuntimeError(f"{THEIRS} 不存在；筛选锁件所指的被冻结臂缺失")

    ours = {p.relative_to(OURS).as_posix(): p for p in OURS.rglob("*") if p.is_file()}
    theirs = {p.relative_to(THEIRS).as_posix(): p for p in THEIRS.rglob("*") if p.is_file()}

    entries, identical, content_equal, differ, uncomparable, absent = [], [], [], [], [], []
    for name in sorted(set(ours) | set(theirs)):
        if name not in ours or name not in theirs:
            absent.append({"artifact": name,
                           "in_round_b_only": name not in theirs,
                           "in_round_a_only": name not in ours})
            continue

        left, right = ours[name], theirs[name]
        entry = {"artifact": name, "sha256_round_b": sha256(left), "sha256_round_a": sha256(right),
                 "bytes_round_b": left.stat().st_size, "bytes_round_a": right.stat().st_size}
        if entry["sha256_round_b"] == entry["sha256_round_a"]:
            entry["verdict"] = "byte_identical"
            identical.append(name)
            entries.append(entry)
            continue

        suffix = Path(name).suffix.lower()
        if suffix == ".parquet":
            detail = compare_parquet(left, right)
            equal = bool(detail.get("equal"))
            entry["verdict"] = "parquet_equal" if equal else "differs"
            entry["detail"] = detail
        elif suffix == ".pt":
            # `left` is round B's file (the loop pairs `ours[name]` first), so its root is round B's.
            detail = compare_pt(left, right, OURS.parent, THEIRS.parent)
            equal = not detail["differing_entries"] and detail["keys_identical"]
            entry["verdict"] = "checkpoint_equal" if equal else "differs"
            entry["detail"] = detail
        else:
            # No reader for this type.  Recorded as its own class and counted as a failure: calling it
            # "differs" would claim a mismatch that was never measured, and dropping it silently would
            # claim a reproduction that was never proven.
            equal = False
            entry["verdict"] = "uncomparable"
            entry["detail"] = {"reason": f"本轮没有逐内容读取器（{suffix}）：既不能判为一致，也不能判为不一致"}
            uncomparable.append(name)
        if suffix in (".parquet", ".pt"):
            (content_equal if equal else differ).append(name)
        entries.append(entry)

    path_only = {entry["artifact"]: entry["detail"]["path_only_differences"]
                 for entry in entries
                 if (entry.get("detail") or {}).get("path_only_differences")}

    report = {
        "stage": "20260917_3",
        "purpose": (
            "B2: round B's product, held against the arm round A froze the screen on -- byte-identical "
            "where the chain is deterministic, and characterised where it is not"
        ),
        "round_b_outputs": str(OURS),
        "round_a_outputs": str(THEIRS),
        "counts": {
            "artifacts_compared": len(entries),
            "byte_identical": len(identical),
            "equal_but_not_byte_identical": len(content_equal),
            "differs": len(differ),
            "uncomparable": len(uncomparable),
            "present_in_one_tree_only": len(absent),
            "path_only_differences": sum(len(v) for v in path_only.values()),
        },
        "byte_identical": identical,
        "equal_but_not_byte_identical": content_equal,
        "differs": differ,
        "uncomparable": uncomparable,
        "present_in_one_tree_only": absent,
        "path_only_differences": path_only,
        "caveat": (
            "`.pt` 检查点会记录它自己写在哪棵树下，故两者的字节必然不同；上面按张量键与逐元素相等比较，"
            "其中字符串条目只在「各自相对于本树根之后指向同一位置」时才算一致，并单列进 "
            "path_only_differences —— 这不是豁免，而是一条更强的核对：它证明两个检查点各指向本树内同一个文件，"
            "而不是只证明基名相同。本轮没有把任何未比过的类型算作「一致」：既非 parquet 也非 `.pt` 的文件"
            "单列为 uncomparable，并计入未通过。"
        ),
        "entries": entries,
        "passed": not differ and not absent and not uncomparable,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "b2_reproduction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    counts = report["counts"]
    print("B2 复现核对：本轮产物 vs 被冻结臂的产物")
    print(f"  逐字节相同        {counts['byte_identical']}")
    print(f"  内容相同、字节不同 {counts['equal_but_not_byte_identical']}")
    print(f"  不一致            {counts['differs']}")
    print(f"  无法逐内容比较     {counts['uncomparable']}")
    print(f"  只在一棵树里有     {counts['present_in_one_tree_only']}")
    if path_only:
        print(f"  仅路径前缀不同     {counts['path_only_differences']} 处（已核为各自树内同一位置）：")
        for artifact, items in sorted(path_only.items()):
            for item in items:
                print(f"    {artifact}: {item['key']} → {item['position_in_tree']}")
    for name in differ + uncomparable:
        entry = next(e for e in entries if e["artifact"] == name)
        print(f"    *** {name}：{entry.get('detail')}")
    if report["passed"]:
        print(f"\n通过：{counts['artifacts_compared']} 件产物全部复现")
    else:
        print("\n未通过：见上")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
