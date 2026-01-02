#!/usr/bin/env python3
"""
Secure-ish batch converter: mmCIF (.cif/.mmcif) -> PDB (.pdb)

Security-focused guardrails:
- Skips symlinks (prevents path-tricks)
- Enforces an input size limit (default 200 MB)
- Writes outputs atomically (temp file + rename)
- Avoids overwriting unless --overwrite is set
- Optionally fails on empty/coordinate-less structures (--strict)

Requires: gemmi  (pip install gemmi)
Usage examples:
  ./cif2pdb_secure.py myfile.cif
  ./cif2pdb_secure.py /path/to/cifs --outdir /path/to/pdbs --recursive
  ./cif2pdb_secure.py /path/to/cifs --recursive --strict --max-mb 50
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import gemmi


CIF_EXTS = {".cif", ".mmcif"}


def is_regular_file_no_symlink(p: Path) -> bool:
    try:
        st = p.lstat()  # does not follow symlinks
    except FileNotFoundError:
        return False
    # Reject symlinks
    if p.is_symlink():
        return False
    # Must be a regular file
    return (st.st_mode & 0o170000) == 0o100000  # stat.S_IFREG without importing stat


def within_dir(child: Path, parent: Path) -> bool:
    """Return True if child resolves under parent (prevents ../ tricks)."""
    try:
        child_res = child.resolve(strict=False)
        parent_res = parent.resolve(strict=True)
        child_res.relative_to(parent_res)
        return True
    except Exception:
        return False


def safe_iter_inputs(indir: Path, recursive: bool) -> list[Path]:
    paths: list[Path] = []
    if indir.is_file():
        paths.append(indir)
        return paths

    if not indir.is_dir():
        return paths

    if recursive:
        for root, dirs, files in os.walk(indir, followlinks=False):
            rootp = Path(root)
            # Optionally prune symlinked dirs (os.walk with followlinks=False won’t follow them anyway)
            for fn in files:
                p = rootp / fn
                if p.suffix.lower() in CIF_EXTS:
                    paths.append(p)
    else:
        for p in indir.iterdir():
            if p.is_file() and p.suffix.lower() in CIF_EXTS:
                paths.append(p)

    return sorted(paths)


def structure_has_atoms(st: gemmi.Structure) -> bool:
    if len(st) == 0:
        return False
    # gemmi.Model has count_atom_sites(), not count_atoms()
    total = 0
    for model in st:
        total += model.count_atom_sites()
        if total > 0:
            return True
    return False


def convert_one(
    cif_path: Path,
    outdir: Path,
    overwrite: bool,
    strict: bool,
    max_bytes: int,
    base_input_dir: Path | None,
) -> tuple[bool, str, Path | None]:
    """
    Returns: (ok, message, output_path_or_none)
    """
    # Basic checks
    if not is_regular_file_no_symlink(cif_path):
        return False, "skipped (not a regular file or is a symlink)", None

    # Optional: ensure input file is under the provided base dir (if converting a directory)
    if base_input_dir is not None and not within_dir(cif_path, base_input_dir):
        return False, "skipped (path escapes input dir)", None

    try:
        size = cif_path.stat().st_size
    except OSError as e:
        return False, f"skipped (stat failed: {e})", None

    if size > max_bytes:
        return False, f"skipped (file too large: {size} bytes > limit {max_bytes})", None

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (cif_path.stem + ".pdb")

    if out_path.exists() and not overwrite:
        return False, "skipped (output exists; use --overwrite)", out_path

    # Read + validate
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception as e:
        return False, f"failed (parse error: {e})", None

    if strict and not structure_has_atoms(st):
        return False, "failed (no atom coordinates found; --strict)", None

    # Atomic write: write to a temp file in output dir, then replace
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(outdir), prefix=out_path.stem + ".", suffix=".tmp", delete=False
        ) as tf:
            tmp_name = tf.name

        # gemmi writes by path
        st.write_pdb(tmp_name)

        # Basic sanity: ensure something was written
        try:
            if Path(tmp_name).stat().st_size == 0:
                os.unlink(tmp_name)
                return False, "failed (writer produced empty output)", None
        except OSError:
            pass

        os.replace(tmp_name, out_path)  # atomic on same filesystem
        return True, "ok", out_path

    except Exception as e:
        # cleanup temp file if present
        try:
            if "tmp_name" in locals() and Path(tmp_name).exists():
                os.unlink(tmp_name)
        except Exception:
            pass
        return False, f"failed (write error: {e})", None


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert mmCIF (.cif/.mmcif) to PDB securely.")
    ap.add_argument("input", type=Path, help="Input .cif/.mmcif file OR directory")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default: alongside input)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when input is a directory")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .pdb outputs")
    ap.add_argument("--strict", action="store_true", help="Fail if structure contains zero atom coordinates")
    ap.add_argument("--max-mb", type=int, default=200, help="Max input file size in MB (default: 200)")
    args = ap.parse_args()

    inp: Path = args.input
    max_bytes = int(args.max_mb) * 1024 * 1024

    if not inp.exists():
        print(f"[ERROR] Input does not exist: {inp}", file=sys.stderr)
        return 2

    # Determine output directory default
    if args.outdir is None:
        if inp.is_file():
            outdir = inp.parent
            base_input_dir = None
        else:
            outdir = inp / "pdb_out"
            base_input_dir = inp.resolve()
    else:
        outdir = args.outdir
        base_input_dir = inp.resolve() if inp.is_dir() else None

    inputs = safe_iter_inputs(inp, args.recursive)
    if not inputs:
        print("[INFO] No .cif/.mmcif files found.", file=sys.stderr)
        return 0

    ok_n = 0
    fail_n = 0
    for p in inputs:
        ok, msg, outp = convert_one(
            cif_path=p,
            outdir=outdir,
            overwrite=args.overwrite,
            strict=args.strict,
            max_bytes=max_bytes,
            base_input_dir=base_input_dir,
        )
        tag = "OK" if ok else "NO"
        if ok:
            ok_n += 1
        else:
            fail_n += 1
        if outp is not None:
            print(f"[{tag}] {p} -> {outp} : {msg}")
        else:
            print(f"[{tag}] {p} : {msg}")

    print(f"[DONE] converted={ok_n} skipped/failed={fail_n} outdir={outdir}")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
