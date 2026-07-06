"""Data acquisition & preprocessing for the peptide detectability classifier.

Pipeline
--------
download GPMDB-derived DeepMSPeptide peptide tables  ->  clean
->  deduplicate (dropping conflicting labels)         ->  stratified 70/15/15 split
->  write data/processed/{train,val,test}.csv + split_stats.json (with a data hash)

Data source
-----------
DeepMSPeptide (Serrano, Guruceaga & Segura, *Bioinformatics* 2020),
https://github.com/vsegurar/DeepMSPeptide — peptides derived from the GPMDB
database. In the source tables the ``Class`` column encodes GPMDB observation
frequency, which we map to a binary detectability label::

    Class == "MObs"  (multiply observed / proteotypic)  ->  1  (detectable)
    Class == "LObs"  (lowly observed)                   ->  0  (not detectable)

We build *our own* stratified 70/15/15 split (not the repository's train/test
boundary), so any performance measured here is NOT directly comparable to the
paper's reported numbers.

Usage
-----
    python -m src.data_prep              # download real data into data/raw/, then process
    python -m src.data_prep --synthetic  # skip the network; use a synthetic placeholder

The real files download into ``data/raw/`` (git-ignored). The training set ships
as a ``.rar``; if no RAR extractor is available the pipeline still runs on the
plain-text test table alone (it already contains both classes). If the network
is blocked entirely, the script prints a warning and falls back to a small,
clearly-labelled SYNTHETIC dataset so the rest of the codebase can run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import config

# --------------------------------------------------------------------------- #
# Source definition
# --------------------------------------------------------------------------- #
_RAW_BASE = "https://raw.githubusercontent.com/vsegurar/DeepMSPeptide/master/Datasets"

# (filename in data/raw/, download URL, kind)
SOURCE_FILES: list[tuple[str, str, str]] = [
    ("GPMDB_test_peptides.txt", f"{_RAW_BASE}/GPMDB_test_peptides.txt", "text"),
    ("GPMDB_training_peptides.rar", f"{_RAW_BASE}/GPMDB_training_peptides.rar", "rar"),
]

AA_SET = set(config.AMINO_ACIDS)

# Textual class label -> binary detectability label. Case-insensitive.
LABEL_MAP: dict[str, int] = {
    "mobs": 1, "lobs": 0,                # DeepMSPeptide GPMDB tables
    "1": 1, "0": 0,                       # already-binary tables
    "detectable": 1, "nondetectable": 0, "non-detectable": 0, "not-detectable": 0,
    "proteotypic": 1, "nonproteotypic": 0, "non-proteotypic": 0,
    "yes": 1, "no": 0, "true": 1, "false": 0,
}

_SEQ_COL_NAMES = ["peptide_seq", "peptide", "sequence", "seq"]
_LABEL_COL_NAMES = ["class", "detectability", "label", "proteotypic"]


# --------------------------------------------------------------------------- #
# Download + extraction
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> bool:
    """Stream ``url`` to ``dest``. Returns True on success (or if already cached)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cache] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "peptide-detectability/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            with open(dest, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"  {dest.name}",
                disable=total == 0,
            ) as bar:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    bar.update(len(chunk))
        print(f"  [ok]    downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  [warn]  could not download {dest.name}: {exc}")
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        return False


def _extract_rar(rar_path: Path, out_dir: Path) -> None:
    """Best-effort extraction of a ``.rar`` into ``out_dir`` (side-effect only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) Python 'rarfile' (needs a system unrar/bsdtar/unar backend).
    try:
        import rarfile  # type: ignore

        with rarfile.RarFile(str(rar_path)) as rf:
            rf.extractall(path=str(out_dir))
        print(f"  [ok]    extracted {rar_path.name} via python-rarfile")
        return
    except Exception:
        pass
    # 2) Shell out to a known archiver, if one is installed.
    attempts = [
        ("unrar", ["unrar", "x", "-o+", "-y", str(rar_path), f"{out_dir}/"]),
        ("unar", ["unar", "-force-overwrite", "-o", str(out_dir), str(rar_path)]),
        ("bsdtar", ["bsdtar", "-x", "-f", str(rar_path), "-C", str(out_dir)]),
        ("7z", ["7z", "x", f"-o{out_dir}", "-y", str(rar_path)]),
    ]
    for tool, cmd in attempts:
        if shutil.which(tool):
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  [ok]    extracted {rar_path.name} via {tool}")
                return
            except Exception:
                continue
    print(
        f"  [warn]  could not extract {rar_path.name} (no RAR extractor found). "
        f"Install unrar / unar / libarchive-tools (bsdtar) / p7zip, or unpack it "
        f"manually into {out_dir}, then rerun. Proceeding without it."
    )


# --------------------------------------------------------------------------- #
# Parsing labeled tables
# --------------------------------------------------------------------------- #
def _pick(cols: list[str], names: list[str], default: str) -> str:
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n in lower:
            return lower[n]
    return default


def _labeled_candidates(raw_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.txt", "*.tsv", "*.csv"):
        paths.extend(raw_dir.rglob(pattern))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _read_labeled_table(path: Path) -> "pd.DataFrame | None":
    """Read a peptide table; return a DataFrame with columns [sequence, label]."""
    sep = "\t"
    try:
        header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig")
    except Exception:
        return None
    cols = list(header.columns)
    if len(cols) < 2:  # not tab-separated; try comma
        sep = ","
        try:
            header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig")
        except Exception:
            return None
        cols = list(header.columns)
        if len(cols) < 2:
            return None
    seq_col = _pick(cols, _SEQ_COL_NAMES, default=cols[0])
    lab_col = _pick(cols, _LABEL_COL_NAMES, default=cols[-1])
    if seq_col == lab_col:
        return None
    try:
        df = pd.read_csv(path, sep=sep, usecols=[seq_col, lab_col], dtype=str, encoding="utf-8-sig")
    except Exception as exc:
        print(f"  [skip]  {path.name}: read error ({exc})")
        return None
    return df.rename(columns={seq_col: "sequence", lab_col: "label"})[["sequence", "label"]]


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    return df


# --------------------------------------------------------------------------- #
# Cleaning / dedup / split
# --------------------------------------------------------------------------- #
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    # keep only sequences written over the 20 standard amino acids
    is_valid_aa = df["sequence"].map(lambda s: len(s) > 0 and set(s) <= AA_SET)
    df = df[is_valid_aa]
    # length bounds (inclusive)
    lengths = df["sequence"].str.len()
    df = df[(lengths >= config.MIN_LEN) & (lengths <= config.MAX_LEN)]
    # drop rows whose label could not be mapped
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def _dedup(df: pd.DataFrame) -> "tuple[pd.DataFrame, int]":
    """Collapse duplicate sequences; drop sequences with conflicting labels."""
    grp = df.groupby("sequence")["label"].agg(["nunique", "first"])
    n_conflicting = int((grp["nunique"] > 1).sum())
    kept = grp[grp["nunique"] == 1]
    out = pd.DataFrame({"sequence": kept.index, "label": kept["first"].astype(int).values})
    return out.reset_index(drop=True), n_conflicting


def _split(df: pd.DataFrame, seed: int) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    """Stratified 70/15/15. Unique sequences guarantee no cross-split leakage."""
    train, temp = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=temp["label"])
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
def _make_synthetic(n: int, seed: int) -> pd.DataFrame:
    """A small, clearly-labelled placeholder dataset (NOT real data)."""
    rng = random.Random(seed)
    aas = config.AMINO_ACIDS
    hydrophobic = set("AILMFWVCY")
    seqs: list[str] = []
    scores: list[float] = []
    seen: set[str] = set()
    while len(seqs) < n:
        length = rng.randint(config.MIN_LEN, config.MAX_LEN)
        seq = "".join(rng.choice(aas) for _ in range(length))
        if seq in seen:
            continue
        seen.add(seq)
        seqs.append(seq)
        hydro_frac = sum(c in hydrophobic for c in seq) / length
        scores.append(hydro_frac - 0.004 * length + rng.uniform(-0.1, 0.1))
    threshold = float(np.median(scores))  # split near the median -> roughly balanced
    labels = [1 if s > threshold else 0 for s in scores]
    return pd.DataFrame({"sequence": seqs, "label": labels})


# --------------------------------------------------------------------------- #
# Saving + reporting
# --------------------------------------------------------------------------- #
def _sha256_12(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]


def _save(train, val, test, out_dir: Path, source: str, synthetic: bool) -> "tuple[dict, str]":
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: dict[str, Path] = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        p = out_dir / f"{name}.csv"
        frame[["sequence", "label"]].to_csv(p, index=False)
        csv_paths[name] = p
    data_hash = _sha256_12([csv_paths["train"], csv_paths["val"], csv_paths["test"]])

    stats: dict = {
        "source": source,
        "synthetic": synthetic,
        "random_seed": config.RANDOM_SEED,
        "min_len": config.MIN_LEN,
        "max_len": config.MAX_LEN,
        "data_hash": data_hash,
        "total": 0,
        "splits": {},
    }
    total = 0
    for name, frame in (("train", train), ("val", val), ("test", test)):
        n = len(frame)
        pos = int(frame["label"].sum())
        stats["splits"][name] = {
            "n": n,
            "positives": pos,
            "negatives": n - pos,
            "positive_rate": round(pos / n, 4) if n else 0.0,
        }
        total += n
    stats["total"] = total
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    return stats, data_hash


def _print_summary(stats: dict, source: str, data_hash: str) -> None:
    line = "-" * 66
    print("\n" + line)
    print("DATA PREP SUMMARY")
    print(line)
    print(f"source    : {source}")
    print(f"synthetic : {stats['synthetic']}")
    print(f"data_hash : {data_hash}")
    print(f"seed      : {stats['random_seed']}    length range: {stats['min_len']}-{stats['max_len']}")
    print(f"{'split':<8}{'n':>9}{'positives':>12}{'negatives':>12}{'pos_rate':>11}")
    for name in ("train", "val", "test"):
        s = stats["splits"][name]
        print(f"{name:<8}{s['n']:>9}{s['positives']:>12}{s['negatives']:>12}{s['positive_rate']:>11.4f}")
    print(f"{'TOTAL':<8}{stats['total']:>9}")
    print(line)
    print(f"wrote: {config.TRAIN_CSV}")
    print(f"       {config.VAL_CSV}")
    print(f"       {config.TEST_CSV}")
    print(f"       {config.SPLIT_STATS_JSON}")
    if stats["synthetic"]:
        print("\n[!] Reminder: this is SYNTHETIC placeholder data — replace with real data "
              "before reporting any results.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the peptide detectability dataset.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip download and generate a synthetic placeholder dataset.")
    parser.add_argument("--n-synthetic", type=int, default=3000,
                        help="Number of synthetic peptides to generate (fallback only).")
    args = parser.parse_args()

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    raw_dir = config.RAW_DATA_DIR
    out_dir = config.PROCESSED_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    synthetic = args.synthetic
    pooled: "pd.DataFrame | None" = None

    if not synthetic:
        print("[1/5] Downloading DeepMSPeptide GPMDB data into data/raw/ ...")
        for fname, url, kind in SOURCE_FILES:
            ok = _download(url, raw_dir / fname)
            if ok and kind == "rar":
                _extract_rar(raw_dir / fname, raw_dir)

        print("[2/5] Parsing labeled peptide tables ...")
        frames: list[pd.DataFrame] = []
        for path in _labeled_candidates(raw_dir):
            df = _read_labeled_table(path)
            if df is None:
                print(f"  [skip]  {path.name}: no usable sequence+label columns")
                continue
            df = _normalize_labels(df)
            known = int(df["label"].notna().sum())
            print(f"  [use]   {path.name}: {len(df)} rows ({known} with a known label)")
            frames.append(df)
        if frames:
            pooled = pd.concat(frames, ignore_index=True)

        if pooled is None or len(pooled) == 0:
            print("\n[!] No real data could be obtained (network blocked or files unreadable).")
            synthetic = True

    if synthetic:
        banner = "=" * 70
        print("\n" + banner)
        print("!! FALLBACK: generating a SYNTHETIC placeholder dataset (NOT real data).")
        print("!! Fetch the real DeepMSPeptide data locally, then rerun WITHOUT --synthetic.")
        print("!! (The training set is a .rar — install unrar/unar/libarchive-tools/p7zip")
        print("!!  to include it; the plain-text test table works on its own.)")
        print(banner)
        pooled = _make_synthetic(args.n_synthetic, config.RANDOM_SEED)
        source = "SYNTHETIC_PLACEHOLDER (not real data)"
    else:
        source = ("DeepMSPeptide GPMDB peptides (Serrano et al., Bioinformatics 2020); "
                  "Class: MObs->1 (detectable/proteotypic), LObs->0 (not detectable)")

    print(f"[3/5] Cleaning (uppercase; 20 standard AAs only; length "
          f"{config.MIN_LEN}-{config.MAX_LEN}) ...")
    n_before = len(pooled)
    pooled = _clean(pooled)
    print(f"  kept {len(pooled)} / {n_before} rows after cleaning")

    print("[4/5] Deduplicating before split (dropping conflicting-label sequences) ...")
    deduped, n_conflicting = _dedup(pooled)
    print(f"  {len(deduped)} unique sequences; dropped {n_conflicting} with conflicting labels")

    # deterministic ordering, then stratified split
    deduped = deduped.sort_values("sequence", kind="mergesort").reset_index(drop=True)
    train, val, test = _split(deduped, config.RANDOM_SEED)

    print("[5/5] Writing processed splits + split_stats.json ...")
    stats, data_hash = _save(train, val, test, out_dir, source, synthetic)
    _print_summary(stats, source, data_hash)


if __name__ == "__main__":
    main()
