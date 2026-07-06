"""Benchmark all saved model versions on the SAME held-out test split.

Produces:
  1. reports/benchmark.md              — table sorted by PR-AUC, champion marked
  2. reports/figures/roc.png           — overlaid ROC curves
  3. reports/figures/pr.png            — overlaid precision-recall curves
  4. reports/figures/model_comparison.png — grouped PR-AUC / F1 bar chart

Everything is recomputed from the committed models + regenerated test split, so
the report is fully reproducible.

CLI:
    python -m src.benchmark                       # all versions in the registry
    python -m src.benchmark --versions v001_... v003_...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src import config, evaluate, versioning  # noqa: E402


def _short_label(version: str, model_type: str) -> str:
    return f"{version.split('_')[0]} {model_type}"  # e.g. "v001 baseline"


def _read_champion() -> "str | None":
    try:
        return json.loads(versioning.LATEST.read_text())["champion"]
    except Exception:
        return None


def _test_context() -> dict:
    try:
        stats = json.loads(config.SPLIT_STATS_JSON.read_text())
        s = stats["splits"]["test"]
        return {"n_test": int(s["n"]), "pos_rate": float(s["positive_rate"]),
                "synthetic": bool(stats.get("synthetic", False))}
    except Exception:
        df = pd.read_csv(config.TEST_CSV)
        return {"n_test": int(len(df)), "pos_rate": float(df["label"].mean()),
                "synthetic": None}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plot_roc(results, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        plt.plot(fpr, tpr, lw=1.8,
                 label=f"{r['short']} (AUC={r['metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pr(results, out_path: Path, prevalence: float) -> None:
    plt.figure(figsize=(6, 5))
    for r in results:
        precision, recall, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        plt.plot(recall, precision, lw=1.8,
                 label=f"{r['short']} (AP={r['metrics']['pr_auc']:.3f})")
    plt.hlines(prevalence, 0, 1, colors="k", linestyles="--", lw=1, alpha=0.6,
               label=f"prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curves (test set)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_bars(results, out_path: Path) -> None:
    labels = [r["short"] for r in results]
    pr_auc = [r["metrics"]["pr_auc"] for r in results]
    f1 = [r["metrics"]["f1"] for r in results]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(6.0, 1.7 * len(labels)), 5))
    plt.bar(x - width / 2, pr_auc, width, label="PR-AUC")
    plt.bar(x + width / 2, f1, width, label="F1")
    for xi, (p, f) in enumerate(zip(pr_auc, f1)):
        plt.text(xi - width / 2, p + 0.01, f"{p:.3f}", ha="center", va="bottom", fontsize=7)
        plt.text(xi + width / 2, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=7)
    plt.xticks(x, labels, rotation=20, ha="right", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Model comparison (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _write_markdown(results, champion, out_md: Path, ctx: dict) -> None:
    ranked = sorted(results, key=lambda r: r["metrics"]["pr_auc"], reverse=True)
    lines: list[str] = ["# Benchmark", ""]

    if ctx.get("synthetic"):
        lines += ["> ⚠️ **These results are on SYNTHETIC placeholder data** "
                  "(`split_stats.json` has `synthetic: true`). Re-run "
                  "`python -m src.data_prep` with the real dataset, retrain, and "
                  "re-benchmark before reporting anything.", ""]

    lines += [
        f"Evaluated on the same held-out test split (`data/processed/test.csv`, "
        f"n={ctx['n_test']}, positive rate={ctx['pos_rate']:.1%}). Ranked by "
        f"PR-AUC (descending). Primary metric: **{config.PRIMARY_METRIC}**.",
        "",
        "| version | model_type | param_count | accuracy | precision | recall | f1 | roc_auc | pr_auc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked:
        m = r["metrics"]
        star = " ★" if r["version"] == champion else ""
        lines.append(
            f"| {r['version']}{star} | {r['model_type']} | {r['param_count']:,} | "
            f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |"
        )
    lines += [
        "",
        "★ = current champion (highest PR-AUC, per `models/LATEST.json`).",
        "",
        "## Figures",
        "",
        "![ROC curves](figures/roc.png)",
        "",
        "![Precision–Recall curves](figures/pr.png)",
        "",
        "![Model comparison](figures/model_comparison.png)",
        "",
        "> **Note.** `param_count` is the number of trainable parameters for the "
        "neural models (cnn, esm2) and the total number of leaves across all trees "
        "for the LightGBM baseline.",
        "",
        "> **Footnote — context, not a head-to-head comparison.** The DeepMSPeptide "
        "paper (Serrano et al., *Bioinformatics* 2020) reported roughly **93% "
        "sensitivity / 99% specificity** on **their own** test set. This benchmark "
        "uses a *different* train/val/test split of the GPMDB-derived data, so those "
        "figures are **context only** and are **not** directly comparable to the "
        "numbers above.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all model versions on the test split.")
    parser.add_argument("--versions", nargs="*", default=None,
                        help="Subset of version names to compare (default: all in the registry).")
    args = parser.parse_args()

    versions = args.versions or [e["version"] for e in versioning.list_versions()]
    if not versions:
        raise SystemExit(
            "No model versions found. Train at least one model first, e.g. "
            "`python -m src.train --model baseline`."
        )

    print(f"[benchmark] evaluating {len(versions)} version(s) on the test split ...")
    results = []
    for version in versions:
        try:
            res = evaluate.evaluate_on_test(version)
        except SystemExit as exc:
            print(f"  [skip] {version}: {exc}")
            continue
        except Exception as exc:  # keep going if one model can't be scored
            print(f"  [skip] {version}: {type(exc).__name__}: {exc}")
            continue
        res["short"] = _short_label(res["version"], res["model_type"])
        m = res["metrics"]
        print(f"  [ok]   {res['short']:<18} pr_auc={m['pr_auc']:.4f}  "
              f"roc_auc={m['roc_auc']:.4f}  f1={m['f1']:.4f}")
        results.append(res)

    if not results:
        raise SystemExit("No versions could be evaluated (see skip messages above).")

    ctx = _test_context()
    prevalence = float(np.mean(results[0]["y_true"]))
    champion = _read_champion()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_roc(results, config.FIGURES_DIR / "roc.png")
    _plot_pr(results, config.FIGURES_DIR / "pr.png", prevalence)
    _plot_bars(results, config.FIGURES_DIR / "model_comparison.png")

    out_md = config.REPORTS_DIR / "benchmark.md"
    _write_markdown(results, champion, out_md, ctx)

    best = max(results, key=lambda r: r["metrics"]["pr_auc"])
    print(f"\n[benchmark] wrote {out_md.relative_to(config.PROJECT_ROOT)} and 3 figures "
          f"in {config.FIGURES_DIR.relative_to(config.PROJECT_ROOT)}/")
    print(f"[benchmark] best PR-AUC: {best['short']} = {best['metrics']['pr_auc']:.4f}"
          + (f"  (champion: {champion})" if champion else ""))


if __name__ == "__main__":
    main()
