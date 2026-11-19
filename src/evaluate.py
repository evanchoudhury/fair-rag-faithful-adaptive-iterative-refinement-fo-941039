"""Metrics, bootstrap confidence intervals, failure taxonomy and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)

from .text import exact_match
from .text import f1_score as ans_f1

FAILURES = ["correct", "answer_error", "early_stop", "filter_drop", "retrieval_miss"]
SUMMARY_KEYS = {
    "joint_evidence_recall": "joint", "paragraph_recall": "para_recall",
    "evidence_precision": "precision", "mean_passages": "n_pass",
    "mean_retrieval_calls": "calls", "mean_iterations": "iters", "answer_em": "em",
    "answer_f1": "f1", "citation_precision": "cite_ok", "abstain_rate": "abstain",
}


# ------------------------------------------------------------- per question


def question_record(qi, ev, pool_ids, ans, calls, iters, stopped_early=False):
    gold = set(qi.gold)
    evs = {int(d) for d in ev}
    hit = len(gold & evs)
    pred = ans["pred"]
    abstain = bool(ans.get("abstain", False))
    gold_ans = qi.answer or ""
    return {
        "joint": float(hit == len(gold) and len(gold) > 0),
        "para_recall": hit / max(1, len(gold)),
        "precision": hit / max(1, len(evs)),
        "n_pass": float(len(evs)),
        "calls": float(calls),
        "iters": float(iters),
        "em": 0.0 if abstain else exact_match(pred, gold_ans),
        "f1": 0.0 if abstain else ans_f1(pred, gold_ans),
        "cite_ok": float((not abstain) and ans["cite"] in gold),
        "abstain": float(abstain),
        "pred": pred, "conf": float(ans["conf"]), "cite": int(ans["cite"]),
        "in_pool": float(len(gold & {int(d) for d in pool_ids})),
        "stopped_early": float(stopped_early),
    }


def failure_type(r) -> str:
    if r["f1"] >= 0.5:
        return "correct"
    if r["joint"] == 1.0:
        return "answer_error"
    if r["stopped_early"]:
        return "early_stop"
    if r["in_pool"] >= 2:
        return "filter_drop"
    return "retrieval_miss"


def summarize(recs: pd.DataFrame) -> dict:
    out = {k: float(recs[c].mean()) for k, c in SUMMARY_KEYS.items()}
    out["n_questions"] = int(len(recs))
    return out


def bootstrap_ci(a, n=2000, seed=42):
    a = np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    m = a[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_bootstrap(a, b, n=2000, seed=42):
    """Mean difference a-b with a 95% percentile CI and one-sided p (diff <= 0)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(m, 2.5)),
            "ci_high": float(np.percentile(m, 97.5)), "p_le_zero": float((m <= 0).mean())}


def binary_metrics(y, p, thr) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else 0.0,
           "accuracy": float((pred == y).mean()) if len(y) else 0.0,
           "f1": float(f1_score(y, pred, zero_division=0)),
           "threshold": float(thr)}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["avg_precision"] = float(average_precision_score(y, p))
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out["precision"] = tp / max(1, tp + fp)
    out["recall"] = tp / max(1, tp + fn)
    return out


# ------------------------------------------------------------------- plots

C_MAIN, C_XGB, C_BASE, C_ABL = "#1f77b4", "#2ca02c", "#7f7f7f", "#ff7f0e"


def _color(name: str) -> str:
    if name.startswith("FAIR-RAG (LightGBM"):
        return C_MAIN
    if "XGBoost" in name:
        return C_XGB
    if name.startswith("Ablation"):
        return C_ABL
    return C_BASE


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_all(R: dict, plots_dir: Path) -> list[str]:
    """R is the results dictionary assembled by train.py. Returns written file names."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def done(fig, name):
        _save(fig, plots_dir / name)
        written.append(name)

    systems = R["systems"]  # name -> summary dict incl. ci
    names = list(systems.keys())

    # 1. system comparison with CIs
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(names) + 1.8))
    for ax, key, ci_key, title in [
            (axes[0], "joint_evidence_recall", "joint_ci", "Joint Evidence Recall (primary)"),
            (axes[1], "answer_f1", "f1_ci", "Answer F1")]:
        vals = [systems[n][key] for n in names]
        lo = [systems[n][key] - systems[n][ci_key][0] for n in names]
        hi = [systems[n][ci_key][1] - systems[n][key] for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, xerr=[lo, hi], color=[_color(n) for n in names], capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(names))
        ax.invert_yaxis()
        for yi, v, h in zip(y, vals, hi):
            ax.text(v + h + 0.006, yi, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, min(1.05, max(vals) * 1.25 + 0.05))
        ax.set_title(title + " - test, 95% bootstrap CI")
    done(fig, "system_comparison.png")

    # 2. cost vs quality
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, label in [(axes[0], "mean_passages", "Mean passages in final evidence"),
                           (axes[1], "mean_retrieval_calls", "Mean retrieval calls")]:
        for n in names:
            ax.scatter(systems[n][key], systems[n]["joint_evidence_recall"], s=70,
                       color=_color(n))
            ax.annotate(n, (systems[n][key], systems[n]["joint_evidence_recall"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel(label)
        ax.set_ylabel("Joint Evidence Recall")
        ax.grid(alpha=0.3)
    axes[0].set_title("Quality vs evidence budget")
    axes[1].set_title("Quality vs retrieval cost")
    done(fig, "cost_vs_quality.png")

    # 3. iteration sweep
    sw = pd.DataFrame(R["iteration_sweep"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode, ls in [("sea_stop", "-"), ("fixed", "--")]:
        d = sw[sw["mode"] == mode]
        axes[0].plot(d["max_iter"], d["joint_evidence_recall"], ls, marker="o",
                     label=f"Joint recall ({mode})", color=C_MAIN)
        axes[0].plot(d["max_iter"], d["answer_f1"], ls, marker="s", label=f"F1 ({mode})",
                     color=C_ABL)
        axes[1].plot(d["max_iter"], d["mean_retrieval_calls"], ls, marker="o",
                     label=f"calls ({mode})", color=C_MAIN)
        axes[1].plot(d["max_iter"], d["mean_passages"], ls, marker="s",
                     label=f"passages ({mode})", color=C_BASE)
    axes[0].set_title("FAIR-RAG quality vs max iterations (test)")
    axes[1].set_title("Cost vs max iterations (test)")
    for ax in axes:
        ax.set_xlabel("Max iterations")
        ax.set_xticks(sorted(sw["max_iter"].unique()))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    done(fig, "iteration_sweep.png")

    # 4. Recall@k
    rk = R["recall_at_k"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, col in [("bm25", "#d62728"), ("lsa", "#9467bd"), ("hybrid", C_MAIN)]:
        axes[0].plot(rk["k"], rk[m]["joint"], marker=".", label=m, color=col)
        axes[1].plot(rk["k"], rk[m]["paragraph"], marker=".", label=m, color=col)
    axes[0].set_title("Single-shot joint recall@k (both gold paragraphs)")
    axes[1].set_title("Single-shot paragraph recall@k")
    for ax in axes:
        ax.set_xlabel("k")
        ax.legend()
        ax.grid(alpha=0.3)
    done(fig, "recall_at_k.png")

    # 5. filter ROC / PR
    fc = R["filter_curves"]  # name -> (y, p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for n, (y, p) in fc.items():
        col = C_XGB if "XGB" in n else C_MAIN
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(y, p):.3f})", color=col)
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, label=f"{n} (AP {average_precision_score(y, p):.3f})", color=col)
    axes[0].plot([0, 1], [0, 1], ":", color="gray")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Evidence filter ROC (test pools)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Evidence filter precision-recall (test pools)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    done(fig, "filter_roc_pr.png")

    # 6. SEA calibration + confusion
    sc = R["sea_curves"]  # name -> (y, p, thr)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for n, (y, p, _) in sc.items():
        col = C_XGB if "XGB" in n else C_MAIN
        if len(np.unique(y)) == 2:
            fy, fx = calibration_curve(y, p, n_bins=10, strategy="quantile")
            axes[0].plot(fx, fy, marker="o", label=n, color=col)
    axes[0].plot([0, 1], [0, 1], ":", color="gray")
    axes[0].set_xlabel("Predicted P(sufficient)")
    axes[0].set_ylabel("Observed: both gold paragraphs in evidence")
    axes[0].set_title("SEA sufficiency calibration (test states)")
    axes[0].legend(fontsize=8)
    n0 = next(iter(sc))
    y, p, thr = sc[n0]
    cm = confusion_matrix(np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int),
                          labels=[0, 1])
    axes[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["insufficient", "sufficient"])
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["incomplete", "complete"])
    axes[1].set_xlabel("SEA decision")
    axes[1].set_ylabel("True evidence state")
    axes[1].set_title(f"{n0} confusion on test states (thr {thr:.2f})")
    done(fig, "sea_diagnostics.png")

    # 7. breakdown by question type
    bt = pd.DataFrame(R["by_type"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    types = sorted(bt["type"].unique())
    sysn = list(dict.fromkeys(bt["system"]))
    w = 0.8 / max(1, len(sysn))
    for ax, key, title in [(axes[0], "joint_evidence_recall", "Joint Evidence Recall"),
                           (axes[1], "answer_f1", "Answer F1")]:
        for i, s in enumerate(sysn):
            d = bt[bt["system"] == s].set_index("type").reindex(types)
            ax.bar(np.arange(len(types)) + i * w, d[key], w, label=s, color=_color(s)
                   if s != "Iter-Retgen loop" else "#8c564b")
        ax.set_xticks(np.arange(len(types)) + w * (len(sysn) - 1) / 2)
        ax.set_xticklabels([f"{t} (n={int(bt[bt.type == t].n.iloc[0])})" for t in types])
        ax.set_title(f"{title} by question type (test)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    done(fig, "breakdown_by_type.png")

    # 8. failure taxonomy
    ft = R["failure_taxonomy"]  # system -> {cat: frac}
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(ft) + 1.6))
    cols = {"correct": "#2ca02c", "answer_error": "#ff7f0e", "early_stop": "#9467bd",
            "filter_drop": "#8c564b", "retrieval_miss": "#d62728"}
    left = np.zeros(len(ft))
    sn = list(ft.keys())
    for cat in FAILURES:
        v = np.array([ft[s].get(cat, 0.0) for s in sn])
        ax.barh(sn, v, left=left, color=cols[cat], label=cat)
        for i, (l0, vi) in enumerate(zip(left, v)):
            if vi > 0.04:
                ax.text(l0 + vi / 2, i, f"{vi:.2f}", ha="center", va="center", fontsize=8,
                        color="white")
        left += v
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of test questions")
    ax.set_title("Outcome / failure taxonomy (correct = answer F1 >= 0.5)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    done(fig, "failure_taxonomy.png")

    # 9. feature importance
    fi = R["feature_importance"]  # component -> {feature: gain share}
    fig, axes = plt.subplots(1, len(fi), figsize=(5.2 * len(fi), 5.5))
    for ax, (comp, imp) in zip(np.atleast_1d(axes), fi.items()):
        items = sorted(imp.items(), key=lambda kv: kv[1])[-12:]
        ax.barh([k for k, _ in items], [v for _, v in items], color=C_MAIN)
        ax.set_title(f"{comp} (LightGBM gain share)")
        ax.tick_params(axis="y", labelsize=8)
    done(fig, "feature_importance.png")

    # 10. iterations used
    iu = R["iterations_used"]  # name -> list of stop iterations
    fig, ax = plt.subplots(figsize=(7, 4.2))
    mx = max(max(v) for v in iu.values())
    w = 0.8 / len(iu)
    for i, (n, v) in enumerate(iu.items()):
        cnt = np.bincount(np.asarray(v, dtype=int), minlength=mx + 1)[1:]
        ax.bar(np.arange(1, mx + 1) + i * w - 0.4 + w / 2, cnt / len(v), w, label=n,
               color=C_XGB if "XGB" in n else C_MAIN)
    ax.set_xlabel("Iterations used before SEA declared evidence sufficient (cap = max)")
    ax.set_ylabel("Fraction of test questions")
    ax.set_xticks(np.arange(1, mx + 1))
    ax.legend(fontsize=8)
    ax.set_title("Adaptive stopping: iterations used per question")
    done(fig, "iterations_used.png")

    # 11. dataset overview
    ds = R["dataset"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    sp = pd.DataFrame(ds["split_type_counts"]).T  # split -> type counts
    sp.plot(kind="bar", stacked=True, ax=axes[0], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("Questions per split and type")
    axes[0].set_ylabel("Questions")
    axes[0].tick_params(axis="x", rotation=0)
    axes[1].hist(ds["paragraph_lengths"], bins=60, color=C_BASE)
    axes[1].set_title(f"Corpus paragraph length (content tokens), n={ds['n_paragraphs']}")
    axes[1].set_xlabel("Tokens")
    axes[1].set_yscale("log")
    done(fig, "dataset_overview.png")
    return written
