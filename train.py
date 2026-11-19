"""FAIR-RAG on a laptop - end-to-end training and evaluation.

    python train.py            full run (HotpotQA distractor validation, ~7.4k questions)
    python train.py --quick    small subsample for a fast end-to-end check

Writes artifacts/metrics.json, artifacts/model.joblib, artifacts/plots/*.png,
artifacts/sample_predictions.csv, artifacts/test_traces.jsonl and more.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src.model import (ANSWER_FEATURES, FILTER_FEATURES, SEA_FEATURES, Config,  # noqa: E402
                       FairRAG, KB, QBank, QInfo, answer_candidates, check_findings,
                       make_xgb, predict_proba, sea_features, select_evidence,
                       trace_dict)
from src.retriever import HybridRetriever  # noqa: E402
from src.text import f1_score as ans_f1  # noqa: E402
from src.utils import ART_DIR, DATA_DIR, SEED, Timer, log, set_seed  # noqa: E402


def make_lgbm(**kw):
    """LightGBM is a training-only dependency: it lives here, never in the dashboard's
    import path. The dashboard runs the exported XGBoost booster (artifacts/demo/)."""
    from lightgbm import LGBMClassifier
    p = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42,
             n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
    p.update(kw)
    return LGBMClassifier(**p)


DEFAULT_OPTS = {
    "lsa_dim": 256, "folds": 4, "filter_grid": [{"num_leaves": 31}, {"num_leaves": 63}],
    "n_boot": 2000, "retr_cache": DATA_DIR / "retriever.joblib", "tune_sample": 1500,
    "lsa_weights": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0], "tau_grid":
        [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5], "passage_cost": 0.01,
    "sea_grid": [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)] + [0.97, 0.99],
    "call_cost": 0.01,
    "min_child_samples": 20,
}

MAIN = "FAIR-RAG (LightGBM, SEA stop)"
XGBN = "FAIR-RAG (XGBoost components)"


# ------------------------------------------------------------------ helpers


def retrieval_texts(corpus):
    """Title is repeated once (simple title-field boost; chosen on train questions)."""
    return [t + ". " + x for t, x in zip(corpus["titles"], corpus["texts"])]


def get_retriever(texts, opts):
    key = {"n": len(texts), "lsa_dim": opts["lsa_dim"], "v": 4,
           "crc": zlib.crc32("\n".join(texts[:200] + texts[-200:]).encode("utf-8"))}
    path = opts.get("retr_cache")
    if path is not None and Path(path).exists():
        try:
            blob = joblib.load(path)
            if blob.get("key") == key:
                log("Loaded cached retriever index (data/retriever.joblib)")
                return blob["retr"]
        except Exception as exc:  # corrupted cache -> rebuild
            log(f"Retriever cache unreadable ({type(exc).__name__}); rebuilding")
    log(f"Fitting hybrid retriever on {len(texts)} paragraphs (BM25 + LSA-{opts['lsa_dim']})")
    retr = HybridRetriever(lsa_dim=opts["lsa_dim"]).fit(texts)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"key": key, "retr": retr}, path)
    return retr


def gold_mask(qi, pids):
    g = set(qi.gold)
    return np.array([int(d) in g for d in pids], dtype=np.int32)


def is_joint(qi, ev):
    return float(set(qi.gold) <= {int(d) for d in ev}) if qi.gold else 0.0


def snap_assess(qi, snap, probs, kb, cfg):
    pids = snap["pool_ids"]
    idx = select_evidence(pids, probs, np.zeros(len(pids), np.float32), cfg, True,
                          cfg.max_evidence)
    ev = pids[idx] if len(idx) else np.zeros(0, dtype=np.int64)
    evp = probs[idx] if len(idx) else np.zeros(0, dtype=np.float32)
    check = check_findings(qi, ev, kb)
    x = sea_features(qi, ev, evp, pids, probs, idx, check, kb, snap["t"], snap["n_new"],
                     len(pids), snap["calls"])
    return ev, evp, x


def stop_iter(s, thr, max_iter):
    last = min(max_iter, len(s.snaps))
    for t in range(1, last + 1):
        p = s.snaps[t - 1]["sea_p"]
        if not np.isnan(p) and p >= thr:
            return t
    return last


def gain_importance(model, names):
    if hasattr(model, "booster_"):
        g = model.booster_.feature_importance(importance_type="gain").astype(float)
    else:
        g = np.asarray(model.feature_importances_, dtype=float)
    g = g / max(g.sum(), 1e-12)
    return {n: round(float(v), 5) for n, v in zip(names, g)}


def fit_oof(X, y, groups, make_fn, grid, folds):
    """Grouped K-fold: choose params by out-of-fold ROC-AUC; return fold models + OOF."""
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=max(2, min(folds, n_groups)))
    splits = list(gkf.split(X, y, groups))
    best = None
    for params in grid:
        oof = np.zeros(len(y), dtype=np.float32)
        models = []
        for tri, tei in splits:
            m = make_fn(**params).fit(X[tri], y[tri])
            oof[tei] = predict_proba(m, X[tei])
            models.append(m)
        auc = roc_auc_score(y, oof) if len(np.unique(y)) == 2 else 0.5
        log(f"    params {params}: OOF ROC-AUC {auc:.4f}")
        if best is None or auc > best["auc"] + 1e-9:
            best = {"auc": float(auc), "params": params, "models": models, "oof": oof}
    fold_of_group = {}
    for k, (_, tei) in enumerate(splits):
        for g in np.unique(groups[tei]):
            fold_of_group[int(g)] = k
    best["fold_of_group"] = fold_of_group
    best["final"] = make_fn(**best["params"]).fit(X, y)
    return best


def oof_snapshot_probs(states, fold_models, fold_of_group):
    """Out-of-fold filter probabilities for every recorded snapshot of every question."""
    out = [[None] * len(s.snaps) for s in states]
    by_fold: dict[int, list] = {}
    for i, s in enumerate(states):
        by_fold.setdefault(fold_of_group[i], []).extend((i, t) for t in range(len(s.snaps)))
    for k, keys in by_fold.items():
        blocks = [states[i].snaps[t]["X"] for i, t in keys]
        p = predict_proba(fold_models[k], np.vstack(blocks)) if blocks else []
        pos = 0
        for (i, t), b in zip(keys, blocks):
            out[i][t] = p[pos:pos + len(b)]
            pos += len(b)
    return out


def tune_filter_threshold(states, oof, kb, cfg, opts, t_index):
    """Pick tau maximising joint recall - cost * passages on train OOF predictions."""
    rows = []
    for tau in opts["tau_grid"]:
        c = dataclasses.replace(cfg, filter_threshold=tau)
        j, n = [], []
        for s, pr in zip(states, oof):
            t = min(t_index, len(s.snaps)) - 1
            pids = s.snaps[t]["pool_ids"]
            idx = select_evidence(pids, pr[t], np.zeros(len(pids)), c, True, c.max_evidence)
            j.append(is_joint(s.qi, pids[idx]))
            n.append(len(idx))
        jr, mn = float(np.mean(j)), float(np.mean(n))
        rows.append({"tau": tau, "joint_recall": jr, "mean_passages": mn,
                     "score": jr - opts["passage_cost"] * mn})
    best = max(rows, key=lambda r: (r["score"], -r["tau"]))
    return best["tau"], rows


def build_sea_rows(states, oof, kb, cfg):
    X, y, g = [], [], []
    for i, (s, pr) in enumerate(zip(states, oof)):
        for t, sn in enumerate(s.snaps):
            ev, _, x = snap_assess(s.qi, sn, pr[t], kb, cfg)
            X.append(x)
            y.append(is_joint(s.qi, ev))
            g.append(i)
    return np.vstack(X), np.asarray(y, dtype=np.int32), np.asarray(g)


def build_answer_rows(states, oof, kb, cfg, t_index):
    X, y, g = [], [], []
    for i, (s, pr) in enumerate(zip(states, oof)):
        t = min(t_index, len(s.snaps)) - 1
        ev, evp, _ = snap_assess(s.qi, s.snaps[t], pr[t], kb, cfg)
        texts, _, Xa = answer_candidates(s.qi, ev, evp, kb)
        if not texts:
            continue
        X.append(Xa)
        y += [int(ans_f1(tx, s.qi.answer or "") >= 0.5) for tx in texts]
        g += [i] * len(texts)
    return np.vstack(X), np.asarray(y, dtype=np.int32), np.asarray(g)


def tune_sea_threshold(states, max_iter, opts):
    """Threshold maximising val joint recall - call_cost * mean retrieval calls
    (an extra retrieval call must buy at least `call_cost` joint recall)."""
    ref = float(np.mean([is_joint(s.qi, s.snaps[min(max_iter, len(s.snaps)) - 1]["ev"])
                         for s in states]))
    rows = []
    for thr in opts["sea_grid"]:
        ts = [stop_iter(s, thr, max_iter) for s in states]
        jr = float(np.mean([is_joint(s.qi, s.snaps[t - 1]["ev"]) for s, t in zip(states, ts)]))
        calls = float(np.mean([s.snaps[t - 1]["calls"] for s, t in zip(states, ts)]))
        rows.append({"threshold": thr, "joint_recall": jr, "mean_calls": calls,
                     "mean_iterations": float(np.mean(ts)),
                     "score": jr - opts["call_cost"] * calls})
    best = max(rows, key=lambda r: (r["score"], -r["threshold"]))
    return best["threshold"], ref, rows


class Answers:
    """Caches answers per (question index, iteration) for one run."""

    def __init__(self, system, states, models):
        self.system, self.states, self.models, self.c = system, states, models, {}

    def get(self, ts):
        need = [(i, t) for i, t in enumerate(ts) if (i, t) not in self.c]
        if need:
            items = [(self.states[i].qi, self.states[i].snaps[t - 1]["ev"],
                      self.states[i].snaps[t - 1]["ev_probs"]) for i, t in need]
            for key, a in zip(need, self.system.answer(items, models=self.models)):
                self.c[key] = a
        return [self.c[(i, t)] for i, t in enumerate(ts)]


def records(states, ts, answers, early=None):
    rows = []
    for i, (s, t, a) in enumerate(zip(states, ts, answers)):
        sn = s.snaps[t - 1]
        r = E.question_record(s.qi, sn["ev"], sn["pool_ids"], a, sn["calls"], t,
                              bool(early[i]) if early is not None else False)
        r["qid"], r["type"] = s.qi.qid, s.qi.qtype
        rows.append(r)
    return pd.DataFrame(rows)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return o.name
    return o


# --------------------------------------------------------------- experiment


def run_experiment(df, corpus, art_dir: Path, opts: dict | None = None, quick=False,
                   n_dropped=0) -> dict:
    opts = {**DEFAULT_OPTS, **(opts or {})}
    set_seed(SEED)
    clock = Timer()
    art_dir = Path(art_dir)
    plots_dir = art_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te = D.split(df)
    if quick:
        tr, va, te = tr[:1200], va[:400], te[:400]
    log(f"Split: train {len(tr)} / val {len(va)} / test {len(te)} questions; "
        f"corpus {len(corpus['titles'])} paragraphs")

    retr = get_retriever(retrieval_texts(corpus), opts)
    kb = KB(corpus["titles"], corpus["sents"])
    qinfos = [QInfo(r.question, r.gold_ids, r.answer, r.type, r.level, r.id)
              for r in df.itertuples()]

    # ---- 1. fusion weight of the dense ranking, chosen on train questions only
    tq = [qinfos[i] for i in tr[: opts["tune_sample"]]]
    fusion_rows = []
    for w in opts["lsa_weights"]:
        retr.set_lsa_weight(w)
        r = retr.rank([q.text for q in tq], "hybrid", 10)
        pr = float(np.mean([len(set(q.gold) & set(x.tolist())) / 2 for q, x in zip(tq, r)]))
        fusion_rows.append({"lsa_weight": w, "train_paragraph_recall_at_10": pr})
    best_w = max(fusion_rows, key=lambda r: (r["train_paragraph_recall_at_10"],
                                             -r["lsa_weight"]))["lsa_weight"]
    retr.set_lsa_weight(best_w)
    log("RRF dense weight chosen on train: " + str(best_w) + "  ("
        + ", ".join(f"{r['lsa_weight']}:{r['train_paragraph_recall_at_10']:.3f}"
                    for r in fusion_rows) + ")")

    cfg = Config()
    system = FairRAG(retr, kb, cfg)
    bank_tr = QBank([qinfos[i] for i in tr], retr)
    bank_va = QBank([qinfos[i] for i in va], retr)
    bank_te = QBank([qinfos[i] for i in te], retr)
    mcs = opts["min_child_samples"]
    mk_l = lambda **kw: make_lgbm(min_child_samples=mcs, **kw)  # noqa: E731
    mk_x = lambda **kw: make_xgb(**kw)  # noqa: E731

    # ---- 2. evidence filter v1 on iteration-1 pools (train)
    log("Stage A: iteration-1 retrieval on train -> evidence filter v1")
    stA = system.run(bank_tr, max_iter=1, record_X=True, models={})
    XA = np.vstack([s.snaps[0]["X"] for s in stA])
    yA = np.concatenate([gold_mask(s.qi, s.snaps[0]["pool_ids"]) for s in stA])
    filter1 = mk_l().fit(XA, yA)

    # ---- 3. fixed 4-iteration loop on train, driven by filter v1
    log("Stage B: fixed 4-iteration FAIR-RAG loop on train (records every state)")
    stB = system.run(bank_tr, max_iter=4, record_X=True, models={"filter": filter1})
    XF = np.vstack([s.snaps[-1]["X"] for s in stB])
    yF = np.concatenate([gold_mask(s.qi, s.snaps[-1]["pool_ids"]) for s in stB])
    gF = np.concatenate([[i] * len(s.snaps[-1]["pool_ids"]) for i, s in enumerate(stB)])
    log(f"  filter training rows: {len(yF)} (positives {yF.mean():.3f})")

    comps = {}
    tuning = {"rrf_lsa_weight": fusion_rows, "rrf_lsa_weight_chosen": best_w}
    for tag, mk, grid in [("lgbm", mk_l, opts["filter_grid"]), ("xgb", mk_x, [{}])]:
        log(f"Training {tag.upper()} components")
        log("  evidence filter (grouped CV over questions)")
        fo = fit_oof(XF, yF, gF, mk, grid, opts["folds"])
        oof = oof_snapshot_probs(stB, fo["models"], fo["fold_of_group"])
        tau, tau_rows = tune_filter_threshold(stB, oof, kb, cfg, opts, cfg.max_iter)
        c = dataclasses.replace(cfg, filter_threshold=tau)
        log(f"  filter threshold (train OOF) = {tau}")
        log("  SEA sufficiency model")
        Xs, ys, gs = build_sea_rows(stB, oof, kb, c)
        so = fit_oof(Xs, ys, gs, mk, [{}], opts["folds"])
        log("  faithful answer ranker")
        Xa, ya, ga = build_answer_rows(stB, oof, kb, c, cfg.max_iter)
        ao = fit_oof(Xa, ya, ga, mk, [{}], opts["folds"])
        comps[tag] = {"models": {"filter": fo["final"], "sea": so["final"], "answer": ao["final"]},
                      "cfg": c}
        tuning[tag] = {
            "filter_params": fo["params"], "filter_oof_auc": fo["auc"],
            "filter_threshold_grid": tau_rows, "filter_threshold": tau,
            "sea_oof_auc": so["auc"], "sea_rows": int(len(ys)),
            "sea_positive_rate": float(ys.mean()), "answer_oof_auc": ao["auc"],
            "answer_rows": int(len(ya)), "answer_positive_rate": float(ya.mean()),
            "filter_rows": int(len(yF)),
        }
    del stA, stB, XA, XF

    # ---- 4. SEA threshold on validation
    systems_ = {}
    for tag in ("lgbm", "xgb"):
        c = comps[tag]["cfg"]
        sysm = FairRAG(retr, kb, c, comps[tag]["models"])
        log(f"Validation run ({tag.upper()}): fixed 4 iterations, tune SEA threshold")
        stV = sysm.run(bank_va, max_iter=4)
        thr, ref, rows = tune_sea_threshold(stV, c.max_iter, opts)
        comps[tag]["cfg"] = dataclasses.replace(c, sea_threshold=thr)
        sysm.cfg = comps[tag]["cfg"]
        systems_[tag] = sysm
        tuning[tag].update({"sea_threshold": thr, "sea_threshold_grid": rows,
                            "val_fixed_loop_joint_recall": ref})
        log(f"  SEA threshold = {thr} (val fixed-loop joint recall {ref:.4f})")
    del stV

    # ---- 5. test runs
    cfgL = comps["lgbm"]["cfg"]
    mainsys, xgbsys = systems_["lgbm"], systems_["xgb"]
    log("Test: FAIR-RAG (LightGBM) fixed 4-iteration run")
    stF = mainsys.run(bank_te, max_iter=4, record_X=True)
    ansF = Answers(mainsys, stF, mainsys.models)
    log("Test: FAIR-RAG (XGBoost) fixed 4-iteration run")
    stX = xgbsys.run(bank_te, max_iter=4, record_X=True)
    ansX = Answers(xgbsys, stX, xgbsys.models)

    def sea_system(states, ans, c, max_iter, use_stop=True):
        ts = ([stop_iter(s, c.sea_threshold, max_iter) for s in states] if use_stop
              else [min(max_iter, len(s.snaps)) for s in states])
        early = [t < max_iter for t in ts] if use_stop else None
        return records(states, ts, ans.get(ts), early), ts

    recs = {}
    recs[MAIN], tsF = sea_system(stF, ansF, cfgL, cfgL.max_iter)
    recs[XGBN], tsX = sea_system(stX, ansX, comps["xgb"]["cfg"], cfgL.max_iter)
    K = int(math.ceil(recs[MAIN]["n_pass"].mean()))
    log(f"Matched passage budget K = ceil(mean FAIR-RAG evidence size) = {K}")

    def simple(name, **kw):
        log(f"Test: {name}")
        st = mainsys.run(bank_te, **kw)
        ts = [len(s.snaps) for s in st]
        recs[name] = records(st, ts, Answers(mainsys, st, mainsys.models).get(ts))
        return st

    simple("Standard RAG (top-5)", max_iter=1, decompose_mode="question", use_filter=False,
           max_evidence=5)
    std_name = f"Standard RAG (matched budget, top-{K})"
    simple(std_name, max_iter=1, decompose_mode="question", use_filter=False,
           max_evidence=K, topk=max(cfgL.topk, K))
    rr_name = f"Standard RAG + filter rerank (top-{K})"
    log(f"Test: {rr_name}")
    rrsys = FairRAG(retr, kb, dataclasses.replace(cfgL, filter_threshold=0.0), mainsys.models)
    st_rr = rrsys.run(bank_te, max_iter=1, decompose_mode="question", max_evidence=K, topk=30)
    recs[rr_name] = records(st_rr, [1] * len(st_rr), Answers(rrsys, st_rr, rrsys.models).get(
        [1] * len(st_rr)))
    simple("Iter-Retgen loop (3 rounds)", max_iter=3, decompose_mode="question",
           refine="retgen", use_filter=False, max_evidence=K, topk=max(cfgL.topk, K))
    recs["Ablation A: no SEA stop (fixed 3)"], _ = sea_system(stF, ansF, cfgL, cfgL.max_iter,
                                                              use_stop=False)
    log("Test: Ablation B (no gap-aware refinement)")
    stB2 = mainsys.run(bank_te, max_iter=cfgL.max_iter, refine="retgen")
    recs["Ablation B: no gap refinement"], _ = sea_system(
        stB2, Answers(mainsys, stB2, mainsys.models), cfgL, cfgL.max_iter)
    simple("Ablation C: no evidence filter", max_iter=cfgL.max_iter, use_filter=False,
           max_evidence=K)

    # ---- 6. metrics
    log("Computing metrics, bootstrap CIs and plots")
    summ = {}
    for n, r in recs.items():
        s = E.summarize(r)
        s["joint_ci"] = E.bootstrap_ci(r["joint"], opts["n_boot"])
        s["f1_ci"] = E.bootstrap_ci(r["f1"], opts["n_boot"])
        summ[n] = s
    boot = {}
    for n in recs:
        if n == MAIN:
            continue
        boot[n] = {"joint_evidence_recall": E.paired_bootstrap(recs[MAIN]["joint"],
                                                               recs[n]["joint"], opts["n_boot"]),
                   "answer_f1": E.paired_bootstrap(recs[MAIN]["f1"], recs[n]["f1"],
                                                   opts["n_boot"])}

    sweep = []
    for m in range(1, 5):
        for mode in ("sea_stop", "fixed"):
            r, _ = sea_system(stF, ansF, cfgL, m, use_stop=(mode == "sea_stop"))
            s = E.summarize(r)
            s.update({"max_iter": m, "mode": mode})
            sweep.append(s)

    # components on test
    def filter_rows(states, t):
        X = np.vstack([s.snaps[t - 1]["X"] for s in states])
        y = np.concatenate([gold_mask(s.qi, s.snaps[t - 1]["pool_ids"]) for s in states])
        return X, y
    XtF, ytF = filter_rows(stF, cfgL.max_iter)
    pL = predict_proba(mainsys.models["filter"], XtF)
    pX = predict_proba(xgbsys.models["filter"], XtF)

    def sea_rows(states, max_iter):
        y, p = [], []
        for s in states:
            for sn in s.snaps[:max_iter]:
                y.append(is_joint(s.qi, sn["ev"]))
                p.append(sn["sea_p"])
        return np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    syL, spL = sea_rows(stF, cfgL.max_iter)
    syX, spX = sea_rows(stX, cfgL.max_iter)
    cx = comps["xgb"]["cfg"]
    sea_L = E.binary_metrics(syL, spL, cfgL.sea_threshold)
    sea_X = E.binary_metrics(syX, spX, cx.sea_threshold)
    for d in (sea_L, sea_X):
        d["false_sufficient_rate"] = 1.0 - d["precision"]

    def cand_recall(states, ts):
        hits = []
        for s, t in zip(states, ts):
            sn = s.snaps[t - 1]
            texts, _, _ = answer_candidates(s.qi, sn["ev"], sn["ev_probs"], kb)
            hits.append(float(any(ans_f1(x, s.qi.answer or "") >= 0.5 for x in texts)))
        return float(np.mean(hits))
    rm = recs[MAIN]
    complete = rm["joint"] == 1
    answerer = {
        "f1_when_evidence_complete": float(rm.loc[complete, "f1"].mean()) if complete.any() else 0.0,
        "em_when_evidence_complete": float(rm.loc[complete, "em"].mean()) if complete.any() else 0.0,
        "candidate_recall": cand_recall(stF, tsF),
        "xgb_f1_when_evidence_complete": float(
            recs[XGBN].loc[recs[XGBN]["joint"] == 1, "f1"].mean())
        if (recs[XGBN]["joint"] == 1).any() else 0.0,
    }
    components = {
        "filter_lightgbm": E.binary_metrics(ytF, pL, cfgL.filter_threshold),
        "filter_xgboost": E.binary_metrics(ytF, pX, cx.filter_threshold),
        "sea_lightgbm": sea_L, "sea_xgboost": sea_X, "answerer": answerer,
    }

    # recall@k on test (single-shot, question as query)
    tq = [s.qi for s in stF]
    ks = list(range(1, 21))
    rak = {"k": ks}
    for m in ("bm25", "lsa", "hybrid"):
        r = retr.rank([q.text for q in tq], m, 20)
        rak[m] = {"joint": [float(np.mean([set(q.gold) <= set(x[:k].tolist())
                                           for q, x in zip(tq, r)])) for k in ks],
                  "paragraph": [float(np.mean([len(set(q.gold) & set(x[:k].tolist())) / 2
                                               for q, x in zip(tq, r)])) for k in ks]}

    by_type = []
    for n in [MAIN, XGBN, std_name, rr_name, "Iter-Retgen loop (3 rounds)"]:
        for t, g in recs[n].groupby("type"):
            by_type.append({"system": n, "type": t, "n": int(len(g)),
                            "joint_evidence_recall": float(g["joint"].mean()),
                            "answer_f1": float(g["f1"].mean()), "answer_em": float(g["em"].mean())})

    fail = {}
    for n in [MAIN, XGBN, "Ablation A: no SEA stop (fixed 3)", std_name, rr_name,
              "Iter-Retgen loop (3 rounds)"]:
        cats = recs[n].apply(E.failure_type, axis=1)
        fail[n] = {c: float((cats == c).mean()) for c in E.FAILURES}

    imp = {"Evidence filter": gain_importance(mainsys.models["filter"], FILTER_FEATURES),
           "SEA sufficiency": gain_importance(mainsys.models["sea"], SEA_FEATURES),
           "Answer ranker": gain_importance(mainsys.models["answer"], ANSWER_FEATURES)}
    imp_x = {"Evidence filter": gain_importance(xgbsys.models["filter"], FILTER_FEATURES),
             "SEA sufficiency": gain_importance(xgbsys.models["sea"], SEA_FEATURES),
             "Answer ranker": gain_importance(xgbsys.models["answer"], ANSWER_FEATURES)}

    split_counts = {nm: df.iloc[ix]["type"].value_counts().sort_index().to_dict()
                    for nm, ix in [("train", tr), ("val", va), ("test", te)]}
    dataset = {"split_type_counts": split_counts, "n_paragraphs": len(corpus["titles"]),
               "paragraph_lengths": retr.doc_len, "n_questions": int(len(df))}

    R = {"systems": summ, "iteration_sweep": sweep, "recall_at_k": rak,
         "filter_curves": {"LightGBM filter": (ytF, pL), "XGBoost filter": (ytF, pX)},
         "sea_curves": {"LightGBM SEA": (syL, spL, cfgL.sea_threshold),
                        "XGBoost SEA": (syX, spX, cx.sea_threshold)},
         "by_type": by_type, "failure_taxonomy": fail, "feature_importance": imp,
         "iterations_used": {MAIN: tsF, XGBN: tsX}, "dataset": dataset}
    plots = E.plot_all(R, plots_dir)

    # ---- 7. save artifacts
    bundle = {"filter": mainsys.models["filter"], "sea": mainsys.models["sea"],
              "answer": mainsys.models["answer"], "config": cfgL.to_dict(),
              "lsa_weight": best_w, "feature_names": {"filter": FILTER_FEATURES,
                                                      "sea": SEA_FEATURES,
                                                      "answer": ANSWER_FEATURES}}
    joblib.dump(bundle, art_dir / "model.joblib", compress=3)
    joblib.dump({**bundle, "filter": xgbsys.models["filter"], "sea": xgbsys.models["sea"],
                 "answer": xgbsys.models["answer"], "config": cx.to_dict()},
                art_dir / "xgb_components.joblib", compress=3)

    # per-question results for all systems (dashboard)
    allrec = []
    for n, r in recs.items():
        d = r.drop(columns=["pred"]).copy()
        d.insert(0, "system", n)
        allrec.append(d)
    pd.concat(allrec).to_csv(art_dir / "test_records.csv", index=False)

    # sample predictions
    rs, rx = recs[std_name], recs[XGBN]
    sp_rows = []
    for i, s in enumerate(stF):
        r = rm.iloc[i]
        sn = s.snaps[tsF[i] - 1]
        sp_rows.append({
            "id": s.qi.qid, "question": s.qi.text, "type": s.qi.qtype,
            "true_answer": s.qi.answer, "prediction": r["pred"],
            "confidence": round(r["conf"], 4), "exact_match": r["em"], "f1": round(r["f1"], 4),
            "cited_paragraph": kb.titles[r["cite"]] if r["cite"] >= 0 else "",
            "citation_is_gold": r["cite_ok"],
            "gold_paragraphs": " | ".join(kb.titles[d] for d in s.qi.gold),
            "evidence_paragraphs": " | ".join(kb.titles[int(d)] for d in sn["ev"]),
            "joint_evidence_recall": r["joint"], "iterations": tsF[i],
            "retrieval_calls": r["calls"], "failure_type": E.failure_type(r),
            "std_rag_prediction": rs.iloc[i]["pred"], "std_rag_joint_recall": rs.iloc[i]["joint"],
            "xgb_prediction": rx.iloc[i]["pred"], "xgb_joint_recall": rx.iloc[i]["joint"],
        })
    pd.DataFrame(sp_rows).to_csv(art_dir / "sample_predictions.csv", index=False,
                                 encoding="utf-8")

    with open(art_dir / "test_traces.jsonl", "w", encoding="utf-8") as fh:
        for i, s in enumerate(stF):
            a = ansF.c[(i, tsF[i])]
            tr_ = trace_dict(s, tsF[i], kb, a, cfgL.sea_threshold)
            tr_.update({"id": s.qi.qid, "type": s.qi.qtype, "gold_answer": s.qi.answer,
                        "gold_titles": [kb.titles[d] for d in s.qi.gold],
                        "joint_evidence_recall": float(rm.iloc[i]["joint"]),
                        "f1": float(rm.iloc[i]["f1"]),
                        "failure_type": E.failure_type(rm.iloc[i])})
            fh.write(json.dumps(tr_, ensure_ascii=False) + "\n")

    def clean(s):
        return {k: v for k, v in s.items()}

    main_m = clean(summ[MAIN])
    metrics = {
        "task": "Multi-hop evidence collection and faithful extractive QA (FAIR-RAG control loop)",
        "dataset": "HotpotQA distractor validation (hotpotqa/hotpot_qa), pooled open-retrieval corpus",
        "primary_metric": "joint_evidence_recall",
        "higher_is_better": True,
        "value": main_m["joint_evidence_recall"],
        "baseline": summ[std_name]["joint_evidence_recall"],
        "baseline_name": std_name,
        "metrics": {
            **main_m,
            "xgboost": clean(summ[XGBN]),
            "systems": summ,
            "paired_bootstrap_vs_main": boot,
            "components": components,
            "iteration_sweep": sweep,
            "recall_at_k": rak,
            "by_type": by_type,
            "failure_taxonomy": fail,
            "feature_importance": {"lightgbm": imp, "xgboost": imp_x},
            "iterations_used_hist": {n: np.bincount(np.asarray(v), minlength=5)[1:].tolist()
                                     for n, v in R["iterations_used"].items()},
            "matched_budget_k": K,
            "config": cfgL.to_dict(),
            "xgb_config": cx.to_dict(),
            "tuning": tuning,
            "data": {"n_questions": int(len(df)), "n_dropped_questions": int(n_dropped),
                     "n_paragraphs": len(corpus["titles"]), "split_type_counts": split_counts,
                     "n_val": int(len(va)), "quick_mode": bool(quick)},
            "plots": plots,
        },
        "baseline_metrics": clean(summ[std_name]),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_seconds": round(clock(), 1),
    }
    with open(art_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(jsonable(metrics), fh, indent=2)
    log(f"Wrote {art_dir.name}/metrics.json and {len(plots)} plots")
    return metrics


def print_summary(m):
    log("=" * 72)
    log(f"Test questions: {m['n_test']}   train: {m['n_train']}   time: {m['train_seconds']}s")
    log(f"{'system':46s} {'jointRec':>8s} {'F1':>6s} {'EM':>6s} {'pass':>5s} {'calls':>5s}")
    for n, s in m["metrics"]["systems"].items():
        log(f"{n[:46]:46s} {s['joint_evidence_recall']:8.4f} {s['answer_f1']:6.3f} "
            f"{s['answer_em']:6.3f} {s['mean_passages']:5.2f} {s['mean_retrieval_calls']:5.2f}")
    b = m["metrics"]["paired_bootstrap_vs_main"][m["baseline_name"]]["joint_evidence_recall"]
    log(f"PRIMARY joint evidence recall: FAIR-RAG {m['value']:.4f} vs {m['baseline_name']} "
        f"{m['baseline']:.4f}  diff {b['diff']:+.4f} [95% CI {b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")
    log("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small subsample for a fast check")
    ap.add_argument("--demo-only", action="store_true",
                    help="only rebuild artifacts/demo/ (live dashboard demo) from saved models")
    args = ap.parse_args()
    raw = D.load_raw()
    corpus, df, n_drop = D.build_corpus(raw)
    log(f"Questions kept: {len(df)} (dropped {n_drop}); paragraphs: {len(corpus['titles'])}")
    if not args.demo_only:
        m = run_experiment(df, corpus, ART_DIR, quick=args.quick, n_dropped=n_drop)
        print_summary(m)
    # browser-safe live-demo bundle for the dashboard (numpy + XGBoost JSON only)
    from src.demo_export import export_demo
    export_demo(df, corpus, get_retriever(retrieval_texts(corpus), DEFAULT_OPTS), ART_DIR,
                n_questions=100 if args.quick else 200)


if __name__ == "__main__":
    main()
