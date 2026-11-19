"""Offline smoke test (no network, no dataset download, < 60 s).

Builds a small synthetic multi-hop QA set with the same schema as the HotpotQA rows used by
train.py, runs the whole pipeline end to end into a scratch folder, checks shapes, files and
edge cases, and prints `ok`. Exits non-zero on any failure.

    python selfcheck.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REQUIRED_KEYS = ["task", "dataset", "primary_metric", "higher_is_better", "value", "baseline",
                 "baseline_name", "metrics", "baseline_metrics", "n_train", "n_test",
                 "train_seconds"]


def synthetic_df(n_items: int = 48, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    syl = ["ka", "lo", "mi", "ra", "ven", "tor", "qui", "zel", "dra", "pon", "sul", "fen",
           "bri", "mox", "tal", "ser"]

    def name(k=2):
        return "".join(rng.choice(syl) for _ in range(k)).capitalize()

    items = []
    used = set()
    while len(items) < n_items:
        dev, comp, per = name(3), name(2) + " Corp", name(2) + " " + name(2)
        if dev in used or comp in used:
            continue
        used.update([dev, comp])
        year = rng.randint(1900, 2010)
        items.append({
            "dev": dev, "comp": comp, "per": per, "year": year,
            "p_dev": [f"The {dev} is a handheld device.", f"It is made by {comp}.",
                      f"The {dev} was released in {year + 5}."],
            "p_comp": [f"{comp} is a technology company.",
                       f"{comp} was founded by {per} in {year}.",
                       "It is based in a large city."],
        })
    rows = []
    for i, it in enumerate(items):
        others = [items[(i + k) % len(items)] for k in (1, 2, 3)]
        ctx_t = [it["dev"], it["comp"]] + [o["dev"] for o in others]
        ctx_s = [it["p_dev"], it["p_comp"]] + [o["p_dev"] for o in others]
        rows.append({"id": f"b{i}", "question": f"Who founded the company that makes the {it['dev']}?",
                     "answer": it["per"], "type": "bridge", "level": "hard",
                     "sf_titles": [it["dev"], it["comp"]], "ctx_titles": ctx_t, "ctx_sents": ctx_s})
        j = items[(i + 5) % len(items)]
        first = it["comp"] if it["year"] <= j["year"] else j["comp"]
        rows.append({"id": f"c{i}", "question": f"Which was founded first, {it['comp']} or {j['comp']}?",
                     "answer": first, "type": "comparison", "level": "hard",
                     "sf_titles": [it["comp"], j["comp"]],
                     "ctx_titles": [it["comp"], j["comp"], others[0]["comp"]],
                     "ctx_sents": [it["p_comp"], j["p_comp"], others[0]["p_comp"]]})
    # edge case: a question with no usable tokens and gold titles missing from the corpus
    rows.append({"id": "edge", "question": "??", "answer": "yes", "type": "bridge",
                 "level": "hard", "sf_titles": ["Missing title"], "ctx_titles": [],
                 "ctx_sents": []})
    return pd.DataFrame(rows)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def unit_checks():
    from src import text as T
    check(T.normalize_answer("The  Cat!") == "cat", "normalize_answer")
    check(T.f1_score("yes", "no") == 0.0 and T.exact_match("Yes", "yes") == 1.0, "yes/no EM/F1")
    check(abs(T.f1_score("new york city", "york city") - 0.8) < 1e-9, "token F1")
    check(T.entity_spans("") == [] and T.content_words("") == [], "empty text")
    ents = T.entity_spans("Were Scott Derrickson and Ed Wood of the same nationality?")
    check(ents == ["Scott Derrickson", "Ed Wood"], f"entity spans {ents}")
    check(T.wh_type("Are both cats?") == T.WH_NAMES.index("yesno"), "wh yes/no")
    check(T.title_mention("Gone (film)") == "gone", "title mention")


def main() -> int:
    t0 = time.time()
    scratch = ROOT / "artifacts" / "_selfcheck"
    try:
        unit_checks()
        from src import data as D
        from src.model import QBank, QInfo
        import train as TR

        raw = synthetic_df()
        corpus, df, n_drop = D.build_corpus(raw)
        check(n_drop == 1 and len(df) == len(raw) - 1, "edge-case question must be dropped")
        check(all(len(g) == 2 for g in df["gold_ids"]), "gold ids")
        tr, va, te = D.split(df)
        check(len(set(tr) & set(te)) == 0 and len(tr) + len(va) + len(te) == len(df), "split")

        if scratch.exists():
            shutil.rmtree(scratch)
        opts = {"lsa_dim": 16, "folds": 2, "filter_grid": [{}], "n_boot": 50,
                "retr_cache": None, "tune_sample": 40, "min_child_samples": 2,
                "lsa_weights": [0.0, 0.5]}
        m = TR.run_experiment(df, corpus, scratch, opts=opts)

        for k in REQUIRED_KEYS:
            check(k in m, f"metrics key {k}")
        on_disk = json.loads((scratch / "metrics.json").read_text(encoding="utf-8"))
        check(set(REQUIRED_KEYS) <= set(on_disk), "metrics.json keys")
        check("xgboost" in on_disk["metrics"], "xgboost metrics")
        check(0.0 <= on_disk["value"] <= 1.0 and 0.0 <= on_disk["baseline"] <= 1.0, "ranges")
        check(on_disk["n_test"] == len(te), "n_test")
        plots = list((scratch / "plots").glob("*.png"))
        check(len(plots) >= 4, f"plots: {len(plots)}")
        sp = pd.read_csv(scratch / "sample_predictions.csv")
        check(len(sp) == len(te), "sample predictions rows")
        for col in ("true_answer", "prediction", "confidence"):
            check(col in sp.columns, f"column {col}")
        check(sp["confidence"].between(0, 1).all(), "confidence range")
        n_traces = sum(1 for _ in open(scratch / "test_traces.jsonl", encoding="utf-8"))
        check(n_traces == len(te), "traces")
        check((scratch / "model.joblib").stat().st_size < 20 * 1024 * 1024, "model size")

        # reload the saved model and answer free-text questions, incl. degenerate ones
        import joblib
        from src.model import FairRAG, KB, Config
        bundle = joblib.load(scratch / "model.joblib")
        retr = TR.get_retriever(TR.retrieval_texts(corpus), opts)
        retr.set_lsa_weight(bundle["lsa_weight"])
        sysm = FairRAG(retr, KB(corpus["titles"], corpus["sents"]), Config(**bundle["config"]),
                       {k: bundle[k] for k in ("filter", "sea", "answer")})
        q = df.iloc[0]["question"]
        tr_ = sysm.ask(q)
        check(1 <= tr_["n_iter"] <= sysm.cfg.max_iter and isinstance(tr_["answer"], str), "ask")
        check(len(tr_["iterations"][0]["evidence"]) >= 1, "evidence kept")
        for weird in ["", "???", "zzzz qqqq unknownword"]:
            out = sysm.ask(weird)
            check(isinstance(out["answer"], str), f"ask({weird!r})")
        bank = QBank([QInfo("")], retr)
        st = sysm.run(bank, max_iter=2)
        check(len(st) == 1 and len(st[0].snaps) == 2 and len(st[0].ev) == 0, "empty question")
        ans = sysm.answer([(st[0].qi, st[0].ev, st[0].ev_probs)])
        check(ans[0]["abstain"] and ans[0]["pred"] == "", "abstain on empty evidence")
        X = sysm.retr.search(["", "Corp"])
        check(len(X) == 2 and len(X[0][0]) == 0, "search shapes")
    except Exception:
        traceback.print_exc()
        print("selfcheck FAILED")
        return 1
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
    print(f"selfcheck passed in {time.time() - t0:.1f}s")
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
