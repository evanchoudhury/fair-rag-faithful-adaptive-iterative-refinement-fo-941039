"""Export the browser-safe live-demo bundle into artifacts/demo/ (training side).

The bundle holds: a demo corpus (context paragraphs of the first N test questions), the matching
slice of the trained BM25 / LSA index, the XGBoost components as JSON boosters and the demo
questions. It also runs FAIR-RAG and Standard RAG on the demo corpus and verifies that the
sliced index reproduces the full index scores; results go into metrics.json["demo"].
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from . import data as D
from . import text as T
from .demo import load_demo
from .model import QBank, QInfo
from .utils import log


def export_demo(df, corpus, retr, art_dir: Path, n_questions: int = 200) -> dict:
    art_dir = Path(art_dir)
    out = art_dir / "demo"
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(art_dir / "xgb_components.joblib")
    retr.set_lsa_weight(bundle["lsa_weight"])

    _, _, te = D.split(df)
    sel = te[:n_questions]
    t2i = corpus["title2id"]
    ids = sorted({t2i[t] for i in sel for t in df.iloc[i]["ctx_titles"] if t in t2i})
    ids = np.asarray(ids, dtype=np.int64)
    pos = {int(d): j for j, d in enumerate(ids)}
    log(f"Demo bundle: {len(sel)} test questions, {len(ids)} paragraphs")

    # ---- index slice
    Wd = retr.W[ids]
    # vocabulary = terms of the demo paragraphs + terms of the demo questions (so the LSA
    # projection of the demo questions is complete)
    qcols = retr.vec.transform([df.iloc[i]["question"] for i in sel]).indices
    cols = np.unique(np.concatenate([Wd.indices, qcols]))
    Wd = Wd[:, cols].tocsr()
    Wd.sort_indices()
    names = retr.vec.get_feature_names_out()
    uni_sub = retr.unigram_mask[cols]
    uni_cols = cols[uni_sub]
    comp_pos = np.searchsorted(retr.uni_idx, uni_cols)
    assert np.all(retr.uni_idx[comp_pos] == uni_cols)
    np.savez_compressed(
        out / "index.npz", n_docs=np.int64(len(ids)),
        W_data=Wd.data.astype(np.float32), W_indices=Wd.indices.astype(np.int32),
        W_indptr=Wd.indptr.astype(np.int64), unigram_mask=uni_sub,
        doc_len=retr.doc_len[ids].astype(np.float32), D=retr.D[ids].astype(np.float32),
        uni_idx=np.where(uni_sub)[0].astype(np.int32),
        tfidf_idf=retr.tfidf.idf_[comp_pos].astype(np.float32),
        # float16 halves the largest array (keeps artifacts/ small); effect is checked below
        components=retr.svd.components_[:, comp_pos].astype(np.float16))
    mentions = sorted({m for m in (T.title_mention(t) for t in corpus["titles"]) if m})
    meta = {"vocab": [str(x) for x in names[cols]],
            "full_unigrams": [str(x) for x in names[retr.unigram_mask]],
            "full_title_mentions": mentions, "lsa_weight": float(retr.lsa_weight),
            "rrf_k": int(retr.rrf_k), "depth": int(retr.depth), "config": bundle["config"],
            "n_full_paragraphs": len(corpus["titles"])}
    with open(out / "index_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    questions = []
    for i in sel:
        r = df.iloc[i]
        questions.append({"id": r["id"], "question": r["question"], "answer": r["answer"],
                          "type": r["type"], "level": r["level"],
                          "gold_titles": [corpus["titles"][d] for d in r["gold_ids"]]})
    with open(out / "corpus.json", "w", encoding="utf-8") as fh:
        json.dump({"titles": [corpus["titles"][d] for d in ids],
                   "sents": [corpus["sents"][d] for d in ids], "questions": questions}, fh,
                  ensure_ascii=False)
    for k in ("filter", "sea", "answer"):
        bundle[k].get_booster().save_model(str(out / f"xgb_{k}.json"))

    # ---- verification: sliced index and JSON boosters reproduce the originals
    system, dcorp, _ = load_demo(out)
    qs = [q["question"] for q in questions]
    Qb, Qu, qd = retr.encode(qs)
    qb2, qu2, qd2 = system.retr.encode(qs)
    qrows = [i for i, r in enumerate(sel) for t in df.iloc[r]["ctx_titles"] if t in t2i]
    dfull = [t2i[t] for r in sel for t in df.iloc[r]["ctx_titles"] if t in t2i]
    a = retr.pair_scores(Qb, Qu, qd, qrows, dfull)
    b = system.retr.pair_scores(qb2, qu2, qd2, qrows, [pos[d] for d in dfull])
    diffs = [float(np.max(np.abs(x - y))) for x, y in zip(a, b)]
    rng = np.random.default_rng(0)
    Xr = rng.random((500, len(bundle["feature_names"]["filter"]))).astype(np.float32) * 10
    pdiff = float(np.max(np.abs(bundle["filter"].predict_proba(Xr)[:, 1]
                                - system.models["filter"].predict_proba(Xr)[:, 1])))
    log(f"Demo slice check: max |diff| bm25 {diffs[0]:.2e}, cos {diffs[1]:.2e}, "
        f"coverage {diffs[2]:.2e}; JSON booster vs classifier {pdiff:.2e}")

    # ---- run FAIR-RAG (SEA stop) and Standard RAG on the demo corpus
    with open(art_dir / "metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    K = int(metrics["metrics"].get("matched_budget_k", 6))
    qinfos = [QInfo(q["question"], tuple(sorted(pos[t2i[t]] for t in q["gold_titles"])),
                    q["answer"], q["type"], q["level"], q["id"]) for q in questions]
    cfg = system.cfg
    res = {}
    for name, kw in [("FAIR-RAG (XGBoost, SEA stop)", {"stop_threshold": cfg.sea_threshold}),
                     (f"Standard RAG (top-{K})", {"max_iter": 1, "decompose_mode": "question",
                                                 "use_filter": False, "max_evidence": K,
                                                 "topk": max(cfg.topk, K)})]:
        bank = QBank(qinfos, system.retr)
        st = system.run(bank, **kw)
        items = [(s.qi, s.snaps[-1]["ev"], s.snaps[-1]["ev_probs"]) for s in st]
        ans = system.answer(items)
        joint = [float(set(s.qi.gold) <= {int(d) for d in s.snaps[-1]["ev"]}) for s in st]
        f1 = [T.f1_score(a_["pred"], s.qi.answer) for s, a_ in zip(st, ans)]
        em = [T.exact_match(a_["pred"], s.qi.answer) for s, a_ in zip(st, ans)]
        res[name] = {"joint_evidence_recall": round(float(np.mean(joint)), 6),
                     "answer_f1": round(float(np.mean(f1)), 6),
                     "answer_em": round(float(np.mean(em)), 6),
                     "mean_passages": round(float(np.mean([len(s.snaps[-1]["ev"]) for s in st])), 6),
                     "mean_iterations": round(float(np.mean([len(s.snaps) for s in st])), 6),
                     "mean_retrieval_calls": round(float(np.mean([s.calls for s in st])), 6)}
        log(f"  demo corpus | {name}: joint {res[name]['joint_evidence_recall']:.3f}  "
            f"F1 {res[name]['answer_f1']:.3f}")
    size = sum(p.stat().st_size for p in out.iterdir())
    demo = {"n_questions": len(sel), "n_paragraphs": int(len(ids)),
            "n_full_paragraphs": len(corpus["titles"]), "vocab_size": int(len(cols)),
            "bundle_mb": round(size / 2 ** 20, 2), "sea_threshold": cfg.sea_threshold,
            "slice_max_abs_diff": {"bm25": diffs[0], "cos": diffs[1], "coverage": diffs[2],
                                   "booster_vs_classifier": pdiff},
            "systems": res}
    metrics["demo"] = demo
    with open(art_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"Wrote artifacts/demo/ ({demo['bundle_mb']} MB) and metrics.json['demo']")
    return demo
