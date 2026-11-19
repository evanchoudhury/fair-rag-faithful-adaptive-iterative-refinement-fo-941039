"""Browser-safe live demo: a compact slice of the trained retriever + XGBoost components.

Only numpy, scipy, xgboost and the standard library are used here (plus src.model / src.text),
so the dashboard can run the real FAIR-RAG loop inside the browser (stlite / Pyodide).

The demo corpus is a subset of the full 66k-paragraph corpus (the context paragraphs of a set
of test questions). BM25 weights, paragraph lengths, LSA vectors and the title list are copied
from the full index, so scores for the demo paragraphs are the same as in the full system; only
the pool of candidate paragraphs is smaller.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .model import KB, Config, FairRAG
from .text import analyzer


def _top(scores: np.ndarray, k: int, positive_only: bool) -> np.ndarray:
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    idx = idx[np.lexsort((idx, -scores[idx]))]
    if positive_only:
        idx = idx[scores[idx] > 0]
    return idx


class DemoRetriever:
    """Same interface as src.retriever.HybridRetriever (encode / pair_scores / search)."""

    def __init__(self, npz: dict, meta: dict):
        self.vocab = {t: i for i, t in enumerate(meta["vocab"])}
        self.full_unigrams = frozenset(meta["full_unigrams"])
        self.lsa_weight = float(meta["lsa_weight"])
        self.rrf_k = int(meta["rrf_k"])
        self.depth = int(meta["depth"])
        V = len(meta["vocab"])
        n = int(npz["n_docs"])
        self.W = sp.csr_matrix((npz["W_data"].astype(np.float32), npz["W_indices"],
                                npz["W_indptr"]), shape=(n, V))
        self.WT = self.W.T.tocsr()
        self.unigram_mask = npz["unigram_mask"].astype(bool)
        U = self.W.copy()
        U.data = self.unigram_mask[U.indices].astype(np.float32)
        U.eliminate_zeros()
        self.U = U
        self.doc_len = npz["doc_len"].astype(np.float32)
        self.D = npz["D"].astype(np.float32)
        self.uni_idx = npz["uni_idx"].astype(np.int64)        # vocab positions of LSA terms
        self.idf = npz["tfidf_idf"].astype(np.float32)         # idf of those terms
        self.comp = npz["components"].astype(np.float32)       # (dim, len(uni_idx))
        self._cache: dict = {}

    # ---------------------------------------------------------- encoding
    def encode(self, texts):
        rows, cols, vals, nq = [], [], [], []
        for i, t in enumerate(texts):
            toks = analyzer(t)
            cnt: dict[int, int] = {}
            for w in toks:
                j = self.vocab.get(w)
                if j is not None:
                    cnt[j] = cnt.get(j, 0) + 1
            for j, c in cnt.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
            # question length for coverage counts unigrams of the FULL vocabulary
            nq.append(len({w for w in toks if " " not in w and w in self.full_unigrams}))
        Qc = sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                           shape=(len(texts), len(self.vocab)))
        Qc.sum_duplicates()
        Qb = Qc.copy()
        Qb.data[:] = 1.0
        Qu = Qb.copy()
        Qu.data = self.unigram_mask[Qb.indices].astype(np.float32)
        Qu.eliminate_zeros()
        T = Qc[:, self.uni_idx].tocsr().astype(np.float32)
        T.data = (1.0 + np.log(T.data)) * self.idf[T.indices]
        qd = np.asarray(T @ self.comp.T, dtype=np.float32)
        nrm = np.linalg.norm(qd, axis=1, keepdims=True)
        qd = qd / np.maximum(nrm, 1e-12)
        Qu._nq = np.asarray(nq, dtype=np.float32)
        return Qb, Qu, qd.astype(np.float32)

    def pair_scores(self, Qb, Qu, qd, qrows, dids):
        qrows = np.asarray(qrows, dtype=np.int64)
        dids = np.asarray(dids, dtype=np.int64)
        if len(dids) == 0:
            z = np.zeros(0, dtype=np.float32)
            return z, z, z
        bm = np.asarray(Qb[qrows].multiply(self.W[dids]).sum(1)).ravel()
        cov = np.asarray(Qu[qrows].multiply(self.U[dids]).sum(1)).ravel()
        nq = Qu._nq[qrows]
        cov = cov / np.maximum(nq, 1)
        cos = np.einsum("ij,ij->i", qd[qrows], self.D[dids])
        return bm.astype(np.float32), cos.astype(np.float32), cov.astype(np.float32)

    def search(self, queries):
        todo = [q for q in dict.fromkeys(queries) if q not in self._cache]
        if todo:
            Qb, _, qd = self.encode(todo)
            S = (Qb @ self.WT).toarray()
            C = qd @ self.D.T
            nnz = np.diff(Qb.indptr)
            for i, q in enumerate(todo):
                if nnz[i] == 0:
                    e = np.zeros(0, dtype=np.float32)
                    self._cache[q] = (np.zeros(0, dtype=np.int32), e, e, e)
                    continue
                rb = _top(S[i], self.depth, True)
                rl = _top(C[i], self.depth, False)
                ids = np.concatenate([rb, rl])
                contrib = np.concatenate([
                    1.0 / (self.rrf_k + 1 + np.arange(len(rb))),
                    self.lsa_weight / (self.rrf_k + 1 + np.arange(len(rl)))])
                uids, inv = np.unique(ids, return_inverse=True)
                fused = np.bincount(inv, weights=contrib)
                order = np.lexsort((uids, -fused))[: self.depth]
                top = uids[order]
                self._cache[q] = (top.astype(np.int32), fused[order].astype(np.float32),
                                  S[i, top].astype(np.float32), C[i, top].astype(np.float32))
        return [self._cache[q] for q in queries]


class BoosterClassifier:
    """Minimal predict_proba wrapper around an XGBoost Booster loaded from JSON."""

    def __init__(self, path: Path):
        import xgboost as xgb
        self._xgb = xgb
        self.booster = xgb.Booster()
        self.booster.load_model(str(path))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        p = self.booster.predict(self._xgb.DMatrix(X)).astype(np.float32)
        return np.column_stack([1.0 - p, p])


def load_demo(demo_dir: Path):
    """Return (FairRAG system, corpus dict, demo meta) built from artifacts/demo/."""
    demo_dir = Path(demo_dir)
    with open(demo_dir / "corpus.json", encoding="utf-8") as fh:
        corpus = json.load(fh)
    with open(demo_dir / "index_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    with np.load(demo_dir / "index.npz", allow_pickle=False) as z:
        npz = {k: z[k] for k in z.files}
    retr = DemoRetriever(npz, meta)
    kb = KB(corpus["titles"], corpus["sents"])
    # membership tests use the titles of the FULL corpus (as in training)
    kb.title_index = {m: 1 for m in meta["full_title_mentions"]}
    cfg = Config(**meta["config"])
    models = {k: BoosterClassifier(demo_dir / f"xgb_{k}.json") for k in ("filter", "sea", "answer")}
    return FairRAG(retr, kb, cfg, models), corpus, meta
