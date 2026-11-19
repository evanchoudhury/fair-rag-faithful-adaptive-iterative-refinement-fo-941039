"""Hybrid retriever: BM25 (sparse) + LSA (dense) fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from .text import analyzer


def _top(scores: np.ndarray, k: int, positive_only: bool) -> np.ndarray:
    """Indices of the k largest scores, sorted by (score desc, index asc)."""
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    idx = idx[np.lexsort((idx, -scores[idx]))]
    if positive_only:
        idx = idx[scores[idx] > 0]
    return idx


class HybridRetriever:
    def __init__(self, k1=1.5, b=0.75, lsa_dim=256, rrf_k=60, depth=50, min_df=2, seed=42,
                 lsa_weight=1.0):
        self.k1, self.b, self.lsa_dim, self.rrf_k = k1, b, lsa_dim, rrf_k
        self.lsa_weight = lsa_weight  # weight of the dense ranking inside RRF (tuned)
        self.depth, self.min_df, self.seed = depth, min_df, seed
        self._cache: dict = {}

    # ------------------------------------------------------------- fitting
    def fit(self, texts: list[str]) -> "HybridRetriever":
        self.vec = CountVectorizer(analyzer=analyzer, min_df=self.min_df, dtype=np.float32)
        X = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        if hasattr(self.vec, "stop_words_"):
            self.vec.stop_words_ = None  # large and not needed
        N, V = X.shape
        df_ = np.bincount(X.indices, minlength=V)
        idf = np.log1p((N - df_ + 0.5) / (df_ + 0.5)).astype(np.float32)
        dl = np.asarray(X.sum(1)).ravel()
        rel = np.repeat(dl / max(dl.mean(), 1e-9), np.diff(X.indptr)).astype(np.float32)
        tf = X.data
        W = X.copy()
        W.data = (tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * rel))
                  * idf[X.indices]).astype(np.float32)
        self.W = W
        self.WT = W.T.tocsr()
        uni = np.array([" " not in f for f in self.vec.get_feature_names_out()])
        self.unigram_mask = uni
        U = X.copy()
        U.data = uni[X.indices].astype(np.float32)
        U.eliminate_zeros()
        self.U = U
        self.doc_len = np.asarray((X[:, uni] if V else X).sum(1)).ravel().astype(np.float32)
        # LSA runs on the unigram columns only (bigrams add cost but little topical signal)
        self.uni_idx = np.where(uni)[0]
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        Tm = self.tfidf.fit_transform(X[:, self.uni_idx])
        dim = int(max(2, min(self.lsa_dim, len(self.uni_idx) - 1, N - 1)))
        self.svd = TruncatedSVD(n_components=dim, random_state=self.seed, n_iter=5)
        self.D = normalize(self.svd.fit_transform(Tm)).astype(np.float32)
        self._cache = {}
        return self

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_cache"] = {}
        st.pop("WT", None)
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.__dict__.setdefault("lsa_weight", 1.0)
        self.WT = self.W.T.tocsr()

    # ------------------------------------------------------------ encoding
    def encode(self, texts: list[str]):
        """Binary BM25 query vectors, unigram-only binary vectors, LSA vectors."""
        Qc = self.vec.transform(texts).tocsr().astype(np.float32)
        Qb = Qc.copy()
        Qb.data[:] = 1.0
        Qu = Qb.copy()
        Qu.data = self.unigram_mask[Qb.indices].astype(np.float32)
        Qu.eliminate_zeros()
        qd = normalize(self.svd.transform(self.tfidf.transform(Qc[:, self.uni_idx])))
        qd = qd.astype(np.float32)
        return Qb, Qu, qd

    def pair_scores(self, Qb, Qu, qd, qrows, dids):
        """BM25(question, doc), cosine and question-unigram coverage for (q, doc) pairs."""
        qrows = np.asarray(qrows, dtype=np.int64)
        dids = np.asarray(dids, dtype=np.int64)
        if len(dids) == 0:
            z = np.zeros(0, dtype=np.float32)
            return z, z, z
        bm = np.asarray(Qb[qrows].multiply(self.W[dids]).sum(1)).ravel()
        cov = np.asarray(Qu[qrows].multiply(self.U[dids]).sum(1)).ravel()
        nq = np.asarray(Qu.sum(1)).ravel()[qrows]
        cov = cov / np.maximum(nq, 1)
        cos = np.einsum("ij,ij->i", qd[qrows], self.D[dids])
        return bm.astype(np.float32), cos.astype(np.float32), cov.astype(np.float32)

    # ------------------------------------------------------------- search
    def search(self, queries: list[str], batch_size: int = 256):
        """For each query: (ids, fused, bm25, cos) arrays of the top `depth` RRF results."""
        todo = [q for q in dict.fromkeys(queries) if q not in self._cache]
        for s in range(0, len(todo), batch_size):
            chunk = todo[s:s + batch_size]
            Qb, _, qd = self.encode(chunk)
            S = (Qb @ self.WT).toarray()
            C = qd @ self.D.T
            nnz = np.diff(Qb.indptr)
            for i, q in enumerate(chunk):
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

    def set_lsa_weight(self, w: float) -> None:
        self.lsa_weight = float(w)
        self._cache = {}

    def rank(self, queries: list[str], method: str, k: int, batch_size: int = 256):
        """Single-method top-k ids (bm25 / lsa / hybrid) for the Recall@k curve."""
        if method == "hybrid":
            return [r[0][:k] for r in self.search(queries)]
        out = []
        for s in range(0, len(queries), batch_size):
            chunk = queries[s:s + batch_size]
            Qb, _, qd = self.encode(chunk)
            if method == "bm25":
                S = (Qb @ self.WT).toarray()
                out += [_top(S[i], k, True) for i in range(len(chunk))]
            else:
                C = qd @ self.D.T
                out += [_top(C[i], k, False) if Qb[i].nnz else np.zeros(0, dtype=np.int64)
                        for i in range(len(chunk))]
        return out
