"""FAIR-RAG components and loop, with the LLM agents replaced by small trained models.

Decompose -> hybrid retrieve -> evidence filter -> Structured Evidence Assessment (SEA)
-> gap-aware query refinement -> (loop) -> faithful extractive answer with citation.

Baselines (Standard RAG, Iter-Retgen-style loop) and ablations reuse the same loop with
different switches, so that every system shares the retriever and the answerer.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from . import text as T

# ---------------------------------------------------------------- config


@dataclass
class Config:
    topk: int = 10              # new paragraphs taken per sub-query
    max_sub: int = 4            # max sub-queries per round
    max_iter: int = 3           # FAIR-RAG iteration cap (paper default)
    max_evidence: int = 8       # max paragraphs kept as evidence
    min_evidence: int = 2       # always keep at least this many
    filter_threshold: float = 0.2   # tuned on train out-of-fold predictions
    sea_threshold: float = 0.6      # tuned on the validation split
    anchors: int = 5            # paragraphs used as link anchors in the filter features

    def to_dict(self):
        return asdict(self)


SRC = {"question": 0, "entity": 1, "gap": 2, "bridge": 3, "retgen": 4}
SRC_NAMES = {v: k for k, v in SRC.items()}

FILTER_FEATURES = [
    "q_bm25", "q_bm25_rel", "q_bm25_rank", "q_cos", "q_cos_rank", "q_cov", "best_rank",
    "best_fused", "fused_sum", "sub_bm25", "n_hits", "first_iter", "src", "title_qcov",
    "title_in_q", "ent_cover", "in_link", "n_in_links", "out_link", "doc_len", "is_comp",
    "n_pool", "iter",
]
SEA_FEATURES = [
    "n_ev", "p1", "p2", "p3", "p_sum", "p12", "max_out", "n_find", "frac_conf", "n_unconf",
    "ent_frac", "n_ent", "bridge_slot", "bridge_conf", "link_score", "n_title_match",
    "is_comp", "is_yesno", "iter", "n_new", "n_pool", "calls",
]
ANSWER_FEATURES = [
    "ctype", "wh", "is_comp", "is_yesno", "in_q", "is_q_ent", "para_prob", "para_rank",
    "sent_idx", "sent_ov", "sent_ov_frac", "n_occ", "n_paras", "n_tok", "is_ev_title",
    "is_corpus_title", "has_digit", "is_year", "type_match", "n_cands", "win_ov",
    "para_linked", "para_title_in_q", "wh_noun_sent", "wh_noun_para", "yn_overlap",
]
CTYPES = {"ent": 0, "date": 1, "num": 2, "title": 3, "yes": 4, "no": 5, "word": 6}
_WH = {n: i for i, n in enumerate(T.WH_NAMES)}
_LOWER = re.compile(r"(?<![\w-])[a-z][a-z\-]{3,}(?![\w-])")


def make_xgb(**kw):
    from xgboost import XGBClassifier
    p = dict(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=5, tree_method="hist", random_state=42,
             n_jobs=4, eval_metric="logloss")
    p.update(kw)
    return XGBClassifier(**p)


def predict_proba(model, X) -> np.ndarray:
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    return model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1].astype(np.float32)


# ------------------------------------------------------------- knowledge


class KB:
    """Corpus view used by the components (normalised text, title mentions, spans)."""

    def __init__(self, titles, sents):
        self.titles = list(titles)
        self.sents = list(sents)
        self.text = [" " + T.norm_text(t + " " + " ".join(s)) + " "
                     for t, s in zip(self.titles, self.sents)]
        ments = [T.title_mention(t) for t in self.titles]
        self.pmention = [(" " + m + " ") if m else None for m in ments]
        self.mention = ments
        self.title_tok = [frozenset(T.content_words(m)) for m in ments]
        self.title_index: dict[str, int] = {}
        for i, m in enumerate(ments):
            if m and m not in self.title_index:
                self.title_index[m] = i
        self._sent_cache: dict[int, list] = {}

    def sent_info(self, d: int):
        """Per sentence of paragraph d: (content-token set, [(span, kind, window tokens)],
        [lower-case content words]). Window = content words within ~60 chars of the span."""
        r = self._sent_cache.get(d)
        if r is None:
            r = []
            for s in self.sents[d]:
                spans = []
                for span, kind in T.sentence_spans(s):
                    p = s.find(span)
                    win = (s[max(0, p - 60):p] + " " + s[p + len(span):p + len(span) + 60]
                           if p >= 0 else s)
                    spans.append((span, kind, frozenset(T.content_words(win))))
                low = list(dict.fromkeys(w for w in _LOWER.findall(s) if w not in T.STOP))
                r.append((frozenset(T.content_words(s)), spans, low))
            self._sent_cache[d] = r
        return r


class QInfo:
    """Static analysis of one question (no labels are used except for evaluation)."""

    def __init__(self, text, gold=(), answer=None, qtype=None, level=None, qid=None):
        text = text or ""
        self.text, self.gold, self.answer = text, tuple(gold), answer
        self.qtype, self.level, self.qid = qtype, level, qid
        self.words = T.content_words(text)
        self.wset = frozenset(self.words)
        self.entities = T.entity_spans(text)[:5]
        self.ent_norm = [n for n in (T.norm_text(e) for e in self.entities) if n]
        self.ent_keys = {T.normalize_answer(e) for e in self.entities}
        self.wh = T.wh_type(text)
        self.is_yesno = self.wh == _WH["yesno"]
        self.is_comp = T.is_comparison(text, self.entities)
        ew = set(T.content_words(" ".join(self.entities)))
        self.focus = list(dict.fromkeys(w for w in self.words if w not in ew))
        self.numbers = [w for w in dict.fromkeys(self.words) if T.YEAR_RE.match(w)]
        self.norm = " " + T.norm_text(text) + " "
        # expected answer type noun(s): content words right after what/which ("what city")
        ws = T.words(text)
        nouns = []
        for i, w in enumerate(ws):
            if w in ("what", "which"):
                nouns = [x for x in ws[i + 1:i + 4] if x not in T.STOP][:2]
                break
        self.wh_nouns = frozenset(nouns)
        spec =[("entity", e, n) for e, n in zip(self.entities, self.ent_norm)]
        spec += [("number", y, y) for y in self.numbers]
        topic = [w for w in self.focus if not w.isdigit()][:5]
        if topic:
            spec.append(("topic", " ".join(topic), " ".join(topic)))
        if not self.is_comp:
            spec.append(("bridge", "bridge entity linking two evidence paragraphs", ""))
        self.checklist_spec = spec
        self.row = -1


class QBank:
    """A list of QInfo plus their encoded query vectors."""

    def __init__(self, qinfos, retr):
        self.q = list(qinfos)
        for i, qi in enumerate(self.q):
            qi.row = i
        self.Qb, self.Qu, self.qd = retr.encode([qi.text for qi in self.q])


class QState:
    __slots__ = ("qi", "pool", "order", "issued", "issued_set", "snaps", "calls",
                 "retgen_used", "ev", "ev_probs", "check", "done")

    def __init__(self, qi):
        self.qi = qi
        # pool[d] = [first_iter, best_rank, best_fused, sub_bm25, n_hits, src, fused_sum]
        self.pool: dict[int, list] = {}
        self.order: list[int] = []
        self.issued: list = []
        self.issued_set: set = set()
        self.snaps: list[dict] = []
        self.calls = 0
        self.retgen_used: set = set()
        self.ev = np.zeros(0, dtype=np.int64)
        self.ev_probs = np.zeros(0, dtype=np.float32)
        self.check: list = []
        self.done = False


# ------------------------------------------------------ query generation


def decompose(qi: QInfo, cfg: Config, mode: str = "full"):
    """Rule-based decomposition: the full question plus entity-focused sub-queries."""
    subs = [("question", qi.text)]
    if mode == "full":
        for e in qi.entities[:3]:
            if T.norm_text(e) != qi.norm.strip():
                subs.append(("entity", e))
    out, seen = [], set()
    for src, q in subs:
        if q.strip() and q not in seen:
            seen.add(q)
            out.append((src, q))
    return out[: cfg.max_sub]


def bridge_candidates(qi: QInfo, ev_docs, kb: KB):
    """Entities mentioned in the top evidence paragraphs that are not in the question."""
    best: dict[str, tuple[float, str]] = {}
    for rank, d in enumerate(ev_docs):
        d = int(d)
        for si, (toks, spans, _) in enumerate(kb.sent_info(d)):
            ov = len(toks & qi.wset)
            for span, kind, _w in spans:
                if kind != "ent":
                    continue
                n = T.norm_text(span)
                if not n or n == kb.mention[d]:
                    continue
                cw = set(T.content_words(span))
                if not cw or cw <= qi.wset:
                    continue
                sc = 1.0 / (1 + rank) + 0.3 * ov - 0.05 * si + (1.0 if n in kb.title_index else 0.0)
                if n not in best or sc > best[n][0]:
                    best[n] = (sc, span)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [v[1] for _, v in ranked]


def refine_gap(s: QState, kb: KB, cfg: Config):
    """FAIR-RAG adaptive refinement: one targeted query per unconfirmed finding."""
    qi = s.qi
    unconf = [(k, t) for k, t, by in s.check if by < 0]
    bridge_open = any(k == "bridge" for k, _ in unconf)
    focus = " ".join(qi.focus[:3])
    bc = []
    if not qi.is_comp:
        bc = [c for c in bridge_candidates(qi, s.ev[:3], kb) if c not in s.issued_set]
    out = []
    if bridge_open:
        out += [("bridge", c) for c in bc[:2]]
    for k, t in unconf:
        if k in ("entity", "number"):
            out.append(("gap", (t + " " + focus).strip()))
        elif k == "topic":
            out.append(("gap", (t + " " + (qi.entities[0] if qi.entities else "")).strip()))
    if not bridge_open:
        out += [("bridge", c) for c in bc[:1]]
    res, seen = [], set()
    for src, q in out:
        if q and q not in s.issued_set and q not in seen:
            seen.add(q)
            res.append((src, q))
    if not res and qi.text.strip():
        res = [("gap", qi.text)]  # deeper search for the question itself (seen excluded)
    return res[: cfg.max_sub]


def refine_retgen(s: QState, kb: KB, cfg: Config):
    """Iter-Retgen stand-in: next query = question + text of the current top paragraph."""
    for d in s.ev:
        d = int(d)
        if d not in s.retgen_used:
            s.retgen_used.add(d)
            body = " ".join(kb.sents[d][:2])
            return [("retgen", f"{s.qi.text} {kb.titles[d]} {body}")]
    return []


# ---------------------------------------------------------- evidence filter


def _rank_desc(vals: np.ndarray, ids: np.ndarray) -> np.ndarray:
    r = np.empty(len(vals), dtype=np.float32)
    r[np.lexsort((ids, -vals))] = np.arange(len(vals), dtype=np.float32)
    return r


def pool_features(states, bank: QBank, retr, kb: KB, cfg: Config, it: int):
    """Feature matrix for every (question, pooled paragraph) pair of the given states."""
    qrows, dids = [], []
    for s in states:
        qrows += [s.qi.row] * len(s.order)
        dids += s.order
    bm, cos, cov = retr.pair_scores(bank.Qb, bank.Qu, bank.qd, qrows, dids)
    X = np.zeros((len(dids), len(FILTER_FEATURES)), dtype=np.float32)
    pos = 0
    for s in states:
        n = len(s.order)
        if n == 0:
            continue
        qi = s.qi
        sl = slice(pos, pos + n)
        ids = np.asarray(s.order, dtype=np.int64)
        b, c, cv = bm[sl], cos[sl], cov[sl]
        rel = b / max(float(b.max()), 1e-6)
        rb, rc = _rank_desc(b, ids), _rank_desc(c, ids)
        anchors = [int(j) for j in np.lexsort((ids, rb + rc))[: cfg.anchors]]
        rows = X[sl]
        for j, d in enumerate(s.order):
            e = s.pool[d]
            pm = kb.pmention[d]
            tt = kb.title_tok[d]
            in_link, n_in, out_link = 0.0, 0, 0.0
            txt = kb.text[d]
            for a in anchors:
                if a == j:
                    continue
                if pm is not None and pm in kb.text[ids[a]]:
                    n_in += 1
                    if rel[a] > in_link:
                        in_link = rel[a]
                pa = kb.pmention[ids[a]]
                if pa is not None and pa in txt and rel[a] > out_link:
                    out_link = rel[a]
            ent_cover = (sum(1 for en in qi.ent_norm if (" " + en + " ") in txt)
                         / max(1, len(qi.ent_norm)))
            rows[j] = (b[j], rel[j], rb[j], c[j], rc[j], cv[j], e[1], e[2], e[6], e[3],
                       e[4], e[0], e[5], len(tt & qi.wset) / max(1, len(tt)),
                       float(pm is not None and pm in qi.norm), ent_cover, in_link, n_in,
                       out_link, retr.doc_len[d], float(qi.is_comp), n, it)
        pos += n
    return X


# ------------------------------------------------ structured evidence assessment


def select_evidence(pool_ids, probs, key, cfg: Config, use_filter: bool, max_ev: int):
    """Keep the best paragraphs: by filter probability (>= threshold) or by retrieval score."""
    if len(pool_ids) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((pool_ids, -(probs if use_filter else key)))
    keep = []
    for i in order[:max_ev]:
        if not use_filter or probs[i] >= cfg.filter_threshold or len(keep) < cfg.min_evidence:
            keep.append(i)
        else:
            break
    return np.asarray(keep, dtype=np.int64)


def check_findings(qi: QInfo, ev, kb: KB):
    """SEA checklist audit: (kind, text, confirming paragraph id or -1) per required finding."""
    ev = [int(d) for d in ev]
    out = []
    for kind, txt, norm in qi.checklist_spec:
        by = -1
        if kind == "entity":
            pn = " " + norm + " "
            if qi.is_comp:  # each compared entity needs its own paragraph
                for d in ev:
                    m = kb.mention[d]
                    if m and (m == norm or (" " + m + " ") in pn or pn in (" " + m + " ")):
                        by = d
                        break
            else:
                for d in ev:
                    if pn in kb.text[d]:
                        by = d
                        break
        elif kind == "number":
            for d in ev:
                if " " + norm + " " in kb.text[d]:
                    by = d
                    break
        elif kind == "topic":
            ws = norm.split()
            best, bd = 0.0, -1
            for d in ev:
                f = sum(1 for w in ws if " " + w + " " in kb.text[d]) / max(1, len(ws))
                if f > best:
                    best, bd = f, d
            by = bd if best >= 0.6 else -1
        elif kind == "bridge":
            top = ev[:4]
            for a in top:
                for b in top:
                    if a != b and kb.pmention[b] is not None and kb.pmention[b] in kb.text[a]:
                        by = b
                        break
                if by >= 0:
                    break
        out.append((kind, txt, by))
    return out


def sea_features(qi: QInfo, ev, ev_probs, pool_ids, probs, ev_idx, check, kb, it, n_new,
                 n_pool, calls):
    p = np.sort(ev_probs)[::-1]
    p1 = float(p[0]) if len(p) > 0 else 0.0
    p2 = float(p[1]) if len(p) > 1 else 0.0
    p3 = float(p[2]) if len(p) > 2 else 0.0
    mask = np.ones(len(pool_ids), dtype=bool)
    mask[ev_idx] = False
    max_out = float(probs[mask].max()) if mask.any() else 0.0
    n_find = len(check)
    n_conf = sum(1 for _, _, by in check if by >= 0)
    ents = [by for k, _, by in check if k == "entity"]
    bridge = [by for k, _, by in check if k == "bridge"]
    link = 0.0
    top = [int(d) for d in ev[:4]]
    for i, a in enumerate(top):
        for j, b in enumerate(top):
            if i != j and kb.pmention[b] is not None and kb.pmention[b] in kb.text[a]:
                link = max(link, float(ev_probs[i] * ev_probs[j]))
    n_tm = sum(1 for d in ev if kb.pmention[int(d)] is not None and kb.pmention[int(d)] in qi.norm)
    return np.array([
        len(ev), p1, p2, p3, float(np.sum(ev_probs)), p1 * p2, max_out, n_find,
        n_conf / max(1, n_find), n_find - n_conf,
        (sum(1 for b in ents if b >= 0) / len(ents)) if ents else 1.0, len(ents),
        float(bool(bridge)), float(bool(bridge) and bridge[0] >= 0), link, n_tm,
        float(qi.is_comp), float(qi.is_yesno), it, n_new, n_pool, calls,
    ], dtype=np.float32)


def assess(s: QState, pool_ids, probs, kb, cfg, it, n_new, use_filter=True, max_ev=None):
    """Evidence selection + checklist + SEA feature vector for one question state."""
    max_ev = max_ev or cfg.max_evidence
    key = np.array([s.pool[int(d)][6] for d in pool_ids], dtype=np.float32)
    idx = select_evidence(pool_ids, probs, key, cfg, use_filter, max_ev)
    ev = pool_ids[idx] if len(idx) else np.zeros(0, dtype=np.int64)
    evp = probs[idx] if len(idx) else np.zeros(0, dtype=np.float32)
    check = check_findings(s.qi, ev, kb)
    x = sea_features(s.qi, ev, evp, pool_ids, probs, idx, check, kb, it, n_new,
                     len(pool_ids), s.calls)
    return ev, evp, check, x


# ------------------------------------------------------------- answerer


def _yes_no_overlap(qi: QInfo, ev, kb: KB) -> float:
    """Yes/no signal: Jaccard overlap of the first sentences of the top-2 evidence paragraphs
    (title and question-entity words removed), e.g. both say 'American'."""
    if len(ev) < 2:
        return 0.0
    ent = set(T.content_words(" ".join(qi.entities)))
    sets = []
    for d in ev[:2]:
        first = kb.sents[d][0] if kb.sents[d] else ""
        sets.append(set(T.content_words(first)) - kb.title_tok[d] - ent)
    a, b = sets
    return len(a & b) / max(1, len(a | b))


def answer_candidates(qi: QInfo, ev, evp, kb: KB, max_cands: int = 500):
    """Faithful candidate set: only spans that occur in the evidence (plus yes/no)."""
    ev = [int(d) for d in ev]
    evp = [float(p) for p in evp]
    # paragraph-level properties
    linked = {d: float(kb.pmention[d] is not None
                       and any(a != d and kb.pmention[d] in kb.text[a] for a in ev)) for d in ev}
    tinq = {d: float(kb.pmention[d] is not None and kb.pmention[d] in qi.norm) for d in ev}
    whn = qi.wh_nouns
    first_has = {d: float(bool(whn) and bool(kb.sent_info(d))
                          and bool(whn & kb.sent_info(d)[0][0])) for d in ev}
    C: dict[str, list] = {}

    def add(txt, kind, d, p, r, si, ov, win, wns):
        key = T.normalize_answer(txt)
        if not key:
            return
        c = C.get(key)
        if c is None:
            C[key] = [txt, CTYPES[kind], d, p, r, si, ov, 1, {d}, win, wns]
        else:
            if p > c[3]:
                c[2], c[3] = d, p
            c[4] = min(c[4], r)
            c[5] = min(c[5], si)
            c[6] = max(c[6], ov)
            c[7] += 1
            c[8].add(d)
            c[9] = max(c[9], win)
            c[10] = max(c[10], wns)

    if qi.is_yesno and ev:
        add("yes", "yes", ev[0], evp[0], 0, 0, 0, 0, 0.0)
        add("no", "no", ev[0], evp[0], 0, 0, 0, 0, 0.0)
    for r, (d, p) in enumerate(zip(ev, evp)):
        info = kb.sent_info(d)
        add(T.display_title(kb.titles[d]), "title", d, p, r, 0, len(kb.title_tok[d] & qi.wset),
            len(info[0][0] & qi.wset) if info else 0,
            float(bool(whn) and bool(info) and bool(whn & info[0][0])))
        best_si, best_ov = -1, 0
        for si, (toks, spans, _) in enumerate(info):
            ov = len(toks & qi.wset)
            wns = float(bool(whn & toks))
            if ov > best_ov:
                best_si, best_ov = si, ov
            for span, kind, wtoks in spans:
                add(span, kind, d, p, r, si, ov, len(wtoks & qi.wset), wns)
        if best_si >= 0 and not qi.is_yesno:  # common-noun answers from the best sentence
            toks = info[best_si][0]
            for w in info[best_si][2]:
                if w not in qi.wset:
                    add(w, "word", d, p, r, best_si, best_ov, best_ov, float(bool(whn & toks)))
    items = list(C.items())[:max_cands]
    ev_titles = {T.normalize_answer(T.display_title(kb.titles[d])) for d in ev}
    yn = _yes_no_overlap(qi, ev, kb) if qi.is_yesno else 0.0
    nq = max(1, len(qi.wset))
    X = np.zeros((len(items), len(ANSWER_FEATURES)), dtype=np.float32)
    for i, (key, c) in enumerate(items):
        txt, ct = c[0], c[1]
        cw = T.content_words(txt)
        in_q = sum(1 for w in cw if w in qi.wset) / max(1, len(cw))
        has_digit = any(ch.isdigit() for ch in key)
        is_year = bool(T.YEAR_RE.match(key))
        wh = qi.wh
        if wh == _WH["when"]:
            tm = ct == CTYPES["date"] or is_year
        elif wh in (_WH["how_many"], _WH["how_much"]):
            tm = ct == CTYPES["num"]
        elif wh == _WH["yesno"]:
            tm = ct in (CTYPES["yes"], CTYPES["no"])
        else:
            tm = ct in (CTYPES["ent"], CTYPES["title"])
        d = c[2]
        X[i] = (ct, wh, float(qi.is_comp), float(qi.is_yesno), in_q, float(key in qi.ent_keys),
                c[3], c[4], c[5], c[6], c[6] / nq, c[7], len(c[8]), len(key.split()),
                float(key in ev_titles), float(T.norm_text(txt) in kb.title_index),
                float(has_digit), float(is_year), float(tm), len(items), c[9],
                linked.get(d, 0.0), tinq.get(d, 0.0), c[10], first_has.get(d, 0.0), yn)
    texts = [c[0] for _, c in items]
    docs = [c[2] for _, c in items]
    return texts, docs, X


# ------------------------------------------------------------------ system


class FairRAG:
    """Holds the retriever, corpus view, config and the three learned components."""

    def __init__(self, retr, kb: KB, cfg: Config, models: dict | None = None):
        self.retr, self.kb, self.cfg = retr, kb, cfg
        self.models = models or {}

    # ---------------------------------------------------------- the loop
    def run(self, bank: QBank, max_iter=None, refine="gap", decompose_mode="full",
            use_filter=True, max_evidence=None, topk=None, record_X=False,
            stop_threshold=None, models=None):
        """Run the iterative loop on every question of `bank`.

        Without `stop_threshold` the loop runs a fixed number of iterations and records
        every state. Because refinement depends only on the current state (never on the
        SEA decision), stopping at the first iteration whose SEA probability passes a
        threshold yields exactly the recorded prefix - so stopping rules, iteration caps
        and the no-SEA ablation can all be read from one fixed run.
        """
        cfg, kb = self.cfg, self.kb
        models = models if models is not None else self.models
        fmodel, smodel = models.get("filter"), models.get("sea")
        use_filter = use_filter and fmodel is not None
        max_iter = max_iter or cfg.max_iter
        topk = topk or cfg.topk
        max_ev = max_evidence or cfg.max_evidence
        states = [QState(qi) for qi in bank.q]
        for it in range(1, max_iter + 1):
            active = [s for s in states if not s.done]
            if not active:
                break
            reqs = []
            for s in active:
                if it == 1:
                    qs = decompose(s.qi, cfg, decompose_mode)
                elif refine == "gap":
                    qs = refine_gap(s, kb, cfg)
                elif refine == "retgen":
                    qs = refine_retgen(s, kb, cfg)
                else:
                    qs = []
                for src, q in qs:
                    reqs.append((s, src, q))
                    s.issued.append((src, q))
                    s.issued_set.add(q)
                    s.calls += 1
            results = self.retr.search([q for _, _, q in reqs])
            new_count = {id(s): 0 for s in active}
            new_q = {id(s): [] for s in active}
            for (s, src, q), (ids, fused, bm, _) in zip(reqs, results):
                added = 0
                new_q[id(s)].append((src, q))
                for r in range(len(ids)):
                    if added >= topk:
                        break
                    d = int(ids[r])
                    e = s.pool.get(d)
                    if e is not None:
                        e[1] = min(e[1], r)
                        e[2] = max(e[2], float(fused[r]))
                        e[3] = max(e[3], float(bm[r]))
                        e[4] += 1
                        e[6] += float(fused[r])
                        continue
                    s.pool[d] = [it, r, float(fused[r]), float(bm[r]), 1, SRC[src],
                                 float(fused[r])]
                    s.order.append(d)
                    added += 1
                new_count[id(s)] += added
            X = pool_features(active, bank, self.retr, kb, cfg, it)
            probs_all = (predict_proba(fmodel, X) if fmodel is not None
                         else np.zeros(len(X), dtype=np.float32))
            pos = 0
            sea_rows = []
            for s in active:
                n = len(s.order)
                pids = np.asarray(s.order, dtype=np.int64)
                probs = probs_all[pos:pos + n]
                ev, evp, check, x = assess(s, pids, probs, kb, cfg, it, new_count[id(s)],
                                           use_filter, max_ev)
                s.ev, s.ev_probs, s.check = ev, evp, check
                snap = {"t": it, "queries": new_q[id(s)], "n_new": new_count[id(s)],
                        "pool_ids": pids, "probs": probs.copy(), "ev": ev, "ev_probs": evp,
                        "check": check, "sea_x": x, "sea_p": float("nan"), "calls": s.calls}
                if record_X:
                    snap["X"] = X[pos:pos + n]
                s.snaps.append(snap)
                sea_rows.append(x)
                pos += n
            if smodel is not None and sea_rows:
                sp = predict_proba(smodel, np.vstack(sea_rows))
                for s, p in zip(active, sp):
                    s.snaps[-1]["sea_p"] = float(p)
                    if stop_threshold is not None and p >= stop_threshold:
                        s.done = True
        return states

    # ------------------------------------------------------------ answer
    def answer(self, items, models=None):
        """items: list of (QInfo, evidence ids, evidence probs) -> list of answer dicts."""
        models = models if models is not None else self.models
        amodel = models.get("answer")
        blocks = [answer_candidates(qi, ev, evp, self.kb) for qi, ev, evp in items]
        allX = [b[2] for b in blocks if len(b[2])]
        probs = (predict_proba(amodel, np.vstack(allX)) if allX and amodel is not None
                 else np.zeros(sum(len(b[2]) for b in blocks), dtype=np.float32))
        out, pos = [], 0
        for (qi, ev, evp), (texts, docs, X) in zip(items, blocks):
            n = len(texts)
            if n == 0:
                out.append({"pred": "", "conf": 0.0, "cite": -1, "top": [], "abstain": True})
                continue
            p = probs[pos:pos + n]
            order = np.lexsort((np.arange(n), -p))
            b = int(order[0])
            out.append({"pred": texts[b], "conf": float(p[b]), "cite": int(docs[b]),
                        "top": [(texts[int(i)], float(p[int(i)]), int(docs[int(i)]))
                                for i in order[:3]],
                        "abstain": False})
            pos += n
        return out

    # ------------------------------------------------------ single query
    def ask(self, question: str, max_iter=None, threshold=None):
        """Run FAIR-RAG with SEA stopping on one free-text question; return a trace dict."""
        bank = QBank([QInfo(question)], self.retr)
        thr = self.cfg.sea_threshold if threshold is None else threshold
        st = self.run(bank, max_iter=max_iter or self.cfg.max_iter, stop_threshold=thr)[0]
        last = st.snaps[-1] if st.snaps else None
        ans = (self.answer([(st.qi, last["ev"], last["ev_probs"])])[0] if last is not None
               else {"pred": "", "conf": 0.0, "cite": -1, "top": [], "abstain": True})
        return trace_dict(st, len(st.snaps), self.kb, ans, thr)


def trace_dict(st: QState, stop_t: int, kb: KB, ans: dict, thr: float, max_items: int = 8):
    """JSON-friendly per-iteration trace (used by the dashboard)."""
    iters = []
    for sn in st.snaps[:stop_t]:
        pids, probs = sn["pool_ids"], sn["probs"]
        new = [int(d) for d in pids if st.pool[int(d)][0] == sn["t"]]
        new_sorted = sorted(new, key=lambda d: -float(probs[list(pids).index(d)]))[:max_items]
        pmap = {int(d): float(p) for d, p in zip(pids, probs)}
        iters.append({
            "t": sn["t"],
            "queries": [{"type": s, "text": q} for s, q in sn["queries"]],
            "retrieved": [{"title": kb.titles[d], "p": round(pmap[d], 4)} for d in new_sorted],
            "n_new": int(sn["n_new"]),
            "evidence": [{"title": kb.titles[int(d)], "p": round(float(p), 4)}
                         for d, p in zip(sn["ev"], sn["ev_probs"])],
            "checklist": [{"kind": k, "text": t, "confirmed": by >= 0,
                           "by": kb.titles[by] if by >= 0 else None}
                          for k, t, by in sn["check"]],
            "sea_p": None if np.isnan(sn["sea_p"]) else round(sn["sea_p"], 4),
            "sufficient": bool(not np.isnan(sn["sea_p"]) and sn["sea_p"] >= thr),
            "calls": int(sn["calls"]),
        })
    return {
        "question": st.qi.text, "iterations": iters, "n_iter": len(iters),
        "answer": ans["pred"], "confidence": round(float(ans["conf"]), 4),
        "citation": kb.titles[ans["cite"]] if ans["cite"] >= 0 else None,
        "top_candidates": [{"text": t, "p": round(p, 4), "doc": kb.titles[d]}
                           for t, p, d in ans.get("top", [])],
    }
