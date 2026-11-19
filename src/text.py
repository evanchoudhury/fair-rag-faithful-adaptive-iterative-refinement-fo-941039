"""Text utilities: tokenisation, entity/number spans, question analysis, EM/F1."""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = frozenset(ENGLISH_STOP_WORDS) | {"s", "t"}
_TOK = re.compile(r"[a-z0-9]+")


def to_ascii(s: str) -> str:
    if s.isascii():
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def words(s: str) -> list[str]:
    """Lower-cased ASCII alphanumeric tokens (stop words kept)."""
    return _TOK.findall(to_ascii(s).lower())


def content_words(s: str) -> list[str]:
    return [w for w in words(s) if w not in STOP and (len(w) > 1 or w.isdigit())]


def analyzer(s: str) -> list[str]:
    """Retrieval analyzer: content unigrams + adjacent content bigrams."""
    toks = content_words(s)
    return toks + [a + " " + b for a, b in zip(toks, toks[1:])]


def norm_text(s: str) -> str:
    """Lower-case ASCII, punctuation removed, single spaces (for substring matching)."""
    return " ".join(words(s))


_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def title_mention(title: str) -> str:
    """How a Wikipedia title is usually mentioned in text: drop '(film)' etc."""
    m = norm_text(_PAREN.sub("", title))
    return m if len(m) >= 3 else ""


def display_title(title: str) -> str:
    return _PAREN.sub("", title).strip() or title


# ----------------------------------------------------------------- spans
_CAP = r"[A-ZÀ-Þ][\w'’\.\-&]*"
# "and" is deliberately not a connector: "X and Y" usually names two entities.
_CONN = r"(?:of|the|de|del|della|la|le|von|van|der|du|da|for|in|on|y|&)"
ENT_RE = re.compile(rf"{_CAP}(?:\s+(?:{_CONN}\s+)*{_CAP})*")
QUOTE_RE = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
           "November|December")
DATE_RE = re.compile(
    rf"\b(?:(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}|"
    rf"(?:{_MONTHS})\s+\d{{4}}|\d{{4}})\b")
NUM_RE = re.compile(
    r"\b\d[\d,\.]*\d?(?:\s+(?:million|billion|thousand|hundred|percent))?\b")
YEAR_RE = re.compile(r"^\d{4}$")

_LEAD = {
    "which", "what", "who", "whom", "whose", "where", "when", "why", "how", "the", "a", "an",
    "is", "are", "was", "were", "did", "do", "does", "in", "on", "at", "of", "this", "that",
    "these", "those", "both", "he", "she", "it", "they", "his", "her", "its", "their", "there",
    "after", "before", "as", "by", "for", "from", "with", "and", "or", "if", "although",
    "during", "between", "since", "while", "however", "also", "then", "many", "name", "can",
    "could", "has", "have", "had", "will", "would", "according", "following", "other", "i",
    "we", "you", "one", "all", "some", "most", "more", "than", "not", "no", "yes", "after",
    "until", "about", "into", "upon", "when", "out", "over", "under", "to",
}
_CONN_SET = {"of", "the", "de", "del", "della", "la", "le", "von", "van", "der", "du",
             "da", "for", "in", "on", "y", "&"}


def _clean_span(span: str) -> str:
    span = span.strip(" .,'’-&;:")
    toks = span.split()
    while toks and toks[0].lower().strip(".,'") in _LEAD:
        toks = toks[1:]
    while toks and toks[-1].lower() in _CONN_SET:
        toks = toks[:-1]
    span = " ".join(toks).strip(" .,'’-&;:")
    if span.endswith("'s") or span.endswith("’s"):
        span = span[:-2]
    if not span or not content_words(span):
        return ""
    return span


def entity_spans(s: str) -> list[str]:
    """Capitalised multi-word spans and quoted strings, cleaned, in order, de-duplicated."""
    out, seen = [], set()
    for m in QUOTE_RE.finditer(s):
        c = _clean_span(m.group(1))
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    for m in ENT_RE.finditer(s):
        c = _clean_span(m.group(0))
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def sentence_spans(s: str) -> list[tuple[str, str]]:
    """Answer-candidate spans of one sentence: (span, kind) with kind in ent/date/num."""
    out, seen = [], set()
    for m in DATE_RE.finditer(s):
        t = m.group(0)
        if t not in seen:
            seen.add(t)
            out.append((t, "date"))
    for m in NUM_RE.finditer(s):
        t = m.group(0).rstrip(".,")
        if t and t not in seen:
            seen.add(t)
            out.append((t, "num"))
    for t in entity_spans(s):
        if t not in seen:
            seen.add(t)
            out.append((t, "ent"))
    return out


# ------------------------------------------------------------- questions
AUX = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had",
       "will", "would", "should"}
WH_NAMES = ["other", "how_many", "how_much", "when", "who", "where", "which", "what",
            "yesno", "how"]


def wh_type(q: str) -> int:
    w = words(q)
    ql = " ".join(w)
    if w and w[0] in AUX:
        return WH_NAMES.index("yesno")
    if "how many" in ql:
        return WH_NAMES.index("how_many")
    if "how much" in ql:
        return WH_NAMES.index("how_much")
    if "when" in w or "what year" in ql:
        return WH_NAMES.index("when")
    for key in ("who", "whom", "whose"):
        if key in w:
            return WH_NAMES.index("who")
    for key in ("where", "which", "what", "how"):
        if key in w:
            return WH_NAMES.index(key)
    return 0


def is_comparison(q: str, ents: list[str]) -> bool:
    if len(ents) < 2:
        return False
    w = set(words(q))
    ql = " " + q.lower() + " "
    return " or " in ql or bool(w & {"both", "same", "common", "either"})


# ---------------------------------------------------------------- answers
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") and p != g:
        return 0.0
    if g in ("yes", "no", "noanswer") and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))
