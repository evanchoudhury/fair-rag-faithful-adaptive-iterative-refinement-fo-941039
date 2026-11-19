"""Load HotpotQA (distractor, validation file), build the pooled corpus, split questions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import DATA_DIR, SEED, log

URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
       "distractor/validation-00000-of-00001.parquet")
CACHE_NAME = "hotpot_val.pkl"
COLUMNS = ["id", "question", "answer", "type", "level", "sf_titles", "ctx_titles", "ctx_sents"]


def _as_list(x):
    return [y for y in x] if x is not None else []


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return one row per question. Downloads once, then reuses data/hotpot_val.pkl."""
    data_dir = Path(data_dir)
    cache = data_dir / CACHE_NAME
    if cache.exists():
        log(f"Loading cached dataset {cache.name}")
        return pd.read_pickle(cache)
    data_dir.mkdir(parents=True, exist_ok=True)
    log("Downloading HotpotQA distractor validation parquet (first run only) ...")
    from datasets import load_dataset
    try:
        ds = load_dataset("parquet", data_files={"validation": URL}, split="validation",
                          cache_dir=str(data_dir / "hf_cache"))
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Could not download HotpotQA. Check the internet connection and retry. "
            f"Original error: {type(exc).__name__}: {exc}") from exc
    rows = []
    for ex in ds:
        sf = ex["supporting_facts"]
        ctx = ex["context"]
        sf_titles = list(dict.fromkeys(_as_list(sf["title"])))
        rows.append({
            "id": ex["id"], "question": ex["question"], "answer": ex["answer"],
            "type": ex["type"], "level": ex["level"], "sf_titles": sf_titles,
            "ctx_titles": _as_list(ctx["title"]),
            "ctx_sents": [[s for s in sents] for sents in ctx["sentences"]],
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_pickle(cache)
    log(f"Cached {len(df)} questions to data/{CACHE_NAME}")
    return df


def build_corpus(df: pd.DataFrame):
    """Pool every context paragraph, de-duplicate by title (first occurrence wins).

    Returns (corpus dict, df with gold_ids column, n_dropped). Questions whose supporting
    titles are not exactly two distinct corpus paragraphs are dropped.
    """
    title2id: dict[str, int] = {}
    titles, sents = [], []
    for ts, ss in zip(df["ctx_titles"], df["ctx_sents"]):
        for t, s in zip(ts, ss):
            if t in title2id:
                continue
            title2id[t] = len(titles)
            titles.append(t)
            sents.append([x.strip() for x in s if x and x.strip()])
    gold = []
    for sf in df["sf_titles"]:
        ids = [title2id[t] for t in sf if t in title2id]
        gold.append(tuple(sorted(set(ids))))
    df = df.copy()
    df["gold_ids"] = gold
    keep = df["gold_ids"].map(len) == 2
    n_drop = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
    corpus = {"titles": titles, "sents": sents, "texts": texts, "title2id": title2id}
    return corpus, df, n_drop


def split(df: pd.DataFrame, seed: int = SEED):
    """60/20/20 split of questions, stratified by question type."""
    idx = np.arange(len(df))
    strat = df["type"].values if df["type"].nunique() > 1 else None
    tr, rest = train_test_split(idx, test_size=0.4, random_state=seed, stratify=strat)
    strat_rest = df["type"].values[rest] if strat is not None else None
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=strat_rest)
    return np.sort(tr), np.sort(va), np.sort(te)
