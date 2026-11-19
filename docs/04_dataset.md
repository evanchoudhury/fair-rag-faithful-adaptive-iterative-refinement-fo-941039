# 04 - Dataset

## Name and source
- **HotpotQA**, `distractor` configuration, **validation split only**.
- Paper: Yang et al., 2018, <https://arxiv.org/abs/1809.09600>; site: <https://hotpotqa.github.io/>
- Hugging Face: <https://huggingface.co/datasets/hotpotqa/hotpot_qa>
- File used: `distractor/validation-00000-of-00001.parquet` (27.5 MB as listed on the Hub).
  The full distractor config is ~613 MB because of the train split, so we load only this file to
  stay under the 200 MB limit.

## License
CC BY-SA 4.0 (as stated on the Hugging Face dataset card). Attribution is given in the README.

## Size
- 7,405 questions (dataset card count for the validation split).
- Each question has 10 context paragraphs (2 gold + 8 distractors retrieved by bigram TF-IDF in
  the original dataset construction), so up to ~74k paragraphs before deduplication; the exact
  number of unique titles is computed at run time and written to `artifacts/metrics.json`.

## Fields
| Field | Type | Use |
|---|---|---|
| `id` | string | question id, split key |
| `question` | string | input query |
| `answer` | string | gold answer (span or `yes`/`no`) |
| `type` | string | `bridge` or `comparison` - used for stratification and breakdowns |
| `level` | string | `easy` / `medium` / `hard` - breakdowns |
| `supporting_facts` | {title: list[str], sent_id: list[int]} | gold paragraph titles -> labels for filter, SEA and Joint Evidence Recall |
| `context` | {title: list[str], sentences: list[list[str]]} | paragraphs pooled into the retrieval corpus |

There are no "classes" in the usual sense; the prediction targets are (a) paragraph relevance
(binary), (b) evidence sufficiency (binary), (c) the answer span.

## Exact loading call
```python
from datasets import load_dataset

URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
       "distractor/validation-00000-of-00001.parquet")
ds = load_dataset("parquet", data_files={"validation": URL}, split="validation",
                  cache_dir=str(Path(__file__).parent / "data" / "hf_cache"))
```
A processed copy (`data/hotpot_val.pkl`) is cached after the first run.

## Preprocessing
1. Convert to a pandas DataFrame; gold titles = unique `supporting_facts.title`.
2. Corpus: for every question, every `(title, sentences)` pair; deduplicate by title (keep the
   first occurrence); paragraph text = `title + ". " + " ".join(sentences)`; keep sentence
   boundaries for answer extraction and citation.
3. Map each question's gold titles to corpus ids. Questions whose gold titles are not exactly 2
   distinct corpus paragraphs are counted and dropped (the count is logged).
4. Text normalisation for retrieval: lowercase, strip accents to ASCII, remove punctuation,
   English stop words removed for BM25/TF-IDF.
5. Answer normalisation for EM/F1: the standard SQuAD/HotpotQA rule (lowercase, remove punctuation
   and articles, collapse whitespace).

## Split plan
- Unit = question. Random 60 / 20 / 20 split into train / val / test, stratified by `type`,
  seed 42 (approx. 4,443 / 1,481 / 1,481; exact counts logged).
- The retrieval corpus is shared by all splits (it is unlabeled text, like Wikipedia in the paper).
- Learned components see train labels only; thresholds and K are tuned on val; test is used once
  for the final report.
- `--quick` mode subsamples questions (e.g. 1,500) for a fast smoke run; final numbers come from
  the full run.
