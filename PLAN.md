# PLAN - FAIR-RAG on a laptop

## Files to create (phases 2-4)
```
requirements.txt          # numpy pandas scipy scikit-learn lightgbm matplotlib plotly datasets streamlit joblib
README.md                 # what, how to run, results table filled from artifacts/metrics.json
train.py                  # end-to-end: load -> corpus -> retriever -> train filter/SEA/answerer -> evaluate -> plots
app.py                    # Streamlit dashboard
src/__init__.py
src/utils.py              # paths (Path(__file__).parent), ASCII-safe logging, seeding, timers
src/data.py               # load HotpotQA validation parquet, cache, build corpus, split
src/text.py               # tokenisation, stop words, entity span extraction, answer normalisation, EM/F1
src/retriever.py          # BM25 + LSA + RRF hybrid retriever (batch search, exclude-seen)
src/components.py         # Decomposer, EvidenceFilter, SEA (checklist + sufficiency), Refiner, FaithfulAnswerer
src/pipeline.py           # FairRAG loop with full per-iteration trace; config flags for ablations
src/baselines.py          # StandardRAG (top-5, matched budget), IterRetgenLoop
src/evaluate.py           # metrics, bootstrap CIs, breakdowns, failure taxonomy, iteration sweep
src/plots.py              # matplotlib PNGs + plotly JSON for the dashboard
tests/smoke_test.py       # python tests/smoke_test.py -> runs train.py --quick end-to-end checks
artifacts/                # metrics.json, models (*.joblib), traces (test_traces.jsonl), plots/
data/                     # cached download (git-ignored)
```

## Dataset loading call
```python
load_dataset("parquet",
    data_files={"validation": "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"},
    split="validation", cache_dir=<project>/data/hf_cache)
```
7,405 questions -> pooled, title-deduplicated paragraph corpus; 60/20/20 split by question
(stratified by type, seed 42).

## Models
- Hybrid retriever: BM25 + LSA(256) fused with RRF.
- Evidence filter: LightGBM classifier.
- SEA: rule-based checklist + LightGBM sufficiency classifier (threshold tuned on val).
- Refiner: gap and bridge-entity queries (rules).
- Faithful answerer: LightGBM span scorer over evidence-only candidates, with citation.

## Primary metric and baseline
- Primary: **Joint Evidence Recall** on test (both gold supporting paragraphs in final evidence).
- Baseline to beat: **Standard RAG at matched passage budget**; also Standard RAG top-5,
  Iter-Retgen-style fixed loop, and ablations (no SEA stop, no gap refinement, no filter).
- Secondary: answer EM/F1, citation precision, passages used, retrieval calls; paired bootstrap CIs.

## Plots (artifacts/plots/)
1. System comparison bar chart: Joint Evidence Recall and F1 with 95% CI.
2. Cost vs quality scatter: mean passages / retrieval calls vs Joint Evidence Recall per system.
3. Iteration sweep (max iter 1-4): recall, F1, calls.
4. Single-shot Recall@k curve (BM25 vs LSA vs hybrid).
5. Evidence filter ROC and precision-recall curves.
6. SEA sufficiency calibration curve + confusion matrix.
7. Breakdown by question type and level (grouped bars).
8. Failure taxonomy stacked bar (FAIR-RAG vs Standard RAG).
9. Feature importance for filter, SEA, answerer.
10. Distribution of iterations used per question.

## Dashboard pages (app.py, Streamlit)
1. **Overview** - problem, pipeline diagram, headline numbers read from metrics.json.
2. **Live explorer** - pick a test question or type your own; step through each iteration:
   sub-queries, retrieved paragraphs with filter scores, SEA checklist (confirmed / gaps),
   refined queries, final cited answer vs gold.
3. **Compare systems** - table + charts for all baselines and ablations, bootstrap CIs.
4. **Components** - filter, SEA and answerer diagnostics, feature importances, iteration sweep.
5. **Error analysis** - failure taxonomy, browse failed questions with their traces.
6. **Dataset** - corpus stats, split sizes, sample questions by type/level.
7. **About** - paper summary, literature table, limitations, how to reproduce.

## Constraints checklist
- Only `python ...` commands; no installs; CPU; full `python train.py` < 15 min (timed).
- ASCII-only console output; paths via Path(__file__).parent; no absolute paths or secrets.
- Every number in README/dashboard comes from artifacts/metrics.json.
