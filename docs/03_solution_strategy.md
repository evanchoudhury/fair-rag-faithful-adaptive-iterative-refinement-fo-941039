# 03 - Solution Strategy

## Approach
Keep FAIR-RAG's **architecture and control flow** exactly (decompose -> retrieve -> filter ->
SEA -> refine -> loop -> faithful answer), but implement each LLM-prompted agent as a small,
trained, inspectable model that runs on CPU. HotpotQA gives us supervision for every module:
`supporting_facts` tells us which paragraphs are needed (filter and SEA labels) and `answer`
gives the target span (answer ranker labels). Because every module is cheap, we can afford the
ablations the paper skipped and evaluate on ~1,500 test questions with confidence intervals.

Why this approach:
- The paper's central claim is about *control* (when to continue, what to ask next), not about the
  LLM itself. A faithful CPU replica of the control is the most informative thing a laptop can test.
- Supervised small models give calibrated probabilities, which SEA needs for a stopping threshold.
- Extractive answering is faithful by construction: the answer is a span of a cited evidence paragraph.

## Alternatives considered
| Alternative | Why not chosen |
|---|---|
| Run a small LLM (e.g. 0.5-1B) for every agent | No `transformers` library allowed; even with it, ~8 calls x 1,500 questions exceeds 15 min on CPU |
| Train a neural reader (BiDAF-style) in torch | Feasible but slow to train well on CPU in the budget; reading quality is not the claim under test. Kept as future work |
| Dense retriever from pretrained embeddings (e5) | Pretrained weights cannot be downloaded through allowed libraries; LSA (TF-IDF + SVD) gives a learned dense space instead |
| Use the per-question 10-paragraph distractor context directly | Too easy (only 10 candidates) and it hides the retrieval problem; we pool all paragraphs into one corpus |
| 2WikiMultiHopQA / MuSiQue | No standard small HF copy confirmed; HotpotQA is the paper's headline dataset and has a 27.5 MB validation parquet |

## Pipeline (step by step)
1. **Load** HotpotQA distractor validation (7,405 questions) from the Hugging Face parquet file.
2. **Corpus:** pool every (title, sentences) paragraph from all questions, deduplicate by title,
   text = title + sentences.
3. **Split** questions 60/20/20 into train/val/test (seed 42, stratified by `type`).
4. **Retriever:** BM25 (sparse, own implementation on a scipy sparse matrix) and LSA dense vectors
   (TF-IDF -> TruncatedSVD, 256 dims, L2-normalised); fused with Reciprocal Rank Fusion (k = 60).
5. **Decomposer (rules):** comparison patterns ("X or Y", "both X and Y", "Which ... first, X or Y")
   produce one sub-query per entity; every question also keeps the full question as a sub-query;
   capitalised entity spans can produce entity-focused sub-queries. Max 4 sub-queries.
6. **Evidence filter (LightGBM):** features of (question, paragraph): BM25 score, dense cosine,
   fused rank, question-term coverage, entity overlap, title-in-question, title-mentioned-by-other-
   evidence (bridge link), retrieved-at-iteration, paragraph length. Label: paragraph is gold.
7. **SEA:** build the *required findings* checklist from the question (entities, quoted titles,
   numbers, content-word groups, and for bridge-like questions an implicit "bridge entity" slot);
   mark each finding confirmed/unconfirmed against kept evidence; a LightGBM *sufficiency* model on
   evidence-state features (coverage, top-2 filter probabilities, bridge-link found, count kept,
   iteration) predicts P(both gold paragraphs present). Sufficient if P >= threshold (tuned on val).
8. **Refiner:** for each gap, build a targeted query: unconfirmed finding terms + question focus
   words; for the bridge slot, take candidate entities mentioned in confirmed paragraphs but not in
   the question (title-like capitalised spans) and query them. Already-seen paragraphs are excluded.
9. **Loop** steps 6-8 until SEA says sufficient or max iterations (default 3) reached.
10. **Faithful answerer:** generate candidate spans only from kept evidence sentences (entities,
    numbers/dates, yes/no for yes-no questions); a LightGBM ranker scores them; output the answer
    with a citation to its source paragraph, or "insufficient evidence" if no candidate qualifies.
11. **Evaluate** all systems on the test split; write `artifacts/metrics.json` and plots.
12. **Dashboard** (Streamlit) reads only artifacts plus the saved models for live queries.

## Training order (no test leakage)
1. Fit retriever on the corpus (unsupervised; the corpus contains no labels).
2. Run iteration-1 retrieval on **train** questions -> train evidence filter.
3. Run the loop with SEA disabled (fixed 4 iterations) on **train** questions, recording each
   evidence state -> train SEA sufficiency model.
4. Collect candidates from final train evidence -> train answer ranker.
5. Tune filter threshold, SEA threshold and K values on **val**. Report once on **test**.

## Risks and handling
| Risk | Handling |
|---|---|
| Extractive answerer is weak, so answer F1 is low | Primary metric is evidence recall, which does not depend on the reader; all systems share the same answerer so comparisons stay fair; report F1 honestly |
| FAIR-RAG wins only because it reads more paragraphs | Matched-budget Standard RAG baseline; report passages used per question for every system |
| Corpus built from the same questions leaks answers | Corpus contains only paragraph text (no questions/answers/labels); learned models trained on train questions only |
| SEA threshold overfits | Tuned on val only; SEA accuracy and false-sufficient rate reported on test |
| Runtime above 15 min | Vectorised sparse retrieval in batches, cap candidates per query (top-10), LightGBM with small trees; a `--quick` flag for a small subsample; timing logged in metrics.json |
| Non-ASCII text crashes the Windows console | Console logging passes strings through an ASCII-safe helper; files written as UTF-8 |
| Dataset download fails | Clear error message; loader tries the `datasets` parquet path and caches under `data/` |
