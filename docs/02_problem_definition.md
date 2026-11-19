# 02 - Problem Definition

## Problem statement
Given a multi-hop question and a large pool of Wikipedia paragraphs, collect a **small set of
evidence paragraphs that contains every fact the question needs**, then produce an answer that is
taken from - and cites - that evidence. Single-shot retrieval often finds the first hop (the
paragraph that shares words with the question) but misses the second hop (a "bridge" paragraph
whose title is only mentioned inside the first one). The system must notice this gap and fill it
with targeted follow-up queries, and stop as soon as the evidence is sufficient.

## Motivation
- In RAG, if the needed evidence is never retrieved, no reader can answer faithfully; it either
  guesses (hallucination) or fails. Evidence completeness is the upper bound on faithful accuracy.
- Iterative methods (IRCoT, Iter-Retgen, FLARE) retrieve more, but use fixed loops or uncertainty
  triggers. FAIR-RAG claims that an *explicit* gap check (SEA) plus gap-targeted queries is what
  makes the loop work - but never ablated it (see docs/01). We test that claim in a controlled,
  reproducible, CPU-only setting.

## Objectives (measurable)
1. Build an open-retrieval corpus from HotpotQA (distractor, validation file) and a hybrid
   BM25 + LSA retriever with Reciprocal Rank Fusion; report single-shot Recall@k for k = 2, 5, 10, 20.
2. Train an evidence filter (paragraph relevance) and report its ROC-AUC and F1 on the test split.
3. Train a Structured Evidence Assessment (SEA) sufficiency classifier and report its accuracy,
   ROC-AUC and the rate of false "sufficient" decisions on the test split.
4. Run the full FAIR-RAG loop and report **Joint Evidence Recall** (primary), evidence precision,
   average passages used and average retrieval calls per question.
5. Compare against: Standard RAG (top-5), Standard RAG at matched passage budget, an
   Iter-Retgen-style fixed loop, and two ablations (no SEA stopping; no gap-aware refinement).
6. Report answer EM / F1 and citation precision of the faithful extractive answerer for every system.
7. Sweep the maximum number of iterations (1-4) and report recall and cost per setting.
8. Break results down by question type (bridge / comparison) and difficulty level, and produce an
   automatic failure taxonomy (retrieval miss, filter drop, early SEA stop, answer error).

## Scope
- English, Wikipedia-paragraph, multi-hop QA (HotpotQA).
- Closed corpus: all paragraphs appearing in the 7,405 validation questions (gold + distractors
  pooled across questions), searched in open-retrieval mode (the per-question distractor grouping
  is not given to the system).
- Lightweight learned components (gradient boosting, logistic regression) plus transparent rules,
  trained in under 15 minutes on CPU.

## Non-goals
- No LLM generation, no fine-tuning of neural language models, no GPU.
- No full-Wikipedia index (5M+ articles) - the pooled corpus is a laptop-scale stand-in.
- Not trying to match the paper's absolute F1 numbers (different reader, different corpus); the
  goal is the *relative* effect of the gap-aware loop.
- No adaptive model routing (the paper also disabled it in its benchmarks).

## Success criteria
- **Primary metric:** *Joint Evidence Recall* on the held-out test split = fraction of questions for
  which **both** gold supporting paragraphs (HotpotQA `supporting_facts` titles) are in the final
  evidence set handed to the answerer.
- **Baseline to beat:** *Standard RAG at matched budget* - single-shot hybrid retrieval of the
  question, keeping the top-K paragraphs where K equals FAIR-RAG's average final evidence size
  (rounded up). Also reported: Standard RAG top-5 (the paper's setting).
- The project is a success if FAIR-RAG's Joint Evidence Recall is higher than the matched-budget
  baseline and the paired-bootstrap 95% confidence interval of the difference excludes zero.
- Secondary (reported honestly whatever the outcome): answer F1 vs Standard RAG, and whether SEA
  stopping beats the fixed loop at lower cost.
