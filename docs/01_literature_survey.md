# 01 - Literature Survey

All entries below were looked up on arXiv (abstract page and, where noted, the HTML full text)
during the design phase. Numbers in the "Key result" column are quoted from those pages. Where a
page gave only qualitative claims, the column says so instead of inventing a figure.

## 1. Related work

| Paper | Year | Method | Dataset | Key result | Limitation |
|---|---|---|---|---|---|
| [Lewis et al. - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 2020 | Seq2seq generator + dense passage retriever over Wikipedia; RAG-Sequence and RAG-Token variants | Open-domain QA and other knowledge-intensive tasks | Abstract reports state of the art on three open-domain QA tasks (no numbers in abstract) | Single-shot retrieval: one retrieval for the whole question, so a missing hop is never recovered |
| [Yang et al. - HotpotQA](https://arxiv.org/abs/1809.09600) | 2018 | Dataset: Wikipedia questions requiring reasoning over multiple documents, with sentence-level supporting facts and comparison questions | HotpotQA (about 113k QA pairs) | Abstract: dataset is challenging for the latest QA systems | Some questions can be solved by shortcuts within one paragraph; distractors were chosen by bigram TF-IDF |
| [Trivedi et al. - IRCoT](https://arxiv.org/abs/2212.10509) | 2022 (ACL 2023) | Interleaves BM25 retrieval with each chain-of-thought step; each reasoning sentence becomes the next query | HotpotQA, 2WikiMultihopQA, MuSiQue, IIRC | HotpotQA QA F1: 59.1 vs 49.7 for one-step retrieval (Flan-T5-XXL); 60.7 vs 53.6 (GPT3). Retrieval recall +7.9 / +11.3 points | One retrieval per reasoning step: many LLM calls; no explicit test of whether the evidence is already sufficient |
| [Yao et al. - ReAct](https://arxiv.org/abs/2210.03629) | 2022 (ICLR 2023) | Prompted interleaving of reasoning traces and actions (e.g. Wikipedia search/lookup) | HotpotQA, FEVER, ALFWorld, WebShop | +34% and +10% absolute success over imitation/RL on ALFWorld and WebShop (abstract) | Free-form action loop; small models often fail to follow the format (FAIR-RAG reports ReAct EM .000 on all four datasets with Llama-3-8B) |
| [Shao et al. - Iter-RetGen](https://arxiv.org/abs/2305.15294) | 2023 | Alternate retrieval and generation; the previous full generation is used as the next retrieval query | HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle, Feverous, StrategyQA | HotpotQA EM/F1 39.2/53.9 at iteration 1 and 44.1/58.6 at iteration 2; largest gain at the 2nd iteration, then plateau | Fixed number of iterations; query = whole generation, so noise from a wrong draft is carried into retrieval |
| [Jiang et al. - FLARE (Active RAG)](https://arxiv.org/abs/2305.06983) | 2023 (EMNLP) | Predict the next sentence; if it has low-confidence tokens, use it as a query and regenerate | 2WikiMultihopQA, StrategyQA, ASQA, WikiAsp | 2WikiMultihopQA EM/F1 51.0/59.7 vs 39.4/48.8 for single-time retrieval | Needs token probabilities; triggers on model uncertainty, not on what evidence is actually missing |
| [Asai et al. - Self-RAG](https://arxiv.org/abs/2310.11511) | 2023 | Fine-tuned LM emits reflection tokens to decide when to retrieve and to critique passages and its own output | PopQA, TriviaQA, PubHealth, ARC-Challenge, Biography, ASQA | Accuracy 7B/13B: PopQA 54.9/55.8, TriviaQA 66.4/69.3, PubHealth 72.4/74.5, ARC-C 67.3/73.1 | Requires fine-tuning a dedicated model; critique is per passage, not a structured checklist of what the question needs |
| [Jeong et al. - Adaptive-RAG](https://arxiv.org/abs/2403.14403) | 2024 (NAACL) | Small classifier predicts query complexity and routes to no-retrieval, single-step or multi-step RAG | SQuAD, NQ, TriviaQA, MuSiQue, HotpotQA, 2WikiMultiHopQA | Averaged F1 46.94 vs 48.85 (multi-step) with FLAN-T5-XL at 3.60 vs 8.81 relative time | Decision is made once, before retrieval; cannot react to what the first retrieval actually found |
| [Kim et al. - SuRe](https://arxiv.org/abs/2404.13081) | 2024 (ICLR) | Generate answer candidates, summarise retrieved passages per candidate, verify and rank the summaries | Several open-domain QA benchmarks (abstract does not list them) | Up to +4.6% EM and +4.0% F1 over standard prompting (abstract) | Single retrieval round; improves reading, not evidence collection |
| [Su et al. - DRAGIN](https://arxiv.org/abs/2403.10081) | 2024 | Decides when to retrieve and what query to use from the LLM's real-time information needs (attention/uncertainty) | 4 knowledge-intensive generation datasets (abstract) | Abstract claims superior performance on all tasks (no numbers in abstract) | Needs access to model internals; trigger is still uncertainty-based rather than evidence-gap-based |

## 2. Critical review of the source paper

**Paper:** [FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation](http://arxiv.org/abs/2510.22344v1)
(Aghajani Asl, Asgari-Bidhendi, Minaei-Bidgoli, arXiv 2510.22344, Oct 2025). Read from the abstract
page and the arXiv HTML full text (first ~100k characters of 175k; the tail of the appendix was not read).

### What it does
An agentic loop built from prompted LLM modules (no fine-tuning):
1. Optional adaptive routing (OBVIOUS / SMALL / LARGE / REASONING) - disabled in the benchmarks.
2. Query decomposition into up to 4 keyword-rich sub-queries.
3. Retrieval - described as hybrid dense+sparse with Reciprocal Rank Fusion, but the benchmarks
   used dense-only e5-base-v2 + Faiss over DPR Wikipedia, top-5 per query.
4. LLM evidence filtering (drop irrelevant passages).
5. **Structured Evidence Assessment (SEA):** turn the question into a checklist of required findings,
   audit the evidence, output confirmed findings and explicit gaps; sufficient only if all findings are confirmed.
6. **Adaptive query refinement:** write new targeted queries for the gaps only.
7. Repeat up to 3 iterations, then **faithful generation** using only the evidence, with [n] citations.

### Reported results (Table 2, 1,000 samples per dataset, Llama-3-8B-Instruct)
- HotpotQA F1: Standard RAG .342, Iter-Retgen .370, FAIR-RAG 3 .447, FAIR-RAG 3 (Adpt.) .453.
- 2WikiMultiHopQA F1: Self-RAG .251, FAIR-RAG 2 .325, FAIR-RAG 3 (Adpt.) .320.
- MuSiQue F1: Iter-Retgen .190, FAIR-RAG 3 .267.
- Component quality (Table 3, HotpotQA): evidence filter F1 67.3%, SEA accuracy 72.0%.
- Iteration study (Table 4): multi-hop datasets peak at 2-3 iterations and get worse at 4;
  on TriviaQA (single-hop) extra iterations hurt.
- Failure analysis of 200 errors: retrieval 32.5%, generation 31.0%, SEA 24.5%, decomposition 9.0%.

### Strengths
- Clear, modular design; every module is inspectable, which makes errors attributable.
- The gap-driven query (only the missing part) is a sensible fix for Iter-Retgen's noisy whole-draft query.
- Consistent gains over seven baselines on four datasets under a shared retriever.
- Honest component-level evaluation and an error taxonomy, plus a human check of the LLM judge (90% agreement).

### Weaknesses
- **No fixed-loop ablation.** The authors explicitly did not run "same loop without SEA / without
  gap-aware refinement", so it is not shown whether gains come from SEA or simply from more retrieval rounds.
- **Budget is not matched.** FAIR-RAG sees more passages (several sub-queries x top-5 x iterations)
  than Standard RAG (top-5); part of the gain may be recall from a larger budget.
- The "Adpt." configuration uses 70B and DeepSeek-R1 models while baselines use 8B.
- Hybrid retrieval and routing are described as core but were switched off in the experiments.
- SEA accuracy is only 72% on HotpotQA, and SEA is responsible for about a quarter of errors.
- Only 1,000 samples per dataset; no confidence intervals or significance tests reported.
- The iteration cap of 3 is a heuristic (acknowledged by the authors).

### What can be reproduced on a CPU laptop
Running Llama-3-8B for ~8 calls per question is not possible in 15 minutes on CPU, and no LLM
library is available. What *can* be reproduced faithfully is the **control logic** that the paper
claims is the source of its gains:
- hybrid sparse+dense retrieval with RRF (as described in the method section),
- query decomposition,
- an evidence filter,
- a structured sufficiency check with explicit confirmed findings and gaps,
- gap-targeted query refinement with early stopping,
- an answer step that may only use and must cite the gathered evidence.
LLM prompts are replaced by small, trained, inspectable models (LightGBM / logistic regression)
and rules, trained on HotpotQA supporting-fact labels. The retrieval side (evidence recall) can be
measured exactly, because HotpotQA labels which paragraphs are needed.

## 3. Gap this project addresses
1. **The missing ablation.** We run the same pipeline with (a) SEA stopping removed (fixed loop) and
   (b) gap-aware refinement replaced by Iter-Retgen-style whole-text re-querying, to measure what the
   evidence-gap mechanism itself contributes.
2. **Budget-matched comparison.** Standard RAG is also evaluated at the same average number of
   passages that FAIR-RAG ends up using, so "more passages" cannot explain the gain.
3. **Evidence-level evaluation.** Instead of only answer F1 (which mixes retrieval and reading),
   the primary metric is whether *all* required supporting paragraphs were collected - the exact
   thing SEA is meant to guarantee - with paired bootstrap confidence intervals.
4. **Laptop-scale, fully transparent version** of the loop that anyone can train in minutes and
   inspect step by step in a dashboard.

Source not read in full: the last ~75k characters (appendix) of the FAIR-RAG HTML version. A follow-up
paper, FARSIQA (arXiv 2510.25621), appeared in search results but was not read and is not cited above.
