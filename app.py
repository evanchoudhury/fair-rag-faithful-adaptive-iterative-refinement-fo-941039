"""FAIR-RAG dashboard.   Run:  streamlit run app.py

Everything shown here is read from artifacts/ (written by `python train.py`) and docs/.
The "Try it" tab runs the real FAIR-RAG loop live, using a compact slice of the trained index
and the XGBoost components saved in artifacts/demo/ (numpy + xgboost only, browser-safe).
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
PLOTS = ART / "plots"
DEMO = ART / "demo"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="FAIR-RAG on a laptop", page_icon=":mag:", layout="wide")

MAIN = "FAIR-RAG (LightGBM, SEA stop)"
XGBN = "FAIR-RAG (XGBoost components)"
COLORS = {"FAIR": "#2563eb", "STD": "#9ca3af", "ABL": "#f59e0b", "OTHER": "#10b981"}
FAIL_LABELS = {
    "correct": "Correct (evidence complete and answer right)",
    "answer_error": "Answer error (evidence complete or partial, wrong span chosen)",
    "retrieval_miss": "Retrieval miss (a gold paragraph was never retrieved)",
    "filter_drop": "Filter drop (gold paragraph retrieved but filtered out)",
    "early_stop": "Early stop (SEA stopped before the evidence was complete)",
}


# ------------------------------------------------------------------ helpers


def show_fig(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def color_of(name: str) -> str:
    if name.startswith("FAIR-RAG"):
        return COLORS["FAIR"]
    if name.startswith("Standard RAG"):
        return COLORS["STD"]
    if name.startswith("Ablation"):
        return COLORS["ABL"]
    return COLORS["OTHER"]


def pct(x) -> str:
    return f"{100 * float(x):.1f}%"


@st.cache_data(show_spinner=False)
def load_metrics():
    p = ART / "metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_csv(name: str):
    p = ART / name
    if not p.exists():
        return None
    return pd.read_csv(p, keep_default_na=False, na_values=[""])


@st.cache_data(show_spinner=False)
def load_traces():
    p = ART / "test_traces.jsonl"
    out = {}
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    out[t["id"]] = t
    return out


@st.cache_data(show_spinner=False)
def load_doc(name: str) -> str:
    p = DOCS / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def doc_section(text: str, heading: str) -> str:
    """Body of a '## heading' section of a markdown file."""
    m = re.search(rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def strip_mermaid(text: str) -> str:
    return re.sub(r"```mermaid.*?```", "_(diagram shown in the Model tab)_", text, flags=re.S)


@st.cache_resource(show_spinner=False)
def load_demo_system():
    """Live FAIR-RAG system (XGBoost components + compact index slice)."""
    needed = ["corpus.json", "index.npz", "index_meta.json", "xgb_filter.json",
              "xgb_sea.json", "xgb_answer.json"]
    if not all((DEMO / n).exists() for n in needed):
        return None, "missing"
    try:
        from src.demo import load_demo
        return load_demo(DEMO), None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"{type(exc).__name__}: {exc}"


M = load_metrics()

st.title("FAIR-RAG on a laptop")
st.caption("Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation - "
           "a CPU-only, fully reproducible study on HotpotQA multi-hop questions.")

if M is None:
    st.error("No trained model found. The folder `artifacts/` (with `metrics.json`) is missing.")
    st.markdown("Train the model first (about 10 minutes on a laptop CPU), then reload this page:")
    st.code("python train.py", language="bash")
    st.stop()

MM = M["metrics"]
SYS = MM["systems"]
BASE = M["baseline_name"]
SP = load_csv("sample_predictions.csv")
REC = load_csv("test_records.csv")

tabs = st.tabs(["Overview", "Data", "Model", "Results", "Try it", "Research"])

# ====================================================================== OVERVIEW
with tabs[0]:
    st.header("The idea in plain words")
    st.markdown(
        """
Many questions need **two facts from two different documents**. Example: *"Which country is the
director of film X from?"* First you must find the page about film X (it names the director),
and then the page about that director. A normal RAG system searches **once** with the question.
It usually finds the first page but misses the second: the second page shares few words with
the question.

**FAIR-RAG** turns retrieval into a short loop:

1. **Decompose**: split the question into simple search queries.
2. **Retrieve and filter**: search, then keep only paragraphs that a trained *evidence filter*
   thinks are relevant.
3. **Structured Evidence Assessment (SEA)**: make a checklist of what the question needs
   (each named entity, the topic, the "bridge" link between two pages). Tick off what the
   evidence already confirms, and let a trained model judge whether the evidence is *sufficient*.
4. **Refine**: if something is missing, write new queries aimed at the gap (for example,
   search for a person named inside the first paragraph), then go back to step 2.
5. **Answer faithfully**: pick an answer span that literally appears in the evidence, and
   **cite** the paragraph it came from.

The paper uses a large language model for every step. Here, every step is a small rule set or a
gradient-boosted tree model, so the whole system trains and runs on a CPU in minutes.
"""
    )
    st.header("Headline results (test set)")
    st.markdown(
        f"Measured on **{M['n_test']} held-out test questions** and compared with "
        f"**{BASE}**: one search, keeping the same number of paragraphs that FAIR-RAG uses "
        "on average. This keeps the comparison fair."
    )
    b = M["baseline_metrics"]
    c = st.columns(4)
    c[0].metric("Joint evidence recall", pct(M["value"]),
                delta=f"{100 * (M['value'] - M['baseline']):+.1f} pts vs baseline")
    c[1].metric("Baseline joint evidence recall", pct(M["baseline"]))
    c[2].metric("Answer F1", pct(MM["answer_f1"]),
                delta=f"{100 * (MM['answer_f1'] - b['answer_f1']):+.1f} pts")
    c[3].metric("Citation precision", pct(MM["citation_precision"]),
                delta=f"{100 * (MM['citation_precision'] - b['citation_precision']):+.1f} pts")
    c = st.columns(4)
    c[0].metric("Paragraphs used per question", f"{MM['mean_passages']:.2f}",
                delta=f"{MM['mean_passages'] - b['mean_passages']:+.2f}", delta_color="inverse")
    c[1].metric("Search calls per question", f"{MM['mean_retrieval_calls']:.2f}",
                delta=f"{MM['mean_retrieval_calls'] - b['mean_retrieval_calls']:+.2f}",
                delta_color="inverse")
    c[2].metric("Loop rounds per question", f"{MM['mean_iterations']:.2f}")
    c[3].metric("Answer exact match", pct(MM["answer_em"]),
                delta=f"{100 * (MM['answer_em'] - b['answer_em']):+.1f} pts")
    bt = MM.get("paired_bootstrap_vs_main", {}).get(BASE, {}).get("joint_evidence_recall")
    if bt:
        st.info(
            f"The gain in joint evidence recall is **{100 * bt['diff']:+.1f} points**, with a "
            f"95% confidence interval of [{100 * bt['ci_low']:+.1f}, {100 * bt['ci_high']:+.1f}] "
            "points (paired bootstrap over test questions). The interval does not include 0, so "
            "the gain is very unlikely to be luck.")
    with st.expander("How to read these numbers"):
        st.markdown(
            """
- **Joint evidence recall** (the main metric): the share of questions where the final evidence
  set contains **both** gold paragraphs. Without both, no reader can answer faithfully.
- **Answer F1 / exact match**: word overlap / exact equality between the predicted and the true
  answer, after removing case, punctuation and articles. The answerer here is a small span
  ranker, not a large language model, so these numbers are low in absolute terms. Look at the
  *differences* between systems that share the same answerer.
- **Citation precision**: how often the cited paragraph is one of the gold paragraphs.
- **Paragraphs used / search calls**: the cost. Fewer is cheaper.
- Green deltas are improvements. For the cost cards, a red delta means FAIR-RAG spends more.
"""
        )
    pdoc = load_doc("02_problem_definition.md")
    if pdoc:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Problem")
            st.markdown(doc_section(pdoc, "Problem statement"))
            st.markdown(doc_section(pdoc, "Motivation"))
        with c2:
            st.subheader("Objectives")
            st.markdown(doc_section(pdoc, "Objectives (measurable)"))

# ====================================================================== DATA
with tabs[1]:
    st.header("Dataset: HotpotQA (distractor, validation file)")
    ddoc = load_doc("04_dataset.md")
    dd = MM.get("data", {})
    c = st.columns(4)
    c[0].metric("Questions", f"{dd.get('n_questions', 0):,}")
    c[1].metric("Unique paragraphs (corpus)", f"{dd.get('n_paragraphs', 0):,}")
    c[2].metric("Train questions", f"{M['n_train']:,}")
    c[3].metric("Test questions", f"{M['n_test']:,}")
    st.markdown(
        """
Each HotpotQA question needs facts from **two Wikipedia paragraphs** (the *gold* paragraphs).
The dataset ships 10 paragraphs per question: 2 gold paragraphs and 8 look-alike distractors.
We pool the paragraphs of **all** questions into one large search corpus, so each system must
find the 2 gold paragraphs among tens of thousands. Two question types:

- **bridge**: the second paragraph is reached through an entity named in the first one.
- **comparison**: two entities are compared, e.g. *"Are A and B both American?"*
"""
    )
    stc = dd.get("split_type_counts", {})
    if stc:
        sdf = pd.DataFrame([{"split": s, "type": t, "questions": n}
                            for s, v in stc.items() for t, n in v.items()])
        fig = px.bar(sdf, x="split", y="questions", color="type", barmode="group", text="questions",
                     title="Questions per split and type (60/20/20 split, stratified by type)")
        show_fig(fig, key="split_counts")
    with st.expander("Dataset card details (source, license, fields, preprocessing)"):
        st.markdown(ddoc or "docs/04_dataset.md not found.")

    st.subheader("Explore the test questions")
    if SP is None:
        st.warning("artifacts/sample_predictions.csv is missing - run `python train.py`.")
    else:
        sp = SP.copy()
        sp["question_words"] = sp["question"].astype(str).str.split().str.len()
        sp["answer_words"] = sp["true_answer"].astype(str).str.split().str.len()
        sp["answer_kind"] = np.where(sp["true_answer"].astype(str).str.lower().isin(["yes", "no"]),
                                     "yes/no", "span")
        c1, c2, c3 = st.columns([1, 1, 2])
        types = c1.multiselect("Question type", sorted(sp["type"].unique()),
                               default=sorted(sp["type"].unique()), key="d_type")
        kinds = c2.multiselect("Answer kind", ["span", "yes/no"], default=["span", "yes/no"],
                               key="d_kind")
        query = c3.text_input("Search question text", "", key="d_search")
        f = sp[sp["type"].isin(types) & sp["answer_kind"].isin(kinds)]
        if query.strip():
            f = f[f["question"].str.contains(re.escape(query.strip()), case=False)]
        st.caption(f"{len(f)} of {len(sp)} test questions match the filters.")
        show_df(f[["question", "type", "true_answer", "gold_paragraphs"]].head(300),
                hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(f, x="question_words", color="type", nbins=40, barmode="overlay",
                               opacity=0.7, title="Question length (words)")
            show_fig(fig, key="qlen")
        with c2:
            fig = px.histogram(f, x="answer_words", color="answer_kind", nbins=20,
                               title="Answer length (words)")
            show_fig(fig, key="alen")
        st.caption("Most answers are short names, dates or yes/no. Comparison questions are "
                   "often yes/no; bridge questions usually ask for a name.")
    dc = DEMO / "corpus.json"
    if dc.exists():
        with st.expander("Browse paragraphs of the live-demo corpus"):
            @st.cache_data(show_spinner=False)
            def demo_paras():
                with open(dc, encoding="utf-8") as fh:
                    c_ = json.load(fh)
                return pd.DataFrame({"title": c_["titles"],
                                     "text": [" ".join(s) for s in c_["sents"]]})
            pdf = demo_paras()
            t = st.text_input("Title contains", "", key="d_para")
            sel = pdf[pdf["title"].str.contains(re.escape(t), case=False)] if t else pdf
            show_df(sel.head(200), hide_index=True)

# ====================================================================== MODEL
with tabs[2]:
    st.header("Design and pipeline")
    st.graphviz_chart(
        """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontsize=11];
  Q [label="Question"];
  D [label="1. Decompose\\n(question + entity\\nsub-queries)"];
  R [label="2. Hybrid retrieval\\nBM25 (+ LSA via RRF)\\ntop-10 per query"];
  F [label="3. Evidence filter\\n(boosted trees,\\n23 features)"];
  S [label="4. SEA\\nchecklist + sufficiency\\nmodel (22 features)", fillcolor="#fde68a"];
  RF [label="5. Gap-aware refinement\\n(gap + bridge-entity\\nqueries)"];
  A [label="6. Faithful answerer\\nspan ranker over\\nevidence only + citation", fillcolor="#bbf7d0"];
  Q -> D -> R -> F -> S;
  S -> RF [label="not sufficient\\nand rounds left"];
  RF -> R;
  S -> A [label="sufficient\\nor max rounds"];
}
"""
    )
    st.markdown(
        """
**Where are the "learned" parts?** Three gradient-boosted tree classifiers replace the LLM agents
of the paper:

| Component | What it predicts | Trained on |
|---|---|---|
| Evidence filter | P(paragraph is a gold paragraph) for each retrieved paragraph | train questions, out-of-fold over 4 folds |
| SEA sufficiency | P(the current evidence already contains both gold paragraphs) | states of the loop on train questions |
| Answer ranker | P(candidate span is the true answer) for spans found in the evidence | train questions |

The checklist, decomposition and refinement are transparent rules. The main system uses
**LightGBM**. A twin system with **XGBoost** components, trained the same way, powers the live
demo in the *Try it* tab, because XGBoost can run inside the browser.
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Loop settings (main system)")
        cfg = MM.get("config", {})
        xcfg = MM.get("xgb_config", {})
        desc = {"topk": "new paragraphs taken per sub-query",
                "max_sub": "max sub-queries per round", "max_iter": "max loop rounds",
                "max_evidence": "max paragraphs kept as evidence",
                "min_evidence": "always keep at least this many",
                "filter_threshold": "evidence filter keep threshold (tuned)",
                "sea_threshold": "SEA 'sufficient' threshold (tuned on validation)",
                "anchors": "anchor paragraphs used for link features"}
        show_df(pd.DataFrame([{"setting": k, "LightGBM system": v, "XGBoost system": xcfg.get(k),
                               "meaning": desc.get(k, "")} for k, v in cfg.items()]),
                hide_index=True)
    with c2:
        st.subheader("Model hyper-parameters")
        md = load_doc("05_model_design.md")
        hp = doc_section(md, "Hyper-parameters (defaults; tuned ones marked *)")
        st.markdown(hp or "See docs/05_model_design.md")
        tl = MM.get("tuning", {}).get("lgbm", {})
        if tl:
            st.caption(f"Chosen filter parameters: {tl.get('filter_params')}. Out-of-fold ROC-AUC "
                       f"on train: filter {tl.get('filter_oof_auc')}, SEA "
                       f"{tl.get('sea_oof_auc')}, answer ranker {tl.get('answer_oof_auc')}.")

    st.subheader("Training and tuning curves")
    st.markdown(
        "Tree models are not trained by epochs, so there is no loss-per-epoch curve. Below are "
        "the curves the training script actually used to pick its settings. Each one sweeps a "
        "knob and measures the result on **train (out-of-fold) or validation** questions, never "
        "on test.")
    tun = MM.get("tuning", {})
    c1, c2 = st.columns(2)
    with c1:
        rows = []
        for fam in ("lgbm", "xgb"):
            for r in tun.get(fam, {}).get("filter_threshold_grid", []):
                rows.append({**r, "model": "LightGBM" if fam == "lgbm" else "XGBoost"})
        if rows:
            fdf = pd.DataFrame(rows)
            fig = px.line(fdf, x="tau", y="joint_recall", color="model", markers=True,
                          hover_data=["mean_passages", "score"], log_x=True,
                          title="Evidence filter threshold vs joint recall (train, out-of-fold)")
            fig.update_layout(xaxis_title="keep threshold (log scale)",
                              yaxis_title="joint evidence recall")
            show_fig(fig, key="tau")
            st.caption("A lower threshold keeps more paragraphs, so recall goes up but each answer "
                       "costs more. The score trades the two off; the best score was chosen.")
    with c2:
        rows = []
        for fam in ("lgbm", "xgb"):
            for r in tun.get(fam, {}).get("sea_threshold_grid", []):
                rows.append({**r, "model": "LightGBM" if fam == "lgbm" else "XGBoost"})
        if rows:
            sdf = pd.DataFrame(rows)
            fig = px.line(sdf, x="threshold", y="joint_recall", color="model", markers=True,
                          hover_data=["mean_calls", "mean_iterations", "score"],
                          title="SEA stop threshold vs joint recall (validation)")
            fig.update_layout(yaxis_title="joint evidence recall")
            show_fig(fig, key="sea_thr")
            st.caption("A higher threshold means SEA must be more confident before it stops. The "
                       "loop runs longer and finds more evidence, at the cost of more search calls.")
    c1, c2 = st.columns(2)
    with c1:
        rr = tun.get("rrf_lsa_weight", [])
        if rr:
            fig = px.line(pd.DataFrame(rr), x="lsa_weight", y="train_paragraph_recall_at_10",
                          markers=True, title="Weight of dense (LSA) ranking in fusion (train)")
            fig.update_layout(yaxis_title="paragraph recall@10")
            show_fig(fig, key="rrf")
            st.caption(f"Chosen weight: {tun.get('rrf_lsa_weight_chosen')}. On this corpus "
                       "the LSA ranking only hurt BM25, so the final retriever is effectively "
                       "BM25 (LSA is still used as a filter feature).")
    with c2:
        sw = MM.get("iteration_sweep", [])
        if sw:
            swd = pd.DataFrame(sw)
            fig = px.line(swd, x="max_iter", y="joint_evidence_recall", color="mode", markers=True,
                          hover_data=["mean_retrieval_calls", "mean_passages", "answer_f1"],
                          title="Recall as the loop is allowed more rounds (test)")
            fig.update_layout(xaxis_title="maximum loop rounds", yaxis_title="joint evidence recall")
            show_fig(fig, key="sweep_model")
            st.caption("'fixed' always runs every round; 'sea_stop' lets SEA stop early. Each extra "
                       "round finds more of the missing paragraphs.")
    comp = MM.get("components", {})
    if comp:
        st.subheader("Component quality on the test set")
        rows = []
        for k in ("filter_lightgbm", "filter_xgboost", "sea_lightgbm", "sea_xgboost"):
            if k in comp:
                v = comp[k]
                rows.append({"component": k.replace("_", " "), "rows": v.get("n"),
                             "ROC-AUC": v.get("roc_auc"), "avg precision": v.get("avg_precision"),
                             "accuracy": v.get("accuracy"), "precision": v.get("precision"),
                             "recall": v.get("recall"), "threshold": v.get("threshold"),
                             "false 'sufficient' rate": v.get("false_sufficient_rate")})
        show_df(pd.DataFrame(rows), hide_index=True)
        st.caption("The filter threshold is deliberately low (keep almost every gold paragraph: "
                   "high recall, low precision). The SEA threshold is deliberately strict: when "
                   "it says 'sufficient' it is almost always right (precision), so it rarely "
                   "stops too early.")

# ====================================================================== RESULTS
with tabs[3]:
    st.header("All systems on the test set")
    st.markdown(
        "Every system shares the same retriever, corpus and answer ranker. Only the **retrieval "
        "strategy** changes. *Ablations* switch off one FAIR-RAG part to show what that part "
        "contributes.")
    rows = []
    for n, s in SYS.items():
        rows.append({"system": n, "joint evidence recall": s["joint_evidence_recall"],
                     "recall 95% CI": f"[{s['joint_ci'][0]:.3f}, {s['joint_ci'][1]:.3f}]",
                     "paragraph recall": s["paragraph_recall"],
                     "evidence precision": s["evidence_precision"],
                     "answer F1": s["answer_f1"], "answer EM": s["answer_em"],
                     "citation precision": s["citation_precision"],
                     "paragraphs used": s["mean_passages"],
                     "search calls": s["mean_retrieval_calls"], "rounds": s["mean_iterations"]})
    sysdf = pd.DataFrame(rows)
    show_df(sysdf.round(4), hide_index=True)

    metric_choice = st.radio("Metric for the bar chart",
                             ["joint evidence recall", "answer F1"], horizontal=True, key="r_m")
    key, cik = (("joint_evidence_recall", "joint_ci") if metric_choice == "joint evidence recall"
                else ("answer_f1", "f1_ci"))
    names = list(SYS.keys())
    vals = [SYS[n][key] for n in names]
    lo = [SYS[n][key] - SYS[n][cik][0] for n in names]
    hi = [SYS[n][cik][1] - SYS[n][key] for n in names]
    order = np.argsort(vals)
    fig = go.Figure(go.Bar(
        x=[vals[i] for i in order], y=[names[i] for i in order], orientation="h",
        marker_color=[color_of(names[i]) for i in order],
        error_x=dict(type="data", symmetric=False, array=[hi[i] for i in order],
                     arrayminus=[lo[i] for i in order]),
        text=[f"{vals[i]:.3f}" for i in order], textposition="outside"))
    fig.update_layout(title=f"{metric_choice} with 95% bootstrap confidence interval",
                      xaxis_title=metric_choice, height=460, margin=dict(l=10))
    show_fig(fig, key="sys_bar")
    st.caption("Blue = FAIR-RAG, grey = standard single-shot RAG, orange = ablations, green = "
               "Iter-Retgen-style loop. Black whiskers are 95% confidence intervals. Bars whose "
               "whiskers do not overlap differ clearly.")

    c1, c2 = st.columns(2)
    with c1:
        cdf = sysdf.copy()
        cdf["family"] = [("FAIR-RAG" if n.startswith("FAIR") else "Standard RAG"
                          if n.startswith("Standard") else "Ablation" if n.startswith("Ablation")
                          else "Iter-Retgen") for n in cdf["system"]]
        xcost = st.selectbox("Cost axis", ["search calls", "paragraphs used"], key="r_cost")
        fig = px.scatter(cdf, x=xcost, y="joint evidence recall", color="family", text="system",
                         title="Cost vs quality (up and left is better)")
        fig.update_traces(textposition="top center", textfont_size=9, marker_size=12)
        show_fig(fig, key="cost")
    with c2:
        rak = MM.get("recall_at_k", {})
        if rak:
            rows = []
            for meth in ("bm25", "lsa", "hybrid"):
                for k_, j, p in zip(rak["k"], rak[meth]["joint"], rak[meth]["paragraph"]):
                    rows.append({"method": meth, "k": k_, "joint recall": j,
                                 "paragraph recall": p})
            rdf = pd.DataFrame(rows)
            which = st.selectbox("Recall@k measure", ["joint recall", "paragraph recall"],
                                 key="r_rak")
            fig = px.line(rdf, x="k", y=which, color="method", markers=True,
                          title="Single-shot retrieval: recall in the top-k results")
            fig.add_hline(y=M["value"], line_dash="dash", line_color=COLORS["FAIR"],
                          annotation_text="FAIR-RAG joint recall")
            show_fig(fig, key="rak")
            st.caption("Even 20 single-shot results find both gold paragraphs less often than "
                       "FAIR-RAG does with about 6. Searching smarter beats searching deeper.")

    bt = MM.get("paired_bootstrap_vs_main", {})
    if bt:
        st.subheader(f"Is the difference real? Paired bootstrap vs {MAIN}")
        brow = []
        for n, v in bt.items():
            j, f_ = v["joint_evidence_recall"], v["answer_f1"]
            brow.append({"compared system": n, "recall gain of FAIR-RAG": j["diff"],
                         "recall gain 95% CI": f"[{j['ci_low']:+.3f}, {j['ci_high']:+.3f}]",
                         "F1 gain of FAIR-RAG": f_["diff"],
                         "F1 gain 95% CI": f"[{f_['ci_low']:+.3f}, {f_['ci_high']:+.3f}]"})
        show_df(pd.DataFrame(brow).round(4), hide_index=True)
        st.caption("Positive = the main FAIR-RAG system is better. If a CI contains 0, the two "
                   "systems are statistically tied on this test set.")

    c1, c2 = st.columns(2)
    with c1:
        bty = pd.DataFrame(MM.get("by_type", []))
        if len(bty):
            m_ = st.selectbox("Breakdown metric", ["joint_evidence_recall", "answer_f1", "answer_em"],
                              key="r_bt")
            fig = px.bar(bty, x="type", y=m_, color="system", barmode="group",
                         hover_data=["n"], title="By question type")
            show_fig(fig, key="bytype")
    with c2:
        ft = MM.get("failure_taxonomy", {})
        if ft:
            fdf = pd.DataFrame([{"system": n, "outcome": k, "share": v}
                                for n, d in ft.items() for k, v in d.items()])
            fig = px.bar(fdf, y="system", x="share", color="outcome", orientation="h",
                         title="Where does each system fail?",
                         category_orders={"outcome": list(FAIL_LABELS)})
            fig.update_layout(legend=dict(orientation="h", y=-0.25))
            show_fig(fig, key="fail")
    with st.expander("What the outcome categories mean"):
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in FAIL_LABELS.items()))

    c1, c2 = st.columns(2)
    with c1:
        fi = MM.get("feature_importance", {})
        if fi:
            fam = st.selectbox("Model family", list(fi.keys()), key="r_fam")
            compn = st.selectbox("Component", list(fi[fam].keys()), key="r_comp")
            imp = pd.Series(fi[fam][compn]).sort_values().tail(15)
            fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                         title=f"Top features - {compn} ({fam}), share of total gain")
            fig.update_layout(xaxis_title="importance (share of gain)", yaxis_title="")
            show_fig(fig, key="fi")
    with c2:
        ih = MM.get("iterations_used_hist", {})
        if ih:
            idf = pd.DataFrame([{"system": n, "rounds used": i + 1, "questions": v}
                                for n, h in ih.items() for i, v in enumerate(h)])
            fig = px.bar(idf, x="rounds used", y="questions", color="system", barmode="group",
                         title="How many loop rounds SEA allowed before stopping")
            show_fig(fig, key="iters")
            st.caption("Questions whose evidence looked sufficient stop after round 1; harder "
                       "ones use the gap-refinement rounds.")

    st.subheader("Error analysis (main system, per test question)")
    if SP is None:
        st.warning("artifacts/sample_predictions.csv is missing - run `python train.py`.")
    else:
        sp = SP.copy()
        c1, c2 = st.columns(2)
        with c1:
            sp["confidence bin"] = pd.cut(sp["confidence"], bins=[0, .1, .2, .3, .4, .5, .6, .8, 1.0],
                                          include_lowest=True).astype(str)
            cal = sp.groupby("confidence bin", observed=True).agg(
                questions=("f1", "size"), exact_match=("exact_match", "mean")).reset_index()
            fig = px.bar(cal, x="confidence bin", y="exact_match", text="questions",
                         title="Is the answer confidence meaningful? (exact match per bin)")
            fig.update_layout(yaxis_title="share answered exactly right",
                              xaxis_title="answer ranker probability")
            show_fig(fig, key="cal")
            st.caption("Numbers on bars = questions in the bin. Higher-confidence answers are "
                       "right more often, so the confidence can be used to flag doubtful answers.")
        with c2:
            g = sp.groupby(["iterations"]).agg(questions=("f1", "size"),
                                               joint=("joint_evidence_recall", "mean"),
                                               f1=("f1", "mean")).reset_index()
            fig = go.Figure()
            fig.add_bar(x=g["iterations"], y=g["joint"], name="joint evidence recall")
            fig.add_bar(x=g["iterations"], y=g["f1"], name="answer F1")
            fig.update_layout(barmode="group", title="Quality by number of rounds used",
                              xaxis_title="rounds used")
            show_fig(fig, key="byiter")
            st.caption("Questions that needed more rounds are the harder ones.")
        wl = pd.DataFrame({
            "FAIR-RAG found both, Standard RAG did not":
                [int(((sp.joint_evidence_recall == 1) & (sp.std_rag_joint_recall == 0)).sum())],
            "Standard RAG found both, FAIR-RAG did not":
                [int(((sp.joint_evidence_recall == 0) & (sp.std_rag_joint_recall == 1)).sum())],
            "both found both": [int(((sp.joint_evidence_recall == 1)
                                     & (sp.std_rag_joint_recall == 1)).sum())],
            "neither": [int(((sp.joint_evidence_recall == 0)
                             & (sp.std_rag_joint_recall == 0)).sum())]}).T.reset_index()
        wl.columns = ["case", "questions"]
        fig = px.bar(wl, x="questions", y="case", orientation="h", text="questions",
                     title=f"Head-to-head on evidence: FAIR-RAG vs {BASE}")
        show_fig(fig, key="wl")

        st.markdown("**Browse individual test questions** (filter, then pick one to see its "
                    "full loop trace).")
        c1, c2, c3 = st.columns(3)
        fsel = c1.multiselect("Outcome", sorted(sp["failure_type"].unique()),
                              default=[x for x in ["retrieval_miss", "filter_drop", "early_stop"]
                                       if x in set(sp["failure_type"])] or None, key="e_fail")
        tsel = c2.multiselect("Type", sorted(sp["type"].unique()),
                              default=sorted(sp["type"].unique()), key="e_type")
        only_diff = c3.checkbox("Only where FAIR-RAG and Standard RAG differ on evidence",
                                key="e_diff")
        e = sp[sp["type"].isin(tsel)]
        if fsel:
            e = e[e["failure_type"].isin(fsel)]
        if only_diff:
            e = e[e["joint_evidence_recall"] != e["std_rag_joint_recall"]]
        st.caption(f"{len(e)} questions")
        show_df(e[["question", "true_answer", "prediction", "confidence", "f1",
                   "joint_evidence_recall", "iterations", "failure_type", "std_rag_prediction",
                   "std_rag_joint_recall", "gold_paragraphs", "evidence_paragraphs"]].head(500),
                hide_index=True)
        if len(e):
            traces = load_traces()
            qpick = st.selectbox("Show the trace of", e["question"].head(500).tolist(),
                                 key="e_pick")
            qid = e[e["question"] == qpick]["id"].iloc[0]
            tr = traces.get(qid)
            if tr:
                st.markdown(f"**Gold answer:** {tr['gold_answer']} | **Predicted:** "
                            f"{tr['answer']} | **Gold paragraphs:** "
                            f"{', '.join(tr['gold_titles'])} | **Outcome:** {tr['failure_type']}")
                for it in tr["iterations"]:
                    with st.expander(f"Round {it['t']} - SEA P(sufficient) = {it['sea_p']}"
                                     f" -> {'stop' if it['sufficient'] else 'continue'}"):
                        st.markdown("Queries: " + "; ".join(f"`{q['type']}`: {q['text']}"
                                                            for q in it["queries"]))
                        ev = pd.DataFrame(it["evidence"])
                        if len(ev):
                            ev["gold"] = ev["title"].isin(tr["gold_titles"])
                            show_df(ev, hide_index=True)
                        st.markdown("Checklist: " + "; ".join(
                            f"{'[x]' if c['confirmed'] else '[ ]'} {c['kind']}: {c['text']}"
                            for c in it["checklist"]))

    imgs = [p for p in (MM.get("plots") or []) if (PLOTS / p).exists()]
    if imgs:
        with st.expander(f"Static figures saved by train.py ({len(imgs)} PNG files)"):
            for p in imgs:
                st.image(str(PLOTS / p), caption=p)

# ====================================================================== TRY IT
with tabs[4]:
    st.header("Try FAIR-RAG live")
    (demo_sys, err) = load_demo_system()
    if demo_sys is None:
        if err == "missing":
            st.warning("The live-demo bundle `artifacts/demo/` is missing. Create it with "
                       "`python train.py` (or, after a full run, `python train.py --demo-only`).")
        else:
            st.error(f"Could not load the live demo: {err}")
    else:
        system, dcorp, dmeta = demo_sys
        dm = M.get("demo", {})
        st.markdown(
            f"""
This runs the **real trained loop** (XGBoost components) in your browser or Python session.
To keep it light, it searches a **demo corpus of {len(dcorp['titles']):,} paragraphs**: the
context paragraphs of {len(dcorp['questions'])} test questions, out of
{dmeta.get('n_full_paragraphs', 0):,} in the full corpus. The search scores are copied from
the full index. The pool is smaller, though, so finding evidence is easier here than in the
reported test results.
"""
        )
        if dm.get("systems"):
            with st.expander("Accuracy of this exact demo setup (from metrics.json)"):
                show_df(pd.DataFrame(dm["systems"]).T.round(4))
                st.caption(f"Measured by train.py on the {dm.get('n_questions')} demo questions "
                           "over the demo corpus. These numbers are higher than the main results "
                           "because the demo corpus is much smaller.")
        mode = st.radio("Input", ["Pick a test question", "Type your own question"],
                        horizontal=True, key="t_mode")
        qs = dcorp["questions"]
        gold = None
        if mode == "Pick a test question":
            tfilter = st.selectbox("Question type", ["any", "bridge", "comparison"], key="t_type")
            pool = [q for q in qs if tfilter == "any" or q["type"] == tfilter]
            labels = [f"{q['question']}  [{q['type']}]" for q in pool]
            idx = st.selectbox("Question", range(len(pool)), format_func=lambda i: labels[i],
                               key="t_q")
            gold = pool[idx]
            question = gold["question"]
        else:
            question = st.text_area(
                "Your question (in English; it can only be answered from the demo corpus topics,"
                " so browse 'Data > Browse paragraphs' for ideas)",
                value=qs[0]["question"], key="t_text", height=80)
        c1, c2, c3, c4 = st.columns(4)
        max_iter = c1.slider("Max loop rounds", 1, 4, int(system.cfg.max_iter), key="t_iter")
        thr = c2.slider("SEA stop threshold", 0.30, 0.99, float(system.cfg.sea_threshold), 0.01,
                        key="t_thr", help="Stop when P(sufficient) >= this value.")
        max_ev = c3.slider("Max evidence paragraphs", 2, 10, int(system.cfg.max_evidence),
                           key="t_ev")
        compare = c4.checkbox("Also run Standard RAG", value=True, key="t_cmp")

        @st.cache_data(show_spinner=False, max_entries=64)
        def run_live(q, max_iter, thr, max_ev):
            from src.model import FairRAG
            cfg = dataclasses.replace(system.cfg, max_evidence=max_ev, sea_threshold=thr)
            s2 = FairRAG(system.retr, system.kb, cfg, system.models)
            return s2.ask(q, max_iter=max_iter, threshold=thr)

        @st.cache_data(show_spinner=False, max_entries=64)
        def run_std(q, k):
            from src.model import FairRAG, QBank, QInfo
            st_ = system.run(QBank([QInfo(q)], system.retr), max_iter=1,
                             decompose_mode="question", use_filter=False, max_evidence=k,
                             topk=max(system.cfg.topk, k))[0]
            sn = st_.snaps[-1]
            a = system.answer([(st_.qi, sn["ev"], sn["ev_probs"])])[0]
            return {"answer": a["pred"], "conf": a["conf"],
                    "evidence": [system.kb.titles[int(d)] for d in sn["ev"]],
                    "citation": system.kb.titles[a["cite"]] if a["cite"] >= 0 else None}

        if not question or not question.strip():
            st.info("Type a question to start.")
        else:
            with st.spinner("Running the FAIR-RAG loop ..."):
                tr = run_live(question.strip(), max_iter, thr, max_ev)
            last = tr["iterations"][-1] if tr["iterations"] else None
            ev_titles = [e["title"] for e in last["evidence"]] if last else []
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FAIR-RAG answer", tr["answer"] or "(none)")
            c2.metric("Answer confidence", pct(tr["confidence"]))
            c3.metric("Rounds used", tr["n_iter"])
            c4.metric("Search calls", last["calls"] if last else 0)
            if gold is not None:
                from src.text import exact_match, f1_score
                got = set(gold["gold_titles"]) <= set(ev_titles)
                ok = exact_match(tr["answer"], gold["answer"]) == 1.0
                f1v = f1_score(tr["answer"], gold["answer"])
                (st.success if ok else st.warning)(
                    f"True answer: **{gold['answer']}**. Prediction is "
                    f"{'exactly right' if ok else f'not exact (word-overlap F1 = {f1v:.2f})'}. "
                    f"Both gold paragraphs ({' + '.join(gold['gold_titles'])}) "
                    f"{'are' if got else 'are NOT both'} in the final evidence.")

            # plain-language explanation
            why = []
            if last:
                if last["sufficient"]:
                    why.append(f"The loop stopped after round {tr['n_iter']} because the SEA "
                               f"model judged the evidence sufficient (P = {last['sea_p']:.2f}, "
                               f"threshold {thr:.2f}).")
                else:
                    why.append(f"The loop used all {tr['n_iter']} allowed round(s); SEA never "
                               f"reached the threshold (last P = {last['sea_p']:.2f}).")
                conf = [c for c in last["checklist"] if c["confirmed"]]
                gaps = [c for c in last["checklist"] if not c["confirmed"]]
                if conf:
                    why.append("Confirmed in the evidence: " + "; ".join(
                        f"{c['kind']} '{c['text']}' (by *{c['by']}*)" for c in conf) + ".")
                if gaps:
                    why.append("Still unconfirmed: " + "; ".join(
                        f"{c['kind']} '{c['text']}'" for c in gaps) + ".")
            if tr["citation"]:
                cid = system.kb.titles.index(tr["citation"])
                sent = next((s for s in system.kb.sents[cid]
                             if tr["answer"].lower() in s.lower()), None)
                why.append(f"The answer is cited from **{tr['citation']}**"
                           + (f": \"{sent}\"" if sent and tr["answer"].lower()
                              not in ("yes", "no") else "."))
            if len(tr["top_candidates"]) > 1:
                why.append("Runner-up candidates: " + ", ".join(
                    f"{c['text']} ({c['p']:.2f})" for c in tr["top_candidates"][1:]) + ".")
            st.markdown("**Why this answer?**\n\n" + "\n".join(f"- {w}" for w in why))
            st.caption("Confidence is the answer ranker's probability that this span is the true "
                       "answer. Below about 0.2, the answer is right less often "
                       "(see Results > Error analysis).")

            if tr["top_candidates"]:
                cdf = pd.DataFrame(tr["top_candidates"])
                fig = px.bar(cdf, x="p", y="text", orientation="h", hover_data=["doc"],
                             title="Top answer candidates (all taken from the evidence)")
                fig.update_layout(yaxis=dict(autorange="reversed"), height=260,
                                  xaxis_title="answer ranker probability", yaxis_title="")
                show_fig(fig, key="t_cands")

            st.subheader("Step by step")
            sea_ps = [it["sea_p"] for it in tr["iterations"]]
            if sea_ps:
                fig = go.Figure(go.Scatter(x=list(range(1, len(sea_ps) + 1)), y=sea_ps,
                                           mode="lines+markers", name="P(sufficient)"))
                fig.add_hline(y=thr, line_dash="dash", annotation_text="stop threshold")
                fig.update_layout(title="SEA sufficiency per round", xaxis_title="round",
                                  yaxis_title="P(evidence sufficient)", yaxis_range=[0, 1],
                                  height=260, xaxis=dict(dtick=1))
                show_fig(fig, key="t_sea")
            for it in tr["iterations"]:
                with st.expander(f"Round {it['t']}: {len(it['queries'])} queries, "
                                 f"{it['n_new']} new paragraphs, SEA P = {it['sea_p']}",
                                 expanded=(it["t"] == tr["n_iter"])):
                    st.markdown("**Queries issued**")
                    for q in it["queries"]:
                        st.markdown(f"- `{q['type']}` {q['text']}")
                    st.markdown("**Evidence kept** (filter probability = how relevant the "
                                "paragraph looks)")
                    ev = pd.DataFrame(it["evidence"])
                    if len(ev):
                        if gold is not None:
                            ev["gold"] = ev["title"].isin(gold["gold_titles"])
                        show_df(ev.rename(columns={"p": "filter probability"}), hide_index=True)
                    st.markdown("**SEA checklist**")
                    for c in it["checklist"]:
                        st.markdown(f"- {'[x]' if c['confirmed'] else '[ ]'} **{c['kind']}**: "
                                    f"{c['text']}" + (f" - confirmed by *{c['by']}*"
                                                      if c["confirmed"] else " - gap"))
            if ev_titles:
                with st.expander("Read the final evidence paragraphs"):
                    for t in ev_titles:
                        i = system.kb.titles.index(t)
                        st.markdown(f"**{t}**: " + " ".join(system.kb.sents[i]))

            if compare:
                k = int(MM.get("matched_budget_k", 6))
                sr = run_std(question.strip(), k)
                st.subheader(f"Standard RAG (one search, top-{k}) on the same question")
                c1, c2 = st.columns([1, 2])
                c1.metric("Standard RAG answer", sr["answer"] or "(none)")
                msg = "Evidence: " + ", ".join(sr["evidence"])
                if gold is not None:
                    msg += (f"  \nBoth gold paragraphs found: "
                            f"{set(gold['gold_titles']) <= set(sr['evidence'])}")
                c2.markdown(msg)
            if gold is not None:
                ft = load_traces().get(gold["id"])
                if ft:
                    st.caption(f"For reference, the main LightGBM system on the FULL "
                               f"{dmeta.get('n_full_paragraphs', 0):,}-paragraph corpus answered "
                               f"'{ft['answer']}' after {ft['n_iter']} round(s) (recorded by "
                               "train.py).")

# ====================================================================== RESEARCH
with tabs[5]:
    st.header("Research notes")
    st.markdown("Source paper: *FAIR-RAG: Faithful Adaptive Iterative Refinement for "
                "Retrieval-Augmented Generation* - [arXiv:2510.22344]"
                "(http://arxiv.org/abs/2510.22344v1)")
    docs = [("Literature survey", "01_literature_survey.md"),
            ("Problem definition", "02_problem_definition.md"),
            ("Solution strategy", "03_solution_strategy.md"),
            ("Dataset", "04_dataset.md"), ("Model design", "05_model_design.md")]
    extra = sorted(p.name for p in DOCS.glob("*.md")) if DOCS.exists() else []
    for n in extra:
        if n not in {d[1] for d in docs}:
            docs.append((n.replace(".md", "").replace("_", " ").strip(" 0123456789").title()
                         or n, n))
    for p in ("paper.md", "PAPER.md"):
        if (ROOT / p).exists():
            docs.append(("Paper", "../" + p))
    pick = st.radio("Document", [d[0] for d in docs], horizontal=True, key="res_doc")
    fname = dict(docs)[pick]
    txt = load_doc(fname)
    if txt:
        st.markdown(strip_mermaid(txt))
    else:
        st.info(f"docs/{fname} not found.")
