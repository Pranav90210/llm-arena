import streamlit as st
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
from pipeline import (
    run_all_models, judge_response, evaluate_question,
    run_benchmark, save_results, MODELS, QUESTION_BANK
)

st.set_page_config(
    page_title="LLM Arena",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background: #0a0a0f;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #1e1e2e;
}

/* Score card */
.score-card {
    background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a3e;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.2s;
}
.score-card:hover {
    border-color: #4a4a7e;
    transform: translateY(-2px);
}
.score-card.winner {
    border-color: #f0c040;
    background: linear-gradient(135deg, #1a1810 0%, #2a2418 100%);
}
.model-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.5rem;
}
.score-number {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.5rem 0;
}
.score-dims {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 0.75rem;
    font-size: 0.8rem;
    color: #666;
}
.dim-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
}
.dim-value {
    font-weight: 700;
    font-size: 1rem;
    color: #aaa;
}
.latency-badge {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.75rem;
}
.winner-badge {
    display: inline-block;
    background: #f0c040;
    color: #000;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

/* Response box */
.response-box {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.25rem;
    height: 320px;
    overflow-y: auto;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #ccc;
}
.response-box::-webkit-scrollbar { width: 4px; }
.response-box::-webkit-scrollbar-track { background: #0f0f1a; }
.response-box::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 2px; }

.feedback-text {
    font-style: italic;
    color: #666;
    font-size: 0.8rem;
    margin-bottom: 0.75rem;
    padding-left: 0.5rem;
    border-left: 2px solid #2a2a3e;
}

/* Section headers */
.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e2e;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #888 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: #444;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
}

/* Question display */
.question-box {
    background: #111118;
    border-left: 3px solid #4a4a7e;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    font-size: 1rem;
    color: #ccc;
    margin-bottom: 1.5rem;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2a3e, transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

MODEL_COLORS = {
    "llama-3.1-8b": "#7c6af7",
    "qwen3.6-27b":  "#34d399",
    "gpt-oss-20b":  "#f97316"
}

MODEL_EMOJIS = {
    "llama-3.1-8b": "🦙",
    "qwen3.6-27b":  "🌸",
    "gpt-oss-20b":  "⚡"
}

def strip_think(text: str) -> str:
    """Strip <think>...</think> blocks from model output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def radar_chart(scores_by_model: dict):
    categories = ["Accuracy", "Clarity", "Reasoning"]
    fig = go.Figure()
    for model, scores in scores_by_model.items():
        color = MODEL_COLORS.get(model, "#fff")
        values = [
            scores.get("accuracy", 0),
            scores.get("clarity", 0),
            scores.get("reasoning", 0)
        ]
        # Convert hex to rgba
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=f"{MODEL_EMOJIS.get(model,'')} {model}",
            line=dict(color=color, width=2),
            fillcolor=f"rgba({r},{g},{b},0.15)",
            opacity=1.0
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(color="#444", size=10),
                gridcolor="#1e1e2e",
                linecolor="#1e1e2e"
            ),
            angularaxis=dict(
                tickfont=dict(color="#888", size=12),
                gridcolor="#1e1e2e",
                linecolor="#1e1e2e"
            )
        ),
        showlegend=True,
        legend=dict(font=dict(color="#888"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888"),
        height=380,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    return fig

def leaderboard_chart(df: pd.DataFrame):
    fig = go.Figure()
    for _, row in df.iterrows():
        color = MODEL_COLORS.get(row["model"], "#888")
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fig.add_trace(go.Bar(
            x=[row["model"]],
            y=[row["avg_overall"]],
            name=row["model"],
            marker=dict(
                color=f"rgba({r},{g},{b},0.85)",
                line=dict(color=color, width=1),
                cornerradius=8
            ),
            text=[f"{row['avg_overall']:.1f}"],
            textposition="outside",
            textfont=dict(color="#fff", size=14, family="Inter")
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=280,
        yaxis=dict(range=[0, 11], gridcolor="#1e1e2e", tickfont=dict(color="#444"), zeroline=False),
        xaxis=dict(tickfont=dict(color="#888")),
        margin=dict(l=20, r=20, t=20, b=20),
        bargap=0.4
    )
    return fig

def win_rate_chart(win_counts: pd.DataFrame):
    fig = px.bar(
        win_counts, x="category", y="wins", color="winner",
        color_discrete_map=MODEL_COLORS, barmode="group",
        labels={"wins": "Wins", "category": "Category", "winner": "Model"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
        height=280,
        xaxis=dict(gridcolor="#1e1e2e"),
        yaxis=dict(gridcolor="#1e1e2e", zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20),
        bargap=0.3
    )
    return fig

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚔️ LLM Arena")
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Single Question", "Batch Benchmark"], label_visibility="collapsed")

    if mode == "Single Question":
        category = st.selectbox("Category", ["factual", "reasoning", "creative"])
        questions = QUESTION_BANK[category]
        selected_idx = st.selectbox(
            "Question",
            range(len(questions)),
            format_func=lambda x: f"{questions[x]['id']}: {questions[x]['question'][:55]}..."
        )
        selected_q = questions[selected_idx]
    else:
        batch_category = st.selectbox("Category", ["all", "factual", "reasoning", "creative"])
        max_q = st.slider("Max questions", 1, 30, 6)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Models</div>', unsafe_allow_html=True)
    for model, color in MODEL_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{color}"></div>'
            f'<span style="color:#888;font-size:0.8rem">{model}</span></div>',
            unsafe_allow_html=True
        )

# ── HERO ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">⚔️ LLM Arena</div>
    <div class="hero-sub">BENCHMARK · EVALUATE · COMPARE</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── SINGLE QUESTION MODE ─────────────────────────────
if mode == "Single Question":
    st.markdown(f'<div class="section-header">{selected_q["category"].upper()} / {selected_q["id"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="question-box">{selected_q["question"]}</div>', unsafe_allow_html=True)

    if st.button("⚔️ Run Arena", type="primary", use_container_width=True):
        with st.spinner("Running models..."):
            result = evaluate_question(selected_q)

        res = result["results"]
        best_model = max(res.keys(), key=lambda m: res[m]["scores"].get("overall", 0))

        # Score cards
        cols = st.columns(3)
        for i, (model, data) in enumerate(res.items()):
            overall = data["scores"].get("overall", 0)
            color = MODEL_COLORS.get(model, "#fff")
            is_winner = model == best_model
            acc = data["scores"].get("accuracy", 0)
            clr = data["scores"].get("clarity", 0)
            rsn = data["scores"].get("reasoning", 0)
            lat = data.get("latency_s", "—")
            emoji = MODEL_EMOJIS.get(model, "")
            with cols[i]:
                if is_winner:
                    st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1810,#2a2418);border:1px solid #f0c040;border-radius:16px;padding:1.5rem;text-align:center;">
<div style="background:#f0c040;color:#000;border-radius:20px;padding:2px 12px;font-size:0.7rem;font-weight:800;display:inline-block;margin-bottom:0.5rem;">🏆 WINNER</div>
<div style="font-size:0.8rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#888;margin-bottom:0.5rem;">{emoji} {model}</div>
<div style="font-size:3.5rem;font-weight:800;color:{color};line-height:1;">{overall}</div>
<div style="display:flex;justify-content:center;gap:1.5rem;margin-top:0.75rem;">
<div style="text-align:center;"><div style="font-size:0.6rem;color:#555;text-transform:uppercase;">ACC</div><div style="font-size:1rem;font-weight:700;color:#aaa;">{acc}</div></div>
<div style="text-align:center;"><div style="font-size:0.6rem;color:#555;text-transform:uppercase;">CLR</div><div style="font-size:1rem;font-weight:700;color:#aaa;">{clr}</div></div>
<div style="text-align:center;"><div style="font-size:0.6rem;color:#555;text-transform:uppercase;">RSN</div><div style="font-size:1rem;font-weight:700;color:#aaa;">{rsn}</div></div>
</div>
<div style="margin-top:0.75rem;background:#1e1e2e;border:1px solid #2a2a3e;border-radius:20px;padding:2px 10px;display:inline-block;font-size:0.75rem;color:#666;">⏱ {lat}s</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="background:linear-gradient(135deg,#13131f,#1a1a2e);border:1px solid #2a2a3e;border-radius:16px;padding:1.5rem;text-align:center;">
<div style="font-size:0.8rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#888;margin-bottom:0.5rem;">{emoji} {model}</div>
<div style="font-size:3.5rem;font-weight:800;color:{color};line-height:1;">{overall}</div>
<div style="display:flex;justify-content:center;gap:1.5rem;margin-top:0.75rem;">
<div style="text-align:center;"><div style="font-size:0.6rem;color:#555;text-transform:uppercase;">ACC</div><div style="font-size:1rem;font-weight:700;color:#aaa;">{acc}</div></div>
<div style="text-align:center;"><div style="font-size:0.6rem;color:#555;text-transform:uppercase;">CLR</div><div style="font-size:1rem;font-weight:700;color:#aaa;">{clr}</div></div>
<div style="text-align:center;"><div style="font-size:0.6rem;color:#555;text-transform:uppercase;">RSN</div><div style="font-size:1rem;font-weight:700;color:#aaa;">{rsn}</div></div>
</div>
<div style="margin-top:0.75rem;background:#1e1e2e;border:1px solid #2a2a3e;border-radius:20px;padding:2px 10px;display:inline-block;font-size:0.75rem;color:#666;">⏱ {lat}s</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Radar chart
        st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
        scores_by_model = {m: res[m]["scores"] for m in res}
        st.plotly_chart(radar_chart(scores_by_model), use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Responses
        st.markdown('<div class="section-header">Responses</div>', unsafe_allow_html=True)
        rcols = st.columns(3)
        for i, (model, data) in enumerate(res.items()):
            color = MODEL_COLORS.get(model, "#fff")
            clean_answer = strip_think(data["answer"])
            with rcols[i]:
                st.markdown(f'<div style="font-weight:700;color:{color};margin-bottom:0.5rem">{MODEL_EMOJIS.get(model,"")} {model}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="feedback-text">{data["scores"].get("feedback","")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="response-box">{clean_answer}</div>', unsafe_allow_html=True)

# ── BATCH MODE ───────────────────────────────────────
else:
    cat_filter = None if batch_category == "all" else batch_category
    total_q = sum(len(v) for v in QUESTION_BANK.values()) if not cat_filter else len(QUESTION_BANK[batch_category])

    st.markdown('<div class="section-header">Batch Benchmark</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#555;font-size:0.85rem;margin-bottom:1.5rem">Running {min(max_q, total_q)} questions · {batch_category} category · 3 models · LLM-as-a-Judge scoring</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    run_btn = col1.button("🚀 Run Benchmark", type="primary", use_container_width=True)
    load_prev = col2.button("📂 Load Previous Results", use_container_width=True)

    results = None

    if load_prev:
        try:
            with open("results.json") as f:
                results = json.load(f)
            st.success(f"Loaded {len(results)} results")
        except:
            st.error("No results.json found.")

    if run_btn:
        progress = st.progress(0)
        status = st.empty()
        results = []
        questions = QUESTION_BANK[cat_filter] if cat_filter else [q for cat in QUESTION_BANK.values() for q in cat]
        questions = questions[:max_q]
        for i, q in enumerate(questions):
            status.text(f"Running {i+1}/{len(questions)}: {q['id']}...")
            results.append(evaluate_question(q))
            progress.progress((i + 1) / len(questions))
            time.sleep(0.5)
        save_results(results)
        status.empty()
        st.success(f"Done — {len(results)} questions evaluated")

    if results:
        rows = []
        for r in results:
            for model, data in r["results"].items():
                rows.append({
                    "question_id": r["question_id"],
                    "category": r["category"],
                    "model": model,
                    "accuracy": data["scores"].get("accuracy", 0),
                    "clarity": data["scores"].get("clarity", 0),
                    "reasoning": data["scores"].get("reasoning", 0),
                    "overall": data["scores"].get("overall", 0),
                    "latency_s": data.get("latency_s", 0)
                })
        df = pd.DataFrame(rows)

        summary = df.groupby("model").agg(
            avg_overall=("overall", "mean"),
            avg_accuracy=("accuracy", "mean"),
            avg_clarity=("clarity", "mean"),
            avg_reasoning=("reasoning", "mean"),
            avg_latency=("latency_s", "mean")
        ).round(2).reset_index().sort_values("avg_overall", ascending=False)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Leaderboard
        st.markdown('<div class="section-header">Leaderboard</div>', unsafe_allow_html=True)
        st.plotly_chart(leaderboard_chart(summary), use_container_width=True)

        # Stats table
        lcols = st.columns(3)
        for i, row in enumerate(summary.itertuples()):
            color = MODEL_COLORS.get(row.model, "#fff")
            with lcols[i]:
                st.markdown(f"""
                <div class="score-card">
                    <div class="model-label" style="color:{color}">#{i+1} {row.model}</div>
                    <div class="score-number" style="color:{color}">{row.avg_overall}</div>
                    <div class="score-dims">
                        <div class="dim-item"><span style="font-size:0.65rem;color:#555">ACC</span><span class="dim-value">{row.avg_accuracy}</span></div>
                        <div class="dim-item"><span style="font-size:0.65rem;color:#555">CLR</span><span class="dim-value">{row.avg_clarity}</span></div>
                        <div class="dim-item"><span style="font-size:0.65rem;color:#555">RSN</span><span class="dim-value">{row.avg_reasoning}</span></div>
                    </div>
                    <div class="latency-badge">avg ⏱ {row.avg_latency}s</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Win rate by category
        st.markdown('<div class="section-header">Win Rate by Category</div>', unsafe_allow_html=True)
        win_rows = []
        for r in results:
            best = max(r["results"].keys(), key=lambda m: r["results"][m]["scores"].get("overall", 0))
            win_rows.append({"category": r["category"], "winner": best})
        win_df = pd.DataFrame(win_rows)
        win_counts = win_df.groupby(["category", "winner"]).size().reset_index(name="wins")
        st.plotly_chart(win_rate_chart(win_counts), use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Per question breakdown
        st.markdown('<div class="section-header">Per Question Breakdown</div>', unsafe_allow_html=True)
        for r in results:
            with st.expander(f"[{r['question_id'].upper()}] {r['question'][:75]}..."):
                qcols = st.columns(3)
                for i, (model, data) in enumerate(r["results"].items()):
                    color = MODEL_COLORS.get(model, "#fff")
                    clean = strip_think(data["answer"])
                    with qcols[i]:
                        st.markdown(f'<span style="color:{color};font-weight:700">{model}</span>', unsafe_allow_html=True)
                        st.markdown(f'**{data["scores"].get("overall",0)}/10** — *{data["scores"].get("feedback","")}*')
                        st.caption(clean[:250] + "...")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Export
        st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="llm_arena_results.csv",
            mime="text/csv",
            use_container_width=True
        )
