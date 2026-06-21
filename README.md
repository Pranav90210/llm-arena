# ⚔️ LLM Arena

A benchmarking dashboard that pits LLMs against each other and uses a 4th model as an automated judge.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-API-orange?style=flat-square)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-green?style=flat-square)

## What it does

Send the same question to 3 LLMs simultaneously. A 4th model scores each response on **accuracy**, **clarity**, and **reasoning** using the LLM-as-a-Judge pattern. Results are visualized in a Streamlit dashboard with radar charts, leaderboards, and per-question breakdowns.

## Models

| Model | Role |
|-------|------|
| `llama-3.1-8b-instant` | Contestant + Judge |
| `qwen/qwen3.6-27b` | Contestant |
| `openai/gpt-oss-20b` | Contestant |

All inference via [Groq](https://console.groq.com) — fast, free tier available.

## Features

- **Single question mode** — run one question live, see side-by-side responses and scores
- **Batch benchmark mode** — run up to 30 questions, get full leaderboard and win rates
- **LLM-as-a-Judge** — automated scoring on 3 dimensions (1-10 each)
- **Radar charts** — per-question score breakdown per model
- **Win rate by category** — factual vs reasoning vs creative performance
- **CSV export** — download full results
- **LangSmith tracing** — every API call logged and traceable

## Question bank

30 questions across 3 categories:
- **Factual** — ML concepts, statistics, deep learning fundamentals
- **Reasoning** — logic puzzles, math word problems, debugging scenarios
- **Creative** — analogies, story prompts, unusual applications

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-arena.git
cd llm-arena

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.template .env
# Fill in GROQ_API_KEY and LANGCHAIN_API_KEY
```

Get your keys:
- Groq: [console.groq.com](https://console.groq.com) — free
- LangSmith: [smith.langchain.com](https://smith.langchain.com) — free tier

## Usage

**Run the benchmark pipeline first:**
```bash
python pipeline.py
```
This runs 3 test questions and saves `results.json`.

For full 30-question benchmark, edit the bottom of `pipeline.py`:
```python
results = run_benchmark()  # all 30 questions
```

**Launch the dashboard:**
```bash
streamlit run app.py
```

## Project structure

```
llm-arena/
├── pipeline.py       # Core benchmark pipeline — model runner + LLM judge
├── app.py            # Streamlit dashboard
├── questions.json    # 30 benchmark questions across 3 categories
├── requirements.txt  # Dependencies
└── .env.template     # API key template
```

## Key concept: LLM-as-a-Judge

Instead of hardcoded metrics, a separate LLM evaluates each response using a structured rubric. This is how production AI teams run large-scale evals without human reviewers on every output.

Known limitation: **self-preference bias** — judge models subtly favor responses that match their own style. This is an open research problem in LLM evaluation.

## Built with

- [Groq](https://groq.com) — LLM inference
- [LangSmith](https://smith.langchain.com) — tracing and observability
- [Streamlit](https://streamlit.io) — dashboard
- [Plotly](https://plotly.com) — charts

---

Built by [Pranav Kumar S A]([https://www.linkedin.com/in/pranavkumarsa/])
