import os
import json
import time
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "llm-arena"

MODELS = {
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "qwen3.6-27b":  "qwen/qwen3.6-27b",
    "gpt-oss-20b":  "openai/gpt-oss-20b"
}

JUDGE_MODEL = "llama-3.1-8b-instant"

with open("questions.json", "r") as f:
    QUESTION_BANK = json.load(f)

def get_questions(category=None):
    if category:
        return QUESTION_BANK[category]
    return [q for cat in QUESTION_BANK.values() for q in cat]

def run_model(model_key: str, question: str) -> dict:
    model_id = MODELS[model_key]
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely."},
                {"role": "user", "content": question}
            ],
            max_tokens=512,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        # Strip think blocks from model answers too
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        return {
            "model": model_key,
            "answer": answer,
            "latency_s": round(time.time() - start, 2),
            "tokens": response.usage.completion_tokens,
            "error": None
        }
    except Exception as e:
        print(f"  MODEL ERROR ({model_key}): {e}")
        return {"model": model_key, "answer": "", "latency_s": None, "tokens": 0, "error": str(e)}

def run_all_models(question: str) -> dict:
    results = {}
    for model_key in MODELS:
        print(f"  Running {model_key}...")
        results[model_key] = run_model(model_key, question)
    return results

def judge_response(question: str, category: str, response: str) -> dict:
    if not response:
        return {"accuracy": 0, "clarity": 0, "reasoning": 0, "overall": 0, "feedback": "No response."}

    prompt = f"""Score this AI response. Return ONLY a JSON object, nothing else. No thinking, no explanation.

Question: {question}
Response: {response}

Return ONLY this JSON (replace numbers with your scores):
{{"accuracy": 8, "clarity": 7, "reasoning": 9, "overall": 8.0, "feedback": "One sentence here."}}"""

    try:
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0
        )
        raw = result.choices[0].message.content.strip()

        # Strip <think>...</think> blocks
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        # Strip markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()

        print(f"    JUDGE (after strip): {raw[:100]}")

        # Find JSON object
        json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            scores["overall"] = round(
                (scores.get("accuracy", 0) + scores.get("clarity", 0) + scores.get("reasoning", 0)) / 3, 1
            )
            return scores

        print(f"    PARSE FAILED. Full raw: {raw[:300]}")
        return {"accuracy": 0, "clarity": 0, "reasoning": 0, "overall": 0, "feedback": "Parse failed"}
    except Exception as e:
        print(f"    JUDGE EXCEPTION: {e}")
        return {"accuracy": 0, "clarity": 0, "reasoning": 0, "overall": 0, "feedback": str(e)}

def evaluate_question(question_obj: dict) -> dict:
    question = question_obj["question"]
    category = question_obj["category"]
    qid = question_obj["id"]

    print(f"\n{'='*60}")
    print(f"Q [{qid}]: {question[:80]}...")
    print(f"{'='*60}")

    responses = run_all_models(question)

    print("  Judging responses...")
    results = {}
    for model_key, resp_data in responses.items():
        scores = judge_response(question, category, resp_data["answer"])
        results[model_key] = {**resp_data, "scores": scores}
        print(f"    {model_key}: overall={scores.get('overall', 'N/A')}")

    return {"question_id": qid, "question": question, "category": category, "results": results}

def run_benchmark(category=None, max_questions=None) -> list:
    questions = get_questions(category)
    if max_questions:
        questions = questions[:max_questions]
    all_results = []
    for q in questions:
        all_results.append(evaluate_question(q))
        time.sleep(1)
    return all_results

def save_results(results: list, filename="results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def print_leaderboard(results: list):
    scores = {m: {"total": 0, "wins": 0, "count": 0} for m in MODELS}
    for r in results:
        best_score, best_model = -1, None
        for model, data in r["results"].items():
            overall = data["scores"].get("overall", 0)
            scores[model]["total"] += overall
            scores[model]["count"] += 1
            if overall > best_score:
                best_score = overall
                best_model = model
        if best_model:
            scores[best_model]["wins"] += 1

    print("\n" + "="*50)
    print("LEADERBOARD")
    print("="*50)
    for rank, (model, data) in enumerate(
        sorted(scores.items(), key=lambda x: x[1]["total"] / max(x[1]["count"], 1), reverse=True), 1
    ):
        avg = round(data["total"] / max(data["count"], 1), 2)
        print(f"{rank}. {model:<20} Avg: {avg}/10   Wins: {data['wins']}")

if __name__ == "__main__":
    print("LLM Arena — Starting benchmark...\n")
    results = run_benchmark()  # all 30 questions

    print_leaderboard(results)
    save_results(results, "results.json")
    print("\nDay 1 done. Run app.py tomorrow for the dashboard.")
