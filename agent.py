import os
import json
import re
import time
import requests
from llama_cpp import Llama

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
MODEL_PATH = "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

print("Loading local Llama-3.2 model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1536,
    n_threads=2,
    n_batch=256,
    verbose=False,
)
print("Model loaded.\n")

HARD_TASK_PROMPTS = {
    "NER": "You are a strict data extraction system. Output ONLY a raw, valid JSON array of strings. Do not use markdown formatting (no ```json), and do not add conversational text.",
    "DEBUG": "You are an expert code debugger. Output ONLY the corrected function. Do not include explanations, and do not use markdown code blocks.",
    "LOGIC": "You are an expert logician and mathematician. Solve the problem accurately. Your final line MUST be exactly written as: 'Answer: [Your final short answer]'.",
    "CODE_GEN": "You are a senior developer. Output ONLY raw code. No explanations, no markdown code blocks.",
    "SENTIMENT": "You are a sentiment classification system. Classify the sentiment as Positive, Negative, or Neutral, then give a one-sentence justification. Format exactly as: 'Sentiment: <label>. Justification: <reason>'.",
    "SUMMARY": "You are a precise summarization system. Follow the exact length or format constraint given in the instruction. Output ONLY the summary text, no preamble.",
    "FACTUAL": "You are a knowledgeable assistant. Give an accurate, concise, direct answer. No conversational preamble, no hedging.",
}

MAX_API_TOKENS = 200

def analyze_intent(task_prompt):
    p_lower = task_prompt.lower()

    if any(w in p_lower for w in ["def ", "function", "code", "script", "snippet"]) and \
       any(w in p_lower for w in ["bug", "fix", "error", "wrong", "fail", "incorrect", "issue"]):
        return True, "DEBUG"

    if any(w in p_lower for w in ["write a", "create a", "implement", "generate", "develop"]) and \
       any(w in p_lower for w in ["function", "script", "component", "query", "bash", "sql", "program"]):
        return True, "CODE_GEN"

    action_match = any(word in p_lower for word in ["extract", "identify", "list all", "find all"])
    entity_match = any(word in p_lower for word in ["named entities", "json", "mentioned in", "from the following", "from this", "in the following"])
    if action_match and entity_match:
        return True, "NER"

    has_digits = any(char.isdigit() for char in task_prompt)
    math_words = ["how many", "percentage", "calculate", "remain", "total", "derivative", "solve", "evaluate", "value of", "probability", "equation"]
    if has_digits and any(word in p_lower for word in math_words):
        return True, "LOGIC"

    logic_words = ["friends", "lies", "truth", "puzzle", "robot", "who is", "order", "assume", "statements", "sequence"]
    if any(word in p_lower for word in logic_words):
        return True, "LOGIC"

    sentiment_words = ["sentiment", "positive or negative", "classify the tone", "classify the emotion", "how does this review", "opinion expressed"]
    if any(word in p_lower for word in sentiment_words):
        return True, "SENTIMENT"

    summary_words = ["summarize", "summarise", "condense", "in one sentence", "in exactly", "tl;dr", "shorten", "brief summary"]
    if any(word in p_lower for word in summary_words):
        return True, "SUMMARY"

    return True, "FACTUAL"


def sanitize_output(raw_text, task_type):
    if not raw_text:
        return ""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip("`").strip()

    if task_type == "NER":
        candidate = cleaned
        if not candidate.startswith("["):
            candidate = "[" + candidate
        if not candidate.endswith("]"):
            candidate = candidate + "]"
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return candidate 
    elif task_type == "LOGIC":
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        for line in reversed(lines):
            if line.startswith("Answer:"):
                return line.replace("Answer:", "").strip()
        return cleaned
    return cleaned


def call_fireworks_api(system_prompt, user_prompt, task_type, retries=1):
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models = os.environ.get("ALLOWED_MODELS", "llama-v3-8b-instruct")

    target_model = allowed_models.split(",").strip()
    if "accounts/fireworks/models/" not in target_model:
        target_model = f"accounts/fireworks/models/{target_model}"

    if not api_key:
        return "Error: FIREWORKS_API_KEY environment variable is not set."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": MAX_API_TOKENS,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            content = response.json()["choices"]["message"]["content"].strip()

            if task_type == "NER":
                try:
                    json.loads(sanitize_output(content, "NER"))
                except Exception:
                    if attempt < retries:
                        time.sleep(1)
                        continue
            return content
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1)
                continue
    return f"API Fallback Error ({target_model}): {str(last_err)}"


def route_and_solve(task_prompt):
    route_to_api, task_type = analyze_intent(task_prompt)

    if route_to_api:
        print(f" -> [ROUTED TO API] Signature matched: {task_type}")
        sys_prompt = HARD_TASK_PROMPTS.get(task_type, HARD_TASK_PROMPTS["FACTUAL"])
        raw_answer = call_fireworks_api(sys_prompt, task_prompt, task_type)
        return sanitize_output(raw_answer, task_type)

    print(" -> [ROUTED LOCALLY] Safe task detected (0 Tokens)...")
    try:
        formatted_prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are a helpful hackathon assistant. Provide a highly accurate, direct, and concise answer without any conversational preamble.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n{task_prompt}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        output = llm(
            formatted_prompt,
            max_tokens=150,
            temperature=0.1,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        raw_answer = output["choices"]["text"].strip()
        return sanitize_output(raw_answer, "GENERAL")
    except Exception as e:
        print(f" -> [LOCAL CRASH] {str(e)}. Fallback to API...")
        raw_answer = call_fireworks_api(HARD_TASK_PROMPTS["FACTUAL"], task_prompt, "FACTUAL")
        return sanitize_output(raw_answer, "GENERAL")


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    results = []
    print("Processing tasks...\n")
    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        prompt = task.get("prompt", "")

        print(f"[{i}/{len(tasks)}] Processing Task: {task_id}")
        try:
            answer = route_and_solve(prompt)
        except Exception as e:
            answer = f"Error: {str(e)}"

        results.append({
            "task_id": task_id,
            "answer": answer
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\nExecution complete. Results written to /output/results.json")


if __name__ == "__main__":
    main()