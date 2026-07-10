import os
import json
import re
import requests
from llama_cpp import Llama

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
MODEL_PATH = "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

print("Loading local Llama-3.2 model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=2,  
    verbose=False,
)
print("Model loaded.\n")
HARD_TASK_PROMPTS = {
    "NER": "You are a strict data extraction system. Output ONLY a raw, valid JSON array of strings. Do not use markdown formatting (no ```json), and do not add conversational text.",
    "DEBUG": "You are an expert Python debugger. Output ONLY the corrected Python function. Do not include explanations, and do not use markdown code blocks (no ```python).",
    "LOGIC": "You are an expert logician. Solve the puzzle accurately. Your final line MUST be exactly written as: 'Answer: [Your final short answer]'.",
    "CODE_GEN": "You are a senior developer. Output ONLY raw Python code. No explanations, no markdown code blocks."
}

def dev4_classify_and_prompt(task_prompt):
    p_lower = task_prompt.lower()
    if "named entities" in p_lower or "json list" in p_lower: return HARD_TASK_PROMPTS["NER"], "NER"
    elif "bug" in p_lower or "debugger" in p_lower: return HARD_TASK_PROMPTS["DEBUG"], "DEBUG"
    elif "friends" in p_lower or "puzzle" in p_lower: return HARD_TASK_PROMPTS["LOGIC"], "LOGIC"
    else: return HARD_TASK_PROMPTS["CODE_GEN"], "CODE_GEN"

def sanitize_output(raw_text, task_type):
    if not raw_text: return ""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip("`").strip()
    
    if task_type == "NER":
        if not cleaned.startswith("["): cleaned = "[" + cleaned
        if not cleaned.endswith("]"): cleaned = cleaned + "]"
    elif task_type == "LOGIC":
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        for line in reversed(lines):
            if line.startswith("Answer:"): return line
    return cleaned

def call_fireworks_api(system_prompt, user_prompt):
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models = os.environ.get("ALLOWED_MODELS", "llama-v3-8b-instruct")
    
    target_model = allowed_models.split(",")[0].strip()
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
        "temperature": 0.0 
    }
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"API Fallback Error ({target_model}): {str(e)}"

def route_and_solve(task_prompt):
    prompt_lower = task_prompt.lower()
    
    hard_keywords = ["def ", "bug:", "named entities", "json list", "python function", "each own a different"]
    
    if any(keyword in prompt_lower for keyword in hard_keywords):
        print("   -> [ROUTED TO API] Trapdoor detected.")
        sys_prompt, task_type = dev4_classify_and_prompt(task_prompt)
        raw_answer = call_fireworks_api(sys_prompt, task_prompt)
        return sanitize_output(raw_answer, task_type)
        
    print("   -> [ROUTED LOCALLY] Running Llama-3.2 (0 Tokens)...")
    try:
        formatted_prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are a helpful hackathon assistant. Provide a highly accurate, direct, and concise answer in one sentence.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{task_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        
        output = llm(formatted_prompt, max_tokens=100, temperature=0.1)
        raw_answer = output["choices"][0]["text"].strip()
        return sanitize_output(raw_answer, "GENERAL")
    except Exception as e:
        print(f"   -> [LOCAL CRASH] {str(e)}. Fallback to API...")
        sys_prompt, task_type = dev4_classify_and_prompt(task_prompt)
        raw_answer = call_fireworks_api(sys_prompt, task_prompt)
        return sanitize_output(raw_answer, task_type)

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
        answer = route_and_solve(prompt)
        
        results.append({
            "task_id": task_id,
            "question": prompt,
            "answer": answer
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nExecution complete. Results written to /output/results.json")

if __name__ == "__main__":
    main()