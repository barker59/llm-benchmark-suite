import os, csv, json, time, subprocess, statistics as stats
import datetime as dt
import ollama
from ollama._types import ResponseError

# -------------------- CONFIG --------------------
MODELS = ["llama3.1:8b"]      # or ["llama3.1:8b","phi3:instruct"]
PROMPTS_FILE = "prompts.txt"  # one prompt per line
OUTDIR = "bench_out"
REPEAT = 3                    # repeats per (model, prompt) to average
USE_GPU = True                # True = GPU, False = CPU
GPU_ID = 0                    # which GPU to use if USE_GPU
SEED = 42                     # None for nondeterministic
MAX_NEW_TOKENS = 150          # sets options.num_predict; None to leave default
PROMPT_PREVIEW_CHARS = 100
# ------------------------------------------------

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def ns_to_s(ns): return (ns or 0) / 1e9

def preview(text, n=100):
    t = text.replace("\n", " ")
    return (t[:n-3] + "...") if len(t) > n else t

def ensure_model(model: str):
    have = {m["model"] for m in ollama.list().get("models", [])}
    if model not in have:
        print(f"Pulling model '{model}'…")
        ollama.pull(model=model)

def get_gpu_info(gpu_id=0):
    try:
        name = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=name",
             "--format=csv,noheader"], text=True).strip()
        mem = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True).strip()
        return name, int(mem)
    except Exception:
        return "N/A", 0

def bench_once(model, prompt, options):
    t0 = time.perf_counter()
    try:
        r = ollama.generate(model=model, prompt=prompt, options=options)
    except ResponseError as e:
        if "not found" in str(e).lower():
            ensure_model(model)
            r = ollama.generate(model=model, prompt=prompt, options=options)
        else:
            raise
    wall = time.perf_counter() - t0
    d = r.model_dump()
    # Primary timing metric = generation (eval) time
    gen_s = ns_to_s(d.get("eval_duration", 0))
    # Fallback to wall if eval_duration missing
    if not gen_s:
        gen_s = wall
    return d, gen_s, wall

def main():
    prompts = read_lines(PROMPTS_FILE)
    os.makedirs(OUTDIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_csv = os.path.join(OUTDIR, f"summary_{stamp}.csv")
    outputs_jsonl = os.path.join(OUTDIR, f"outputs_{stamp}.jsonl")

    device = "gpu" if USE_GPU else "cpu"
    gpu_name, _ = get_gpu_info(GPU_ID)  # just once for the header; per-row we also sample mem

    # Write CSV header matching your desired format
    fields = [
        "model","device","prompt_number","prompt_preview",
        "avg_generation_time","std_deviation","max_new_tokens",
        "memory_used_mb","gpu_name","timestamp"
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as sfile, \
         open(outputs_jsonl, "w", encoding="utf-8") as ofile:
        sw = csv.DictWriter(sfile, fieldnames=fields)
        sw.writeheader()

        for model in MODELS:
            try: ensure_model(model)
            except Exception as e: print(f"Warning pulling {model}: {e}")

            for idx, prompt in enumerate(prompts, 1):
                gens = []
                responses = []
                mem_samples = []

                for _ in range(REPEAT):
                    opts = {"num_gpu": 999, "main_gpu": GPU_ID} if USE_GPU else {"num_gpu": 0}
                    if SEED is not None: opts["seed"] = SEED
                    if MAX_NEW_TOKENS is not None: opts["num_predict"] = MAX_NEW_TOKENS

                    # (Optional) sample memory right after warm load begins is tricky;
                    # we just record post-generation snapshot for simplicity
                    d, gen_s, _ = bench_once(model, prompt, opts)
                    _, mem_used = get_gpu_info(GPU_ID) if USE_GPU else ("N/A", 0)

                    gens.append(gen_s)
                    mem_samples.append(mem_used)
                    responses.append({
                        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model,
                        "device": device,
                        "gpu_id": GPU_ID if USE_GPU else None,
                        "options": opts,
                        "prompt_number": idx,
                        "prompt": prompt,
                        "response": d.get("response",""),
                        "raw": {k:v for k,v in d.items() if k != "response"},
                    })

                avg_gen = float(sum(gens)/len(gens))
                std_gen = float(stats.stdev(gens)) if len(gens) > 1 else 0.0
                mem_used_mb = max(mem_samples) if mem_samples else 0

                row = {
                    "model": model,
                    "device": device,
                    "prompt_number": idx,
                    "prompt_preview": preview(prompt, PROMPT_PREVIEW_CHARS),
                    "avg_generation_time": avg_gen,
                    "std_deviation": std_gen,
                    "max_new_tokens": MAX_NEW_TOKENS or 0,
                    "memory_used_mb": mem_used_mb if USE_GPU else 0,
                    "gpu_name": gpu_name,
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                sw.writerow(row)

                # write each repeat to outputs JSONL
                for rec in responses:
                    ofile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Summary CSV: {summary_csv}\n✅ Outputs JSONL: {outputs_jsonl}")

if __name__ == "__main__":
    main()
