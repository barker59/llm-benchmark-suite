import os, csv, json, time, datetime as dt, ollama
from ollama._types import ResponseError

# ---------- CONFIG ----------
MODELS = ["llama3"]       # or e.g. ["llama3.1:8b","phi3:instruct"]
PROMPTS_FILE = "prompts.txt"
OUTDIR = "bench_out"
USE_GPU = True
GPU_ID = 0
REPEAT = 1
SEED = None
# ---------------------------

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def ns_to_s(ns): return (ns or 0)/1e9

def ensure_model(model: str):
    have = {m["model"] for m in ollama.list().get("models", [])}
    if model not in have:
        print(f"Pulling model '{model}'…")
        ollama.pull(model=model)

def bench_one(model: str, prompt: str, opts: dict):
    t0 = time.perf_counter()
    try:
        r = ollama.generate(model=model, prompt=prompt, options=opts)
    except ResponseError as e:
        if "not found" in str(e).lower():
            ensure_model(model)
            r = ollama.generate(model=model, prompt=prompt, options=opts)
        else:
            raise
    wall = time.perf_counter() - t0
    d = r.model_dump()
    pe_cnt, ge_cnt = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
    pe_dur, ge_dur = d.get("prompt_eval_duration", 0), d.get("eval_duration", 0)
    tot_dur = d.get("total_duration", 0)
    total_tokens = (pe_cnt or 0) + (ge_cnt or 0)
    gen_tps = (ge_cnt / ns_to_s(ge_dur)) if ge_dur else 0.0
    total_tps = (total_tokens / ns_to_s(tot_dur)) if tot_dur else 0.0
    return d, wall, total_tokens, gen_tps, total_tps

def main():
    prompts = read_lines(PROMPTS_FILE)
    os.makedirs(OUTDIR, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    metrics_path = os.path.join(OUTDIR, f"metrics_{stamp.replace(':','-')}.csv")
    outputs_path = os.path.join(OUTDIR, f"outputs_{stamp.replace(':','-')}.jsonl")

    fields = ["ts","model","run","prompt_id","gpu","gpu_id",
              "prompt_chars","response_chars",
              "prompt_tokens","gen_tokens","total_tokens",
              "eval_s","prompt_eval_s","total_s","wall_s",
              "gen_tok_per_s","total_tok_per_s"]

    with open(metrics_path, "w", newline="", encoding="utf-8") as mfile, \
         open(outputs_path, "w", encoding="utf-8") as ofile:
        writer = csv.DictWriter(mfile, fieldnames=fields); writer.writeheader()

        for model in MODELS:
            # preflight: pull if missing
            try: ensure_model(model)
            except Exception as e: print(f"Warning pulling {model}: {e}")

            for run in range(1, REPEAT+1):
                for pid, prompt in enumerate(prompts, 1):
                    opts = {"num_gpu": 999, "main_gpu": GPU_ID} if USE_GPU else {"num_gpu": 0}
                    if SEED is not None: opts["seed"] = SEED

                    d, wall, total_tokens, gen_tps, total_tps = bench_one(model, prompt, opts)

                    row = {
                        "ts": d.get("created_at", dt.datetime.now(dt.timezone.utc).isoformat()),
                        "model": model, "run": run, "prompt_id": pid,
                        "gpu": USE_GPU, "gpu_id": GPU_ID,
                        "prompt_chars": len(prompt),
                        "response_chars": len(d.get("response","")),
                        "prompt_tokens": d.get("prompt_eval_count",0),
                        "gen_tokens": d.get("eval_count",0),
                        "total_tokens": total_tokens,
                        "eval_s": f"{ns_to_s(d.get('eval_duration',0)):.4f}",
                        "prompt_eval_s": f"{ns_to_s(d.get('prompt_eval_duration',0)):.4f}",
                        "total_s": f"{ns_to_s(d.get('total_duration',0)):.4f}",
                        "wall_s": f"{wall:.4f}",
                        "gen_tok_per_s": f"{gen_tps:.2f}",
                        "total_tok_per_s": f"{total_tps:.2f}",
                    }
                    writer.writerow(row)

                    # full output line
                    out = {
                        "timestamp": row["ts"],
                        "model": model,
                        "run": run,
                        "prompt_id": pid,
                        "gpu": USE_GPU,
                        "gpu_id": GPU_ID,
                        "options": opts,
                        "prompt": prompt,
                        "response": d.get("response",""),
                        "raw": {k:v for k,v in d.items() if k != "response"},
                    }
                    ofile.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"✅ Metrics: {metrics_path}\n✅ Outputs: {outputs_path}")

if __name__ == "__main__":
    main()
