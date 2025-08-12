import argparse, os, time, ollama

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--prompts", help="path to prompts.txt")
p.add_argument("--gpu", action="store_true", help="use GPU (else CPU)")
p.add_argument("--gpu-id", type=int, default=0)
args = p.parse_args()

prompts = (open(args.prompts).read().splitlines()
           if args.prompts and os.path.exists(args.prompts)
           else ['Say hi in 1 sentence','What is 2+2?','Write a rain haiku'])

opts = {"num_gpu": 999, "main_gpu": args.gpu_id} if args.gpu else {"num_gpu": 0}

t0 = time.perf_counter()
for i, prompt in enumerate(prompts, 1):
    t1 = time.perf_counter()
    r = ollama.generate(model=args.model, prompt=prompt, options=opts)
    dt = time.perf_counter() - t1
    print(f"\n### {i}. {prompt}\n{r.get('response','').strip()}\n-- {dt:.2f}s")
print(f"\nTotal time: {time.perf_counter()-t0:.2f}s  | Mode: {'GPU' if args.gpu else 'CPU'}")
