import sys, ollama

m = sys.argv[1] if len(sys.argv)>1 else 'llama3'
prompts = (open(sys.argv[2]).read().splitlines() if len(sys.argv)>2
           else ['Say hi in 1 sentence','What is 2+2?','Write a rain haiku'])
for i, p in enumerate(prompts, 1):
#   r = ollama.generate(model=m, prompt=p)
    r = ollama.generate(
    model=m,
    prompt=p,
    options={
        "num_gpu": 999,   # push as many layers as possible to GPU VRAM
        "main_gpu": 0     # pick which NVIDIA GPU to use (0-based)
    },
)
    print(f'### Prompt {i}: {p}\n{r.get("response","").strip()}\n')