import llm_benchmark
import os
from llm_benchmark.run_benchmark import run_benchmark

# Get package directory
pkg_dir = os.path.dirname(llm_benchmark.__file__)

# Use the models file for 16GB (which includes deepseek-r1:8b)
models_file = os.path.join(pkg_dir, 'data', 'benchmark_models_16gb_ram.yml')
benchmark_file = os.path.join(pkg_dir, 'data', 'benchmark1.yml')

print("Starting LLM Benchmark...")
print(f"Models file: {models_file}")
print(f"Benchmark file: {benchmark_file}")
print(f"Benchmark type: benchmark")

try:
    # Run the benchmark
    result = run_benchmark(models_file, benchmark_file, 'benchmark')
    
    print("\n--- Benchmark Results ---")
    if result:
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print("No results returned - this may indicate that models need to be downloaded")
        print("The benchmark may be running in the background to download required models")
        
except Exception as e:
    print(f"Error running benchmark: {e}")
    import traceback
    traceback.print_exc()
