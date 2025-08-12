import llm_benchmark
import os
import yaml

# Get package directory
pkg_dir = os.path.dirname(llm_benchmark.__file__)
models_file = os.path.join(pkg_dir, 'data', 'benchmark_models_16gb_ram.yml')
benchmark_file = os.path.join(pkg_dir, 'data', 'benchmark1.yml')

print("Package directory:", pkg_dir)
print("Models file:", models_file)
print("Benchmark file:", benchmark_file)

# Read models file
print("\n--- Models Configuration ---")
try:
    with open(models_file, 'r') as f:
        models_content = yaml.safe_load(f)
    print(models_content)
except Exception as e:
    print(f"Error reading models file: {e}")

# Read benchmark file
print("\n--- Benchmark Configuration ---")
try:
    with open(benchmark_file, 'r') as f:
        benchmark_content = yaml.safe_load(f)
    print(benchmark_content)
except Exception as e:
    print(f"Error reading benchmark file: {e}")

# Check available models in ollama
print("\n--- Available Ollama Models ---")
try:
    import subprocess
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"Error checking ollama models: {e}")
