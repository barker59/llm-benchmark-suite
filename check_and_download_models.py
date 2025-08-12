import llm_benchmark
import os
from llm_benchmark.check_models import check_models

# Get package directory
pkg_dir = os.path.dirname(llm_benchmark.__file__)
models_file = os.path.join(pkg_dir, 'data', 'benchmark_models_16gb_ram.yml')

print("Checking required models...")
print(f"Models file: {models_file}")

try:
    # Check which models are available and which need to be downloaded
    result = check_models(models_file)
    print("\n--- Model Check Results ---")
    
    if result:
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print("Model check completed")
        
except Exception as e:
    print(f"Error checking models: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Current Ollama Models ---")
import subprocess
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
    print(result.stdout)
except Exception as e:
    print(f"Error listing ollama models: {e}")
