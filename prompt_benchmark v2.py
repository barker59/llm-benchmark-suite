import time, psutil, pandas as pd, os, gc, requests, argparse, subprocess
from datetime import datetime

# Try to import GPUtil, but don't fail if it's not available
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU detection will be disabled.")

class PromptBenchmark:
    def __init__(self, prompts_file="prompts.txt", models_file="models.txt"):
        self.prompts_file = prompts_file
        self.models_file = models_file
        self.results = []
        self.prompts = []
        self.models = []
        self.run_timestamp = None

    # --- IO helpers ---
    def load_prompts(self):
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.prompts = [p.strip() for p in content.split('\n\n') if p.strip()]
            print(f"Loaded {len(self.prompts)} prompts from {self.prompts_file}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {self.prompts_file}")
            return False
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return False

    def load_models(self):
        try:
            with open(self.models_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.models = [m.strip() for m in content.split('\n') if m.strip()]
            print(f"Loaded {len(self.models)} models from {self.models_file}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {self.models_file}")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    # --- System/GPU info ---
    def _safe_gpu_detection(self):
        """Safely detect GPUs without failing on nvidia-smi errors."""
        if not GPUTIL_AVAILABLE:
            return []
        try:
            gpus = GPUtil.getGPUs()
            return gpus if gpus else []
        except Exception as e:
            # Common errors: nvidia-smi not found, no NVIDIA drivers, etc.
            return []
    
    def has_gpu_support(self):
        """Check if GPU support is available without failing."""
        return bool(self._safe_gpu_detection())
    
    def get_gpu_name(self):
        """Safely get GPU name without failing."""
        try:
            gpu_info = self.get_gpu_info()
            return gpu_info['name'] if gpu_info else "N/A"
        except Exception:
            return "N/A"
    
    def get_gpu_info(self):
        try:
            gpus = self._safe_gpu_detection()
            if gpus:
                gpu = gpus[0]
                return {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'load': gpu.load * 100
                }
        except Exception:
            pass
        return None

    def get_system_memory(self):
        mem = psutil.virtual_memory()
        return {'total': mem.total, 'available': mem.available, 'used': mem.used, 'percent': mem.percent}

    # --- Ollama helpers ---
    def _ollama_base(self):
        return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    def validate_model_exists(self, tag: str):
        """Check if a model exists locally or can be pulled from Ollama registry."""
        try:
            # First check if model exists locally
            r = requests.get(f"{self._ollama_base()}/api/tags", timeout=30)
            r.raise_for_status()
            local_tags = {m.get("name") or m.get("model") for m in r.json().get("models", [])}
            
            if tag in local_tags:
                print(f"✓ Model '{tag}' found locally")
                return True
                
            # If not local, try to pull it
            print(f"Model '{tag}' not found locally. Attempting to pull...")
            return self._pull_model(tag)
            
        except requests.exceptions.ConnectionError:
            print(f"✗ Error: Cannot connect to Ollama service at {self._ollama_base()}")
            print("  Make sure Ollama is running: 'ollama serve'")
            return False
        except requests.exceptions.Timeout:
            print(f"✗ Error: Timeout connecting to Ollama service")
            return False
        except Exception as e:
            print(f"✗ Error checking model '{tag}': {e}")
            return False

    def _pull_model(self, tag: str):
        """Attempt to pull a model from Ollama registry with comprehensive error handling."""
        try:
            print(f"Pulling model '{tag}'...")
            result = subprocess.run(
                ["ollama", "pull", tag], 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=600  # 10 minute timeout for model pulls
            )
            print(f"✓ Successfully pulled model '{tag}'")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"✗ Error: Timeout while pulling model '{tag}' (exceeded 10 minutes)")
            print("  Large models may take longer. Try manually: ollama pull {tag}")
            return False
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            if "not found" in error_msg.lower() or "404" in error_msg:
                print(f"✗ Error: Model '{tag}' does not exist in Ollama registry")
                print(f"  Check available models at: https://ollama.ai/library")
                print(f"  Or list local models: ollama list")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                print(f"✗ Error: Network issue while pulling '{tag}'")
                print("  Check your internet connection and try again")
            else:
                print(f"✗ Error pulling model '{tag}': {error_msg}")
            return False
            
        except FileNotFoundError:
            print(f"✗ Error: 'ollama' command not found")
            print("  Please install Ollama: https://ollama.ai/")
            return False
            
        except Exception as e:
            print(f"✗ Unexpected error pulling model '{tag}': {e}")
            return False

    def ensure_model(self, tag: str):
        """Legacy method for backward compatibility - now uses validate_model_exists."""
        return self.validate_model_exists(tag)

    def generate_ollama(self, model_tag: str, prompt: str, max_new_tokens: int, use_gpu: bool):
        endpoint = f"{self._ollama_base()}/api/generate"
        options = {"temperature": 0.7, "num_predict": max_new_tokens}
        # CPU/GPU toggle via layer offload
        options["num_gpu"] = 999 if use_gpu else 0
        if use_gpu:
            options["main_gpu"] = 0

        payload = {"model": model_tag, "prompt": prompt, "stream": False, "options": options}
        start = time.time()
        
        try:
            resp = requests.post(endpoint, json=payload, timeout=600)
            
            # Handle specific HTTP error codes
            if resp.status_code == 404:
                print(f"Model '{model_tag}' not found. Attempting to pull...")
                if self.validate_model_exists(model_tag):
                    # Model was successfully pulled, retry generation
                    resp = requests.post(endpoint, json=payload, timeout=600)
                else:
                    error_msg = f"Model '{model_tag}' could not be found or pulled"
                    return {"response": "", "error": error_msg, "error_type": "model_not_found"}, time.time() - start
            
            elif resp.status_code == 400:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get("error", "Bad request")
                    if "model" in error_msg.lower():
                        print(f"✗ Model error for '{model_tag}': {error_msg}")
                        return {"response": "", "error": error_msg, "error_type": "model_error"}, time.time() - start
                    else:
                        print(f"✗ Request error: {error_msg}")
                        return {"response": "", "error": error_msg, "error_type": "request_error"}, time.time() - start
                except:
                    error_msg = f"Bad request (HTTP 400) for model '{model_tag}'"
                    return {"response": "", "error": error_msg, "error_type": "request_error"}, time.time() - start
            
            elif resp.status_code == 500:
                error_msg = f"Ollama server error when processing model '{model_tag}'"
                print(f"✗ Server error: {error_msg}")
                return {"response": "", "error": error_msg, "error_type": "server_error"}, time.time() - start
            
            resp.raise_for_status()
            data = resp.json()
            
            # Validate response structure
            if "response" not in data:
                error_msg = f"Invalid response format from model '{model_tag}'"
                print(f"✗ {error_msg}")
                return {"response": "", "error": error_msg, "error_type": "invalid_response"}, time.time() - start
            
            elapsed = time.time() - start
            return data, elapsed
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to Ollama service at {self._ollama_base()}"
            print(f"✗ Connection error: {error_msg}")
            print("  Make sure Ollama is running: 'ollama serve'")
            return {"response": "", "error": error_msg, "error_type": "connection_error"}, time.time() - start
            
        except requests.exceptions.Timeout:
            error_msg = f"Timeout generating text with model '{model_tag}' (exceeded 10 minutes)"
            print(f"✗ Timeout error: {error_msg}")
            return {"response": "", "error": error_msg, "error_type": "timeout_error"}, time.time() - start
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error with model '{model_tag}': {str(e)}"
            print(f"✗ Network error: {error_msg}")
            return {"response": "", "error": error_msg, "error_type": "network_error"}, time.time() - start
            
        except ValueError as e:
            error_msg = f"JSON parsing error from model '{model_tag}': {str(e)}"
            print(f"✗ Response parsing error: {error_msg}")
            return {"response": "", "error": error_msg, "error_type": "parsing_error"}, time.time() - start
            
        except Exception as e:
            error_msg = f"Unexpected error with model '{model_tag}': {str(e)}"
            print(f"✗ Unexpected error: {error_msg}")
            return {"response": "", "error": error_msg, "error_type": "unknown_error"}, time.time() - start

    # --- Benchmark core (Ollama for all models) ---
    def benchmark_ollama_device(self, model_tag: str, device: str, max_new_tokens: int = 100):
        device_results = []
        use_gpu = (device.lower() == "gpu")
        print(f"\n{'='*60}\nBenchmarking {model_tag} on {device.upper()} (Ollama)\n{'='*60}")

        # Warmup
        try:
            _ = self.generate_ollama(model_tag, "Warmup prompt.", 16, use_gpu)
        except Exception as e:
            print(f"Warmup warning: {e}")

        for i, prompt in enumerate(self.prompts):
            print(f"\nProcessing prompt {i+1}/{len(self.prompts)}...")
            print(f"Prompt preview: {prompt[:100]}...")
            times = []
            mem_deltas = []

            for itr in range(3):
                # Clear Python garbage (GPU cache mgmt is internal to Ollama)
                gc.collect()

                # Memory before
                before_used = 0
                if use_gpu:
                    info = self.get_gpu_info()
                    if info: before_used = info['memory_used']

                data, wall = self.generate_ollama(model_tag, prompt, max_new_tokens, use_gpu)

                # Check for generation errors
                if "error" in data:
                    error_type = data.get("error_type", "unknown")
                    error_msg = data.get("error", "Unknown error")
                    print(f"  ✗ Iteration {itr+1} failed: {error_msg}")
                    
                    # For critical errors (model not found, connection issues), skip remaining iterations
                    if error_type in ["model_not_found", "connection_error", "model_error"]:
                        print(f"  ⚠️  Critical error detected. Skipping remaining iterations for this prompt.")
                        # Return error result for this prompt
                        return [{
                            "model": model_tag,
                            "device": "gpu" if use_gpu else "cpu",
                            "prompt_number": i + 1,
                            "prompt_preview": (prompt[:100] + "...") if len(prompt) > 100 else prompt,
                            "avg_generation_time": -1,  # Indicate error
                            "std_deviation": -1,
                            "max_new_tokens": max_new_tokens,
                            "memory_used_mb": -1,
                            "gpu_name": self.get_gpu_name() if use_gpu else "N/A",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "error": error_msg,
                            "error_type": error_type
                        }]
                    
                    # For other errors, continue but don't add to valid measurements
                    continue

                # Memory after (only if generation was successful)
                after_used = 0
                if use_gpu:
                    info2 = self.get_gpu_info()
                    if info2: after_used = info2['memory_used']

                times.append(wall)
                mem_deltas.append(max(0, after_used - before_used))
                gen_text = data.get("response", "")
                print(f"  ✓ Iteration {itr+1}: {wall:.3f}s, Generated: {len(gen_text)} chars")

            # Check if we have any successful iterations
            if not times:
                print(f"  ⚠️  All iterations failed for prompt {i+1}. Skipping...")
                result = {
                    "model": model_tag,
                    "device": "gpu" if use_gpu else "cpu",
                    "prompt_number": i + 1,
                    "prompt_preview": (prompt[:100] + "...") if len(prompt) > 100 else prompt,
                    "avg_generation_time": -1,  # Indicate failure
                    "std_deviation": -1,
                    "max_new_tokens": max_new_tokens,
                    "memory_used_mb": -1,
                    "gpu_name": self.get_gpu_name() if use_gpu else "N/A",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": "All iterations failed",
                    "error_type": "all_iterations_failed"
                }
            else:
                avg_time = sum(times) / len(times)
                std_dev = (sum((x - avg_time) ** 2 for x in times) / len(times)) ** 0.5
                avg_mem = sum(mem_deltas) / len(mem_deltas) if any(mem_deltas) else 0
                gpu_name = self.get_gpu_name() if use_gpu else "N/A"

                result = {
                    "model": model_tag,
                    "device": "gpu" if use_gpu else "cpu",
                    "prompt_number": i + 1,
                    "prompt_preview": (prompt[:100] + "...") if len(prompt) > 100 else prompt,
                    "avg_generation_time": avg_time,
                    "std_deviation": std_dev,
                    "max_new_tokens": max_new_tokens,
                    "memory_used_mb": avg_mem,
                    "gpu_name": gpu_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                print(f"  ✓ Average time: {avg_time:.3f}s ± {std_dev:.3f}s ({len(times)}/3 successful iterations)")
                
            device_results.append(result)

        return device_results

    def validate_all_models(self):
        """Pre-validate all models before starting benchmark to catch issues early."""
        print("\n" + "="*80)
        print("VALIDATING MODELS")
        print("="*80)
        
        valid_models = []
        invalid_models = []
        
        for i, model_tag in enumerate(self.models, 1):
            print(f"\nValidating model {i}/{len(self.models)}: {model_tag}")
            if self.validate_model_exists(model_tag):
                valid_models.append(model_tag)
                print(f"✓ Model '{model_tag}' is ready")
            else:
                invalid_models.append(model_tag)
                print(f"✗ Model '{model_tag}' failed validation")
        
        if invalid_models:
            print(f"\n{'='*80}")
            print(f"MODEL VALIDATION SUMMARY")
            print(f"{'='*80}")
            print(f"✓ Valid models: {len(valid_models)}")
            print(f"✗ Invalid models: {len(invalid_models)}")
            
            if invalid_models:
                print(f"\nInvalid models that will be skipped:")
                for model in invalid_models:
                    print(f"  - {model}")
                    
                print(f"\nSuggestions:")
                print(f"  1. Check model names in '{self.models_file}'")
                print(f"  2. Browse available models: https://ollama.ai/library")
                print(f"  3. List local models: ollama list")
                print(f"  4. Pull models manually: ollama pull <model_name>")
        
        # Update the models list to only include valid models
        self.models = valid_models
        
        if not self.models:
            print(f"\n✗ No valid models found! Cannot proceed with benchmark.")
            return False
            
        print(f"\n✓ Proceeding with {len(self.models)} valid models")
        return True

    def run_comparison_benchmark(self, max_new_tokens=100, model_name_filter=None, skip_cpu=False, skip_gpu=False):
        if not self.load_prompts(): return
        if not self.load_models(): return

        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Starting Multi-Model Prompt Benchmark (Ollama only)")
        print("=" * 80)
        print(f"Run timestamp: {self.run_timestamp}")

        if model_name_filter:
            before = len(self.models)
            self.models = [m for m in self.models if model_name_filter.lower() in m.lower()]
            print(f"Applied model filter '{model_name_filter}': {before} -> {len(self.models)} models")

        # Pre-validate all models before starting benchmark
        if not self.validate_all_models():
            print("❌ Benchmark cancelled due to model validation failures")
            return []

        print(f"\n{'='*80}")
        print(f"BENCHMARK CONFIGURATION")
        print(f"{'='*80}")
        print(f"Models to test: {len(self.models)}")
        for i, m in enumerate(self.models, 1):
            print(f"  {i}. {m}")
        print(f"Number of prompts: {len(self.prompts)}")
        print(f"Max new tokens per generation: {max_new_tokens}")
        print(f"System Memory: {self.get_system_memory()['total'] / (1024**3):.2f} GB")

        has_gpu = self.has_gpu_support()
        gpu_count = len(self._safe_gpu_detection())
        print(f"GPUs detected: {gpu_count}")
        print(f"CPU benchmarks: {'❌ DISABLED' if skip_cpu else '✅ ENABLED'}")
        print(f"GPU benchmarks: {'❌ DISABLED' if skip_gpu else '✅ ENABLED' if has_gpu else '❌ NO GPU DETECTED'}")
        
        # Validate configuration
        if skip_cpu and skip_gpu:
            print("❌ Error: Both --skip-cpu and --skip-gpu flags used!")
            print("   At least one device type must be enabled.")
            return []
        
        if skip_cpu and not has_gpu:
            print("❌ Error: --skip-cpu flag used but no GPUs detected!")
            print("   Cannot run GPU-only benchmark without GPUs.")
            return []
        
        if skip_gpu and not has_gpu:
            print("ℹ️  --skip-gpu flag used but no GPUs detected anyway")
        
        if skip_cpu:
            print("🚀 GPU-only mode enabled (skipping CPU benchmarks)")
        elif skip_gpu:
            print("🚀 CPU-only mode enabled (skipping GPU benchmarks)")
        
        for model_idx, model_tag in enumerate(self.models, 1):
            print(f"\n{'='*100}\nBENCHMARKING MODEL {model_idx}/{len(self.models)}: {model_tag}\n{'='*100}")

            # CPU (unless skipped)
            if not skip_cpu:
                print(f"\n{'='*80}\nSTARTING CPU BENCHMARK FOR {model_tag}\n{'='*80}")
                cpu_res = self.benchmark_ollama_device(model_tag, "cpu", max_new_tokens)
                self.results.extend(cpu_res)
            else:
                print(f"\n⏭️  SKIPPING CPU BENCHMARK FOR {model_tag} (--skip-cpu flag)")

            # GPU (if present and not skipped)
            if has_gpu and not skip_gpu:
                print(f"\n{'='*80}\nSTARTING GPU BENCHMARK FOR {model_tag}\n{'='*80}")
                gpu_res = self.benchmark_ollama_device(model_tag, "gpu", max_new_tokens)
                self.results.extend(gpu_res)
            else:
                if skip_gpu:
                    print(f"\n⏭️  SKIPPING GPU BENCHMARK FOR {model_tag} (--skip-gpu flag)")
                elif not has_gpu:
                    if skip_cpu:
                        print(f"❌ No GPUs found for {model_tag} - cannot run GPU-only benchmark")
                    else:
                        print(f"⚠️  Skipping GPU benchmark for {model_tag} - No GPU support detected")

        return self.results

    def save_results(self, base_filename="multi_model_benchmark"):
        if not self.results:
            print("No results to save")
            return None
        timestamp = getattr(self, 'run_timestamp', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        filename = f"{base_filename}_{timestamp}.csv"
        save_path = r"C:\Users\barke\Desktop\Code"
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        try:
            df = pd.DataFrame(self.results)
            df['run_timestamp'] = timestamp
            # Ensure CSV columns order, including error columns if present
            base_cols = [
                "model","device","prompt_number","prompt_preview",
                "avg_generation_time","std_deviation","max_new_tokens",
                "memory_used_mb","gpu_name","timestamp","run_timestamp"
            ]
            
            # Add error columns if any results contain errors
            if any("error" in result for result in self.results):
                base_cols.extend(["error", "error_type"])
            
            # Only include columns that exist in the dataframe
            cols = [col for col in base_cols if col in df.columns]
            df = df[cols]
            df.to_csv(full_path, index=False)
            print(f"\nResults saved to {full_path}")
            print(f"File exists: {os.path.exists(full_path)}")
            return df
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def print_comparison_summary(self):
        if not self.results:
            print("No results available")
            return
        print("\n" + "="*80)
        print("MULTI-MODEL PERFORMANCE COMPARISON SUMMARY (Ollama)")
        print("="*80)
        df = pd.DataFrame(self.results)
        
        # Filter out failed results for performance calculations (negative times indicate errors)
        successful_df = df[df['avg_generation_time'] >= 0] if 'avg_generation_time' in df.columns else df
        failed_df = df[df['avg_generation_time'] < 0] if 'avg_generation_time' in df.columns else pd.DataFrame()
        
        # Report any errors
        if not failed_df.empty:
            print(f"\n⚠️  ERRORS DETECTED:")
            print(f"   Failed results: {len(failed_df)}/{len(df)} ({len(failed_df)/len(df)*100:.1f}%)")
            
            if 'error_type' in failed_df.columns:
                error_summary = failed_df['error_type'].value_counts()
                print(f"   Error breakdown:")
                for error_type, count in error_summary.items():
                    print(f"     - {error_type}: {count}")
            
            print(f"   Failed models:")
            failed_models = failed_df.groupby(['model', 'device']).size().reset_index(name='failed_count')
            for _, row in failed_models.iterrows():
                print(f"     - {row['model']} ({row['device']}): {row['failed_count']} failures")
        
        if successful_df.empty:
            print("\n❌ No successful benchmark results to analyze")
            return
            
        print(f"\n✅ SUCCESSFUL RESULTS: {len(successful_df)}/{len(df)} ({len(successful_df)/len(df)*100:.1f}%)")
        
        if 'device' in successful_df.columns and 'model' in successful_df.columns:
            # Only calculate stats for successful results
            device_stats = successful_df.groupby('device')['avg_generation_time'].agg(['mean','std','min','max']).round(3)
            print("\nOverall Performance by Device (Successful Results Only):")
            print(device_stats)
            
            devices = successful_df['device'].unique()
            if 'cpu' in devices and 'gpu' in devices:
                cpu_avg = successful_df[successful_df['device']=='cpu']['avg_generation_time'].mean()
                gpu_avg = successful_df[successful_df['device']=='gpu']['avg_generation_time'].mean()
                if gpu_avg > 0 and cpu_avg > 0:
                    print(f"\nOverall GPU Speedup: {cpu_avg / gpu_avg:.2f}x")

            print("\nPer-Model Performance (Successful Results Only):")
            print("-"*80)
            for m in sorted(successful_df['model'].unique()):
                md = successful_df[successful_df['model']==m]
                if not md.empty:
                    stats = md.groupby('device')['avg_generation_time'].agg(['mean','count']).round(3)
                    print(f"\nModel: {m}")
                    for dev, row in stats.iterrows():
                        total_attempts = len(df[(df['model']==m) & (df['device']==dev)])
                        success_rate = int(row['count']) / total_attempts * 100 if total_attempts > 0 else 0
                        print(f"  {dev.upper()}: {row['mean']:.3f}s avg ({int(row['count'])} successful, {success_rate:.1f}% success rate)")

def main():
    parser = argparse.ArgumentParser(description="Multi-model prompt benchmark (Ollama only)")
    parser.add_argument("--prompts-file", default="prompts.txt")
    parser.add_argument("--models-file", default="models.txt")
    parser.add_argument("--only-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU benchmarks and only run GPU benchmarks")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU benchmarks and only run CPU benchmarks")
    args = parser.parse_args()

    bench = PromptBenchmark(args.prompts_file, args.models_file)
    bench.run_comparison_benchmark(max_new_tokens=args.max_new_tokens, model_name_filter=args.only_model, skip_cpu=args.skip_cpu, skip_gpu=args.skip_gpu)
    bench.print_comparison_summary()
    bench.save_results("multi_model_benchmark")
    print("\nMulti-Model Benchmark complete!")
    ts = getattr(bench, 'run_timestamp', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"Results saved with timestamp: {ts}")
    print(f"Total benchmark results: {len(bench.results)} entries")

if __name__ == "__main__":
    main()
