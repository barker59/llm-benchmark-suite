import torch
import time
import psutil
import GPUtil
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import os
from datetime import datetime
import gc
import requests
import argparse

class PromptBenchmark:
    def __init__(self, prompts_file="challenging_prompts.txt", models_file="models.txt"):
        self.prompts_file = prompts_file
        self.models_file = models_file
        self.results = []
        self.prompts = []
        self.models = []
        
    def parse_model_entry(self, entry: str):
        """Parse a model entry into (source, identifier)
        Supported:
          - 'ollama:MODEL_TAG'
          - 'lmstudio:MODEL_NAME'
          - 'local:ABSOLUTE_OR_RELATIVE_PATH' (Hugging Face format dir)
          - 'huggingface:ORG/MODEL' or bare 'ORG/MODEL' or short id like 'gpt2'
        """
        if ":" in entry:
            prefix, value = entry.split(":", 1)
            prefix = prefix.strip().lower()
            value = value.strip()
            if prefix in {"ollama", "lmstudio", "local", "huggingface", "hf"}:
                if prefix == "hf":
                    prefix = "huggingface"
                return prefix, value
        return "huggingface", entry.strip()
        
    def load_prompts(self):
        """Load prompts from text file"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Split by double newlines to separate prompts
                self.prompts = [prompt.strip() for prompt in content.split('\n\n') if prompt.strip()]
            print(f"Loaded {len(self.prompts)} prompts from {self.prompts_file}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {self.prompts_file}")
            return False
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return False
    
    def load_models(self):
        """Load models from text file"""
        try:
            with open(self.models_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Split by newlines to get individual models
                self.models = [model.strip() for model in content.split('\n') if model.strip()]
            print(f"Loaded {len(self.models)} models from {self.models_file}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {self.models_file}")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_gpu_info(self):
        """Get detailed GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'load': gpu.load * 100
                }
        except:
            pass
        return None
    
    def get_system_memory(self):
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
    
    def benchmark_model_on_device(self, model_name, device, max_new_tokens=100):
        """Benchmark a model on a specific device with all prompts"""
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name} on {device.upper()}")
        print('='*60)
        
        device_results = []
        
        try:
            # Load model and tokenizer
            print("Loading model and tokenizer...")
            # Support local HF directories via 'local:' prefix
            load_target = model_name
            local_files_only = False
            if isinstance(model_name, str) and model_name.startswith("local:"):
                load_target = model_name.split(":", 1)[1]
                local_files_only = True
            
            tokenizer = AutoTokenizer.from_pretrained(load_target, use_fast=False, local_files_only=local_files_only)
            
            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Force device selection based on request
            if device == "gpu" or device == "cuda":
                # Try to use GPU acceleration 
                try:
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        # Check if GPU is compatible
                        capability = torch.cuda.get_device_capability(0)
                        print(f"GPU capability: sm_{capability[0]}{capability[1]}")
                        
                        # Check if capability is supported
                        if capability[0] >= 12:  # sm_120+ (RTX 50xx series)
                            print("WARNING: GPU architecture sm_120 may not be fully supported")
                            print("Attempting to use GPU anyway - some operations may fail")
                            # Don't raise exception, let's try to use it anyway
                        
                        # Load model with device_map for GPU – prefer bfloat16, fallback to float16
                        model_dtype_used = None
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.bfloat16,
                                low_cpu_mem_usage=True,
                                device_map="auto"
                            )
                            model_dtype_used = torch.bfloat16
                            print("Loaded model in bfloat16")
                        except Exception as bf_err:
                            print(f"bfloat16 load failed, retrying float16: {bf_err}")
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True,
                                device_map="auto"
                            )
                            model_dtype_used = torch.float16
                        device_type = "gpu"
                        # Don't specify device for pipeline when using accelerate
                        pipeline_device = None  
                        print(f"Model loaded with device_map='auto' - using GPU acceleration")
                    else:
                        raise Exception("CUDA not available or no GPUs found")
                        
                except Exception as e:
                    print(f"GPU loading failed, falling back to CPU: {e}")
                    model = AutoModelForCausalLM.from_pretrained(
                        load_target,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    model = model.to("cpu")
                    device_type = "cpu"
                    pipeline_device = -1
            else:
                # Force CPU usage
                model = AutoModelForCausalLM.from_pretrained(
                    load_target,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to("cpu")
                device_type = "cpu"
                pipeline_device = -1
            
            print(f"Model loaded successfully on {device_type}")
            
            # Create text generation pipeline
            if pipeline_device is None:
                # For accelerate-loaded models, don't specify device
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer
                )
            else:
                # For CPU or manual device placement
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=pipeline_device
                )
            
            # Warmup
            print("Running warmup...")
            try:
                warmup_prompt = "This is a test prompt for warmup."
                generator(warmup_prompt, max_new_tokens=10, do_sample=False)
            except Exception as e:
                print(f"Warmup warning: {e}")
            
            # Benchmark each prompt
            for i, prompt in enumerate(self.prompts):
                print(f"\nProcessing prompt {i+1}/{len(self.prompts)}...")
                print(f"Prompt preview: {prompt[:100]}...")
                
                prompt_results = []
                memory_usage = []
                
                # Run 3 iterations per prompt
                for iteration in range(3):
                    # Clear cache
                    if device_type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Get memory before
                    memory_before = 0
                    if device_type == "cuda":
                        gpu_info = self.get_gpu_info()
                        if gpu_info:
                            memory_before = gpu_info['memory_used']
                    
                    # Time the generation
                    start_time = time.time()
                    
                    try:
                        # Generate text
                        output = generator(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                            return_full_text=False
                        )
                        
                        generated_text = output[0]['generated_text'] if output else ""
                        
                    except Exception as e:
                        print(f"Generation error: {e}")
                        generated_text = ""
                    
                    end_time = time.time()
                    
                    # Get memory after
                    memory_after = 0
                    if device_type == "cuda":
                        gpu_info = self.get_gpu_info()
                        if gpu_info:
                            memory_after = gpu_info['memory_used']
                    
                    generation_time = end_time - start_time
                    memory_used = memory_after - memory_before
                    
                    prompt_results.append(generation_time)
                    memory_usage.append(memory_used)
                    
                    print(f"  Iteration {iteration+1}: {generation_time:.3f}s, Generated: {len(generated_text)} chars")
                
                # Calculate averages for this prompt
                avg_time = sum(prompt_results) / len(prompt_results)
                avg_memory = sum(memory_usage) / len(memory_usage) if any(memory_usage) else 0
                std_dev = (sum((x - avg_time)**2 for x in prompt_results) / len(prompt_results))**0.5
                
                # Store result
                result = {
                    'model': model_name,
                    'device': device_type,
                    'prompt_number': i + 1,
                    'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'avg_generation_time': avg_time,
                    'std_deviation': std_dev,
                    'max_new_tokens': max_new_tokens,
                    'memory_used_mb': avg_memory,
                    'gpu_name': self.get_gpu_info()['name'] if self.get_gpu_info() else "N/A",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                device_results.append(result)
                print(f"  Average time: {avg_time:.3f}s ± {std_dev:.3f}s")
            
            # Clean up
            del model
            del generator
            if device_type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return device_results
            
        except Exception as e:
            print(f"Error benchmarking {model_name} on {device}: {e}")
            return []

    def benchmark_ollama(self, model_tag: str, max_new_tokens: int = 100):
        """Benchmark a local Ollama model via REST API (http://127.0.0.1:11434)."""
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        endpoint = f"{base_url}/api/generate"
        device_results = []
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_tag} via OLLAMA API")
        print('='*60)
        for i, prompt in enumerate(self.prompts):
            print(f"\nProcessing prompt {i+1}/{len(self.prompts)}...")
            print(f"Prompt preview: {prompt[:100]}...")
            times = []
            for iteration in range(3):
                payload = {
                    "model": model_tag,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": max_new_tokens}
                }
                start_time = time.time()
                try:
                    resp = requests.post(endpoint, json=payload, timeout=600)
                    resp.raise_for_status()
                    data = resp.json()
                    _gen = data.get("response", "")
                except Exception as e:
                    print(f"Generation error (ollama): {e}")
                    _gen = ""
                end_time = time.time()
                dt = end_time - start_time
                times.append(dt)
                print(f"  Iteration {iteration+1}: {dt:.3f}s, Generated: {len(_gen)} chars")
            avg_time = sum(times) / len(times)
            std_dev = (sum((x - avg_time)**2 for x in times) / len(times))**0.5
            result = {
                'model': model_tag,
                'device': 'ollama',
                'prompt_number': i + 1,
                'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'avg_generation_time': avg_time,
                'std_deviation': std_dev,
                'max_new_tokens': max_new_tokens,
                'memory_used_mb': 0,
                'gpu_name': self.get_gpu_info()['name'] if self.get_gpu_info() else "N/A",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            device_results.append(result)
            print(f"  Average time: {avg_time:.3f}s ± {std_dev:.3f}s")
        return device_results

    def benchmark_lmstudio(self, model_name: str, max_new_tokens: int = 100):
        """Benchmark an LM Studio model via OpenAI-compatible local server.
        Default endpoint: http://localhost:1234/v1/completions
        """
        base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234")
        endpoint = f"{base_url}/v1/completions"
        device_results = []
        headers = {"Content-Type": "application/json"}
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name} via LM Studio API")
        print('='*60)
        for i, prompt in enumerate(self.prompts):
            print(f"\nProcessing prompt {i+1}/{len(self.prompts)}...")
            print(f"Prompt preview: {prompt[:100]}...")
            times = []
            for iteration in range(3):
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_new_tokens,
                    "temperature": 0.7
                }
                start_time = time.time()
                try:
                    resp = requests.post(endpoint, json=payload, headers=headers, timeout=600)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("text", "")
                except Exception as e:
                    print(f"Generation error (lmstudio): {e}")
                    text = ""
                end_time = time.time()
                dt = end_time - start_time
                times.append(dt)
                print(f"  Iteration {iteration+1}: {dt:.3f}s, Generated: {len(text)} chars")
            avg_time = sum(times) / len(times)
            std_dev = (sum((x - avg_time)**2 for x in times) / len(times))**0.5
            result = {
                'model': model_name,
                'device': 'lmstudio',
                'prompt_number': i + 1,
                'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'avg_generation_time': avg_time,
                'std_deviation': std_dev,
                'max_new_tokens': max_new_tokens,
                'memory_used_mb': 0,
                'gpu_name': self.get_gpu_info()['name'] if self.get_gpu_info() else "N/A",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            device_results.append(result)
            print(f"  Average time: {avg_time:.3f}s ± {std_dev:.3f}s")
        return device_results
    
    def run_comparison_benchmark(self, max_new_tokens=100, model_name_filter: str | None = None):
        """Run benchmark comparing CPU vs GPU performance for all models
        If model_name_filter is provided, only models whose entry contains the filter (case-insensitive) are run.
        """
        if not self.load_prompts():
            return
        
        if not self.load_models():
            return
        
        # Generate run timestamp
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        print("Starting Multi-Model Prompt Benchmark - CPU vs GPU Comparison")
        print("=" * 80)
        print(f"Run timestamp: {run_timestamp}")
        # Optional filter to narrow models list
        if model_name_filter:
            before = len(self.models)
            self.models = [m for m in self.models if model_name_filter.lower() in m.lower()]
            print(f"Applied model filter '{model_name_filter}': {before} -> {len(self.models)} models")

        print(f"Models to test: {len(self.models)}")
        for i, model in enumerate(self.models, 1):
            print(f"  {i}. {model}")
        print(f"Number of prompts: {len(self.prompts)}")
        print(f"Max new tokens per generation: {max_new_tokens}")
        print(f"System Memory: {self.get_system_memory()['total'] / (1024**3):.2f} GB")
        
        # Check GPU availability
        cuda_available = torch.cuda.is_available()
        gpu_count = len(GPUtil.getGPUs()) if cuda_available else 0
        print(f"CUDA available: {cuda_available}")
        print(f"Available GPUs: {gpu_count}")
        
        if cuda_available:
            print("CUDA device capability check:")
            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                print(f"  GPU {i}: sm_{capability[0]}{capability[1]}")
        
        # Check GPU compatibility more thoroughly
        gpu_compatible = False
        if cuda_available and gpu_count > 0:
            try:
                capability = torch.cuda.get_device_capability(0)
                if capability[0] < 12:  # Fully supported architectures
                    gpu_compatible = True
                    print(f"\nGPU sm_{capability[0]}{capability[1]} fully supported")
                else:
                    # Try to use RTX 5090 anyway despite warnings
                    gpu_compatible = True  
                    print(f"\nGPU sm_{capability[0]}{capability[1]} may have limited support - trying anyway")
            except:
                print("\nError checking GPU compatibility")
        
        # Store run timestamp for results
        self.run_timestamp = run_timestamp
        
        # Benchmark each model
        for model_idx, raw_model_entry in enumerate(self.models, 1):
            source, model_identifier = self.parse_model_entry(raw_model_entry)
            print(f"\n" + "="*100)
            print(f"BENCHMARKING MODEL {model_idx}/{len(self.models)}: {raw_model_entry}")
            print("="*100)
            
            if source == "ollama":
                api_results = self.benchmark_ollama(model_identifier, max_new_tokens)
                self.results.extend(api_results)
                continue
            if source == "lmstudio":
                api_results = self.benchmark_lmstudio(model_identifier, max_new_tokens)
                self.results.extend(api_results)
                continue

            # Hugging Face (remote or local path) using transformers
            hf_model_name = raw_model_entry if source == "huggingface" else f"local:{model_identifier}"

            # Benchmark on CPU
            print(f"\n{'='*80}")
            print(f"STARTING CPU BENCHMARK FOR {hf_model_name}")
            print("="*80)
            cpu_results = self.benchmark_model_on_device(hf_model_name, "cpu", max_new_tokens)
            self.results.extend(cpu_results)

            # Benchmark on GPU (if available and compatible)
            if gpu_compatible:
                print(f"\n{'='*80}")
                print(f"STARTING GPU BENCHMARK FOR {hf_model_name}")
                print("="*80)
                gpu_results = self.benchmark_model_on_device(hf_model_name, "gpu", max_new_tokens)
                self.results.extend(gpu_results)
            else:
                if not cuda_available:
                    print(f"\nSkipping GPU benchmark for {hf_model_name} - CUDA not available")
                elif gpu_count == 0:
                    print(f"\nSkipping GPU benchmark for {hf_model_name} - No GPUs found")
                else:
                    print(f"\nSkipping GPU benchmark for {hf_model_name} - GPU architecture not compatible with PyTorch version")
        
        return self.results
    
    def save_results(self, base_filename="prompt_benchmark_cpu_vs_gpu"):
        """Save results to CSV in the Code folder with timestamp"""
        if self.results:
            # Create timestamped filename
            timestamp = getattr(self, 'run_timestamp', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            filename = f"{base_filename}_{timestamp}.csv"
            
            save_path = r"C:\Users\barke\Desktop\Code"
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, filename)
            
            try:
                df = pd.DataFrame(self.results)
                # Add run_timestamp column to each row for tracking
                df['run_timestamp'] = timestamp
                df.to_csv(full_path, index=False)
                print(f"\nResults saved to {full_path}")
                print(f"File exists: {os.path.exists(full_path)}")
                return df
            except PermissionError:
                # Try alternative location if permission denied
                alt_path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
                try:
                    df.to_csv(alt_path, index=False)
                    print(f"\nResults saved to {alt_path} (alternative location due to permissions)")
                    return df
                except Exception as e2:
                    print(f"Error saving to alternative location: {e2}")
                    return None
            except Exception as e:
                print(f"Error saving file: {e}")
                return None
        else:
            print("No results to save")
            return None
    
    def print_comparison_summary(self):
        """Print a comparison summary between CPU and GPU performance"""
        if not self.results:
            print("No results available")
            return
        
        print("\n" + "="*80)
        print("MULTI-MODEL PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        
        df = pd.DataFrame(self.results)
        
        if 'device' in df.columns and 'model' in df.columns:
            # Overall stats by device
            device_stats = df.groupby('device')['avg_generation_time'].agg(['mean', 'std', 'min', 'max']).round(3)
            
            print("\nOverall Performance by Device (All Models):")
            print(device_stats)
            
            # Calculate overall speedup if both CPU and GPU results exist
            devices = df['device'].unique()
            if 'cpu' in devices and 'gpu' in devices:
                cpu_avg = df[df['device'] == 'cpu']['avg_generation_time'].mean()
                gpu_avg = df[df['device'] == 'gpu']['avg_generation_time'].mean()
                speedup = cpu_avg / gpu_avg
                print(f"\nOverall GPU Speedup: {speedup:.2f}x faster than CPU")
            
            # Per-model comparison
            print("\nPer-Model Performance Comparison:")
            print("-" * 80)
            
            for model in sorted(df['model'].unique()):
                model_data = df[df['model'] == model]
                print(f"\nModel: {model}")
                
                # Calculate averages per device for this model
                model_device_stats = model_data.groupby('device')['avg_generation_time'].agg(['mean', 'count']).round(3)
                for device, stats in model_device_stats.iterrows():
                    print(f"  {device.upper()}: {stats['mean']:.3f}s avg ({int(stats['count'])} prompts)")
                
                # Calculate speedup for this model
                if 'cpu' in model_data['device'].values and 'gpu' in model_data['device'].values:
                    cpu_avg = model_data[model_data['device'] == 'cpu']['avg_generation_time'].mean()
                    gpu_avg = model_data[model_data['device'] == 'gpu']['avg_generation_time'].mean()
                    speedup = cpu_avg / gpu_avg
                    print(f"  Speedup: {speedup:.2f}x")
            
            # Summary table by model and device
            print(f"\nSummary Table:")
            print("-" * 80)
            summary = df.pivot_table(
                values='avg_generation_time', 
                index='model', 
                columns='device', 
                aggfunc='mean'
            ).round(3)
            print(summary)

def main():
    parser = argparse.ArgumentParser(description="Multi-model CPU vs GPU prompt benchmark")
    parser.add_argument("--prompts-file", default="challenging_prompts.txt", help="Path to prompts file")
    parser.add_argument("--models-file", default="models.txt", help="Path to models file")
    parser.add_argument("--only-model", default=None, help="Substring filter; only run models containing this text")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max new tokens to generate per prompt")
    args = parser.parse_args()

    # Initialize benchmark
    benchmark = PromptBenchmark(args.prompts_file, args.models_file)
    
    # Run comparison benchmark for all (or filtered) models
    results = benchmark.run_comparison_benchmark(
        max_new_tokens=args.max_new_tokens,
        model_name_filter=args.only_model
    )
    
    # Print comparison summary
    benchmark.print_comparison_summary()
    
    # Save results with timestamp
    df = benchmark.save_results("multi_model_benchmark")
    
    print("\nMulti-Model Benchmark complete!")
    timestamp = getattr(benchmark, 'run_timestamp', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"Results saved with timestamp: {timestamp}")
    print(f"Total benchmark results: {len(benchmark.results)} entries")

if __name__ == "__main__":
    main()
