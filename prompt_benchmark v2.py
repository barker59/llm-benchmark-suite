import time, psutil, pandas as pd, os, gc, requests, argparse, subprocess, threading, json, platform, sys
from datetime import datetime
from collections import defaultdict

# Try to import GPUtil, but don't fail if it's not available
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU detection will be disabled.")

class SystemMonitor:
    """Comprehensive system monitoring during benchmarks."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.data = defaultdict(list)
        self.start_time = None
        
    def start_monitoring(self, interval=1.0):
        """Start system monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.data.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        return dict(self.data)
        
    def _monitor_loop(self, interval):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time() - self.start_time
                
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                self.data['cpu_percent_overall'].append(psutil.cpu_percent(interval=None))
                self.data['cpu_percent_per_core'].append(cpu_percent)
                self.data['cpu_count_logical'].append(psutil.cpu_count(logical=True))
                self.data['cpu_count_physical'].append(psutil.cpu_count(logical=False))
                
                # CPU frequency
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    self.data['cpu_freq_current'].append(cpu_freq.current)
                    self.data['cpu_freq_max'].append(cpu_freq.max)
                
                # Memory monitoring
                memory = psutil.virtual_memory()
                self.data['memory_total'].append(memory.total)
                self.data['memory_used'].append(memory.used)
                self.data['memory_percent'].append(memory.percent)
                self.data['memory_available'].append(memory.available)
                
                # Swap memory
                swap = psutil.swap_memory()
                self.data['swap_total'].append(swap.total)
                self.data['swap_used'].append(swap.used)
                self.data['swap_percent'].append(swap.percent)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.data['disk_read_bytes'].append(disk_io.read_bytes)
                    self.data['disk_write_bytes'].append(disk_io.write_bytes)
                    self.data['disk_read_count'].append(disk_io.read_count)
                    self.data['disk_write_count'].append(disk_io.write_count)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    self.data['network_bytes_sent'].append(net_io.bytes_sent)
                    self.data['network_bytes_recv'].append(net_io.bytes_recv)
                    self.data['network_packets_sent'].append(net_io.packets_sent)
                    self.data['network_packets_recv'].append(net_io.packets_recv)
                
                # GPU monitoring (if available)
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Monitor first GPU
                            self.data['gpu_load'].append(gpu.load * 100)
                            self.data['gpu_memory_used'].append(gpu.memoryUsed)
                            self.data['gpu_memory_total'].append(gpu.memoryTotal)
                            self.data['gpu_memory_percent'].append((gpu.memoryUsed / gpu.memoryTotal) * 100)
                            self.data['gpu_temperature'].append(getattr(gpu, 'temperature', 0))
                    except Exception:
                        pass
                
                # Process-specific monitoring
                current_process = psutil.Process()
                self.data['process_cpu_percent'].append(current_process.cpu_percent())
                self.data['process_memory_rss'].append(current_process.memory_info().rss)
                self.data['process_memory_vms'].append(current_process.memory_info().vms)
                self.data['process_num_threads'].append(current_process.num_threads())
                
                # System load average (Unix-like systems)
                try:
                    loadavg = os.getloadavg()
                    self.data['load_avg_1min'].append(loadavg[0])
                    self.data['load_avg_5min'].append(loadavg[1])
                    self.data['load_avg_15min'].append(loadavg[2])
                except (AttributeError, OSError):
                    # Not available on Windows
                    pass
                
                self.data['timestamp'].append(timestamp)
                
            except Exception as e:
                print(f"Warning: Monitoring error: {e}")
                
            time.sleep(interval)
    
    def get_summary_stats(self, data):
        """Generate summary statistics from monitoring data."""
        if not data or not data.get('timestamp'):
            return {}
            
        stats = {}
        duration = data['timestamp'][-1] - data['timestamp'][0] if len(data['timestamp']) > 1 else 0
        stats['monitoring_duration_seconds'] = duration
        stats['monitoring_samples'] = len(data['timestamp'])
        
        # CPU stats
        if 'cpu_percent_overall' in data:
            cpu_data = data['cpu_percent_overall']
            stats['cpu_avg_percent'] = sum(cpu_data) / len(cpu_data) if cpu_data else 0
            stats['cpu_max_percent'] = max(cpu_data) if cpu_data else 0
            stats['cpu_min_percent'] = min(cpu_data) if cpu_data else 0
            
        if 'cpu_percent_per_core' in data and data['cpu_percent_per_core']:
            per_core = data['cpu_percent_per_core']
            # Average across all samples and cores
            core_averages = []
            for sample in per_core:
                if sample:
                    core_averages.append(sum(sample) / len(sample))
            if core_averages:
                stats['cpu_avg_across_cores'] = sum(core_averages) / len(core_averages)
                stats['cpu_max_across_cores'] = max(core_averages)
        
        if 'cpu_count_logical' in data and data['cpu_count_logical']:
            stats['cpu_cores_logical'] = data['cpu_count_logical'][0]
        if 'cpu_count_physical' in data and data['cpu_count_physical']:
            stats['cpu_cores_physical'] = data['cpu_count_physical'][0]
            
        # Memory stats
        if 'memory_percent' in data:
            mem_data = data['memory_percent']
            stats['memory_avg_percent'] = sum(mem_data) / len(mem_data) if mem_data else 0
            stats['memory_max_percent'] = max(mem_data) if mem_data else 0
            
        if 'memory_total' in data and data['memory_total']:
            stats['memory_total_gb'] = data['memory_total'][0] / (1024**3)
            
        # GPU stats
        if 'gpu_load' in data and data['gpu_load']:
            gpu_load = data['gpu_load']
            stats['gpu_avg_load_percent'] = sum(gpu_load) / len(gpu_load)
            stats['gpu_max_load_percent'] = max(gpu_load)
            
        if 'gpu_memory_percent' in data and data['gpu_memory_percent']:
            gpu_mem = data['gpu_memory_percent']
            stats['gpu_avg_memory_percent'] = sum(gpu_mem) / len(gpu_mem)
            stats['gpu_max_memory_percent'] = max(gpu_mem)
            
        # Process stats
        if 'process_cpu_percent' in data:
            proc_cpu = data['process_cpu_percent']
            stats['process_avg_cpu_percent'] = sum(proc_cpu) / len(proc_cpu) if proc_cpu else 0
            stats['process_max_cpu_percent'] = max(proc_cpu) if proc_cpu else 0
            
        return stats

class PromptBenchmark:
    def __init__(self, prompts_file="prompts.txt", models_file="models.txt"):
        self.prompts_file = prompts_file
        self.models_file = models_file
        self.results = []
        self.prompts = []
        self.models = []
        self.run_timestamp = None
        self.system_monitor = SystemMonitor()
        self.enable_monitoring = True

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
    
    def get_comprehensive_system_info(self):
        """Get detailed system information for performance analysis."""
        info = {
            'system': {
                'platform': platform.platform(),
                'system': platform.system(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'hostname': platform.node()
            },
            'cpu': {
                'cores_logical': psutil.cpu_count(logical=True),
                'cores_physical': psutil.cpu_count(logical=False),
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_percent': psutil.virtual_memory().percent
            },
            'disk': {},
            'gpu': {}
        }
        
        # CPU frequency info
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info['cpu']['freq_current_mhz'] = cpu_freq.current
                info['cpu']['freq_max_mhz'] = cpu_freq.max
                info['cpu']['freq_min_mhz'] = cpu_freq.min
        except:
            pass
            
        # Disk info for current drive
        try:
            disk_usage = psutil.disk_usage('.')
            info['disk']['total_gb'] = disk_usage.total / (1024**3)
            info['disk']['free_gb'] = disk_usage.free / (1024**3)
            info['disk']['used_percent'] = (disk_usage.used / disk_usage.total) * 100
        except:
            pass
            
        # Network interfaces
        try:
            net_stats = psutil.net_if_stats()
            active_interfaces = [name for name, stats in net_stats.items() if stats.isup]
            info['network'] = {'active_interfaces': active_interfaces}
        except:
            pass
            
        # GPU info
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info['gpu'] = {
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'driver_version': getattr(gpu, 'driver', 'unknown'),
                        'temperature': getattr(gpu, 'temperature', 'unknown')
                    }
                else:
                    info['gpu'] = {'status': 'no_gpus_detected'}
            except Exception as e:
                info['gpu'] = {'status': f'gpu_detection_error: {e}'}
        else:
            info['gpu'] = {'status': 'gputil_not_available'}
            
        return info
    
    def print_system_info(self):
        """Print comprehensive system information."""
        info = self.get_comprehensive_system_info()
        
        print("\n" + "="*80)
        print("SYSTEM PERFORMANCE ANALYSIS")
        print("="*80)
        
        print(f"\n🖥️  SYSTEM OVERVIEW:")
        print(f"  Platform: {info['system']['platform']}")
        print(f"  Processor: {info['system']['processor']}")
        print(f"  Hostname: {info['system']['hostname']}")
        print(f"  Python: {info['system']['python_version']}")
        
        print(f"\n🔧 CPU SPECIFICATIONS:")
        print(f"  Physical Cores: {info['cpu']['cores_physical']}")
        print(f"  Logical Cores: {info['cpu']['cores_logical']}")
        if 'freq_current_mhz' in info['cpu']:
            print(f"  Current Frequency: {info['cpu']['freq_current_mhz']:.0f} MHz")
            print(f"  Max Frequency: {info['cpu']['freq_max_mhz']:.0f} MHz")
            
        print(f"\n💾 MEMORY SPECIFICATIONS:")
        print(f"  Total RAM: {info['memory']['total_gb']:.2f} GB")
        print(f"  Available RAM: {info['memory']['available_gb']:.2f} GB")
        print(f"  Memory Usage: {info['memory']['used_percent']:.1f}%")
        
        if info['disk']:
            print(f"\n💿 DISK SPECIFICATIONS:")
            print(f"  Total Space: {info['disk']['total_gb']:.2f} GB")
            print(f"  Free Space: {info['disk']['free_gb']:.2f} GB")
            print(f"  Disk Usage: {info['disk']['used_percent']:.1f}%")
            
        print(f"\n🎮 GPU SPECIFICATIONS:")
        if info['gpu'].get('name'):
            print(f"  GPU: {info['gpu']['name']}")
            print(f"  VRAM: {info['gpu']['memory_total_mb']} MB")
            if info['gpu'].get('driver_version') != 'unknown':
                print(f"  Driver: {info['gpu']['driver_version']}")
        else:
            print(f"  Status: {info['gpu']['status']}")
            
        # Performance expectations
        print(f"\n⚡ PERFORMANCE EXPECTATIONS:")
        cores = info['cpu']['cores_logical']
        ram_gb = info['memory']['total_gb']
        
        if cores >= 8 and ram_gb >= 16:
            perf_class = "High-Performance"
            expected = "Fast inference (< 10s per prompt)"
        elif cores >= 4 and ram_gb >= 8:
            perf_class = "Mid-Range"
            expected = "Moderate inference (10-30s per prompt)"
        else:
            perf_class = "Limited"
            expected = "Slow inference (30s+ per prompt)"
            
        print(f"  Performance Class: {perf_class}")
        print(f"  Expected Speed: {expected}")
        
        if ram_gb < 8:
            print(f"  ⚠️  WARNING: Low RAM may cause model swapping to disk")
        if cores < 4:
            print(f"  ⚠️  WARNING: Few CPU cores may limit parallel processing")
            
        return info

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

                # Start system monitoring for this iteration
                if self.enable_monitoring:
                    self.system_monitor.start_monitoring(interval=0.5)

                data, wall = self.generate_ollama(model_tag, prompt, max_new_tokens, use_gpu)
                
                # Stop monitoring and collect data
                monitoring_data = {}
                monitoring_stats = {}
                if self.enable_monitoring:
                    monitoring_data = self.system_monitor.stop_monitoring()
                    monitoring_stats = self.system_monitor.get_summary_stats(monitoring_data)

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
                
                # Print iteration results with monitoring info
                monitoring_summary = ""
                if monitoring_stats:
                    cpu_avg = monitoring_stats.get('cpu_avg_percent', 0)
                    mem_avg = monitoring_stats.get('memory_avg_percent', 0)
                    monitoring_summary = f", CPU: {cpu_avg:.1f}%, RAM: {mem_avg:.1f}%"
                    
                print(f"  ✓ Iteration {itr+1}: {wall:.3f}s, Generated: {len(gen_text)} chars{monitoring_summary}")

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
                
                # Add monitoring statistics if available (use last iteration's stats)
                if monitoring_stats:
                    result.update({
                        "cpu_avg_percent": monitoring_stats.get('cpu_avg_percent', 0),
                        "cpu_max_percent": monitoring_stats.get('cpu_max_percent', 0),
                        "cpu_cores_logical": monitoring_stats.get('cpu_cores_logical', 0),
                        "cpu_cores_physical": monitoring_stats.get('cpu_cores_physical', 0),
                        "memory_avg_percent": monitoring_stats.get('memory_avg_percent', 0),
                        "memory_max_percent": monitoring_stats.get('memory_max_percent', 0),
                        "memory_total_gb": monitoring_stats.get('memory_total_gb', 0),
                        "process_avg_cpu_percent": monitoring_stats.get('process_avg_cpu_percent', 0),
                        "process_max_cpu_percent": monitoring_stats.get('process_max_cpu_percent', 0),
                        "monitoring_duration_seconds": monitoring_stats.get('monitoring_duration_seconds', 0),
                        "monitoring_samples": monitoring_stats.get('monitoring_samples', 0),
                    })
                    
                    # Add GPU stats if available
                    if 'gpu_avg_load_percent' in monitoring_stats:
                        result.update({
                            "gpu_avg_load_percent": monitoring_stats.get('gpu_avg_load_percent', 0),
                            "gpu_max_load_percent": monitoring_stats.get('gpu_max_load_percent', 0),
                            "gpu_avg_memory_percent": monitoring_stats.get('gpu_avg_memory_percent', 0),
                            "gpu_max_memory_percent": monitoring_stats.get('gpu_max_memory_percent', 0),
                        })
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

        # Show comprehensive system information
        self.print_system_info()
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK CONFIGURATION")
        print(f"{'='*80}")
        print(f"Models to test: {len(self.models)}")
        for i, m in enumerate(self.models, 1):
            print(f"  {i}. {m}")
        print(f"Number of prompts: {len(self.prompts)}")
        print(f"Max new tokens per generation: {max_new_tokens}")
        print(f"System monitoring: {'✅ ENABLED' if self.enable_monitoring else '❌ DISABLED'}")

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
                "memory_used_mb","gpu_name","timestamp","run_timestamp",
                # System monitoring columns
                "cpu_avg_percent","cpu_max_percent","cpu_cores_logical","cpu_cores_physical",
                "memory_avg_percent","memory_max_percent","memory_total_gb",
                "process_avg_cpu_percent","process_max_cpu_percent",
                "gpu_avg_load_percent","gpu_max_load_percent",
                "gpu_avg_memory_percent","gpu_max_memory_percent",
                "monitoring_duration_seconds","monitoring_samples"
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
    parser.add_argument("--disable-monitoring", action="store_true", help="Disable system monitoring during benchmarks")
    parser.add_argument("--system-info-only", action="store_true", help="Show system information and exit (no benchmarking)")
    args = parser.parse_args()

    bench = PromptBenchmark(args.prompts_file, args.models_file)
    bench.enable_monitoring = not args.disable_monitoring
    
    # Handle system info only mode
    if args.system_info_only:
        bench.print_system_info()
        print("\n" + "="*80)
        print("SYSTEM ANALYSIS COMPLETE")
        print("="*80)
        print("\n💡 PERFORMANCE INSIGHTS:")
        print("  - Compare CPU cores/frequency between desktop vs server")
        print("  - Check RAM amount and speed differences") 
        print("  - Verify disk type (SSD vs HDD) and free space")
        print("  - Look for network latency if using remote Ollama")
        print("  - Monitor temperature throttling on server")
        print("\n🚀 To run actual benchmark:")
        print(f"  python \"{sys.argv[0]}\" --max-new-tokens 50 --only-model <small-model>")
        return
    
    bench.run_comparison_benchmark(max_new_tokens=args.max_new_tokens, model_name_filter=args.only_model, skip_cpu=args.skip_cpu, skip_gpu=args.skip_gpu)
    bench.print_comparison_summary()
    bench.save_results("multi_model_benchmark")
    print("\nMulti-Model Benchmark complete!")
    ts = getattr(bench, 'run_timestamp', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"Results saved with timestamp: {ts}")
    print(f"Total benchmark results: {len(bench.results)} entries")

if __name__ == "__main__":
    main()
