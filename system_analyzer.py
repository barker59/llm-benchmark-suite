#!/usr/bin/env python3
"""
System Performance Analyzer for LLM Benchmarking

This script analyzes system specifications and provides performance insights
to help diagnose why server performance differs from desktop performance.

Usage:
    python system_analyzer.py
    python system_analyzer.py --export results.json
    python system_analyzer.py --compare desktop.json server.json
"""

import json, platform, time, argparse
import psutil
from datetime import datetime

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

class SystemAnalyzer:
    def __init__(self):
        self.analysis = {}
        
    def gather_system_info(self):
        """Collect comprehensive system information."""
        print("🔍 Gathering system information...")
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'platform': platform.platform(),
                'system': platform.system(), 
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'hostname': platform.node(),
                'architecture': platform.architecture()[0]
            },
            'cpu': {
                'cores_logical': psutil.cpu_count(logical=True),
                'cores_physical': psutil.cpu_count(logical=False),
            },
            'memory': {
                'total_bytes': psutil.virtual_memory().total,
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_bytes': psutil.virtual_memory().available,
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_percent': psutil.virtual_memory().percent
            },
            'disk': {},
            'network': {},
            'gpu': {},
            'performance': {}
        }
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info['cpu']['freq_current_mhz'] = cpu_freq.current
                info['cpu']['freq_max_mhz'] = cpu_freq.max
                info['cpu']['freq_min_mhz'] = cpu_freq.min
        except:
            pass
            
        # Disk performance
        try:
            disk_usage = psutil.disk_usage('.')
            info['disk']['total_gb'] = disk_usage.total / (1024**3)
            info['disk']['free_gb'] = disk_usage.free / (1024**3)
            info['disk']['used_percent'] = (disk_usage.used / disk_usage.total) * 100
            
            # Basic disk speed test
            print("  📊 Testing disk speed...")
            start_time = time.time()
            test_data = b'0' * (10 * 1024 * 1024)  # 10MB
            with open('disk_speed_test.tmp', 'wb') as f:
                f.write(test_data)
                f.flush()
            write_time = time.time() - start_time
            
            start_time = time.time()
            with open('disk_speed_test.tmp', 'rb') as f:
                f.read()
            read_time = time.time() - start_time
            
            info['disk']['write_speed_mbps'] = (10 / write_time) if write_time > 0 else 0
            info['disk']['read_speed_mbps'] = (10 / read_time) if read_time > 0 else 0
            
            # Cleanup
            import os
            try:
                os.remove('disk_speed_test.tmp')
            except:
                pass
                
        except Exception as e:
            info['disk']['error'] = str(e)
            
        # Network interfaces
        try:
            net_stats = psutil.net_if_stats()
            info['network']['interfaces'] = {}
            for name, stats in net_stats.items():
                if stats.isup:
                    info['network']['interfaces'][name] = {
                        'speed_mbps': stats.speed,
                        'mtu': stats.mtu
                    }
        except:
            pass
            
        # GPU information
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info['gpu'] = {
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_free_mb': gpu.memoryFree,
                        'load_percent': gpu.load * 100,
                        'temperature': getattr(gpu, 'temperature', 'unknown'),
                        'driver_version': getattr(gpu, 'driver', 'unknown')
                    }
                else:
                    info['gpu'] = {'status': 'no_gpus_detected'}
            except Exception as e:
                info['gpu'] = {'status': f'gpu_detection_error: {e}'}
        else:
            info['gpu'] = {'status': 'gputil_not_available'}
            
        # CPU performance test
        print("  ⚡ Testing CPU performance...")
        start_time = time.time()
        # Simple CPU-bound calculation
        result = sum(i * i for i in range(1000000))
        cpu_test_time = time.time() - start_time
        info['performance']['cpu_test_seconds'] = cpu_test_time
        info['performance']['cpu_test_score'] = 1.0 / cpu_test_time if cpu_test_time > 0 else 0
        
        # Memory bandwidth test
        print("  💾 Testing memory bandwidth...")
        start_time = time.time()
        data = [0] * (10 * 1024 * 1024)  # 10M integers
        for i in range(len(data)):
            data[i] = i
        memory_test_time = time.time() - start_time
        info['performance']['memory_test_seconds'] = memory_test_time
        info['performance']['memory_test_score'] = 1.0 / memory_test_time if memory_test_time > 0 else 0
        
        self.analysis = info
        return info
        
    def print_analysis(self):
        """Print comprehensive system analysis."""
        info = self.analysis
        
        print("\n" + "="*80)
        print("🖥️  SYSTEM PERFORMANCE ANALYSIS")
        print("="*80)
        
        print(f"\n📋 SYSTEM OVERVIEW:")
        print(f"  Hostname: {info['system']['hostname']}")
        print(f"  Platform: {info['system']['platform']}")
        print(f"  Architecture: {info['system']['architecture']}")
        print(f"  Processor: {info['system']['processor']}")
        print(f"  Python: {info['system']['python_version']}")
        
        print(f"\n🔧 CPU SPECIFICATIONS:")
        print(f"  Physical Cores: {info['cpu']['cores_physical']}")
        print(f"  Logical Cores: {info['cpu']['cores_logical']}")
        if 'freq_current_mhz' in info['cpu']:
            print(f"  Current Frequency: {info['cpu']['freq_current_mhz']:.0f} MHz")
            print(f"  Max Frequency: {info['cpu']['freq_max_mhz']:.0f} MHz")
        if 'cpu_test_seconds' in info['performance']:
            print(f"  Performance Score: {info['performance']['cpu_test_score']:.2f} (higher is better)")
            
        print(f"\n💾 MEMORY SPECIFICATIONS:")
        print(f"  Total RAM: {info['memory']['total_gb']:.2f} GB")
        print(f"  Available RAM: {info['memory']['available_gb']:.2f} GB")
        print(f"  Memory Usage: {info['memory']['used_percent']:.1f}%")
        if 'memory_test_seconds' in info['performance']:
            print(f"  Memory Score: {info['performance']['memory_test_score']:.2f} (higher is better)")
            
        if info['disk'] and 'total_gb' in info['disk']:
            print(f"\n💿 DISK SPECIFICATIONS:")
            print(f"  Total Space: {info['disk']['total_gb']:.2f} GB")
            print(f"  Free Space: {info['disk']['free_gb']:.2f} GB")
            print(f"  Disk Usage: {info['disk']['used_percent']:.1f}%")
            if 'read_speed_mbps' in info['disk']:
                print(f"  Read Speed: {info['disk']['read_speed_mbps']:.1f} MB/s")
                print(f"  Write Speed: {info['disk']['write_speed_mbps']:.1f} MB/s")
                
        print(f"\n🌐 NETWORK:")
        if info['network']['interfaces']:
            for name, stats in info['network']['interfaces'].items():
                speed = stats['speed_mbps']
                print(f"  {name}: {speed} Mbps" if speed > 0 else f"  {name}: speed unknown")
        else:
            print("  No active network interfaces detected")
            
        print(f"\n🎮 GPU SPECIFICATIONS:")
        if info['gpu'].get('name'):
            print(f"  GPU: {info['gpu']['name']}")
            print(f"  VRAM: {info['gpu']['memory_total_mb']} MB")
            print(f"  VRAM Used: {info['gpu']['memory_used_mb']} MB ({info['gpu']['load_percent']:.1f}%)")
            if info['gpu'].get('temperature') != 'unknown':
                print(f"  Temperature: {info['gpu']['temperature']}°C")
        else:
            print(f"  Status: {info['gpu']['status']}")
            
        # Performance classification
        self.classify_performance()
        
    def classify_performance(self):
        """Classify system performance for LLM inference."""
        info = self.analysis
        
        print(f"\n⚡ PERFORMANCE CLASSIFICATION:")
        
        cores = info['cpu']['cores_logical']
        ram_gb = info['memory']['total_gb']
        cpu_score = info['performance'].get('cpu_test_score', 0)
        
        # Overall classification
        if cores >= 16 and ram_gb >= 32 and cpu_score > 5:
            perf_class = "🚀 High-Performance"
            expected = "Very fast inference (< 5s per prompt)"
        elif cores >= 8 and ram_gb >= 16 and cpu_score > 2:
            perf_class = "⚡ Good Performance" 
            expected = "Fast inference (5-15s per prompt)"
        elif cores >= 4 and ram_gb >= 8 and cpu_score > 1:
            perf_class = "✅ Adequate Performance"
            expected = "Moderate inference (15-30s per prompt)"
        else:
            perf_class = "⚠️  Limited Performance"
            expected = "Slow inference (30s+ per prompt)"
            
        print(f"  Overall: {perf_class}")
        print(f"  Expected LLM Speed: {expected}")
        
        # Bottleneck analysis
        print(f"\n🔍 BOTTLENECK ANALYSIS:")
        
        bottlenecks = []
        if cores < 4:
            bottlenecks.append("❌ CPU cores (< 4 cores)")
        elif cores < 8:
            bottlenecks.append("⚠️  CPU cores (limited parallelism)")
            
        if ram_gb < 8:
            bottlenecks.append("❌ RAM (< 8GB, will cause swapping)")
        elif ram_gb < 16:
            bottlenecks.append("⚠️  RAM (limited for large models)")
            
        if cpu_score < 1:
            bottlenecks.append("❌ CPU performance (very slow)")
        elif cpu_score < 2:
            bottlenecks.append("⚠️  CPU performance (below average)")
            
        if 'read_speed_mbps' in info['disk'] and info['disk']['read_speed_mbps'] < 100:
            bottlenecks.append("⚠️  Disk speed (< 100 MB/s, likely HDD)")
            
        if bottlenecks:
            for bottleneck in bottlenecks:
                print(f"  {bottleneck}")
        else:
            print("  ✅ No obvious bottlenecks detected")
            
        # Recommendations
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        if ram_gb < 16:
            print("  🔧 Add more RAM for better model caching")
        if cores < 8:
            print("  🔧 Upgrade CPU for better parallel processing")
        if 'read_speed_mbps' in info['disk'] and info['disk']['read_speed_mbps'] < 200:
            print("  🔧 Consider SSD upgrade for faster model loading")
        if not info['gpu'].get('name'):
            print("  🔧 Add GPU for hardware acceleration")
            
        print("  🔧 Ensure Ollama has sufficient resources allocated")
        print("  🔧 Monitor system during inference to identify real bottlenecks")
        
    def export_analysis(self, filename):
        """Export analysis to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.analysis, f, indent=2)
        print(f"✅ Analysis exported to {filename}")
        
    def compare_systems(self, file1, file2):
        """Compare two system analysis files."""
        try:
            with open(file1, 'r') as f:
                system1 = json.load(f)
            with open(file2, 'r') as f:
                system2 = json.load(f)
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            return
            
        print("\n" + "="*80)
        print("📊 SYSTEM COMPARISON")
        print("="*80)
        
        print(f"\n🖥️  SYSTEM 1: {system1['system']['hostname']}")
        print(f"🖥️  SYSTEM 2: {system2['system']['hostname']}")
        
        # CPU comparison
        print(f"\n🔧 CPU COMPARISON:")
        print(f"  Logical Cores: {system1['cpu']['cores_logical']} vs {system2['cpu']['cores_logical']}")
        print(f"  Physical Cores: {system1['cpu']['cores_physical']} vs {system2['cpu']['cores_physical']}")
        
        if 'freq_max_mhz' in system1['cpu'] and 'freq_max_mhz' in system2['cpu']:
            print(f"  Max Frequency: {system1['cpu']['freq_max_mhz']:.0f} MHz vs {system2['cpu']['freq_max_mhz']:.0f} MHz")
            
        if 'cpu_test_score' in system1['performance'] and 'cpu_test_score' in system2['performance']:
            score1 = system1['performance']['cpu_test_score']
            score2 = system2['performance']['cpu_test_score']
            ratio = score1 / score2 if score2 > 0 else float('inf')
            print(f"  Performance Score: {score1:.2f} vs {score2:.2f} (ratio: {ratio:.2f}x)")
            
        # Memory comparison  
        print(f"\n💾 MEMORY COMPARISON:")
        print(f"  Total RAM: {system1['memory']['total_gb']:.2f} GB vs {system2['memory']['total_gb']:.2f} GB")
        
        if 'memory_test_score' in system1['performance'] and 'memory_test_score' in system2['performance']:
            score1 = system1['performance']['memory_test_score']
            score2 = system2['performance']['memory_test_score']
            ratio = score1 / score2 if score2 > 0 else float('inf')
            print(f"  Memory Score: {score1:.2f} vs {score2:.2f} (ratio: {ratio:.2f}x)")
            
        # Disk comparison
        if 'read_speed_mbps' in system1['disk'] and 'read_speed_mbps' in system2['disk']:
            print(f"\n💿 DISK COMPARISON:")
            print(f"  Read Speed: {system1['disk']['read_speed_mbps']:.1f} MB/s vs {system2['disk']['read_speed_mbps']:.1f} MB/s")
            print(f"  Write Speed: {system1['disk']['write_speed_mbps']:.1f} MB/s vs {system2['disk']['write_speed_mbps']:.1f} MB/s")
            
        # Performance prediction
        print(f"\n⚡ PERFORMANCE PREDICTION:")
        if 'cpu_test_score' in system1['performance'] and 'cpu_test_score' in system2['performance']:
            score1 = system1['performance']['cpu_test_score']
            score2 = system2['performance']['cpu_test_score']
            cores1 = system1['cpu']['cores_logical']
            cores2 = system2['cpu']['cores_logical']
            
            # Simple performance estimation
            perf1 = score1 * cores1
            perf2 = score2 * cores2
            perf_ratio = perf1 / perf2 if perf2 > 0 else float('inf')
            
            if perf_ratio > 1.5:
                print(f"  System 1 should be ~{perf_ratio:.1f}x faster")
            elif perf_ratio < 0.67:
                print(f"  System 2 should be ~{1/perf_ratio:.1f}x faster")
            else:
                print(f"  Systems should have similar performance")
                
def main():
    parser = argparse.ArgumentParser(description="System Performance Analyzer for LLM Benchmarking")
    parser.add_argument("--export", help="Export analysis to JSON file")
    parser.add_argument("--compare", nargs=2, metavar=('FILE1', 'FILE2'), help="Compare two system analysis files")
    args = parser.parse_args()
    
    analyzer = SystemAnalyzer()
    
    if args.compare:
        analyzer.compare_systems(args.compare[0], args.compare[1])
    else:
        analyzer.gather_system_info()
        analyzer.print_analysis()
        
        if args.export:
            analyzer.export_analysis(args.export)
            
if __name__ == "__main__":
    main()
