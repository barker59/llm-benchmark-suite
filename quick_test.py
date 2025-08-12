#!/usr/bin/env python3
"""
Quick LLM Performance Test

A lightweight script to quickly test LLM performance and identify bottlenecks.
Perfect for comparing desktop vs server performance.

Usage:
    python quick_test.py
    python quick_test.py --model llama3.2:3b --tokens 50
"""

import time
import argparse
import requests
import psutil
import os
from datetime import datetime

def test_ollama_performance(model="llama3.2:3b", prompt="Hello, how are you?", max_tokens=50):
    """Test Ollama performance with system monitoring."""
    
    print(f"🚀 Quick LLM Performance Test")
    print(f"="*50)
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System info
    print(f"\n💻 System Info:")
    print(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total, {psutil.virtual_memory().available / (1024**3):.1f} GB available")
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    
    # Test Ollama connection
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    print(f"\n🔗 Testing Ollama connection ({base_url})...")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m.get("name", m.get("model", "unknown")) for m in models]
        print(f"  ✅ Connected! Available models: {len(model_names)}")
        
        if model not in model_names:
            print(f"  ⚠️  Model '{model}' not found. Available: {', '.join(model_names[:3])}...")
            if model_names:
                model = model_names[0]
                print(f"  🔄 Using '{model}' instead")
            else:
                print(f"  ❌ No models available. Run: ollama pull {model}")
                return
                
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        print(f"  💡 Make sure Ollama is running: ollama serve")
        return
    
    # Performance test
    print(f"\n⚡ Performance Test:")
    print(f"  Testing model '{model}'...")
    
    # Record system state before
    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory()
    
    # Start timer
    start_time = time.time()
    
    # Make request
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7
        }
    }
    
    try:
        print(f"  🔄 Generating response...")
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        # End timer
        end_time = time.time()
        duration = end_time - start_time
        
        # Record system state after
        cpu_after = psutil.cpu_percent(interval=None)
        memory_after = psutil.virtual_memory()
        
        # Results
        generated_text = data.get("response", "")
        token_count = len(generated_text.split())
        
        print(f"\n📊 Results:")
        print(f"  ⏱️  Generation Time: {duration:.2f} seconds")
        print(f"  📝 Generated Text: {len(generated_text)} characters, ~{token_count} tokens")
        print(f"  🏃 Speed: {token_count/duration:.1f} tokens/second")
        print(f"  🖥️  CPU Usage: {cpu_before:.1f}% → {cpu_after:.1f}%")
        print(f"  💾 Memory: {memory_before.percent:.1f}% → {memory_after.percent:.1f}%")
        
        # Performance classification
        if duration < 10:
            perf = "🚀 Excellent"
        elif duration < 30:
            perf = "✅ Good"
        elif duration < 60:
            perf = "⚠️  Slow"
        else:
            perf = "❌ Very Slow"
            
        print(f"  📈 Performance: {perf}")
        
        # Show generated text preview
        preview = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
        print(f"\n💬 Generated Text Preview:")
        print(f"  \"{preview}\"")
        
        # Bottleneck analysis
        print(f"\n🔍 Quick Bottleneck Analysis:")
        
        if duration > 60:
            print(f"  ❌ Very slow performance detected!")
            
        if psutil.cpu_count(logical=True) < 4:
            print(f"  ⚠️  Limited CPU cores ({psutil.cpu_count(logical=True)})")
            
        if psutil.virtual_memory().total < 8 * (1024**3):
            print(f"  ⚠️  Limited RAM ({psutil.virtual_memory().total / (1024**3):.1f} GB)")
            
        if memory_after.percent > 90:
            print(f"  ⚠️  High memory usage ({memory_after.percent:.1f}%)")
            
        # Recommendations
        print(f"\n💡 Recommendations:")
        if duration > 30:
            print(f"  🔧 Try a smaller model (e.g., llama3.2:1b)")
            print(f"  🔧 Reduce max tokens (current: {max_tokens})")
            print(f"  🔧 Check system resources and close other applications")
        
        print(f"  📊 Run full benchmark: python \"prompt_benchmark v2.py\" --system-info-only")
        print(f"  📊 Compare systems: python system_analyzer.py --export desktop.json")
        
        return {
            'duration': duration,
            'tokens_per_second': token_count/duration if duration > 0 else 0,
            'cpu_usage_change': cpu_after - cpu_before,
            'memory_usage_after': memory_after.percent,
            'success': True
        }
        
    except requests.exceptions.Timeout:
        print(f"  ❌ Request timeout (> 2 minutes)")
        print(f"  💡 Model may be too large or system too slow")
        return {'success': False, 'error': 'timeout'}
        
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Quick LLM Performance Test")
    parser.add_argument("--model", default="llama3.2:3b", help="Model to test (default: llama3.2:3b)")
    parser.add_argument("--prompt", default="Explain quantum physics in simple terms.", help="Test prompt")
    parser.add_argument("--tokens", type=int, default=50, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    result = test_ollama_performance(args.model, args.prompt, args.tokens)
    
    if result and result.get('success'):
        print(f"\n✅ Test completed successfully!")
        
        # Save result for comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import json
        with open(f"quick_test_{timestamp}.json", "w") as f:
            json.dump({
                'timestamp': timestamp,
                'model': args.model,
                'result': result,
                'system': {
                    'cpu_cores': psutil.cpu_count(logical=True),
                    'ram_gb': psutil.virtual_memory().total / (1024**3)
                }
            }, f, indent=2)
        print(f"📁 Results saved to quick_test_{timestamp}.json")
    else:
        print(f"\n❌ Test failed!")

if __name__ == "__main__":
    main()
