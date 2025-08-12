# LLM Benchmark Suite

A comprehensive benchmarking suite for Large Language Models (LLMs) using Ollama, designed to evaluate performance across CPU and GPU configurations with multiple models and prompts.

## Features

- **Multi-Model Testing**: Benchmark multiple LLMs simultaneously
- **CPU vs GPU Comparison**: Performance analysis across different hardware configurations
- **Comprehensive Metrics**: Generation time, memory usage, error tracking, and statistical analysis
- **Robust Error Handling**: Graceful handling of model failures and network issues
- **Automated Model Management**: Automatic model validation and pulling from Ollama registry
- **Detailed Reporting**: CSV output with statistical summaries and performance comparisons

## Prerequisites

- **Python 3.8+**
- **Ollama**: Install from [https://ollama.ai/](https://ollama.ai/)
- **NVIDIA GPU** (optional, for GPU benchmarks)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd llm-benchmark-suite
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama service:**
   ```bash
   ollama serve
   ```

## Configuration

### Models Configuration (`models.txt`)
List the models you want to benchmark, one per line:
```
gpt-oss
llama3
qwen3-coder-30b-a3b:latest
```

### Prompts Configuration (`prompts.txt`)
Add your test prompts, separated by blank lines. The included prompts cover various domains:
- Economic analysis
- Scientific explanations
- Creative writing
- Technical documentation
- Philosophical discussions

## Usage

### Basic Benchmark
```bash
python "prompt_benchmark v2.py"
```

### Advanced Options
```bash
# Benchmark specific model only
python "prompt_benchmark v2.py" --only-model llama3

# Skip CPU benchmarks (GPU only)
python "prompt_benchmark v2.py" --skip-cpu

# Custom token limit
python "prompt_benchmark v2.py" --max-new-tokens 200

# Custom files
python "prompt_benchmark v2.py" --prompts-file custom_prompts.txt --models-file custom_models.txt
```

### Command Line Arguments
- `--prompts-file`: Path to prompts file (default: `prompts.txt`)
- `--models-file`: Path to models file (default: `models.txt`)
- `--only-model`: Filter to specific model name (substring match)
- `--max-new-tokens`: Maximum tokens to generate per prompt (default: 100)
- `--skip-cpu`: Skip CPU benchmarks and only run GPU tests

## Output

### CSV Results
Results are saved as `multi_model_benchmark_YYYY-MM-DD_HH-MM-SS.csv` with columns:
- `model`: Model name
- `device`: cpu/gpu
- `prompt_number`: Sequential prompt identifier
- `prompt_preview`: First 100 characters of the prompt
- `avg_generation_time`: Average time across 3 iterations (seconds)
- `std_deviation`: Standard deviation of generation times
- `max_new_tokens`: Token limit used
- `memory_used_mb`: GPU memory usage (MB)
- `gpu_name`: GPU model name
- `timestamp`: Individual test timestamp
- `run_timestamp`: Benchmark run identifier
- `error`: Error message (if any)
- `error_type`: Error category (if any)

### Console Summary
The tool provides real-time progress updates and a comprehensive summary including:
- Per-device performance statistics
- GPU speedup calculations
- Model-by-model performance breakdown
- Error analysis and success rates

## Architecture

### Core Components

1. **PromptBenchmark Class**: Main orchestrator
   - Model and prompt management
   - Benchmark execution coordination
   - Results aggregation and reporting

2. **Model Validation**: Pre-flight checks
   - Local model detection
   - Automatic model pulling from Ollama registry
   - Comprehensive error handling

3. **Performance Measurement**:
   - Multi-iteration averaging (3 runs per test)
   - Memory usage tracking
   - Statistical analysis

4. **Error Handling**:
   - Network connectivity issues
   - Model availability problems
   - Generation timeouts
   - Memory constraints

### Supported Models
Any model available in the Ollama registry can be benchmarked. Popular options include:
- `llama3`, `llama3:8b`, `llama3:70b`
- `mistral`, `mixtral`
- `qwen`, `qwen2`
- `codellama`, `deepseek-coder`

## Server Deployment

### Quick Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
systemctl start ollama  # Linux
# or
ollama serve            # Manual start

# Clone and setup benchmark
git clone <your-repo-url>
cd llm-benchmark-suite
pip install -r requirements.txt

# Run benchmark
python "prompt_benchmark v2.py"
```

### Systemd Service (Linux)
Create `/etc/systemd/system/ollama.service`:
```ini
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=ollama
ExecStart=/usr/local/bin/ollama serve
Restart=always

[Install]
WantedBy=multi-user.target
```

### Environment Variables
- `OLLAMA_BASE_URL`: Custom Ollama endpoint (default: `http://127.0.0.1:11434`)

## Troubleshooting

### Common Issues

1. **"Cannot connect to Ollama service"**
   - Ensure Ollama is running: `ollama serve`
   - Check the service URL: `curl http://localhost:11434/api/tags`

2. **"Model not found"**
   - List available models: `ollama list`
   - Pull missing models: `ollama pull <model-name>`

3. **GPU not detected**
   - Verify NVIDIA drivers: `nvidia-smi`
   - Check CUDA installation
   - Restart Ollama service after driver updates

4. **Memory errors**
   - Reduce `--max-new-tokens` value
   - Use smaller models for testing
   - Monitor system resources: `htop`, `nvidia-smi`

### Performance Tips

1. **Model Selection**: Start with smaller models (7B-13B parameters) for initial testing
2. **Batch Testing**: Use `--only-model` flag to test individual models
3. **Resource Monitoring**: Monitor GPU memory usage with `nvidia-smi -l 1`
4. **Network Stability**: Ensure stable internet connection for model downloads

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with your configuration
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Ollama documentation: [https://ollama.ai/](https://ollama.ai/)
3. Open an issue in this repository

