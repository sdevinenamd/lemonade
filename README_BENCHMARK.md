# Benchmark with Power Monitoring

This script runs lemonade benchmarks while monitoring GPU power usage.

## Quick Start

```bash
# Make sure you're in the lemonade directory
cd /home/user/Downloads/lemonade

# Run the script
./benchmark_with_power.sh
```

## What It Does

1. **Starts GPU power monitoring** using `nvidia-smi`
   - Logs timestamp, power draw, temperature, utilization, and clock speeds
   - Samples every 1 second
   - Saves to `benchmark_results/gpu_power_TIMESTAMP.csv`

2. **Runs lemonade benchmark**
   - Model: Qwen2.5-0.5B-Instruct (Q4_0)
   - Device: igpu (integrated GPU)
   - 5 iterations with prompt lengths: 64, 128, 256, 512 tokens
   - Saves output to `benchmark_results/benchmark_TIMESTAMP.log`

3. **Shows power usage summary**
   - Average, min, and max power consumption
   - Number of samples collected

4. **Automatic cleanup**
   - Stops nvidia-smi when benchmark completes
   - Handles Ctrl+C gracefully

## Customization

Edit the script to change parameters:

```bash
# Model to benchmark
MODEL="Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf"

# Device: cpu, igpu, or gpu
DEVICE="igpu"

# Number of benchmark iterations
ITERATIONS=5

# Prompt token lengths to test
PROMPTS="64 128 256 512"
```

## Output Files

All results are saved to `benchmark_results/`:
- `gpu_power_YYYYMMDD_HHMMSS.csv` - GPU power/temperature/utilization data
- `benchmark_YYYYMMDD_HHMMSS.log` - Benchmark output with TPS and TTFT metrics

## Viewing Results

### Power Data
```bash
# View raw CSV
cat benchmark_results/gpu_power_*.csv

# Or import into your favorite tool (Excel, Python pandas, etc.)
```

### Benchmark Results
```bash
# View log file
cat benchmark_results/benchmark_*.log

# Or use lemonade's built-in report
lemonade report --perf
```

## Example Output

```
==========================================
Lemonade Benchmark with Power Monitoring
==========================================
Model: Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf
Device: igpu
Iterations: 5
Prompts: 64 128 256 512
Power log: ./benchmark_results/gpu_power_20251031_183000.csv
Benchmark log: ./benchmark_results/benchmark_20251031_183000.log
==========================================

Starting GPU power monitoring...
Power monitoring started (PID: 12345)

Starting benchmark...

✓ Loading llama.cpp model
✓ Benchmarking LLM

Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf:
    Status: Successful build!
    Seconds To First Token: [0.336, 0.728, 1.487, 3.136]
    Token Generation Tokens Per Second: 182.200
    ...

==========================================
Benchmark completed successfully!
==========================================

Power usage summary:
----------------------------------------
  Average Power: 45.32 W
  Max Power:     68.50 W
  Min Power:     22.10 W
  Samples:       187
----------------------------------------
```

## Troubleshooting

### nvidia-smi not found
If you don't have NVIDIA GPU or nvidia-smi is not installed, the script will skip power monitoring and run the benchmark only.

### Permission denied
Make sure the script is executable:
```bash
chmod +x benchmark_with_power.sh
```

### lemonade command not found
The script adds conda environment to PATH automatically. If it still fails, activate your conda environment first:
```bash
conda activate lemon
./benchmark_with_power.sh
```

### Want to use Python instead?
While bash is recommended for this use case, here's a Python equivalent if needed:

```python
#!/usr/bin/env python3
import subprocess
import time
import signal
import sys
from datetime import datetime

# Start nvidia-smi
power_log = f"gpu_power_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
nvidia_proc = subprocess.Popen([
    'nvidia-smi',
    '--query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu',
    '--format=csv',
    '-l', '1'
], stdout=open(power_log, 'w'))

# Cleanup handler
def cleanup(signum=None, frame=None):
    nvidia_proc.terminate()
    nvidia_proc.wait()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

try:
    # Run benchmark
    subprocess.run([
        'lemonade',
        '-i', 'Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf',
        'llamacpp-load', '--device', 'igpu',
        'llamacpp-bench', '--iterations', '5', '--prompts', '64', '128', '256', '512'
    ], check=True)
finally:
    cleanup()
```

But the bash script is simpler and more robust for this use case!
