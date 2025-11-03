# NVIDIA GPU Power Profiling for Lemonade

This feature adds built-in NVIDIA GPU power profiling to Lemonade, allowing you to track energy consumption during model benchmarks.

## Installation

Install the pynvml package (NVIDIA Management Library Python bindings):

```bash
pip install pynvml
```

Or install lemonade with the nvidia extras:

```bash
pip install -e .[nvidia]
```

## Usage

Add the `--power-nvidia` flag to your lemonade benchmark commands:

```bash
lemonade -i Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf \
  --power-nvidia \
  llamacpp-load --device igpu \
  llamacpp-bench --iterations 10 --prompts 64 128 256 512 1024 2048
```

### Optional: Custom Warmup Period

You can specify a custom warmup period (in seconds) before and after the benchmark:

```bash
lemonade -i MODEL --power-nvidia 3 llamacpp-load --device igpu llamacpp-bench ...
```

Default warmup period is 5 seconds.

### Optional: Select Specific GPU

If you have multiple GPUs, set the `GPU_DEVICE_INDEX` environment variable:

```bash
GPU_DEVICE_INDEX=1 lemonade -i MODEL --power-nvidia llamacpp-load ...
```

Default is GPU 0.

## Output

The profiler generates the following outputs:

### 1. CSV File
Location: `~/.cache/lemonade/builds/<build_name>/power_usage_nvidia.csv`

Contains timestamped measurements of:
- GPU power draw (W)
- GPU temperature (°C)
- GPU utilization (%)
- Memory utilization (%)
- GPU clock speed (MHz)
- Memory clock speed (MHz)

### 2. PNG Plot
Location: Current directory and cache directory

Three subplots showing:
1. **GPU Power Draw**: Power consumption over time with benchmark stages annotated, showing:
   - Average power per stage (dashed lines)
   - Energy consumed per stage (in Joules)
   - Duration of each stage

2. **Clock Speeds & Temperature**: GPU/memory clocks and temperature over time

3. **Utilization**: GPU and memory utilization percentages over time

### 3. Console Summary

The benchmark output includes:
- Peak GPU Power (W)
- Path to the power usage plot
- Energy consumed per benchmark stage (displayed in performance reports)

## Example Output

```
Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf:
    Peak Gpu Power Nvidia:              36.5 W
    Power Usage Plot Nvidia:            /path/to/power_usage_nvidia.png
```

The plot legend shows energy consumed for each stage:
- `warmup (0.0s, 0.0 J)`
- `llamacpp-load (0.1s, 0.5 J)`
- `llamacpp-bench (11.1s, 275.2 J)` ← **Main benchmark stage**
- `cool down (1.8s, 58.1 J)`

## Metrics Calculated

The profiler automatically calculates:
- **Energy consumed** (Joules): Integrated power over time using trapezoidal integration
- **Average power** (Watts): Energy divided by duration for each benchmark stage
- **Peak power** (Watts): Maximum power draw during the entire benchmark

## Comparison with External nvidia-smi Script

### Before (External Script):
- Requires manual bash script wrapper
- Separate CSV files for power and benchmark data
- Manual correlation of power data with benchmark stages

### After (Built-in Profiler):
- Single command with `--power-nvidia` flag
- Automatic correlation of power with benchmark stages
- Integrated reporting in lemonade output
- Energy-per-stage calculations
- Professional plots with stage annotations

## Requirements

- Linux or Windows with NVIDIA GPU
- NVIDIA drivers installed
- pynvml Python package
- Works with any CUDA-capable NVIDIA GPU

## Troubleshooting

### "pynvml is not installed"
```bash
pip install pynvml
```

### "No NVIDIA GPUs found"
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Check GPU is visible to the system

### "GPU index X is out of range"
- Check available GPUs: `nvidia-smi -L`
- Adjust `GPU_DEVICE_INDEX` environment variable

### FutureWarning about pynvml deprecation
- This is a harmless warning from pynvml
- The library works correctly despite the warning
- You can alternatively install `nvidia-ml-py` directly

## Integration with lemonade report

Power usage data is automatically integrated into lemonade's reporting system. Use:

```bash
lemonade report --perf
```

To export power usage statistics along with other performance metrics to CSV.
