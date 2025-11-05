

import os
import platform
import textwrap
import time
import threading
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lemonade.common.printing as printing
from lemonade.profilers import Profiler
from lemonade.tools.report.table import LemonadePerfTable, DictListStat

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

DEFAULT_TRACK_POWER_INTERVAL_S = 0.05  # Sample every 1 second
DEFAULT_TRACK_POWER_WARMUP_PERIOD = 5  # 5 second warmup

POWER_USAGE_CSV_FILENAME = "power_usage_nvidia.csv"
POWER_USAGE_PNG_FILENAME = "power_usage_nvidia.png"


class Keys:
    # Path to the file containing the power usage plot
    POWER_USAGE_PLOT = "power_usage_plot_nvidia"
    # Power usage data
    POWER_USAGE_DATA = "power_usage_data_nvidia"
    # Path to the CSV file containing the power usage data
    POWER_USAGE_DATA_CSV = "power_usage_data_file_nvidia"

    # pynvml metrics
    PYNVML_PEAK_GPU_POWER = "pynvml_peak_gpu_power"
    PYNVML_AVG_GPU_POWER = "pynvml_avg_gpu_power"
    PYNVML_PEAK_GPU_TEMP = "pynvml_peak_gpu_temp"
    PYNVML_AVG_GPU_TEMP = "pynvml_avg_gpu_temp"

    # nvidia-smi metrics
    SMI_PEAK_GPU_POWER = "smi_peak_gpu_power"
    SMI_AVG_GPU_POWER = "smi_avg_gpu_power"
    SMI_PEAK_GPU_TEMP = "smi_peak_gpu_temp"
    SMI_AVG_GPU_TEMP = "smi_avg_gpu_temp"


# Add column to the Lemonade performance report table for the power data
LemonadePerfTable.table_descriptor["stat_columns"].append(
    DictListStat(
        "Power Usage (NVIDIA)",
        Keys.POWER_USAGE_DATA,
        [
            ("name", "{0}:"),
            ("duration", "{0:.1f}s,"),
            ("energy consumed", "{0:.1f} J"),
        ],
    )
)


class NVIDIAPowerProfiler(Profiler):

    unique_name = "power-nvidia"

    @staticmethod
    def add_arguments_to_parser(parser):
        parser.add_argument(
            f"--{NVIDIAPowerProfiler.unique_name}",
            nargs="?",
            metavar="WARMUP_PERIOD",
            type=int,
            default=None,
            const=DEFAULT_TRACK_POWER_WARMUP_PERIOD,
            help="Track NVIDIA GPU power consumption using NVML (NVIDIA Management Library) "
            "and plot the results. Requires pynvml Python package (pip install pynvml). "
            "Optionally, set the warmup period in seconds "
            f"(default: {DEFAULT_TRACK_POWER_WARMUP_PERIOD}). "
            "This works on Linux/Windows systems with NVIDIA GPUs. "
            "You can optionally set the GPU_DEVICE_INDEX environment variable to select "
            "a specific GPU (default: 0).",
        )

    def __init__(self, parser_arg_value):
        super().__init__()
        self.warmup_period = parser_arg_value
        self.status_stats += [
            Keys.PYNVML_PEAK_GPU_POWER,
            Keys.PYNVML_AVG_GPU_POWER,
            Keys.PYNVML_PEAK_GPU_TEMP,
            Keys.PYNVML_AVG_GPU_TEMP,
            Keys.SMI_PEAK_GPU_POWER,
            Keys.SMI_AVG_GPU_POWER,
            Keys.SMI_PEAK_GPU_TEMP,
            Keys.SMI_AVG_GPU_TEMP,
            Keys.POWER_USAGE_PLOT,
        ]
        self.tracking_active = False
        self.build_dir = None
        self.csv_path = None
        self.data = None
        self.power_data = []
        self.nvidia_smi_data = []
        self.monitoring_thread = None
        self.nvidia_smi_thread = None
        self.gpu_handle = None
        self.gpu_index = int(os.getenv("GPU_DEVICE_INDEX", "0"))

    def _monitor_power(self):
        """Background thread that monitors GPU power consumption using pynvml."""
        start_time = time.time()

        while self.tracking_active:
            try:
                current_time = time.time() - start_time

                # Get GPU metrics
                power_draw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert mW to W
                temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = utilization.gpu
                mem_util = utilization.memory

                # Get clock speeds
                # try:
                #     gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
                #     mem_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
                # except pynvml.NVMLError:
                #     gpu_clock = 0
                #     mem_clock = 0

                # Store data
                self.power_data.append({
                    'time': current_time,
                    'power_draw': power_draw,
                    'temperature': temperature,
                    'gpu_utilization': gpu_util,
                    # 'memory_utilization': mem_util,
                    # 'gpu_clock': gpu_clock,
                    # 'memory_clock': mem_clock,
                })

                time.sleep(DEFAULT_TRACK_POWER_INTERVAL_S)

            except pynvml.NVMLError as e:
                printing.log_info(f"Error reading GPU metrics: {e}")
                break

    def _monitor_nvidia_smi(self):
        """Background thread that monitors GPU power and temperature using nvidia-smi."""
        start_time = time.time()

        while self.tracking_active:
            try:
                current_time = time.time() - start_time

                # Use nvidia-smi to get power and temperature
                # Format: power.draw,temperature.gpu
                cmd = [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    # Parse output: "power, temperature"
                    values = result.stdout.strip().split(',')
                    if len(values) == 2:
                        power_draw = float(values[0].strip())
                        temperature = float(values[1].strip())

                        self.nvidia_smi_data.append({
                            'time': current_time,
                            'power_draw': power_draw,
                            'temperature': temperature,
                        })

                time.sleep(DEFAULT_TRACK_POWER_INTERVAL_S)

            except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
                printing.log_info(f"Error reading nvidia-smi metrics: {e}")
                break

    def start(self, build_dir):
        if self.tracking_active:
            raise RuntimeError("Cannot start power tracking while already tracking")

        if not PYNVML_AVAILABLE:
            raise RuntimeError(
                "pynvml is not installed. Please install it with: pip install pynvml"
            )

        # Save the folder where data and plot will be stored
        self.build_dir = build_dir

        # The csv file where power data will be stored
        self.csv_path = os.path.join(build_dir, POWER_USAGE_CSV_FILENAME)

        # Initialize NVML
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                raise RuntimeError("No NVIDIA GPUs found on this system.")

            if self.gpu_index >= device_count:
                raise RuntimeError(
                    f"GPU index {self.gpu_index} is out of range. "
                    f"Found {device_count} GPU(s). Use GPU_DEVICE_INDEX environment variable."
                )

            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)

            printing.log_info(f"Monitoring GPU {self.gpu_index}: {gpu_name}")

        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}")

        # Start monitoring in background threads
        self.tracking_active = True
        self.power_data = []
        self.nvidia_smi_data = []

        # Start pynvml monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_power, daemon=True)
        self.monitoring_thread.start()

        # Start nvidia-smi monitoring thread
        self.nvidia_smi_thread = threading.Thread(target=self._monitor_nvidia_smi, daemon=True)
        self.nvidia_smi_thread.start()

        # Warmup period
        time.sleep(self.warmup_period)

    def stop(self):
        if self.tracking_active:
            self.tracking_active = False

            # Wait for monitoring threads to finish
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            if self.nvidia_smi_thread:
                self.nvidia_smi_thread.join(timeout=5)

            # Cooldown period
            time.sleep(self.warmup_period)

            # Cleanup NVML
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def generate_results(self, state, timestamp, start_times):
        if not self.power_data:
            printing.log_info("No power data collected")
            state.save_stat(Keys.POWER_USAGE_PLOT, "NONE")
            return

        if self.tracking_active:
            self.stop()

        # Convert power data to DataFrame
        df = pd.DataFrame(self.power_data)

        # Save CSV
        df.to_csv(self.csv_path, index=False)

        # Remap time to start at 0 when first tool starts
        if start_times:
            tool_start_times = sorted(start_times.values())
            # First tool after warmup (if no tools, then will be time of start of cool down)
            first_tool_time = tool_start_times[1]

            # Find the offset in our power data
            initial_power_time = df['time'].iloc[0]
            time_offset = first_tool_time - time.time() + initial_power_time

            # For simplicity, just make the first measurement time 0
            df['time'] = df['time'] - df['time'].iloc[0]

        # Calculate statistics from pynvml data
        pynvml_peak_power = max(df['power_draw'])
        pynvml_peak_temp = max(df['temperature'])
        pynvml_avg_power = df['power_draw'].mean()
        pynvml_avg_temp = df['temperature'].mean()

        printing.log_info(f"pynvml: Peak Power={pynvml_peak_power:.1f}W, "
                        f"Avg Power={pynvml_avg_power:.1f}W, "
                        f"Peak Temp={pynvml_peak_temp:.1f}°C, "
                        f"Avg Temp={pynvml_avg_temp:.1f}°C")

        # Calculate statistics from nvidia-smi data if available
        smi_peak_power = None
        smi_peak_temp = None
        smi_avg_power = None
        smi_avg_temp = None

        if self.nvidia_smi_data:
            smi_df = pd.DataFrame(self.nvidia_smi_data)
            smi_peak_power = max(smi_df['power_draw'])
            smi_peak_temp = max(smi_df['temperature'])
            smi_avg_power = smi_df['power_draw'].mean()
            smi_avg_temp = smi_df['temperature'].mean()

            printing.log_info(f"nvidia-smi: Peak Power={smi_peak_power:.1f}W, "
                            f"Avg Power={smi_avg_power:.1f}W, "
                            f"Peak Temp={smi_peak_temp:.1f}°C, "
                            f"Avg Temp={smi_avg_temp:.1f}°C")
        else:
            printing.log_info("nvidia-smi data not available")

        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 8))

        if start_times:
            tool_starts = sorted(start_times.items(), key=lambda item: item[1])
            tool_name_list = [item[0] for item in tool_starts]

            # Adjust to common time frame as power measurements
            tool_start_list = [
                max(df['time'].iloc[0], item[1] - (tool_starts[1][1] if len(tool_starts) > 1 else 0))
                for item in tool_starts
            ]
            tool_stop_list = tool_start_list[1:] + [df['time'].values[-1]]

            # Extract power data time series
            x_time = df['time'].to_numpy()
            y_power = df['power_draw'].to_numpy()

            # Extract data for each stage in the build
            self.data = []
            for name, t0, tf in zip(tool_name_list, tool_start_list, tool_stop_list):
                x = x_time[(x_time >= t0) * (x_time <= tf)]
                if len(x) == 0:
                    continue
                x = np.insert(x, 0, t0)
                x = np.insert(x, len(x), tf)
                y = np.interp(x, x_time, y_power)
                energy = np.trapz(y, x)
                avg_power = energy / (tf - t0) if (tf - t0) > 0 else 0
                stage = {
                    "name": name,
                    "t": x.tolist(),
                    "power": y.tolist(),
                    "duration": float(tf - t0),
                    "energy consumed": float(energy),
                    "average power": float(avg_power),
                }
                self.data.append(stage)

            # Plot power usage for each stage
            for stage in self.data:
                p = ax1.plot(
                    stage["t"],
                    stage["power"],
                    label=f"{stage['name']} ({stage['duration']:.1f}s, "
                    f"{stage['energy consumed']:0.1f} J)",
                )
                # Add a dashed line to show average power
                ax1.plot(
                    [stage["t"][0], stage["t"][-1]],
                    [stage["average power"], stage["average power"]],
                    linestyle="--",
                    c=p[0].get_c(),
                )
                # Add average power text to plot
                ax1.text(
                    stage["t"][0],
                    stage["average power"],
                    f"{stage['average power']:.1f} W ",
                    horizontalalignment="right",
                    verticalalignment="center",
                    c=p[0].get_c(),
                )
        else:
            ax1.plot(df['time'], df['power_draw'])

        # Draw prompt and iteration boundary lines if benchmark data is available
        # import lemonade.common.filesystem as fs
        # stats = fs.Stats(state.cache_dir, state.build_name)
        # if 'bench_prompt_boundaries' in stats.stats:
        #     prompt_boundaries = stats.stats['bench_prompt_boundaries']
        #     if prompt_boundaries and start_times:
        #         # Get the benchmark start time offset
        #         bench_start_time = start_times.get('llamacpp-bench', None)
        #         first_tool_time = sorted(start_times.values())[1] if len(start_times) > 1 else None

        #         if bench_start_time and first_tool_time:
        #             # Calculate the time offset between power measurements and benchmark start
        #             time_offset = bench_start_time - first_tool_time

        #             # Draw vertical lines for each prompt boundary and estimate iteration boundaries
        #             for i, boundary in enumerate(prompt_boundaries):
        #                 # Calculate the time in power plot coordinates for this prompt
        #                 prompt_start_time = boundary['timestamp'] - bench_start_time + time_offset

        #                 # Calculate end time for this prompt (start of next prompt or end of benchmark)
        #                 if i < len(prompt_boundaries) - 1:
        #                     next_boundary = prompt_boundaries[i + 1]
        #                     prompt_end_time = next_boundary['timestamp'] - bench_start_time + time_offset
        #                 else:
        #                     # Last prompt - use end of benchmark
        #                     prompt_end_time = df['time'].iloc[-1]

        #                 # Calculate duration for this prompt and estimate iteration duration
        #                 prompt_duration = prompt_end_time - prompt_start_time
        #                 num_iterations = boundary['iterations']
        #                 iteration_duration = prompt_duration / num_iterations if num_iterations > 0 else 0

        #                 # Only draw lines if within the plot range
        #                 if df['time'].iloc[0] <= prompt_start_time <= df['time'].iloc[-1]:
        #                     # Solid red line for prompt boundary (on all 3 plots)
        #                     ax1.axvline(x=prompt_start_time, color='red', linestyle='-',
        #                                linewidth=2, alpha=0.8, zorder=10)
        #                     ax2.axvline(x=prompt_start_time, color='red', linestyle='-',
        #                                linewidth=2, alpha=0.8, zorder=10)
        #                     ax3.axvline(x=prompt_start_time, color='red', linestyle='-',
        #                                linewidth=2, alpha=0.8, zorder=10)

        #                     # Add text annotation showing prompt size on first plot
        #                     y_pos = ax1.get_ylim()[1] * 0.95  # 95% of y-axis height
        #                     ax1.text(prompt_start_time, y_pos, f" {boundary['prompt_size']}tok",
        #                             rotation=90, verticalalignment='top',
        #                             fontsize=9, color='red', fontweight='bold')

        #                     # Draw dashed gray lines for iteration boundaries within this prompt
        #                     if num_iterations > 1 and iteration_duration > 0:
        #                         for iter_num in range(1, num_iterations):
        #                             iter_time = prompt_start_time + (iter_num * iteration_duration)
        #                             if df['time'].iloc[0] <= iter_time <= df['time'].iloc[-1]:
        #                                 # Dashed gray line for iteration boundaries (on all 3 plots)
        #                                 ax1.axvline(x=iter_time, color='gray', linestyle='--',
        #                                            linewidth=1, alpha=0.5, zorder=5)
        #                                 ax2.axvline(x=iter_time, color='gray', linestyle='--',
        #                                            linewidth=1, alpha=0.5, zorder=5)
        #                                 ax3.axvline(x=iter_time, color='gray', linestyle='--',
        #                                            linewidth=1, alpha=0.5, zorder=5)

        # Add title and labels to first plot
        ax1.set_ylabel("GPU Power Draw [W]")
        title_str = "NVIDIA GPU Power Stats\n" + "\n".join(textwrap.wrap(state.build_name, 60))
        ax1.set_title(title_str)
        ax1.legend()
        ax1.grid(True)

        # Second plot: Clock speeds and temperature
        # ax2.plot(df['time'], df['gpu_clock'], label='GPU Clock [MHz]')
        # ax2.plot(df['time'], df['memory_clock'], label='Memory Clock [MHz]')
        # ax2.set_ylabel('Clock Frequency [MHz]')
        # ax2.legend(loc=2)
        # ax2.grid(True)

        # Add second y-axis for temperature
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df['time'], df['temperature'], label='Temperature [°C]', c='r')
        ax2_twin.set_ylabel('Temperature [°C]')
        ax2_twin.legend(loc=1)

        # Third plot: GPU and memory utilization
        ax3.plot(df['time'], df['gpu_utilization'], label='GPU Utilization [%]')
        #ax3.plot(df['time'], df['memory_utilization'], label='Memory Utilization [%]')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Utilization [%]')
        ax3.set_ylim([0, 100])
        ax3.legend()
        ax3.grid(True)

        # Save plot to current folder AND save to cache
        plot_path = os.path.join(
            self.build_dir, f"{timestamp}_{POWER_USAGE_PNG_FILENAME}"
        )
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_path = os.path.join(os.getcwd(), f"{timestamp}_{POWER_USAGE_PNG_FILENAME}")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        state.save_stat(Keys.POWER_USAGE_PLOT, plot_path)
        state.save_stat(Keys.POWER_USAGE_DATA, self.data)
        state.save_stat(Keys.POWER_USAGE_DATA_CSV, self.csv_path)

        # Save pynvml statistics
        state.save_stat(Keys.PYNVML_PEAK_GPU_POWER, f"{pynvml_peak_power:0.1f} W")
        state.save_stat(Keys.PYNVML_AVG_GPU_POWER, f"{pynvml_avg_power:0.1f} W")
        state.save_stat(Keys.PYNVML_PEAK_GPU_TEMP, f"{pynvml_peak_temp:0.1f} °C")
        state.save_stat(Keys.PYNVML_AVG_GPU_TEMP, f"{pynvml_avg_temp:0.1f} °C")

        # Save nvidia-smi statistics if available
        if smi_peak_power is not None:
            state.save_stat(Keys.SMI_PEAK_GPU_POWER, f"{smi_peak_power:0.1f} W")
            state.save_stat(Keys.SMI_AVG_GPU_POWER, f"{smi_avg_power:0.1f} W")
            state.save_stat(Keys.SMI_PEAK_GPU_TEMP, f"{smi_peak_temp:0.1f} °C")
            state.save_stat(Keys.SMI_AVG_GPU_TEMP, f"{smi_avg_temp:0.1f} °C")
        else:
            state.save_stat(Keys.SMI_PEAK_GPU_POWER, "N/A")
            state.save_stat(Keys.SMI_AVG_GPU_POWER, "N/A")
            state.save_stat(Keys.SMI_PEAK_GPU_TEMP, "N/A")
            state.save_stat(Keys.SMI_AVG_GPU_TEMP, "N/A")


# Copyright (c) 2025 AMD
