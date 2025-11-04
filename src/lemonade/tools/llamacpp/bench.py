import argparse
import statistics
from statistics import StatisticsError
from lemonade.state import State
from lemonade.tools.tool import Tool
from lemonade.tools.llamacpp.utils import LlamaCppAdapter
from lemonade.tools.bench import (
    Bench,
    default_prompt_length,
    default_iterations,
    default_output_tokens,
    default_warmup_runs,
)


class LlamaCppBench(Bench):
    """
    Benchmark a llama.cpp model
    """

    unique_name = "llamacpp-bench"

    def __init__(self, monitor_message="Benchmarking LLM"):
        super().__init__(monitor_message)

        # Don't track memory usage since we are using a llamacpp executable for compute
        self.save_max_memory_used = False

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Benchmark an LLM in llama.cpp",
            add_help=add_help,
        )

        parser = Bench.parser(parser)

        parser.add_argument(
            "--cli",
            action="store_true",
            help="Set this flag to use llama-cli.exe to benchmark model performance. This executable will be called "
            "once per iteration.  Otherwise, llama-bench.exe is used by default.  In this default behavior behavior, "
            "the only valid prompt format is integer token lengths. Also, the warmup-iterations parameter is "
            "ignored and the default value for number of threads is 16.",
        )

        return parser

    def parse(self, state: State, args, known_only=True) -> argparse.Namespace:
        """
        Helper function to parse CLI arguments into the args expected by run()
        """
        import os

        # Call Tool parse method, NOT the Bench parse method
        parsed_args = Tool.parse(self, state, args, known_only)

        if parsed_args.cli:
            parsed_args = super().parse(state, args, known_only)
        else:
            # Make sure prompts is a list of integers
            if parsed_args.prompts is None:
                parsed_args.prompts = [default_prompt_length]

            # Check if any prompts are file paths or non-integer strings
            # If so, auto-enable CLI mode
            needs_cli_mode = False
            for prompt_item in parsed_args.prompts:
                if not prompt_item.isdigit():
                    # Check if it's a file path
                    if os.path.exists(prompt_item):
                        needs_cli_mode = True
                        break
                    # Also enable for non-integer strings (direct prompt text)
                    else:
                        needs_cli_mode = True
                        break

            if needs_cli_mode:
                # Auto-enable CLI mode and print warning
                print(f"\nWarning: Detected prompt file paths or text strings. Automatically enabling --cli mode.")
                print(f"    CLI mode uses llama-cli instead of llama-bench for better prompt support.\n")
                parsed_args.cli = True
                # Re-parse with Bench.parse() to handle file paths and strings properly
                parsed_args = super().parse(state, args, known_only)
            else:
                # All prompts are integers, proceed normally
                prompt_ints = []
                for prompt_item in parsed_args.prompts:
                    prompt_ints.append(int(prompt_item))
                parsed_args.prompts = prompt_ints

        return parsed_args

    def run_prompt(
        self,
        state: State,
        report_progress_fn,
        prompt: str,
        iterations: int,
        warmup_iterations: int,
        output_tokens: int,
    ) -> State:
        """
        Benchmark llama.cpp model that was loaded by LoadLlamaCpp.
        """

        if self.first_run_prompt:

            if not hasattr(state, "model") or not isinstance(
                state.model, LlamaCppAdapter
            ):
                raise Exception(
                    f"{self.__class__.unique_name} requires a LlamaCppAdapter model to be "
                    "loaded first. Please run load-llama-cpp before this tool."
                )
        model: LlamaCppAdapter = state.model

        per_iteration_tokens_per_second = []
        per_iteration_time_to_first_token = []

        for iteration in range(iterations + warmup_iterations):
            try:
                # Use the adapter's generate method which already has the timeout
                # and error handling
                model.time_to_first_token = None
                model.tokens_per_second = None
                raw_output, stderr = model.generate(
                    prompt, max_new_tokens=output_tokens, return_raw=True
                )

                if model.time_to_first_token is None or model.tokens_per_second is None:
                    error_msg = (
                        "Could not find timing information in llama.cpp output.\n"
                    )
                    error_msg += "Raw output:\n" + raw_output + "\n"
                    error_msg += "Stderr:\n" + stderr
                    raise Exception(error_msg)

                self.tokens_out_len_list.append(model.response_tokens)

                if iteration > warmup_iterations - 1:
                    per_iteration_tokens_per_second.append(model.tokens_per_second)
                    per_iteration_time_to_first_token.append(model.time_to_first_token)

                report_progress_fn((iteration + 1) / (warmup_iterations + iterations))

            except Exception as e:
                error_msg = f"Failed to run benchmark: {str(e)}"
                raise Exception(error_msg)

        self.input_ids_len_list.append(model.prompt_tokens)
        mean_time_to_first_token = statistics.mean(per_iteration_time_to_first_token)
        self.mean_time_to_first_token_list.append(mean_time_to_first_token)
        self.prefill_tokens_per_second_list.append(
            model.prompt_tokens / mean_time_to_first_token
        )
        self.token_generation_tokens_per_second_list.append(
            statistics.mean(per_iteration_tokens_per_second)
        )
        try:
            self.std_dev_time_to_first_token_list.append(
                statistics.stdev(per_iteration_time_to_first_token)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_time_to_first_token_list.append(None)
        try:
            self.std_dev_token_generation_tokens_per_second_list.append(
                statistics.stdev(per_iteration_tokens_per_second)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_token_generation_tokens_per_second_list.append(None)

    def run_llama_bench_exe(self, state, prompts, iterations, output_tokens):
        import time

        if prompts is None:
            prompts = [default_prompt_length]
        elif isinstance(prompts, int):
            prompts = [prompts]

        state.save_stat("prompts", prompts)
        state.save_stat("iterations", iterations)
        state.save_stat("output_tokens", output_tokens)

        model: LlamaCppAdapter = state.model

        # Initialize aggregation lists for collecting results from multiple benchmark runs
        all_prompt_lengths = []
        all_pp_tps = []
        all_pp_tps_sd = []
        all_tg_tps = []
        all_tg_tps_sd = []

        # Track prompt boundaries for power profiling visualization
        prompt_boundaries = []

        # Run benchmark separately for each prompt size to get per-prompt token generation speeds
        for prompt in prompts:
            # Record timestamp at the start of this prompt's benchmark
            prompt_start_time = time.time()
            prompt_boundaries.append({
                'prompt_size': prompt,
                'timestamp': prompt_start_time,
                'iterations': iterations
            })

            prompt_lengths, pp_tps, pp_tps_sd, tg_tps, tg_tps_sd = model.benchmark(
                [prompt], iterations, output_tokens
            )

            # Aggregate results from this prompt size
            all_prompt_lengths.extend(prompt_lengths)
            all_pp_tps.extend(pp_tps)
            all_pp_tps_sd.extend(pp_tps_sd)
            all_tg_tps.append(tg_tps)
            all_tg_tps_sd.append(tg_tps_sd)

        # Store aggregated results
        self.input_ids_len_list = all_prompt_lengths
        self.prefill_tokens_per_second_list = all_pp_tps
        if iterations > 1:
            self.std_dev_prefill_tokens_per_second_list = all_pp_tps_sd
        self.mean_time_to_first_token_list = [
            tokens / tps for tokens, tps in zip(all_prompt_lengths, all_pp_tps)
        ]
        self.token_generation_tokens_per_second_list = all_tg_tps
        if iterations > 1:
            self.std_dev_token_generation_tokens_per_second_list = all_tg_tps_sd
        self.tokens_out_len_list = [output_tokens] * len(prompts) * iterations

        # Save prompt boundaries for profiler visualization
        state.save_stat("bench_prompt_boundaries", prompt_boundaries)

        self.save_stats(state)
        return state

    def run(
        self,
        state: State,
        prompts: list[str] = None,
        iterations: int = default_iterations,
        warmup_iterations: int = default_warmup_runs,
        output_tokens: int = default_output_tokens,
        cli: bool = False,
        **kwargs,
    ) -> State:
        """
        Args:
            - prompts: List of input prompts used as starting points for LLM text generation
            - iterations: Number of benchmarking samples to take; results are
                reported as the median and mean of the samples.
            - warmup_iterations: Subset of the iterations to treat as warmup,
                and not included in the results.
            - output_tokens: Number of new tokens LLM to create.
            - ggml: Use llama-bench.exe directly
            - kwargs: Additional parameters used by bench tools
        """

        # Check that state has the attribute model and it is a LlamaCPP model
        if not hasattr(state, "model") or not isinstance(state.model, LlamaCppAdapter):
            raise Exception("Load model using llamacpp-load first.")

        if cli:
            state = super().run(
                state, prompts, iterations, warmup_iterations, output_tokens, **kwargs
            )
        else:
            state = self.run_llama_bench_exe(state, prompts, iterations, output_tokens)

        return state


# This file was originally licensed under Apache 2.0. It has been modified.
# Modifications Copyright (c) 2025 AMD
