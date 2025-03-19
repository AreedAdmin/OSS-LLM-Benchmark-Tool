import subprocess
import time
import argparse
import csv
import os
import uuid

def count_tokens(text):
    """
    Naively count tokens by splitting the text on whitespace.
    For more accurate token counts, integrate a proper tokenizer.
    """
    tokens = text.strip().split()
    return len(tokens)

def run_benchmark(model, prompt, iterations):
    iteration_tokens = []
    iteration_times = []
    sample_output = None

    print(f"\n{'='*50}")
    print(f"Running benchmark for model '{model}' with prompt:")
    print(f"\"{prompt}\"\n")
    print(f"Iterations: {iterations}\n{'-'*50}")

    for i in range(iterations):
        start = time.time()
        # Run the 'ollama run' command with the specified model and prompt.
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True
        )
        end = time.time()

        elapsed = end - start
        output_text = result.stdout.strip()
        tokens = count_tokens(output_text)

        # Save sample output from first iteration (in one line)
        if i == 0:
            sample_output = output_text.replace("\n", " ")

        iteration_tokens.append(tokens)
        iteration_times.append(elapsed)

        print(f"Iteration {i+1}: {tokens} tokens, {elapsed:.2f} seconds")
    
    total_time = sum(iteration_times)
    total_tokens = sum(iteration_tokens)
    avg_token_count = total_tokens / iterations if iterations > 0 else 0
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    avg_inference_time = total_time / iterations if iterations > 0 else 0

    # Calculate token range as a percentage of the maximum token count.
    if iteration_tokens and max(iteration_tokens) != 0:
        min_tokens = min(iteration_tokens)
        max_tokens = max(iteration_tokens)
        min_max_variance_pct = ((max_tokens - min_tokens) / max_tokens) * 100
    else:
        min_max_variance_pct = 0

    # Print summary statistics
    print("\nBenchmark Results for model '{}' :".format(model))
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average tokens per iteration: {avg_token_count:.2f}")
    print(f"Token Range Variance (%): {min_max_variance_pct:.2f}%")
    print(f"Average tokens per second (TPS): {avg_tps:.2f}")
    print(f"Average inference time per iteration: {avg_inference_time:.2f} seconds")

    # Return summary stats as a dictionary
    summary = {
        "model_name": model,
        "iteration_no": iterations,
        "average_token_count": avg_token_count,
        "min_max_token_variance_%": min_max_variance_pct,
        "average_tps": avg_tps,
        "average_inference_time": avg_inference_time,
        "sample_output": sample_output
    }
    return summary

def log_benchmark_data(prompt_id, summary, csv_filename="raw_data.csv"):
    # If the file does not exist, write the header.
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "prompt_id", "model_name", "iteration_no", "average_token_count",
            "min_max_token_variance_%", "average_tps", "average_inference_time", "sample_output"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {"prompt_id": prompt_id}
        row.update(summary)
        writer.writerow(row)
    print(f"\nLogged benchmark data for prompt_id {prompt_id} into '{csv_filename}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the performance of one or more Ollama models by running repeated inference requests."
    )
    parser.add_argument("--models", type=str, default="deepseek-r1:7b",
                        help="Comma-separated list of model names to run (default: deepseek-r1:7b)")
    parser.add_argument("--prompt", type=str, default="Please generate a summary of the latest research trends in AI.",
                        help="Test prompt to send to the model(s)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations to run per model")
    args = parser.parse_args()

    # Generate a unique prompt ID
    prompt_id = str(uuid.uuid4())

    # Split models by comma and strip any whitespace
    model_list = [m.strip() for m in args.models.split(",")]

    # For each model, run benchmark and log the summary data
    for model in model_list:
        summary = run_benchmark(model, args.prompt, args.iterations)
        log_benchmark_data(prompt_id, summary)

if __name__ == "__main__":
    main()
