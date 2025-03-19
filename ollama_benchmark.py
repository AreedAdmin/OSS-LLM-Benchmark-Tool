import subprocess
import time
import argparse

def count_tokens(text):
    """
    Naively count tokens by splitting the text on whitespace.
    For more accurate token counts, integrate a proper tokenizer.
    """
    tokens = text.strip().split()
    return len(tokens)

def run_benchmark(model, prompt, iterations):
    total_time = 0.0
    total_tokens = 0

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
        
        total_time += elapsed
        total_tokens += tokens

        print(f"Iteration {i+1}: {tokens} tokens, {elapsed:.2f} seconds")
    
    if total_time > 0:
        average_tps = total_tokens / total_time
    else:
        average_tps = 0

    print("\nBenchmark Results for model '{}' :".format(model))
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average tokens per second (TPS): {average_tps:.2f}")

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

    # Split models by comma and strip any whitespace
    model_list = [m.strip() for m in args.models.split(",")]

    # Run benchmark for each model in the list
    for model in model_list:
        run_benchmark(model, args.prompt, args.iterations)

if __name__ == "__main__":
    main()
