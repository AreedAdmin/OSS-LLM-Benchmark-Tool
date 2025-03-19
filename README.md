Below is an example `README.md` file tailored for your script:

```markdown
# Ollama Benchmark

This script benchmarks the performance of models available via the Ollama command-line tool. It sends a specified prompt to a chosen model repeatedly, measuring the time taken and token output for each iteration, and then summarizes the overall results by reporting the total tokens generated, total elapsed time, and average tokens per second (TPS).

## Features

- **Model Flexibility:** Easily run benchmarks on any model available in your Ollama installation by specifying the model name through the command line.
- **Customizable Prompt:** Provide any prompt to test how the model responds.
- **Multiple Iterations:** Execute a configurable number of iterations to obtain robust average performance metrics.
- **Token Counting:** Uses a simple whitespace-based token counting method to estimate the number of tokens generated.
- **Performance Metrics:** Outputs per-iteration statistics along with aggregate results (total tokens, total time, and average TPS).

## Requirements

- **Python 3.x**
- **Ollama CLI:** Ensure that the `ollama` command-line tool is installed and configured.
- Standard Python libraries: `subprocess`, `time`, and `argparse` (included in the Python standard library).

## Usage

Run the script from the command line with your desired parameters. For example:

```bash
python ollama_benchmark.py --model deepseek-r1:1.5b --prompt "How many r's are clear?" --iterations 10
```

### Command-line Arguments

- `--model`: The name of the model to benchmark (default: `deepseek-r1:7b`).
- `--prompt`: The prompt to send to the model (default: `"Please generate a summary of the latest research trends in AI."`).
- `--iterations`: The number of benchmark iterations to run (default: `10`).

## How It Works

1. **Argument Parsing:** Uses `argparse` to capture the model name, prompt, and number of iterations from the command line.
2. **Running the Benchmark:** For each iteration, the script invokes the Ollama CLI using `subprocess.run()` with the specified model and prompt.
3. **Token Counting:** Counts tokens in the model's output by splitting the text on whitespace.
4. **Performance Calculation:** Records the elapsed time for each iteration and computes the average tokens per second (TPS) across all iterations.
5. **Results Display:** After all iterations are complete, prints a summary of the benchmark results.

## Example Output

When you run the script, the output may look like this:

```
Running benchmark for model 'deepseek-r1:1.5b' with prompt:
"How many r's are clear?"

Iterations: 10
----------------------------------------
Iteration 1: 45 tokens, 1.20 seconds
Iteration 2: 47 tokens, 1.18 seconds
...
Iteration 10: 46 tokens, 1.25 seconds

Benchmark Results:
Total tokens: 460
Total time: 12.30 seconds
Average tokens per second (TPS): 37.40
```

## License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) [2023] Shehab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.]

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

For any questions or feedback, please contact:
- **Name:** Shehab
- **Email:** [Shehab.hassani@areednow.com](mailto:Shehab.hassani@areednow.com)
```
