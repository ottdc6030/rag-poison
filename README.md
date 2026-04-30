# rag-poison

A small-scale RAG (Retrieval-Augmented Generation) testing framework using [Ollama](https://ollama.com/) for local LLM inference and LlamaIndex for document indexing.

## Prerequisites

- Python 3
- [Ollama](https://ollama.com/) installed on your system

---

## Setup

### 1. Start Ollama in the background

Ollama must be running before any queries can be made. Run this once per session:

```bash
ollama serve > /dev/null 2>&1 &
```

### 2. Install a model into Ollama

The Python scripts cannot install models — you must add them to Ollama manually before running any queries.

**To pull a model from the Ollama library:**
```bash
ollama pull <model-name>
# e.g. ollama pull llama3
```

**To add a locally-downloaded model using a Modelfile:**
```bash
ollama create <model-name> -f ./Modelfile
```

You can verify available models with:
```bash
ollama list
```

### 3. Create the Python virtual environment

Run this once to set up the virtual environment and install dependencies:

```bash
bash make_venv.sh
```

Then activate it:

```bash
source llama_venv/bin/activate
```

---

## Usage

### `zero-shot.py` — One-off RAG query

Loads all files from a local data folder, builds a vector index, and runs a single predefined query against it. Useful for quick small-scale tests with a set of attachment files.

```bash
python zero-shot.py <model-name> [--data_folder <path>] [--embedding <hf-model>]
```

| Argument | Description | Default |
|---|---|---|
| `llm` | Name of an Ollama model (must already be installed) | *(required)* |
| `--data_folder` | Path to the folder containing documents to query | `data` |
| `--embedding` | HuggingFace embedding model for semantic search | `BAAI/bge-small-en-v1.5` |

**Example:**
```bash
python zero-shot.py llama3 --data_folder ./data
```

Place the files you want to query (e.g. resumes, PDFs, text files) in the `data/` folder before running.

---

### `rag_poison_test.py` — Iterative RAG poisoning harness

Runs a multi-turn red-team experiment in which an **adversary model** tries to hijack a **target model** via a poisoned RAG document. Each attempt, the adversary crafts hidden directives and embeds them into a document. The target loads the poisoned document and answers a benign user prompt. The adversary then judges whether the attack succeeded; if not, it reflects on the failure and tries again. Successful attacks are recorded to `successes.csv`.

Two modes are supported:
- **Poisoned file** (default): the adversary reads an existing benign document and appends a hidden injection to it.
- **Standalone** (`--standalone`): the adversary writes an entirely adversarial file from scratch, with no benign source document required.

```bash
python rag_poison_test.py <adversary_model> <target_model> <poisoned_file> <poison_goal> <target_prompt> [benign_file] [options]
```

| Argument | Description |
|---|---|
| `adversary_model` | Ollama model that crafts the injections and evaluates results |
| `target_model` | Ollama model being attacked |
| `poisoned_file` | Path where the poisoned document is written on each attempt |
| `poison_goal` | Text file describing what the adversary wants the target to do |
| `target_prompt` | Text file containing the benign query the target believes it is answering |
| `benign_file` | Source document to poison (omit when using `--standalone`) |

| Option | Description | Default |
|---|---|---|
| `--embedding` | HuggingFace embedding model for RAG | `BAAI/bge-small-en-v1.5` |
| `--max_attempts` | Maximum poisoning attempts before giving up | `100` |
| `--top_k` | Document chunks retrieved per RAG query | `4000` |
| `--max_doc_chars` | Max characters of the benign document sent to the adversary | `4000` |
| `--standalone` | Adversary writes the poisoned file from scratch; `benign_file` is not used | — |
| `--keep_target_thinking` | Pass the target's raw `<think>` blocks to the adversary instead of stripping them | — |
| `--injection_position` | Where to insert the injection: `-1` = append, `0` = prepend, `N` = after line N | `-1` |
| `--evaluator` | Which model judges whether an injection succeeded (`adversary` or `target`) | `adversary` |
| `--critic` | Which model reflects on failures and suggests improvements (`adversary` or `target`) | `adversary` |

**Example — poison an existing file:**
```bash
python rag_poison_test.py llama3 llama3 poisoned/people_poisoned.csv badgoal.txt goodgoal.txt data/people.csv
```

**Example — standalone mode (no benign file):**
```bash
python rag_poison_test.py llama3 llama3 poisoned/standalone.txt badgoal.txt goodgoal.txt --standalone
```

Successful injections are appended to `successes.csv` with the injection text, attempt number, and full target response.

---

### `injection_tester.py` — Injection robustness tester

Reads a CSV of pre-crafted poison prompts and measures how reliably each one hijacks a target model. For every prompt, the injection is embedded into a benign file and the target is queried `TRIALS_PER_PROMPT` (default: 20) times. A separate evaluator LLM judges each response and the per-prompt hijack rate is reported.

The input CSV must have a `poison_prompt` column. An optional `poison_goal` column enables more precise evaluation; without it a generic hijack-detection prompt is used.

```bash
python injection_tester.py <poisoned_prompts_csv> <benign_file> <target_model> <benign_prompt_file> <output_csv> [options]
```

| Argument | Description |
|---|---|
| `poisoned_prompts_csv` | CSV with a `poison_prompt` column (and optionally a `poison_goal` column that represents the original goal of the adversary) |
| `benign_file` | Original document that each injection is embedded into |
| `target_model` | Ollama model being tested for resilience |
| `benign_prompt_file` | Text file containing the benign query the target is asked |
| `output_csv` | CSV file where results are appended (`poison_prompt`, `poison_goal`, `trial`, `response`, `defense_detected`, `hijacked`) |

| Option | Description | Default |
|---|---|---|
| `--defense` | Defense mechanism to apply before querying the target (`none`, `known_answer`, `bert`) | `none` |
| `--defense_model` | HuggingFace NLI model to use when `--defense bert` is set | `typeform/mobilebert-uncased-mnli` |
| `--adversary_model` | Ollama model used as the evaluator when `--evaluator adversary` is set | *(required if adversary evaluator)* |
| `--embedding` | HuggingFace embedding model for RAG | `BAAI/bge-small-en-v1.5` |
| `--top_k` | Document chunks retrieved per RAG query | `4000` |
| `--poison_file` | Path to write the temporary poisoned file | auto (temp file) |
| `--manual` | Human user evaluates each response (Y/N) instead of an LLM evaluator | — |
| `--injection_position` | Where to insert the injection: `-1` = append, `0` = prepend, `N` = after line N | `-1` |
| `--evaluator` | Which model judges whether an injection succeeded (`adversary` or `target`) | `adversary` |

**Example:**

```bash
python injection_tester.py statistics/lying-csv/round_1_successes.csv data/people.csv llama3 goodgoal.txt results.csv
```
```bash
python injection_tester.py statistics/lying-html/successes.csv data/resume.html llama3 goodgoal.txt results.csv
```

---

### `evaluate_bert_baseline.py` — BERT defense baseline check

Runs the BERT defense against a clean, un-poisoned file to measure whether it produces false positives. Useful for calibrating the defense threshold before deploying it in `injection_tester.py`.

```bash
python evaluate_bert_baseline.py <context_file> [--model <hf-model>]
```

| Argument | Description | Default |
|---|---|---|
| `context_file` | Un-poisoned file to scan | *(required)* |
| `--model` | HuggingFace NLI model to use | `typeform/mobilebert-uncased-mnli` |

**Example:**
```bash
python evaluate_bert_baseline.py data/people.csv
```

Prints the entailment score and a `blocked` / `allowed` verdict.

---

### `summarize_results.py` — Aggregate injection test results

Post-processes the output CSV from `injection_tester.py`. Groups rows by `(poison_prompt, poison_goal)` pair and computes the hijack rate for each injection across all trials.

```bash
python summarize_results.py <input_csv> <output_csv>
```

| Argument | Description |
|---|---|
| `input_csv` | Raw results CSV produced by `injection_tester.py` |
| `output_csv` | Summary CSV to write (`poison_prompt`, `poison_goal`, `total_trials`, `hijack_count`, `hijack_rate`) |

**Example:**
```bash
python summarize_results.py results.csv summary.csv
```
