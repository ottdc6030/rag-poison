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

