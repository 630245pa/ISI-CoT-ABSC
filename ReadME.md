# LLM Prompting Strategies for Implicit Aspect-Based Sentiment Classification

This repository implements and evaluates a family of prompting strategies including a ** iterative Self-Improvement (ISI)** loop — for Aspect-Based Sentiment Classification (ABSC). Experiments are run against locally-hosted LLMs (via [Ollama](https://ollama.com/)) on the SemEval-2014 Restaurants/Laptops datasets.

---

## Methods Implemented

| Method | Description |
|---|---|
| `vanilla` | Direct zero-shot polarity prediction, no reasoning. |
| `zs_cot` | Zero-shot Chain-of-Thought ("let's think step by step"). |
| `zs_scot` | Structured CoT: opinion cue → justification → polarity. |
| `zs_hcot` | Syntax-guided CoT using a spaCy-derived CoNLL-U dependency parse. |
| `fs_cot` | Few-shot CoT with BM25/random/SimCSE demonstration retrieval from a pre-built demo bank. |
| `*_isi` variants | Each method above extended with a **Iterative Self-Improvement loop**: an LLM critiques its own (reasoning, prediction) against explicit criteria and re-executes from the earliest broken reasoning step, for up to *T* iterations. |

All ISI variants reuse the corresponding base method's cached outputs as their starting point rather than re-querying the LLM, to save compute.

---

## Repository Structure

```
.
├── data_loader.py         # Parses SemEval ABSA XML into aspect-term records
├── sentiment_predictor.py # All prompting methods (LLM prompt construction + parsing)
├── fs_cot.py              # Few-shot demo bank, retrieval (random/BM25/SimCSE), chain generation
├── eval.py                # Runner (checkpointing, parallel execution) + Evaluator (accuracy/F1)
├── main.py                # End-to-end pipeline entry point
└── demo_bank.py            # Ad-hoc script for building the demo bank / debugging
```

---

## How It Works

1. **`DataLoader`** parses a SemEval ABSA XML file into aspect-term records, filtering to those with a labeled `implicit_sentiment` attribute.
2. **`SentimentPredictor`** dispatches each item to the requested method via a `predict()` entry point, prompting the LLM (Ollama, or optionally Groq) and parsing its output into one of `positive` / `neutral` / `negative`.
3. **`Runner`** executes a method over the full dataset with checkpointing (resumable, crash-safe JSON checkpoints) and optional multi-threaded parallel execution.
4. **`Evaluator`** computes accuracy, macro/weighted F1, ISE (implicit) vs. ESE (explicit) sentiment breakdowns, and iteration statistics for SI methods.

---

## Usage

```python
from data_loader import DataLoader
from sentiment_predictor import SentimentPredictor
from eval import Evaluator, Runner

data = DataLoader("path/to/Restaurants_Test_Gold_Implicit_Labeled.xml").load_data()

predictor = SentimentPredictor(model="mistral")   # any Ollama-served model
runner    = Runner(predictor, output_dir="results")
evaluator = Evaluator()

results = runner.run_all(data, methods=["vanilla", "zs_scot", "zs_hcot", "zs_cot_si"])

evaluator.print_compare(results)
for method, r in results.items():
    evaluator.print_by_split(r, label=method)
```

> **Note:** `*_si` methods require a completed checkpoint of their base method first (e.g. run `zs_scot` to completion before `zs_scot_si`), since SI reuses the base run as its starting cache.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with the desired model pulled
- `spacy` (+ `en_core_web_sm`), `scikit-learn`, `tqdm`
- Optional: `rank-bm25` and/or `sentence-transformers` for `fs_cot` retrieval strategies
- Optional: `groq` + `python-dotenv` if using a hosted Groq model instead of Ollama

```bash
pip install ollama spacy scikit-learn tqdm rank-bm25 sentence-transformers groq python-dotenv
python -m spacy download en_core_web_sm
```

---

## Data

Expects SemEval-2014 style ABSA XML with an `implicit_sentiment` attribute on each `aspectTerm` (`True`/`False`), e.g. the Restaurants/Laptops test and training sets used for demo-bank construction in `fs_cot.py`.
