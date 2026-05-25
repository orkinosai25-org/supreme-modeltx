# Dataset Overview — SMTX-Baby (T-101)

This document summarises the synthetic pretraining and SFT seed dataset located under `data/raw/`.  
All content is entirely original synthetic text. No real individuals, copyrighted material, personal information, or unsafe content is included.

---

## Directory Structure

```
data/raw/
├── general_text/        Plain-text encyclopedic paragraphs
├── wiki_style/          Synthetic wiki-style articles (.txt)
├── qa_pairs/            Question/answer pairs (.jsonl)
├── conversations/       Multi-turn assistant dialogues (.jsonl)
├── code_samples/        Synthetic code examples (.txt)
├── reasoning/           Chain-of-thought reasoning samples (.jsonl)
├── instructions/        Instruction → response pairs (.jsonl)
└── math/                Synthetic mathematics dataset (see below)
    ├── arithmetic/
    ├── algebra/
    ├── word_problems/
    ├── sequences/
    ├── reasoning_steps/
    └── explanations/
```

---

## Dataset Summary

| Folder | File(s) | Format | Count |
|---|---|---|---|
| `general_text/` | `corpus.txt` | Plain text | 50 paragraphs |
| `wiki_style/` | 7 × `.txt` articles | Plain text | 7 articles (~10 paragraphs each) |
| `qa_pairs/` | `qa_pairs.jsonl` | JSONL `{"question","answer"}` | 113 pairs |
| `conversations/` | `conversations.jsonl` | JSONL `{"conversation":[…]}` | 25 dialogues |
| `code_samples/` | 9 × `.txt` files | Plain text | 9 code examples |
| `reasoning/` | `reasoning.jsonl` | JSONL `{"question","reasoning","answer"}` | 30 samples |
| `instructions/` | `instructions.jsonl` | JSONL `{"instruction","response"}` | 30 samples |
| `math/arithmetic/` | `arithmetic_examples.txt` | Plain text | 1,000 examples |
| `math/algebra/` | `algebra_examples.txt` | Plain text | 600 examples |
| `math/word_problems/` | `word_problems.txt` | Plain text | 400 examples |
| `math/sequences/` | `sequences.txt` | Plain text | 300 examples |
| `math/reasoning_steps/` | `reasoning_steps.jsonl` | JSONL `{"question","reasoning","answer"}` | 400 samples |
| `math/explanations/` | `concept_explanations.txt` | Plain text | 15 concepts |

---

## Content Breakdown

### A. General Text (`general_text/corpus.txt`)
- 50 paragraphs of clean, neutral, encyclopedic-style English
- Topics: science, history, technology, agriculture, AI, UK infrastructure
- Format: plain UTF-8 text with blank-line paragraph separators

### B. Wiki-style Articles (`wiki_style/`)
| File | Topic |
|---|---|
| `Offshore_Wind_Energy.txt` | Offshore wind turbines, foundations, grid integration |
| `Distributed_Computing_Systems.txt` | CAP theorem, consensus, message queues |
| `Cloud_Computing_Fundamentals.txt` | IaaS/PaaS/SaaS, virtualisation, object storage |
| `Neural_Network_Architectures.txt` | CNNs, RNNs, transformers, residual connections |
| `Robotics_and_Automation.txt` | Industrial robots, sensors, path planning, drones |
| `Renewable_Energy_Storage.txt` | Batteries, pumped hydro, hydrogen, flow batteries |
| `British_AI_Innovation.txt` | UK AI ecosystem, research, safety, healthcare AI |

### C. QA Pairs (`qa_pairs/qa_pairs.jsonl`)
- 113 question/answer pairs
- Topics: mathematics, logic, computing, everyday reasoning, science, AI/ML
- Safe, factual, original answers

### D. Conversations (`conversations/conversations.jsonl`)
- 25 multi-turn dialogues
- Style: helpful AI assistant, step-by-step explanations
- Topics: programming, cloud, networking, mathematics, technology concepts
- No personal information, politics, or harmful content

### E. Code Samples (`code_samples/`)
| File | Language | Topic |
|---|---|---|
| `python_loops.txt` | Python | for, while, list comprehensions |
| `python_functions.txt` | Python | functions, recursion, defaults |
| `python_file_io.txt` | Python | file read/write, JSON |
| `python_classes.txt` | Python | OOP, inheritance |
| `csharp_loops.txt` | C# | for, while, foreach, do-while |
| `csharp_classes.txt` | C# | classes, properties, methods |
| `javascript_functions.txt` | JavaScript | arrow functions, closures, HOF |
| `javascript_async.txt` | JavaScript | Promises, async/await |
| `javascript_array_methods.txt` | JavaScript | map, filter, reduce, sort |

### F. Reasoning (`reasoning/reasoning.jsonl`)
- 30 chain-of-thought reasoning samples
- Format: `{"question": "…", "reasoning": "…", "answer": "…"}`
- Topics: arithmetic, logic puzzles, probability, combinatorics, geometry

### G. Instructions (`instructions/instructions.jsonl`)
- 30 instruction → response pairs
- Format: `{"instruction": "…", "response": "…"}`
- Topics: cloud computing, ML/AI, software engineering, databases, DevOps

### H. Synthetic Mathematics Dataset (`math/`)

Generated as part of **SMTX-Baby (T-101) foundational-reasoning pre-training**.
All content is fully synthetic and original; generated deterministically (`random.seed(42)`) for reproducibility.

| Sub-directory | File | Examples | Description |
|---|---|---|---|
| `arithmetic/` | `arithmetic_examples.txt` | 1,000 | Addition, subtraction, multiplication, and division with numbered step-by-step workings. |
| `algebra/` | `algebra_examples.txt` | 600 | One-step and two-step linear equations, factored quadratics, and inequalities with full reasoning. |
| `word_problems/` | `word_problems.txt` | 400 | Narrative problems across six archetypes (purchases, sharing, age, distance, percentages, combining groups). |
| `sequences/` | `sequences.txt` | 300 | Arithmetic, geometric, Fibonacci-style, and square/cube-number sequences with missing-term identification. |
| `reasoning_steps/` | `reasoning_steps.jsonl` | 400 | Chain-of-thought triples `{"question","reasoning","answer"}` covering arithmetic, algebra, word problems, and sequences. |
| `explanations/` | `concept_explanations.txt` | 15 | Paragraph-length explanations of core concepts: addition through ratios. |

**Math sub-total: 2,715 examples**

---

## File Format Notes

- All files are UTF-8 encoded with no BOM
- `.txt` files are plain text consumed directly by `dataset_pipeline.py`
- `.jsonl` files contain one JSON object per line
- The pipeline's `iter_text_files()` function reads `.txt` files directly and extracts the `text` field from `.jsonl` files. To ensure JSONL samples flow through the pipeline, each JSONL record should be pre-processed to include a `"text"` field (e.g. by concatenating `question + " " + answer`) or the pipeline can be extended to handle domain-specific keys.

---

## Alignment with `training/config_t101.json`

The pipeline ingests data from `DATA_INPUT_DIR` (default `data/raw`) as set in `scripts/run_training.sh` and configured in `training/config_t101.json`:

```json
"data": {
  "train_file": "data/train.jsonl",
  "validation_file": "data/val.jsonl"
}
```

Run `dataset_pipeline.py` to tokenise and split `data/raw/` into `data/processed/train.jsonl` and `data/processed/val.jsonl` before launching training:

```bash
python training/dataset_pipeline.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --tokenizer_path tmodels/t101
```
