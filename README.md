# Beyond the Needle: Stress-Testing Long-Context Intelligence

## Overview
This project evaluates how well large language models (LLMs) understand and reason over long contexts using the LongBench v2 benchmark.

We test performance under:
- Different context lengths
- Different answer positions
- Different prompting strategies

The goal is to understand *why* models fail.

Errors are categorized into:
- Retrieval Failure (RF)
- Reasoning Failure (RsF)
- Instruction Non-Compliance (INC)

---

## Environment Setup

### 1. Clone the repository
git clone <your-repo-link>
cd longbench-diagnostic

### 2. Create a virtual environment
python -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download NLTK data (required once)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

---

## System / Device Used

- OS: macOS
- Python: 3.13
- Hardware: CPU for preprocessing/testing, GPU for inference (if available)
- Models:
  - Llama-3.1-8B-Instruct (local)
  - GPT-4o-mini (API)

---

## Project Structure

longbench-diagnostic/
├── data/
├── src/
│   ├── logging_utils.py
│   ├── span_detect.py
│   ├── dataset_loader.py
│   ├── truncate.py
│   ├── models/
│   ├── prompts/
│   └── iaa.py
├── tests/
├── results/
├── annotation/
├── requirements.txt
└── README.md

---

## How to Run the Code

Make sure dataset is placed in:
data/longbench_v2/

### 1. Load dataset
python src/dataset_loader.py

### 2. Generate truncated datasets
python src/truncate.py

### 3. Run inference (pilot experiments)

8K context → results/pilot_8k.jsonl  
32K context → results/pilot_32k.jsonl  

---

## How Results Are Generated

Each experiment follows these steps:

1. Load a LongBench instance  
2. Truncate context to a token budget  
3. Build a prompt (Direct Answer or Extract-Then-Answer)  
4. Run model inference  
5. Log results using InferenceLogger  

Example output:

{
  "instance_id": "...",
  "category": "...",
  "budget": 8192,
  "prompt_strategy": "DA",
  "model_id": "...",
  "prediction": "B",
  "ground_truth": "A",
  "correct": false,
  "input_tokens": 7841,
  "output_tokens": 3,
  "latency_ms": 4210.3
}

---

## Error Analysis

Incorrect predictions are manually annotated into:

- Retrieval Failure (RF)
- Reasoning Failure (RsF)
- Instruction Non-Compliance (INC)

Annotations are stored in:
annotation/iaa_pilot.csv

Inter-annotator agreement is computed using Cohen’s kappa.

---

## Key Outputs

- Accuracy vs context length
- Error type distribution
- Latency vs accuracy trade-offs

---

## Testing

pytest
