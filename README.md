Beyond the Needle: Stress-Testing Long-Context Intelligence
Overview

This project evaluates how well large language models (LLMs) understand and reason over long contexts. Using the LongBench v2 benchmark, we systematically test model performance under different conditions such as context length, answer position, and prompting strategy.

The goal is not just to measure accuracy, but to understand why models fail. Errors are categorized into:

Retrieval Failure (RF)
Reasoning Failure (RsF)
Instruction Non-Compliance (INC)
Environment Setup
1. Clone the repository
git clone <your-repo-link>
cd longbench-diagnostic
2. Create a virtual environment
python -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Download NLTK data (required once)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
System / Device Used
OS: macOS
Python: 3.13
Hardware: CPU for preprocessing/testing, GPU used for model inference (if available)
Models tested:
Llama-3.1-8B-Instruct (local)
GPT-4o-mini (API)
Project Structure
longbench-diagnostic/
├── data/                  Dataset files
├── src/                   Core code
│   ├── logging_utils.py   Logging
│   ├── span_detect.py     Span detection
│   ├── dataset_loader.py  Data loading
│   ├── truncate.py        Context truncation
│   ├── models/            Model backends
│   ├── prompts/           Prompt templates
│   └── iaa.py             Annotation metrics
├── tests/                 Unit tests
├── results/               JSONL output files
├── annotation/            Annotation CSV and schema
├── requirements.txt
└── README.md
How to Run the Code

Make sure the dataset is placed in:

data/longbench_v2/
1. Load dataset
python src/dataset_loader.py
2. Generate truncated datasets
python src/truncate.py
3. Run inference (pilot experiments)

Run pilot experiments on:

8K context → outputs to results/pilot_8k.jsonl
32K context → outputs to results/pilot_32k.jsonl

(Exact commands depend on the inference script used in the repository.)

How Results Are Generated

Each experiment follows these steps:

Load a LongBench instance
Truncate context to a specified token budget
Build a prompt (Direct Answer or Extract-Then-Answer)
Run model inference
Log results using the InferenceLogger

Each result is stored in JSONL format:

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
Error Analysis

Incorrect predictions are manually annotated into:

Retrieval Failure (RF)
Reasoning Failure (RsF)
Instruction Non-Compliance (INC)

Annotations are stored in:

annotation/iaa_pilot.csv

Inter-annotator agreement is computed using Cohen’s kappa.

Key Outputs
Accuracy vs context length
Error type distribution
Latency vs accuracy trade-offs
Testing

Run unit tests:

pytest
