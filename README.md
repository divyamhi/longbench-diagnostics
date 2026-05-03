# Beyond the Needle: Stress-Testing Long-Context Intelligence

## Environment Setup

### 1. Clone the repository
```
git clone https://github.com/divyamhi/longbench-diagnostics.git
cd longbench-diagnostics
```

### 2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Download NLTK data (required once)
```
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## System / Device Used

- OS: macOS / Windows  
- Python: 3.13  
- Hardware: CPU for preprocessing/testing; GPU used for model inference
- Models used:
  - Llama-3.1-8B-Instruct 

---

## How to Run the Code

### Dataset Setup

This project uses the LongBench v2 dataset.

Download the dataset from:
https://huggingface.co/datasets/THUDM/LongBench

Download the dataset using the HuggingFace datasets library or manually download the JSON files from the link above.

Place the raw JSON files in:

```
data/longbench_v2/
```

Create the folder if needed:

```
mkdir -p data/longbench_v2
```

---

### Convert dataset to project format

Run:

```
python convert_longbench.py
```

This will generate processed dataset files in:

```
data/processed_longbench/
```

---

### Run pipeline

### 1. Load dataset
```
python src/dataset_loader.py --data_dir data/processed_longbench
```


### 2. Generate truncated datasets
```
python src/truncate.py
```

### 3. Run inference (pilot experiments)

- 8K context → results/pilot_8k.jsonl  
- 32K context → results/pilot_32k.jsonl  

---

## How Results Are Generated

Each experiment follows these steps:

1. Load a LongBench instance  
2. Truncate context to a specified token budget  
3. Build a prompt (Direct Answer or Extract-Then-Answer)  
4. Run model inference  
5. Log results using the InferenceLogger  

Each result is stored in JSONL format:

```json
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
```
