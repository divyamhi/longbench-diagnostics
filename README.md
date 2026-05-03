# Beyond the Needle: Stress-Testing Long-Context Intelligence

## Environment Setup

### 1. Clone the repository
```
git clone https://github.com/divyamhi/longbench-diagnostics.git
cd longbench-diagnostic
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

- OS: macOS  
- Python: 3.13  
- Hardware: CPU for preprocessing/testing, GPU for inference (if available)  
- Models used:
  - Llama-3.1-8B-Instruct (local)
  - GPT-4o-mini (API)

---

## How to Run the Code

Ensure dataset is placed in:
```
data/longbench_v2/
```

### 1. Load dataset
```
python src/dataset_loader.py
```

### 2. Generate truncated datasets
```
python src/truncate.py
```

### 3. Run inference (pilot experiments)

- 8K context → `results/pilot_8k.jsonl`  
- 32K context → `results/pilot_32k.jsonl`  

---

## How Results Are Generated

Each experiment follows these steps:

1. Load a LongBench instance  
2. Truncate context to a specified token budget  
3. Build a prompt (Direct Answer or Extract-Then-Answer)  
4. Run model inference  
5. Log results using the `InferenceLogger`  

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
