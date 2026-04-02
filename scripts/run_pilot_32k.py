import sys, os, json, random
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset_loader import load, get_by_category
from src.prompts.e2a_pipeline import run_e2a
from src.models.local_hf import LocalHFBackend
from src.logging_utils import InferenceLogger

SPLIT_PATH          = "data/splits/32k.jsonl"
OUTPUT_PATH         = "results/pilot_32k.jsonl"
MODEL_ID            = "meta-llama/Llama-3.1-8B-Instruct"
BUDGET              = 32768
PROMPT_STRAT        = "E2A"
SAMPLE_PER_CATEGORY = 50 // 6   # 8 per category (floor)
EXTRA               = 50 % 6    # 2 extra to reach exactly 50
SEED                = 42

# 1. Load the 32K split
instances = []
with open(SPLIT_PATH, encoding="utf-8") as f:
    for line in f:
        instances.append(json.loads(line.strip()))
print(f"Loaded {len(instances)} instances from {SPLIT_PATH}")

# 2. Stratified sampling
random.seed(SEED)
grouped = get_by_category(instances)
sampled = []
categories_sorted = sorted(grouped.keys())
for i, cat in enumerate(categories_sorted):
    n = SAMPLE_PER_CATEGORY + (1 if i < EXTRA else 0)
    cat_instances = grouped[cat]
    if len(cat_instances) < n:
        print(f"WARNING: {cat} has only {len(cat_instances)} instances, sampling all")
        n = len(cat_instances)
    sampled.extend(random.sample(cat_instances, n))

print(f"Sampled {len(sampled)} instances across {len(categories_sorted)} categories")

# 3. Load model
print(f"Loading model: {MODEL_ID}")
backend = LocalHFBackend(model_id=MODEL_ID, device="cuda")
print("Model loaded.")

# 4. Run inference and log
Path("results").mkdir(exist_ok=True)

correct = 0
with InferenceLogger(OUTPUT_PATH) as logger:
    for idx, instance in enumerate(sampled):
        result = run_e2a(instance, backend)

        log_record = {
            **result,
            "category":        instance["category"],
            "budget":          BUDGET,
            "prompt_strategy": PROMPT_STRAT,
            "model_id":        MODEL_ID,
        }
        logger.log(log_record)

        if result["correct"]:
            correct += 1

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/50] accuracy so far: {correct/(idx+1):.1%}")

print(f"\nDone. Accuracy: {correct}/50 = {correct/50:.1%}")
print(f"Results written to {OUTPUT_PATH}")

# 5. Final verification
lines = Path(OUTPUT_PATH).read_text(encoding="utf-8").strip().split("\n")
print(f"\nVerification: {len(lines)} records in {OUTPUT_PATH}")

from collections import defaultdict
cat_correct = defaultdict(int)
cat_total   = defaultdict(int)
for line in lines:
    rec = json.loads(line)
    cat_total[rec["category"]] += 1
    if rec["correct"]:
        cat_correct[rec["category"]] += 1

print("\nPer-category accuracy:")
for cat in sorted(cat_total.keys()):
    n = cat_total[cat]
    c = cat_correct[cat]
    print(f"  {cat}: {c}/{n} = {c/n:.1%}")
