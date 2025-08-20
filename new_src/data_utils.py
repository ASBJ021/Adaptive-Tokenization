
from datasets import load_dataset

def load_data(DATASET_NAME, NUM_SAMPLES):
    ds = load_dataset(DATASET_NAME, split="test")
    ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))
    names = ds.features["fine_label"].names
    prompts = [f"a photo of a {n.replace('_',' ')}" for n in names]
    return ds, prompts