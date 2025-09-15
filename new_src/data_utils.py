
from datasets import load_dataset

import pickle
import numpy as np

def load_data(DATASET_NAME, NUM_SAMPLES):
    ds = load_dataset(DATASET_NAME, split="test")
    ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))
    names = ds.features["fine_label"].names
    prompts = [f"a photo of a {n.replace('_',' ')}" for n in names]
    return ds, prompts


def load_data_folder(file_path, num_samples = 0):
    with open(file_path, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data
  
def load_data_normal(DATASET_NAME, NUM_SAMPLES, SPLIT="test"):
    ds = load_dataset(DATASET_NAME,split = SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))
    # ds = ds.select(range(NUM_SAMPLES))
    print(ds[0])
    names = ds.features["label"].names
    prompts = [f"a photo of a {n.replace('_',' ')}" for n in names]
    return ds, prompts

import matplotlib.pyplot as plt

def plot_dataset(ds, prompts):
    """
    Plot images one by one from a HuggingFace dataset.
    
    Args:
        ds: HuggingFace dataset (must contain "img" and "fine_label")
        prompts: optional list of prompt strings (e.g., class names)
        max_images: limit how many images to plot (default None = all)
    """
    n = len(ds) 
    
    for i in range(n):
        sample = ds[i]
        print(sample)
        img = sample["image"]    # already a PIL Image (HF datasets return PIL for image columns)
        label = sample["label"]
        print(label)
        print(f'{img.size = }')

        plt.imshow(img)
        
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()
