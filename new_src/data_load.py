

import pickle
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data_normal, plot_dataset



if __name__ == "__main__":

    num_samples = 10
    dataset_name = "clane9/imagenet-100"
    split = "validation"
    
    ds, prompts = load_data_normal(dataset_name, num_samples, split)
    plot_dataset(ds, prompts)

    