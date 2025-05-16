# here we will conduct the tests for the functions in the src directory
# import torch
from transformers import pipeline

clip = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
result = clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)
print(f'{result}')