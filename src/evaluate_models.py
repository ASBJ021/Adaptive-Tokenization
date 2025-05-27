import time
import yaml
import argparse
import requests
# import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModel, AutoTokenizer

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def show_image_with_label(image_url, ground_truth):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Ground Truth: {ground_truth}", fontsize=14)
    plt.show()

def get_model_info(model_name):
    try:
        model = AutoModel.from_pretrained(model_name)
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(t.element_size() * t.nelement() for t in model.state_dict().values()) / (1024 ** 2)
        return num_params, model_size_mb
    except Exception as e:
        print(f"Warning: Could not fetch model info for {model_name}: {e}")
        return None, None

def evaluate_model(model_name, image_url, labels):
    clf = pipeline("zero-shot-image-classification", model=model_name, device=0)
    start_time = time.time()
    result = clf(image_url, candidate_labels=labels)
    end_time = time.time()

    top_label = result[0]["label"]
    inference_time = end_time - start_time

    return {
        "model": model_name,
        "top_label": top_label,
        "inference_time": inference_time,
        "scores": {item["label"]: item["score"] for item in result},
        "full_result": result
    }

def compare_results(results, ground_truth):
    print("\n=== Model Comparison ===")
    for alias, res in results:
        print(f"{alias}:")
        print(f"  Model Name     : {res['model']}")
        print(f"  Top Prediction : {res['top_label']}")
        print(f"  Inference Time : {res['inference_time']:.4f} seconds")
        print(f"  Parameters     : {res.get('num_params', 'N/A'):,}")
        print(f"  Model Size     : {res.get('model_size_mb', 'N/A'):.2f} MB" if res.get('model_size_mb') else "")
        print("  Label Scores   :")
        for label, score in res["scores"].items():
            print(f"    {label:20} {score:.4f}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Compare zero-shot image classification models.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = config["labels"]
    image_url = config["image_url"]
    ground_truth = config["ground_truth"]

    show_image_with_label(image_url, ground_truth)

    results = []
    for model_alias, model_name in config["models"].items():
        print(f"Evaluating {model_alias}...")
        result = evaluate_model(model_name, image_url, labels)

        num_params, model_size_mb = get_model_info(model_name)
        result["num_params"] = num_params
        result["model_size_mb"] = model_size_mb

        results.append((model_alias, result))

    compare_results(results, ground_truth)

if __name__ == "__main__":
    main()
