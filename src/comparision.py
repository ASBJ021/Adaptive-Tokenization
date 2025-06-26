import time
import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPModel, CLIPProcessor
from torch.nn.functional import normalize
from tinyclip_bigclip_pipeline import (
    load_models, get_patch_tokens, get_text_embedding,
    select_top_k_indices, inject_selected_bigclip_tokens
)
from collections import defaultdict
from transformers import AutoModel, AutoProcessor



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Config ---
NUM_SAMPLES = 100 
TOP_K_PERCENT = 100
DATASET_NAME = "cifar100"
MODEL_ID = "openai/clip-vit-base-patch32"

# --- Dataset and Prompts ---
def load_data():
    dataset = load_dataset(DATASET_NAME, split="test").shuffle(seed=42).select(range(NUM_SAMPLES))
    classnames = dataset.features["fine_label"].names
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
    return dataset, prompts

# --- Preprocessing ---
def get_transform():
    return Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758))
    ])

# --- Benchmark: Standard CLIP ---
def benchmark_clip(dataset, prompts):
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = normalize(text_features, dim=-1)

    transform = get_transform()
    correct, total, total_time = 0, 0, 0.0

    for item in dataset:
        label = item["fine_label"]
        image_tensor = transform(item["img"]).unsqueeze(0).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_tensor)
            image_features = normalize(image_features, dim=-1)
            logits = image_features @ text_features.T
            pred = logits.argmax(dim=-1).item()
        end = time.time()

        correct += int(pred == label)
        total += 1
        total_time += (end - start)

    return {
        "model": "Standard CLIP",
        "accuracy": correct / total,
        "avg_inference_time": total_time / total
    }

# --- Benchmark: TinyCLIP → BigCLIP Pipeline ---
# def benchmark_tiny_bigclip(dataset, prompts):
#     tinyclip, tiny_proc, bigclip, big_proc = load_models()

#     text_inputs = big_proc(text=prompts, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         text_features = bigclip.get_text_features(**{k: v.to(DEVICE) for k, v in text_inputs.items()})
#         text_features = normalize(text_features, dim=-1)

#     correct, total, total_time = 0, 0, 0.0

#     for item in dataset:
#         label = item["fine_label"]
#         print(f'{label = }')
#         prompt = prompts[label]
#         print(f'{prompt = }')
#         image = item["img"]

#         start = time.time()
#         with torch.no_grad():
#             tiny_patches = get_patch_tokens(tinyclip, tiny_proc, image)
#             text_emb = get_text_embedding(tinyclip, tiny_proc, prompt)
#             selected_indices, _ = select_top_k_indices(tiny_patches, text_emb, top_k_percent=TOP_K_PERCENT)

#             big_patches = get_patch_tokens(bigclip, big_proc, image)
#             selected_big_patches = big_patches[selected_indices]

#             image_emb = inject_selected_bigclip_tokens(bigclip, selected_big_patches)
#             logits = image_emb @ text_features.T
#             pred = logits.argmax(dim=-1).item()
#         end = time.time()

#         # print(f'{pred = }')

#         correct += int(pred == label)
#         total += 1
#         total_time += (end - start)

#     return {
#         "model": f"Tiny→BigCLIP Top-{TOP_K_PERCENT}%",
#         "accuracy": correct / total,
#         "avg_inference_time": total_time / total
#     }

def benchmark_tiny_bigclip(dataset, prompts):

    tinyclip, tiny_proc, bigclip, big_proc = load_models()

    text_inputs = big_proc(text=prompts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = bigclip.get_text_features(**{k: v.to(DEVICE) for k, v in text_inputs.items()})
        text_features = normalize(text_features, dim=-1)

    correct, total, total_time = 0, 0, 0.0
    log_lines = []
    step_timings = defaultdict(list)

    for idx, item in enumerate(dataset):
        label = item["fine_label"]
        prompt = prompts[label]
        image = item["img"]

        times = {}
        start_total = time.time()

        with torch.no_grad():
            t0 = time.time()
            tiny_patches = get_patch_tokens(tinyclip, tiny_proc, image)
            times["tiny_patch_extraction"] = time.time() - t0

            t0 = time.time()
            text_emb = get_text_embedding(tinyclip, tiny_proc, prompt)
            times["text_embedding"] = time.time() - t0

            t0 = time.time()
            selected_indices, _ = select_top_k_indices(tiny_patches, text_emb, top_k_percent=TOP_K_PERCENT)
            times["patch_selection"] = time.time() - t0

            t0 = time.time()
            big_patches = get_patch_tokens(bigclip, big_proc, image)
            selected_big_patches = big_patches[selected_indices]
            times["big_patch_extraction"] = time.time() - t0

            t0 = time.time()
            image_emb = inject_selected_bigclip_tokens(bigclip, selected_big_patches)
            times["bigclip_injection"] = time.time() - t0

            t0 = time.time()
            logits = image_emb @ text_features.T
            pred = logits.argmax(dim=-1).item()
            times["prediction"] = time.time() - t0

        end_total = time.time()
        elapsed_total = end_total - start_total
        total += 1
        correct += int(pred == label)
        total_time += elapsed_total

        # Log per-step times for averaging
        for step, duration in times.items():
            step_timings[step].append(duration)
        step_timings["total"].append(elapsed_total)

        # --- Logging per image ---
        log_lines.append(f"\n🧪 Image {idx+1}/{len(dataset)} - Label: {label}, Prompt: \"{prompt}\"")
        for step, duration in times.items():
            log_lines.append(f"  {step:25s}: {duration:.4f} sec")
        log_lines.append(f"  TOTAL time per image       : {elapsed_total:.4f} sec")

    # --- Compute Averages ---
    log_lines.append("\n📊 Average Timing Per Step:")
    for step, durations in step_timings.items():
        avg = sum(durations) / len(durations)
        log_lines.append(f"  {step:25s}: {avg:.4f} sec")

    # --- Save to TXT file ---
    txt_path = "result/tiny_bigclip_timings.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n📝 Timing breakdown + averages saved to: {txt_path}")

    return {
        "model": f"Tiny→BigCLIP Top-{TOP_K_PERCENT}%",
        "accuracy": correct / total,
        "avg_inference_time": total_time / total
    }



def benchmark_tinyclip(dataset, prompts):
    # Load TinyCLIP
    model_id = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
    # model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
    # processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    # Encode text prompts
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = normalize(text_features, dim=-1)

    transform = get_transform()
    correct, total, total_time = 0, 0, 0.0

    for item in dataset:
        label = item["fine_label"]
        image_tensor = transform(item["img"]).unsqueeze(0).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_tensor)
            image_features = normalize(image_features, dim=-1)
            logits = image_features @ text_features.T
            pred = logits.argmax(dim=-1).item()
        end = time.time()

        correct += int(pred == label)
        total += 1
        total_time += (end - start)

    return {
        "model": "Standard CLIP",
        "accuracy": correct / total,
        "avg_inference_time": total_time / total
    }



# --- Run Benchmarks ---
def run_all_benchmarks():
    dataset, prompts = load_data()
    results = []

    print("\n▶ Running benchmark: Standard CLIP")
    results.append(benchmark_clip(dataset, prompts))

    # print("\n▶ Running benchmark: TinyCLIP → BigCLIP")
    # results.append(benchmark_tiny_bigclip(dataset, prompts))

    print("\n▶ Running benchmark: Tiny CLIP")
    results.append(benchmark_tinyclip(dataset, prompts))

    print(f"\n Final Results on {DATASET_NAME} for {NUM_SAMPLES} images:")
    for res in results:
        print(f"{res['model']:30} | Accuracy: {res['accuracy']:.4f} | Inference Time/Image: {res['avg_inference_time']:.4f} sec")

if __name__ == "__main__":
    run_all_benchmarks()
