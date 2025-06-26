import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPModel, AutoProcessor, AutoModel
from tinyclip_bigclip_pipeline import (
    load_models, get_patch_tokens, get_text_embedding,
    select_top_k_indices, inject_into_big_clip
)
from torch.nn.functional import cosine_similarity, normalize
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CIFAR-100 setup ===
NUM_SAMPLES = 10
dataset = load_dataset("cifar100", split="test").shuffle(seed=42).select(range(NUM_SAMPLES))
classnames = dataset.features["fine_label"].names
prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
# print(f'{prompts = }')

# === Load models ===
tinyclip, tiny_proc, bigclip, big_proc = load_models()

# === Encode class prompts with BigCLIP ===
text_inputs = big_proc(text=prompts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    text_features = bigclip.get_text_features(**{k: v.to(DEVICE) for k, v in text_inputs.items()})
    text_features = normalize(text_features, dim=-1)

# === Preprocessing transform for CIFAR images ===
transform = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758))
])

# === Evaluation loop ===
correct = 0
total = 0
TOP_K_PERCENT = 100  # adjust as needed

for item in dataset:
    image: Image.Image = transform(item["img"]).unsqueeze(0).to(DEVICE)
    label = item["fine_label"]
    true_prompt = prompts[label]
    print(f'{true_prompt = }')

    with torch.no_grad():
        # TinyCLIP patch tokens and text embedding
        patch_tokens = get_patch_tokens(tinyclip, tiny_proc, item["img"])       # [N, D_small]
        text_emb = get_text_embedding(tinyclip, tiny_proc, true_prompt)         # [1, D_small]
        selected_indices, _ = select_top_k_indices(patch_tokens, text_emb, top_k_percent=TOP_K_PERCENT)

        # Use BigCLIP's patches and inject selected ones
        big_patch_tokens = get_patch_tokens(bigclip, big_proc, item["img"])     # [N, D_big]
        selected_patches = big_patch_tokens[selected_indices]                   # [K, D_big]
        image_embedding = inject_into_big_clip(selected_patches, bigclip)       # [1, D_big]
        image_embedding = normalize(image_embedding, dim=-1)

        # Compute similarities to all prompts
        logits = image_embedding @ text_features.T                              # [1, 100]
        pred = logits.argmax(dim=-1).item()

        correct += int(pred == label)
        total += 1

print(f"\n✅ TinyCLIP→BigCLIP Top-1 Accuracy on CIFAR-100 ({total} samples): {correct / total:.4f}")
