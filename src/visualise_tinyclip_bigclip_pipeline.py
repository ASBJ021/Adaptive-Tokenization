import torch
from torch.nn.functional import normalize, cosine_similarity
from transformers import AutoModel, AutoProcessor, CLIPModel
from PIL import Image, ImageDraw
from datasets import load_dataset
import time
import matplotlib.pyplot as plt
import numpy as np
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------- Load Models -------------------------
def load_models():
    tinyclip_model_id = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
    bigclip_model_id = "openai/clip-vit-base-patch16"

    tinyclip = AutoModel.from_pretrained(tinyclip_model_id).eval().to(DEVICE)
    tinyclip_processor = AutoProcessor.from_pretrained(tinyclip_model_id)

    bigclip = CLIPModel.from_pretrained(bigclip_model_id).eval().to(DEVICE)
    bigclip_processor = AutoProcessor.from_pretrained(bigclip_model_id)

    return tinyclip, tinyclip_processor, bigclip, bigclip_processor

# ------------------------- Feature Extraction -------------------------
def get_patch_tokens(model, processor, image):
    inputs = processor(images=image, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.vision_model(**inputs)
        patch_tokens = out.last_hidden_state[:, 1:, :].squeeze(0)  # [N, D]
    return patch_tokens

def get_text_embedding(model, processor, text):
    inputs = processor(text=text, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        output = model.text_model(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
    return output.last_hidden_state[:, 0, :]  # [1, D]

# ------------------------- Patch Selection -------------------------
def select_top_k_indices(patch_tokens, text_embedding, top_k_percent=25):
    patch_tokens = normalize(patch_tokens, dim=-1)
    text_embedding = normalize(text_embedding, dim=-1)
    sims = cosine_similarity(patch_tokens, text_embedding[0], dim=-1)  # [N]
    top_k = max(1, int(patch_tokens.shape[0] * top_k_percent / 100))
    top_indices = torch.topk(sims, top_k).indices
    return top_indices, sims[top_indices]

# ------------------------- Injection into Big CLIP -------------------------
def inject_into_big_clip(selected_patches, big_clip_model):
    vision_model = big_clip_model.vision_model
    proj = big_clip_model.visual_projection

    D_big = vision_model.config.hidden_size
    D_input = selected_patches.shape[1]

    if D_input != D_big:
        mapper = torch.nn.Linear(D_input, D_big).to(DEVICE)
        selected_patches = mapper(selected_patches)

    cls_token = vision_model.embeddings.class_embedding.clone().detach().unsqueeze(0).unsqueeze(0)
    seq_len = selected_patches.shape[0] + 1
    pos_embed = vision_model.embeddings.position_embedding.weight[:seq_len].unsqueeze(0)

    x = torch.cat([cls_token, selected_patches.unsqueeze(0)], dim=1) + pos_embed
    x = vision_model.pre_layrnorm(x)
    x = vision_model.encoder(x)
    x = vision_model.post_layernorm(x[0])
    image_emb = proj(x[:, 0, :])

    return normalize(image_emb, dim=-1)

# ------------------------- Visualization -------------------------
# def visualize_selected_patches(image, selected_indices, grid_size):
#     img = image.resize((grid_size * 16, grid_size * 16))
#     draw = ImageDraw.Draw(img)
#     patch_size = 16
#     for idx in selected_indices:
#         row, col = divmod(idx.item(), grid_size)
#         x0, y0 = col * patch_size, row * patch_size
#         x1, y1 = x0 + patch_size, y0 + patch_size
#         draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
#     img.show()


def visualize_selected_patches(image, selected_indices, processor):
    # Resize image using CLIP's internal logic (exactly as CLIP saw it)
    processed = processor.image_processor(image, return_tensors="pt")
    image_tensor = processed["pixel_values"][0]  # shape: [3, H, W]
    h, w = image_tensor.shape[1:]  # get spatial size

    # Convert back to PIL for visualization
    np_img = image_tensor.permute(1, 2, 0).cpu().numpy() * 255
    np_img = np_img.astype("uint8")
    img_resized = Image.fromarray(np_img)

    # Grid size from patch count
    grid_size = int(math.sqrt(image_tensor.shape[1] * image_tensor.shape[2] / (16 * 16)))
    patch_size = w // grid_size

    draw = ImageDraw.Draw(img_resized)
    for idx in selected_indices:
        row, col = divmod(idx.item(), grid_size)
        x0, y0 = col * patch_size, row * patch_size
        x1, y1 = x0 + patch_size, y0 + patch_size
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    img_resized.show()

# ------------------------- Classification -------------------------
def classify_image(image_emb, prompts, clip_model):
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_embs = clip_model.get_text_features(**inputs)
    text_embs = normalize(text_embs, dim=-1)
    sims = cosine_similarity(image_emb, text_embs)
    top_idx = sims.argmax().item()
    return prompts[top_idx], sims

# ------------------------- Main Pipeline -------------------------
def main():
    dataset = load_dataset("timm/imagenet-1k-wds", split="validation")
    image = dataset[10]['jpg']
    prompt = "main object in the image"
    candidate_prompts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a car",
        "a photo of a bird",
        "a person standing",
        "a sleeping animal"
    ]

    tinyclip, tiny_proc, bigclip, big_proc = load_models()

    start = time.perf_counter()
    tiny_patch_tokens = get_patch_tokens(tinyclip, tiny_proc, image)
    text_emb = get_text_embedding(tinyclip, tiny_proc, prompt)
    selected_indices, sims = select_top_k_indices(tiny_patch_tokens, text_emb, top_k_percent=25)
    patch_time = time.perf_counter() - start

    big_patch_tokens = get_patch_tokens(bigclip, big_proc, image)
    selected_patches = big_patch_tokens[selected_indices]

    start = time.perf_counter()
    image_emb = inject_into_big_clip(selected_patches, bigclip)
    inject_time = time.perf_counter() - start

    start = time.perf_counter()
    full_inputs = big_proc(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        full_out = bigclip(**full_inputs)
    full_image_emb = normalize(full_out.image_embeds, dim=-1)
    full_text_emb = normalize(full_out.text_embeds, dim=-1)
    full_sim = cosine_similarity(full_image_emb, full_text_emb).item()
    full_time = time.perf_counter() - start

    # Classification
    pred_label, all_sims = classify_image(image_emb, candidate_prompts, bigclip)

    print("\n📊 Classification Results:")
    for label, score in zip(candidate_prompts, all_sims):
        print(f"{label:30s} → {score.item():.4f}")

    print(f"\n✅ Final predicted label: {pred_label}")
    print(f"\n🔍 Patch-based similarity score: {all_sims[candidate_prompts.index(pred_label)].item():.4f}")
    print(f"⏱️ TinyCLIP + patch selection time: {patch_time:.4f} sec")
    print(f"⏱️ Big CLIP patch injection time: {inject_time:.4f} sec")
    print(f"⏱️ Full CLIP inference time: {full_time:.4f} sec")
    print(f"⚡ Speed gain: {(full_time - (patch_time + inject_time)) / full_time * 100:.2f}%")

    # Visualize patches
    # visualize_selected_patches(image, selected_indices, grid_size=int(tiny_patch_tokens.shape[0] ** 0.5))
    visualize_selected_patches(image, selected_indices, tiny_proc)

if __name__ == "__main__":
    main()
