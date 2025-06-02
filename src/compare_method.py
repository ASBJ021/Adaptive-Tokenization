import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw
import requests
import math
import os
import clip
import time

# ------------------- Configuration -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 16

# Load TinyCLIP
model_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
model = AutoModel.from_pretrained(model_name).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained(model_name)

# ------------------- TinyCLIP Patch Embeddings -------------------
def get_patch_embeddings(image):
    inputs = processor(images=image, return_tensors="pt")
    vision_inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k.startswith("pixel_values")}
    with torch.no_grad():
        output = model.vision_model(**vision_inputs)

    patch_tokens = output.last_hidden_state[:, 1:, :]  # remove CLS
    patch_tokens = patch_tokens.squeeze(0)

    num_patches = patch_tokens.shape[0]
    grid_size = int(math.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, "Non-square grid not supported"

    resized_image_tensor = processor.image_processor(image, return_tensors="pt")["pixel_values"][0]
    resized_np = resized_image_tensor.permute(1, 2, 0).cpu().numpy()
    resized_np = (resized_np * 255).astype("uint8")
    resized_pil = Image.fromarray(resized_np)

    return patch_tokens, resized_pil, (grid_size, grid_size), image.size, resized_pil.size

# ------------------- Text Embedding -------------------
def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        output = model.text_model(**text_inputs)
    return output.last_hidden_state[:, 0, :]  # CLS

# ------------------- Select Top-K -------------------
def select_top_k_patches(patch_tokens, text_embedding, top_k_percentage):
    patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
    text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)
    sims = torch.nn.functional.cosine_similarity(patch_tokens, text_embedding[0], dim=-1)
    top_k = max(1, int(patch_tokens.shape[0] * top_k_percentage / 100))
    top_indices = torch.topk(sims, top_k).indices
    return top_indices.cpu(), sims.cpu()

# ------------------- Visualization -------------------
def visualize_top_patches(image, top_indices, grid_size):
    draw = ImageDraw.Draw(image)
    cols, rows = grid_size
    patch_w = image.width // cols
    patch_h = image.height // rows
    for idx in top_indices:
        r = idx // cols
        c = idx % cols
        x0 = c * patch_w
        y0 = r * patch_h
        x1 = x0 + patch_w
        y1 = y0 + patch_h
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    return image

def visualize_on_original(original_image, top_indices, grid_size, resized_size):
    draw = ImageDraw.Draw(original_image)
    cols, rows = grid_size
    res_w, res_h = resized_size
    orig_w, orig_h = original_image.size
    patch_w = res_w // cols
    patch_h = res_h // rows
    x_scale = orig_w / res_w
    y_scale = orig_h / res_h
    for idx in top_indices:
        r = idx // cols
        c = idx % cols
        x0 = c * patch_w * x_scale
        y0 = r * patch_h * y_scale
        x1 = x0 + patch_w * x_scale
        y1 = y0 + patch_h * y_scale
        draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
    return original_image

# ------------------- Big CLIP Comparison -------------------
def compare_big_clip_full_vs_selected(original_image, top_indices, grid_size, resized_size, prompt):
    big_model, big_preprocess = clip.load("ViT-L/14", device=DEVICE)
    big_model.eval()

    cols, rows = grid_size
    res_w, res_h = resized_size
    orig_w, orig_h = original_image.size
    patch_w = res_w // cols
    patch_h = res_h // rows
    x_scale = orig_w / res_w
    y_scale = orig_h / res_h

    cropped_patches = []
    for idx in top_indices:
        r = idx // cols
        c = idx % cols
        x0 = int(c * patch_w * x_scale)
        y0 = int(r * patch_h * y_scale)
        x1 = int(x0 + patch_w * x_scale)
        y1 = int(y0 + patch_h * y_scale)
        cropped = original_image.crop((x0, y0, x1, y1))
        cropped_patches.append(cropped)

    patch_size = cropped_patches[0].size
    grid_side = int(math.ceil(len(cropped_patches) ** 0.5))
    canvas = Image.new("RGB", (patch_size[0] * grid_side, patch_size[1] * grid_side), color=(0, 0, 0))
    for i, patch in enumerate(cropped_patches):
        r = i // grid_side
        c = i % grid_side
        canvas.paste(patch, (c * patch_size[0], r * patch_size[1]))

    input_canvas = big_preprocess(canvas).unsqueeze(0).to(DEVICE)
    full_input = big_preprocess(original_image).unsqueeze(0).to(DEVICE)
    tokenized = clip.tokenize([prompt]).to(DEVICE)

    with torch.no_grad():
        t0 = time.time()
        canvas_feat = big_model.encode_image(input_canvas)
        prompt_feat = big_model.encode_text(tokenized)
        canvas_feat /= canvas_feat.norm(dim=-1, keepdim=True)
        prompt_feat /= prompt_feat.norm(dim=-1, keepdim=True)
        sim_canvas = (canvas_feat @ prompt_feat.T).item()
        t1 = time.time()

        t2 = time.time()
        full_feat = big_model.encode_image(full_input)
        full_feat /= full_feat.norm(dim=-1, keepdim=True)
        sim_full = (full_feat @ prompt_feat.T).item()
        t3 = time.time()

    print(f"\n[Big CLIP] Prompt: \"{prompt}\"")
    print(f"  Selected Patches → Similarity: {sim_canvas:.2f} | Time: {t1 - t0:.3f} s")
    print(f"  Full Image       → Similarity: {sim_full:.2f} | Time: {t3 - t2:.3f} s")

    return {
        "similarity_selected": sim_canvas,
        "similarity_full": sim_full,
        "time_selected": t1 - t0,
        "time_full": t3 - t2,
        "patch_image": canvas,
    }

# ------------------- Pipeline -------------------
def run_pipeline(image_url, text_prompt, top_k_percentage=25, visualize=True, save_path=None):
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    patch_tokens, resized_image, grid_size, original_size, resized_size = get_patch_embeddings(image)
    text_embedding = get_text_embedding(text_prompt)
    top_indices, similarities = select_top_k_patches(patch_tokens, text_embedding, top_k_percentage)

    print(f"Top patch similarity scores:\n{similarities[top_indices]}")

    vis_resized = visualize_top_patches(resized_image.copy(), top_indices, grid_size)
    vis_original = visualize_on_original(image.copy(), top_indices, grid_size, resized_size)

    result = compare_big_clip_full_vs_selected(image, top_indices, grid_size, resized_size, text_prompt)

    if save_path:
        base, ext = os.path.splitext(save_path)
        vis_resized.save(f"{base}_resized{ext}")
        vis_original.save(f"{base}_original{ext}")
        result["patch_image"].save(f"{base}_patches{ext}")
        print(f"Saved: {base}_resized{ext}")
        print(f"Saved: {base}_original{ext}")
        print(f"Saved: {base}_patches{ext}")

    if visualize:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(vis_resized)
        axs[0].set_title("Model Input (Resized)")
        axs[1].imshow(vis_original)
        axs[1].set_title("Original Image")
        axs[2].imshow(result["patch_image"])
        axs[2].set_title("Selected Patches Only")
        for ax in axs: ax.axis("off")
        plt.tight_layout()
        plt.show()

    return result

# ------------------- Entry -------------------
def main():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt = "a sleeping cat"
    save_path = "t1.png"
    run_pipeline(url, prompt, top_k_percentage=25, save_path=save_path)

if __name__ == "__main__":
    main()
