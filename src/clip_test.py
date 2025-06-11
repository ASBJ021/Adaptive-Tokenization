import torch
from torch.nn.functional import normalize
from transformers import AutoProcessor, AutoModel, CLIPModel, AutoModelForZeroShotImageClassification
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import math
import os
import time



# ------------------- Configuration -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 16

# Load TinyCLIP
model_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
model = AutoModel.from_pretrained(model_name).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained(model_name)

# --------------------- Loaders ---------------------
def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def preprocess_image(processor, image: Image.Image) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

# --------------------- TinyCLIP Encoding ---------------------
def encode_text_tinyclip(model, processor, text: str) -> torch.Tensor:
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    return normalize(text_emb, dim=-1)  # [1, D]

# ------------------- Text Embedding -------------------
def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        output = model.text_model(**text_inputs)
    return output.last_hidden_state[:, 0, :]  # CLS

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

def encode_image_tinyclip(model, image_tensor: torch.Tensor) -> torch.Tensor:
    # with torch.no_grad():
    #     image_emb = model.get_image_features(image_tensor)  # [1, N, D]

    # patch_tokens = normalize(image_emb, dim=-1)  # remove CLS
    # patch_tokens = patch_tokens.squeeze(0)
    # return patch_tokens # normalized [1, N, D]

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

def encode_prompts(prompts, clip_model):
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embs = clip_model.get_text_features(**{k: v.to(DEVICE) for k, v in inputs.items()})
    return normalize(text_embs, dim=-1)
# --------------------- Patch Selection ---------------------
# def select_top_patches(image_tokens: torch.Tensor, text_embedding: torch.Tensor, top_k_ratio=0.25):
#     """
#     image_tokens: [1, N, D]
#     text_embedding: [1, D]
#     Returns: [K, D]
#     """
#     sim = torch.matmul(image_tokens, text_embedding.unsqueeze(-1)).squeeze(-1)  # [1, N]
#     top_k = int(sim.shape[1] * top_k_ratio)
#     top_indices = torch.topk(sim, top_k, dim=1).indices.squeeze(0)  # [K]
#     # selected = image_tokens.squeeze(0)[top_indices]  # [K, D]


#     return top_indices, sim

# ------------------- Select Top-K -------------------
def select_top_k_patches(patch_tokens, text_embedding, top_k_percentage):
    patch_tokens = normalize(patch_tokens, dim=-1)
    text_embedding = normalize(text_embedding, dim=-1)
    sims = torch.nn.functional.cosine_similarity(patch_tokens, text_embedding[0], dim=-1)
    top_k = max(1, int(patch_tokens.shape[0] * top_k_percentage / 100))
    top_indices = torch.topk(sims, top_k).indices
    return patch_tokens[top_indices], top_indices.cpu(), sims.cpu()

# --------------------- Patch Injection to Big CLIP ---------------------
def inject_into_big_clip(selected_patches, clip_model_id="openai/clip-vit-base-patch16"):
    big_clip = CLIPModel.from_pretrained(clip_model_id).eval().to(DEVICE)
    vision_model = big_clip.vision_model
    proj = big_clip.visual_projection

    D_big = vision_model.config.hidden_size
    D_small = selected_patches.shape[1]
    # print(f'{D_big=}')
    # print(f'{D_small=}')
    selected_patches = selected_patches.to(DEVICE)
    # print(f'{selected_patches=}')

    if D_small != D_big:
        mapper = torch.nn.Linear(D_small, D_big).to(DEVICE)
        selected_patches = mapper(selected_patches)

    
    K = selected_patches.shape[0]
    # print(f'{K=}')

    # Create fake [CLS] token
    cls_token = torch.zeros(1, 1, D_big, device=selected_patches.device)
    # print(f'{cls_token.shape=}')

    # Combine CLS + patches
    x = torch.cat([cls_token, selected_patches.unsqueeze(0)], dim=1)  # [1, K+1, D]
    # print(f'{x.shape=}')

    # Use proper positional embedding indices
    position_ids = torch.arange(0, K + 1, device=selected_patches.device).unsqueeze(0)  # [1, K+1]
    pos_embed = vision_model.embeddings.position_embedding(position_ids)  # [1, K+1, D]
    x = x + pos_embed

    # Norm and Transformer
    x = vision_model.pre_layrnorm(x)
    x = vision_model.encoder(x)
    x = vision_model.post_layernorm(x[0])
    image_emb = proj(x[:, 0, :])  # CLS output

    return normalize(image_emb, dim=-1), big_clip




def classify_image(final_embedding, candidate_prompts, clip_model):
    text_embs = encode_prompts(candidate_prompts, clip_model)
    sims = torch.nn.functional.cosine_similarity(final_embedding, text_embs)
    best_idx = sims.argmax().item()
    return candidate_prompts[best_idx], sims



def run_baseline_clip_inference(image, prompt):
    model_id = "openai/clip-vit-base-patch16"
    processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval().to(DEVICE)

    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(**inputs)
        elapsed = time.perf_counter() - start

    img_emb = normalize(outputs.image_embeds, dim=-1)
    txt_emb = normalize(outputs.text_embeds, dim=-1)
    sim = torch.nn.functional.cosine_similarity(img_emb, txt_emb).item()
    return sim, elapsed



# --------------------- Main Pipeline ---------------------
# def main():
#     # image_url = "https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png"
#     # prompt = "A group of colorful birds sitting on a tree branch"

#     image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     prompt = "a sleeping cat"

#     # Load TinyCLIP
    
#     tinyclip_model_id = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
#     processor = AutoProcessor.from_pretrained(tinyclip_model_id)
#     tinyclip = AutoModel.from_pretrained(tinyclip_model_id)
#     # print(tinyclip)
#     print(f'loaded ... {tinyclip_model_id}')

#     # # Preprocess
#     # image = load_image_from_url(image_url)
#     # image_tensor = preprocess_image(processor, image)

#     # # Encode
#     # text_emb = encode_text_tinyclip(tinyclip, processor, prompt)  # [1, D]
#     # image_tokens = encode_image_tinyclip(tinyclip, image_tensor)  # [1, N, D]
#     # # print(f'Text Embd: {text_emb}')
#     # # print(f'Img Tokens: {image_tokens}')


#     # # Select top 25% patches
#     # top_indices, sim = select_top_k_patches(image_tokens, text_emb, top_k_percentage=0.25)
#     # # print(f'Selected tokens are : {selected}')
#     # # print(f"Top patch similarity scores:\n{sim[top_indices]}")
#     # print(f"Top Indices: {top_indices}")

#     image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

#     patch_tokens, resized_image, grid_size, original_size, resized_size = get_patch_embeddings(image)
#     text_embedding = get_text_embedding(prompt)
#     selected_patches, top_indices, similarities = select_top_k_patches(patch_tokens, text_embedding, top_k_percentage=25)

#     print(f"Top patch similarity scores:\n{len(similarities[top_indices])}")

    

#     # Inject into Big CLIP
#     final_img_embedding, big_clip = inject_into_big_clip(selected_patches)

#     print("Final image embedding shape:", final_img_embedding.shape)
#     print("Final embedding (first 5 dims):", final_img_embedding[0, :5])


#     candidate_labels = [
#     "a photo of a cat",
#     "a photo of a dog",
#     "a photo of a bird",
#     "a photo of a car",
#     "a photo of a tree",
#     "a sleeping cat",
#     ]

#     pred_label, all_scores = classify_image(final_img_embedding, candidate_labels, big_clip)

#     print("\n🧪 Prediction:")
#     for lbl, score in zip(candidate_labels, all_scores):
#         print(f"{lbl:30s} → {score.item():.4f}")

#     print(f"\n✅ Final predicted label: {pred_label}")

    

# def main():
#     image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     prompt = "a sleeping cat"

#     print(f"📥 Loading image...")
#     image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

#     # ---------------- TinyCLIP Patch-Based Path ----------------
#     print("\n🚀 Running TinyCLIP → patch selection → Big CLIP")
#     start_patch = time.perf_counter()

#     patch_tokens, _, grid_size, _, _ = get_patch_embeddings(image)
#     text_embedding = get_text_embedding(prompt)
#     selected_patches, top_indices, similarities = select_top_k_patches(patch_tokens, text_embedding, top_k_percentage=25)

#     final_img_embedding, big_clip = inject_into_big_clip(selected_patches)

#     start_bc = time.perf_counter()

#     text_emb_clip = encode_prompts([prompt], big_clip)
    
#     sim_patch = torch.nn.functional.cosine_similarity(final_img_embedding, text_emb_clip).item()

#     elapsed_patch = time.perf_counter() - start_patch
#     diff_time = time.perf_counter() - start_bc

#     print(f"\n🔍 Patch-based similarity score: {sim_patch:.4f}")
#     print(f"⏱️ Patch-based time: {elapsed_patch:.4f} sec")
#     print(f"⏱️ Patch-based time (Only Big CLip ): {diff_time:.4f} sec")

#     # ---------------- Standard CLIP Path ----------------
#     print("\n⚙️ Running standard full-image CLIP inference")
#     sim_full, elapsed_full = run_baseline_clip_inference(image, prompt)

#     print(f"\n🔍 Full CLIP similarity score: {sim_full:.4f}")
#     print(f"⏱️ Full CLIP time: {elapsed_full:.4f} sec")

#     # ---------------- Classification Comparison ----------------
#     candidate_labels = [
#         "a photo of a cat",
#         "a photo of a dog",
#         "a photo of a bird",
#         "a photo of a car",
#         "a photo of a tree",
#         "a sleeping cat",
#     ]

#     pred_label, all_scores = classify_image(final_img_embedding, candidate_labels, big_clip)

#     print("\n📊 Patch-Based Classification:")
#     for lbl, score in zip(candidate_labels, all_scores):
#         print(f"{lbl:30s} → {score.item():.4f}")

#     print(f"\n✅ Final predicted label (patch-based): {pred_label}")
#     print(f"\n⚡ Speed gain from patch-based inference: {(elapsed_full - elapsed_patch)/elapsed_full * 100:.2f}%")
   

def main():
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt = "a sleeping cat"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    print(f"📥 Image loaded: {image.size}")

    full_start = time.perf_counter()

    # --- PATCH EXTRACTION ---
    t0 = time.perf_counter()
    patch_tokens, _, grid_size, _, _ = get_patch_embeddings(image)
    t_patch = time.perf_counter() - t0
    print(f"⏱️ Patch extraction time: {t_patch:.4f} sec")

    # --- TEXT EMBEDDING (TinyCLIP) ---
    t0 = time.perf_counter()
    text_embedding = get_text_embedding(prompt)
    t_text = time.perf_counter() - t0
    print(f"⏱️ Text embedding (TinyCLIP) time: {t_text:.4f} sec")

    # --- SIMILARITY & PATCH SELECTION ---
    t0 = time.perf_counter()
    selected_patches, top_indices, similarities = select_top_k_patches(
        patch_tokens, text_embedding, top_k_percentage=25
    )
    t_select = time.perf_counter() - t0
    print(f"⏱️ Patch selection time: {t_select:.4f} sec")

    # --- PATCH INJECTION TO BIG CLIP ---
    t0 = time.perf_counter()
    final_img_embedding, big_clip = inject_into_big_clip(selected_patches)
    t_inject = time.perf_counter() - t0
    print(f"⏱️ Big CLIP injection time: {t_inject:.4f} sec")

    # --- FINAL PROMPT ENCODING & SIMILARITY ---
    t0 = time.perf_counter()
    text_emb_clip = encode_prompts([prompt], big_clip)
    sim_patch = torch.nn.functional.cosine_similarity(final_img_embedding, text_emb_clip).item()
    t_infer = time.perf_counter() - t0
    print(f"⏱️ Final inference time (text + similarity): {t_infer:.4f} sec")

    # --- TOTAL PATCH-BASED TIME ---
    total_patch_time = time.perf_counter() - full_start
    print(f"\n🧠 Final similarity score (patch-based): {sim_patch:.4f}")
    print(f"⏱️ Total patch-based pipeline time: {total_patch_time:.4f} sec")

    # --- FULL CLIP BASELINE ---
    sim_full, time_full = run_baseline_clip_inference(image, prompt)
    print(f"\n🧠 Full CLIP similarity: {sim_full:.4f}")
    print(f"⏱️ Full CLIP inference time: {time_full:.4f} sec")

    # --- CLASSIFICATION COMPARISON ---
    candidate_labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a car",
        "a photo of a tree",
        "a sleeping cat",
    ]
    pred_label, all_scores = classify_image(final_img_embedding, candidate_labels, big_clip)

    print("\n📊 Patch-Based Classification:")
    for lbl, score in zip(candidate_labels, all_scores):
        print(f"{lbl:30s} → {score.item():.4f}")

    print(f"\n✅ Predicted label: {pred_label}")
    print(f"⚡ Speed gain from patch-based inference: {(time_full - total_patch_time)/time_full * 100:.2f}%")


if __name__ == "__main__":
    main()
