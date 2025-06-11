
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import os
import time



# ------------------- Configuration -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 16

# Load TinyCLIP
model_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
model = AutoModel.from_pretrained(model_name).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained(model_name)

big_model_name = "openai/clip-vit-base-patch16"
big_model = AutoModel.from_pretrained(big_model_name ).eval().to(DEVICE)
big_processor = AutoProcessor.from_pretrained(big_model_name )

# ------------------- Configuration -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 16
TOPK_RATIO = 0.25  # Top 25% patch selection
os.makedirs("result", exist_ok=True)

# ------------------- Load Models -------------------
print("Loading models...")
# TinyCLIP
tinyclip_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
tinyclip = AutoModel.from_pretrained(tinyclip_name).eval().to(DEVICE)
tinyclip_processor = AutoProcessor.from_pretrained(tinyclip_name)

# Big CLIP
bigclip_name = "openai/clip-vit-base-patch16"
bigclip = AutoModel.from_pretrained(bigclip_name).eval().to(DEVICE)
bigclip_processor = AutoProcessor.from_pretrained(bigclip_name)

# ------------------- TinyCLIP Encoding -------------------

def encode_image_tinyclip(images):
    inputs = tinyclip_processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = tinyclip.vision_model(pixel_values=inputs["pixel_values"])
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # remove CLS, [B, N_patches, D]
    return patch_tokens

def encode_text_tinyclip(texts):
    inputs = tinyclip_processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_outputs = tinyclip.text_model(**inputs)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]  # CLS token
    return text_embeds



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

# ------------------- Patch Selection -------------------

def select_topk_patches(patch_embeds, text_embeds, topk_ratio=0.25):
    patch_embeds = F.normalize(patch_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1).unsqueeze(1)
    sims = torch.sum(patch_embeds * text_embeds, dim=-1)  # [B, N_patches]
    topk = max(1, int(patch_embeds.shape[1] * topk_ratio))
    top_indices = sims.topk(topk, dim=1).indices  # [B, topk]
    batch_indices = torch.arange(patch_embeds.size(0), device=DEVICE).unsqueeze(1)
    selected = patch_embeds[batch_indices, top_indices]  # [B, topk, D]
    return selected, top_indices

# ------------------- BigCLIP Encoding -------------------

def encode_image_bigclip(images):
    inputs = bigclip_processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = bigclip.get_image_features(inputs["pixel_values"])  # [B, D]
    return features

# ------------------- Main Pipeline -------------------

def run_pipeline(images, prompts, topk_ratio=TOPK_RATIO):
    patch_embeds = encode_image_tinyclip(images)
    text_embeds = encode_text_tinyclip(prompts)
    selected_patches, selected_indices = select_topk_patches(patch_embeds, text_embeds, topk_ratio)
    bigclip_features = encode_image_bigclip(images)
    return {
        "selected_patch_embeddings": selected_patches,
        "selected_indices": selected_indices,
        "bigclip_features": bigclip_features
    }

# ------------------- Save Output -------------------

def save_patch_embeddings(tensor, filename="selected_patch_embeddings.pt"):
    torch.save(tensor.cpu(), filename)
    print(f"Saved patch embeddings to {filename}")

# ------------------- Run Example -------------------

if __name__ == "__main__":
    # Define image and prompt
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt = "a sleeping cat"

    # Download image
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Run pipeline
    results = run_pipeline([image], [prompt])

    # Create save paths
    timestamp = time.time()
    image_path = f"result/test_{timestamp}.png"
    embed_path = image_path.replace(".png", ".pt")

    # Save image and patch embeddings
    image.save(image_path)
    save_patch_embeddings(results["selected_patch_embeddings"], embed_path)

    print(f"Saved original image to {image_path}")
