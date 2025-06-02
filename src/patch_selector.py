import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw
import requests
import math
import numpy as np
from matplotlib import cm


# Load model and processor
model_name = "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M"
model = AutoModel.from_pretrained(model_name).eval().cuda()
processor = AutoProcessor.from_pretrained(model_name)

PATCH_SIZE = 16

# ---- Get Patch Embeddings ----
def get_patch_embeddings(image):
    inputs = processor(images=image, return_tensors="pt")
    vision_inputs = {k: v.cuda() for k, v in inputs.items() if k.startswith("pixel_values")}
    with torch.no_grad():
        output = model.vision_model(**vision_inputs)
    patch_tokens = output.last_hidden_state[:, 1:, :]  # exclude [CLS]
    return patch_tokens.squeeze(0)  # [num_patches, hidden_dim]

# ---- Get Text Embedding ----
def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    text_inputs = {k: v.cuda() for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        output = model.text_model(**text_inputs)
    return output.last_hidden_state[:, 0, :]  # CLS token

# ---- Select Top-K Patches ----
def select_top_k_patches(patch_tokens, text_embedding, top_k_percentage):

    patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
    text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)
    sims = torch.nn.functional.cosine_similarity(patch_tokens, text_embedding[0], dim=-1)
    
    # Compute top_k count from percentage
    total_patches = patch_tokens.shape[0]
    top_k = max(1, int(total_patches * top_k_percentage/100))

    # Select top-k indices
    top_indices = torch.topk(sims, top_k).indices

    return top_indices.cpu(), sims.cpu()

def overlay_heatmap_on_image(image, similarity_scores, grid_size, alpha=0.5, cmap='jet'):
    """
    similarity_scores: 1D tensor of length num_patches (e.g., 196 for 14x14)
    grid_size: (cols, rows) — typically (14, 14) for ViT-Base
    """

    # Normalize similarity scores to [0, 1]
    sim = similarity_scores.clone()
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    
    # Reshape to 2D grid
    heatmap = sim.view(grid_size[1], grid_size[0]).cpu().numpy()  # [rows, cols]
    heatmap = np.uint8(cm.get_cmap(cmap)(heatmap) * 255)  # RGBA

    # Convert to PIL image
    heatmap_img = Image.fromarray(heatmap).convert("RGBA")
    heatmap_img = heatmap_img.resize(image.size, resample=Image.BILINEAR)

    # Original image as RGBA
    overlay = Image.blend(image.convert("RGBA"), heatmap_img, alpha=alpha)
    return overlay
# ---- Visualize Top Patches ----
def visualize_top_patches(image, top_indices, grid_size):
    draw = ImageDraw.Draw(image)
    cols, rows = grid_size
    for idx in top_indices:
        r = idx // cols
        c = idx % cols
        x0 = c * PATCH_SIZE
        y0 = r * PATCH_SIZE
        x1 = x0 + PATCH_SIZE
        y1 = y0 + PATCH_SIZE
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    return image


# ---- Run Full Flow for One Image ----
def run_pipeline(image_url, text_prompt, top_k_percentage=10, visualize=True, resize_to_fit=True, save_path=None):
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    w, h = image.size

    if resize_to_fit:
        new_w = (w // PATCH_SIZE) * PATCH_SIZE
        new_h = (h // PATCH_SIZE) * PATCH_SIZE
        image = image.resize((new_w, new_h))
        w, h = new_w, new_h

    # cols, rows = w // PATCH_SIZE, h // PATCH_SIZE

    

    patch_tokens = get_patch_embeddings(image)

    num_patches = patch_tokens.shape[0]
    grid_size = int(math.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, "Non-square patch grid not supported"
    cols, rows = grid_size, grid_size

    text_embedding = get_text_embedding(text_prompt)
    top_indices, similarities = select_top_k_patches(patch_tokens, text_embedding, top_k_percentage)

    print(f"Top patch similarity scores:\n{similarities[top_indices]}")

    heatmap_img = overlay_heatmap_on_image(image.copy(), similarities, (cols, rows), alpha=0.5)
    

    
    if visualize or save_path:
        vis_image = visualize_top_patches(image.copy(), top_indices, (cols, rows))

        if save_path:
            vis_image.save(save_path)
            print(f"Saved visualization to: {save_path}")
            heatmap_img.save(save_path.replace(".png", "_heatmap.png"))
            print(f"Heatmap saved to: {save_path.replace('.png', '_heatmap.png')}")

        if visualize:
            plt.imshow(vis_image)
            plt.title(f"Top-{top_k_percentage} % Patches for '{text_prompt}'")
            plt.axis("off")
            plt.show()

    return top_indices, similarities[top_indices]



url = "http://images.cocodataset.org/val2017/000000039769.jpg"
prompt = "a sleeping cat"
save_path = "top_patches_p4.png"

run_pipeline(url, prompt, top_k_percentage=25, save_path=save_path)