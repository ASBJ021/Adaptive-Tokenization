import time
import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPModel, CLIPProcessor
from torch.nn.functional import normalize
from collections import defaultdict
from transformers import AutoModel, AutoProcessor
import clip
from clip.model import VisionTransformer


# from tinyclip_bigclip_pipeline import (
#     get_patch_tokens, get_text_embedding,
#     select_top_k_indices, inject_selected_bigclip_tokens
# )

# ------------------------- Global Variables & Constants  -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'{DEVICE = }')

# --- Config ---
NUM_SAMPLES = 100 
TOP_K_PERCENT = 90
DATASET_NAME = "cifar100"
MODEL_ID =  'ViT-B/16' # "openai/clip-vit-base-patch16"


# --- Dataset and Prompts ---
def load_data():
    dataset = load_dataset(DATASET_NAME, split="test").shuffle(seed=42).select(range(NUM_SAMPLES))
    classnames = dataset.features["fine_label"].names
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
    return dataset, prompts


# ------------------------- Load Models -------------------------
# def load_model(model_id):
    
#     model   = CLIPModel.from_pretrained(model_id).eval().to(DEVICE)
#     proc = AutoProcessor.from_pretrained(model_id)


#     print(f'{model_id} loaded successfully')
#     return model, proc



def modified_clip(dataset, prompts,model_id=MODEL_ID, keep_pct=0.9):
    idx = 10
    print(dataset[idx])
    image = dataset[idx]['img']
    label = dataset[idx]['fine_label']
    prompt = prompts[label]
    print(f'Prompt for {label} =  {prompt}')
    # model, proc = load_model(model_id)
    # print(f'{clip.available_models() = }')
        # Load CLIP model
    model, proc = clip.load(model_id, DEVICE)
    # print(f'{model.dtype = }')
    print(f'{keep_pct = }')

    # Process image
    image_input = proc(image).unsqueeze(0).to(DEVICE).half()

    with torch.no_grad():
        x = model.visual.conv1(image_input).half()  # explicitly set half
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1).half()

        # Randomly remove 50% of patches
        num_patches = x.shape[1]
        print(f'{num_patches = }')
        keep_count = max(1, int((keep_pct) * num_patches))
        print(f'{keep_count = }')
        keep_indices = torch.randperm(num_patches, device=DEVICE)[:keep_count]
        print(f'before sorting {keep_indices = }')
        keep_indices, _ = torch.sort(keep_indices)
        print(f'after sorting {keep_indices = }')

        # keep_indices = torch.randperm(num_patches)[: num_patches // 2].to(DEVICE)
        x = x[:, keep_indices, :].half()

        cls_token = model.visual.class_embedding.half() + torch.zeros(x.shape[0], 1, x.shape[-1], device=DEVICE).half()
        x = torch.cat([cls_token, x], dim=1).half()

        positional_embedding = model.visual.positional_embedding[:x.size(1), :].unsqueeze(0).half()
        x = x + positional_embedding
        x = model.visual.ln_pre(x).half()

        x = x.permute(1, 0, 2).half()  # LND
        x = model.visual.transformer(x).half()
        x = x.permute(1, 0, 2).half()  # NLD

        x = model.visual.ln_post(x[:, 0, :]).half()

        if model.visual.proj is not None:
            image_features = x @ model.visual.proj.half()


    # Text features
    text_inputs = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(3)

    return values, indices

def og_clip(dataset, prompts, model_id=MODEL_ID):

    idx = 10
    print(dataset[idx])
    image = dataset[idx]['img']
    label = dataset[idx]['fine_label']
    prompt = prompts[label]
    print(f'Prompt for {label} =  {prompt}')
    # model, proc = load_model(model_id)
    # print(f'{clip.available_models() = }')
        # Load CLIP model
    model, proc = clip.load(model_id, DEVICE)
    image = proc(image).unsqueeze(0).to(DEVICE)
    text_inputs = clip.tokenize(prompts).to(DEVICE)


    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(3)    


    return values, indices


def main():

    dataset, prompts = load_data()
    # results = []
    Keep_PCT = 0.5
    mod_clip = modified_clip(dataset, prompts, keep_pct=Keep_PCT)
    clip_res = og_clip(dataset, prompts)

    print(f'Keeping {Keep_PCT} patches {mod_clip = }')
    print(f'{clip_res = }')


if __name__ == "__main__":
    main()