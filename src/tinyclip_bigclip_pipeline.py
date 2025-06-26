import torch
from torch.nn.functional import normalize, cosine_similarity
from transformers import AutoModel, AutoProcessor, CLIPModel
from PIL import Image
from datasets import load_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'{DEVICE = }')
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
    print(f'{D_big = } & {D_input = }')


    # if D_input != D_big:
    #     print(f'D_big != D_input')
    #     mapper = torch.nn.Linear(D_input, D_big).to(DEVICE)
    #     selected_patches = mapper(selected_patches)

    # Reuse Big CLIP's learned CLS token and positional embeddings
    # cls_token = vision_model.embeddings.class_embedding.clone().detach().unsqueeze(0)  # [1, D]
    cls_token = vision_model.embeddings.class_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
    # cls_token = cls_token.unsqueeze(0)  # [1, 1, D]

    seq_len = selected_patches.shape[0] + 1
    pos_embed = vision_model.embeddings.position_embedding.weight[:seq_len].unsqueeze(0)  # [1, seq_len, D]

    x = torch.cat([cls_token, selected_patches.unsqueeze(0)], dim=1) + pos_embed
    x = vision_model.pre_layrnorm(x)
    x = vision_model.encoder(x)
    x = vision_model.post_layernorm(x[0])
    image_emb = proj(x[:, 0, :])  # CLS token

    return normalize(image_emb, dim=-1)



def inject_selected_bigclip_tokens(big_clip_model, selected_patch_tokens):
    """
    Inject selected patch tokens into BigCLIP's transformer, preserving CLS and position handling.

    Args:
        selected_patch_tokens: Tensor [K, D] — patch tokens selected from get_patch_tokens()

    Returns:
        image_emb: Tensor [1, D] — normalized image embedding from BigCLIP
    """
    vision_model = big_clip_model.vision_model
    proj = big_clip_model.visual_projection
    K = selected_patch_tokens.shape[0]
    D = vision_model.config.hidden_size
    # print(f'{selected_patch_tokens.shape = }')
    # print(f'Vision Hidden Parans : {D = }')

    # [CLS] + selected tokens
    cls_token = vision_model.embeddings.class_embedding.unsqueeze(0).unsqueeze(0)  # [1,1,D]
    patch_seq = torch.cat([cls_token, selected_patch_tokens.unsqueeze(0)], dim=1)  # [1,K+1,D]

    # print(f'{cls_token.shape = }')
    # print(f'{patch_seq.shape = }')


    # Positional embedding
    pos_embed = vision_model.embeddings.position_embedding.weight[:K+1].unsqueeze(0).to(patch_seq.device)
    x = patch_seq + pos_embed

    # ViT forward
    x = vision_model.pre_layrnorm(x)
    x = vision_model.encoder(x)[0]
    x = vision_model.post_layernorm(x)
    image_emb = proj(x[:, 0, :])  # CLS token
    # print('Image Embd: ')
    # print(image_emb.shape)
    return normalize(image_emb, dim=-1)



# --------------------- Full CLIP Prediction ---------------------
def classify_full_clip(big_clip_model, big_clip_processor, img, prompt):
    inputs = big_clip_processor(text=prompt, images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    big_clip_model.eval()
    with torch.no_grad():
        outputs = big_clip_model(**inputs)

    # Extract similarity scores
    logits_per_image = outputs.logits_per_image   # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    return probs



# ------------------------- Main Pipeline -------------------------

def main():
    # Load sample image
    dataset = load_dataset("timm/imagenet-1k-wds", split="validation")
    image = dataset[10]['jpg']
    prompt = "main object in the image"

    # Load models
    tinyclip, tiny_proc, bigclip, big_proc = load_models()

    # Step 1: TinyCLIP patch tokens for similarity
    tiny_patch_tokens = get_patch_tokens(tinyclip, tiny_proc, image)         # [196, 512]
    tiny_text_emb = get_text_embedding(tinyclip, tiny_proc, prompt)          # [1, 512]
    selected_indices, sims = select_top_k_indices(tiny_patch_tokens, tiny_text_emb, top_k_percent=100)

    print(f"tiny_patch_tokens.shape = {tiny_patch_tokens.shape}")
    print(f"sims.shape = {sims.shape}")

    # Step 2: BigCLIP patch tokens (used for injection)
    big_patch_tokens = get_patch_tokens(bigclip, big_proc, image)            # [196, 768]
    selected_big_tokens = big_patch_tokens[selected_indices]  
    print(f'{selected_big_tokens.shape = }')               # [K, 768]

    # Step 3: Inject into BigCLIP
    injected_emb = inject_selected_bigclip_tokens(bigclip, selected_big_tokens)  # [1, 768]
    print(f"injected_emb.shape = {injected_emb.shape}")

    # Step 4: Get full CLIP embedding from get_image_features
    full_inputs = big_proc(images=image, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        full_emb = normalize(bigclip.get_image_features(**full_inputs), dim=-1)  # [1, 768]

    print(f'{full_emb.shape = }')

    # Step 5: Compare full vs injected
    cos_sim = cosine_similarity(full_emb, injected_emb).item()
    print(f"✅ Cosine similarity (full CLIP vs 100% patch injection): {cos_sim:.6f}")

    # Step 6: Compare injected embedding to text embedding
    text_emb = get_text_embedding(bigclip, big_proc, prompt)
    text_emb = normalize(text_emb, dim=-1)
    relevance = cosine_similarity(injected_emb, text_emb).item()
    print(f"✅ Similarity score (injected BigCLIP vs '{prompt}'): {relevance:.4f}")



if __name__ == "__main__":
    main()
