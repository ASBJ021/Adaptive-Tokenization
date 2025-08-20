import time, math, torch
import torch.nn.functional as F
from torch.nn.functional import normalize
import clip
from visual_utils import patchify, viz_patches, plot_heatmap_overlay, visualize_on_original
from data_utils import load_data
import yaml
import os
from genetic_patch_pruning import GeneticPatchPruner
import numpy as np
import random


def mask_patches(patches, indices_to_remove, model):
    with torch.no_grad():
        outputs = model.vision_model(pixel_values=patches)
        tokens = outputs.last_hidden_state.clone()
        for idx in indices_to_remove:
            tokens[:, idx + 1, :] = 0  # +1 offset for cls token
        masked_cls = model.visual_projection(tokens[:, 0, :])
        return masked_cls

# Placeholder functions (implement based on your model)
def get_confidence_score(patches, indices_to_remove, model, proc):
    # Return model confidence after removing indices
    masked_cls_embedding = mask_patches(patches, indices_to_remove, model)
    masked_cls_embedding /= masked_cls_embedding.norm(dim=-1, keepdim=True)
    logits = masked_cls_embedding @ text_features.T
    probs = logits.softmax(dim=-1)
    return probs.max(dim=-1)[0].item()
    # return random.uniform(0, 1)

def get_feature_preservation_score(patches, indices_to_remove, model, proc):
    # Return similarity of features before/after removal
    masked_cls_embedding = mask_patches(img, indices_to_remove, model)
    masked_cls_embedding /= masked_cls_embedding.norm(dim=-1, keepdim=True)
    orig_cls_norm = orig_cls_token / orig_cls_token.norm(dim=-1, keepdim=True)
    similarity = F.cosine_similarity(orig_cls_norm, masked_cls_embedding).item()
    # return random.uniform(0, 1)



# Fitness function combining multiple scores
def fitness_function(patches, indices, model, proc):
    if weights is None:
        weights = {'confidence': 0.4, 'feature': 0.6}

    conf = get_confidence_score(patches, indices, model , proc)
    feat = get_feature_preservation_score(patches, indices, model, proc)

    return weights['confidence'] * conf + weights['feature'] * feat

# Genetic Algorithm
def genetic_algorithm(
    img,
    patches,      
    keep,
    model, #clip model
    proc,  #clip processor 
    population_size = 20,
    generations = 30,
    mutation_rate = 0.1,
    
):
    # Initialize population (random indices)
    num_patches = patches.size(1)
    assert keep < num_patches, f"{keep = } is more than {num_patches = }"
    population = [
        random.sample(range(num_patches), keep)
        for _ in range(population_size)
    ]

    for gen in range(generations):
        # Evaluate fitness of each individual
        scores = [fitness_function(img, patches, model, proc) for indiv in population]
        total = sum(scores) or 1.0
        probs = [s / total for s in scores]

        # Select parents via roulette wheel
        parent_idxs = np.random.choice(
            range(population_size),
            size=population_size,
            p=probs
        )
        parents = [population[i] for i in parent_idxs]

        # Crossover and mutation
        next_gen = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % population_size]
            cp = random.randint(1, keep - 1)
            c1 = p1[:cp] + p2[cp:]
            c2 = p2[:cp] + p1[cp:]
            for child in (c1, c2):
                if random.random() < mutation_rate:
                    idx = random.randrange(keep)
                    child[idx] = random.randrange(num_patches)
                # ensure exactly `keep` unique indices
                child[:] = sorted(set(child))[:keep]
                while len(child) < keep:
                    child.append(random.randrange(num_patches))
                next_gen.append(child)
        population = next_gen

    # Return best individual by fitness
    best = max(population, key=lambda indiv: fitness_function(patches, indiv))
    return best


def load_model(model_id, device):
    model, proc = clip.load(model_id, device); model = model.float()
    return model, proc

def patch_modified_clip(dataset, prompts, MODEL_ID, DEVICE, keep_pct):

    model, proc = clip.load(MODEL_ID, DEVICE); model = model.float()

    # model = model.float()
    # precompute text embeddings
    toks = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        txt_feats = model.encode_text(toks)
        txt_feats /= txt_feats.norm(dim=-1,keepdim=True)
    results = []

    for item in dataset:
        print(f'Image id is {item = }')
        img, label = item["img"], item["fine_label"]
        img_input = proc(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # start = time.time()
            # extract patch tokens
            x = model.visual.conv1(img_input)
        B,D,N = x.shape[0], x.shape[1], x.numel()//(x.shape[0]*x.shape[1])
        x = x.reshape(B, D, -1).permute(0,2,1)
        num_patches = x.size(0)        
        keep = max(1, int(keep_pct * x.shape[1]))


        # select indices using Genetic Patch Prunning method
        iteration = 0
        while True:
            selected_indices = genetic_algorithm(img = img, patches = x, keep=keep, model=model, proc=proc)

            cls = model.visual.class_embedding + torch.zeros(1, 1, D, device=DEVICE)
            seq_all = torch.cat([cls, x], dim=1)
            pos_all = model.visual.positional_embedding[:seq_all.size(1)].unsqueeze(0)
            seq_all = model.visual.ln_pre(seq_all + pos_all)
            keep_idx = torch.tensor([0] + [i + 1 for i in selected_indices], device=DEVICE)
            seq = seq_all[:, keep_idx, :]

            # Classification
            z = model.visual.transformer(seq.permute(1,0,2))
            z = model.visual.ln_post(z.permute(1,0,2)[:,0])

            img_f = (z @ model.visual.proj) if model.visual.proj is not None else z
            img_f /= img_f.norm(dim=-1,keepdim=True)

            sim2 = (100*img_f @ txt_feats.T).softmax(-1)
            pred = sim2.argmax().item()

            print(f"Iteration {iteration}: Prediction={pred}, GT={label}")
            iteration += 1

            if pred == label or iteration >= 10:
                break
            

            results.append({
                'image_id': item['id'] if 'id' in item else None,
                'selected_indices': selected_indices
            })

    return results
            


def main():
    # ─── load config.yaml ────────────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    num_samples  = cfg["num_samples"]
    dataset_name = cfg["dataset_name"]
    model_id     = cfg["model_id"]
    keep_pct = cfg["keep_pct"]

    # --- logic -----------------------------------

    # 1) sampling info
    if num_samples != 0:
        print(f"Evaluating on {num_samples} samples of {dataset_name} dataset")
    else:
        print(f"Evaluating on full {dataset_name} dataset")

    # 2) load data & baseline
    dataset, prompts = load_data(dataset_name, num_samples)

    result  = patch_modified_clip(dataset, prompts, model_id, device,keep_pct)
    print(f'Top Selected Indices are: {result}')
    
if __name__ == "__main__":
    main()