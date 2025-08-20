# model_utils.py

import time, math, torch
from torch.nn.functional import normalize
import clip
from visual_utils import patchify, viz_patches, plot_heatmap_overlay, visualize_on_original


def original_clip(dataset, prompts, MODEL_ID, DEVICE):
    model, proc = clip.load(MODEL_ID, DEVICE)
    model = model.float()
    total, correct = 0.0, 0
    for item in dataset:
        img, label = item["img"], item["fine_label"]
        img_input = proc(img).unsqueeze(0).to(DEVICE)
        toks = clip.tokenize(prompts).to(DEVICE)
        start = time.time()
        with torch.no_grad():
            img_f = model.encode_image(img_input)
            txt_f = model.encode_text(toks)
            img_f /= img_f.norm(dim=-1,keepdim=True)
            txt_f /= txt_f.norm(dim=-1,keepdim=True)
            sim = (100*img_f@txt_f.T).softmax(-1)
            pred = sim.argmax().item()
        total += time.time() - start
        correct += (pred==label)
    return correct/len(dataset), total/len(dataset)


def modified_clip_dropout(dataset, prompts,  MODEL_ID, DEVICE, keep_pct=0.5, strategy="random", seed=42, visualize=False):
    torch.manual_seed(seed)
    model, proc = clip.load(MODEL_ID, DEVICE); model = model.float()
    # precompute text embeddings
    toks = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        txt_feats = model.encode_text(toks)
        txt_feats /= txt_feats.norm(dim=-1,keepdim=True)

    
    total, correct = 0.0, 0
    for item in dataset:
        img, label = item["img"], item["fine_label"]
        feat_tgt = txt_feats[label:label+1]
        img_input = proc(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            start = time.time()
            # extract patch tokens
            x = model.visual.conv1(img_input)
            B,D,N = x.shape[0], x.shape[1], x.numel()//(x.shape[0]*x.shape[1])
            x = x.reshape(B, D, -1).permute(0,2,1)
            keep = max(1, int(keep_pct * x.shape[1]))

            # choose indices
            if strategy=="random":
                idx = torch.randperm(x.shape[1], device=DEVICE)[:keep]
            elif strategy=="uniform":
                g = int(math.sqrt(x.shape[1])); step = max(1, g//int(math.sqrt(keep)))
                coords = [(i,j) for i in range(0,g,step) for j in range(0,g,step)]
                idx = torch.tensor([i*g+j for i,j in coords],device=DEVICE)[:keep]
            elif strategy=="similarity":
                pos = model.visual.positional_embedding[1:N+1].unsqueeze(0)
                x_pe = model.visual.ln_pre(x + pos)
                patch_e = x_pe @ model.visual.proj if model.visual.proj is not None else x_pe
                sims = normalize(patch_e.squeeze(0),dim=-1) @ normalize(feat_tgt.squeeze(0),dim=-1)
                idx = sims.topk(keep).indices
                # plot_heatmap_overlay(img, sims.cpu().numpy(), (g,g), alpha=0.4)
            else:
                raise ValueError(f"Unknown strategy {strategy}")
            idx = torch.sort(idx)[0]

            
            # rebuild sequence & forward
            cls = model.visual.class_embedding + torch.zeros(1,1,D,device=DEVICE)
            seq_all = torch.cat([cls, x], dim=1)
            pos_all = model.visual.positional_embedding[:seq_all.size(1)].unsqueeze(0)
            seq_all = model.visual.ln_pre(seq_all + pos_all)
            keep_idx = torch.cat([torch.zeros(1,dtype=torch.long,device=DEVICE), idx+1])
            seq = seq_all[:, keep_idx, :]
            # learnable_img_tokens = ''

            
            z = model.visual.transformer(seq.permute(1,0,2))
            z = model.visual.ln_post(z.permute(1,0,2)[:,0])

            img_f = (z @ model.visual.proj) if model.visual.proj is not None else z
            img_f /= img_f.norm(dim=-1,keepdim=True)
            sim2 = (100*img_f @ txt_feats.T).softmax(-1)
            pred = sim2.argmax().item()
            total += time.time() - start
            correct += (pred==label)
            # optional: viz patches
            if visualize:
                patches = patchify(img, resolution=224, patch_size=16)
                viz_patches(patches, topk=idx.cpu(), img_title=f"{strategy}_{keep_pct}_{label}")
    return correct/len(dataset), total/len(dataset)
