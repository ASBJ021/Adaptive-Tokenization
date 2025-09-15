import time, math, torch
from torch.nn.functional import normalize
import clip
from visual_utils import patchify, viz_patches, plot_heatmap_overlay, visualize_on_original
from data_utils import load_data, load_data_normal
import os
import yaml
from beautifultable import BeautifulTable

# ─── load config.yaml ────────────────────────────────────────────────
cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

device = cfg.get("device", "cuda")
if not torch.cuda.is_available():
    device = "cpu"

# device = "cpu"

num_samples  = cfg["num_samples"]
dataset_name = cfg["dataset_name"]
model_id     = cfg["model_id"]
split = cfg["split"]
viz = cfg['visualize']

# load model
model, processor = clip.load(model_id, device); model = model.float()

# modified transformer larer

def modified_clip_tflayers(dataset, prompts, model, proc, DEVICE, keep_pct=0.9):
    """
    Runs one example from `dataset` through CLIP’s vision tower,
    pruning the bottom (1–keep_pct) fraction of patches **at each Transformer layer** 
    according to the CLS-token’s self-attention scores.
    """
    # 1) tokenize & encode all prompts
    toks = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        txt_feats = model.encode_text(toks)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)

    # 2) grab a single example
    # idx        = 10
    # item       = dataset[idx]

    # print(f'{keep_pct = }')

    total, correct, prun_ratio = 0.0, 0, 0.0
    for item in dataset:
        img, label = item["image"], item["label"]
        img_input  = proc(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            start = time.time()
            # --- patchify & add class token + pos embed ---
            x = model.visual.conv1(img_input)                                   # B×C×H'×W'
            x = x.reshape(x.shape[0], x.shape[1], -1)                            # B×C×(H'·W')
            x = x.permute(0, 2, 1)                                               # B×N×C
            cls = model.visual.class_embedding.to(x.dtype) \
                + torch.zeros(x.shape[0], 1, x.shape[-1],
                                dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)                                       # B×(N+1)×C
            x = x + model.visual.positional_embedding.to(x.dtype)                # B×(N+1)×C
            x = model.visual.ln_pre(x)                                           # B×(N+1)×C

            # shape for transformer: (seq, batch, dim)
            x = x.permute(1, 0, 2)    
            og_shape = x.shape                                           # L×B×D

            # print(f'{og_shape = }')

            # --- iterate Transformer layers and prune ---
            for block in model.visual.transformer.resblocks:
                # 1) self-attn + residual
                res = x
                x_ln1 = block.ln_1(x)
                attn_out, attn_w = block.attn(
                    x_ln1, x_ln1, x_ln1,
                    need_weights=True,
                    attn_mask=block.attn_mask
                )
                x = res + attn_out

                # 2) MLP + residual
                res = x
                x = res + block.mlp(block.ln_2(x))

                # 3) CLS-attention pruning
                L, B, D = x.shape               # seq_len, batch, dim
                # print(f'Initial {x.shape = }')
                if L > 2:
                    # ensure attn_w has a head dimension
                    # attn_w is either (B, heads, L, L) or (B, L, L)
                    aw = attn_w
                    # print(f'attn_w = {aw.shape = }')
                    if aw.dim() == 3:
                        aw = aw.unsqueeze(1)    # make it (B,1,L,L)
                    aw = aw.mean(dim=1)        # now (B,L,L)

                    # take CLS (query idx=0) → all other tokens (1…L-1)
                    cls_scores = aw[:, 0, 1:]  # (B, L-1)

                    # how many to keep?
                    keep = max(1, int(math.ceil(keep_pct * (L - 1))))

                    # top-k indices per example
                    _, topk = torch.topk(cls_scores, keep, dim=1)  # (B, keep)

                    # gather CLS + those topk tokens
                    x_b = x.permute(1, 0, 2)                       # B×L×D
                    batch_idx = torch.arange(B, device=x.device)[:, None]
                    gather_idx = torch.cat([
                        torch.zeros(B, 1, dtype=torch.long, device=x.device), 
                        topk + 1
                    ], dim=1)                                       # B×(keep+1)

                    pruned = x_b[batch_idx, gather_idx]            # B×(keep+1)×D
                    x = pruned.permute(1, 0, 2)                    # new_L×B×D
                    # print(f'After Prunning {x.shape = }')

            # back to (batch, seq, dim)
            x = x.permute(1, 0, 2)                                   # B×L'×D
            prunned_shape = x.shape
            # print(f'{prunned_shape = }')

            # --- final pooling & projection as usual ---
            x = model.visual.ln_post(x[:, 0, :])                    # B×D
            if model.visual.proj is not None:
                img_f = x @ model.visual.proj
            else:
                img_f = x

            img_f /= img_f.norm(dim=-1, keepdim=True)
            sim2  = (100 * img_f @ txt_feats.T).softmax(-1)
            pred  = sim2.argmax().item()

            total += time.time() - start
            correct += (pred==label)
            prun_ratio += prunned_shape[1] / og_shape[0]
            # print(f'{prun_ratio = }')

            # if pred == label:
                # print(f'Correct Prediction: {pred = } == {label = }')

    return correct/len(dataset), total/len(dataset), prun_ratio/len(dataset)

# ─── main ────────────────────────────────────────────────

def main():

     # 1) sampling info
    if num_samples != 0:
        print(f"Evaluating on {num_samples} samples of {dataset_name} dataset ({device = })")
    else:
        print(f"Evaluating on full {dataset_name} dataset ({device = })")


    ds, prompts = load_data_normal(dataset_name, num_samples, split)
    # idx = 10

    keep_pct = [1.0, 0.95, 0.9, 0.85, 0.80]
    records = []

    for pct in keep_pct:
        print(f'{pct = }')
        acc, avg_time, avg_prun_ratio = modified_clip_tflayers(ds, prompts, model, processor, device, keep_pct=pct)
        
        records.append({
                'keep_pct':      pct,
                'accuracy':      acc,
                'avg_time':      avg_time,
                'avg_pr':       avg_prun_ratio,           
            })

    # print(records)
    table = BeautifulTable()
    table.columns.header = ['Keep %', 'Accuracy', 'Avg. Time (s)', 'Average Prunning Ratio (in % Tokens Left)']

    # Populate the table with records
    for rec in records:
        table.rows.append([
            f"{rec['keep_pct']*100:.0f}%",
            f"{rec['accuracy']*100:.2f}%",
            f"{rec['avg_time']:.3f}",
            f"{rec['avg_pr']*100:.2f}%"

        ])

    # Print the formatted table
    print(table)


    

if __name__ == "__main__":
    main()