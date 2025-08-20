import math
import numpy as np
import torch
import clip
from PIL import Image, ImageDraw
import torchvision.transforms as T
from datasets import load_dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 100
DATASET_NAME = "cifar100"
MODEL_ID = "ViT-B/16"   # ViT-B/16 → 16px patches

# ----------------------------------------
# 1) Your helper functions
# ----------------------------------------
def load_image(img_path, resize=None, pil=False):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return image if pil else np.asarray(image).astype(np.float32) / 255.

def patchify(image_pil, resolution, patch_size, patch_stride=None):
    """
    image_pil: PIL Image, e.g. orig_img
    returns: Tensor of shape (N, 3, patch_size, patch_size)
    """
    image_pil = image_pil.resize((resolution, resolution))
    img_t = T.ToTensor()(image_pil)                # 3×H×W
    if patch_stride is None:
        patch_stride = patch_size
    # patches = img_t.unfold(1, patch_size, patch_stride) \
    #                .unfold(2, patch_size, patch_stride)
    # # → (3, n_h, n_w, p, p)
    # patches = patches.contiguous()\
    #                  .view(3, -1, patch_size, patch_size)\
    #                  .permute(1, 0, 2, 3)          # (N,3,p,p)

    patches = img_t.unfold(
        1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
    patches = patches.reshape(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    return patches

def viz_patches(x, figsize=(8, 8), topk=None, t=5, img_title='test'):
    """
    x: (N,3,patch,patch)
    topk: list or Tensor of selected patch indices
    draws a grid of patches and outlines those in topk in yellow
    """
    n = x.shape[0]
    ncols = int(math.sqrt(n))
    fig, axes = plt.subplots(ncols, ncols, figsize=figsize)
    
    fig.suptitle(img_title, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        im = x[i].permute(1,2,0).numpy()
        im = (im*255).round().astype(np.uint8)
        if topk is not None and i in set(int(j) for j in topk):
            # draw yellow border
            im[0:t] = (255, 255, 0)
            im[im.shape[0]-t:] = (255, 255, 0)
            im[:, 0:t] = (255, 255, 0)
            im[:, im.shape[1]-t:] = (255, 255, 0)
        ax.imshow(im)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_heatmap_overlay(orig_img: Image.Image,
                         patch_sims,
                         grid_size,
                         alpha=0.5,
                         cmap="jet"):
    """
    orig_img:    PIL Image
    patch_sims:  1D array-like of length N
    grid_size:   (rows, cols)
    alpha:       float opacity of the heatmap overlay
    cmap:        matplotlib colormap
    """
    # 1) make the low-res heatmap image
    heatmap = np.array(patch_sims).reshape(grid_size)
    # normalize to [0..1]
    hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 2) upsample to full image size
    H, W = orig_img.size[1], orig_img.size[0]   # PIL: size=(W,H)
    hm_img = Image.fromarray(np.uint8(hm_norm * 255), mode="L") \
                  .resize((W, H), resample=Image.BILINEAR)

    # 3) colorize via a colormap
    hm_colored = plt.get_cmap(cmap)(np.array(hm_img) / 255.0)[..., :3]
    hm_colored = np.uint8(hm_colored * 255)

    # 4) blend with original
    blended = np.array(orig_img).astype(np.float32) * (1 - alpha) \
            + hm_colored.astype(np.float32) * alpha
    blended = np.uint8(np.clip(blended, 0, 255))

    # 5) display
    plt.figure(figsize=(6,6))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Original + Patch Heatmap")
    plt.show()


# ----------------------------------------
# 2) Main logic
# ----------------------------------------
def load_data():
    ds = load_dataset(DATASET_NAME, split="test")
    ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))
    names = ds.features["fine_label"].names
    prompts = [f"a photo of a {n.replace('_',' ')}" for n in names]
    return ds, prompts

def modified_clip_dropout(dataset, prompts, keep_pct=0.5, seed=42):
    torch.manual_seed(seed)
    model, preprocess = clip.load(MODEL_ID, DEVICE)
    model = model.eval().float()

    # text features
    toks = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_feats = model.encode_text(toks)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    # pick one sample
    img_pil, label = dataset[50]["img"], dataset[10]["fine_label"]
    img_t = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    img_pil.show()

    with torch.no_grad():
        # extract patch tokens
        x = model.visual.conv1(img_t)                  # B×D×H'×W'
        x = x.reshape(x.shape[0], x.shape[1], -1)      # B×D×N
        x = x.permute(0,2,1)                           # B×N×D
        B, N, D = x.shape

        k = max(1, int(keep_pct * N))
        idx = torch.randperm(N, device=DEVICE)[:k].sort()[0]

        # build trimmed sequence
        cls = model.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(B,1,D,device=DEVICE,dtype=x.dtype)
        seq_full = torch.cat([cls, x], dim=1)
        pos    = model.visual.positional_embedding[: seq_full.size(1)]
        seq = model.visual.ln_pre(seq_full + pos.unsqueeze(0))[:, torch.cat([torch.zeros(1,dtype=torch.long,device=DEVICE), idx+1]), :]
        v = seq.permute(1,0,2)
        out = model.visual.transformer(v).permute(1,0,2)
        out = model.visual.ln_post(out[:,0])
        feat = out @ model.visual.proj if model.visual.proj is not None else out
        feat /= feat.norm(dim=-1, keepdim=True)

        sims = (100*feat @ text_feats.T).softmax(dim=-1)
        pred = sims.argmax(-1).item()

    print(f"GT = {label}  |  Pred = {pred}")

    # 3) DRAW on the original
    # visualize_on_original(img_pil, idx)

    # 4) SHOW patch grid with highlights
    patches = patchify(img_pil, resolution=224, patch_size=16)
    viz_patches(patches, topk=idx.cpu())

def visualize_on_original(orig_img, selected_idx):
    """
    Draws red rectangles on orig_img PIL object and shows/saves it.
    """
    # extract CLIP’s resize+crop
    pre = clip.load(MODEL_ID, DEVICE)[1]  # just to get transforms
    resize = next(t for t in pre.transforms if isinstance(t, T.Resize))
    crop   = next(t for t in pre.transforms if isinstance(t, T.CenterCrop))

    target = resize.size if isinstance(resize.size,int) else resize.size[0]
    crop_s = crop.size   if isinstance(crop.size,int)   else crop.size[0]
    ow, oh = orig_img.size
    if ow<oh:
        nw, nh = target, int(target*oh/ow)
    else:
        nh, nw = target, int(target*ow/oh)

    left = (nw - crop_s)//2;  top = (nh - crop_s)//2
    sx = ow / nw;  sy = oh / nh

    draw = ImageDraw.Draw(orig_img)
    for ii in selected_idx.cpu().tolist():
        r,c = divmod(ii, crop_s//16)
        x0c, y0c = c*16, r*16
        rect = [
            (x0c+left)*sx, (y0c+top)*sy,
            (x0c+left+16)*sx, (y0c+top+16)*sy
        ]
        draw.rectangle(rect, outline="red", width=1)

    orig_img.save("highlighted.png")
    orig_img.show()


def main():
    ds, prompts = load_data()
    modified_clip_dropout(ds, prompts)

if __name__ == "__main__":
    main()
