import time
import torch
from datasets import load_dataset
import clip
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'{DEVICE = }')

NUM_SAMPLES = 1000
DATASET_NAME = "cifar100"
MODEL_ID = 'ViT-B/16'


# --- Dataset and Prompts ---
def load_data():
    if NUM_SAMPLES == 0:
        dataset = load_dataset(DATASET_NAME, split="test").shuffle(seed=42)
    else:
        dataset = load_dataset(DATASET_NAME, split="test").shuffle(seed=42).select(range(NUM_SAMPLES))

    
    classnames = dataset.features["fine_label"].names
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
    return dataset, prompts

# --- Original CLIP Inference ---
def original_clip(dataset, prompts, model_id=MODEL_ID):
    model, proc = clip.load(model_id, DEVICE)
    # model = model.float()  # use float32

    total_time = 0.0
    correct = 0
    for item in dataset:
        image = item['img']
        label = item['fine_label']
        text_inputs = clip.tokenize(prompts).to(DEVICE)

        # preprocess image
        image_input = proc(image).unsqueeze(0).to(DEVICE).float()

        start = time.time()
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred = similarity[0].argmax().item()
        elapsed = time.time() - start

        total_time += elapsed
        correct += int(pred == label)

    accuracy = correct / len(dataset)
    avg_time = total_time / len(dataset)
    return accuracy, avg_time

# --- Modified CLIP with 50% Patch Dropout ---
def modified_clip(dataset, prompts, model_id=MODEL_ID):
    model, proc = clip.load(model_id, DEVICE)
    model = model.float()

    total_time = 0.0
    correct = 0
    for item in dataset:
        image = item['img']
        label = item['fine_label']
        prompt = prompts[label]
        text_inputs = clip.tokenize(prompts).to(DEVICE)

        # preprocess image
        image_input = proc(image).unsqueeze(0).to(DEVICE).float()

        with torch.no_grad():
            start = time.time()
            # original ViT stem
            x = model.visual.conv1(image_input)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

            # randomly keep 50% of patches
            num_patches = x.shape[1]
            keep = torch.randperm(num_patches)[: num_patches // 2].to(DEVICE)
            x = x[:, keep, :]

            # prepend cls token
            cls = model.visual.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=DEVICE)
            x = torch.cat([cls, x], dim=1)

            # positional embeddings
            pos = model.visual.positional_embedding[: x.size(1), :].unsqueeze(0)
            x = x + pos
            x = model.visual.ln_pre(x)

            # transformer
            x = x.permute(1, 0, 2)
            x = model.visual.transformer(x)
            x = x.permute(1, 0, 2)
            x = model.visual.ln_post(x[:, 0, :])

            # projection
            if model.visual.proj is not None:
                image_features = x @ model.visual.proj

            # text features and similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred = similarity[0].argmax().item()
            elapsed = time.time() - start

        total_time += elapsed
        correct += int(pred == label)

    accuracy = correct / len(dataset)
    avg_time = total_time / len(dataset)
    return accuracy, avg_time


def modified_clip_dropout(dataset, prompts, model_id=MODEL_ID, keep_pct=0.9):
    """
    Runs CLIP inference while randomly dropping a fraction of visual patches.

    Args:
        dataset: HuggingFace dataset of images and labels.
        prompts: List of text prompts corresponding to class labels.
        model_id: CLIP model identifier.
        dropout_pct: Fraction of patches to drop (0.0 = no dropout; 1.0 = drop all patches).
    Returns:
        (accuracy, avg_inference_time)
    """
    model, proc = clip.load(model_id, DEVICE)
    print(f'{keep_pct = }')

    total_time = 0.0
    correct = 0
    for item in dataset:
        image = item['img']
        label = item['fine_label']
        text_inputs = clip.tokenize(prompts).to(DEVICE)

        # preprocess image
        image_input = proc(image).unsqueeze(0).to(DEVICE).half()

        with torch.no_grad():
            start = time.time()
            # original ViT stem
            x = model.visual.conv1(image_input).half()
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1).half()

            # randomly drop a fraction of patches
            num_patches = x.shape[1]
            keep_count = max(1, int((keep_pct) * num_patches))
            # print(f'{keep_count = }')
            keep_indices = torch.randperm(num_patches, device=DEVICE)[:keep_count]
            keep_indices, _ = torch.sort(keep_indices)
            x = x[:, keep_indices, :].half()

            # prepend cls token
            cls = model.visual.class_embedding.half() + torch.zeros(x.shape[0], 1, x.shape[-1], device=DEVICE).half()
            x = torch.cat([cls, x], dim=1).half()

            # positional embeddings for remaining tokens
            pos = model.visual.positional_embedding[: x.size(1), :].unsqueeze(0).half()
            x = x + pos
            x = model.visual.ln_pre(x).half()

            # transformer
            x = x.permute(1, 0, 2).half()  # LND
            x = model.visual.transformer(x).half()
            x = x.permute(1, 0, 2).half()  # NLD

            x = model.visual.ln_post(x[:, 0, :]).half()

            if model.visual.proj is not None:
                image_features = x @ model.visual.proj.half()

            # text features and similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred = similarity[0].argmax().item()
            elapsed = time.time() - start

        total_time += elapsed
        correct += int(pred == label)

    accuracy = correct / len(dataset)
    avg_time = total_time / len(dataset)
    return accuracy, avg_time

# --- Main: Compare both ---
def main():
    dataset, prompts = load_data()
    # Keep_PCT = 0.8
    results = []
    # mod_clip = modified_clip(dataset, prompts, keep_pct=Keep_PCT)

    orig_acc, orig_time = original_clip(dataset, prompts)
    # mod_acc, mod_time = modified_clip_dropout(dataset, prompts, keep_pct=Keep_PCT)

    # print(f'Evaluating on {NUM_SAMPLES}')

    # print(f"Original CLIP   - Accuracy: {orig_acc*100:.2f}%, Avg Time: {orig_time:.4f}s")
    # print(f"Modified ({Keep_PCT*100} % selected ) - Accuracy: {mod_acc*100:.2f}%, Avg Time: {mod_time:.4f}s")

    # start at 100%, decrease by 10% down to 0%
    for pct in [i/10 for i in range(10, -1, -1)]:
        acc, t = modified_clip_dropout(dataset, prompts, keep_pct=pct)
        results.append({'keep_pct': pct, 'accuracy': acc, 'avg_time': t})

    df = pd.DataFrame(results)
    if NUM_SAMPLES != 0:
        print(f"Evaluating on {NUM_SAMPLES} samples of {DATASET_NAME} dataset")
    else: 
        print(f"Evaluating on full {DATASET_NAME} dataset")
    print(f"Original CLIP (100% patches) - Accuracy: {orig_acc*100:.2f}%, Time: {orig_time:.4f}s")
    print(df.to_string(index=False, formatters={
        'keep_pct': '{:.0%}'.format,
        'accuracy': lambda x: f"{x*100:.2f}%",
        'avg_time': lambda x: f"{x:.4f}s"
    }))

if __name__ == "__main__":
    main()
