import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1. Load CIFAR-100 test set
dataset = load_dataset("cifar100", split="test")

# 2. Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 3. Get class names and convert to CLIP-style prompts
classnames = dataset.features["fine_label"].names
prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]

# 4. Preprocess text prompts
text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    text_features = model.get_text_features(**{k: v.cuda() for k, v in text_inputs.items()})
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 5. Define image transform (resize to 224x224 + CLIP normalization)
transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758))
])

# 6. Evaluate on N samples
correct = 0
total = 0
NUM_SAMPLES = 100  # change to 10000 for full evaluation

for item in dataset.shuffle(seed=42).select(range(NUM_SAMPLES)):
    image: Image.Image = item["img"]
    label = item["fine_label"]

    image_tensor = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        pred = logits.argmax(dim=-1).item()
        # print(f'{pred = }')

        correct += int(pred == label)
        total += 1

print(f"Zero-shot Top-1 Accuracy on CIFAR-100 ({total} samples): {correct / total:.4f}")
