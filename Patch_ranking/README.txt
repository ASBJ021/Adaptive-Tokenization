CNN for predicting selected patch indices from images

Files:
- Patch_ranking/dataset.py: Dataset wrapper joining HF dataset and JSONL annotations.
- Patch_ranking/cnn.py: Small 3-layer CNN producing multi-label logits.
- Patch_ranking/train_cnn.py: Training script using load_data_normal.

Example:
python -m Patch_ranking.train_cnn \
  --dataset_name cifar100 \
  --annotations patch_ranking/cifar100_10_final_patches_50.jsonl \
  --epochs 10 --batch_size 64 --lr 1e-3 --val_split 0.1

Notes:
- The script imports load_data_normal from new_src/data_utils.py. If run from repo root, it will add new_src to PYTHONPATH automatically.
- Targets are multi-hot vectors over patch indices; loss is BCEWithLogitsLoss.
