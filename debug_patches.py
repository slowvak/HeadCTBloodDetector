#!/usr/bin/env python3
"""Quick check: load one batch and verify shapes, dtypes, and label values."""
import sys
import os
os.chdir('/Users/bje/repos/HeadCTBloodDetector')

import json
import torch
import numpy as np

with open('finetune_config.json') as f:
    config = json.load(f)

print("Config num_classes:", config['data']['num_classes'])
print("Config class_names:", config['data']['class_names'])

from blast_ct.read_config import get_model, get_train_loader

model = get_model(config)
input_size  = tuple(config['training']['input_patch_size'])
output_size = model.get_output_size(input_size)
print(f"\nInput patch : {input_size}  ({np.prod(input_size):,} voxels)")
print(f"Output patch: {output_size}  ({np.prod(output_size):,} voxels)")

loader = get_train_loader(config, model, 'finetune_data/train.csv', use_cuda=False)
batch  = next(iter(loader))

target = batch['target']
image  = batch['image']
mask   = batch['sampling_mask']

print(f"\nimage  shape : {image.shape}   dtype: {image.dtype}")
print(f"target shape : {target.shape}  dtype: {target.dtype}")
print(f"mask   shape : {mask.shape}   dtype: {mask.dtype}")
print(f"target unique values : {torch.unique(target).tolist()}")
print(f"target has nan       : {torch.isnan(target.float()).any().item()}")
print(f"target min/max       : {target.min().item()} / {target.max().item()}")
print(f"mask unique (<=20)   : {torch.unique(mask)[:20].tolist()}")

# Simulate forward pass + metrics
model.eval()
with torch.no_grad():
    logits, _ = model(image)

print(f"\nlogits shape : {logits.shape}")
_, pred = torch.max(logits, dim=1)
print(f"pred   shape : {pred.shape}")
print(f"pred   unique: {torch.unique(pred).tolist()}")

print(f"\ntarget == pred shape match: {target.shape == pred.shape}")
if target.shape != pred.shape:
    print("  *** MISMATCH — target won't align with pred in confusion matrix ***")
else:
    # Build confusion matrix manually
    n = config['data']['num_classes']
    cm = torch.zeros(n, n, dtype=torch.long)
    t = target.flatten().long()
    p = pred.flatten().long()
    valid = (t >= 0) & (t < n)
    print(f"  Valid target voxels: {valid.sum().item()} / {valid.numel()}")
    for i in range(n):
        for j in range(n):
            cm[i, j] = ((t == i) & (p == j)).sum()
    print(f"\nConfusion matrix:\n{cm.numpy()}")
    print(f"Row sums (target class counts): {cm.sum(dim=1).tolist()}")
    print(f"Col sums (pred class counts)  : {cm.sum(dim=0).tolist()}")
