"""Convert blast-ct PyTorch .torch_model weights to MLX-compatible .npz files.

PyTorch layout: Conv3d weight (C_out, C_in, kD, kH, kW)
MLX layout:     Conv3d weight (C_out, kD, kH, kW, C_in)  — channels last

Key mapping:
  path(5, 5, 5)0.path.N.bn.weight  →  paths.0.path.layers.N.bn.weight
  fully_connected.N.conv.weight     →  fully_connected.layers.N.conv.weight
"""
import os
import re
from pathlib import Path

import numpy as np


def pytorch_to_mlx_key(key: str) -> str:
    """Map a PyTorch state-dict key to the equivalent MLX parameter path.

    Two transformations needed:
      1. path(5, 5, 5)0  →  paths.0   (top-level Python list — direct index, no .layers.)
      2. .path.N.        →  .path.layers.N.    (nn.Sequential attribute inside Path)
         fully_connected.N.  →  fully_connected.layers.N.  (nn.Sequential)
    """
    # path(X, Y, Z)N → paths.N
    key = re.sub(r'path\([^)]+\)(\d+)', lambda m: f'paths.{m.group(1)}', key)
    # .path.N. → .path.layers.N.   (the Sequential inside each Path module)
    key = re.sub(r'(?<=\.path\.)(\d+)(?=\.)', r'layers.\1', key)
    # fully_connected.N. → fully_connected.layers.N.
    key = re.sub(r'(?<=fully_connected\.)(\d+)(?=\.)', r'layers.\1', key)
    return key


def convert_torch_model(src: os.PathLike, dst: os.PathLike) -> None:
    """Convert a single .torch_model file to an MLX-compatible .npz file."""
    import torch
    state_dict = torch.load(str(src), map_location='cpu')
    mlx_weights: dict = {}
    for pt_key, tensor in state_dict.items():
        if 'num_batches_tracked' in pt_key:
            continue  # not used by MLX BatchNorm
        arr = tensor.float().numpy()
        if pt_key.endswith('conv.weight'):
            # (C_out, C_in, kD, kH, kW) → (C_out, kD, kH, kW, C_in)
            arr = arr.transpose(0, 2, 3, 4, 1)
        mlx_weights[pytorch_to_mlx_key(pt_key)] = arr
    np.savez(str(dst), **mlx_weights)
    print(f"  Saved {len(mlx_weights)} tensors → {Path(dst).name}")


def convert_saved_models(install_dir: str | None = None) -> Path:
    """Convert all .torch_model files in data/saved_models/ to .npz.

    Skips files that have already been converted.
    Returns the directory containing the .npz files.
    """
    if install_dir is None:
        install_dir = os.path.dirname(os.path.realpath(__file__))
    src_dir = Path(install_dir) / 'data' / 'saved_models'
    dst_dir = Path(install_dir) / 'data' / 'mlx_models'
    dst_dir.mkdir(exist_ok=True)
    for src in sorted(src_dir.glob('*.torch_model')):
        dst = dst_dir / (src.stem + '.npz')
        if dst.exists():
            continue
        print(f"  Converting {src.name} …")
        convert_torch_model(src, dst)
    return dst_dir


if __name__ == '__main__':
    print('Converting blast-ct PyTorch weights → MLX …')
    out = convert_saved_models()
    print(f'Done. Files written to {out}')
