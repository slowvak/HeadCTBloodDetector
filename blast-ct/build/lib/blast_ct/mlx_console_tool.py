"""MLX-accelerated single-image inference entry point.

Mirrors console_tool.py but runs the forward pass on Apple Silicon via MLX.
The existing PyTorch data pipeline (DataLoader, NiftiPatchSaver) is reused
unchanged — only the model and inference loop are swapped.

Usage (after `uv sync`):
    blast-ct-mlx --input scan.nii.gz --output prediction.nii.gz
"""
import argparse
import json
import os
import shutil
import sys

import mlx.core as mx
import numpy as np
import pandas as pd
import torch

from blast_ct.mlx_convert import convert_saved_models
from blast_ct.mlx_model import (
    DROPOUT, FEATURE_MAPS, FULLY_CONNECTED, DeepMedic,
)
from blast_ct.nifti.savers import NiftiPatchSaver
from blast_ct.read_config import get_test_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(config: dict) -> DeepMedic:
    dm_cfg = config['model']['DeepMedic']
    return DeepMedic(
        input_channels=config['data']['input_channels'],
        num_classes=config['data']['num_classes'],
        feature_maps=dm_cfg.get('feature_maps', FEATURE_MAPS),
        fully_connected=dm_cfg.get('fully_connected', FULLY_CONNECTED),
    )


def _load_weights(model: DeepMedic, npz_path: str) -> None:
    weights = mx.load(npz_path)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())


def _to_ndhwc(pt_tensor) -> np.ndarray:
    """PyTorch NCDHW tensor → numpy NDHWC array."""
    return pt_tensor.numpy().transpose(0, 2, 3, 4, 1)


def _prob_to_torch(prob_ndhwc: np.ndarray):
    """numpy NDHWC probability map → PyTorch NCDHW tensor."""
    return torch.from_numpy(prob_ndhwc.transpose(0, 4, 1, 2, 3).copy())


def _run_model(model: DeepMedic, image_np: np.ndarray):
    """Run one forward pass; returns (prob_ndhwc, pred_ndhw) as numpy arrays."""
    image_mlx = mx.array(image_np)
    logits, _ = model(image_mlx)
    prob = mx.softmax(logits, axis=-1)
    mx.eval(prob)
    prob_np = np.array(prob)
    pred_np = prob_np.argmax(axis=-1)
    return prob_np, pred_np


# ---------------------------------------------------------------------------
# Inference loops
# ---------------------------------------------------------------------------

def _infer_single(model: DeepMedic, dataloader, saver: NiftiPatchSaver) -> None:
    model.eval()
    for inputs in dataloader:
        prob_np, pred_np = _run_model(model, _to_ndhwc(inputs['image']))
        state = {
            'prob': _prob_to_torch(prob_np),
            'pred': torch.from_numpy(pred_np),
        }
        msg = saver(state)
        if msg:
            print(msg)


def _infer_ensemble(model: DeepMedic, dataloader, saver: NiftiPatchSaver,
                    model_paths: list) -> None:
    """Average probabilities from multiple models, then save once."""
    model.eval()
    # Collect per-model probability arrays for each batch
    all_probs: list[list[np.ndarray]] = []
    for npz in model_paths:
        _load_weights(model, str(npz))
        model_probs: list[np.ndarray] = []
        for inputs in dataloader:
            prob_np, _ = _run_model(model, _to_ndhwc(inputs['image']))
            model_probs.append(prob_np)
        all_probs.append(model_probs)

    n_batches = len(all_probs[0])
    for b in range(n_batches):
        avg_prob = np.mean([all_probs[m][b] for m in range(len(model_paths))], axis=0)
        state = {
            'prob': _prob_to_torch(avg_prob),
            'pred': torch.from_numpy(avg_prob.argmax(axis=-1)),
        }
        msg = saver(state)
        if msg:
            print(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _path(string):
    if os.path.exists(string):
        return string
    sys.exit(f'File not found: {string}')


def console_tool_mlx():
    parser = argparse.ArgumentParser(
        description='blast-ct inference via MLX (Apple Silicon).'
    )
    parser.add_argument('--input', type=_path, required=True,
                        help='Input NIfTI file (.nii or .nii.gz).')
    parser.add_argument('--output', type=str, required=True,
                        help='Output prediction path (.nii.gz).')
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='Use ensemble of 12 models (slower, more accurate).')
    parser.add_argument('--do-localisation', action='store_true', default=False,
                        help='Calculate lesion volume per brain region.')
    args, _ = parser.parse_known_args()

    if not (args.input.endswith('.nii.gz') or args.input.endswith('.nii')):
        raise IOError('Input must be .nii or .nii.gz')
    if not args.output.endswith('.nii.gz'):
        raise IOError('Output must be .nii.gz')

    install_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(install_dir, 'data/config.json')) as f:
        config = json.load(f)
    config['test']['batch_size'] = 1  # safer for large patches on MLX

    # --- Build MLX model ---
    model = _build_model(config)

    # --- Convert weights on first run (skipped if already done) ---
    print('Checking MLX weights …')
    mlx_dir = convert_saved_models(install_dir)

    # --- Data pipeline (existing PyTorch DataLoader, unchanged) ---
    job_dir = '/tmp/blast_ct_mlx'
    os.makedirs(job_dir, exist_ok=True)
    test_csv = os.path.join(job_dir, 'test.csv')
    pd.DataFrame([['im_0', args.input]], columns=['id', 'image']).to_csv(test_csv, index=False)

    test_loader = get_test_loader(config, model, test_csv, use_cuda=False)
    saver = NiftiPatchSaver(job_dir, test_loader, write_prob_maps=False,
                            do_localisation=args.do_localisation)

    # --- Run inference ---
    if args.ensemble:
        model_paths = sorted(mlx_dir.glob('model_*.npz'))[:12]
        if not model_paths:
            sys.exit('No converted MLX models found in ' + str(mlx_dir))
        print(f'Ensemble inference with {len(model_paths)} models …')
        _infer_ensemble(model, test_loader, saver, model_paths)
    else:
        npz = mlx_dir / 'model_1.npz'
        if not npz.exists():
            sys.exit(f'Model not found: {npz}')
        _load_weights(model, str(npz))
        print('Running MLX inference …')
        _infer_single(model, test_loader, saver)

    # --- Copy output to requested path ---
    pred_csv = os.path.join(job_dir, 'predictions/prediction.csv')
    output_df = pd.read_csv(pred_csv)
    shutil.copyfile(output_df.loc[0, 'prediction'], args.output)
    shutil.rmtree(job_dir)
    print(f'\nDone. Prediction saved to: {args.output}')
