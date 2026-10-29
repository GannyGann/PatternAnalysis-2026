"""
predict.py, used to evaluate the trained basic 2D U-Net, and Improved 2D U-Net

Produces the following:
per-class Dice, precision, recall, prostate HD95 in mm, comparison figures, and failure case autopsy panels

Usage:
python predict.py --model unet2d
python predict.py --model improved
"""
import argparse
import os
import glob

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import label as cc_label, binary_erosion, distance_transform_edt
import time

from modules import UNet2D
from dataset import HipMRIDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")

PROSTATE = 5
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

# per class metrics
def per_class_metrics(pred, gt, num_classes=6, eps=1e-6):
    """
    returns dict of arrays of dice, prec, and recall
    """
    dice = np.zeros(num_classes)
    prec = np.zeros(num_classes)
    rec = np.zeros(num_classes)

    for c in range(num_classes):
        p = (pred == c)
        t = (gt == c)
        tp = (p & t).sum()
        fp = (p & ~t).sum()
        fn = (~p & t).sum()
        dice[c] = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        prec[c] = (tp + eps) / (tp + fp + eps)
        rec[c] = (tp + eps) / (tp + fn + eps)

    return {"dice": dice, "precision": prec, "recall": rec}

# hd95 function
def hd95_2d(pred_mask, gt_mask, spacing=(1.0, 1.0)):
    """
    returns HD95 in mm between 2 binary masks, returns NaN if either is empty
    uses symmetric surface distances, 95th percentile
    """
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")

    # extract surface via binary erosion
    pred_surf = pred_mask ^ binary_erosion(pred_mask)
    gt_surf = gt_mask ^ binary_erosion(gt_mask)

    # distance from each surface point to other surface points
    dt_to_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = dt_to_gt[pred_surf]
    d_gt_to_pred = dt_to_pred[gt_surf]

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    return float(np.percentile(all_d, 95))

# fragment count function
def count_fragments(mask, min_size=50):
    """
    counts the number of connected components larger than the min size
    """
    labeled, n = cc_label(mask)
    if n == 0:
        return 0
    sizes = np.bincount(labeled.ravel())
    return int((sizes[1:] > min_size).sum())

# parameter counting function
def count_parameters(model):
    """
    counts how many learnable numbers are in the model
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# latency measuring function
@torch.no_grad()
def measure_latency(model, test_dataset, device, n_warmup=5, n_measure=50):
    """
    measures how long the model takes to segment a slice. Timing starts after 5 passes (n_warmup)
    and times 50 passes (n_measure). Median over mean as mean is sensitive to outliers. P95 for the worst latency
    95% of slices experience
    """
    model.eval()
    for i in range(n_warmup):
        img, _ = test_dataset[i % len(test_dataset)]
        _ = model(img.unsqueeze(0).to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for i in range(n_measure):
        img, _ = test_dataset[i % len(test_dataset)]
        img = img.unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {"median": np.median(times), "p95": np.percentile(times, 95)}


# evaluation function

# plots

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet2d", "improvedunet2d"], default="unet2d")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--n-panels", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load model
    if args.model == "unet2d":
        model = UNet2D(in_channels=1, num_classes=6).to(device)
        chk_path = os.path.join(SCRIPT_DIR, "models", "best_unet2d.pt")
    else:
        chk_path = os.path.join(SCRIPT_DIR, "models", "best_improvedunet2d.pt")

    
    chk = torch.load(chk_path, map_location=device)
    model.load_state_dict(chk["model_state"])

    print(f"Loaded {chk_path}")
    print(f"Epoch: {chk.get('epoch', '?')}")
    print(f"Val Dice {chk.get('val_dice', float('nan')):.4f}")

    # load test dataset
    test_dataset = HipMRIDataset(
        os.path.join(args.data_dir, "keras_slices_test"),
        os.path.join(args.data_dir, "keras_slices_seg_test"),
        transform=None,
    )
    print(f"Test slices: {len(test_dataset)}")

if __name__ == "__main__":
    main()