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
from scipy.ndimage import label as cc_label

from modules import UNet2D
from dataset import HipMRIDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet2d", "improveduned2d"], default="unet2d")
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