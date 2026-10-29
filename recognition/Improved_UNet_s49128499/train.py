from dataset import build_dataloaders
from modules import UNet2D

import os
import argparse
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")

train_loader, val_loader, test_loader = build_dataloaders(base_dir=DATA_DIR, batch_size=4, num_workers=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet2d", "improvedunet2d"], default="unet2d")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    # seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)