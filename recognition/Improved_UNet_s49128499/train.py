from dataset import build_dataloaders
from modules import UNet2D

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")



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

    # data
    train_loader, val_loader, _ = build_dataloaders(base_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    # print("Train batches: ", len(train_loader))
    # print("Validation batches: ", len(val_loader))

    # model
    if args.model == "unet2d":
        model = UNet2D(in_channels=1, num_classes=6).to(device)

    # weighted loss
    weights = torch.tensor([0.1, 0.1, 0.5, 2.0, 5.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training

    for epoch in range(1, args.epochs + 1):
        pass

if __name__ == "__main__":
    main()