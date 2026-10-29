from dataset import build_dataloaders
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")

train_loader, val_loader, test_loader = build_dataloaders(base_dir=DATA_DIR, batch_size=4, num_workers=0)

print("Train slices:", len(train_loader.dataset))
print("Val slices:", len(val_loader.dataset))
print("Test slices:", len(test_loader.dataset))

xb, yb = next(iter(train_loader))
print("Batch shape:", xb.shape, yb.shape)