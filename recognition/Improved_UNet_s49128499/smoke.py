from dataset import load_data_2D
import glob, os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from modules import UNet2D as BasicUNet, dice_per_class

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")

image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "keras_slices_train", "*.nii.gz")))
label_paths = sorted(glob.glob(os.path.join(DATA_DIR, "keras_slices_seg_train", "*.nii.gz")))

# take tiny subset from image and labels
subs_images = image_paths[:20]
subs_labels = label_paths[:20]

# load some slices
X = load_data_2D(subs_images, normImage=True, categorical=False, early_stop=True)
y = load_data_2D(subs_labels, normImage=False, categorical=True, early_stop=True)

# print("X shape: ", X.shape)
# print("y shape: ", y.shape)

# convert to tensors
# for X: (N,H,W) to (N,1,H,W)
X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

# for y: (N,H,W,C) to (N,C,H,W)
y_t = torch.tensor(y, dtype=torch.float32).permute(0, 3, 1, 2)

# print("X_t shape: ", X_t.shape)
# print("y_t shape: ", y_t.shape)

loader = DataLoader(TensorDataset(X_t, y_t), batch_size=4, shuffle=True)

# device loading and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device:", device)

model = BasicUNet(in_channels=1, num_classes=6).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)