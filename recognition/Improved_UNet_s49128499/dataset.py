"""
Code for reading Nifti image files are from 
Section B: Reading Nifti files of the Report Task Sheet (COMP3710_Report_2026_draft_v2.3)
"""

import numpy as np
import nibabel as nib
from tqdm import tqdm
import glob
import os
import torch
from torch.utils.data import Dataset

def to_channels(arr: np.ndarray, classes=(0, 1, 2, 3, 4, 5), dtype=np.uint8) -> np.ndarray:
    # channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(classes) ,), dtype=dtype)
    for idx, c in enumerate(classes):
        # c = int(c)
        res[..., idx][arr == c] = 1

    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick
    loading and testing scripts
    '''

    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_Study data

        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break
    if getAffines:
        return images, affines
    else:
        return images

class HipMRIDataset(Dataset):
    """
    Loads each slice each time for each call instead of loading everything in ram
    (Parts of the code are adapted from load_data_2D, specifically code under __getitem__)

    image_dir: directory of nifti images
    label_dir: directory of nifti masks
    transform: whether to apply a transform or not
    classes: class list for the one hot encoding
    """
    def __init__(self, image_dir, label_dir, transform=None, classes=(0, 1, 2, 3, 4, 5), dtype=np.float32):
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))
        self.transform = transform
        self.classes = classes
        self.dtype = dtype

    def __getitem__(self, key):
        # load img
        img = nib.load(self.image_paths[key]).get_fdata(caching='unchanged')
        if len(img) == 3:
            img = img[:, :, 0]

        # load mask
        lbl = nib.load(self.label_dir[key]).get_fdata(caching='unchanged')
        if len(lbl) == 3:
            lbl = lbl[:, :, 0]

        # normalise image
        img = img.astype(self.dtype)
        img = (img - img.mean()) / (img.std() + 1e-8)

        # on hot encode the mask
        lbl = lbl.astype(self.dtype)
        mask = to_channels(lbl, classes=self.classes, dtype=self.dtype)

        # to tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)

        # transform the image and mask tgt

        return img_tensor, mask_tensor

        