"""
Code for reading Nifti image files are from 
Section B: Reading Nifti files of the Report Task Sheet (COMP3710_Report_2026_draft_v2.3)
"""

import numpy as np
import nibabel as nib
from tqdm import tqdm

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels) ,), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

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