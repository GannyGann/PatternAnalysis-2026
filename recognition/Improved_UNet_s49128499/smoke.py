from dataset import load_data_2D
import glob, os

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

print("X shape: ", X.shape)
print("y shape: ", y.shape)