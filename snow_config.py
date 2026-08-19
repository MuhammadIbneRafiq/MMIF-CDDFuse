"""Shared paths/settings for the snowpole signal+range CDDFuse experiment.

Override the dataset location without editing code by setting the
SNOW_DATASET_ROOT environment variable (useful when running on a remote GPU box):

    export SNOW_DATASET_ROOT=/home/me/data/SnowPole_Detection_Dataset
"""
import os

# Root containing signal/, range/, reflec/, nearir/ and the centralized labels/
DEFAULT_DATASET_ROOT = (
    r"C:\Users\wifi stuff\OneDrive - TU Eindhoven\Snowpole_image_fusion"
    r"\Effect-of-Image-Fusion-on-Snowpole-Detection\SnowPole_Detection_Dataset"
)
DATASET_ROOT = os.environ.get("SNOW_DATASET_ROOT", DEFAULT_DATASET_ROOT)

# Modality roles. CDDFuse is a 2-input network: one image fills the "VIS" role
# (the decoder's residual input) and one fills the "IR" role.
MODALITY_VIS = "signal"
MODALITY_IR = "range"

PAIR_NAME = f"{MODALITY_VIS}_{MODALITY_IR}"

# Splits as they exist on disk. Fusion training pools train+valid; test is held out.
FUSION_TRAIN_SPLITS = ["train", "valid"]
ALL_SPLITS = ["train", "valid", "test"]

# Patch extraction (same values as the original dataprocessing.py)
IMG_SIZE = 128
STRIDE = 128

H5_NAME = f"Snow_{PAIR_NAME}_imgsize_{IMG_SIZE}_stride_{STRIDE}.h5"
H5_PATH = os.path.join("data", H5_NAME)


def modality_dir(modality, split):
    return os.path.join(DATASET_ROOT, modality, "images", split)


def labels_dir(split):
    """Labels are centralized (shared across modalities) at <root>/labels/<split>."""
    return os.path.join(DATASET_ROOT, "labels", split)
