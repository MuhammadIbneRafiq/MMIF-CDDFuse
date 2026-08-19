"""STEP 1a - build the fusion training set.

Copy of dataprocessing.py repointed at the snowpole signal+range modality pair
instead of MSRS ir/vi. train and valid are pooled into one patch pool; test is
never touched here so it stays clean for evaluation.

    python dataprocessing_snow.py

Writes data/Snow_signal_range_imgsize_128_stride_200.h5
"""
import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread

import snow_config as C

VALID_EXT = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')


def get_img_file(dirs):
    files = []
    for d in dirs:
        for filename in sorted(os.listdir(d)):
            if filename.lower().endswith(VALID_EXT):
                files.append(os.path.join(d, filename))
    return files


def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y


def read_as_y(path):
    """Both signal and range are stored as 3-channel PNGs with R==G==B, so both
    go through the same RGB->Y path (unlike MSRS, where IR is natively 1-channel)."""
    img = imread(path).astype(np.float32)
    if img.ndim == 2:
        return img[None, :, :] / 255.
    img = img.transpose(2, 0, 1) / 255.
    return rgb2y(img[:3])


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10, upper_percentile=90):
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    if limits[1] <= 0:
        return True
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


VIS_files = get_img_file([C.modality_dir(C.MODALITY_VIS, s) for s in C.FUSION_TRAIN_SPLITS])
IR_files = get_img_file([C.modality_dir(C.MODALITY_IR, s) for s in C.FUSION_TRAIN_SPLITS])

assert len(IR_files) == len(VIS_files), \
    f"{C.MODALITY_IR}: {len(IR_files)} vs {C.MODALITY_VIS}: {len(VIS_files)}"
assert len(IR_files) > 0, f"No images found under {C.DATASET_ROOT}"
for a, b in zip(VIS_files, IR_files):
    assert os.path.basename(a) == os.path.basename(b), f"Unpaired: {a} vs {b}"

print(f"{len(VIS_files)} paired {C.MODALITY_VIS}/{C.MODALITY_IR} images "
      f"from splits {C.FUSION_TRAIN_SPLITS}")

os.makedirs("data", exist_ok=True)
h5f = h5py.File(C.H5_PATH, 'w')
h5_ir = h5f.create_group('ir_patchs')
h5_vis = h5f.create_group('vis_patchs')
train_num = 0
for i in tqdm(range(len(IR_files))):
    I_VIS = read_as_y(VIS_files[i])   # [1, H, W]
    I_IR = read_as_y(IR_files[i])     # [1, H, W]

    I_IR_Patch_Group = Im2Patch(I_IR, C.IMG_SIZE, C.STRIDE)
    I_VIS_Patch_Group = Im2Patch(I_VIS, C.IMG_SIZE, C.STRIDE)

    for ii in range(I_IR_Patch_Group.shape[-1]):
        bad_IR = is_low_contrast(I_IR_Patch_Group[0, :, :, ii])
        bad_VIS = is_low_contrast(I_VIS_Patch_Group[0, :, :, ii])
        if not (bad_IR or bad_VIS):
            avl_IR = I_IR_Patch_Group[0, :, :, ii][None, ...]
            avl_VIS = I_VIS_Patch_Group[0, :, :, ii][None, ...]

            h5_ir.create_dataset(str(train_num), data=avl_IR, dtype=avl_IR.dtype, shape=avl_IR.shape)
            h5_vis.create_dataset(str(train_num), data=avl_VIS, dtype=avl_VIS.dtype, shape=avl_VIS.shape)
            train_num += 1

h5f.close()
print(f"\nWrote {train_num} patch pairs -> {C.H5_PATH}")

with h5py.File(C.H5_PATH, "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
