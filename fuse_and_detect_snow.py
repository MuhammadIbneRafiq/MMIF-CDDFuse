"""STEP 2 - run the trained fusion net over every split, build a YOLO dataset
from the fused images, and (optionally) fine-tune YOLOv11n on it.

The fusion weights cannot detect anything by themselves: this script uses them
to *render* fused PNGs, then hands those images to YOLO as an ordinary detection
dataset. Bounding boxes are reused unchanged from the centralized labels/ dir,
because the fused image has the same size and pixel alignment as signal/range.

    # generate fused images + data.yaml only
    python fuse_and_detect_snow.py --ckpt models/CDDFuse_snow_signal_range_<ts>.pth

    # ... and immediately fine-tune YOLOv11n on them
    python fuse_and_detect_snow.py --ckpt models/... --train-yolo

Outputs:
    <out-root>/images/{train,valid,test}/*.png    fused images
    <out-root>/labels/{train,valid,test}/*.txt    copied labels
    <out-root>/data.yaml                          YOLO dataset config
"""
import argparse
import os
import shutil
import warnings
import logging

import numpy as np
import torch
import torch.nn as nn
import cv2

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.img_read_save import image_read_cv2
import snow_config as C

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)


def _pick(ckpt, *candidates):
    for k in candidates:
        if k in ckpt:
            return ckpt[k]
    raise KeyError(f"none of {candidates} in checkpoint (has: {list(ckpt.keys())})")


def load_fusion_models(ckpt_path, device):
    """Checkpoints are saved from nn.DataParallel, so wrap the same way to match keys.

    train.py in this repo saves CDDF_*, but the released pretrained weights use
    the paper's original DIDF_* naming, so accept either.
    """
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    Encoder.load_state_dict(_pick(ckpt, 'CDDF_Encoder', 'DIDF_Encoder'))
    Decoder.load_state_dict(_pick(ckpt, 'CDDF_Decoder', 'DIDF_Decoder'))
    BaseFuseLayer.load_state_dict(ckpt['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(ckpt['DetailFuseLayer'])

    for m in (Encoder, Decoder, BaseFuseLayer, DetailFuseLayer):
        m.eval()
    return Encoder, Decoder, BaseFuseLayer, DetailFuseLayer


def fuse_split(models, split, out_root, device):
    Encoder, Decoder, BaseFuseLayer, DetailFuseLayer = models

    vis_dir = C.modality_dir(C.MODALITY_VIS, split)
    ir_dir = C.modality_dir(C.MODALITY_IR, split)
    img_out = os.path.join(out_root, "images", split)
    lbl_out = os.path.join(out_root, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    names = sorted(n for n in os.listdir(vis_dir) if n.lower().endswith('.png'))
    src_lbl_dir = C.labels_dir(split)
    missing_labels = []

    with torch.no_grad():
        for k, img_name in enumerate(names, 1):
            data_VIS = image_read_cv2(os.path.join(vis_dir, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            data_IR = image_read_cv2(os.path.join(ir_dir, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0

            data_VIS = torch.FloatTensor(data_VIS).to(device)
            data_IR = torch.FloatTensor(data_IR).to(device)

            feature_V_B, feature_V_D, _ = Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            # data_VIS as residual input, matching Phase II of train_snow.py
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)

            data_min, data_max = torch.min(data_Fuse), torch.max(data_Fuse)
            data_Fuse = (data_Fuse - data_min) / (data_max - data_min).clamp_min(1e-8)
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())

            # Save 3-channel so the fused set is a drop-in match for the
            # single-modality baselines (which are RGB with R==G==B).
            fi = np.clip(np.round(fi), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(img_out, img_name), cv2.cvtColor(fi, cv2.COLOR_GRAY2BGR))

            stem = os.path.splitext(img_name)[0]
            src_lbl = os.path.join(src_lbl_dir, stem + ".txt")
            if os.path.exists(src_lbl):
                shutil.copyfile(src_lbl, os.path.join(lbl_out, stem + ".txt"))
            else:
                missing_labels.append(stem)

            if k % 200 == 0 or k == len(names):
                print(f"  [{split}] {k}/{len(names)}")

    if missing_labels:
        print(f"  WARNING [{split}]: {len(missing_labels)} images had no label file "
              f"(e.g. {missing_labels[:3]})")
    return len(names)


def write_data_yaml(out_root, path):
    content = (
        f"# CDDFuse-fused {C.MODALITY_VIS}+{C.MODALITY_IR} snowpole dataset\n"
        f"path: {os.path.abspath(out_root)}\n"
        f"train: images/train\n"
        f"val: images/valid\n"
        f"test: images/test\n"
        f"\nnc: 1\n"
        f"names:\n  0: pole\n"
    )
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fusion weights from train_snow.py")
    ap.add_argument("--out-root", default=None,
                    help="where to write the fused YOLO dataset "
                         "(default: <dataset-root>/cddfuse_<pair>)")
    ap.add_argument("--splits", nargs="+", default=C.ALL_SPLITS)
    ap.add_argument("--skip-fusion", action="store_true",
                    help="reuse already-generated fused images, just (re)train YOLO")
    ap.add_argument("--train-yolo", action="store_true")
    ap.add_argument("--yolo-model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None, help="ultralytics run name")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(C.DATASET_ROOT, f"cddfuse_{C.PAIR_NAME}")
    yaml_path = os.path.join(out_root, "data.yaml")

    if not args.skip_fusion:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading fusion weights {args.ckpt} on {device}")
        models = load_fusion_models(args.ckpt, device)
        for split in args.splits:
            n = fuse_split(models, split, out_root, device)
            print(f"Fused {n} images for split '{split}'")

    os.makedirs(out_root, exist_ok=True)
    write_data_yaml(out_root, yaml_path)

    if args.train_yolo:
        from ultralytics import YOLO
        run_name = args.name or f"cddfuse_{C.PAIR_NAME}_11n"
        print(f"\nFine-tuning {args.yolo_model} on fused images -> run '{run_name}'")
        model = YOLO(args.yolo_model)
        model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, name=run_name)
    else:
        print("\nFused dataset ready. To fine-tune YOLOv11n on it:")
        print(f'  yolo train model=yolo11n.pt data="{yaml_path}" '
              f'epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} '
              f'device={args.device} name=cddfuse_{C.PAIR_NAME}_11n')


if __name__ == "__main__":
    main()
