"""STEP 3 - evaluation, both halves of the experiment.

  A) Fusion quality on the held-out test split: EN SD SF MI SCD VIF Qabf SSIM,
     printed in the same table format as test_MIF.py / test_IVF.py. Any number of
     rival fusion directories can be scored in the same table so you can see
     whether CDDFuse beats the classical methods you already have.

  B) Detection quality on the test split: runs ultralytics val() on trained
     YOLO weights and prints P / R / mAP50 / mAP50-95.

    # fusion metrics only, CDDFuse vs the classical baselines you already built
    python evaluate_snow.py --fused-dir <root>/cddfuse_signal_range \
        --compare fused_avg fused_laplacian fused_wavelet fused_max

    # detection metrics only
    python evaluate_snow.py --skip-fusion-metrics \
        --yolo-weights runs/detect/cddfuse_signal_range_11n/weights/best.pt \
        --yolo-data <root>/cddfuse_signal_range/data.yaml

    # both
    python evaluate_snow.py --fused-dir ... --yolo-weights ... --yolo-data ...

All MI/SCD/VIF/Qabf/SSIM values are computed against the SAME two sources
(signal + range), so rows stay comparable even when a rival method fused a
different set of modalities.
"""
import argparse
import os
import warnings
import logging

import numpy as np
from skimage.metrics import structural_similarity

from utils.Evaluator import Evaluator
from utils.img_read_save import image_read_cv2
import snow_config as C

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

METRIC_HEADER = "\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM"

# utils/Evaluator.SSIM calls skimage without data_range, which modern skimage
# rejects for float input. Images here are 0-255, so state it explicitly; this
# lands near the paper's published values, whereas skimage 0.19's old float
# default (data_range=2) does not.
SSIM_DATA_RANGE = 255


def ssim_sum(image_F, image_A, image_B):
    """Same definition as Evaluator.SSIM: ssim(F,A) + ssim(F,B)."""
    return (structural_similarity(image_F, image_A, data_range=SSIM_DATA_RANGE)
            + structural_similarity(image_F, image_B, data_range=SSIM_DATA_RANGE))


def find_split_dir(root, split):
    """Accept both <root>/images/<split> and <root>/<split> layouts."""
    for cand in (os.path.join(root, "images", split), os.path.join(root, split)):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"No '{split}' images under {root}")


def score_fused_dir(fused_dir, vis_dir, ir_dir, image_names):
    """Mean of the 8 CDDFuse metrics over image_names. A=range(IR), B=signal(VIS),
    matching the (fi, ir, vi) argument order used in test_IVF.py."""
    acc = np.zeros(8)
    used = 0
    for img_name in image_names:
        fused_path = os.path.join(fused_dir, img_name)
        if not os.path.exists(fused_path):
            continue
        ir = image_read_cv2(os.path.join(ir_dir, img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(vis_dir, img_name), 'GRAY')
        fi = image_read_cv2(fused_path, 'GRAY')
        acc += np.array([
            Evaluator.EN(fi), Evaluator.SD(fi),
            Evaluator.SF(fi), Evaluator.MI(fi, ir, vi),
            Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi),
            Evaluator.Qabf(fi, ir, vi), ssim_sum(fi, ir, vi),
        ])
        used += 1
    if used == 0:
        return None, 0
    return acc / used, used


def print_row(label, result):
    cells = "\t".join(str(np.round(v, 2)) for v in result)
    pad = label if len(label) >= 15 else label + " " * (15 - len(label))
    print(pad + "\t" + cells)


def run_fusion_metrics(args):
    split = args.split
    vis_dir = C.modality_dir(C.MODALITY_VIS, split)
    ir_dir = C.modality_dir(C.MODALITY_IR, split)
    image_names = sorted(n for n in os.listdir(vis_dir) if n.lower().endswith('.png'))

    print("\n" + "=" * 80)
    print(f"Fusion metrics on '{split}' split  ({len(image_names)} images)")
    print(f"sources: {C.MODALITY_VIS} + {C.MODALITY_IR}")
    print(METRIC_HEADER)

    # The two source modalities themselves, as a floor to beat.
    if args.include_sources:
        for label, d in ((C.MODALITY_VIS, vis_dir), (C.MODALITY_IR, ir_dir)):
            res, n = score_fused_dir(d, vis_dir, ir_dir, image_names)
            if res is not None:
                print_row(label, res)

    targets = []
    if args.fused_dir:
        targets.append(("CDDFuse", args.fused_dir))
    for name in args.compare:
        root = name if os.path.isabs(name) else os.path.join(C.DATASET_ROOT, name)
        targets.append((os.path.basename(name.rstrip("/\\")), root))

    for label, root in targets:
        try:
            d = find_split_dir(root, split)
        except FileNotFoundError as e:
            print(f"  (skipped {label}: {e})")
            continue
        res, n = score_fused_dir(d, vis_dir, ir_dir, image_names)
        if res is None:
            print(f"  (skipped {label}: no matching filenames in {d})")
            continue
        if n != len(image_names):
            print(f"  note: {label} scored on {n}/{len(image_names)} images")
        print_row(label, res)

    print("=" * 80)


def run_detection_metrics(args):
    from ultralytics import YOLO

    print("\n" + "=" * 80)
    print(f"Detection metrics on '{args.split}' split")
    print(f"weights: {args.yolo_weights}")
    print(f"data:    {args.yolo_data}")

    model = YOLO(args.yolo_weights)
    metrics = model.val(data=args.yolo_data, split=args.split,
                        imgsz=args.imgsz, batch=args.batch,
                        conf=args.conf, iou=args.iou, device=args.device)
    b = metrics.box
    print("\n\t\t P\t R\tmAP50\tmAP50-95")
    print("YOLOv11n\t" + "\t".join(str(np.round(v, 4)) for v in
                                   (b.mp, b.mr, b.map50, b.map)))
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=C.ALL_SPLITS)

    # A) fusion metrics
    ap.add_argument("--fused-dir", default=None,
                    help="CDDFuse output root from fuse_and_detect_snow.py")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="other fusion dirs to score in the same table; bare names "
                         "are resolved against the dataset root")
    ap.add_argument("--include-sources", action="store_true",
                    help="also score the raw signal/range images as a baseline row")
    ap.add_argument("--skip-fusion-metrics", action="store_true")

    # B) detection metrics
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--yolo-data", default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if not args.skip_fusion_metrics and (args.fused_dir or args.compare):
        run_fusion_metrics(args)

    if args.yolo_weights:
        if not args.yolo_data:
            ap.error("--yolo-data is required alongside --yolo-weights")
        run_detection_metrics(args)


if __name__ == "__main__":
    main()
