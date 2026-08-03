import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TestDataset
from model import UniV, get_backbone_input_config, is_dinov3_backbone
from train import collate_video_batch, move_batch_to_device, resolve_checkpoint_path

DATA = "/222010008/yuncheng/Data"
RES = "/222010008/yuncheng/UniV/results"
DATASETS = [
    ("echocp", f"{DATA}/EchoCP/preprocessed/", 2, f"{RES}/univ_dinov3_echocp/log_2026-05-14_03-57-28"),
    ("erus", f"{DATA}/ERUS/", 5, f"{RES}/univ_dinov3_erus/log_2026-05-14_03-56-41"),
    ("jnu", f"{DATA}/JNU-IFM/preprocessed/", 4, f"{RES}/univ_dinov3_jnu/log_2026-05-13_07-32-25"),
    ("tus", f"{DATA}/TUS/preprocessed/", 2, f"{RES}/univ_dinov3_tus/log_2026-05-14_07-01-41"),
]


def build_args(num_classes, backbone="vit_base_patch16_dinov3"):
    return argparse.Namespace(backbone=backbone, num_classes=num_classes, clip_size=3, train_size=256, temporal_backend="pytorch", ssm=True, use_mode_pseudo_labels=True, pretrained=None)


@torch.no_grad()
def run_dataset(name, data_root, num_classes, ckpt_dir, device, out_dir):
    args = build_args(num_classes)
    cfg = get_backbone_input_config(args.backbone)
    if is_dinov3_backbone(args.backbone):
        args.train_size = cfg["train_size"]
    model = UniV(args).to(device).eval()
    ckpt_path = resolve_checkpoint_path(os.path.join(ckpt_dir, "best_dice.pth"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()}, strict=False)
    ds = TestDataset(data_root, args.train_size, args.clip_size, split="test", image_mean=cfg["mean"], image_std=cfg["std"], image_interpolation=cfg["interpolation"], use_mode_pseudo_labels=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_video_batch)
    center = args.clip_size // 2
    rows = []
    for batch in loader:
        b = move_batch_to_device(batch, device)
        out = model(b["images"], b["mode_label"])
        H, W = b["seg_masks"].shape[-2:]
        logits = F.interpolate(out["seg_logits"], size=(H, W), mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits).view(b["images"].shape[0], args.clip_size, 1, H, W)
        pred = (probs[:, center, 0] > 0.5).float()
        gt = (b["seg_masks"][:, center, 0] > 0.5).float()
        inter = (pred * gt).sum(dim=(1, 2))
        denom = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
        dice = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
        for i, vid in enumerate(batch["video_id"]):
            m = (pred[i].cpu().numpy() * 255).astype(np.uint8)
            rows.append((vid, batch["center_frame"][i], float(dice[i]), float(gt[i].sum()), m))
    ddir = os.path.join(out_dir, name)
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "dice.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "frame", "dice", "gt_area"])
        for vid, fr, d, ga, _ in rows:
            w.writerow([vid, fr, f"{d:.6f}", int(ga)])
    lesion = [r for r in rows if r[3] > 0]
    top10 = sorted(lesion, key=lambda r: r[2], reverse=True)[:10]
    mdir = os.path.join(ddir, "top10_masks")
    os.makedirs(mdir, exist_ok=True)
    for vid, fr, d, ga, mask in top10:
        stem = fr.replace(".jpg", "").replace(".png", "")
        cv2.imwrite(os.path.join(mdir, f"{vid}_{stem}.png"), mask)
    mean_all = float(np.mean([r[2] for r in rows])) if rows else 0.0
    mean_lesion = float(np.mean([r[2] for r in lesion])) if lesion else 0.0
    print(f"\n===== {name} ===== frames={len(rows)} lesion_frames={len(lesion)} mean_dice_all={mean_all:.4f} mean_dice_lesion={mean_lesion:.4f}")
    print(f"{'rank':<5}{'video_id':<22}{'frame':<24}{'dice':<10}{'gt_area':<8}")
    for i, (vid, fr, d, ga, _) in enumerate(top10, 1):
        print(f"{i:<5}{vid:<22}{fr:<24}{d:<10.4f}{int(ga):<8}")
    return name, len(rows), mean_all, mean_lesion


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(RES, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir: {out_dir}\ndevice: {device}")
    summary = []
    for name, root, ncls, ckpt in DATASETS:
        summary.append(run_dataset(name, root, ncls, ckpt, device, out_dir))
    print("\n===== SUMMARY =====")
    print(f"{'dataset':<10}{'frames':<10}{'mean_dice_all':<16}{'mean_dice_lesion':<16}")
    for n, fr, ma, ml in summary:
        print(f"{n:<10}{fr:<10}{ma:<16.4f}{ml:<16.4f}")
    print(f"\nresults saved under: {out_dir}")
