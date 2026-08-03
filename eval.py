import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import TestDataset
from model import ASTR, get_backbone_input_config, is_dinov3_backbone
from train import collate_video_batch, evaluate, move_batch_to_device, resolve_checkpoint_path


def qualitative_eval(model, loader, device, save_path, clip_size, limit_batches=0):
    output_dir = os.path.join(save_path, "qualitative")
    os.makedirs(output_dir, exist_ok=True)
    center_index = clip_size // 2

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if limit_batches and step >= limit_batches:
                break
            batch_cpu = batch
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], batch["mode_label"])
            seg_logits = outputs["seg_logits"]
            seg_logits = torch.nn.functional.interpolate(
                seg_logits,
                size=batch["seg_masks"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            seg_probs = torch.sigmoid(seg_logits).view(batch["images"].shape[0], clip_size, 1, batch["seg_masks"].shape[-2], batch["seg_masks"].shape[-1])
            center_masks = (seg_probs[:, center_index] > 0.5).float().cpu().numpy()

            for sample_index, video_id in enumerate(batch_cpu["video_id"]):
                center_name = batch_cpu["center_frame"][sample_index].replace(".jpg", ".png")
                video_dir = os.path.join(output_dir, str(video_id))
                os.makedirs(video_dir, exist_ok=True)
                mask = np.uint8(center_masks[sample_index, 0] * 255)
                cv2.imwrite(os.path.join(video_dir, center_name), mask)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_dinov3", help="model backbone")
    parser.add_argument("--pretrained", type=str, default=None, help="optional local pretrained backbone path")
    parser.add_argument("--resume", type=str, required=True, help="checkpoint directory or checkpoint file")
    parser.add_argument("--clip_size", type=int, default=3, help="clip size")
    parser.add_argument("--train_size", type=int, default=352, help="image size")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
    parser.add_argument("--data_root", type=str, required=True, help="dataset root")
    parser.add_argument("--task", type=str, default="both", choices=["quantitative", "qualitative", "both"], help="evaluation task")
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test"], help="evaluation split")
    parser.add_argument("--num_classes", type=int, default=5, help="number of video classes")
    parser.add_argument("--temporal_backend", type=str, default="pytorch", help="temporal backend")
    parser.add_argument("--use_mode_pseudo_labels", action="store_true", help="enable geometry-based pseudo labels")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--limit_eval_batches", type=int, default=0, help="limit eval batches for smoke tests")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backbone_input_config = get_backbone_input_config(args.backbone)
    if is_dinov3_backbone(args.backbone) and args.train_size != backbone_input_config["train_size"]:
        print(f"Adjust train_size from {args.train_size} to {backbone_input_config['train_size']} for {args.backbone}")
        args.train_size = backbone_input_config["train_size"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    model = ASTR(args).to(device)
    checkpoint_path = resolve_checkpoint_path(args.resume)
    if checkpoint_path is None:
        raise FileNotFoundError(f"Could not resolve checkpoint from {args.resume}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict({key.replace("module.", ""): value for key, value in checkpoint.items()}, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")

    dataset = TestDataset(
        args.data_root,
        args.train_size,
        args.clip_size,
        split=args.eval_split,
        image_mean=backbone_input_config["mean"],
        image_std=backbone_input_config["std"],
        image_interpolation=backbone_input_config["interpolation"],
        use_mode_pseudo_labels=args.use_mode_pseudo_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_video_batch,
    )

    output_path = os.path.dirname(checkpoint_path)
    if args.task in {"quantitative", "both"}:
        metrics = evaluate(model, loader, device, args.num_classes, limit_batches=args.limit_eval_batches)
        message = (
            f"Split={args.eval_split} "
            f"MAE={metrics['mae']:.4f} IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f} "
            f"Sen={metrics['sen']:.4f} Spe={metrics['spe']:.4f} Acc={metrics['acc']:.4f} "
            f"ClsAcc={metrics['cls_acc']:.4f} F1={metrics['cls_f1']:.4f} AUC={metrics['cls_auc']:.4f} "
            f"mu_cs={metrics['mu_cs']:.4f} kappa_c={metrics['kappa_c']:.4f}"
        )
        print(message)
        with open(os.path.join(output_path, f"quantitative_{args.eval_split}.txt"), "w", encoding="utf-8") as handle:
            handle.write(message)

    if args.task in {"qualitative", "both"}:
        qualitative_eval(model, loader, device, output_path, args.clip_size, limit_batches=args.limit_eval_batches)
