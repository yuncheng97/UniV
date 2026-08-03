import argparse
import logging
import os
import random
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from tabulate import tabulate
from torch import amp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from dataset import TestDataset, TrainDataset
from losses import UniVLoss
from model import UniV, get_backbone_input_config, is_dinov3_backbone
from utils import ModelEma, clip_gradient, get_rank, init_distributed_mode


def collate_video_batch(batch):
    collated = {}
    tensor_keys = {"images", "seg_masks", "heatmap_targets", "video_label", "mode_label"}
    meta_keys = {"video_id", "frame_ids", "center_frame"}
    for key in batch[0]:
        if key in tensor_keys:
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
        elif key in meta_keys:
            collated[key] = [sample[key] for sample in batch]
        else:
            collated[key] = [sample[key] for sample in batch]
    return collated


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def resolve_checkpoint_path(resume_arg):
    if resume_arg is None:
        return None
    if os.path.isdir(resume_arg):
        preferred = [
            os.path.join(resume_arg, "best_mu_cs.pth"),
            os.path.join(resume_arg, "best_dice.pth"),
            os.path.join(resume_arg, "best_cls_acc.pth"),
            os.path.join(resume_arg, "epoch_bestDice.pth"),
        ]
        for path in preferred:
            if os.path.exists(path):
                return path
        return None
    if os.path.exists(resume_arg):
        return resume_arg
    return None


def compute_segmentation_metrics(logits, targets):
    logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    target_long = (targets > 0.5).long()

    tp = (preds * target_long).sum().item()
    fp = (preds * (1 - target_long)).sum().item()
    fn = ((1 - preds) * target_long).sum().item()
    tn = ((1 - preds) * (1 - target_long)).sum().item()
    mae = torch.abs(probs - targets).mean().item()
    return tp, fp, fn, tn, mae


def summarize_segmentation(tp, fp, fn, tn, mae_sum, sample_count):
    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    sen = tp / (tp + fn + eps)
    spe = tn / (tn + fp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    mae = mae_sum / max(sample_count, 1)
    return {
        "mae": mae,
        "iou": iou,
        "dice": dice,
        "sen": sen,
        "spe": spe,
        "acc": acc,
    }


def summarize_classification(video_predictions, num_classes):
    labels = []
    probs = []
    seg_probs = []
    for _, payload in sorted(video_predictions.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0]):
        labels.append(payload["label"])
        probs.append(payload["cls_prob_sum"] / payload["count"])
        seg_probs.append(payload["seg_prob_sum"] / payload["count"])

    if not probs:
        zero_metrics = {"cls_acc": 0.0, "cls_f1": 0.0, "cls_auc": 0.0, "kappa_c": 0.0}
        return zero_metrics

    probs = np.stack(probs, axis=0)
    seg_probs = np.stack(seg_probs, axis=0)
    labels = np.asarray(labels)
    preds = probs.argmax(axis=1)

    cls_acc = float((labels == preds).mean())
    cls_f1 = macro_f1_score(labels, preds, num_classes)
    cls_auc = multiclass_ovr_auc(labels, probs, num_classes)
    kappa_c = float(1.0 - np.mean(np.abs(probs - seg_probs)))
    return {
        "cls_acc": cls_acc,
        "cls_f1": cls_f1,
        "cls_auc": cls_auc,
        "kappa_c": kappa_c,
    }


def macro_f1_score(labels, preds, num_classes):
    f1_scores = []
    for class_index in range(num_classes):
        tp = np.sum((labels == class_index) & (preds == class_index))
        fp = np.sum((labels != class_index) & (preds == class_index))
        fn = np.sum((labels == class_index) & (preds != class_index))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def binary_auc(labels, scores):
    positive = labels == 1
    negative = labels == 0
    pos_count = positive.sum()
    neg_count = negative.sum()
    if pos_count == 0 or neg_count == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks = ranks[positive]
    auc = (pos_ranks.sum() - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def multiclass_ovr_auc(labels, probs, num_classes):
    aucs = []
    for class_index in range(num_classes):
        binary_labels = (labels == class_index).astype(np.int64)
        auc = binary_auc(binary_labels, probs[:, class_index])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def evaluate(model, loader, device, num_classes, limit_batches=0):
    model.eval()
    tp = fp = fn = tn = 0
    mae_sum = 0.0
    frame_count = 0
    video_predictions = {}

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if limit_batches and step >= limit_batches:
                break
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], batch["mode_label"])

            seg_logits = outputs["seg_logits"]
            prior_logits = outputs["prior_logits"]
            batch_tp, batch_fp, batch_fn, batch_tn, batch_mae = compute_segmentation_metrics(
                seg_logits,
                batch["seg_masks"].flatten(0, 1),
            )
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            tn += batch_tn
            mae_sum += batch_mae
            frame_count += 1

            cls_probs = torch.softmax(outputs["video_logits"], dim=-1).cpu().numpy()
            seg_probs = torch.softmax(outputs["features_for_loss"]["seg_video_logits"], dim=-1).cpu().numpy()
            labels = batch["video_label"].cpu().numpy()
            for sample_index, video_id in enumerate(batch["video_id"]):
                payload = video_predictions.setdefault(video_id, {
                    "label": int(labels[sample_index]),
                    "cls_prob_sum": np.zeros(num_classes, dtype=np.float64),
                    "seg_prob_sum": np.zeros(num_classes, dtype=np.float64),
                    "count": 0,
                })
                payload["cls_prob_sum"] += cls_probs[sample_index]
                payload["seg_prob_sum"] += seg_probs[sample_index]
                payload["count"] += 1

    seg_metrics = summarize_segmentation(tp, fp, fn, tn, mae_sum, frame_count)
    cls_metrics = summarize_classification(video_predictions, num_classes)
    metrics = {**seg_metrics, **cls_metrics}
    metrics["mu_cs"] = 0.5 * metrics["dice"] + 0.5 * metrics["cls_acc"]
    return metrics


def format_metrics_table(metrics, epoch, title="EVAL"):
    seg_headers = ["Metric", "MAE", "IoU", "Dice", "Sen", "Spe", "Acc"]
    seg_values = [
        "Seg",
        f"{metrics['mae']:.4f}",
        f"{metrics['iou']:.4f}",
        f"{metrics['dice']:.4f}",
        f"{metrics['sen']:.4f}",
        f"{metrics['spe']:.4f}",
        f"{metrics['acc']:.4f}",
    ]
    cls_headers = ["Metric", "cls_acc", "cls_f1", "cls_auc", "kappa_c"]
    cls_values = [
        "Cls",
        f"{metrics['cls_acc']:.4f}",
        f"{metrics['cls_f1']:.4f}",
        f"{metrics['cls_auc']:.4f}",
        f"{metrics['kappa_c']:.4f}",
    ]
    combined_headers = ["Metric", "mu_cs"]
    combined_values = ["Comb", f"{metrics['mu_cs']:.4f}"]

    lines = [f"=== {title} Epoch [{epoch:03d}] ==="]
    lines.append(tabulate([seg_values], headers=seg_headers, tablefmt="simple", numalign="right", stralign="right"))
    lines.append(tabulate([cls_values], headers=cls_headers, tablefmt="simple", numalign="right", stralign="right"))
    lines.append(tabulate([combined_values], headers=combined_headers, tablefmt="simple", numalign="right", stralign="right"))
    return "\n".join(lines)


def update_best_records(best_records, metrics, epoch):
    tracked = [
        ("mu_cs", "max"),
        ("dice", "max"),
        ("mae", "min"),
        ("iou", "max"),
        ("cls_acc", "max"),
        ("cls_f1", "max"),
        ("cls_auc", "max"),
    ]
    for key, direction in tracked:
        if key not in metrics:
            continue
        current = best_records[key]["value"]
        new_val = metrics[key]
        if direction == "max" and new_val > current:
            best_records[key]["value"] = new_val
            best_records[key]["epoch"] = epoch
        elif direction == "min" and new_val < current:
            best_records[key]["value"] = new_val
            best_records[key]["epoch"] = epoch


def make_best_table(best_records):
    rows = []
    for key, rec in best_records.items():
        if rec["epoch"] is not None:
            rows.append([key, f"{rec['value']:.4f}", str(rec["epoch"])])
    if not rows:
        return ""
    return tabulate(rows, headers=["Metric", "Best Value", "Epoch"], tablefmt="simple", stralign="right")


def log_metrics(writer, metrics, prefix, step):
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, global_step=step)


def train_one_epoch(args, train_loader, model, criterion, optimizer, scaler, device, epoch, writer, ema=None):
    model.train()
    running = {}
    steps = 0
    autocast_enabled = scaler is not None and scaler.is_enabled() and device.type == "cuda"

    for step, batch in enumerate(train_loader):
        if args.limit_train_batches and step >= args.limit_train_batches:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast("cuda", enabled=autocast_enabled):
            outputs = model(batch["images"], batch["mode_label"])
            loss_dict = criterion(outputs, batch)
            total_loss = loss_dict["total_loss"]

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(optimizer, args.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        steps += 1
        for key, value in loss_dict.items():
            running[key] = running.get(key, 0.0) + float(value.detach().item())

        global_step = (epoch - 1) * max(len(train_loader), 1) + step + 1
        steps_per_epoch = len(train_loader)
        if step == 0 or (step + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            message = (
                f"TRAIN Epoch [{epoch:03d}] Step [{step + 1:04d}/{steps_per_epoch:04d}] "
                f"Lr: {lr:.5e} "
                f"loss={loss_dict['total_loss'].item():.4f} "
                f"seg={loss_dict['seg_loss'].item():.4f} "
                f"cls={loss_dict['cls_loss'].item():.4f} "
                f"mode={loss_dict['mode_loss'].item():.4f}"
            )
            print(message)
            logging.info(message)
        if writer is not None:
            writer.add_scalar("train/iter_total_loss", loss_dict["total_loss"].item(), global_step=global_step)

    averaged = {key: value / max(steps, 1) for key, value in running.items()}
    log_metrics(writer, averaged, "train_epoch", epoch)
    return averaged


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def save_checkpoint(model, save_dir, filename):
    state_dict = unwrap_model(model).state_dict()
    torch.save(state_dict, os.path.join(save_dir, filename))


def create_dataloaders(args, backbone_input_config):
    train_set = TrainDataset(
        args.data_root,
        args.train_size,
        args.clip_size,
        augment=args.augmentation,
        image_mean=backbone_input_config["mean"],
        image_std=backbone_input_config["std"],
        image_interpolation=backbone_input_config["interpolation"],
        use_mode_pseudo_labels=args.use_mode_pseudo_labels,
    )
    val_set = TestDataset(
        args.data_root,
        args.train_size,
        args.clip_size,
        split=args.eval_split,
        image_mean=backbone_input_config["mean"],
        image_std=backbone_input_config["std"],
        image_interpolation=backbone_input_config["interpolation"],
        use_mode_pseudo_labels=args.use_mode_pseudo_labels,
    )

    if args.distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)
    else:
        train_sampler = None
    val_sampler = None

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_video_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batchsize,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_video_batch,
    )
    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=20, help="epoch number")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_dinov3", help="model backbone")
    parser.add_argument("--pretrained", type=str, default=None, help="local pretrained checkpoint path for the selected backbone")
    parser.add_argument("--resume", type=str, default=None, help="resume model checkpoint path")
    parser.add_argument("--batchsize", type=int, default=4, help="training batch size")
    parser.add_argument("--clip_size", type=int, default=3, help="a clip size")
    parser.add_argument("--train_size", type=int, default=352, help="training dataset size")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer for training")
    parser.add_argument("--scheduler", type=str, default="cos", help="scheduler for training")
    parser.add_argument("--gpu_id", type=str, default="0", help="train use gpu")
    parser.add_argument("--data_root", type=str, required=True, help="the training images root")
    parser.add_argument("--save_path", type=str, default="./results", help="the path to save models and logs")
    parser.add_argument("--note", type=str, default="univ", help="the experiment note")
    parser.add_argument("--augmentation", action="store_true", help="whether use dataset augmentation")
    parser.add_argument("--mp_train", action="store_true", help="whether use mixed precision training")
    parser.add_argument("--model_ema", action="store_true", help="whether use model ema training")
    parser.add_argument("--distributed", action="store_true", help="use distribution training")
    parser.add_argument("--world_size", type=int, default=1, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url to set up distributed training")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--num_classes", type=int, default=5, help="number of video-level classes")
    parser.add_argument("--temporal_backend", type=str, default="pytorch", help="temporal backend implementation")
    parser.add_argument("--use_mode_pseudo_labels", action="store_true", help="enable geometry-based pseudo scan-mode labels")
    parser.add_argument("--famh_alpha", type=float, default=0.5, help="FAMH style-mix alpha")
    parser.add_argument("--lesion_gamma", type=float, default=1.0, help="lesion-aware gamma")
    parser.add_argument("--lata_temperature", type=float, default=1.0, help="LATA softmax temperature")
    parser.add_argument("--ctsi_gate_dim", type=int, default=128, help="CTSI gating dimension")
    parser.add_argument("--ssm_d_state", type=int, default=16, help="Mamba SSM state dimension")
    parser.add_argument("--mode_contrastive_temperature", type=float, default=0.2, help="mode contrastive temperature")
    parser.add_argument("--mode_contrastive_weight", type=float, default=0.05, help="mode contrastive loss weight")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="prior loss weight")
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test"], help="evaluation split used during training")
    parser.add_argument("--limit_train_batches", type=int, default=0, help="limit train batches per epoch for smoke tests")
    parser.add_argument("--limit_val_batches", type=int, default=0, help="limit validation batches for smoke tests")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--ssm", action="store_true", help="use mamba-ssm native implementation for spatial scan")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.system("")

    backbone_input_config = get_backbone_input_config(args.backbone)
    if is_dinov3_backbone(args.backbone) and args.train_size != backbone_input_config["train_size"]:
        print(f"Adjust train_size from {args.train_size} to {backbone_input_config['train_size']} for {args.backbone}")
        args.train_size = backbone_input_config["train_size"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.distributed:
        init_distributed_mode(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    model = UniV(args).to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu] if device.type == "cuda" else None, find_unused_parameters=True)

    ema = ModelEma(unwrap_model(model), decay=0.9998) if args.model_ema else None
    scaler = amp.GradScaler("cuda", enabled=args.mp_train and device.type == "cuda")

    checkpoint_path = resolve_checkpoint_path(args.resume)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = model.module if hasattr(model, "module") else model
        model_state.load_state_dict({key.replace("module.", ""): value for key, value in checkpoint.items()}, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    criterion = UniVLoss(
        num_classes=args.num_classes,
        mode_contrastive_weight=args.mode_contrastive_weight,
        prior_loss_weight=args.prior_loss_weight,
        contrastive_temperature=args.mode_contrastive_temperature,
    ).to(device)
    parameter_groups = list(model.parameters()) + list(criterion.parameters())
    optimizer_dict = {
        "adam": torch.optim.Adam(parameter_groups, lr=args.lr, weight_decay=5e-4),
        "adamw": torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=5e-4),
        "sgd": torch.optim.SGD(parameter_groups, lr=args.lr, momentum=0.9, weight_decay=5e-4),
    }
    optimizer = optimizer_dict[args.optimizer]

    scheduler_dict = {
        "step": torch.optim.lr_scheduler.MultiStepLR(optimizer, [max(args.epoch // 2, 1)], gamma=0.1, last_epoch=-1),
        "cos": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epoch, 1), eta_min=1e-6),
        "exp": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, last_epoch=-1),
    }
    scheduler = scheduler_dict[args.scheduler]
    warmup_epochs = max(args.epoch // 8, 1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    if args.distributed and dist.is_initialized():
        timestamp_payload = [timestamp if get_rank() == 0 else None]
        dist.broadcast_object_list(timestamp_payload, src=0)
        timestamp = timestamp_payload[0]

    save_root = os.path.join(args.save_path, args.note)
    exp_path = os.path.join(save_root, f"log_{timestamp}")
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)

    if get_rank() == 0:
        logging.basicConfig(
            filename=os.path.join(exp_path, "log.log"),
            format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
            level=logging.INFO,
            filemode="a",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        writer = SummaryWriter(os.path.join(exp_path, "summary"))
    else:
        writer = None

    train_loader, val_loader = create_dataloaders(args, backbone_input_config)

    print("=== UniV training config ===")
    print(f"backbone: {args.backbone}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"eval samples ({args.eval_split}): {len(val_loader.dataset)}")
    print(f"train size: {args.train_size}")
    print(f"clip size: {args.clip_size}")
    print(f"device: {device}")
    print(f"seed: {seed}")
    print(f"save path: {exp_path}")
    logging.info("=== UniV training config ===")
    logging.info(f"args: {args}")
    if get_rank() == 0:
        with open(os.path.join(exp_path, "run_config.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)

    best_scores = {"mu_cs": -1.0, "dice": -1.0, "cls_acc": -1.0}
    best_records = {
        "mu_cs": {"epoch": None, "value": -1.0},
        "dice": {"epoch": None, "value": -1.0},
        "mae": {"epoch": None, "value": float("inf")},
        "iou": {"epoch": None, "value": -1.0},
        "cls_acc": {"epoch": None, "value": -1.0},
        "cls_f1": {"epoch": None, "value": -1.0},
        "cls_auc": {"epoch": None, "value": -1.0},
    }
    start_time = time.time()
    for epoch in range(1, args.epoch + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_losses = train_one_epoch(args, train_loader, model, criterion, optimizer, scaler, device, epoch, writer, ema=ema)
        scheduler.step()

        if args.distributed and dist.is_initialized():
            dist.barrier()

        if get_rank() == 0:
            eval_model = unwrap_model(ema.module if ema is not None else model)
            metrics = evaluate(eval_model, val_loader, device, args.num_classes, limit_batches=args.limit_val_batches)
            log_metrics(writer, metrics, f"{args.eval_split}_metrics", epoch)

            update_best_records(best_records, metrics, epoch)

            table = format_metrics_table(metrics, epoch)
            print(table)
            logging.info(table)
            logging.info(f"train_losses: {train_losses}")

            if metrics["mu_cs"] > best_scores["mu_cs"]:
                best_scores["mu_cs"] = metrics["mu_cs"]
                save_checkpoint(eval_model, exp_path, "best_mu_cs.pth")
            if metrics["dice"] > best_scores["dice"]:
                best_scores["dice"] = metrics["dice"]
                save_checkpoint(eval_model, exp_path, "best_dice.pth")
            if metrics["cls_acc"] > best_scores["cls_acc"]:
                best_scores["cls_acc"] = metrics["cls_acc"]
                save_checkpoint(eval_model, exp_path, "best_cls_acc.pth")

        if args.distributed and dist.is_initialized():
            dist.barrier()

    elapsed = time.time() - start_time
    if get_rank() == 0:
        print(f"Training completed in {elapsed:.1f}s")
        logging.info(f"Training completed in {elapsed:.1f}s")
        best_table = make_best_table(best_records)
        header = "\n=== Best Metrics Summary ==="
        print(header + "\n" + best_table)
        logging.info(header + "\n" + best_table)
        if writer is not None:
            writer.close()

    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()
