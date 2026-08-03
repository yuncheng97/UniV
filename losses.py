import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_like(logits, target):
    spatial_size = target.shape[-2:]
    if logits.shape[-2:] != spatial_size:
        logits = F.interpolate(logits, size=spatial_size, mode="bilinear", align_corners=False)
    return logits


def dice_loss_from_logits(logits, target, eps=1.0):
    logits = resize_like(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return dice.mean()


def bce_dice_mae_loss(logits, target):
    logits = resize_like(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    mae = F.l1_loss(torch.sigmoid(logits), target)
    return bce + dice + mae


def symmetric_kl_from_logits(student_logits, teacher_logits):
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    forward_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    backward_kl = F.kl_div(teacher_log_probs, student_probs, reduction="batchmean")
    return 0.5 * (forward_kl + backward_kl)


def cross_mode_contrastive_loss(features, class_labels, mode_labels, temperature=0.2):
    if features.ndim != 2 or features.shape[0] < 2:
        return features.new_tensor(0.0)

    features = F.normalize(features, dim=-1)
    logits = torch.matmul(features, features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    same_class = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)
    diff_mode = mode_labels.unsqueeze(1) != mode_labels.unsqueeze(0)
    positive_mask = same_class & diff_mode
    identity = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = positive_mask & (~identity)
    valid_mask = ~identity

    if positive_mask.sum() == 0:
        return features.new_tensor(0.0)

    exp_logits = torch.exp(logits) * valid_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    positive_log_prob = (positive_mask.float() * log_prob).sum(dim=1)
    positive_count = positive_mask.sum(dim=1).clamp(min=1)
    loss = -(positive_log_prob / positive_count)
    valid_rows = positive_mask.sum(dim=1) > 0
    return loss[valid_rows].mean()


class UniVLoss(nn.Module):
    def __init__(self, num_classes=5, mode_contrastive_weight=0.05, prior_loss_weight=1.0, contrastive_temperature=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.mode_contrastive_weight = mode_contrastive_weight
        self.prior_loss_weight = prior_loss_weight
        self.contrastive_temperature = float(contrastive_temperature)
        self.log_vars = nn.Parameter(torch.zeros(2))

    def set_mode_contrastive_weight(self, weight):
        self.mode_contrastive_weight = weight

    def uncertainty_weighted_loss(self, seg_group, cls_group):
        seg_weight = torch.exp(-self.log_vars[0])
        cls_weight = torch.exp(-self.log_vars[1])
        weighted = 0.5 * seg_weight * seg_group + 0.5 * cls_weight * cls_group
        regularizer = 0.5 * (self.log_vars[0] + self.log_vars[1])
        return weighted + regularizer

    def forward(self, outputs, batch):
        seg_targets = batch["seg_masks"]
        prior_targets = batch["heatmap_targets"]
        video_labels = batch["video_label"]
        mode_labels = batch["mode_label"]

        seg_logits = resize_like(outputs["seg_logits"], seg_targets.flatten(0, 1))
        prior_logits = resize_like(outputs["prior_logits"], prior_targets.flatten(0, 1))
        seg_targets_flat = seg_targets.flatten(0, 1)
        prior_targets_flat = prior_targets.flatten(0, 1)

        seg_loss = bce_dice_mae_loss(seg_logits, seg_targets_flat)
        prior_loss = bce_dice_mae_loss(prior_logits, prior_targets_flat)
        sbc_loss = F.mse_loss(torch.sigmoid(seg_logits), torch.sigmoid(prior_logits))

        video_logits = outputs["video_logits"]
        cls_loss = F.cross_entropy(video_logits, video_labels)

        seg_video_logits = outputs["features_for_loss"]["seg_video_logits"]
        cpc_loss = symmetric_kl_from_logits(seg_video_logits, video_logits)

        mode_logits = outputs["mode_logits"]
        mode_ce = F.cross_entropy(mode_logits, mode_labels)
        phase_embeddings = outputs["features_for_loss"]["phase_embeddings"]
        contrastive = cross_mode_contrastive_loss(
            phase_embeddings,
            video_labels,
            mode_labels,
            temperature=self.contrastive_temperature,
        )
        mode_loss = mode_ce + self.mode_contrastive_weight * contrastive

        seg_group = seg_loss + self.prior_loss_weight * prior_loss + sbc_loss
        cls_group = cls_loss + cpc_loss
        multitask_loss = self.uncertainty_weighted_loss(seg_group, cls_group)
        total_loss = mode_loss + multitask_loss

        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "prior_loss": prior_loss,
            "sbc_loss": sbc_loss,
            "cls_loss": cls_loss,
            "cpc_loss": cpc_loss,
            "mode_loss": mode_loss,
            "mode_ce": mode_ce,
            "mode_contrastive": contrastive,
            "seg_group": seg_group,
            "cls_group": cls_group,
        }
