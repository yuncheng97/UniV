import csv
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

import transforms_clip


def _sorted_video_names(root: str) -> List[str]:
    return sorted(os.listdir(root), key=lambda name: int(name) if name.isdigit() else name)


def _read_video_labels(label_path: str) -> Dict[str, int]:
    labels = {}
    with open(label_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            labels[str(row["video_id"])] = int(row["video_label"])
    return labels


class VideoDataset(data.Dataset):
    def __init__(
        self,
        root,
        split,
        image_size,
        clip_len=3,
        augment=False,
        image_mean=None,
        image_std=None,
        image_interpolation="bilinear",
        use_mode_pseudo_labels=True,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.clip_len = clip_len
        self.augment = augment and split == "train"
        self.use_mode_pseudo_labels = use_mode_pseudo_labels
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]
        self.image_interpolation = image_interpolation

        self.video_labels = _read_video_labels(os.path.join(root, "video_labels.txt"))
        self.samples = self._build_samples()
        self.mode_labels = self._build_mode_labels() if use_mode_pseudo_labels else {video_id: 0 for video_id in self.video_labels}
        self.videos = sorted({sample["video_id"] for sample in self.samples}, key=lambda name: int(name) if str(name).isdigit() else name)
        self.size = len(self.samples)

        if split == "train":
            self.transform = transforms_clip.Compose([
                transforms_clip.RandomVerticalFlip(),
                transforms_clip.RandomHorizontalFlip(),
                transforms_clip.Resize(self.image_size, image_interpolation=self.image_interpolation),
                transforms_clip.ToTensor(),
                transforms_clip.Normalize(self.image_mean, self.image_std),
            ])
        else:
            interpolation = getattr(InterpolationMode, self.image_interpolation.upper())
            self.image_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), interpolation=interpolation),
                transforms.ToTensor(),
                transforms.Normalize(self.image_mean, self.image_std),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])

    def _build_samples(self):
        samples = []
        image_roots = [os.path.join(self.root, self.split, "image")]
        if self.augment:
            aug_root = os.path.join(self.root, "train_aug", "image")
            if os.path.isdir(aug_root):
                image_roots.append(aug_root)

        for image_root in image_roots:
            if not os.path.isdir(image_root):
                continue
            for video_id in _sorted_video_names(image_root):
                video_dir = os.path.join(image_root, video_id)
                frame_names = sorted(os.listdir(video_dir))
                if not frame_names:
                    continue
                for center_idx in range(len(frame_names)):
                    clip_frame_names = self._build_clip_frame_names(frame_names, center_idx)
                    image_paths = [os.path.join(video_dir, frame_name) for frame_name in clip_frame_names]
                    mask_paths = [path.replace(os.sep + "image" + os.sep, os.sep + "mask" + os.sep) for path in image_paths]
                    samples.append({
                        "video_id": video_id,
                        "frame_ids": clip_frame_names,
                        "image_paths": image_paths,
                        "mask_paths": mask_paths,
                        "center_frame": clip_frame_names[len(clip_frame_names) // 2],
                    })
        return samples

    def _build_clip_frame_names(self, frame_names, center_idx):
        half = self.clip_len // 2
        clip = []
        for offset in range(-half, self.clip_len - half):
            index = min(max(center_idx + offset, 0), len(frame_names) - 1)
            clip.append(frame_names[index])
        return clip

    def _build_mode_labels(self):
        scores = {}
        split_roots = [os.path.join(self.root, split, "image") for split in ("train", "val", "test")]
        for split_root in split_roots:
            if not os.path.isdir(split_root):
                continue
            for video_id in _sorted_video_names(split_root):
                if video_id in scores:
                    continue
                frame_names = sorted(os.listdir(os.path.join(split_root, video_id)))
                if not frame_names:
                    scores[video_id] = 0.0
                    continue
                first_frame = os.path.join(split_root, video_id, frame_names[0])
                scores[video_id] = self._geometry_mode_score(first_frame)

        if not scores:
            return {video_id: 0 for video_id in self.video_labels}

        score_values = np.array(list(scores.values()), dtype=np.float32)
        initial_labels = {video_id: int(score > 0.2) for video_id, score in scores.items()}
        unique_labels = set(initial_labels.values())
        if len(unique_labels) == 1:
            threshold = float(np.median(score_values))
            initial_labels = {video_id: int(score >= threshold) for video_id, score in scores.items()}
            unique_labels = set(initial_labels.values())
        if len(unique_labels) == 1:
            ranked = sorted(scores.items(), key=lambda item: item[1])
            split_point = max(1, len(ranked) // 2)
            initial_labels = {}
            for index, (video_id, _) in enumerate(ranked):
                initial_labels[video_id] = int(index >= split_point)

        for video_id in self.video_labels:
            initial_labels.setdefault(video_id, 0)
        return initial_labels

    def _geometry_mode_score(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0.0
        _, thresh = cv2.threshold(image, 8, 255, cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 5)
        points = cv2.findNonZero(thresh)
        if points is None:
            return 0.0
        x, y, w, h = cv2.boundingRect(points)
        roi = thresh[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0
        top_band = roi[: max(1, h // 8)]
        mid_start = max(0, h // 2 - max(1, h // 10))
        mid_end = min(h, h // 2 + max(1, h // 10))
        mid_band = roi[mid_start:mid_end]
        top_width = float(np.count_nonzero(top_band.max(axis=0) > 0))
        mid_width = float(np.count_nonzero(mid_band.max(axis=0) > 0))
        fill_ratio = float(np.count_nonzero(roi)) / float(max(roi.size, 1))
        top_ratio = top_width / max(mid_width, 1.0)
        score = (1.0 - top_ratio) + 0.5 * (1.0 - fill_ratio)
        return float(score)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = [self.rgb_loader(path) for path in sample["image_paths"]]
        masks = [self.binary_loader(path) for path in sample["mask_paths"]]

        if self.split == "train":
            images, masks, aux_maps = self.transform(images, masks, masks)
        else:
            images = [self.image_transform(image) for image in images]
            masks = [self.mask_transform(mask) for mask in masks]
            aux_maps = masks

        image_tensor = torch.stack(images, dim=0)
        mask_tensor = torch.stack(masks, dim=0).float()
        heatmap_tensor = self._build_heatmap_targets(aux_maps if self.split == "train" else masks)

        return {
            "images": image_tensor,
            "seg_masks": mask_tensor,
            "heatmap_targets": heatmap_tensor,
            "video_label": torch.tensor(self.video_labels[sample["video_id"]], dtype=torch.long),
            "mode_label": torch.tensor(self.mode_labels[sample["video_id"]], dtype=torch.long),
            "video_id": sample["video_id"],
            "frame_ids": sample["frame_ids"],
            "center_frame": sample["center_frame"],
        }

    def _build_heatmap_targets(self, masks):
        heatmaps = []
        for mask in masks:
            mask_tensor = mask if torch.is_tensor(mask) else transforms.ToTensor()(mask)
            mask_tensor = mask_tensor.float()
            heatmaps.append(self._mask_to_heatmap(mask_tensor))
        return torch.stack(heatmaps, dim=0)

    def _mask_to_heatmap(self, mask_tensor):
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_binary = (mask_tensor[0] > 0.5)
        if not mask_binary.any():
            return torch.zeros_like(mask_tensor)

        coords = torch.nonzero(mask_binary, as_tuple=False).float()
        cy = coords[:, 0].mean()
        cx = coords[:, 1].mean()
        area = float(coords.shape[0])
        radius = max(2.0, np.sqrt(area / np.pi) * 0.5)

        height, width = mask_binary.shape
        ys = torch.arange(height, dtype=torch.float32).unsqueeze(1)
        xs = torch.arange(width, dtype=torch.float32).unsqueeze(0)
        sigma = max(radius / 2.0, 1.0)
        heatmap = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
        heatmap = heatmap * mask_binary.float()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap.unsqueeze(0)

    def rgb_loader(self, path):
        with open(path, "rb") as handle:
            image = Image.open(handle)
            return image.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as handle:
            image = Image.open(handle)
            return image.convert("L")

    def __len__(self):
        return self.size


class TrainDataset(VideoDataset):
    def __init__(self, root, trainsize, clip_len=3, augment=False, image_mean=None, image_std=None, image_interpolation="bilinear", use_mode_pseudo_labels=True):
        super().__init__(
            root=root,
            split="train",
            image_size=trainsize,
            clip_len=clip_len,
            augment=augment,
            image_mean=image_mean,
            image_std=image_std,
            image_interpolation=image_interpolation,
            use_mode_pseudo_labels=use_mode_pseudo_labels,
        )


class TestDataset(VideoDataset):
    def __init__(self, root, testsize, clip_len=3, split="test", image_mean=None, image_std=None, image_interpolation="bilinear", use_mode_pseudo_labels=True):
        super().__init__(
            root=root,
            split=split,
            image_size=testsize,
            clip_len=clip_len,
            augment=False,
            image_mean=image_mean,
            image_std=image_std,
            image_interpolation=image_interpolation,
            use_mode_pseudo_labels=use_mode_pseudo_labels,
        )
