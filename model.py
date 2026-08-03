import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pvtv2 import pvt_v2_b2
from res2net import Res2Net50

try:
    from mamba_ssm import Mamba as _MambaSSM
    _HAS_MAMBA_SSM = True
except ImportError:
    _MambaSSM = None
    _HAS_MAMBA_SSM = False


TIMM_VIT_BASE_DINOV3 = "vit_base_patch16_dinov3.lvd1689m"
TIMM_DINOV3_ALIASES = {TIMM_VIT_BASE_DINOV3, "vit_base_patch16_dinov3"}


def is_dinov3_backbone(backbone_name):
    return backbone_name in TIMM_DINOV3_ALIASES


def get_backbone_input_config(backbone_name):
    if is_dinov3_backbone(backbone_name):
        return {
            "train_size": 256,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bicubic",
            "timm_model_name": TIMM_VIT_BASE_DINOV3,
        }
    return {
        "train_size": 352,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bilinear",
        "timm_model_name": None,
    }


def init_module(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.apply(init_module)


class TimmDinoV3Backbone(nn.Module):
    def __init__(
        self,
        model_name,
        checkpoint_path=None,
        out_indices=(2, 5, 8, 11),
        out_channels=(128, 256, 512, 768),
        reductions=(4, 8, 16, 32),
    ):
        super().__init__()
        self.reductions = reductions
        self.channels = list(out_channels)
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=checkpoint_path,
            features_only=True,
            out_indices=out_indices,
        )
        in_channels = self.encoder.feature_info.channels()
        self.projections = nn.ModuleList([
            ConvNormAct(in_ch, out_ch, kernel_size=1, padding=0)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])

    def forward(self, x):
        height, width = x.shape[-2:]
        features = self.encoder(x)
        pyramid = []
        for feature, projector, reduction in zip(features, self.projections, self.reductions):
            feature = projector(feature)
            target_size = (max(height // reduction, 1), max(width // reduction, 1))
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            pyramid.append(feature)
        return tuple(pyramid)


def build_backbone(args):
    backbone_name = args.backbone
    pretrained_path = getattr(args, "pretrained", None)
    if backbone_name == "res2net50":
        backbone = Res2Net50()
        channels = [256, 512, 1024, 2048]
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            backbone.load_state_dict(state_dict, strict=False)
    elif backbone_name == "pvt_v2_b2":
        backbone = pvt_v2_b2()
        channels = [64, 128, 320, 512]
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            backbone.load_state_dict(state_dict, strict=False)
    elif is_dinov3_backbone(backbone_name):
        backbone = TimmDinoV3Backbone(
            model_name=get_backbone_input_config(backbone_name)["timm_model_name"],
            checkpoint_path=pretrained_path,
        )
        channels = backbone.channels
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return backbone, channels


class FAMHBlock(nn.Module):
    def __init__(self, channels, embed_dim=128, style_mix_alpha=0.5):
        super().__init__()
        self.style_mix_alpha = float(style_mix_alpha)
        self.mode_classifier = nn.Linear(channels, 2)
        self.alpha_gate = nn.Linear(channels, channels)
        self.style_mean_bank = nn.Parameter(torch.randn(2, channels))
        self.style_std_bank = nn.Parameter(torch.ones(2, channels))
        self.phase_projector = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, embed_dim),
        )
        self.out_norm = nn.GroupNorm(num_groups=8 if channels % 8 == 0 else 1, num_channels=channels)
        self.apply(init_module)

    def forward(self, x):
        b, t, c, h, w = x.shape
        video_descriptor = x.mean(dim=(1, 3, 4))
        mode_logits = self.mode_classifier(video_descriptor)
        mode_probs = torch.softmax(mode_logits, dim=-1)
        alpha = torch.sigmoid(self.alpha_gate(video_descriptor)).view(b, 1, c, 1)

        fft_feature = torch.fft.fft2(x.float(), dim=(-2, -1))
        amplitude = torch.abs(fft_feature).view(b, t, c, -1)
        phase = torch.angle(fft_feature).view(b, t, c, -1)

        amp_mean = amplitude.mean(dim=-1, keepdim=True)
        amp_std = amplitude.std(dim=-1, keepdim=True).clamp(min=1e-6)
        target_mean = torch.matmul(mode_probs, self.style_mean_bank).view(b, 1, c, 1)
        target_std = F.softplus(torch.matmul(mode_probs, self.style_std_bank)).view(b, 1, c, 1) + 1e-4
        normalized_amp = (amplitude - amp_mean) / amp_std
        transformed_amp = normalized_amp * target_std + target_mean
        harmonized_amp = self.style_mix_alpha * (alpha * amplitude + (1.0 - alpha) * transformed_amp)
        harmonized_amp = harmonized_amp + (1.0 - self.style_mix_alpha) * amplitude

        harmonized_fft = torch.polar(harmonized_amp, phase).view(b, t, c, h, w)
        harmonized = torch.fft.ifft2(harmonized_fft, dim=(-2, -1)).real
        harmonized = self.out_norm(harmonized.flatten(0, 1)).view(b, t, c, h, w)

        phase_descriptor = torch.cat([
            torch.cos(phase).mean(dim=(1, 3)),
            torch.sin(phase).mean(dim=(1, 3)),
        ], dim=-1)
        phase_embedding = self.phase_projector(phase_descriptor)
        return harmonized, mode_logits, phase_embedding


class SpatialSelectiveScanner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.delta_proj = nn.Linear(channels, channels)
        self.update_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.apply(init_module)

    def forward(self, tokens, weights, gamma=1.0):
        B, C, L = tokens.shape
        delta_raw = self.delta_proj(tokens.transpose(1, 2)).transpose(1, 2)
        delta = torch.sigmoid(delta_raw) * (1.0 + gamma * weights)
        update_raw = self.update_proj(tokens.transpose(1, 2)).transpose(1, 2)
        update = torch.tanh(update_raw)
        a = 1.0 - delta
        b = delta * update
        state = torch.zeros(B, C, device=tokens.device, dtype=tokens.dtype)
        for index in range(L):
            state = a[:, :, index] * state + b[:, :, index]
        return self.out_proj(state)


class MambaSpatialScanner(nn.Module):
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not _HAS_MAMBA_SSM:
            raise ImportError(
                "mamba-ssm is not installed. Install it with: pip install mamba-ssm causal-conv1d"
            )
        self.input_norm = nn.LayerNorm(channels)
        self.mamba = _MambaSSM(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=True,
        )
        self.apply(init_module)

    def forward(self, tokens, weights, gamma=1.0):
        B, C, L = tokens.shape
        x = tokens.transpose(1, 2)
        x = x * (1.0 + gamma * weights.transpose(1, 2).contiguous())
        x = self.input_norm(x).contiguous()
        out = self.mamba(x)
        return out[:, -1, :]


class TemporalSelectiveScan(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.forward_gate = nn.Linear(channels + 1, channels)
        self.backward_gate = nn.Linear(channels + 1, channels)
        self.out_proj = nn.Linear(channels * 2, channels)
        self.apply(init_module)

    def forward(self, sequence, lesion_scores):
        b, t, c = sequence.shape
        forward_states = []
        backward_states = [None] * t

        state = torch.zeros(b, c, device=sequence.device, dtype=sequence.dtype)
        for step in range(t):
            gate_input = torch.cat([sequence[:, step], lesion_scores[:, step:step + 1]], dim=-1)
            gate = torch.sigmoid(self.forward_gate(gate_input))
            state = gate * state + (1.0 - gate) * sequence[:, step]
            forward_states.append(state)

        state = torch.zeros(b, c, device=sequence.device, dtype=sequence.dtype)
        for offset, step in enumerate(range(t - 1, -1, -1)):
            gate_input = torch.cat([sequence[:, step], lesion_scores[:, step:step + 1]], dim=-1)
            gate = torch.sigmoid(self.backward_gate(gate_input))
            state = gate * state + (1.0 - gate) * sequence[:, step]
            backward_states[step] = state

        forward_tensor = torch.stack(forward_states, dim=1)
        backward_tensor = torch.stack(backward_states, dim=1)
        return self.out_proj(torch.cat([forward_tensor, backward_tensor], dim=-1))


class UATSSMBlock(nn.Module):
    def __init__(self, channels, use_mamba_ssm=False, lesion_gamma=1.0, d_state=16):
        super().__init__()
        self.lesion_gamma = float(lesion_gamma)
        if use_mamba_ssm:
            self.spatial_scan = MambaSpatialScanner(channels, d_state=d_state)
        else:
            self.spatial_scan = SpatialSelectiveScanner(channels)
        self.temporal_scan = TemporalSelectiveScan(channels)
        self.context_gate = nn.Linear(channels, channels)
        self.apply(init_module)
        self._scan_index_cache = {}

    def _scan_indices(self, height, width, mode_value, device):
        key = (height, width, int(mode_value))
        if key in self._scan_index_cache:
            return self._scan_index_cache[key].to(device)
        if int(mode_value) == 0:
            order = torch.arange(height * width, dtype=torch.long)
        else:
            ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            cy = (height - 1) / 2.0
            cx = (width - 1) / 2.0
            radial = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
            angle = torch.atan2(ys.float() - cy, xs.float() - cx)
            order = torch.argsort((radial * 10.0 + angle).reshape(-1))
        self._scan_index_cache[key] = order
        return order.to(device)

    def _spatial_summary(self, x, prior_probs, mode_labels):
        b, t, c, h, w = x.shape
        x_tokens = x.view(b, t, c, h * w)
        weight_tokens = prior_probs.view(b, t, 1, h * w)

        order0 = self._scan_indices(h, w, 0, x.device)
        order1 = self._scan_indices(h, w, 1, x.device)

        group0 = [(bi, si) for bi in range(b) for si in range(t) if int(mode_labels[bi].item()) == 0]
        group1 = [(bi, si) for bi in range(b) for si in range(t) if int(mode_labels[bi].item()) == 1]

        summaries = torch.zeros(b, t, c, device=x.device, dtype=x.dtype)
        lesion_scores = torch.zeros(b, t, device=x.device, dtype=x.dtype)

        for group, order in [(group0, order0), (group1, order1)]:
            if not group:
                continue
            bi_list, si_list = zip(*group)
            batch_tokens = torch.stack([x_tokens[bi, si, :, order] for bi, si in group], dim=0)
            batch_weights = torch.stack([weight_tokens[bi, si, :, order] for bi, si in group], dim=0)
            batch_summaries = self.spatial_scan(batch_tokens, batch_weights, gamma=self.lesion_gamma)
            batch_scores = batch_weights.mean(dim=(-2, -1))
            for idx, (bi, si) in enumerate(group):
                summaries[bi, si] = batch_summaries[idx]
                lesion_scores[bi, si] = batch_scores[idx]

        return summaries, lesion_scores

    def forward(self, x, prior_probs, mode_labels):
        spatial_summary, lesion_scores = self._spatial_summary(x, prior_probs, mode_labels)
        temporal_context = self.temporal_scan(spatial_summary, lesion_scores)
        context_gate = torch.sigmoid(self.context_gate(temporal_context)).unsqueeze(-1).unsqueeze(-1)
        return x + context_gate * x


class LATAAggregator(nn.Module):
    def __init__(self, channels, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.score_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1),
        )
        self.apply(init_module)

    def forward(self, x, prior_probs):
        pooled = (x * prior_probs).mean(dim=(3, 4))
        frame_scores = self.score_head(pooled).squeeze(-1)
        weights = torch.softmax(frame_scores / max(self.temperature, 1e-6), dim=1)
        video_feature = torch.sum(weights.unsqueeze(-1) * pooled, dim=1)
        return video_feature, weights


class UniVDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.fuse_high = ConvNormAct(hidden_channels * 2, hidden_channels)
        self.up_low = ConvNormAct(hidden_channels + hidden_channels, hidden_channels)
        self.refine = ConvNormAct(hidden_channels + hidden_channels, hidden_channels)
        self.seg_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.prior_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.apply(init_module)

    def forward(self, x1, x2, x3, x4):
        x4_up = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        fuse3 = self.fuse_high(torch.cat([x4_up, x3], dim=1))
        prior_logits = self.prior_head(fuse3)

        fuse2 = F.interpolate(fuse3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        fuse2 = self.up_low(torch.cat([fuse2, x2], dim=1))

        fuse1 = F.interpolate(fuse2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        fuse1 = self.refine(torch.cat([fuse1, x1], dim=1))
        seg_logits = self.seg_head(fuse1)
        return seg_logits, prior_logits, fuse3


class UniVModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = getattr(args, "num_classes", 5)
        self.temporal_backend = getattr(args, "temporal_backend", "pytorch")
        self.famh_alpha = float(getattr(args, "famh_alpha", 0.5))
        self.lesion_gamma = float(getattr(args, "lesion_gamma", 1.0))
        self.lata_temperature = float(getattr(args, "lata_temperature", 1.0))
        self.ctsi_gate_dim = int(getattr(args, "ctsi_gate_dim", 128))
        self.ssm_d_state = int(getattr(args, "ssm_d_state", 16))
        self.backbone, channels = build_backbone(args)

        hidden = 128
        self.proj1 = ConvNormAct(channels[0], hidden, kernel_size=1, padding=0)
        self.proj2 = ConvNormAct(channels[1], hidden, kernel_size=1, padding=0)
        self.proj3 = ConvNormAct(channels[2], hidden, kernel_size=1, padding=0)
        self.proj4 = ConvNormAct(channels[3], hidden, kernel_size=1, padding=0)

        self.famh3 = FAMHBlock(hidden, embed_dim=hidden, style_mix_alpha=self.famh_alpha)
        self.famh4 = FAMHBlock(hidden, embed_dim=hidden, style_mix_alpha=self.famh_alpha)
        use_mamba_ssm = getattr(args, "ssm", False)
        self.uatssm3 = UATSSMBlock(hidden, use_mamba_ssm=use_mamba_ssm, lesion_gamma=self.lesion_gamma, d_state=self.ssm_d_state)
        self.uatssm4 = UATSSMBlock(hidden, use_mamba_ssm=use_mamba_ssm, lesion_gamma=self.lesion_gamma, d_state=self.ssm_d_state)
        self.decoder = UniVDecoder(hidden)
        self.lata = LATAAggregator(hidden, temperature=self.lata_temperature)

        self.seg_to_cls_gate = nn.Sequential(
            nn.Linear(hidden, self.ctsi_gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ctsi_gate_dim, hidden),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, self.num_classes),
        )
        self.seg_cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_classes),
        )
        self.cls_to_seg = nn.Sequential(
            nn.Linear(self.num_classes, self.ctsi_gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ctsi_gate_dim, hidden),
        )
        self.phase_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.phase_fusion.apply(init_module)

    def _reshape_feature(self, feature, batch_size, clip_len):
        channels, height, width = feature.shape[1:]
        return feature.view(batch_size, clip_len, channels, height, width)

    def _roi_pool(self, feature, mask_weights):
        weights = mask_weights / (mask_weights.sum(dim=(3, 4), keepdim=True) + 1e-6)
        return torch.sum(feature * weights, dim=(3, 4))

    def forward(self, x, mode_labels=None):
        batch_size, clip_len, channels, height, width = x.shape
        x = x.view(batch_size * clip_len, channels, height, width)
        feat1, feat2, feat3, feat4 = self.backbone(x)

        feat1 = self._reshape_feature(self.proj1(feat1), batch_size, clip_len)
        feat2 = self._reshape_feature(self.proj2(feat2), batch_size, clip_len)
        feat3 = self._reshape_feature(self.proj3(feat3), batch_size, clip_len)
        feat4 = self._reshape_feature(self.proj4(feat4), batch_size, clip_len)

        famh3, mode_logits3, phase3 = self.famh3(feat3)
        famh4, mode_logits4, phase4 = self.famh4(feat4)
        mode_logits = 0.5 * (mode_logits3 + mode_logits4)
        fused_phase = self.phase_fusion(torch.cat([phase3, phase4], dim=-1))

        if mode_labels is None:
            mode_labels = mode_logits.argmax(dim=-1)

        coarse_prior = torch.sigmoid(famh3.mean(dim=2, keepdim=True))
        coarse_prior = coarse_prior / (coarse_prior.amax(dim=(3, 4), keepdim=True) + 1e-6)
        feat3 = self.uatssm3(famh3, coarse_prior, mode_labels)

        prior_for_high = F.interpolate(
            coarse_prior.flatten(0, 1),
            size=famh4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, clip_len, 1, famh4.shape[-2], famh4.shape[-1])
        feat4 = self.uatssm4(famh4, prior_for_high, mode_labels)

        seg_logits, prior_logits, decoder_feature = self.decoder(
            feat1.flatten(0, 1),
            feat2.flatten(0, 1),
            feat3.flatten(0, 1),
            feat4.flatten(0, 1),
        )

        decoder_feature = self._reshape_feature(decoder_feature, batch_size, clip_len)
        prior_probs = torch.sigmoid(prior_logits)
        prior_probs_clip = prior_probs.view(batch_size, clip_len, 1, prior_probs.shape[-2], prior_probs.shape[-1])

        video_feature, lata_weights = self.lata(feat4, prior_for_high)
        roi_context = self._roi_pool(decoder_feature, prior_probs_clip)
        roi_context = torch.sum(lata_weights.unsqueeze(-1) * roi_context, dim=1)
        gated_video_feature = video_feature + self.seg_to_cls_gate(roi_context) * roi_context
        video_logits = self.cls_head(gated_video_feature)
        seg_video_logits = self.seg_cls_head(roi_context)

        cls_gate = torch.sigmoid(self.cls_to_seg(torch.softmax(video_logits, dim=-1))).view(batch_size, 1, -1, 1, 1)
        gated_decoder_feature = decoder_feature * (1.0 + cls_gate)
        gated_fuse3 = gated_decoder_feature.flatten(0, 1)
        fuse2 = F.interpolate(gated_fuse3, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        fuse2 = self.decoder.up_low(torch.cat([fuse2, feat2.flatten(0, 1)], dim=1))
        fuse1 = F.interpolate(fuse2, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
        fuse1 = self.decoder.refine(torch.cat([fuse1, feat1.flatten(0, 1)], dim=1))
        seg_logits = self.decoder.seg_head(fuse1)

        return {
            "seg_logits": seg_logits,
            "prior_logits": prior_logits,
            "video_logits": video_logits,
            "mode_logits": mode_logits,
            "lata_weights": lata_weights,
            "features_for_loss": {
                "phase_embeddings": fused_phase,
                "seg_video_logits": seg_video_logits,
                "roi_context": roi_context,
            },
        }


UniV = UniVModel
