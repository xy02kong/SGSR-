import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer

from model.utils import (
    MS_ConvBlockV2,
    build_ailif,
    build_lif,
)

__all__ = ["SGSR"]


class Spike_Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = layer.Conv2d(nIn, nOut, kernel_size=kSize,
                                 stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias,
                                 step_mode="m")

        if self.bn_acti:
            # self.bn_prelu = BNPReLU(nOut)
            self.bn_lif = BNLIF(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            # output = self.bn_prelu(output)
            output = self.bn_lif(output)

        return output


class BNLIF(nn.Module):
    def __init__(self, nIn, lif=True, spike="lif", tau=2.0, backend="cupy"):
        super().__init__()
        self.bn = layer.BatchNorm2d(nIn, eps=1e-3, step_mode="m")
        self.lif = build_lif(spike=spike, tau=tau, backend=backend)
        self.lif_acti = lif

    def forward(self, input):
        output = self.bn(input)

        if self.lif_acti:
            output = self.lif(output)

        return output


class BasicInterpolate(nn.Module):
    def __init__(self, size, mode, align_corners):
        super(BasicInterpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        times_window, batch_size = x.shape[0], x.shape[1]
        # [t,b,c,h,w,]->[t*b,c,h,w]
        x = x.reshape(-1, *x.shape[2:])
        x = F.interpolate(x, size=self.size, mode=self.mode,
                          align_corners=self.align_corners)
        # [t*b,c,h,w]->[t,b,c,h,w]
        x = x.view(times_window, batch_size, *x.shape[1:])
        return x


class PixelUnshuffleM(nn.Module):
    """PixelUnshuffle for multi-step tensors [T, B, C, H, W]."""

    def __init__(self, downscale_factor=2):
        super().__init__()
        self.downscale_factor = int(downscale_factor)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        x = F.pixel_unshuffle(x, self.downscale_factor)
        _, C2, H2, W2 = x.shape
        return x.reshape(T, B, C2, H2, W2)


class SpikeConvBNNoLIF(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bias=False):
        super().__init__()

        self.conv = layer.Conv2d(
            nIn,
            nOut,
            kernel_size=kSize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            step_mode="m",
        )

        self.bn = layer.BatchNorm2d(
            nOut,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )

    def forward(self, x):
        return self.bn(self.conv(x))


class HalfResolutionLocalBlock(nn.Module):
    """
    1/2尺度局部细节提取块。

    该模块比单纯DW3×3更完整，但明显轻于MS_ConvBlockV2。

    输入输出：
        [T,B,channels,H/2,W/2]
    """

    def __init__(
            self,
            channels=24,
            expansion=1.5,
            tau=2.0,
            threshold=0.5,
            backend="cupy",
            residual_scale_init=0.1,
            use_pre_neuron=True,
            use_adaptive_pre_neuron=False,
            pre_neuron_threshold=None,
            ailif_base_threshold=0.25,
            ailif_threshold_min=0.20,
            ailif_threshold_max=0.35,
            ailif_adaptation_decay=0.90,
            ailif_adaptation_increment=0.05,
            ailif_adaptation_increment_max=0.08,
    ):
        super().__init__()

        hidden_channels = max(
            channels,
            int(round(channels * expansion)),
        )

        if not use_pre_neuron and use_adaptive_pre_neuron:
            raise ValueError(
                "use_adaptive_pre_neuron requires use_pre_neuron=True."
            )

        if not use_pre_neuron:
            # Optional path for callers that already provide spikes.
            self.pre_bn = nn.Identity()
            self.pre_lif = nn.Identity()
        elif use_adaptive_pre_neuron:
            self.pre_bn = layer.BatchNorm2d(
                channels,
                eps=1e-3,
                momentum=0.1,
                step_mode="m",
            )
            # AiLIF is intentionally restricted to the first event encoder.
            self.pre_lif = build_ailif(
                tau=tau,
                base_threshold=ailif_base_threshold,
                threshold_min=ailif_threshold_min,
                threshold_max=ailif_threshold_max,
                adaptation_decay=ailif_adaptation_decay,
                adaptation_increment=ailif_adaptation_increment,
                adaptation_increment_max=ailif_adaptation_increment_max,
            )
        else:
            self.pre_bn = layer.BatchNorm2d(
                channels,
                eps=1e-3,
                momentum=0.1,
                step_mode="m",
            )
            self.pre_lif = build_lif(
                spike="lif",
                tau=tau,
                backend=backend,
                v_threshold=(
                    threshold
                    if pre_neuron_threshold is None
                    else float(pre_neuron_threshold)
                ),
            )

        # 通道扩展。
        self.expand = layer.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            step_mode="m",
        )

        self.expand_bn = layer.BatchNorm2d(
            hidden_channels,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )

        self.spatial_lif = build_lif(
            spike="lif",
            tau=tau,
            backend=backend,
            v_threshold=threshold,
        )

        # 所有隐藏通道进行3×3局部空间提取。
        self.dw = layer.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels,
            bias=False,
            step_mode="m",
        )

        self.dw_bn = layer.BatchNorm2d(
            hidden_channels,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )

        self.project_lif = build_lif(
            spike="lif",
            tau=tau,
            backend=backend,
            v_threshold=threshold,
        )

        self.project = layer.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            step_mode="m",
        )

        self.out_bn = layer.BatchNorm2d(
            channels,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )

        self.residual_scale = nn.Parameter(
            torch.full(
                (1, 1, channels, 1, 1),
                float(residual_scale_init),
            )
        )

    def forward(self, x):
        residual = self.pre_bn(
            x
        )

        residual = self.pre_lif(
            residual
        )

        residual = self.expand(
            residual
        )

        residual = self.expand_bn(
            residual
        )

        residual = self.spatial_lif(
            residual
        )

        residual = self.dw(
            residual
        )

        residual = self.dw_bn(
            residual
        )

        residual = self.project_lif(
            residual
        )

        residual = self.project(
            residual
        )

        residual = self.out_bn(
            residual
        )

        return (
                x
                + self.residual_scale * residual
        )


class DownSamplingBlock(nn.Module):
    # 输出是 membrane potential
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Spike_Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = layer.MaxPool2d(3, stride=2, padding=1, step_mode="m")
        self.bn = layer.BatchNorm2d(nOut, eps=1e-3, step_mode="m")

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)

            if max_pool.shape[-2:] != output.shape[-2:]:
                T, B = max_pool.shape[0], max_pool.shape[1]
                max_pool = max_pool.reshape(T * B, *max_pool.shape[2:])
                max_pool = F.interpolate(
                    max_pool,
                    size=output.shape[-2:],
                    mode='nearest'
                )
                max_pool = max_pool.view(T, B, *max_pool.shape[1:])

            output = torch.cat([output, max_pool], dim=2)

        return self.bn(output)


class ShallowLearnedDownSamplingBlock(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()

        self.proj = SpikeConvBNNoLIF(
            nIn,
            nOut,
            kSize=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.proj(x)


class MultiStepBilinearUpsample(nn.Module):
    """连续膜电位双线性上采样，输入/输出均为 [T, B, C, H, W]。"""

    def forward(self, x, target_size):
        if x.dim() != 5:
            raise ValueError(
                f"Expected [T,B,C,H,W], but got shape={tuple(x.shape)}"
            )

        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return x.reshape(T, B, C, target_size[0], target_size[1])


class MembraneProjection(nn.Module):
    """连续膜电位通道投影；跨尺度融合前不执行 LIF。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = layer.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            step_mode="m",
        )
        self.bn = layer.BatchNorm2d(
            out_ch,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )

    def forward(self, x):
        return self.bn(self.proj(x))


class DynamicSpaceTimeGate(nn.Module):
    """
    内容相关的空间—时间 Skip gate。

    输入和输出均为 [T, B, C, H, W]。时间信息由时序均值和
    相邻时间步差分显式注入，最终 gate 对样本、时间、空间和通道均自适应。
    """

    def __init__(self, channels):
        super().__init__()
        hidden_ch = max(16, channels // 4)

        self.reduce = layer.Conv2d(
            channels * 5,
            hidden_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            step_mode="m",
        )
        self.reduce_bn = layer.BatchNorm2d(
            hidden_ch,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )
        self.dw = layer.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_ch,
            bias=False,
            step_mode="m",
        )
        self.dw_bn = layer.BatchNorm2d(
            hidden_ch,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )
        self.out = layer.Conv2d(
            hidden_ch,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            step_mode="m",
        )

    def reset_parameters(self):
        # 初始 gate 约为 0.5，保持与原静态 gate 的初始融合强度接近。
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, deep_mem, skip_mem):
        if deep_mem.shape != skip_mem.shape:
            raise RuntimeError(
                "Dynamic gate shape mismatch: "
                f"deep={tuple(deep_mem.shape)}, "
                f"skip={tuple(skip_mem.shape)}."
            )

        mixed = 0.5 * (deep_mem + skip_mem)
        temporal_mean = mixed.mean(dim=0, keepdim=True).expand_as(mixed)

        temporal_delta = torch.cat(
            [
                torch.zeros_like(mixed[:1]),
                torch.abs(mixed[1:] - mixed[:-1]),
            ],
            dim=0,
        )

        gate_input = torch.cat(
            [
                deep_mem,
                skip_mem,
                torch.abs(deep_mem - skip_mem),
                temporal_mean,
                temporal_delta,
            ],
            dim=2,
        )

        gate_feature = self.reduce(gate_input)
        gate_feature = F.silu(self.reduce_bn(gate_feature))
        gate_feature = self.dw(gate_feature)
        gate_feature = F.silu(self.dw_bn(gate_feature))

        return torch.sigmoid(self.out(gate_feature))


class DynamicSpikeDecoderBlock(nn.Module):
    """
    1/16、1/8、1/4 解码块。

    结构：
        连续膜电位投影 -> 上采样 -> 动态空间—时间门控融合
        -> BN -> 单次 LIF -> 轻量空间精炼 -> 残差输出。
    """

    def __init__(
            self,
            deep_ch,
            skip_ch,
            out_ch,
            tau=2.0,
            threshold=0.5,
            backend="cupy",
            use_dilated_context=False,
            residual_scale_init=1e-2,
            skip_scale_init=None,
    ):
        super().__init__()

        self.upsample = MultiStepBilinearUpsample()
        self.deep_proj = MembraneProjection(deep_ch, out_ch)
        self.skip_proj = MembraneProjection(skip_ch, out_ch)
        self.gate = DynamicSpaceTimeGate(out_ch)

        # 可选的逐通道 Skip 缩放参数。
        # 使用 sigmoid 将实际缩放值限制在 0～1。
        if skip_scale_init is None:
            self.register_parameter("skip_scale_logit", None)
        else:
            skip_scale_init = float(skip_scale_init)

            if not 0.0 < skip_scale_init < 1.0:
                raise ValueError(
                    "skip_scale_init must lie strictly inside (0, 1), "
                    f"got {skip_scale_init}."
                )

            initial_scale = torch.full(
                (1, 1, out_ch, 1, 1),
                skip_scale_init,
                dtype=torch.float32,
            )

            self.skip_scale_logit = nn.Parameter(
                torch.logit(initial_scale)
            )

        self.fuse_bn = layer.BatchNorm2d(
            out_ch,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )
        self.fuse_lif = build_lif(
            spike="lif",
            tau=tau,
            backend=backend,
            v_threshold=threshold,
        )

        self.local_dw = layer.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_ch,
            bias=False,
            step_mode="m",
        )

        self.context_dw = None
        if use_dilated_context:
            self.context_dw = layer.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                groups=out_ch,
                bias=False,
                step_mode="m",
            )

        self.spatial_bn = layer.BatchNorm2d(
            out_ch,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )
        self.pw = layer.Conv2d(
            out_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            step_mode="m",
        )
        self.out_bn = layer.BatchNorm2d(
            out_ch,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )

        self.residual_scale = nn.Parameter(
            torch.full(
                (1, 1, out_ch, 1, 1),
                float(residual_scale_init),
            )
        )

    @property
    def skip_scale(self):
        """返回经过 sigmoid 约束后的实际 Skip 缩放值。"""
        if self.skip_scale_logit is None:
            return None

        return torch.sigmoid(self.skip_scale_logit)

    def reset_gate_parameters(self):
        self.gate.reset_parameters()

    def forward(self, deep, skip):
        # 深层特征先在低分辨率完成 1x1 投影，再上采样，减少计算量。
        deep_mem = self.deep_proj(deep)
        deep_mem = self.upsample(deep_mem, skip.shape[-2:])
        skip_mem = self.skip_proj(skip)

        gate = self.gate(deep_mem, skip_mem)

        if self.skip_scale_logit is None:
            # Decoder16 和 Decoder4 继续使用原始融合方式。
            fused_mem = deep_mem + gate * skip_mem
        else:
            # Decoder8 使用受约束的可学习 Skip 缩放。
            fused_mem = (
                deep_mem
                + self.skip_scale * gate * skip_mem
            )

        # 融合前两条分支均不做 LIF；只在融合后统一脉冲化一次。
        spike = self.fuse_lif(self.fuse_bn(fused_mem))

        residual = self.local_dw(spike)
        if self.context_dw is not None:
            residual = 0.5 * (residual + self.context_dw(spike))

        residual = self.spatial_bn(residual)
        residual = self.pw(residual)
        residual = self.out_bn(residual)

        return fused_mem + self.residual_scale * residual


class MembraneReadoutHead(nn.Module):
    """Continuous-membrane classifier on decoder4 (1/4 scale).

    Average membranes over time, then BN → 1×1 Conv. Semantic logits are
    bilinear-upsampled to full resolution; boundary logits stay at 1/4.
    """

    def __init__(self, in_ch, classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            in_ch,
            eps=1e-3,
            momentum=0.1,
        )
        self.classifier = nn.Conv2d(
            in_ch,
            classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(
                "MembraneReadoutHead expects [T,B,C,H,W], "
                f"got {tuple(x.shape)}."
            )
        x = x.mean(dim=0)
        return self.classifier(self.bn(x))


class SpikeBottleneck(nn.Module):
    expansion = 2

    def __init__(
            self,
            in_ch,
            planes,
            stride=1,
            downsample=None,
            no_lif=True,
    ):
        super().__init__()

        out_ch = planes * self.expansion

        self.conv1 = Spike_Conv(
            in_ch,
            planes,
            kSize=1,
            stride=1,
            padding=0,
            bn_acti=True,
        )

        self.conv2 = Spike_Conv(
            planes,
            planes,
            kSize=3,
            stride=stride,
            padding=1,
            bn_acti=True,
        )

        self.conv3 = Spike_Conv(
            planes,
            out_ch,
            kSize=1,
            stride=1,
            padding=0,
            bn_acti=False,
        )
        self.bn3 = layer.BatchNorm2d(
            out_ch,
            eps=1e-3,
            momentum=0.1,
            step_mode="m",
        )

        self.downsample = downsample
        self.no_lif = no_lif
        self.lif3 = build_lif(spike="lif", tau=2.0, backend="cupy")

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        if self.no_lif:
            return out

        return self.lif3(out)


class SpikeDAPPM(nn.Module):

    def __init__(self, in_ch=64, branch_ch=16, out_ch=64):
        super().__init__()

        self.scale0 = nn.Sequential(
            BNLIF(in_ch),
            Spike_Conv(
                in_ch, branch_ch,
                kSize=1,
                stride=1,
                padding=0,
                bn_acti=False,
            ),
        )

        self.scale1 = nn.Sequential(
            BNLIF(in_ch),
            Spike_Conv(
                in_ch, branch_ch,
                kSize=1,
                stride=1,
                padding=0,
                bn_acti=False,
            ),
        )

        self.scale2 = nn.Sequential(
            BNLIF(in_ch),
            Spike_Conv(
                in_ch, branch_ch,
                kSize=1,
                stride=1,
                padding=0,
                bn_acti=False,
            ),
        )

        # 全局分支：不做 BNLIF，避免在 [T,B,C,1,1] 上 BN 不稳定
        self.scale3 = Spike_Conv(
            in_ch,
            branch_ch,
            kSize=1,
            stride=1,
            padding=0,
            bn_acti=False,
        )

        self.process1 = nn.Sequential(
            BNLIF(branch_ch),
            Spike_Conv(
                branch_ch, branch_ch,
                kSize=3,
                stride=1,
                padding=1,
                bn_acti=False,
            ),
        )

        self.process2 = nn.Sequential(
            BNLIF(branch_ch),
            Spike_Conv(
                branch_ch, branch_ch,
                kSize=3,
                stride=1,
                padding=1,
                bn_acti=False,
            ),
        )

        self.process3 = nn.Sequential(
            BNLIF(branch_ch),
            Spike_Conv(
                branch_ch, branch_ch,
                kSize=3,
                stride=1,
                padding=1,
                bn_acti=False,
            ),
        )

        self.compression = nn.Sequential(
            BNLIF(branch_ch * 4),
            Spike_Conv(
                branch_ch * 4,
                out_ch,
                kSize=1,
                stride=1,
                padding=0,
                bn_acti=False,
            ),
        )

        self.shortcut = nn.Sequential(
            BNLIF(in_ch),
            Spike_Conv(
                in_ch,
                out_ch,
                kSize=1,
                stride=1,
                padding=0,
                bn_acti=False,
            ),
        )

    @staticmethod
    def temporal_resize(x, size):
        T, B, C, H, W = x.shape

        x = x.reshape(T * B, C, H, W)
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        return x.reshape(
            T, B, C,
            size[0], size[1],
        )

    @staticmethod
    def temporal_pool(x, size):
        T, B, C, H, W = x.shape

        x = x.reshape(T * B, C, H, W)
        x = F.adaptive_avg_pool2d(x, output_size=size)

        return x.reshape(
            T, B, C,
            size[0], size[1],
        )

    def forward(self, x):
        _, _, _, H, W = x.shape
        target_size = (H, W)

        x0 = self.scale0(x)

        pool1 = self.temporal_pool(
            x,
            size=(
                max(1, H // 2),
                max(1, W // 2),
            ),
        )
        x1 = self.scale1(pool1)
        x1 = self.temporal_resize(x1, target_size)
        x1 = self.process1(x1 + x0)

        pool2 = self.temporal_pool(
            x,
            size=(
                max(1, H // 4),
                max(1, W // 4),
            ),
        )
        x2 = self.scale2(pool2)
        x2 = self.temporal_resize(x2, target_size)
        x2 = self.process2(x2 + x1)

        pool3 = self.temporal_pool(x, size=(1, 1))
        x3 = self.scale3(pool3)
        x3 = self.temporal_resize(x3, target_size)
        x3 = self.process3(x3 + x2)

        out = torch.cat(
            [x0, x1, x2, x3],
            dim=2,
        )

        out = self.compression(out)
        out = out + self.shortcut(x)

        return out


class SGSR(nn.Module):
    def __init__(
            self,
            classes=19,
            num_classes=None,
            in_channels=1,
            temporal_steps=3,
            # Recommended topology:
            #   1/8 uses three mixed local/context blocks plus one local block;
            #   1/16 uses six bottleneck blocks.
            block_1=3,
            block_2=4,
            block_3=6,
            decoder_channels=(128, 96, 64),
            decoder_threshold=0.5,
            decoder_tau=2.0,
            decoder_backend="cupy",
            first_neuron_type="ailif",
            first_lif_threshold=0.2,
            ailif_base_threshold=0.25,
            ailif_threshold_min=0.20,
            ailif_threshold_max=0.35,
            ailif_adaptation_decay=0.90,
            ailif_adaptation_increment=0.05,
            ailif_adaptation_increment_max=0.08,
            **kwargs
    ):
        super().__init__()

        if num_classes is not None:
            classes = num_classes

        self.in_channels = in_channels
        self.temporal_steps = temporal_steps

        first_neuron_type = str(first_neuron_type).lower()
        if first_neuron_type not in {"fixed_lif", "ailif"}:
            raise ValueError(
                "first_neuron_type must be 'fixed_lif' or 'ailif', "
                f"got {first_neuron_type!r}."
            )
        if float(first_lif_threshold) <= 0.0:
            raise ValueError(
                "first_lif_threshold must be positive, "
                f"got {first_lif_threshold}."
            )
        self.first_neuron_type = first_neuron_type

        for name, value in (
                ("block_1", block_1),
                ("block_2", block_2),
                ("block_3", block_3),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be at least 1, got {value}.")

        block_1 = int(block_1)
        block_2 = int(block_2)
        block_3 = int(block_3)

        if len(decoder_channels) != 3:
            raise ValueError(
                "decoder_channels must be (d16, d8, d4)."
            )

        d16, d8, d4 = [
            int(v) for v in decoder_channels
        ]

        decoder_threshold = float(
            decoder_threshold
        )

        if decoder_threshold <= 0:
            raise ValueError(
                "decoder_threshold must be positive, "
                f"but got {decoder_threshold}."
            )

        th16 = decoder_threshold
        th8 = decoder_threshold
        th4 = decoder_threshold

        half_ch = 32

        # =====================================================
        # Stem: PixelUnshuffle(2) → 1x1 Conv → BN → first AiLIF/LIF
        # =====================================================

        self.stem_half_proj = nn.Sequential(
            PixelUnshuffleM(2),
            SpikeConvBNNoLIF(
                in_channels * 4,
                half_ch,
                kSize=1,
                stride=1,
                padding=0,
            ),
        )

        self.stem_half_local = nn.Sequential(
            HalfResolutionLocalBlock(
                channels=half_ch,
                expansion=1.5,
                tau=2.0,
                threshold=0.5,
                backend=decoder_backend,
                residual_scale_init=0.1,
                use_pre_neuron=True,
                use_adaptive_pre_neuron=(
                    self.first_neuron_type == "ailif"
                ),
                pre_neuron_threshold=first_lif_threshold,
                ailif_base_threshold=ailif_base_threshold,
                ailif_threshold_min=ailif_threshold_min,
                ailif_threshold_max=ailif_threshold_max,
                ailif_adaptation_decay=ailif_adaptation_decay,
                ailif_adaptation_increment=ailif_adaptation_increment,
                ailif_adaptation_increment_max=(
                    ailif_adaptation_increment_max
                ),
            ),

            HalfResolutionLocalBlock(
                channels=half_ch,
                expansion=1.5,
                tau=2.0,
                threshold=0.5,
                backend=decoder_backend,
                residual_scale_init=0.1,
                use_pre_neuron=True,
                use_adaptive_pre_neuron=False,
            ),
        )

        self.stem_quarter_down = SpikeConvBNNoLIF(
            half_ch,
            64,
            kSize=3,
            stride=2,
            padding=1,
        )

        # 先完成通道映射→空间聚合，再统一进行一次脉冲化。
        self.bn_lif_1 = BNLIF(64)

        # block_1 now controls this stage.  The default value of 3 preserves
        # the original local-context-local topology exactly.
        stage1_blocks = []
        for index in range(block_1):
            is_boundary_block = index == 0 or index == block_1 - 1
            stage1_blocks.append(
                MS_ConvBlockV2(
                    dim=64,
                    expansion=1.5 if is_boundary_block else 2.0,
                    context_ratio=0.0 if is_boundary_block else 0.25,
                    spike="lif",
                    tau=2.0,
                    threshold=0.5,
                    backend=decoder_backend,
                    residual_scale_init=0.1,
                )
            )
        self.DAB_Block_1 = nn.Sequential(*stage1_blocks)

        self.bn_lif_2 = BNLIF(64)

        # DAB Block 2
        # 1/4→1/8 使用 64→128，增强中层语义与细粒度目标表达能力；
        # 全部由可学习的 3×3 stride-2 卷积产生，避免过度依赖最大池化。
        self.downsample_2 = ShallowLearnedDownSamplingBlock(
            nIn=64,
            nOut=128,
        )

        # The recommended 1/8 stage uses the original three mixed
        # local/context blocks. Custom configurations with extra blocks keep
        # the previous lightweight local-only fallback for compatibility.
        stage2_blocks = []
        for index in range(block_2):
            is_added_local_block = index >= 3
            stage2_blocks.append(
                MS_ConvBlockV2(
                    dim=128,
                    expansion=(
                        1.5
                        if is_added_local_block
                        else 2.0
                    ),
                    context_ratio=(
                        0.0
                        if is_added_local_block
                        else 0.25
                    ),
                    spike="lif",
                    tau=2.0,
                    threshold=0.5,
                    backend=decoder_backend,
                    residual_scale_init=(
                        0.05
                        if is_added_local_block
                        else 0.1
                    ),
                )
            )
        self.DAB_Block_2 = nn.Sequential(*stage2_blocks)

        self.bn_lif_3 = BNLIF(128)

        self.downsample_3 = DownSamplingBlock(
            nIn=128,
            nOut=256,
        )

        self.bn_lif_4 = BNLIF(256)

        # Six 1/16 bottlenecks provide the main increase in semantic capacity.
        self.bottleneck_16 = self._make_bottleneck_stage(
            inplanes=256,
            planes=128,
            blocks=block_3,
            stride=1,
        )

        # Keep the 1/16 -> 1/32 signal semantics aligned with 1/8 -> 1/16:
        # DownSamplingBlock always receives spikes, never raw membrane values.
        self.bn_lif_5 = BNLIF(
            256,
            spike="lif",
            tau=2.0,
            backend=decoder_backend,
        )

        self.downsample_4 = DownSamplingBlock(
            nIn=256,
            nOut=512,
        )

        self.dappm = SpikeDAPPM(
            in_ch=512,
            branch_ch=128,
            out_ch=512,
        )

        self.decode16 = DynamicSpikeDecoderBlock(
            deep_ch=512,
            skip_ch=256,
            out_ch=d16,
            tau=decoder_tau,
            threshold=th16,
            backend=decoder_backend,
            use_dilated_context=True,
            residual_scale_init=1e-2,
        )

        self.decode8 = DynamicSpikeDecoderBlock(
            deep_ch=d16,
            skip_ch=128,
            out_ch=d8,
            tau=decoder_tau,
            threshold=th8,
            backend=decoder_backend,
            use_dilated_context=True,
            residual_scale_init=1e-2,
            skip_scale_init=0.3,
        )

        self.decode4 = DynamicSpikeDecoderBlock(
            deep_ch=d8,
            skip_ch=64,
            out_ch=d4,
            tau=decoder_tau,
            threshold=th4,
            backend=decoder_backend,
            use_dilated_context=False,
            residual_scale_init=0.1,
        )

        # 语义与边界均在 decoder4（1/4）连续膜电位上直接分类。
        self.seg_head = MembraneReadoutHead(
            in_ch=d4,
            classes=classes,
        )
        self.boundary_head = MembraneReadoutHead(
            in_ch=d4,
            classes=1,
        )

        self.apply(self.kaiming_init)

        for decoder in (self.decode16, self.decode8, self.decode4):
            decoder.reset_gate_parameters()

        if self.boundary_head.classifier.bias is not None:
            nn.init.zeros_(
                self.boundary_head.classifier.bias
            )

        for m in self.modules():
            if isinstance(m, SpikeBottleneck):
                nn.init.zeros_(m.bn3.weight)

        self._assert_multistep_configuration()

    @staticmethod
    def _make_bottleneck_stage(
            inplanes,
            planes,
            blocks,
            stride=1,
    ):
        if blocks < 1:
            raise ValueError(
                "blocks must be greater than or equal to 1"
            )

        outplanes = (
                planes
                * SpikeBottleneck.expansion
        )

        downsample = None

        if (
                stride != 1
                or inplanes != outplanes
        ):
            downsample = nn.Sequential(
                Spike_Conv(
                    inplanes,
                    outplanes,
                    kSize=1,
                    stride=stride,
                    padding=0,
                    bn_acti=False,
                ),

                layer.BatchNorm2d(
                    outplanes,
                    eps=1e-3,
                    momentum=0.1,
                    step_mode="m",
                ),
            )

        layers = []

        # -------------------------------------------------
        # 第一块
        #
        # 保留：
        #   conv1后的LIF
        #   conv2后的LIF
        #
        # 删除：
        #   残差相加后的lif3
        # -------------------------------------------------
        layers.append(
            SpikeBottleneck(
                in_ch=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,

                # 所有1/16 Bottleneck均不在残差后
                # 额外进行一次脉冲化。
                no_lif=True,
            )
        )

        # -------------------------------------------------
        # 后续块同样只保留两个卷积路径中的LIF
        # -------------------------------------------------
        for _ in range(1, blocks):
            layers.append(
                SpikeBottleneck(
                    in_ch=outplanes,
                    planes=planes,
                    stride=1,
                    downsample=None,

                    no_lif=True,
                )
            )

        return nn.Sequential(
            *layers
        )

    def _assert_multistep_configuration(self):
        invalid = []

        for name, module in self.named_modules():
            if hasattr(module, "step_mode") and module.step_mode != "m":
                invalid.append(
                    f"{name or '<root>'}: "
                    f"{module.__class__.__name__}(step_mode={module.step_mode!r})"
                )

        if invalid:
            details = "\n  ".join(invalid)
            raise RuntimeError(
                "SGSR is a multi-step model, but some step-aware modules "
                f"are not configured with step_mode='m':\n  {details}"
            )

    def _format_input(self, x):
        if x.dim() == 4:
            B, TC, H, W = x.shape
            expected_tc = self.temporal_steps * self.in_channels

            if TC != expected_tc:
                raise ValueError(
                    f"Expected channel dimension T*C={expected_tc}, "
                    f"but got {TC}. shape={tuple(x.shape)}"
                )

            x = x.view(
                B,
                self.temporal_steps,
                self.in_channels,
                H,
                W,
            )

        if x.dim() != 5:
            raise ValueError(f"Expected 5D input, got {tuple(x.shape)}")

        if x.shape[1] == self.temporal_steps and x.shape[2] == self.in_channels:
            x = x.permute(1, 0, 2, 3, 4).contiguous()
        elif x.shape[0] == self.temporal_steps and x.shape[2] == self.in_channels:
            x = x.contiguous()
        else:
            raise ValueError(
                f"Expected [B,T,C,H,W] or [T,B,C,H,W] with "
                f"T={self.temporal_steps}, C={self.in_channels}, "
                f"but got shape={tuple(x.shape)}"
            )

        return x

    def trunc_init(self, m):
        conv_types = (
            nn.Conv1d,
            nn.Conv2d,
            nn.ConvTranspose2d,
            layer.Conv1d,
            layer.Conv2d,
            layer.ConvTranspose2d,
        )
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            layer.BatchNorm1d,
            layer.BatchNorm2d,
        )

        if isinstance(m, conv_types):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, bn_types):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, layer.Conv1d, layer.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, layer.BatchNorm1d, layer.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        input = self._format_input(
            input
        )

        input_hw = input.shape[-2:]

        # =====================================================
        # 1/2 high-resolution stem
        # =====================================================

        stem_half = self.stem_half_proj(
            input
        )

        stem_half_mem = self.stem_half_local(
            stem_half
        )

        # =====================================================
        # 1/2 -> 1/4 learned downsampling
        # =====================================================

        output0_mem = self.stem_quarter_down(
            stem_half_mem
        )

        output0 = self.bn_lif_1(
            output0_mem
        )

        # =====================================================
        # Encoder
        # =====================================================

        output1_mem = self.DAB_Block_1(
            output0
        )

        output1 = self.bn_lif_2(
            output1_mem
        )

        output2_0 = self.downsample_2(
            output1
        )

        output2_mem = self.DAB_Block_2(
            output2_0
        )

        output2 = self.bn_lif_3(
            output2_mem
        )

        output3 = self.downsample_3(
            output2
        )

        output3 = self.bn_lif_4(
            output3
        )

        output16 = self.bottleneck_16(
            output3
        )

        output16_spike = self.bn_lif_5(
            output16
        )

        output32 = self.downsample_4(
            output16_spike
        )

        output32 = self.dappm(
            output32
        )

        # =====================================================
        # Decoder
        # =====================================================

        decoder16 = self.decode16(
            output32,
            output16,
        )

        decoder8 = self.decode8(
            decoder16,
            output2_mem,
        )

        decoder4 = self.decode4(
            decoder8,
            output1_mem,
        )

        # =====================================================
        # Semantic + boundary prediction at decoder4 (1/4)
        # =====================================================

        # 原生1/4语义logits。
        # Semantic Boundary Alignment必须作用在这里，
        # 避免全分辨率双线性插值天然平滑相邻像素。
        seg_logits_quarter = self.seg_head(
            decoder4
        )

        # 独立1/4边界头。
        boundary_logits = self.boundary_head(
            decoder4
        )

        # 仅主语义CE和Dice使用全分辨率输出。
        seg_logits = F.interpolate(
            seg_logits_quarter,
            size=input_hw,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "seg": seg_logits,
            "seg_quarter": seg_logits_quarter,
            "boundary": boundary_logits,
        }
