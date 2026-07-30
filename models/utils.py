import torch
import torch.nn as nn
from spikingjelly.activation_based import base, layer, neuron, surrogate


def build_lif(spike="lif", tau=2.0, backend="cupy", v_threshold=0.5):
    # v_threshold 从默认 1.0 下调到 0.5：BN 后特征≈N(0,1)，阈值 1.0 时仅约 16%
    # 像素过阈发放，弱激活的薄结构/小目标（pole/traffic sign/fence/person）会被
    # 系统性抑制；下调到 0.5 后过阈比例约提升到 31%，缓解弱响应被抹除。
    if spike == "plif":
        return neuron.ParametricLIFNode(
            init_tau=tau,
            v_threshold=v_threshold,
            detach_reset=True,
            step_mode="m",
            backend=backend,
        )

    return neuron.LIFNode(
        tau=tau,
        v_threshold=v_threshold,
        detach_reset=True,
        step_mode="m",
        backend=backend,
    )


class AdaptiveLIFNode(base.MemoryModule):
    """Multi-step AiLIF used only by the first event-encoding neuron.

    The firing threshold rises after a spike and decays back toward a
    trainable base threshold.  This provides short-term event-density
    adaptation without introducing adaptive neurons throughout the network.

    This node intentionally uses the PyTorch backend.  It is instantiated
    only once in SGSR, so the small T-step loop has negligible impact while
    keeping the adaptive state explicit and compatible with reset_net().
    """

    def __init__(
            self,
            tau=2.0,
            base_threshold=0.25,
            threshold_min=0.20,
            threshold_max=0.35,
            adaptation_decay=0.90,
            adaptation_increment=0.05,
            adaptation_increment_max=0.08,
            v_reset=0.0,
            detach_reset=True,
            surrogate_function=None,
    ):
        super().__init__()

        if tau <= 1.0:
            raise ValueError(f"tau must be greater than 1.0, got {tau}.")
        if not threshold_min < threshold_max:
            raise ValueError(
                "threshold_min must be smaller than threshold_max, "
                f"got ({threshold_min}, {threshold_max})."
            )
        if not threshold_min <= base_threshold <= threshold_max:
            raise ValueError(
                "base_threshold must lie inside the threshold range, "
                f"got base={base_threshold}, range="
                f"[{threshold_min}, {threshold_max}]."
            )
        if not 0.0 <= adaptation_decay < 1.0:
            raise ValueError(
                "adaptation_decay must be in [0, 1), "
                f"got {adaptation_decay}."
            )
        if adaptation_increment_max <= 0.0:
            raise ValueError(
                "adaptation_increment_max must be positive, "
                f"got {adaptation_increment_max}."
            )
        if adaptation_increment_max > threshold_max - threshold_min:
            raise ValueError(
                "adaptation_increment_max must not exceed the threshold "
                f"range ({threshold_max - threshold_min}), got "
                f"{adaptation_increment_max}."
            )
        if not 0.0 < adaptation_increment < adaptation_increment_max:
            raise ValueError(
                "adaptation_increment must lie strictly inside (0, "
                f"adaptation_increment_max), got {adaptation_increment} "
                f"and max={adaptation_increment_max}."
            )

        self.tau = float(tau)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.adaptation_decay = float(adaptation_decay)
        self.adaptation_increment_max = float(adaptation_increment_max)
        self.v_reset = float(v_reset)
        self.detach_reset = bool(detach_reset)
        self.step_mode = "m"
        self.backend = "torch"

        # Learn unconstrained raw values and map them smoothly into valid
        # physical ranges.  In contrast to clamping a trainable scalar, this
        # keeps a non-zero gradient when optimization approaches a bound.
        base_fraction = (
                (float(base_threshold) - self.threshold_min)
                / (self.threshold_max - self.threshold_min)
        )
        increment_fraction = (
                float(adaptation_increment) / self.adaptation_increment_max
        )
        eps = 1e-4
        base_fraction = min(max(base_fraction, eps), 1.0 - eps)
        increment_fraction = min(
            max(increment_fraction, eps),
            1.0 - eps,
        )
        self.base_threshold_raw = nn.Parameter(
            torch.logit(torch.tensor(base_fraction, dtype=torch.float32))
        )
        self.adaptation_increment_raw = nn.Parameter(
            torch.logit(
                torch.tensor(increment_fraction, dtype=torch.float32)
            )
        )
        self.surrogate_function = (
            surrogate.ATan()
            if surrogate_function is None
            else surrogate_function
        )

        self.register_memory("v", 0.0)
        self.register_memory("threshold_adaptation", 0.0)

    @property
    def base_threshold(self):
        return self.threshold_min + (
                self.threshold_max - self.threshold_min
        ) * torch.sigmoid(self.base_threshold_raw)

    @property
    def adaptation_increment(self):
        return self.adaptation_increment_max * torch.sigmoid(
            self.adaptation_increment_raw
        )

    def threshold_from_adaptation(self, adaptation):
        """Smoothly saturate the adaptive threshold at threshold_max."""
        base = self.base_threshold
        available_range = (self.threshold_max - base).clamp_min(1e-6)
        adaptation_tensor = torch.as_tensor(
            adaptation,
            dtype=base.dtype,
            device=base.device,
        )
        return base + available_range * (
                1.0 - torch.exp(-adaptation_tensor / available_range)
        )

    @property
    def current_threshold(self):
        return self.threshold_from_adaptation(
            self.threshold_adaptation
        )

    def ailif_parameters(self):
        """Parameters that require the no-weight-decay AiLIF group."""
        return (
            self.base_threshold_raw,
            self.adaptation_increment_raw,
        )

    def single_step_forward(self, x):
        # Match the default decay_input=True LIF dynamics used elsewhere.
        self.v = self.v + (
                x - (self.v - self.v_reset)
        ) / self.tau

        spike = self.surrogate_function(
            self.v - self.current_threshold
        )

        reset_spike = (
            spike.detach()
            if self.detach_reset
            else spike
        )
        self.v = (
                self.v * (1.0 - reset_spike)
                + self.v_reset * reset_spike
        )

        self.threshold_adaptation = (
                self.adaptation_decay * self.threshold_adaptation
                + self.adaptation_increment * spike
        )

        return spike

    def multi_step_forward(self, x_seq):
        if x_seq.dim() < 2:
            raise ValueError(
                "AiLIF expects a multi-step tensor [T, ...], "
                f"got shape={tuple(x_seq.shape)}."
            )

        return torch.stack(
            [self.single_step_forward(x_seq[t]) for t in range(x_seq.shape[0])],
            dim=0,
        )

    def forward(self, x):
        if self.step_mode != "m":
            raise RuntimeError(
                "AdaptiveLIFNode is configured only for step_mode='m'."
            )
        return self.multi_step_forward(x)


def build_ailif(
        tau=2.0,
        base_threshold=0.25,
        threshold_min=0.20,
        threshold_max=0.35,
        adaptation_decay=0.90,
        adaptation_increment=0.05,
        adaptation_increment_max=0.08,
):
    """Build the single adaptive event-encoding neuron used by SGSR."""
    return AdaptiveLIFNode(
        tau=tau,
        base_threshold=base_threshold,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        adaptation_decay=adaptation_decay,
        adaptation_increment=adaptation_increment,
        adaptation_increment_max=adaptation_increment_max,
        detach_reset=True,
    )


class ConvBN(nn.Module):
    def __init__(
            self,
            c1,
            c2,
            k=1,
            p=0,
            s=1,
            g=1,
            d=1,
            bias=False,
            bn_momentum=0.1,
            bn_eps=1e-3,
    ):
        super().__init__()
        self.op = nn.Sequential(
            layer.Conv2d(
                c1, c2, kernel_size=k, stride=s,
                padding=p, dilation=d, groups=g, bias=bias, step_mode="m",
            ),
            layer.BatchNorm2d(c2, eps=bn_eps, momentum=bn_momentum, step_mode="m"),
        )

    def forward(self, x):
        return self.op(x)


class DetailSepConvPostAct(nn.Module):

    def __init__(self, dim, expansion=1.5, kernel_sizes=(3, 5), spike="lif", tau=2.0, backend="cupy",
                 reduced_lif=False, ):
        super().__init__()

        hidden = int(dim * expansion)
        self.kernel_sizes = kernel_sizes
        self.reduced_lif = reduced_lif

        self.pw1 = ConvBN(dim, hidden, k=1)
        self.lif1 = build_lif(spike, tau=tau, backend=backend)

        branch_num = len(kernel_sizes)
        branch_ch = hidden // branch_num
        self.splits = [branch_ch] * branch_num
        self.splits[-1] += hidden - sum(self.splits)

        self.dw_branches = nn.ModuleList()
        for ch, k in zip(self.splits, kernel_sizes):
            self.dw_branches.append(
                ConvBN(ch, ch, k=k, p=k // 2, g=ch, d=1, )
            )

        # reduced_lif=True 时删除深度卷积后的第二次脉冲化，只保留 pw1 后的一次。
        self.lif2 = None if reduced_lif else build_lif(spike, tau=tau, backend=backend)
        self.pw2 = ConvBN(hidden, dim, k=1)

    def forward(self, x):
        out = self.pw1(x)
        out = self.lif1(out)

        xs = torch.split(out, self.splits, dim=2)
        xs = [branch(xi) for branch, xi in zip(self.dw_branches, xs)]
        out = torch.cat(xs, dim=2)

        if not self.reduced_lif:
            out = self.lif2(out)
        out = self.pw2(out)

        return out


class MS_ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=2.0,
            expansion=1.5,
            kernel_sizes=(3, 5),
            spike="lif",
            tau=2.0,
            backend="cupy",
            no_lif=True,
            reduced_lif=False,
    ):
        super().__init__()

        hidden = int(dim * mlp_ratio)
        self.reduced_lif = reduced_lif

        self.conv = DetailSepConvPostAct(
            dim=dim,
            expansion=expansion,
            kernel_sizes=kernel_sizes,
            spike=spike,
            tau=tau,
            backend=backend,
            reduced_lif=reduced_lif,
        )

        # reduced_lif=True 时删除第一次残差后的 LIF。
        self.lif_after_conv = None if reduced_lif else build_lif(spike, tau=tau, backend=backend)

        self.fc1 = ConvBN(dim, hidden, k=1)
        self.lif1 = build_lif(spike, tau=tau, backend=backend)

        self.dw = ConvBN(hidden, hidden, k=3, p=1, g=hidden)
        # reduced_lif=True 时删除 FFN 深度卷积后的 LIF。
        self.lif2 = None if reduced_lif else build_lif(spike, tau=tau, backend=backend)

        self.fc2 = ConvBN(hidden, dim, k=1)

        # reduced_lif=True 时删除第二次残差后的 LIF，浅层 Block 输出连续膜电位。
        self.lif_after_ffn = None if reduced_lif else build_lif(spike, tau=tau, backend=backend)
        self.no_lif = no_lif

        self.gamma1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.gamma2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x):
        scale1 = self.gamma1.view(1, 1, -1, 1, 1)
        scale2 = self.gamma2.view(1, 1, -1, 1, 1)

        residual = x
        out = self.conv(x)
        x = residual + scale1 * out
        if not self.reduced_lif:
            x = self.lif_after_conv(x)

        residual = x
        out = self.fc1(x)
        out = self.lif1(out)

        out = self.dw(out)
        if not self.reduced_lif:
            out = self.lif2(out)

        out = self.fc2(out)
        x = residual + scale2 * out

        if self.reduced_lif or self.no_lif:
            return x

        return self.lif_after_ffn(x)


class MS_ConvBlockV2(nn.Module):
    """
    改进型多尺度卷积残差块。

    设计原则：
        1. 所有隐藏通道首先经过3×3局部空间建模；
        2. 仅部分通道继续经过5×5上下文建模；
        3. 不再将3×3和5×5硬分配给互不重叠的原始通道；
        4. 删除第二套FFN Depthwise 3×3；
        5. 双残差合并为单残差；
        6. 残差缩放由0.01提高至0.1。
    """

    def __init__(
            self,
            dim,
            expansion=2.0,
            context_ratio=0.25,
            spike="lif",
            tau=2.0,
            threshold=0.5,
            backend="cupy",
            residual_scale_init=0.1,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError(
                f"dim must be positive, got {dim}."
            )

        if expansion <= 0:
            raise ValueError(
                f"expansion must be positive, got {expansion}."
            )

        if not 0.0 <= context_ratio < 1.0:
            raise ValueError(
                "context_ratio must be in [0, 1), "
                f"got {context_ratio}."
            )

        hidden = max(
            dim,
            int(round(dim * expansion)),
        )

        context_ch = int(
            round(hidden * context_ratio)
        )

        # context_ratio=0时完全禁用5×5分支。
        if context_ratio > 0.0:
            context_ch = max(
                1,
                min(context_ch, hidden - 1),
            )
        else:
            context_ch = 0

        local_ch = hidden - context_ch

        self.hidden = hidden
        self.local_ch = local_ch
        self.context_ch = context_ch

        # --------------------------------------------------
        # 通道扩展
        # --------------------------------------------------
        self.expand = ConvBN(
            dim,
            hidden,
            k=1,
            p=0,
        )

        self.expand_lif = build_lif(
            spike=spike,
            tau=tau,
            backend=backend,
            v_threshold=threshold,
        )

        # --------------------------------------------------
        # 所有隐藏通道首先经过3×3局部建模
        # --------------------------------------------------
        self.local_dw = ConvBN(
            hidden,
            hidden,
            k=3,
            p=1,
            g=hidden,
        )

        self.local_lif = build_lif(
            spike=spike,
            tau=tau,
            backend=backend,
            v_threshold=threshold,
        )

        # --------------------------------------------------
        # 仅部分通道进一步进行5×5上下文建模
        # --------------------------------------------------
        if context_ch > 0:
            self.context_dw = ConvBN(
                context_ch,
                context_ch,
                k=5,
                p=2,
                g=context_ch,
            )
            # Both multi-scale branches must enter project as binary spikes.
            # Without this LIF, local_feature is a spike tensor while the
            # ConvBN context branch is a continuous membrane tensor.
            self.context_lif = build_lif(
                spike=spike,
                tau=tau,
                backend=backend,
                v_threshold=threshold,
            )
        else:
            self.context_dw = None
            self.context_lif = None

        # --------------------------------------------------
        # 通道投影
        # --------------------------------------------------
        self.project = ConvBN(
            hidden,
            dim,
            k=1,
            p=0,
        )

        self.residual_scale = nn.Parameter(
            torch.full(
                (1, 1, dim, 1, 1),
                float(residual_scale_init),
            )
        )

    def forward(self, x):
        residual = self.expand(
            x
        )

        residual = self.expand_lif(
            residual
        )

        residual = self.local_dw(
            residual
        )

        residual = self.local_lif(
            residual
        )

        if self.context_dw is not None:
            local_feature = residual[
                            :,
                            :,
                            :self.local_ch,
                            :,
                            :,
                            ]

            context_feature = residual[
                              :,
                              :,
                              self.local_ch:,
                              :,
                              :,
                              ]

            context_feature = self.context_dw(
                context_feature
            )

            context_feature = self.context_lif(
                context_feature
            )

            residual = torch.cat(
                [
                    local_feature,
                    context_feature,
                ],
                dim=2,
            )

        residual = self.project(
            residual
        )

        return (
                x
                + self.residual_scale * residual
        )
