# -*- coding: utf-8 -*-
import os
import time
import timeit

import math
import matplotlib
import numpy as np

# 兼容补丁：NumPy>=1.24 已删除 np.int/np.float/np.bool 等旧别名，
# 而 spikingjelly 0.0.0.0.14 内部仍在使用，这里补回以避免 AttributeError。
for _np_alias, _py_type in (("int", int), ("float", float), ("bool", bool),
                            ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _np_alias):
        setattr(np, _np_alias, _py_type)

import torch
from torch import optim
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from utils.utils import setup_seed, netParams
from utils.metric.metric import scores_from_confmat
from utils.losses.loss import (
    CEDiceLoss,
    semantic_to_boundary_target,
    compute_boundary_loss,
    semantic_boundary_probability,
    compute_semantic_boundary_alignment,
)

from utils.optim import RAdam, Ranger
from utils.scheduler.lr_scheduler import WarmupPolyLR, WarmupCosineLR

# event
from dataset.event.base_trainer import BaseTrainer
from datetime import datetime
import json

from spikingjelly.activation_based import functional

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    pass
print(torch_ver)

GLOBAL_SEED = 1234
torch.autograd.set_detect_anomaly(False)

device = torch.device("cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ArgumentParser().error('Boolean value expected for {}'.format(v))


class SNNActivityMonitor:
    """Low-overhead scalar monitor for the modified SGSR encoder.

    Forward hooks are active only on sampled batches. No feature tensor is
    retained: all outputs are detached and immediately reduced to scalars.
    """

    SPIKE_MODULE_SUFFIXES = (
        "stem_half_local.0.pre_lif",
        "stem_half_local.0.spatial_lif",
        "stem_half_local.0.project_lif",
        "stem_half_local.1.pre_lif",
        "stem_half_local.1.spatial_lif",
        "stem_half_local.1.project_lif",
        "bn_lif_1.lif",
        "bn_lif_2.lif",
        "bn_lif_3.lif",
        "bn_lif_4.lif",
        "bn_lif_5.lif",
    )

    MEMBRANE_MODULE_SUFFIXES = (
        "downsample_3",
        "downsample_4",
    )

    GRADIENT_GROUP_SUFFIXES = (
        "stem_half_proj",
        "stem_half_local.0.pre_lif",
        "DAB_Block_1",
        "DAB_Block_2",
        "bottleneck_16",
        "bn_lif_5",
        "downsample_4",
        "dappm",
        "seg_head",
        "boundary_head",
    )

    def __init__(
        self,
        model,
        enabled=True,
        interval=50,
        max_batches=8,
        dead_rate=1e-4,
        high_rate=0.5,
    ):
        self.model = model
        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        self.max_batches = max(1, int(max_batches))
        self.dead_rate = float(dead_rate)
        self.high_rate = float(high_rate)

        self.handles = []
        self.monitored_modules = {}
        self.gradient_groups = {}

        self.phase = None
        self.epoch = None
        self.active = False
        self.sampled_batches = 0
        self.stats = {}
        self.current_input_density = None
        self.current_first_firing_rate = None
        self.density_firing_pairs = []

        if self.enabled:
            self._register_hooks()

    @staticmethod
    def _unwrap_tensor(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        return None

    @staticmethod
    def _matches_suffix(name, suffixes):
        return any(name == suffix or name.endswith("." + suffix)
                   for suffix in suffixes)

    @staticmethod
    def _is_multiscale_spike_module(name):
        in_target_stage = (
            "DAB_Block_1." in name
            or "DAB_Block_2." in name
        )
        return in_target_stage and (
            name.endswith(".local_lif")
            or name.endswith(".context_lif")
        )

    @staticmethod
    def _is_dappm_spike_module(name):
        return (
            (name.startswith("dappm.") or ".dappm." in name)
            and name.endswith(".lif")
        )

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            is_spike = (
                self._matches_suffix(name, self.SPIKE_MODULE_SUFFIXES)
                or self._is_multiscale_spike_module(name)
                or self._is_dappm_spike_module(name)
            )
            is_membrane = self._matches_suffix(
                name,
                self.MEMBRANE_MODULE_SUFFIXES,
            )

            if is_spike or is_membrane:
                kind = "spike" if is_spike else "membrane"
                self.monitored_modules[name] = kind
                self.handles.append(
                    module.register_forward_hook(
                        self._make_forward_hook(name, kind)
                    )
                )

            if self._matches_suffix(name, self.GRADIENT_GROUP_SUFFIXES):
                self.gradient_groups[name] = module

        required = (
            "stem_half_local.0.pre_lif",
            "bn_lif_5.lif",
            "downsample_4",
        )
        missing = [
            suffix for suffix in required
            if not any(
                name == suffix or name.endswith("." + suffix)
                for name in self.monitored_modules
            )
        ]
        if missing:
            raise RuntimeError(
                "SNN monitor could not find required modified SGSR modules: "
                f"{missing}. Make sure the monitored train.py is used with "
                "the modified SGSR.py and utils.py."
            )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start_phase(self, phase, epoch):
        self.phase = str(phase)
        self.epoch = int(epoch)
        self.active = False
        self.sampled_batches = 0
        self.stats = {}
        self.current_input_density = None
        self.current_first_firing_rate = None
        self.density_firing_pairs = []

    def prepare_batch(self, iteration, events):
        if not self.enabled:
            self.active = False
            return

        self.active = (
            self.sampled_batches < self.max_batches
            and int(iteration) % self.interval == 0
        )
        if not self.active:
            return

        self.sampled_batches += 1
        with torch.no_grad():
            detached = events.detach()
            self.current_input_density = float(
                detached.ne(0).float().mean().item()
            )
            self._add("input/nonzero_ratio", self.current_input_density)
            self._add(
                "input/abs_mean",
                float(detached.float().abs().mean().item()),
            )

    def end_forward(self):
        if not self.active:
            return
        if (
            self.current_input_density is not None
            and self.current_first_firing_rate is not None
        ):
            self.density_firing_pairs.append(
                (
                    self.current_input_density,
                    self.current_first_firing_rate,
                )
            )
        self.current_input_density = None
        self.current_first_firing_rate = None

    def record_batch_time(self, batch_time, batch_size):
        if not self.active:
            return
        self._add("runtime/batch_seconds", float(batch_time))
        if batch_time > 0:
            self._add(
                "runtime/samples_per_second",
                float(batch_size) / float(batch_time),
            )

    def _add(self, key, value):
        value = float(value)
        total, count = self.stats.get(key, (0.0, 0))
        self.stats[key] = (total + value, count + 1)

    def _make_forward_hook(self, name, kind):
        def hook(module, inputs, output):
            if not self.active:
                return
            tensor = self._unwrap_tensor(output)
            if tensor is None:
                return

            with torch.no_grad():
                x = tensor.detach().float()
                if kind == "spike":
                    self._record_spike(name, module, x)
                else:
                    self._record_membrane(name, x)

        return hook

    def _record_spike(self, name, module, spike):
        firing_rate = float(spike.mean().item())
        self._add(f"{name}/firing_rate", firing_rate)

        if name.endswith("stem_half_local.0.pre_lif"):
            self.current_first_firing_rate = firing_rate

        if spike.dim() >= 3:
            channel_dim = 2 if spike.dim() == 5 else 1
            reduce_dims = tuple(
                dim for dim in range(spike.dim()) if dim != channel_dim
            )
            channel_rate = spike.mean(dim=reduce_dims)
            self._add(
                f"{name}/dead_channel_ratio",
                channel_rate.le(self.dead_rate).float().mean().item(),
            )
            self._add(
                f"{name}/high_fire_channel_ratio",
                channel_rate.ge(self.high_rate).float().mean().item(),
            )

        if spike.dim() == 5:
            for t in range(spike.shape[0]):
                self._add(
                    f"{name}/firing_rate_t{t}",
                    spike[t].mean().item(),
                )

        if hasattr(module, "base_threshold") and hasattr(
            module,
            "adaptation_increment",
        ):
            self._record_ailif(name, module, spike)

    def _record_ailif(self, name, module, spike):
        base_threshold = module.base_threshold.detach().float()
        increment = module.adaptation_increment.detach().float()
        self._add(f"{name}/base_threshold", base_threshold.item())
        self._add(f"{name}/adaptation_increment", increment.item())

        if spike.dim() != 5:
            threshold = module.threshold_from_adaptation(0.0).detach()
            self._add(f"{name}/threshold_mean", threshold.item())
            return

        adaptation = torch.zeros_like(spike[0])
        eps = 1e-6
        for t in range(spike.shape[0]):
            threshold = module.threshold_from_adaptation(
                adaptation
            ).detach()
            self._add(
                f"{name}/threshold_mean_t{t}",
                threshold.mean().item(),
            )
            self._add(
                f"{name}/threshold_low_clamp_ratio_t{t}",
                threshold.le(float(module.threshold_min) + eps)
                .float().mean().item(),
            )
            self._add(
                f"{name}/threshold_high_clamp_ratio_t{t}",
                threshold.ge(float(module.threshold_max) - eps)
                .float().mean().item(),
            )
            adaptation = (
                float(module.adaptation_decay) * adaptation
                + increment * spike[t]
            )

    def _record_membrane(self, name, membrane):
        self._add(f"{name}/mean", membrane.mean().item())
        self._add(
            f"{name}/std",
            membrane.std(unbiased=False).item(),
        )
        self._add(
            f"{name}/abs_mean",
            membrane.abs().mean().item(),
        )

    def record_module_gradients(self):
        if not self.active or self.phase != "train":
            return
        with torch.no_grad():
            for name, module in self.gradient_groups.items():
                squared_norm = 0.0
                parameter_count = 0
                for parameter in module.parameters():
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad.detach().float()
                    squared_norm += float(grad.square().sum().item())
                    parameter_count += grad.numel()
                if parameter_count > 0:
                    self._add(
                        f"gradient/{name}",
                        math.sqrt(squared_norm),
                    )

    def record_pre_clip_gradient(self, total_norm):
        if not self.active or self.phase != "train":
            return
        if torch.is_tensor(total_norm):
            total_norm = total_norm.detach().item()
        self._add("gradient/global_pre_clip", total_norm)
        self._add("gradient/clip_triggered", float(total_norm > 1.0))

    @staticmethod
    def _pearson(pairs):
        if len(pairs) < 2:
            return None
        x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        if x.std() < 1e-12 or y.std() < 1e-12:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    def finish_phase(self):
        if not self.enabled:
            return {"enabled": False}

        metrics = {
            key: total / max(count, 1)
            for key, (total, count) in self.stats.items()
        }

        correlation = self._pearson(self.density_firing_pairs)
        if correlation is not None:
            metrics["input/density_first_firing_correlation"] = correlation

        for stage in ("DAB_Block_1", "DAB_Block_2"):
            context_values = [
                value for key, value in metrics.items()
                if f"{stage}." in key
                and key.endswith("context_lif/firing_rate")
            ]
            local_values = [
                value for key, value in metrics.items()
                if f"{stage}." in key
                and key.endswith("local_lif/firing_rate")
            ]
            if context_values and local_values:
                context_mean = sum(context_values) / len(context_values)
                local_mean = sum(local_values) / len(local_values)
                metrics[f"derived/{stage}_context_local_fr_ratio"] = (
                    context_mean / max(local_mean, 1e-12)
                )

        summary = {
            "enabled": True,
            "epoch": self.epoch,
            "phase": self.phase,
            "sampled_batches": self.sampled_batches,
            "interval": self.interval,
            "metrics": metrics,
        }
        self.active = False
        return summary


def parse_args():
    parser = ArgumentParser(
        description=(
            'Event-based semantic segmentation '
            'with SGSR decoder4 membrane readout'
        )
    )

    # ---------------------------------------------------------
    # Model and dataset
    # ---------------------------------------------------------
    parser.add_argument('--model', type=str, default='SGSR')
    parser.add_argument('--dataset', type=str, default='DSEC_events', choices=['DSEC_events', 'DDD17_events'])
    parser.add_argument('--input_size', type=str, default='440,640')
    parser.add_argument('--classes', type=int, default=11)
    parser.add_argument('--dataset_path', type=str, default='/root/autodl-tmp/DSEC_Semantic')
    parser.add_argument('--split', type=str, default='train')

    # ---------------------------------------------------------
    # Event representation
    # ---------------------------------------------------------
    parser.add_argument('--nr_events_data', type=int, default=1)
    parser.add_argument('--delta_t_per_data', type=int, default=50)
    parser.add_argument('--nr_events_window', type=int, default=100000)
    parser.add_argument('--data_augmentation_train', type=str2bool, default=True)
    parser.add_argument('--event_representation', type=str, default='voxel_grid')
    parser.add_argument('--nr_temporal_bins', type=int, default=3)
    parser.add_argument('--require_paired_data_train', type=str2bool, default=False)
    parser.add_argument('--require_paired_data_val', type=str2bool, default=False)
    parser.add_argument('--separate_pol', type=str2bool, default=False)
    parser.add_argument('--normalize_event', type=str2bool, default=True)
    parser.add_argument('--fixed_duration', type=str2bool, default=False)

    # ---------------------------------------------------------
    # First event neuron ablation.  Keep every other architecture and loss
    # setting identical between fixed_lif and ailif runs.
    # ---------------------------------------------------------
    parser.add_argument(
        '--first_neuron_type',
        type=str.lower,
        default='ailif',
        choices=['fixed_lif', 'ailif'],
    )
    parser.add_argument('--first_lif_threshold', type=float, default=0.2)
    parser.add_argument('--ailif_base_threshold', type=float, default=0.25)
    parser.add_argument('--ailif_threshold_min', type=float, default=0.20)
    parser.add_argument('--ailif_threshold_max', type=float, default=0.35)
    parser.add_argument('--ailif_adaptation_decay', type=float, default=0.90)
    parser.add_argument('--ailif_adaptation_increment', type=float, default=0.05)
    parser.add_argument('--ailif_adaptation_increment_max', type=float, default=0.08)
    parser.add_argument(
        '--ailif_lr_scale',
        type=float,
        default=0.25,
        help='AiLIF raw-parameter LR as a fraction of the main LR.',
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    # 多轮实验显示性能峰值集中在 40–50 epoch，60 epoch 的余弦周期比 50/100 更贴合
    # 当前模型的实际收敛规律（40 epoch 仍保留约 1e-4 学习率，50 epoch 进入细调）。
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optim', type=str.lower, default='adamw', choices=['sgd', 'adam', 'radam', 'ranger', 'adamw'])
    parser.add_argument('--lr_schedule', type=str, default='warmupcosine',
                        choices=['poly', 'warmpoly', 'StepLR', 'warmupcosine'])
    parser.add_argument('--poly_exp', type=float, default=0.9)
    parser.add_argument('--warmup_iters', type=int, default=500)

    # ---------------------------------------------------------
    # Early stopping
    # 逐轮 mIoU 波动约 0.2–0.8 个百分点，故用较长耐心（12 轮），
    # 避免在真正的性能峰值前提前停止。min_delta=0.0005 即 0.05 个百分点。
    # ---------------------------------------------------------
    parser.add_argument('--early_stop', type=str2bool, default=True)
    parser.add_argument('--early_stop_patience', type=int, default=12)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0005)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)

    # ---------------------------------------------------------
    # Checkpoint and logging
    # ---------------------------------------------------------
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--savedir', default='./checkpoint/')
    parser.add_argument('--logFile', default='log.txt')
    parser.add_argument('--arguFile', default='arguments.txt')

    parser.add_argument(
        '--boundary_weight',
        type=float,
        default=0.15,
    )
    parser.add_argument(
        '--semantic_boundary_weight',
        type=float,
        default=0.08,
    )
    parser.add_argument(
        '--semantic_boundary_kappa',
        type=float,
        default=4.0,
    )

    # ---------------------------------------------------------
    # SNN diagnostics. Hooks reduce outputs to scalars immediately and are
    # active only on sampled batches.
    # ---------------------------------------------------------
    parser.add_argument('--snn_monitor', type=str2bool, default=True)
    parser.add_argument('--snn_monitor_interval', type=int, default=50)
    parser.add_argument('--snn_monitor_batches', type=int, default=8)
    parser.add_argument('--snn_monitor_log', type=str, default='snn_monitor.jsonl')
    parser.add_argument('--snn_dead_rate', type=float, default=1e-4)
    parser.add_argument('--snn_high_rate', type=float, default=0.5)

    return parser.parse_args()


def build_checkpoint_state(args, epoch, model, optimizer, scheduler,
                           best_mIOU_val, best_acc_val, best_mIOU_per_class,
                           mIOU_val_list, lossTr_list, lossVal_list,
                           ceLossTr_list, diceLossTr_list,
                           ceLossVal_list, diceLossVal_list,
                           boundaryLossTr_list, boundaryBceTr_list,
                           boundaryWeightedTr_list,
                           boundaryLossVal_list, boundaryBceVal_list,
                           boundaryWeightedVal_list,
                           semanticBoundaryLossTr_list,
                           semanticBoundaryBceTr_list,
                           semanticBoundaryDiceTr_list,
                           semanticBoundaryWeightedTr_list,
                           semanticBoundaryLossVal_list,
                           semanticBoundaryBceVal_list,
                           semanticBoundaryDiceVal_list,
                           semanticBoundaryWeightedVal_list,
                           semanticBoundaryF1Val_list,
                           semanticBoundaryF1Tol1Val_list):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_mIOU_val,
        "best_acc": best_acc_val,
        "best_miou_class": best_mIOU_per_class,
        "mIOU_val_list": mIOU_val_list,

        # 总损失（语义 + 加权边界 + 加权语义边界对齐）
        "lossTr_list": lossTr_list,
        "lossVal_list": lossVal_list,

        # 分项损失
        "ceLossTr_list": ceLossTr_list,
        "diceLossTr_list": diceLossTr_list,
        "ceLossVal_list": ceLossVal_list,
        "diceLossVal_list": diceLossVal_list,

        "boundaryLossTr_list": boundaryLossTr_list,
        "boundaryBceTr_list": boundaryBceTr_list,
        "boundaryWeightedTr_list": boundaryWeightedTr_list,
        "boundaryLossVal_list": boundaryLossVal_list,
        "boundaryBceVal_list": boundaryBceVal_list,
        "boundaryWeightedVal_list": boundaryWeightedVal_list,

        "semanticBoundaryLossTr_list": semanticBoundaryLossTr_list,
        "semanticBoundaryBceTr_list": semanticBoundaryBceTr_list,
        "semanticBoundaryDiceTr_list": semanticBoundaryDiceTr_list,
        "semanticBoundaryWeightedTr_list": semanticBoundaryWeightedTr_list,
        "semanticBoundaryLossVal_list": semanticBoundaryLossVal_list,
        "semanticBoundaryBceVal_list": semanticBoundaryBceVal_list,
        "semanticBoundaryDiceVal_list": semanticBoundaryDiceVal_list,
        "semanticBoundaryWeightedVal_list": semanticBoundaryWeightedVal_list,
        "semanticBoundaryF1Val_list": semanticBoundaryF1Val_list,
        "semanticBoundaryF1Tol1Val_list": semanticBoundaryF1Tol1Val_list,

        "loss_name": (
            "ce_plus_0.5_seg_dice"
            f"_plus_{args.boundary_weight:g}_boundary"
            f"_plus_{args.semantic_boundary_weight:g}"
            "_semantic_boundary"
        ),
        "loss_config": {
            "ce_weight": 1.0,
            "seg_dice_weight": 0.5,

            "boundary_weight": (
                args.boundary_weight
            ),
            "boundary_loss": (
                "weighted_bce_plus_0.5_soft_dice"
            ),
            "boundary_target": (
                "resize_label_first_then_compact_boundary"
            ),
            "boundary_target_scale": "1/4",
            "boundary_dilation": 0,
            "boundary_dice": True,
            "boundary_dice_weight": 0.5,
            "boundary_pos_weight": None,

            "semantic_boundary_weight": (
                args.semantic_boundary_weight
            ),
            "semantic_boundary_loss": (
                "semantic_probability_boundary_"
                "weighted_bce_plus_0.5_soft_dice"
            ),
            "semantic_boundary_scale": "decoder4_1_4",
            "semantic_boundary_kappa": (
                args.semantic_boundary_kappa
            ),
            "semantic_boundary_dice_weight": 0.5,

            "old_pairwise_structure": False,
            "structure_interior": False,
            "edge_semantic": False,
            "ownership": False,
            "auxiliary_semantic": False,

            "class_weights": None,
            "ignore_index": args.ignore_label,
        },
        "boundary_module_config": {
            "type": (
                "decoder4_quarter_resolution_boundary_head"
            ),

            "boundary_controls_feature_fusion": False,

            "boundary_gate": None,

            "shared_feature": (
                "decoder4_1_4_resolution_64_channels"
            ),

            "stem_half_feature": (
                "1_2_resolution_32_channels_two_local_blocks"
            ),

            "stem_downsampling": (
                "pixel_unshuffle_2_then_1x1"
            ),

            "decode4_type": (
                "unified_spike_decoder"
            ),

            "decode2_type": None,

            "segmentation_and_boundary_share_decoder2": False,

            "segmentation_and_boundary_share_decoder4": True,

            "shared_decoder_receives_both_gradients": True,

            "boundary_classifier_receives_seg_gradient": False,

            "boundary_output_scale": "1/4",
        },
        "encoder_config": {
            "stem_first_downsampling": "pixel_unshuffle_2_then_1x1",

            "first_neuron_type": args.first_neuron_type,

            "first_lif_threshold": args.first_lif_threshold,

            "ailif_base_threshold": args.ailif_base_threshold,

            "ailif_threshold_min": args.ailif_threshold_min,

            "ailif_threshold_max": args.ailif_threshold_max,

            "ailif_adaptation_decay": args.ailif_adaptation_decay,

            "ailif_adaptation_increment": (
                args.ailif_adaptation_increment
            ),

            "ailif_adaptation_increment_max": (
                args.ailif_adaptation_increment_max
            ),

            "ailif_parameterization": "sigmoid_bounded_no_hard_clamp",

            "ailif_lr_scale": args.ailif_lr_scale,

            "ailif_weight_decay": 0.0,

            "stem_second_downsampling": (
                "learned_3x3_stride2"
            ),

            "stem_half_channels": 32,

            "stem_half_blocks": 2,

            "stem_half_block_type": (
                "half_resolution_local_block"
            ),

            "stage4_block_type": (
                "MS_ConvBlockV2"
            ),

            "stage8_block_type": (
                "MS_ConvBlockV2"
            ),

            "stage8_channels": 128,

            "stage8_blocks": 4,

            "stage16_channels": 256,

            "stage16_bottleneck_blocks": 6,

            "stage32_channels": 512,

            "capacity_strategy": (
                "widen_1_8_to_128_and_1_16_to_256_with_six_bottlenecks"
            ),

            "ms_block_local_kernel": 3,

            "ms_block_context_kernel": 5,

            "stage8_context_ratio_schedule": [
                0.25,
                0.25,
                0.25,
                0.0,
            ],

            "stage8_expansion_schedule": [
                2.0,
                2.0,
                2.0,
                1.5,
            ],

            "stage8_residual_scale_schedule": [
                0.1,
                0.1,
                0.1,
                0.05,
            ],

            "ms_block_residual_scale_init": 0.1,
        },
        "readout_config": {
            "type": "mean_membrane_batchnorm_1x1",
            "stage": "decoder4_1_4",
            "semantic_upsample": "bilinear_x4_to_input",
            "boundary_output_scale": "1/4",
            "learnable_temporal_weight": False,
        },
        "ignore_index": args.ignore_label,
        "args": vars(args),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    return state


def train_model(args):
    global device

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))

    # set the seed
    setup_seed(GLOBAL_SEED)

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("=====> building network")

    # build the model and initialization
    in_channels = 2 if args.separate_pol else 1

    if args.model.lower() == 'sgsr':
        # Instantiate SGSR directly so the ablation arguments do not depend
        # on whether an older project-level model_builder forwards **kwargs.
        from model.SGSR import SGSR

        model = SGSR(
            num_classes=args.classes,
            in_channels=in_channels,
            temporal_steps=args.nr_temporal_bins,
            block_1=3,
            block_2=4,
            block_3=6,
            first_neuron_type=args.first_neuron_type,
            first_lif_threshold=args.first_lif_threshold,
            ailif_base_threshold=args.ailif_base_threshold,
            ailif_threshold_min=args.ailif_threshold_min,
            ailif_threshold_max=args.ailif_threshold_max,
            ailif_adaptation_decay=args.ailif_adaptation_decay,
            ailif_adaptation_increment=args.ailif_adaptation_increment,
            ailif_adaptation_increment_max=(
                args.ailif_adaptation_increment_max
            ),
        )
    else:
        model = build_model(
            args.model,
            num_classes=args.classes,
            in_channels=in_channels,
            temporal_steps=args.nr_temporal_bins,
        )

    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
    # SGSR is intrinsically a multi-step model. Training should only verify
    # the configuration here rather than mutating step_mode externally.
    step_modules = [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "step_mode")
    ]
    invalid_step_modules = [
        (name, module.__class__.__name__, module.step_mode)
        for name, module in step_modules
        if module.step_mode != "m"
    ]
    if invalid_step_modules:
        details = "\n".join(
            f"  {name}: {cls_name}(step_mode={step_mode!r})"
            for name, cls_name, step_mode in invalid_step_modules
        )
        raise RuntimeError(
            "Expected an intrinsically multi-step model, but found "
            f"non-'m' modules:\n{details}"
        )
    print(f"=====> verified multi-step modules: {len(step_modules)}")

    # DDD17/DSEC datasets
    base_trainer_instance = BaseTrainer()
    trainLoader, valLoader = base_trainer_instance.createDataLoaders(args)

    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter

    criteria = CEDiceLoss(
        num_classes=args.classes,
        ignore_index=args.ignore_label,
        dice_weight=0.5,
        dice_smooth=1.0,
        dice_p=2.0,
        only_present_classes=True,
    )

    if torch.cuda.is_available():
        criteria = criteria.cuda(device)
        model = model.cuda(device)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.savedir = (args.savedir + args.dataset + '/' + args.model + "_" + current_time + "_" + str(args.split) + '/')

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0
    best_mIOU_per_class = []
    best_mIOU_val = 0
    best_acc_val = 0

    # 早停计数器：连续 patience 轮 mIoU 未获得超过 min_delta 的提升则停止。
    epochs_without_improvement = 0

    mIOU_val_list = []

    # lossTr_list 与 lossVal_list 保存所有加权分项相加后的总损失，
    # 保留名称用于兼容旧绘图与checkpoint接口。
    lossTr_list = []
    lossVal_list = []

    # 分项损失历史
    ceLossTr_list = []
    diceLossTr_list = []
    ceLossVal_list = []
    diceLossVal_list = []

    boundaryLossTr_list = []
    boundaryBceTr_list = []
    boundaryWeightedTr_list = []

    boundaryLossVal_list = []
    boundaryBceVal_list = []
    boundaryWeightedVal_list = []

    semanticBoundaryLossTr_list = []
    semanticBoundaryBceTr_list = []
    semanticBoundaryDiceTr_list = []
    semanticBoundaryWeightedTr_list = []

    semanticBoundaryLossVal_list = []
    semanticBoundaryBceVal_list = []
    semanticBoundaryDiceVal_list = []
    semanticBoundaryWeightedVal_list = []
    semanticBoundaryF1Val_list = []
    semanticBoundaryF1Tol1Val_list = []

    resume_checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            resume_checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = resume_checkpoint['epoch']
            best_mIOU_val = resume_checkpoint.get('best_miou', 0)
            best_mIOU_per_class = resume_checkpoint.get('best_miou_class', [])
            best_acc_val = resume_checkpoint.get('best_acc', 0)
            mIOU_val_list = resume_checkpoint.get('mIOU_val_list', [])
            lossTr_list = resume_checkpoint.get('lossTr_list', [])
            lossVal_list = resume_checkpoint.get('lossVal_list', [])
            ceLossTr_list = resume_checkpoint.get('ceLossTr_list', [])
            diceLossTr_list = resume_checkpoint.get('diceLossTr_list', [])
            ceLossVal_list = resume_checkpoint.get('ceLossVal_list', [])
            diceLossVal_list = resume_checkpoint.get('diceLossVal_list', [])
            boundaryLossTr_list = resume_checkpoint.get('boundaryLossTr_list', [])
            boundaryBceTr_list = resume_checkpoint.get('boundaryBceTr_list', [])
            boundaryWeightedTr_list = resume_checkpoint.get('boundaryWeightedTr_list', [])
            boundaryLossVal_list = resume_checkpoint.get('boundaryLossVal_list', [])
            boundaryBceVal_list = resume_checkpoint.get('boundaryBceVal_list', [])
            boundaryWeightedVal_list = resume_checkpoint.get('boundaryWeightedVal_list', [])
            semanticBoundaryLossTr_list = resume_checkpoint.get('semanticBoundaryLossTr_list', [])
            semanticBoundaryBceTr_list = resume_checkpoint.get('semanticBoundaryBceTr_list', [])
            semanticBoundaryDiceTr_list = resume_checkpoint.get('semanticBoundaryDiceTr_list', [])
            semanticBoundaryWeightedTr_list = resume_checkpoint.get('semanticBoundaryWeightedTr_list', [])
            semanticBoundaryLossVal_list = resume_checkpoint.get('semanticBoundaryLossVal_list', [])
            semanticBoundaryBceVal_list = resume_checkpoint.get('semanticBoundaryBceVal_list', [])
            semanticBoundaryDiceVal_list = resume_checkpoint.get('semanticBoundaryDiceVal_list', [])
            semanticBoundaryWeightedVal_list = resume_checkpoint.get('semanticBoundaryWeightedVal_list', [])
            semanticBoundaryF1Val_list = resume_checkpoint.get('semanticBoundaryF1Val_list', [])
            semanticBoundaryF1Tol1Val_list = resume_checkpoint.get('semanticBoundaryF1Tol1Val_list', [])
            load_result = model.load_state_dict(
                resume_checkpoint['model'],
                strict=False,
            )
            print("Missing keys:", load_result.missing_keys)
            print("Unexpected keys:", load_result.unexpected_keys)
            print("=====> loaded checkpoint '{}' (epoch {})".format(args.resume, resume_checkpoint['epoch']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))

    model.train()

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write("\n%s\t\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)', 'lr'))
    logger.flush()

    # 记录参数
    arguFileLoc = args.savedir + args.arguFile
    if os.path.isfile(arguFileLoc):
        logger_argu = open(arguFileLoc, 'a')
    else:
        logger_argu = open(arguFileLoc, 'w')
        json.dump(args.__dict__, logger_argu, indent=2)
    logger_argu.flush()

    # tensorboard记录loss和iou曲线
    # writer = SummaryWriter(log_dir=args.savedir)

    # define optimization strategy
    if args.ailif_lr_scale <= 0.0:
        raise ValueError(
            f"ailif_lr_scale must be positive, got {args.ailif_lr_scale}."
        )

    trainable_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    ailif_suffixes = (
        "base_threshold_raw",
        "adaptation_increment_raw",
    )
    ailif_named_parameters = [
        (name, parameter)
        for name, parameter in trainable_named_parameters
        if name.endswith(ailif_suffixes)
    ]
    ailif_parameter_ids = {
        id(parameter) for _, parameter in ailif_named_parameters
    }
    main_parameters = [
        parameter
        for _, parameter in trainable_named_parameters
        if id(parameter) not in ailif_parameter_ids
    ]
    ailif_parameters = [
        parameter for _, parameter in ailif_named_parameters
    ]

    if args.first_neuron_type == 'ailif' and len(ailif_parameters) != 2:
        raise RuntimeError(
            "AiLIF mode must expose exactly base_threshold_raw and "
            "adaptation_increment_raw, but found: "
            f"{[name for name, _ in ailif_named_parameters]}"
        )
    if args.first_neuron_type == 'fixed_lif' and ailif_parameters:
        raise RuntimeError(
            "fixed_lif mode unexpectedly contains AiLIF parameters: "
            f"{[name for name, _ in ailif_named_parameters]}"
        )

    def optimizer_parameter_groups(main_weight_decay):
        groups = [
            {
                'params': main_parameters,
                'lr': args.lr,
                'weight_decay': main_weight_decay,
                'group_name': 'main',
            }
        ]
        if ailif_parameters:
            groups.append(
                {
                    'params': ailif_parameters,
                    'lr': args.lr * args.ailif_lr_scale,
                    'weight_decay': 0.0,
                    'group_name': 'ailif',
                }
            )
        return groups

    print(
        "=====> first neuron: {} | AiLIF params: {} | "
        "AiLIF lr: {:.3e} | AiLIF weight_decay: 0".format(
            args.first_neuron_type,
            [name for name, _ in ailif_named_parameters],
            args.lr * args.ailif_lr_scale if ailif_parameters else 0.0,
        )
    )

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            optimizer_parameter_groups(1e-4),
            lr=args.lr,
            momentum=0.9,
        )
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_parameter_groups(1e-4),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif args.optim == 'radam':
        optimizer = RAdam(
            optimizer_parameter_groups(1e-4),
            lr=args.lr,
            betas=(0.90, 0.999),
            eps=1e-08,
        )
    elif args.optim == 'ranger':
        optimizer = Ranger(
            optimizer_parameter_groups(1e-4),
            lr=args.lr,
            betas=(0.95, 0.999),
            eps=1e-08,
        )
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(1e-2),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    args.cur_iter = start_epoch * args.per_iter

    # learming scheduling
    scheduler = None
    if args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'warmpoly':
        scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=args.warmup_iters, power=0.8)
    elif args.lr_schedule == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.92, last_epoch=-1)
    elif args.lr_schedule == 'warmupcosine':
        # 学习率仍按 iteration 平滑更新，只是把总衰减周期对齐到 60 epoch 的总 iteration；
        # eta_min 由 args.min_lr 控制，收敛末端不再降到 0。
        scheduler = WarmupCosineLR(
            optimizer,
            args.max_iter,
            warmup_iters=args.warmup_iters,
            eta_min=args.min_lr,
        )

    if resume_checkpoint is not None:
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler'])

    snn_monitor = SNNActivityMonitor(
        model=model,
        enabled=args.snn_monitor,
        interval=args.snn_monitor_interval,
        max_batches=args.snn_monitor_batches,
        dead_rate=args.snn_dead_rate,
        high_rate=args.snn_high_rate,
    )
    if args.snn_monitor:
        print(
            "=====> SNN monitor enabled: "
            f"modules={len(snn_monitor.monitored_modules)}, "
            f"interval={args.snn_monitor_interval}, "
            f"max_batches={args.snn_monitor_batches}"
        )

    print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        # training

        train_losses, lr, train_monitor_summary = train(
            args,
            trainLoader,
            model,
            criteria,
            optimizer,
            scheduler,
            epoch,
            snn_monitor,
        )

        # 原有loss列表保存总损失，
        # 保持旧绘图和checkpoint接口兼容。
        lossTr_list.append(train_losses['total'])
        ceLossTr_list.append(train_losses['ce'])
        diceLossTr_list.append(train_losses['dice'])
        boundaryLossTr_list.append(train_losses['boundary'])
        boundaryBceTr_list.append(train_losses['boundary_bce'])
        boundaryWeightedTr_list.append(train_losses['boundary_weighted'])
        semanticBoundaryLossTr_list.append(train_losses['semantic_boundary'])
        semanticBoundaryBceTr_list.append(train_losses['semantic_boundary_bce'])
        semanticBoundaryDiceTr_list.append(train_losses['semantic_boundary_dice'])
        semanticBoundaryWeightedTr_list.append(train_losses['semantic_boundary_weighted'])

        if args.lr_schedule == 'StepLR' and scheduler is not None:
            scheduler.step()

        # validation写入log.txt
        # if epoch % 50 == 0 or epoch == (args.max_epochs - 1):#50的整数倍以及最大max_epoch-1 记录mIou在.txt文件中
        if epoch < args.max_epochs:  # 每个epoch都验证且记录 记录mIou在.txt文件中
            (
                mIOU_val,
                per_class_iu,
                acc_val,
                val_losses,
                val_monitor_summary,
            ) = val(
                args,
                valLoader,
                model,
                criteria,
                epoch,
                snn_monitor,
            )
            mIOU_val_list.append(mIOU_val)
            lossVal_list.append(val_losses['total'])
            ceLossVal_list.append(val_losses['ce'])
            diceLossVal_list.append(val_losses['dice'])
            boundaryLossVal_list.append(val_losses['boundary'])
            boundaryBceVal_list.append(val_losses['boundary_bce'])
            boundaryWeightedVal_list.append(val_losses['boundary_weighted'])
            semanticBoundaryLossVal_list.append(val_losses['semantic_boundary'])
            semanticBoundaryBceVal_list.append(val_losses['semantic_boundary_bce'])
            semanticBoundaryDiceVal_list.append(val_losses['semantic_boundary_dice'])
            semanticBoundaryWeightedVal_list.append(val_losses['semantic_boundary_weighted'])
            semanticBoundaryF1Val_list.append(val_losses['semantic_boundary_f1'])
            semanticBoundaryF1Tol1Val_list.append(val_losses['semantic_boundary_f1_tol1'])

            # 早停判定基于「超过 min_delta 的显著提升」，需在更新 best 之前比较。
            is_significant_improvement = (
                mIOU_val > best_mIOU_val + args.early_stop_min_delta
            )

            is_best_miou = mIOU_val > best_mIOU_val
            is_best_acc = acc_val > best_acc_val

            if is_best_miou:
                best_mIOU_val = mIOU_val
                best_mIOU_per_class = list(per_class_iu)

            if is_significant_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if is_best_acc:
                best_acc_val = acc_val

            state = build_checkpoint_state(
                args, epoch + 1, model, optimizer, scheduler,
                best_mIOU_val, best_acc_val, best_mIOU_per_class,
                mIOU_val_list, lossTr_list, lossVal_list,
                ceLossTr_list, diceLossTr_list,
                ceLossVal_list, diceLossVal_list,
                boundaryLossTr_list, boundaryBceTr_list,
                boundaryWeightedTr_list,
                boundaryLossVal_list, boundaryBceVal_list,
                boundaryWeightedVal_list,
                semanticBoundaryLossTr_list,
                semanticBoundaryBceTr_list,
                semanticBoundaryDiceTr_list,
                semanticBoundaryWeightedTr_list,
                semanticBoundaryLossVal_list,
                semanticBoundaryBceVal_list,
                semanticBoundaryDiceVal_list,
                semanticBoundaryWeightedVal_list,
                semanticBoundaryF1Val_list,
                semanticBoundaryF1Tol1Val_list)

            if is_best_miou:
                torch.save(state, os.path.join(args.savedir, 'model_best_miou.pth'))

            if is_best_acc:
                torch.save(state, os.path.join(args.savedir, 'model_best_acc.pth'))

            torch.save(state, os.path.join(args.savedir, 'model_latest.pth'))

            # record train information
            report = format_epoch_report(
                args, args.dataset, epoch, train_losses, val_losses, mIOU_val, best_mIOU_val,
                per_class_iu, best_acc_val, lr,
                train_monitor_summary=train_monitor_summary,
                val_monitor_summary=val_monitor_summary,
            )
            logger.write("\n" + report)
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print(report)

            target_model = (
                model.module
                if hasattr(model, "module")
                else model
            )
            if (
                hasattr(target_model, "decode8")
                and target_model.decode8.skip_scale is not None
            ):
                with torch.no_grad():
                    skip_scale = (
                        target_model.decode8.skip_scale.detach()
                    )
                    print(
                        "Decoder8 skip scale:",
                        f"mean={skip_scale.mean().item():.4f},",
                        f"min={skip_scale.min().item():.4f},",
                        f"max={skip_scale.max().item():.4f},",
                        f"std={skip_scale.std(unbiased=False).item():.4f}",
                    )

            if args.snn_monitor:
                monitor_record = {
                    "epoch": epoch,
                    "train": train_monitor_summary,
                    "validation": val_monitor_summary,
                }
                with open(
                    os.path.join(args.savedir, args.snn_monitor_log),
                    'a',
                    encoding='utf-8',
                ) as monitor_file:
                    monitor_file.write(
                        json.dumps(monitor_record, ensure_ascii=False)
                        + "\n"
                    )

        else:
            # record train information  #其他不用记录mIou
            logger.write("\n%d\t\t%.4f\t\t\t\t%.7f" % (epoch, train_losses['total'], lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("Epoch No.: %d\tTrain Total Loss = %.4f\t lr= %.6f\n" % (epoch, train_losses['total'], lr))

        if epoch % 5 == 0 or epoch == (args.max_epochs - 1):
            # Plot the figures per 5 epochs

            # 总损失（CE + 0.5 Dice）
            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(range(len(lossTr_list)), lossTr_list, label="Train Total Loss")
            ax1.plot(range(len(lossVal_list)), lossVal_list, label="Validation Total Loss")
            ax1.set_title("Total loss (CE + 0.5 SegDice + boundary) vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Total Loss")
            ax1.legend(loc='upper right')

            plt.savefig(args.savedir + "loss_vs_epochs_total.png")

            plt.clf()

            # CE分项
            fig_ce, ax_ce = plt.subplots(figsize=(11, 8))

            ax_ce.plot(range(len(ceLossTr_list)), ceLossTr_list, label="Train CE")
            ax_ce.plot(range(len(ceLossVal_list)), ceLossVal_list, label="Validation CE")
            ax_ce.set_title("CE loss vs epochs")
            ax_ce.set_xlabel("Epochs")
            ax_ce.set_ylabel("CE Loss")
            ax_ce.legend(loc='upper right')

            plt.savefig(args.savedir + "loss_vs_epochs_ce.png")

            plt.clf()

            # Dice分项
            fig_dice, ax_dice = plt.subplots(figsize=(11, 8))

            ax_dice.plot(range(len(diceLossTr_list)), diceLossTr_list, label="Train Dice")
            ax_dice.plot(range(len(diceLossVal_list)), diceLossVal_list, label="Validation Dice")
            ax_dice.set_title("Dice loss vs epochs")
            ax_dice.set_xlabel("Epochs")
            ax_dice.set_ylabel("Dice Loss")
            ax_dice.legend(loc='upper right')

            plt.savefig(args.savedir + "loss_vs_epochs_dice.png")

            plt.clf()

            # Boundary分项
            fig_boundary, ax_boundary = plt.subplots(figsize=(11, 8))

            ax_boundary.plot(
                range(len(boundaryLossTr_list)),
                boundaryLossTr_list,
                label="Train Boundary",
            )
            ax_boundary.plot(
                range(len(boundaryLossVal_list)),
                boundaryLossVal_list,
                label="Validation Boundary",
            )
            ax_boundary.set_title(
                "Boundary loss (weighted BCE + 0.5 Dice)"
            )
            ax_boundary.set_xlabel("Epochs")
            ax_boundary.set_ylabel("Boundary Loss")
            ax_boundary.legend(loc='upper right')

            plt.savefig(args.savedir + "loss_vs_epochs_boundary.png")

            plt.clf()

            # val miou
            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(range(len(mIOU_val_list)), mIOU_val_list, label="Val IoU")
            ax2.set_title("Average IoU vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Current IoU")
            plt.legend(loc='lower right')

            plt.savefig(args.savedir + "iou_vs_epochs_val.png")

            plt.close('all')

        # 早停：连续 patience 轮无显著提升则提前结束训练。
        if args.early_stop and epochs_without_improvement >= args.early_stop_patience:
            stop_msg = (
                "Early stopping at epoch %d, best mIoU=%.2f%% "
                "(no improvement > %.4f for %d epochs)"
                % (
                    epoch + 1,
                    best_mIOU_val * 100,
                    args.early_stop_min_delta,
                    epochs_without_improvement,
                )
            )
            print(stop_msg)
            logger.write("\n" + stop_msg + "\n")
            logger.flush()
            break

    snn_monitor.close()
    logger.close()


def compute_step_losses(
    args,
    criterion,
    seg_logits,
    seg_logits_quarter,
    boundary_logits,
    labels,
):
    """
    Final loss:

        CE
        + 0.5 * SegDice
        + 0.15 * Boundary
        + 0.08 * SemanticBoundary
    """

    # 1. 主语义监督：全分辨率CE + Dice
    seg_loss_dict = criterion(
        seg_logits,
        labels,
    )

    # 2. 在decoder4原生1/4尺度生成统一GT边界
    native_size = boundary_logits.shape[-2:]

    if seg_logits_quarter.shape[-2:] != native_size:
        raise ValueError(
            "seg_logits_quarter and boundary_logits "
            "must have the same resolution."
        )

    boundary_target, boundary_valid = (
        semantic_to_boundary_target(
            labels=labels,
            output_size=native_size,
            ignore_index=args.ignore_label,
            tolerance_radius=1,
            soft_band_value=0.5,
            ignore_radius=1,
        )
    )

    # 3. 独立边界头监督
    boundary_loss_dict = compute_boundary_loss(
        boundary_logits=boundary_logits,
        boundary_target=boundary_target,
        boundary_valid=boundary_valid,
        dice_weight=0.5,
    )

    boundary_weighted = (
        args.boundary_weight
        * boundary_loss_dict["total"]
    )

    # 4. 最终语义预测的边界对齐监督
    semantic_boundary_dict = (
        compute_semantic_boundary_alignment(
            seg_logits=seg_logits_quarter,
            boundary_target=boundary_target,
            boundary_valid=boundary_valid,
            kappa=args.semantic_boundary_kappa,
            dice_weight=0.5,
        )
    )

    semantic_boundary_weighted = (
        args.semantic_boundary_weight
        * semantic_boundary_dict["total"]
    )

    # 5. 总损失
    total_loss = (
        seg_loss_dict["total"]
        + boundary_weighted
        + semantic_boundary_weighted
    )

    return {
        "total": total_loss,
        "ce": seg_loss_dict["ce"],
        "dice": seg_loss_dict["dice"],
        "boundary": boundary_loss_dict["total"],
        "boundary_bce": boundary_loss_dict["bce"],
        "boundary_dice": boundary_loss_dict["dice"],
        "boundary_weighted": boundary_weighted,
        "semantic_boundary": semantic_boundary_dict["total"],
        "semantic_boundary_bce": semantic_boundary_dict["bce"],
        "semantic_boundary_dice": semantic_boundary_dict["dice"],
        "semantic_boundary_weighted": semantic_boundary_weighted,
        "semantic_boundary_probability": (
            semantic_boundary_dict["probability"]
        ),
        "boundary_target": boundary_target,
        "boundary_valid": boundary_valid,
    }



def train(
    args,
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch,
    snn_monitor,
):
    model.train()
    snn_monitor.start_phase("train", epoch)

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    dice_loss_sum = 0.0
    boundary_loss_sum = 0.0
    boundary_bce_sum = 0.0
    boundary_weighted_sum = 0.0
    semantic_boundary_loss_sum = 0.0
    semantic_boundary_bce_sum = 0.0
    semantic_boundary_dice_sum = 0.0
    semantic_boundary_weighted_sum = 0.0

    batch_count = 0
    total_batches = len(train_loader)
    print("=====> iterations per epoch:", total_batches)
    start_epoch_time = time.time()

    for iteration, batch in enumerate(train_loader):
        start_time = time.time()

        events = batch[0]
        labels = batch[1]

        events = events.to(device, non_blocking=False)
        labels = labels.long().to(device, non_blocking=False)

        snn_monitor.prepare_batch(
            iteration=iteration,
            events=events,
        )

        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(events)
        snn_monitor.end_forward()

        if not isinstance(outputs, dict):
            raise TypeError(
                "Boundary-guided SGSR must return a dict, "
                f"but got {type(outputs)}."
            )

        required_keys = {
            "seg",
            "seg_quarter",
            "boundary",
        }
        missing_keys = required_keys - set(outputs.keys())
        if missing_keys:
            raise KeyError(
                "Model output missing keys: "
                f"{sorted(missing_keys)}"
            )

        seg_logits = outputs["seg"]
        seg_logits_quarter = outputs["seg_quarter"]
        boundary_logits = outputs["boundary"]

        if seg_logits.ndim != 4:
            raise ValueError(
                "seg logits must have shape [B,C,H,W], "
                f"but got {tuple(seg_logits.shape)}."
            )

        if boundary_logits.ndim != 4:
            raise ValueError(
                "boundary logits must have shape [B,1,h,w], "
                f"but got {tuple(boundary_logits.shape)}."
            )

        if seg_logits_quarter.ndim != 4:
            raise ValueError(
                "seg_quarter must have shape [B,C,h,w], "
                f"got {tuple(seg_logits_quarter.shape)}."
            )

        if seg_logits.shape[0] != labels.shape[0]:
            raise ValueError(
                'Batch size mismatch: '
                f'seg_logits={seg_logits.shape[0]}, '
                f'labels={labels.shape[0]}.'
            )

        if seg_logits_quarter.shape[0] != labels.shape[0]:
            raise ValueError(
                "seg_quarter batch size mismatch: "
                f"logits={seg_logits_quarter.shape[0]}, "
                f"labels={labels.shape[0]}."
            )

        if seg_logits.shape[1] != args.classes:
            raise ValueError(
                'Class channel mismatch: '
                f'seg_logits={seg_logits.shape[1]}, '
                f'args.classes={args.classes}.'
            )

        if seg_logits_quarter.shape[1] != args.classes:
            raise ValueError(
                "seg_quarter class channel mismatch: "
                f"logits={seg_logits_quarter.shape[1]}, "
                f"classes={args.classes}."
            )

        if seg_logits.shape[-2:] != labels.shape[-2:]:
            raise ValueError(
                'Spatial size mismatch: '
                f'seg_logits={seg_logits.shape[-2:]}, '
                f'labels={labels.shape[-2:]}.'
            )

        if (
            seg_logits_quarter.shape[-2:]
            != boundary_logits.shape[-2:]
        ):
            raise ValueError(
                "seg_quarter and boundary logits must share "
                "the same native decoder4 resolution: "
                f"seg={seg_logits_quarter.shape[-2:]}, "
                f"boundary={boundary_logits.shape[-2:]}."
            )

        loss_dict = compute_step_losses(
            args=args,
            criterion=criterion,
            seg_logits=seg_logits,
            seg_logits_quarter=seg_logits_quarter,
            boundary_logits=boundary_logits,
            labels=labels,
        )

        total_loss = loss_dict['total']
        ce_loss = loss_dict['ce']
        dice_loss = loss_dict['dice']
        boundary_loss = loss_dict['boundary']
        boundary_bce = loss_dict['boundary_bce']
        boundary_weighted = loss_dict['boundary_weighted']
        semantic_boundary_loss = loss_dict['semantic_boundary']
        semantic_boundary_bce = loss_dict['semantic_boundary_bce']
        semantic_boundary_dice = loss_dict['semantic_boundary_dice']
        semantic_boundary_weighted = loss_dict['semantic_boundary_weighted']

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                'Non-finite total loss detected: '
                f'total={total_loss.item()}, '
                f'ce={ce_loss.item()}, '
                f'dice={dice_loss.item()}, '
                f'boundary={boundary_loss.item()}, '
                f'semantic_boundary={semantic_boundary_loss.item()}.'
            )

        total_loss.backward()

        snn_monitor.record_module_gradients()
        pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        snn_monitor.record_pre_clip_gradient(pre_clip_grad_norm)

        optimizer.step()
        functional.reset_net(model)

        if scheduler is not None and args.lr_schedule in ['poly', 'warmpoly', 'warmupcosine']:
            scheduler.step()

        total_value = float(total_loss.detach().item())
        ce_value = float(ce_loss.detach().item())
        dice_value = float(dice_loss.detach().item())
        boundary_value = float(boundary_loss.detach().item())
        boundary_bce_value = float(boundary_bce.detach().item())
        boundary_weighted_value = float(boundary_weighted.detach().item())
        semantic_boundary_value = float(
            semantic_boundary_loss.detach().item()
        )
        semantic_boundary_bce_value = float(
            semantic_boundary_bce.detach().item()
        )
        semantic_boundary_dice_value = float(
            semantic_boundary_dice.detach().item()
        )
        semantic_boundary_weighted_value = float(
            semantic_boundary_weighted.detach().item()
        )

        total_loss_sum += total_value
        ce_loss_sum += ce_value
        dice_loss_sum += dice_value
        boundary_loss_sum += boundary_value
        boundary_bce_sum += boundary_bce_value
        boundary_weighted_sum += boundary_weighted_value
        semantic_boundary_loss_sum += semantic_boundary_value
        semantic_boundary_bce_sum += semantic_boundary_bce_value
        semantic_boundary_dice_sum += semantic_boundary_dice_value
        semantic_boundary_weighted_sum += semantic_boundary_weighted_value

        batch_count += 1

        lr = optimizer.param_groups[0]['lr']
        batch_time = time.time() - start_time
        snn_monitor.record_batch_time(
            batch_time=batch_time,
            batch_size=events.shape[0],
        )

        if iteration % 10 == 0:
            print(
                "=====> epoch[%d/%d] "
                "iter:(%d/%d) "
                "lr: %.8f "
                "total: %.4f "
                "CE: %.4f "
                "SegDice: %.4f "
                "Boundary: %.4f "
                "%.2f*Boundary: %.4f "
                "SemanticBoundary: %.4f "
                "SemanticBoundaryBCE: %.4f "
                "SemanticBoundaryDice: %.4f "
                "%.2f*SemanticBoundary: %.4f "
                "time: %.2f"
                % (
                    epoch + 1,
                    args.max_epochs,
                    iteration + 1,
                    total_batches,
                    lr,
                    total_value,
                    ce_value,
                    dice_value,
                    boundary_value,
                    args.boundary_weight,
                    boundary_weighted_value,
                    semantic_boundary_value,
                    semantic_boundary_bce_value,
                    semantic_boundary_dice_value,
                    args.semantic_boundary_weight,
                    semantic_boundary_weighted_value,
                    batch_time,
                )
            )

    if batch_count == 0:
        raise RuntimeError('Training loader returned no batches.')

    epoch_time = time.time() - start_epoch_time
    remaining_time = epoch_time * (args.max_epochs - 1 - epoch)
    minutes, seconds = divmod(remaining_time, 60)
    hours, minutes = divmod(minutes, 60)
    print(
        'Remaining training time = %d hour %d minutes %d seconds'
        % (hours, minutes, seconds)
    )

    average_losses = {
        "total": total_loss_sum / batch_count,
        "ce": ce_loss_sum / batch_count,
        "dice": dice_loss_sum / batch_count,
        "boundary": boundary_loss_sum / batch_count,
        "boundary_bce": boundary_bce_sum / batch_count,
        "boundary_weighted": boundary_weighted_sum / batch_count,
        "semantic_boundary": (
            semantic_boundary_loss_sum / batch_count
        ),
        "semantic_boundary_bce": (
            semantic_boundary_bce_sum / batch_count
        ),
        "semantic_boundary_dice": (
            semantic_boundary_dice_sum / batch_count
        ),
        "semantic_boundary_weighted": (
            semantic_boundary_weighted_sum / batch_count
        ),
    }

    monitor_summary = snn_monitor.finish_phase()
    return average_losses, lr, monitor_summary



CLASS_NAMES = {
    'DDD17_events': ['flat', 'background', 'object', 'vegetation', 'human', 'vehicle'],
    'DSEC_events': ['background', 'building', 'fence', 'person', 'pole', 'road',
                    'sidewalk', 'vegetation', 'car', 'wall', 'traffic sign'],
}


def _find_monitor_metric(metrics, suffix):
    matches = [
        value for key, value in metrics.items()
        if key == suffix or key.endswith("." + suffix) or key.endswith(suffix)
    ]
    if not matches:
        return None
    return float(sum(matches) / len(matches))


def _format_snn_monitor_summary(summary, label):
    if not summary or not summary.get("enabled", False):
        return [f"SNN Monitor ({label})    : disabled"]

    metrics = summary.get("metrics", {})
    lines = [
        "SNN Monitor (%s)    : %d sampled batches"
        % (label, summary.get("sampled_batches", 0))
    ]

    first_prefix = "stem_half_local.0.pre_lif"
    first_fr = _find_monitor_metric(
        metrics,
        first_prefix + "/firing_rate",
    )
    if first_fr is not None:
        lines.append("  first neuron FR       : %.6f" % first_fr)

    first_fr_t = []
    first_th_t = []
    # Stop automatically at the first missing time step; 32 is only a safe
    # upper bound and does not assume a fixed temporal bin count.
    for t in range(32):
        fr = _find_monitor_metric(
            metrics,
            first_prefix + f"/firing_rate_t{t}",
        )
        threshold = _find_monitor_metric(
            metrics,
            first_prefix + f"/threshold_mean_t{t}",
        )
        if fr is None and threshold is None:
            if t > 0:
                break
            continue
        if fr is not None:
            first_fr_t.append("t%d=%.4f" % (t, fr))
        if threshold is not None:
            first_th_t.append("t%d=%.4f" % (t, threshold))

    if first_fr_t:
        lines.append("  first neuron FR/time  : " + ", ".join(first_fr_t))
    if first_th_t:
        lines.append("  AiLIF threshold/time : " + ", ".join(first_th_t))

    base_threshold = _find_monitor_metric(
        metrics,
        first_prefix + "/base_threshold",
    )
    adaptation_increment = _find_monitor_metric(
        metrics,
        first_prefix + "/adaptation_increment",
    )
    if base_threshold is not None and adaptation_increment is not None:
        lines.append(
            "  AiLIF base/increment : %.5f / %.5f"
            % (base_threshold, adaptation_increment)
        )

    for stage in ("DAB_Block_1", "DAB_Block_2"):
        ratio = metrics.get(f"derived/{stage}_context_local_fr_ratio")
        if ratio is not None:
            lines.append(
                "  %s ctx/local FR : %.4f"
                % (stage, ratio)
            )

    down32_fr = _find_monitor_metric(metrics, "bn_lif_5.lif/firing_rate")
    down16_std = _find_monitor_metric(metrics, "downsample_3/std")
    down32_std = _find_monitor_metric(metrics, "downsample_4/std")
    if down32_fr is not None:
        lines.append("  pre-down32 spike FR  : %.6f" % down32_fr)
    if down16_std is not None and down32_std is not None:
        lines.append(
            "  down16/down32 mem std: %.5f / %.5f"
            % (down16_std, down32_std)
        )

    correlation = metrics.get("input/density_first_firing_correlation")
    if correlation is not None:
        lines.append(
            "  density-firstFR corr : %.4f"
            % correlation
        )

    if label.lower() == "train":
        grad_norm = metrics.get("gradient/global_pre_clip")
        clip_ratio = metrics.get("gradient/clip_triggered")
        if grad_norm is not None:
            lines.append("  pre-clip grad norm   : %.5f" % grad_norm)
        if clip_ratio is not None:
            lines.append("  grad clip ratio      : %.4f" % clip_ratio)

    return lines


def format_epoch_report(
    args,
    dataset,
    epoch,
    train_losses,
    val_losses,
    mean_iou,
    best_mean_iou,
    per_class_iou,
    best_mean_acc,
    lr,
    train_monitor_summary=None,
    val_monitor_summary=None,
):
    """将验证结果格式化为：指标用百分数、每行一个指标、逐类标注类别名。"""
    class_names = CLASS_NAMES.get(dataset)
    if class_names is None or len(class_names) != len(per_class_iou):
        class_names = ["class_%d" % i for i in range(len(per_class_iou))]

    width = max((len(name) for name in class_names), default=0)

    lines = [
        "Epoch No.              : %d" % epoch,

        "Train Total Loss       : %.4f"
        % train_losses["total"],

        "Train CE Loss          : %.4f"
        % train_losses["ce"],

        "Train Seg Dice Loss    : %.4f"
        % train_losses["dice"],

        "Train 0.5*SegDice      : %.4f"
        % (0.5 * train_losses["dice"]),

        "Train Boundary BCE     : %.4f"
        % train_losses["boundary_bce"],

        "Train %.2f*Boundary    : %.4f"
        % (
            args.boundary_weight,
            train_losses["boundary_weighted"],
        ),

        "Train SemanticBoundary Total : %.4f"
        % train_losses["semantic_boundary"],

        "Train SemanticBoundary BCE   : %.4f"
        % train_losses["semantic_boundary_bce"],

        "Train SemanticBoundary Dice  : %.4f"
        % train_losses["semantic_boundary_dice"],

        "Train %.2f*SemanticBoundary  : %.4f"
        % (
            args.semantic_boundary_weight,
            train_losses["semantic_boundary_weighted"],
        ),

        "Validation Total Loss  : %.4f"
        % val_losses["total"],

        "Validation CE Loss     : %.4f"
        % val_losses["ce"],

        "Validation Seg Dice Loss : %.4f"
        % val_losses["dice"],

        "Validation 0.5*SegDice : %.4f"
        % (0.5 * val_losses["dice"]),

        "Validation Boundary BCE : %.4f"
        % val_losses["boundary_bce"],

        "Validation %.2f*Boundary : %.4f"
        % (
            args.boundary_weight,
            val_losses["boundary_weighted"],
        ),

        "Validation SemanticBoundary Total : %.4f"
        % val_losses["semantic_boundary"],

        "Validation SemanticBoundary BCE   : %.4f"
        % val_losses["semantic_boundary_bce"],

        "Validation SemanticBoundary Dice  : %.4f"
        % val_losses["semantic_boundary_dice"],

        "Validation %.2f*SemanticBoundary  : %.4f"
        % (
            args.semantic_boundary_weight,
            val_losses["semantic_boundary_weighted"],
        ),

        "Validation Boundary Precision : %.4f"
        % val_losses.get("boundary_precision", 0.0),

        "Validation Boundary Recall    : %.4f"
        % val_losses.get("boundary_recall", 0.0),

        "Validation Boundary F1        : %.4f"
        % val_losses.get("boundary_f1", 0.0),

        "Validation Boundary F1 tol=1  : %.4f"
        % val_losses.get("boundary_f1_tol1", 0.0),

        "Validation Semantic Boundary Precision : %.4f"
        % val_losses["semantic_boundary_precision"],

        "Validation Semantic Boundary Recall    : %.4f"
        % val_losses["semantic_boundary_recall"],

        "Validation Semantic Boundary F1        : %.4f"
        % val_losses["semantic_boundary_f1"],

        "Validation Semantic Boundary F1 tol=1  : %.4f"
        % val_losses["semantic_boundary_f1_tol1"],

        "mIOU(val)              : %.2f%%"
        % (mean_iou * 100),

        "best_mIOU(val)         : %.2f%%"
        % (best_mean_iou * 100),

        "best_mean_acc(val)     : %.2f%%"
        % (best_mean_acc * 100),

        "lr                     : %.8f"
        % lr,

        "current IoU per class:",
    ]
    for name, iou in zip(class_names, per_class_iou):
        lines.append("  %-*s : %.2f%%" % (width, name, float(iou) * 100))

    lines.extend(
        _format_snn_monitor_summary(
            train_monitor_summary,
            "Train",
        )
    )
    lines.extend(
        _format_snn_monitor_summary(
            val_monitor_summary,
            "Validation",
        )
    )

    return "\n".join(lines) + "\n"


def val(
    args,
    val_loader,
    model,
    criterion,
    epoch,
    snn_monitor,
):
    model.eval()
    snn_monitor.start_phase("validation", epoch)

    total_batches = len(val_loader)

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    dice_loss_sum = 0.0
    boundary_loss_sum = 0.0
    boundary_bce_sum = 0.0
    boundary_weighted_sum = 0.0
    semantic_boundary_loss_sum = 0.0
    semantic_boundary_bce_sum = 0.0
    semantic_boundary_dice_sum = 0.0
    semantic_boundary_weighted_sum = 0.0

    batch_count = 0
    nclass = args.classes

    conf = torch.zeros(nclass, nclass, dtype=torch.int64, device=device)
    boundary_counts = torch.zeros(5, dtype=torch.float64, device=device)
    semantic_boundary_counts = torch.zeros(
        5,
        dtype=torch.float64,
        device=device,
    )

    functional.reset_net(model)

    for index, batch in enumerate(val_loader):
        start_time = time.time()

        events = batch[0]
        labels = batch[1]

        events = events.to(device, non_blocking=False)
        labels = labels.long().to(device, non_blocking=False)

        snn_monitor.prepare_batch(
            iteration=index,
            events=events,
        )

        functional.reset_net(model)

        with torch.no_grad():
            outputs = model(events)
            snn_monitor.end_forward()

            if not isinstance(outputs, dict):
                raise TypeError(
                    "Boundary-guided SGSR must return a dict, "
                    f"but got {type(outputs)}."
                )

            required_keys = {
                "seg",
                "seg_quarter",
                "boundary",
            }
            missing_keys = required_keys - set(outputs.keys())
            if missing_keys:
                raise KeyError(
                    "Model output missing keys: "
                    f"{sorted(missing_keys)}"
                )

            seg_logits = outputs["seg"]
            seg_logits_quarter = outputs["seg_quarter"]
            boundary_logits = outputs["boundary"]

            if seg_logits_quarter.ndim != 4:
                raise ValueError(
                    "seg_quarter must have shape [B,C,h,w], "
                    f"got {tuple(seg_logits_quarter.shape)}."
                )

            if seg_logits_quarter.shape[0] != labels.shape[0]:
                raise ValueError(
                    "seg_quarter batch size mismatch: "
                    f"logits={seg_logits_quarter.shape[0]}, "
                    f"labels={labels.shape[0]}."
                )

            if seg_logits_quarter.shape[1] != args.classes:
                raise ValueError(
                    "seg_quarter class channel mismatch: "
                    f"logits={seg_logits_quarter.shape[1]}, "
                    f"classes={args.classes}."
                )

            if (
                seg_logits_quarter.shape[-2:]
                != boundary_logits.shape[-2:]
            ):
                raise ValueError(
                    "seg_quarter and boundary logits must share "
                    "the same native decoder4 resolution: "
                    f"seg={seg_logits_quarter.shape[-2:]}, "
                    f"boundary={boundary_logits.shape[-2:]}."
                )

            loss_dict = compute_step_losses(
                args=args,
                criterion=criterion,
                seg_logits=seg_logits,
                seg_logits_quarter=seg_logits_quarter,
                boundary_logits=boundary_logits,
                labels=labels,
            )

            boundary_target = loss_dict["boundary_target"]
            boundary_valid = loss_dict["boundary_valid"]

            boundary_pred = torch.sigmoid(boundary_logits).ge(0.5)
            boundary_gt = boundary_target.ge(0.5)
            boundary_mask = boundary_valid.gt(0.5)

            pred_valid = boundary_pred & boundary_mask
            gt_valid = boundary_gt & boundary_mask
            exact_tp = pred_valid & gt_valid

            dilated_gt = F.max_pool2d(
                gt_valid.float(),
                kernel_size=3,
                stride=1,
                padding=1,
            ).gt(0.0)
            dilated_pred = F.max_pool2d(
                pred_valid.float(),
                kernel_size=3,
                stride=1,
                padding=1,
            ).gt(0.0)

            boundary_counts += torch.stack(
                [
                    exact_tp.sum(),
                    pred_valid.sum(),
                    gt_valid.sum(),
                    (pred_valid & dilated_gt).sum(),
                    (gt_valid & dilated_pred).sum(),
                ]
            ).to(boundary_counts.dtype)

            semantic_boundary_probability_map = (
                loss_dict["semantic_boundary_probability"]
            )
            semantic_boundary_pred = (
                semantic_boundary_probability_map.ge(0.5)
            )
            semantic_boundary_gt = boundary_target.ge(0.5)
            semantic_boundary_mask = boundary_valid.gt(0.5)

            semantic_pred_valid = (
                semantic_boundary_pred & semantic_boundary_mask
            )
            semantic_gt_valid = (
                semantic_boundary_gt & semantic_boundary_mask
            )
            semantic_exact_tp = (
                semantic_pred_valid & semantic_gt_valid
            )

            semantic_dilated_gt = F.max_pool2d(
                semantic_gt_valid.float(),
                kernel_size=3,
                stride=1,
                padding=1,
            ).gt(0.0)
            semantic_dilated_pred = F.max_pool2d(
                semantic_pred_valid.float(),
                kernel_size=3,
                stride=1,
                padding=1,
            ).gt(0.0)

            semantic_boundary_counts += torch.stack(
                [
                    semantic_exact_tp.sum(),
                    semantic_pred_valid.sum(),
                    semantic_gt_valid.sum(),
                    (
                        semantic_pred_valid
                        & semantic_dilated_gt
                    ).sum(),
                    (
                        semantic_gt_valid
                        & semantic_dilated_pred
                    ).sum(),
                ]
            ).to(semantic_boundary_counts.dtype)

            pred = seg_logits.argmax(dim=1)

        functional.reset_net(model)

        total_value = float(loss_dict['total'].detach().item())
        ce_value = float(loss_dict['ce'].detach().item())
        dice_value = float(loss_dict['dice'].detach().item())
        boundary_value = float(loss_dict['boundary'].detach().item())
        boundary_bce_value = float(
            loss_dict['boundary_bce'].detach().item()
        )
        boundary_weighted_value = float(
            loss_dict['boundary_weighted'].detach().item()
        )
        semantic_boundary_value = float(
            loss_dict['semantic_boundary'].detach().item()
        )
        semantic_boundary_bce_value = float(
            loss_dict['semantic_boundary_bce'].detach().item()
        )
        semantic_boundary_dice_value = float(
            loss_dict['semantic_boundary_dice'].detach().item()
        )
        semantic_boundary_weighted_value = float(
            loss_dict['semantic_boundary_weighted'].detach().item()
        )

        total_loss_sum += total_value
        ce_loss_sum += ce_value
        dice_loss_sum += dice_value
        boundary_loss_sum += boundary_value
        boundary_bce_sum += boundary_bce_value
        boundary_weighted_sum += boundary_weighted_value
        semantic_boundary_loss_sum += semantic_boundary_value
        semantic_boundary_bce_sum += semantic_boundary_bce_value
        semantic_boundary_dice_sum += semantic_boundary_dice_value
        semantic_boundary_weighted_sum += semantic_boundary_weighted_value

        batch_count += 1

        valid = (
            (labels != args.ignore_label)
            & (labels >= 0)
            & (labels < nclass)
        )
        index_tensor = labels[valid] * nclass + pred[valid]
        conf += torch.bincount(
            index_tensor, minlength=nclass * nclass
        ).reshape(nclass, nclass)

        batch_time = time.time() - start_time
        snn_monitor.record_batch_time(
            batch_time=batch_time,
            batch_size=events.shape[0],
        )

        print(
            "[%d/%d] "
            "total: %.4f "
            "CE: %.4f "
            "SegDice: %.4f "
            "Boundary: %.4f "
            "SemanticBoundary: %.4f "
            "SemanticBoundaryBCE: %.4f "
            "SemanticBoundaryDice: %.4f "
            "%.2f*SemanticBoundary: %.4f "
            "time: %.2f"
            % (
                index + 1,
                total_batches,
                total_value,
                ce_value,
                dice_value,
                boundary_value,
                semantic_boundary_value,
                semantic_boundary_bce_value,
                semantic_boundary_dice_value,
                args.semantic_boundary_weight,
                semantic_boundary_weighted_value,
                batch_time,
            )
        )

    if batch_count == 0:
        raise RuntimeError('Validation loader returned no batches.')

    conf = conf.cpu().numpy()
    mean_iou, per_class_iou, mean_acc = scores_from_confmat(conf, nclass)

    (
        boundary_tp,
        boundary_pred_positive,
        boundary_gt_positive,
        boundary_precision_match_tol1,
        boundary_recall_match_tol1,
    ) = boundary_counts.cpu().tolist()

    boundary_precision = boundary_tp / max(boundary_pred_positive, 1.0)
    boundary_recall = boundary_tp / max(boundary_gt_positive, 1.0)
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall
        / max(boundary_precision + boundary_recall, 1e-12)
    )
    boundary_precision_tol1 = (
        boundary_precision_match_tol1
        / max(boundary_pred_positive, 1.0)
    )
    boundary_recall_tol1 = (
        boundary_recall_match_tol1
        / max(boundary_gt_positive, 1.0)
    )
    boundary_f1_tol1 = (
        2.0 * boundary_precision_tol1 * boundary_recall_tol1
        / max(boundary_precision_tol1 + boundary_recall_tol1, 1e-12)
    )

    (
        semantic_tp,
        semantic_pred_positive,
        semantic_gt_positive,
        semantic_precision_match_tol1,
        semantic_recall_match_tol1,
    ) = semantic_boundary_counts.cpu().tolist()

    semantic_boundary_precision = (
        semantic_tp / max(semantic_pred_positive, 1.0)
    )
    semantic_boundary_recall = (
        semantic_tp / max(semantic_gt_positive, 1.0)
    )
    semantic_boundary_f1 = (
        2.0
        * semantic_boundary_precision
        * semantic_boundary_recall
        / max(
            semantic_boundary_precision + semantic_boundary_recall,
            1e-12,
        )
    )
    semantic_boundary_precision_tol1 = (
        semantic_precision_match_tol1
        / max(semantic_pred_positive, 1.0)
    )
    semantic_boundary_recall_tol1 = (
        semantic_recall_match_tol1
        / max(semantic_gt_positive, 1.0)
    )
    semantic_boundary_f1_tol1 = (
        2.0
        * semantic_boundary_precision_tol1
        * semantic_boundary_recall_tol1
        / max(
            semantic_boundary_precision_tol1
            + semantic_boundary_recall_tol1,
            1e-12,
        )
    )

    average_losses = {
        "total": total_loss_sum / batch_count,
        "ce": ce_loss_sum / batch_count,
        "dice": dice_loss_sum / batch_count,
        "boundary": boundary_loss_sum / batch_count,
        "boundary_bce": boundary_bce_sum / batch_count,
        "boundary_weighted": boundary_weighted_sum / batch_count,
        "semantic_boundary": (
            semantic_boundary_loss_sum / batch_count
        ),
        "semantic_boundary_bce": (
            semantic_boundary_bce_sum / batch_count
        ),
        "semantic_boundary_dice": (
            semantic_boundary_dice_sum / batch_count
        ),
        "semantic_boundary_weighted": (
            semantic_boundary_weighted_sum / batch_count
        ),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "boundary_f1_tol1": boundary_f1_tol1,
        "semantic_boundary_precision": semantic_boundary_precision,
        "semantic_boundary_recall": semantic_boundary_recall,
        "semantic_boundary_f1": semantic_boundary_f1,
        "semantic_boundary_f1_tol1": semantic_boundary_f1_tol1,
    }

    monitor_summary = snn_monitor.finish_phase()
    return (
        mean_iou,
        per_class_iou,
        mean_acc,
        average_losses,
        monitor_summary,
    )



if __name__ == '__main__':

    start = timeit.default_timer()
    args = parse_args()

    if args.dataset == 'DDD17_events':
        args.classes = 6
        args.input_size = '200,346'
        args.ignore_label = 255
    elif args.dataset == 'DSEC_events':
        args.classes = 11
        args.input_size = '440,640'
        args.ignore_label = 255
    else:
        raise ValueError(
            'Supported datasets are DDD17_events and DSEC_events, '
            f'but got {args.dataset}.'
        )

    train_model(args)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    print("training time: %d hour %d minutes" % (int(hour), int(minute)))
