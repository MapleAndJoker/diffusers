# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..loaders import PatchedLoraProjection, text_encoder_attn_modules, text_encoder_mlp_modules
from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def adjust_lora_scale_text_encoder(text_encoder, lora_scale: float = 1.0):
    for _, attn_module in text_encoder_attn_modules(text_encoder):
        if isinstance(attn_module.q_proj, PatchedLoraProjection):
            attn_module.q_proj.lora_scale = lora_scale
            attn_module.k_proj.lora_scale = lora_scale
            attn_module.v_proj.lora_scale = lora_scale
            attn_module.out_proj.lora_scale = lora_scale

    for _, mlp_module in text_encoder_mlp_modules(text_encoder):
        if isinstance(mlp_module.fc1, PatchedLoraProjection):
            mlp_module.fc1.lora_scale = lora_scale
            mlp_module.fc2.lora_scale = lora_scale


class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        #在这加M层
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        #M层初始化

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        #数据类型保持一致
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        #加入M层
        up_hidden_states = self.up(down_hidden_states)

        #up_hidden_states 乘以缩放因子network_alpha /rank
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class TriLoRALinearLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mid_rank_in: int = 4,
        mid_rank_out: int = 4,
 #       rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, mid_rank_in, bias=False, device=device, dtype=dtype)
        #在这加M层
        self.middle = nn.Linear(mid_rank_in, mid_rank_out, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(mid_rank_out, out_features, bias=False, device=device, dtype=dtype)

        self.network_alpha = network_alpha
        self.mid_rank_in = mid_rank_in
        self.mid_rank_out = mid_rank_out
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / mid_rank_in)
        nn.init.normal_(self.middle.weight, std=1 / mid_rank_out)
        nn.init.zeros_(self.up.weight)
        #M层初始化

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        #数据类型保持一致
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        middle_hidden_states = self.middle(down_hidden_states)
        #加入M层
        up_hidden_states = self.up(middle_hidden_states)

        average_rank = (self.mid_rank_in + self.mid_rank_out) / 2

        #up_hidden_states 乘以缩放因子
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / average_rank

        return up_hidden_states.to(orig_dtype)


class LoRAConv2dLayer(nn.Module):
    r"""
    A convolutional layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        kernel_size (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The kernel size of the convolution.
        stride (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The stride of the convolution.
        padding (`int` or `tuple` of two `int` or `str`, `optional`, defaults to 0):
            The padding of the convolution.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
    ):
        super().__init__()

        self.down = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        #加M层
        self.up = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        fusion = fusion.reshape((w_orig.shape))
        '''
        在卷积层中，权重通常是多维的
        对于 nn.Conv2d，权重的形状是 [输出通道数, 输入通道数, 高度, 宽度]
        为了进行矩阵乘法，需要将这些多维权重转换为二维矩阵。这是因为矩阵乘法是定义在二维矩阵上的操作。
        flatten(start_dim=1) 将每个权重张量从其第二个维度开始展平，转换为二维矩阵。
        这样做保留了权重的“输出通道”维度，而将其他维度（输入通道、高度、宽度）展平。
        '''
        fused_weight = w_orig + (lora_scale * fusion)

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        self.w_up = self.w_up.to(device=device).float()
        self.w_down = self.w_down.to(device).float()

        fusion = torch.mm(self.w_up.flatten(start_dim=1), self.w_down.flatten(start_dim=1))
        fusion = fusion.reshape((fused_weight.shape))
        unfused_weight = fused_weight.float() - (self._lora_scale * fusion)
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            original_outputs = F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return original_outputs + (scale * self.lora_layer(hidden_states))


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        #  LoRA 权重融合
        #加入M矩阵要改
        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        '''
        批量矩阵乘法（torch.bmm）
        w_up[None, :] 和 w_down[None, :] 在 w_up 和 w_down 前添加一个新的批次维度来准备的
        即使它们是单个矩阵，也能被 torch.bmm 接受作为批次矩阵乘法的输入。
        torch.bmm 返回的结果是一个批次的矩阵乘积，即使只有一个乘法操作。
        因此，结果将是一个形状为 [1, x, y] 的张量
        我们使用 [0] 来提取这个批次中的第一个元素，这将是一个形状为 [x, y] 的二维张量。
        '''

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return
        # 只有在 LoRA 权重实际存在的情况下，才会尝试进行解融合操作
        # M层在此处检查
        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        # 从融合后的权重中减去 U 和 D 权重乘积的影响
        #再减去M层
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)
        
        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            #传到LoRALinearLayer的forward方法中
            return out
