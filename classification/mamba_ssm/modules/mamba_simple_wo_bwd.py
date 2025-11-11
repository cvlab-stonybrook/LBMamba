# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from ..ops.selective_scan_interface import selective_scan_fn
except ImportError:
    raise ImportError("Could not find selective_scan_interface")

try:
    # split the following imports to multiple lines and import every one as lbm_xxx
    from ..ops.selective_scan_interface_lbm import selective_scan_fn as lbm_selective_scan_fn
except ImportError:
    raise ImportError("Could not find selective_scan_interface_lbm")

try:
    from ..ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from ..ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Not used
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_divide_out=False,
        init_layer_scale=None,
        lbm=False,
        use_norm_after_ssm=False,
        attention_C=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.bimamba_type = bimamba_type
        self.if_divide_out = if_divide_out
        self.attention_C = attention_C

        # LBM added
        self.lbm = lbm
        if lbm:
            self.selective_scan_fn = lbm_selective_scan_fn
        else:
            self.selective_scan_fn = selective_scan_fn

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.use_norm_after_ssm = use_norm_after_ssm
        
        if self.use_norm_after_ssm is True:
            self.layernorm = nn.LayerNorm(self.d_inner)
            
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "none":
            pass
        elif bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
        else:
            raise NotImplementedError

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        x_inp, z = xz.chunk(2, dim=1)

        
        assert self.activation in ["silu", "swish"]
        x = causal_conv1d_fn(
            x=x_inp,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        if self.attention_C:
            C = torch.softmax(C, dim=-1)
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]

        if self.bimamba_type == "none":
            out = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            out = self.out_proj(rearrange(out, "b d l -> b l d"))
        elif self.bimamba_type == "v2":

            A_b = -torch.exp(self.A_b_log.float())  # (d_inner, d_state)
            x_flip_inp = x_inp.flip([-1])
            z = F.silu(rearrange(z, "b d l -> b l d"))

            out = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )

            x_flip = causal_conv1d_fn(
                x=x_flip_inp,
                weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                bias=self.conv1d_b.bias,
                activation=self.activation,
            )

            x_dbl_flip = self.x_proj_b(rearrange(x_flip, "b d l -> (b l) d"))  # (bl d)
            dt_flip, B_flip, C_flip = torch.split(x_dbl_flip, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_flip = self.dt_proj_b.weight @ dt_flip.t()
            dt_flip = rearrange(dt_flip, "d (b l) -> b d l", l=seqlen)
            B_flip = rearrange(B_flip, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            if self.attention_C:
                C_flip = torch.softmax(C_flip, dim=-1)
            C_flip = rearrange(C_flip, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            out_b = selective_scan_fn(
                x_flip,
                dt_flip,
                A_b,
                B_flip,
                C_flip,
                self.D_b.float(),
                z=None,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
        
            if self.use_norm_after_ssm is True:
                out = F.linear((self.layernorm(rearrange(out/2 + out_b.flip([-1])/2, "b d l -> b l d")))*z, self.out_proj.weight, self.out_proj.bias)

            else:
                out = F.linear((rearrange(out/2 + out_b.flip([-1])/2, "b d l -> b l d"))*z, self.out_proj.weight, self.out_proj.bias)
        else:
            raise NotImplementedError
        return out
