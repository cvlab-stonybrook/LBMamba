import json
import os

import math
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET

import projects.ViTDet

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models_mamba_bi",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../vim/"),
)

BiVisionMamba: nn.Module = build.BiVisionMamba

@MODELS_MMDET.register_module()
class Backbone_BiVisionMamba(BiVisionMamba):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 embed_dim=192,
                 out_indices=(23,),
                 pretrained_ckpt=None,
                 out_dim=None,
                 **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, **kwargs)

        del self.head
        del self.norm_f
        self.patch_size = patch_size

        if not isinstance(out_indices, (list, tuple)):
            out_indices = (out_indices,)

        if out_dim is not None and out_dim != embed_dim:
            self.out_map = True
            for i in range(len(out_indices)):
                layer = nn.Linear(embed_dim, out_dim)
                layer_name = f'out_map_{i}'
                self.add_module(layer_name, layer)
        else:
            self.out_map = False

        out_dim = out_dim if out_dim else embed_dim

        self.out_indices = out_indices
        for i in range(len(out_indices)):
            layer = nn.LayerNorm(out_dim)
            layer_name = f'outnorm_{i}'
            self.add_module(layer_name, layer)

        self.load_pretrained(pretrained_ckpt)

        # drop the pos embed for class token
        if self.if_cls_token:
            print("Dropping cls token")
            del self.cls_token
            # self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])

    def load_pretrained(self, ckpt):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)
        if ckpt is None:
            return
        print(f'Load backbone state dict from {ckpt}')
        if ckpt.startswith('http'):
            from mmengine.utils.dl_utils import load_url
            state_dict = load_url(ckpt, map_location='cpu')["model"] #['state_dict']
        else:
            state_dict = torch.load(ckpt, map_location='cpu')["model"] #['state_dict']
        if 'pos_embed' in state_dict:
            pos_size = int(math.sqrt(state_dict['pos_embed'].shape[1]))
            L = state_dict['pos_embed'].shape[1]
            if L % 2 != 0:
                if self.use_middle_cls_token:
                    state_dict['pos_embed'] = torch.cat((state_dict['pos_embed'][:, :L // 2, :],
                                                         state_dict['pos_embed'][:, L // 2 + 1:, :]), 1)
                else:
                    state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:, :]

            state_dict['pos_embed'] = self.resize_pos_embed(
                state_dict['pos_embed'],
                self.token_size,
                (pos_size, pos_size),
                'bicubic'
            )
        if 'rope.freqs_cos' in state_dict:
            pos_size = int(math.sqrt(state_dict['rope.freqs_cos'].shape[0]))
            state_dict['rope.freqs_cos'] = self.resize_pos_embed(
                state_dict['rope.freqs_cos'].unsqueeze(0),
                self.token_size,
                (pos_size, pos_size),
                'bicubic'
            )[0]
        if 'rope.freqs_cos' in state_dict:
            pos_size = int(math.sqrt(state_dict['rope.freqs_sin'].shape[0]))
            state_dict['rope.freqs_sin'] = self.resize_pos_embed(
                state_dict['rope.freqs_sin'].unsqueeze(0),
                self.token_size,
                (pos_size, pos_size),
                'bicubic'
            )[0]
        res = self.load_state_dict(state_dict, strict=False)
        print(res)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        from mmseg.models.utils import resize
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        pos_embed_weight = pos_embed
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return pos_embed_weight

    def forward_features(self, x, inference_params=None):
        B, C, H_img, W_img = x.shape
        # x, (Hp, Wp) = self.patch_embed(x)
        x = self.patch_embed(x)

        batch_size, seq_len, _ = x.size()
        # Hp = Wp = int(math.sqrt(seq_len))
        H, W = math.ceil(H_img / self.patch_size), math.ceil(W_img / self.patch_size)

        if self.pos_embed is not None:
            if H != self.token_size[0] or W != self.token_size[1]:
                # downstream tasks such as det and seg may have various input resolutions
                pos_embed = self.resize_pos_embed(self.pos_embed, (H, W), self.token_size,
                                                                       'bicubic')
                if self.if_rope:
                    freqs_cos = self.resize_pos_embed(self.rope.freqs_cos.unsqueeze(0),
                                                      (H, W), self.token_size,
                                                      'bicubic')[0]
                    freqs_sin = self.resize_pos_embed(self.rope.freqs_sin.unsqueeze(0),
                                                      (H, W),self.token_size,
                                                      'bicubic')[0]
            else:
                pos_embed = self.pos_embed
                freqs_cos = None
                freqs_sin = None
            x = x + pos_embed
            x = self.pos_drop(x)

        residual = None
        hidden_states = x
        features = []
        for layer_idx, layer in enumerate(self.layers):

            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

            # When i + 1 is even (e.g. 24th layer),  the sequence is reversed now,
            # thus we need to flip the sequence to get the original order
            # before saving it to the feature list
            if self.layerwise_reverse_sq and (layer_idx + 1) % 2 == 0:
                hidden_states = hidden_states.flip([1])
                residual = residual.flip([1]) if residual is not None else None

            if self.out_indices is not None and layer_idx in self.out_indices:
                features.append(hidden_states)
            # When i + 1 is odd (e.g. 1th layer), the sequence is in its original order,
            # thus we need to flip the sequence after saving it to the feature list
            if self.layerwise_reverse_sq and (layer_idx + 1) % 2 != 0:
                hidden_states = hidden_states.flip([1])
                residual = residual.flip([1]) if residual is not None else None

        assert len(features) == len(self.out_indices)
        return features, (H, W)

    def forward(self, x):
        C = self.embed_dim
        outs, (H, W) = self.forward_features(x)
        if self.out_map:
            outs = [getattr(self, f'out_map_{i}')(o) for i, o in enumerate(outs)]
            C = outs[0].shape[-1]
        outs = [getattr(self, f'outnorm_{i}')(o) for i, o in enumerate(outs)]
        outs = [o.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous() for o in outs]
        if len(self.out_indices) == 1:
            return outs[0]
        return outs


@MODELS_MMDET.register_module()
class MM_BiVim(BaseModule, Backbone_BiVisionMamba):
    def __init__(self, *args, **kwargs):
        Backbone_BiVisionMamba.__init__(self, *args, **kwargs)
        self._is_init = True
        self.init_cfg = None



# from mmengine.dist import get_dist_info
# from mmengine.logging import print_log
# from mmengine.optim import DefaultOptimWrapperConstructor
#
# from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS
#
#
# @OPTIM_WRAPPER_CONSTRUCTORS.register_module()
# class VimLayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
#     def add_params(self, params, module, prefix='', is_dcn_module=None):
#         """Add all parameters of module to the params list.
#         The parameters of the given module will be added to the list of param
#         groups, with specific rules defined by paramwise_cfg.
#         Args:
#             params (list[dict]): A list of param groups, it will be modified
#                 in place.
#             module (nn.Module): The module to be added.
#             prefix (str): The prefix of the module
#             is_dcn_module (int|float|None): If the current module is a
#                 submodule of DCN, `is_dcn_module` will be passed to
#                 control conv_offset layer's learning rate. Defaults to None.
#         """
#         parameter_groups = {}
#         print(self.paramwise_cfg)
#         num_layers = self.paramwise_cfg.get('num_layers') + 2
#         layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
#         print("Build LayerDecayOptimizerConstructor %f - %d" % (layer_decay_rate, num_layers))
#         weight_decay = self.base_wd
#
#         for name, param in module.named_parameters():
#             if not param.requires_grad:
#                 continue  # frozen weights
#             if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token'):
#                 group_name = "no_decay"
#                 this_weight_decay = 0.
#             else:
#                 group_name = "decay"
#                 this_weight_decay = weight_decay
#
#             layer_id = get_num_layer_for_vit(name, num_layers)
#             group_name = "layer_%d_%s" % (layer_id, group_name)
#
#             if group_name not in parameter_groups:
#                 scale = layer_decay_rate ** (num_layers - layer_id - 1)
#
#                 parameter_groups[group_name] = {
#                     "weight_decay": this_weight_decay,
#                     "params": [],
#                     "param_names": [],
#                     "lr_scale": scale,
#                     "group_name": group_name,
#                     "lr": scale * self.base_lr,
#                 }
#
#             parameter_groups[group_name]["params"].append(param)
#             parameter_groups[group_name]["param_names"].append(name)
#         rank, _ = get_dist_info()
#         if rank == 0:
#             to_display = {}
#             for key in parameter_groups:
#                 to_display[key] = {
#                     "param_names": parameter_groups[key]["param_names"],
#                     "lr_scale": parameter_groups[key]["lr_scale"],
#                     "lr": parameter_groups[key]["lr"],
#                     "weight_decay": parameter_groups[key]["weight_decay"],
#                 }
#             print("Param groups = %s" % json.dumps(to_display, indent=2))
#
#         # state_dict = module.state_dict()
#         # for group_name in parameter_groups:
#         #     group = parameter_groups[group_name]
#         #     for name in group["param_names"]:
#         #         group["params"].append(state_dict[name])
#         params.extend(parameter_groups.values())
#
# def get_num_layer_for_vit(var_name, num_max_layer):
#     if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
#         return 0
#     elif var_name.startswith("backbone.patch_embed"):
#         return 0
#     elif var_name.startswith("backbone.blocks"):
#         layer_id = int(var_name.split('.')[2])
#         return layer_id + 1
#     elif var_name.startswith("backbone.layers"):
#         layer_id = int(var_name.split('.')[2])
#         return layer_id + 1
#     else:
#         return num_max_layer - 1