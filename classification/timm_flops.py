#!/usr/bin/env python
"""
deit_flops.py – Compute params & FLOPs for DeiT models in timm.

Example
-------
# Tiny DeiT at the default 224×224 resolution
python deit_flops.py --model deit_tiny_patch16_224

# Same model at 384×384
python deit_flops.py -m deit_tiny_patch16_224 -s 384

# Get a per-layer table
python deit_flops.py -m deit_base_patch16_224 --detail
"""
import argparse
import timm
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table   # pip install fvcore

def get_args():
    p = argparse.ArgumentParser(description="FLOPs calculator for DeiT models")
    p.add_argument("-m", "--model", default="deit_tiny_patch16_224",
                   help="Model name recognised by timm.create_model()")
    p.add_argument("-s", "--img-size", type=int, default=256,
                   help="Input resolution (square).")
    p.add_argument("-b", "--batch-size", type=int, default=1)
    p.add_argument("-c", "--channels",   type=int, default=3)
    p.add_argument("--detail", action="store_true",
                   help="Print a per-module FLOP table.")
    return p.parse_args()

def main():
    args   = get_args()
    device = "cpu"          # FLOP counting is device-independent
    model  = timm.create_model(args.model,
                               pretrained=False,
                               num_classes=1000,
                               img_size=args.img_size,
                               drop_path_rate = 0.05,
                               ).to(device).eval()

    dummy  = torch.zeros(args.batch_size,
                         args.channels,
                         args.img_size,
                         args.img_size,
                         device=device)

    flops  = FlopCountAnalysis(model, dummy)
    macs   = flops.total()          # fvcore counts a fused multiply-add as 1 FLOP
    params = sum(p.numel() for p in model.parameters())

    print("-" * 60)
    print(f"Model  : {args.model}")
    print(f"Input  : {args.batch_size}×{args.channels}×{args.img_size}×{args.img_size}")
    print(f"Params : {params/1e6: .2f} M")
    print(f"FLOPs  : {macs/1e9  : .2f} G")
    if args.detail:
        print(flop_count_table(flops, max_depth=2))
    print("-" * 60)

if __name__ == "__main__":
    main()
