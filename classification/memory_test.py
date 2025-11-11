#!/usr/bin/env python3
"""
gpu_mem_search.py

Estimate the minimum GPU memory a model needs for a given input size.

Requires:
  * PyTorch ≥ 1.9 (for torch.cuda.set_per_process_memory_fraction)
  * A CUDA-enabled GPU

Usage example:
    python gpu_mem_search.py --model resnet50 \
                             --height 224 --width 224 \
                             --batch 8 --tol 25
"""

import argparse
import contextlib
import sys
import time

import torch

import models_mamba
import models_mamba_bi
import models_mamba_bi_head

from timm.models import create_model



@contextlib.contextmanager
def limited_memory(fraction: float, device: int = 0):
    """
    Context manager that restricts the process to a fraction of GPU memory.

    NOTE:  Setting the limit repeatedly in a loop is officially supported.
           We still clear caches aggressively to avoid artefacts.
    """
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    try:
        yield
    finally:
        # No limit after we leave the block
        torch.cuda.set_per_process_memory_fraction(1.0, device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def try_run(model, inp, use_backward: bool = False) -> bool:
    """
    Try forward (+ optional backward) pass.  Return True if it succeeds.
    """
    try:
        with torch.no_grad() if not use_backward else contextlib.nullcontext():
            out = model(inp)
            if use_backward:
                loss = out.mean()
                loss.backward()
        torch.cuda.synchronize()      # make sure kernels really ran
        return True
    except RuntimeError as e:
        # The canonical OOM message includes this substring
        if "CUDA out of memory" in str(e):
            return False
        raise  # some other error


def search_memory(model, batch, c, h, w, tol_mb=25, use_backward=False, gpu_id=0):
    """
    Binary-search for minimal usable memory.
    Returns memory_in_MB_needed.
    """
    device_prop = torch.cuda.get_device_properties(gpu_id)
    total_mb = device_prop.total_memory / (1024 ** 2)

    low_mb = 0
    high_mb = total_mb  # upper bound

    # Synthetic input once, re-used at every step (keeps host cost tiny)
    inp = torch.randn(batch, c, h, w).cuda()

    best_mb = high_mb  # track smallest successful value

    # convert to fractions because API uses fractions
    while (high_mb - low_mb) > tol_mb:
        mid_mb = (high_mb + low_mb) / 2
        frac = mid_mb / total_mb

        torch.cuda.empty_cache()
        with limited_memory(frac, gpu_id):
            ok = try_run(model, inp, use_backward)

        if ok:
            print(f"{mid_mb:.2f} MB needed")
            best_mb = mid_mb
            high_mb = mid_mb          # tighten upper bound
        else:
            print(f"{mid_mb:.2f} MB not enough")
            low_mb = mid_mb           # raise lower bound

    return round(best_mb, 2)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Binary-search GPU memory usage.")
    parser.add_argument("--model", default="resnet50",
                        help="Any available model")
    parser.add_argument("--batch", type=int, default=128, help="Batch size.")
    parser.add_argument("--height", type=int, default=256, help="Image height.")
    parser.add_argument("--width", type=int, default=256, help="Image width.")
    parser.add_argument("--channels", type=int, default=3, help="Input channels.")
    parser.add_argument("--tol", type=float, default=1,
                        help="Tolerance (MB) at which to stop searching.")
    parser.add_argument("--backward", action="store_true",
                        help="Also run backward pass (gradients).")
    # add gpu id parameter
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    args = parser.parse_args()

    torch.manual_seed(0)

    # set cuda device
    torch.cuda.set_device(args.gpu_id)

    print(f"Building model '{args.model}' …", flush=True)
    # 加载模型
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        img_size=args.height,
        drop_path_rate=0.05,
    ).cuda()

    print(f"Searching memory need for "
          f"batch={args.batch}×{args.channels}×{args.height}×{args.width} …")
    start = time.time()
    need_mb = search_memory(
        model, args.batch, args.channels, args.height, args.width,
        tol_mb=args.tol, use_backward=args.backward, gpu_id=args.gpu_id,
    )
    dur = time.time() - start

    print(f"\n≈ {need_mb} MB of GPU memory needed "
          f"(stop-tol {args.tol} MB, time {dur:.1f}s).")


if __name__ == "__main__":
    # quick sanity check for CUDA presence
    if not torch.cuda.is_available():
        sys.exit("CUDA device not found — please run on a GPU machine.")
    main()
