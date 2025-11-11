import random
from pprint import pprint

import math
import torch

from einops import rearrange
import torch.nn.functional as F
from networkx.generators.trees import prefix_tree

from models_valid.pscan import pscan
from models_valid.pscan_bi import pscan_bi
import lbm.pscan_cuda.pscan as selective_scan_cuda


def random_initialize(BB, LL, DD, NN, dt_proj):
    u = torch.randn(BB, DD, LL).cuda()  # x
    delta = torch.randn(BB, DD, LL).cuda()
    A = torch.randn(DD, NN).cuda()
    B = torch.randn(BB, NN, LL).cuda()
    C = torch.randn(DD, NN, LL).cuda()
    D = torch.randn(DD).cuda()
    z = torch.randn(BB, DD, LL).cuda()
    delta_bias = dt_proj.bias.float()
    # return all these values
    return u, delta, A, B, C, D, z, delta_bias


def one_initialize(BB, LL, DD, NN, dt_proj):
    u = torch.ones(BB, DD, LL, dtype=torch.float).cuda()  # x
    delta = torch.ones(BB, DD, LL, dtype=torch.float).cuda()
    # delta = torch.arange(delta.numel()).float().reshape(BB, DD, LL).cuda()
    A = torch.ones(DD, NN, dtype=torch.float).cuda()
    B = torch.ones(BB, NN, LL, dtype=torch.float).cuda()
    C = torch.ones(DD, NN, LL, dtype=torch.float).cuda()
    D = None  # torch.ones(DD, dtype=torch.float).cuda()
    z = None  # torch.zeros(BB, DD, LL, dtype=torch.float).cuda()
    delta_bias = dt_proj.bias.float() - dt_proj.bias.float()
    # return all these values
    return u, delta, A, B, C, D, z, delta_bias


def python_mamba(u, delta, A, B, C, D, z, delta_bias, delta_softplus, exp=True):
    def selective_scan(x, delta, A, B, exp=True):
        # A : (ED, N)
        # hs : (B, L, ED, N)
        if exp:
            deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        else:
            deltaA = delta.unsqueeze(-1) * A  # (B, L, ED, N)

        if B is None:
            BX = x  # already (B, L, ED, N)
        else:
            deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
            BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan_bi(deltaA, deltaA, BX, len_b=8)

        return hs

    x = u
    x = x.transpose(1, 2)
    B = B.transpose(1, 2)
    C = C.transpose(1, 2)
    if z is not None:
        z = z.transpose(1, 2)

    delta = delta.transpose(1, 2)
    delta = delta + delta_bias
    if delta_softplus:
        delta = F.softplus(delta)

    # prepare for h scan
    # x : (B, L, ED)
    # Δ : (B, L, ED)
    # B : (B, L, N)
    hs = selective_scan(x, delta, A, B, exp=exp)
    # h: (B, L, ED, N)

    # C : (B, L, N)
    # D : (ED)
    y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    if D is not None:
        y = y + D * x

    output = y
    if z is not None:
        z = F.silu(z)
        output = output * z
    return output


def validate(debug=True):
    BB, LL, DD, NN = 1, 256, 1, 1

    d_model = DD
    d_inner = d_model
    dt_proj = torch.nn.Linear(DD, d_inner, bias=True).cuda()

    u, delta, A, B, C, D, z, delta_bias = random_initialize(BB, LL, DD, NN, dt_proj)
    # u, delta, A, B, C, D, z, delta_bias = one_initialize(BB, LL, DD, NN, dt_proj)
    delta_softplus = True
    use_exp = True

    # print the same of u, A, B, C, D, z, delta_bias
    if debug:
        print('u:', u.shape)
        print('delta:', delta.shape)
        print('A:', A.shape)
        print('B:', B.shape)
        print('C:', C.shape)
        if D is not None:
            print('D:', D.shape)
        if z is not None:
            print('z:', z.shape)
        print('delta_bias:', delta_bias.shape)

    from time import perf_counter

    start = perf_counter()
    out_python = python_mamba(u, delta, A, B, C, D, z, delta_bias, delta_softplus, exp=use_exp)
    stop = perf_counter()
    if debug:
        print('python out.shape:', out_python.shape)

    B = rearrange(B, "b dstate l -> b 1 dstate l")
    C = rearrange(C, "b dstate l -> b 1 dstate l")

    start = perf_counter()
    out, _, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
    stop = perf_counter()
    # print('mamba cuda', stop - start)

    if z is not None:
        out = rest[0]
    # swap the dim of out (1 and 2)
    out = rearrange(out, "b dstate l -> b l dstate")
    if debug:
        print('cuda out.shape:', out.shape)

    rtol, atol = (6e-4, 2e-3)
    all_close = torch.allclose(out, out_python, rtol=rtol, atol=atol, equal_nan=True)
    # assert all_close, "cuda out != python out"

    if debug:
        print(f'Result all close? {all_close}')
        print("out:", out)
        print("out_python:", out_python)
        # if not all_close:
        #     print('rel diff:')
        #     rel_diff = torch.abs(out - out_python) / out_python
        #     rel_diff[torch.isnan(rel_diff)] = 0
        #     pprint(rel_diff.cpu().tolist())
        #     print('max rel diff', torch.argmax(rel_diff), torch.max(rel_diff))

    return all_close


if __name__ == '__main__':
    debug = True
    # debug = False
    test_time = 100
    if debug:
        test_time = 1

    for _ in range(test_time):
        # random_seed = random.randint(0, int(1e9))
        random_seed = 246233883
        torch.cuda.manual_seed(random_seed)
        torch.manual_seed(random_seed)

        valid = validate(debug)
        print(f'Test with seed {random_seed} done, valid: {valid}')



