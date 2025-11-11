# validate_cuda_kernel_bwd.py
import random
import math
import torch
from pprint import pprint

from einops import rearrange
from torch import nn
import torch.nn.functional as F
# from models_valid.pscan import pscan as pscan_bi
from models_valid.pscan_bi import pscan_bi
import lbm.pscan_cuda.pscan as selective_scan_cuda
# import selective_scan_cuda

def mamba_block_initialization(dt_rank, d_inner, dt_init, dt_scale, dt_min, dt_max, dt_init_floor):
    # 初始化dt_proj类似MambaBlock的方式
    dt_proj = torch.nn.Linear(dt_rank, d_inner, bias=True)
    
    # dt初始化逻辑
    dt_init_std = dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
        nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == "random":
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    
    # delta bias初始化
    dt = torch.exp(
        torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    
    return dt_proj

def random_initialize(BB, LL, DD, NN, dt_proj, requires_grad=True):
    # 参数初始化（包含梯度需求）
    u = torch.randn(BB, DD, LL, requires_grad=requires_grad)
    # u = torch.ones(BB, DD, LL, requires_grad=requires_grad, dtype=torch.float32)
    delta = torch.randn(BB, DD, LL, requires_grad=requires_grad)
    # delta = torch.ones(BB, DD, LL, requires_grad=requires_grad, dtype=torch.float32)
    
    # 使用MambaBlock风格的A初始化
    A = torch.arange(1, NN + 1, dtype=torch.float32).repeat(DD, 1)
    A_log = torch.log(A).clone().detach()
    A_log.requires_grad_(requires_grad)
    
    B = torch.randn(BB, NN, LL, requires_grad=requires_grad)
    C = torch.randn(BB, NN, LL, requires_grad=requires_grad)
    # B = torch.ones(BB, NN, LL, requires_grad=requires_grad, dtype=torch.float32)
    # C = torch.ones(BB, NN, LL, requires_grad=requires_grad, dtype=torch.float32)
    D = torch.randn(DD, requires_grad=requires_grad)
    z = torch.randn(BB, DD, LL, requires_grad=requires_grad)
    # z = torch.ones(BB, DD, LL, requires_grad=requires_grad, dtype=torch.float32)
    delta_bias = dt_proj.bias.clone().detach().requires_grad_(requires_grad)

    grad = torch.randn(BB, DD, LL)
    
    return u, delta, A_log, B, C, D, z, delta_bias, grad

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
        # hs = pscan_bi(deltaA, BX)

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


def python_backward(u, delta, A, B, C, D, z, delta_bias, delta_softplus, grad):

    grad = grad.transpose(1, 2)
    for p in [u, delta, A, B, C, D, z, delta_bias]:
        if p is not None:
            p.retain_grad()
    # 前向计算
    with torch.enable_grad():
        output = python_mamba(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        output.backward(grad)
    
    # 收集梯度
    grads = [p.grad for p in [u, delta, A, B, C, D, z, delta_bias] if p is not None]
    return grads

def cuda_backward(u, delta, A, B, C, D, z, delta_bias, delta_softplus, grad):
    # 转置维度匹配CUDA实现
    B = rearrange(B, "b dstate l -> b 1 dstate l")
    C = rearrange(C, "b dstate l -> b 1 dstate l")
    
    # 前向+反向计算
    with torch.enable_grad():
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

        has_z = z is not None
        if has_z:
            out_z = rest[0]
        else:
            z = None
            out = None

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, grad, x, out, None, delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if has_z else None

    dB = dB.squeeze(1)
    dC = dC.squeeze(1)

    # 解析梯度顺序与Python实现一致
    return (du, ddelta, dA, dB, dC,
            dD if D is not None else None,
            dz,
            ddelta_bias if delta_bias is not None else None)

def validate_backward(debug=True):
    # 配置参数
    BB, LL, DD, NN = 1, 256, 4, 1
    config = {
        'dt_rank': math.ceil(DD/16),
        'd_inner': DD,
        'dt_init': 'random',
        'dt_scale': 1.0,
        'dt_min': 0.001,
        'dt_max': 0.1,
        'dt_init_floor': 1e-4
    }
    
    # 初始化
    dt_proj = mamba_block_initialization(**config)
    u, delta, A, B, C, D, z, delta_bias, grad = random_initialize(BB, LL, DD, NN, dt_proj)
    delta_softplus = True

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
        print('grad:', grad.shape)

    # move u, delta, A, B, C, D, z, delta_bias, grad to cuda
    u, delta, A, B, C, D, z, delta_bias, grad = (u.cuda(), delta.cuda(), A.cuda(), B.cuda(),
                                                 C.cuda(), D.cuda(), z.cuda(), delta_bias.cuda(), grad.cuda())

    # Python梯度计算
    py_grads = python_backward(u, delta, A, B, C, D, z, delta_bias, delta_softplus, grad)
    
    # CUDA梯度计算
    cuda_grads = cuda_backward(u, delta, A, B, C, D, z, delta_bias, delta_softplus, grad)
    
    # 梯度对比
    tolerance = {'rtol': 1e-4, 'atol': 1e-4}
    all_close = True
    for name, pg, cg in zip(['u','delta','A','B','C','D','z','delta_bias'], py_grads, cuda_grads):
        if pg is None or cg is None:
            print(f'Gradient None in {name} py/cuda: {pg is None} vs {cg is None}')
            all_close = False
            continue
            
        close = torch.allclose(pg, cg, **tolerance)
        if debug and not close:
            print(f'Gradient mismatch in {name}')
            # print(f'Python max: {pg.abs().max().item():.4e}')
            # print(f'CUDA max:   {cg.abs().max().item():.4e}')
            # print python and cuda gradients
            print('Python grad:', pg.cpu().numpy().tolist())
            print('CUDA grad:', cg.cpu().numpy().tolist())
            print("pg - cg: ", pg - cg)
        # elif name == "A":
        #     print(f'Gradient on {name}')
        #     # print(f'Python max: {pg.abs().max().item():.4e}')
        #     # print(f'CUDA max:   {cg.abs().max().item():.4e}')
        #     # print python and cuda gradients
        #     print('Python grad:', pg.cpu().numpy().tolist())
        #     print('CUDA grad:', cg.cpu().numpy().tolist())
        #     print("pg - cg: ", pg - cg)

        if debug and close:
            print(f'Gradient match in {name}')
        all_close = all_close and close
    
    return all_close

if __name__ == '__main__':
    # Globally set cuda device to 2
    torch.cuda.set_device(2)

    seed = random.randint(1, 2147483647)
    # seed = 1383193364
    print(f'Manual seed {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for _ in range(10):
        valid = validate_backward(debug=True)
        print(f'Backward validation: {valid}')
        if not valid:
            break
