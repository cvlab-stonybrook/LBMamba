import random
from pprint import pprint

import math
import torch

from models_valid.mamba_model_bi import MambaConfig, MambaModel


def validate_backward(
    debug=True,
    image_input=True,
    sequence_length=128,
    d_model=256,
    image_size=224,
    patch_size=16,
    n_layers=12,
    batch_size=4,
    n_classes=100,
    rtoal=1e-4,
    atol=1e-5,
    fp16=False  # 新增fp16选项
):
    # 初始化配置与模型（保持与前向验证相同）
    common_config_cuda = {
        'd_model': d_model,
        'n_layers': n_layers,
        'd_state': 16,
        'expand_factor': 2,
        'local_vim': True,
        'pscan': True,
        'python_z': True,
    }
    common_config_py = common_config_cuda
    # {
    #     'd_model': d_model,
    #     'n_layers': n_layers,
    #     'd_state': 16,
    #     'expand_factor': 2,
    #     'local_vim': True,
    #     'pscan': True,
    #     'python_z': True,
    # }

    cuda_config = MambaConfig(**common_config_cuda, use_cuda=True)
    py_config = MambaConfig(**common_config_py, use_cuda=False)

    # 初始化模型（添加fp16支持）
    cuda_model = MambaModel(cuda_config, image_input=image_input,
                            img_size=image_size, patch_size=patch_size,
                            in_channels=3, reture_ft=False, n_classes=n_classes).cuda()
    py_model = MambaModel(py_config, image_input=image_input,
                          img_size=image_size, patch_size=patch_size,
                          in_channels=3, reture_ft=False, n_classes=n_classes).cuda()

    # 参数同步
    py_model.load_state_dict(cuda_model.state_dict())

    loss_fn = torch.nn.CrossEntropyLoss()

    fp16_scaler = None
    if fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # add optimizer
    cuda_optimizer = torch.optim.AdamW(cuda_model.parameters(), lr=1e-4)
    py_optimizer = torch.optim.AdamW(py_model.parameters(), lr=1e-4)


    # 生成输入数据
    if image_input:
        x = torch.rand(batch_size, 3, image_size, image_size).cuda()
    else:
        x = torch.rand(batch_size, sequence_length, d_model).cuda()

    # generate random label
    if n_classes > 0:
        y = torch.randint(0, n_classes, (batch_size,)).cuda()


    # 前向+反向计算
    def compute_grads(model, input):
        model.zero_grad()
        output = model(input)

        if n_classes > 0:
            loss = loss_fn(output, y)
        else:
            loss = output.square().sum()

        # loss.backward()
        if fp16_scaler is None:
            loss.backward()
        else:
            fp16_scaler.scale(loss).backward()

        if fp16_scaler:
            fp16_scaler.unscale_(cuda_optimizer)
            fp16_scaler.unscale_(py_optimizer)

        return output, {n: p.grad for n, p in model.named_parameters() if p.requires_grad}

    # clone and detach x for cuda and py
    x_py = x.clone().detach().requires_grad_(True)
    x_cuda = x.clone().detach().requires_grad_(True)

    y_pred_cuda, cuda_grads = compute_grads(cuda_model, x_cuda)
    y_pred_py, py_grads = compute_grads(py_model, x_py)


    # 梯度对比
    all_close = True

    # compare the prediction of cuda and py
    if not torch.allclose(y_pred_cuda, y_pred_py, rtol=rtoal, atol=atol):
        all_close = False
        if debug:
            print(f"Prediction mismatch:")
            print(f"CUDA pred: {y_pred_cuda.abs().mean().item():.2e}")
            print(f"Python pred: {y_pred_py.abs().mean().item():.2e}")
            print(f"Max difference: {(y_pred_cuda - y_pred_py).abs().max().item():.2e}")
    else:
        if debug:
            print(f"Prediction match")

    max_diff = 0
    for (cuda_name, cuda_grad), (py_name, py_grad) in zip(cuda_grads.items(), py_grads.items()):
        assert cuda_name == py_name, "Parameter name mismatch"

        if cuda_grad is None or py_grad is None:
            # print message and continue
            print(f"Parameter {cuda_name} has no gradient in cuda/py:",
                  str(cuda_grad is None), str(py_grad is None))
            continue

        close = torch.allclose(cuda_grad, py_grad, rtol=rtoal, atol=atol)
        current_diff = (cuda_grad - py_grad).abs().max().item()

        if not close:
            all_close = False
        max_diff = max(max_diff, current_diff)

        if debug:
            if not close:
                print(f"Gradient mismatch in {cuda_name}:")
                print(f"CUDA grad: {cuda_grad.abs().mean().item():.2e}")
                print(f"Python grad: {py_grad.abs().mean().item():.2e}")
                print(f"Max difference: {current_diff:.2e}")
            else:
                print(f"Gradient match in {cuda_name}")

    # check if gradients on x are the same
    if not torch.allclose(x_cuda.grad, x_py.grad, rtol=rtoal, atol=atol):
        all_close = False
        print(f"Gradient mismatch in x:")
        print(f"CUDA grad: {x_cuda.grad.abs().mean().item():.2e}")
        print(f"Python grad: {x_py.grad.abs().mean().item():.2e}")
        print(f"Max difference: {(x_cuda.grad - x_py.grad).abs().max().item():.2e}")

    if debug:
        print(f"Gradients {'match' if all_close else 'differ'}")
        print(f"Max gradient difference: {max_diff:.2e}")

    return all_close


if __name__ == '__main__':

    # set cuda device
    torch.cuda.set_device(2)
    print("Validating backward")

    # debug = True
    debug = False
    test_time = 100
    rtoal = 2e-4
    atol = 1e-4
    fp16 = False

    # make the following key ward args
    kw_args = {
        'image_input': True,
        'sequence_length': 128,
        'd_model': 256,
        'image_size': 224,
        'patch_size': 16,
        'n_layers': 4,
        'batch_size': 10,
        'n_classes': 100,
    }

    if debug:
        test_time = 1

    for _ in range(test_time):
        random_seed = random.randint(0, int(1e9))
        # random_seed = 246233883
        torch.cuda.manual_seed(random_seed)
        torch.manual_seed(random_seed)

        valid = validate_backward(debug=debug, rtoal=rtoal, atol=atol, fp16=fp16, **kw_args)
        # print(f'Test with seed {random_seed} done, valid: {valid}')
        print(f'Test with seed {random_seed:10d} done, valid: {valid}')



