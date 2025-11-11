import random
from pprint import pprint

import math
import torch

from models_valid.mamba_model_bi import MambaConfig, MambaModel


def validate(
    debug=True,
    image_input=True,
    sequence_length=128,
    d_model=256,
    image_size=224,
    patch_size=16,
    n_layers=2,
    batch_size=4,
    rtoal=1e-4,
    atol=1e-5,
):
    # Create config for both implementations
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
    #     'local_vim': False,
    #     'pscan': True,
    #     'python_z': True,
    # }

    # CUDA version config
    cuda_config = MambaConfig(**common_config_cuda, use_cuda=True)
    # Python version config
    py_config = MambaConfig(**common_config_py, use_cuda=False)
    in_channels = 3
    # Create models
    cuda_model = MambaModel(cuda_config, image_input=image_input,
                            img_size=image_size, patch_size=patch_size,
                            in_channels=in_channels, reture_ft=True).cuda()
    py_model = MambaModel(py_config, image_input=image_input,
                          img_size=image_size, patch_size=patch_size,
                          in_channels=in_channels, reture_ft=True).cuda()

    # Copy weights
    py_model.load_state_dict(cuda_model.state_dict())

    # Generate test input
    if image_input:
        x = torch.rand(batch_size, in_channels, image_size, image_size)
    else:
        x = torch.rand(batch_size, sequence_length, d_model)

    # Run inference
    with torch.no_grad():
        cuda_output = cuda_model(x.cuda())
        py_output = py_model(x.cuda())

    # Compare results
    cuda_output = cuda_output
    close = torch.allclose(cuda_output, py_output, rtol=rtoal, atol=atol)
    max_diff = (cuda_output - py_output).abs().max().item()

    if debug:
        print(f"Outputs {'match' if close else 'differ'}")
        print(f"Max difference: {max_diff:.2e}")
        print(f"Allowed tolerance: {atol:.2e}")

    return close


if __name__ == '__main__':

    # set cuda device
    torch.cuda.set_device(2)

    debug = False
    # debug = False
    test_time = 100
    rtoal = 6e-4
    atol = 2e-3

    kw_args = {
        'image_input': True,
        'sequence_length': 128,
        'd_model': 256,
        'image_size': 224,
        'patch_size': 16,
        'n_layers': 12,
        'batch_size': 4,
    }

    if debug:
        test_time = 1

    for _ in range(test_time):
        random_seed = random.randint(0, int(1e9))
        # random_seed = 246233883
        torch.cuda.manual_seed(random_seed)
        torch.manual_seed(random_seed)

        valid = validate(debug=debug, rtoal=rtoal, atol=atol, **kw_args)
        print(f'Test with seed {random_seed:10d} done, valid: {valid}')



