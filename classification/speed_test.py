import argparse
import contextlib

import torch
import time

from scipy.stats.tests.test_continuous_fit_censored import optimizer
from torch.utils.data import DataLoader, Dataset
from timm.models import create_model
from timm.utils import accuracy
from torch.profiler import profile, record_function, ProfilerActivity

import models_mamba
import models_mamba_bi
import models_mamba_bi_head


class DummyDataset(Dataset):
    def __init__(self, image_size, length=100):
        self.image_size = image_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(3, *self.image_size)
        target = torch.randint(0, 1000, (1,))
        return image, target

def get_args_parser():
    parser = argparse.ArgumentParser(description='Speed test script')
    parser.add_argument('--model', default='vim_tiny_patch16_224_bimambav2_final_pool_mean_div2',
                        type=str, help='Model name')
    parser.add_argument('--image_size', default=(224, 224), type=lambda x: tuple(map(int, x.split(','))), 
                        help='Input image dimensions (H,W)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--repeat_batch', type=int, default=100)
    parser.add_argument('--mode', choices=['train', 'inference'], default='inference')
    parser.add_argument('--profiler', action='store_true', help='Enable detailed profiling')
    return parser

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建虚拟数据集和数据加载器
    dataset = DummyDataset(image_size=args.image_size, length=args.batch_size * args.repeat_batch)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    # 加载模型
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        img_size=args.image_size[0],
        drop_path_rate = 0.05,
    ).to(device)
    
    if args.mode == 'train':
        model.train()
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    else:
        model.eval()
        criterion = None
        optimizer = None

    # print args in a one line log
    print(f"Model: {args.model}, Image size: {args.image_size}, Batch size: {args.batch_size}, "
          f"Repeat batch: {args.repeat_batch}, Mode: {args.mode}, Profiler: {args.profiler}")
    
    # 预热步骤
    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        if args.mode == 'train':
            targets = targets.flatten()
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        break
    
    total_time = 0
    with torch.no_grad() if args.mode == 'inference' else contextlib.nullcontext():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            break

        if args.profiler:
            with profile(activities=[ProfilerActivity.CUDA], profile_memory=False, record_shapes=False) as prof:
                for rep in range(args.repeat_batch):
                # for images, targets in data_loader:
                #     images, targets = images.to(device), targets.to(device)

                    with record_function("model_inference"):
                        output = model(images)
                        if args.mode == 'train':
                            targets = targets.flatten()
                            loss = criterion(output, targets)
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
            print(prof.key_averages().table(sort_by="self_cuda_time_total"))
        else:
            start_time = time.time()
            for rep in range(args.repeat_batch):
            # for images, targets in data_loader:
            #     images, targets = images.to(device), targets.to(device)

                if args.mode == 'train':
                    optimizer.zero_grad()
                    output = model(images)
                    targets = targets.flatten()
                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    output = model(images)
            total_time = time.time() - start_time
    
    if not args.profiler:
        avg_second = total_time / args.repeat_batch / args.batch_size
        print(f"Average speed: {avg_second} per image, throughput {1 / avg_second:.2f}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)