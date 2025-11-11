import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, image_size, length=100):
        self.image_size = image_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(*self.image_size)
        target = torch.randint(0, 1000, (1,))
        return image, target

def mmseg_throughput(config=None, input_shape=(3, 512, 2048)):
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()

    model.eval()

    # 创建虚拟数据集和数据加载器
    dataset = DummyDataset(image_size=input_shape, length=1000)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    # 预热步骤
    for images, targets in data_loader:
        images, targets = images.cuda(), targets.cuda()
        output = model(images)
        break

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.cuda(), targets.cuda()
            break

        start_time = time.time()
        for rep in range(repeat_batch):
            output = model(images)
        total_time = time.time() - start_time

    avg_second = total_time / repeat_batch / 1
    print(f"Average speed: {avg_second} per image, throughput {1 / avg_second:.2f}")

if __name__ == "__main__":
    repeat_batch=100
    # import model
    mmseg_throughput(sys.argv[1])
