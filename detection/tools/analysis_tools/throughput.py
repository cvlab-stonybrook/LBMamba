import os
import sys
import time
from functools import partial

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


def mmdet_flops(config=None):
    from mmengine.config import Config
    from mmengine.runner import Runner
    import numpy as np

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()

    model.eval()
    # get_model_complexity_info = mmengine_flop_count(_get_model_complexity_info=True)
    dataloader_cfg = cfg.test_dataloader
    print("Previous batch size:", dataloader_cfg['batch_size'])
    dataloader_cfg['num_workers'] = 1
    dataloader_cfg['batch_size'] = 1
    dataloader_cfg['persistent_workers'] = False
    data_loader = Runner.build_dataloader(dataloader_cfg)
    # data_loader = runner.val_dataloader
    # 创建虚拟数据集和数据加载器
    # dataset = DummyDataset(image_size=(3, 1280, 800), length=1000)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    # 预热步骤
    for data_batch in data_loader:
        data = model.data_preprocessor(data_batch)
        # print(data)
        output = model(**data)
        break

    with torch.no_grad():
        data_list = []
        for i, data_batch in enumerate(data_loader):
            data = model.data_preprocessor(data_batch)
            data_list.append(data)
            if i + 1 == repeat_batch:
                break

        start_time = time.time()
        # for rep in range(repeat_batch):
        #     output = model(**data_list[])
        for data in data_list:
            output = model(**data)
        total_time = time.time() - start_time

    avg_second = total_time / repeat_batch / 1
    print(f"Average speed: {avg_second} per image, throughput {1 / avg_second:.2f}")

    # if True:
    #     data_loader = runner.val_dataloader
    #     num_images = 100
    #     mean_flops = []
    #     for idx, data_batch in enumerate(data_loader):
    #         if idx == num_images:
    #             break
    #         data = model.data_preprocessor(data_batch)
    #         model.forward = partial(model.forward, data_samples=data['data_samples'])
    #         out = get_model_complexity_info(model, input_shape=(3, 1280, 800))
    #         params = out['params_str']
    #         mean_flops.append(out['flops'])
    #     mean_flops = np.average(np.array(mean_flops))
    #     print(params, mean_flops)


if __name__ == "__main__":
    repeat_batch = 100
    import model
    mmdet_flops(sys.argv[1])
