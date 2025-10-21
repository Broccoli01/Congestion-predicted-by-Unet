from torch.utils.data import DataLoader
import time
from .augmentation import Flip, Rotation
from .congestion_dataset import CongestionDataset


class IterLoader:
    """Wrap a DataLoader to yield batches endlessly (restarts on epoch end)."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            # 取下一个 batch
            data = next(self.iter_loader)
        except StopIteration:
            # 一个 epoch 结束，稍等后重置迭代器
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __len__(self):
        # 返回 DataLoader 原始长度（batch 数）
        return len(self._dataloader)

    def __iter__(self):
        return self
    


def build_dataset(opt):
    """
    根据配置构建 CongestionDataset 和相应 DataLoader。
    opt 中常见字段：
      - ann_file, dataroot, test_mode, batch_size
      - aug_pipeline: ['Flip', 'Rotation']（训练时启用）
      - Rotation 所需参数也放在 opt 里
    """

    # 定义可选的增广方法
    aug_methods = {'Flip': Flip(), 'Rotation': Rotation(**opt)}
    # 只有在非测试模式下，且指定了 aug_pipeline 才组装 pipeline
    pipeline = [aug_methods[i] for i in opt.pop(
        'aug_pipeline')] if 'aug_pipeline' in opt and not opt['test_mode'] else None
    # 直接使用 CongestionDataset
    dataset = CongestionDataset(**opt, pipeline=pipeline)
    if opt['test_mode']:
        # 测试时：单样本、无 shuffle、单进程
        return DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    else:
        # 训练时：多进程、打乱、丢弃最后不满 batch
        return IterLoader(DataLoader(dataset=dataset,
                                     num_workers=16,
                                     batch_size=opt.pop('batch_size'),
                                     shuffle=True,
                                     drop_last=True,
                                     pin_memory=True))
