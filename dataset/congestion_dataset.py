# Copyright 2022 CircuitNet. All rights reserved.

import os.path as osp
import copy
import numpy as np
from torchvision.transforms import Compose


class CongestionDataset(object):
    """
    数据集类，用于加载拥塞数据（feature + label）并应用可选的预处理管道。
    ann_file:   注释文件，每行 "feature.npy,label.npy"
    dataroot:   数据根目录，会与注释中的路径拼接
    pipeline:   torchvision.transforms.Compose 对象，定义一系列预处理操作
    test_mode:  是否为测试模式（可根据需要扩展）
    """

    def __init__(self, ann_file, dataroot, pipeline=None, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        # 如果传入了 pipeline 配置，就用 Compose 组合；否则不做任何预处理
        if pipeline:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        # 读取并解析注释文件，得到每条样本的信息列表
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """
        解析 ann_file，读取每行 feature 和 label 的文件名。
        返回一个 dict 列表，每项包含 feature_path 和 label_path。
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                feature, label = line.strip().split(',')
                # 如果指定了 dataroot，就拼接成完整路径
                if self.dataroot is not None:
                    feature_path = osp.join(self.dataroot, feature)
                    label_path = osp.join(self.dataroot, label)
                data_infos.append(
                    dict(feature_path=feature_path, label_path=label_path))
        return data_infos

    def prepare_data(self, idx):
        """
        根据索引加载一条样本：
          1. 拷贝路径信息
          2. 用 numpy.load 加载 .npy 数据
          3. 可选地通过 pipeline 处理
          4. 转置通道为 (C, H, W) 并转成 float32
        返回：feature, label, label_path（可用于跟踪或评估）
        """
        # 深拷贝，避免修改原始 data_infos
        results = copy.deepcopy(self.data_infos[idx])
        # 从磁盘加载 numpy 数据
        results['feature'] = np.load(results['feature_path'])
        results['label'] = np.load(results['label_path'])
        # 应用预处理（如果有）
        results = self.pipeline(results) if self.pipeline else results
        # 转置到 (C, H, W) 并确保数据类型为 float32
        feature = results['feature'].transpose(2, 0, 1).astype(np.float32)
        label = results['label'].transpose(2, 0, 1).astype(np.float32)

        return feature, label, results['label_path']

    def __len__(self):
        # 数据集大小等于注释条数
        return len(self.data_infos)

    def __getitem__(self, idx):
        # 支持索引访问，返回一条样本
        return self.prepare_data(idx)
