# Copyright 2022 CircuitNet. All rights reserved.

import mmcv
import numpy as np


class Flip:
    """
    随机翻转图像（水平或垂直）
    keys:      要翻转的数据字段列表（如 'feature', 'label'）
    flip_ratio: 翻转的概率
    direction:  翻转方向：'horizontal' 或 'vertical'
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys=['feature', 'label'], flip_ratio=0.5, direction='horizontal', **kwargs):
        # 检查方向是否合法
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')
        self.keys = keys
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        # 随机决定是否翻转
        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                # 对每一个 key 对应的数据进行就地翻转
                if isinstance(results[key], list):
                    # 如果是 list，就遍历每一项再翻转
                    for v in results[key]:
                        mmcv.imflip_(v, self.direction)
                else:
                    # 单张图直接翻转
                    mmcv.imflip_(results[key], self.direction)

        return results


class Rotation:
    """
    随机旋转图像 90° 的整数倍
    keys:         要旋转的数据字段列表
    axis:         旋转轴（tuple 或 dict），默认 (0,1) 平面
    rotate_ratio: 旋转的概率
    direction:    可选的旋转次数，对应 90°、180°、270° 等
    """

    def __init__(self, keys=['feature', 'label'], axis=(0, 1), rotate_ratio=0.5, **kwargs):
        self.keys = keys
        # 支持统一轴或为每个 key 指定不同轴
        self.axis = {k: axis for k in keys} if isinstance(
            axis, tuple) else axis
        self.rotate_ratio = rotate_ratio
        # np.rot90 的 k 参数选项：0 (不变), -1, -2, -3
        self.direction = [0, -1, -2, -3]

    def __call__(self, results):
        # 随机决定是否旋转
        rotate = np.random.random() < self.rotate_ratio

        if rotate:
            # 从 direction 中任选一个旋转次数（排除 k=0 可视为“无旋转”）
            rotate_angle = self.direction[int(np.random.random()/(10.0/3.0))+1]
            for key in self.keys:
                if isinstance(results[key], list):
                    # 如果是 list，则整个 list 被重写为最后一帧的旋转结果
                    for v in results[key]:
                        results[key] = np.ascontiguousarray(
                            np.rot90(v, rotate_angle, axes=self.axis[key]))
                else:
                    results[key] = np.ascontiguousarray(
                        np.rot90(results[key], rotate_angle, axes=self.axis[key]))

        return results
