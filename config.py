import torch
# 数据集配置
DATA_CONFIG = {
    "train_ann_file": "/home/xiaojy/CongestionModel_0/training_set_ISPD2015/congestion/file/ISPD2015_train.csv",  # 训练集标注文件
    "val_ann_file": "/home/xiaojy/CongestionModel_0/training_set_ISPD2015/congestion/file/ISPD2015_test.csv",      # 验证集标注文件
    "dataroot": "/home/xiaojy/CongestionModel_0/training_set_ISPD2015/congestion",                         # 数据根目录
    "batch_size": 8,
    "aug_pipeline": ["Flip", "Rotation"],               # 数据增强 pipeline
    "Rotation": {"rotate_ratio": 0.5},                  # 旋转增强参数
    "Flip": {"flip_ratio": 0.5, "direction": "horizontal"}  # 翻转增强参数
}

# 模型配置
MODEL_CONFIG = {
    "n_channels": 3,    # 输入特征通道数
    "n_classes": 1,     # 输出标签通道数
    "bilinear": False   # 是否使用双线性插值上采样
}

# 训练配置
TRAIN_CONFIG = {
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_interval": 10,  # 日志打印间隔
    "save_dir": "checkpoints",  # 模型保存目录
    "val_interval": 5    # 验证间隔（epoch）
}