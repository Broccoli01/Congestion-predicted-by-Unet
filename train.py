import torch
from model import UNet
from trainer import Trainer
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG


def main():
    # 初始化模型
    model = UNet(
        n_channels=MODEL_CONFIG["n_channels"],
        n_classes=MODEL_CONFIG["n_classes"],
        bilinear=MODEL_CONFIG["bilinear"]
    )
    
    # 初始化训练器并开始训练
    trainer = Trainer(model, TRAIN_CONFIG, DATA_CONFIG)
    trainer.train()


if __name__ == "__main__":
    main()