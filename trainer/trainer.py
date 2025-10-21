import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import build_dataset


class Trainer:
    def __init__(self, model, train_config, data_config):
        self.model = model
        self.epochs = train_config["epochs"]
        self.lr = train_config["lr"]
        self.weight_decay = train_config["weight_decay"]
        self.device = train_config["device"]
        self.log_interval = train_config["log_interval"]
        self.save_dir = train_config["save_dir"]
        self.val_interval = train_config["val_interval"]
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 构建数据加载器
        self.train_loader = self._build_dataloader(data_config, test_mode=False)
        self.val_loader = self._build_dataloader(data_config, test_mode=True)
        
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 移动模型到设备
        self.model.to(self.device)

    def _build_dataloader(self, data_config, test_mode):
        """构建数据加载器"""
        opt = {
            "ann_file": data_config["val_ann_file"] if test_mode else data_config["train_ann_file"],
            "dataroot": data_config["dataroot"],
            "test_mode": test_mode,
            "batch_size": data_config["batch_size"],
            "aug_pipeline": data_config.get("aug_pipeline", [])
        }
        # 添加数据增强参数
        if "Rotation" in data_config:
            opt.update(data_config["Rotation"])
        if "Flip" in data_config:
            opt.update(data_config["Flip"])
            
        return build_dataset(opt)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
            for i, (features, labels, _) in enumerate(self.train_loader):
                # 数据移动到设备
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # 打印日志
                if (i + 1) % self.log_interval == 0:
                    avg_loss = total_loss / (i + 1)
                    pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
                
                pbar.update(1)
        
        return total_loss / len(self.train_loader)

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, labels, _ in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)

    def train(self):
        """完整训练过程"""
        best_val_loss = float("inf")
        
        for epoch in range(self.epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.6f}")
            
            # 验证
            if (epoch + 1) % self.val_interval == 0:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.6f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, "best_model.pth")
                    )
                    print(f"Saved best model with val loss: {best_val_loss:.6f}")
        
        # 保存最后一个epoch的模型
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, "last_model.pth")
        )