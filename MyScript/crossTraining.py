from torch.utils.data import DataLoader
from neuralop.models import FNO
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import sys
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
import wandb

from utils.MyDataset import MyDataset as CrossDataset


# ---------- 设置设备 ----------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 如果没有GPU可用，使用CPU
print(f"使用设备: {device}")  # 打印当前使用的设备

# ---------- 设置超参数 ----------
batch_size = 16  # 批大小
n_epochs = 500  # 训练轮数
eval_interval = 1  # 评估间隔（每隔 eval_interval 轮评估一次）
# ---------- 设置WandB ----------

# 初始化 wandb
wandb.init(
    project="fno-cross",  # 你可以改成自己的项目名
    name="cross_64_exp1",  # 每次运行的实验名（可选）
    config={  # 可传入训练配置
        "batch_size": batch_size,
        "learning_rate": 8e-3,
        "hidden_channels": 32,
        "loss": "H1",
        "optimizer": "AdamW",
        "n_layers": 4,
        "n_epochs": n_epochs,
    },
)


# ---------- 导入数据集 ----------
dataset = CrossDataset("./My_data/cross_64/cross900_train_64.pt")
testdata = CrossDataset("./My_data/cross_64/cross900_test_64.pt")


train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True,
                              persistent_workers=False,)

test_loaders = DataLoader(testdata, 
                          batch_size = batch_size,
                          num_workers=0,
                          pin_memory=True,
                          persistent_workers=False,)
test_loaders = {"test": test_loaders}  # 将测试数据加载器放入字典中
# ---------- 定义模型 ----------
model = FNO(
    n_modes=(64, 64),  # 64x64 网格
    in_channels=1,  # 结构图通道数
    out_channels=2,  # 电场 Ex 的实部 + 虚部
    hidden_channels=32,  # 隐藏层通道数
    n_layers=4,
)
model = model.to(device)  # 将模型移动到GPU
wandb.watch(model, log="gradients", log_freq=10) # 监控模型参数和梯度
# 统计模型参数数量
num_params = count_model_params(model)
print(f"模型参数数量: {num_params}")
sys.stdout.flush()
# ---------- 设置优化器和损失函数 ----------

optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

# ---------- 训练模型 ----------
trainer = Trainer(
    model=model,
    n_epochs=n_epochs,
    device=device,
    data_processor=None,
    wandb_log=True,
    eval_interval=eval_interval,
    use_distributed=False,
    verbose=True,
)


trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss,  # 训练损失函数
    eval_losses=eval_losses,  # 测试损失函数
)  # 训练模型


# ---------- 测试模型 ----------

# ---------- 保存模型 ----------

model.save_checkpoint(save_folder="./MyScript/cross", save_name="cross_model")
