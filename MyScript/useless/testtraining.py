from torch.utils.data import DataLoader
from unit_dataset import UnitDataset
from neuralop.models import FNO
import torch.nn as nn
import torch.optim as optim
import torch

# 替换为你的数据路径
dataset = UnitDataset("./My_data/unit_64/unit_train_64.pt")
# dataset = UnitDataset("./neuralop/data/datasets/data/unit_train_64.pt")

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)


model = FNO(
    n_modes=(64, 64),  # 64x64 网格
    in_channels=1,  # 结构图通道数
    out_channels=2,  # 电场 Ex 的实部 + 虚部
    hidden_channels=32, # 隐藏层通道数
    n_layers=4,
).cuda()


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 存储阶段结果
snapshots = {}
save_epochs = {100,300,500}


for epoch in range(501):
    model.train()
    total_loss = 0
    for a, u in train_loader:
        a, u = a.cuda(), u.cuda()
        optimizer.zero_grad()
        pred = model(a)
        loss = criterion(pred, u)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")
    # save the output of the model for 100,300,500 epochs
    if epoch in save_epochs:
        snapshots[epoch] = pred[0, 0].cpu().detach().numpy()
        print(f"Snapshot saved for epoch {epoch+1}")

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    a, u = next(iter(train_loader))
    pred = model(a.cuda()).cpu()
t = u[0, 0, 0].cpu()
plt.subplot(2, 2, 1)
plt.imshow(u[0, 0, 0].cpu().numpy(), cmap="viridis")
plt.title('real part of Ex')

plt.subplot(2, 2, 2)
plt.imshow(snapshots[100], cmap="viridis")
plt.title('epoch=100')

plt.subplot(2, 2, 3)
plt.imshow(snapshots[300], cmap="viridis")
plt.title('epoch=300')

plt.subplot(2, 2, 4)
plt.imshow(snapshots[500], cmap="viridis")
plt.title('epoch=500')
plt.show()
