import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from neuralop.models import FNO  # 确保你训练用的是这个模型类

from utils.metrics import histogram_l2  # 导入自定义的误差计算和绘图函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 如果没有GPU可用，使用CPU

# ---------- 加载模型 ----------
model = FNO.from_checkpoint(save_folder="./MyScript/cross/",save_name="cross_model")#训练保存模型路径
model.eval()
model.cuda()

# ---------- 加载测试数据 ----------
data = torch.load("./My_data/cross_64/cross900_test_64.pt")
a = data['a'].cuda()  # 输入
u = data['u'].cuda()  # 真值

# ---------- 模型预测 ----------
with torch.no_grad():
    pred = model(a)  # 输出: [N, 2, 64, 64]

# ---------- 计算误差 ----------
mse_loss = torch.nn.MSELoss()
loss = mse_loss(pred, u)
print(f" MSE Loss on test set: {loss.item():.6f}")


# ---------- 可视化五个样本 ----------
for sample_id in range(5):
    
    true_real = u[sample_id, 0].cpu()
    pred_real = pred[sample_id, 0].cpu()
    error_map = true_real - pred_real
    input_map = a[sample_id, 0].cpu()

    # 统一色标范围（基于真值和预测）
    vmin = min(true_real.min(), pred_real.min()).item()
    vmax = max(true_real.max(), pred_real.max()).item()

    # 使用 gridspec 创建 2×3 子图布局，最后一列用于 colorbar
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.3, hspace=0.3)

    # 输入图（无 colorbar）
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(input_map, cmap="gray")
    ax0.set_title("Input (mask)")

    # 真值图
    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(true_real, cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title("True Ex real")

    # 预测图
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(pred_real, cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_title("Predicted Ex real")

    # 误差图
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(error_map, cmap="viridis", vmin=vmin, vmax=vmax)
    ax3.set_title("Error Ex real")

    # colorbar：使用右侧空位置
    cax = fig.add_subplot(gs[:, 2])  # 全行共享 colorbar
    fig.colorbar(im, cax=cax)

    plt.show()


# 计算相对L2误差并绘制直方图
histogram_l2(pred, u, bins=100, title="Histogram of Relative L2 Error")  # 显示直方图
