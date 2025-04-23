import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neuralop.models import FNO  
import numpy as np

from utils.metrics import histogram_l2  # 导入自定义的误差计算和绘图函数
# ---------- 设置设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 如果没有GPU可用，使用CPU

# ---------- 加载模型 ----------
model = FNO.from_checkpoint(save_folder="./MyScript/square/",save_name="square_model")#训练保存模型路径
model.eval()
model.cuda()

# ---------- 加载测试数据 ----------
data = torch.load("./My_data/square_64/square_test_64.pt")
a = data['a'].cuda()  # 输入
u = data['u'].cuda()  # 真值

# ---------- 模型预测 ----------
with torch.no_grad():
    pred = model(a)  # 输出: [N, 2, 64, 64]

# ---------- 计算误差 ----------
mse_loss = torch.nn.MSELoss()
loss = mse_loss(pred, u)
print(f" MSE Loss on test set: {loss.item():.6f}")

# ---------- 可视化结果 ----------

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
histogram_l2(pred, u, bins=50, title="Histogram of Relative L2 Error")  # 显示直方图


# # 测试近场外推
# from Math.SurfaceIntegrator import integrate_mode_real

# grid_x = np.array(
#     [
#         0.03906249999999999,
#         0.11718749999999997,
#         0.19531249999999997,
#         0.27343749999999994,
#         0.35156249999999994,
#         0.42968749999999994,
#         0.5078124999999999,
#         0.5859374999999999,
#         0.6640624999999999,
#         0.7421874999999999,
#         0.8203124999999999,
#         0.8984374999999999,
#         0.9765624999999998,
#         1.0546874999999998,
#         1.1328124999999998,
#         1.2109374999999998,
#         1.2890624999999998,
#         1.3671874999999998,
#         1.4453124999999998,
#         1.5234374999999998,
#         1.6015624999999998,
#         1.6796874999999998,
#         1.7578124999999998,
#         1.8359374999999998,
#         1.9140624999999996,
#         1.9921874999999996,
#         2.0703124999999996,
#         2.1484374999999996,
#         2.2265624999999996,
#         2.3046874999999996,
#         2.3828124999999996,
#         2.4609374999999996,
#         2.5390624999999996,
#         2.6171874999999996,
#         2.6953124999999996,
#         2.7734374999999996,
#         2.8515624999999996,
#         2.9296874999999996,
#         3.0078124999999996,
#         3.0859374999999996,
#         3.1640624999999996,
#         3.2421874999999996,
#         3.3203124999999996,
#         3.3984374999999996,
#         3.4765624999999996,
#         3.5546874999999996,
#         3.6328124999999996,
#         3.7109374999999996,
#         3.789062499999999,
#         3.867187499999999,
#         3.945312499999999,
#         4.023437499999999,
#         4.101562499999999,
#         4.179687499999999,
#         4.257812499999999,
#         4.335937499999999,
#         4.414062499999999,
#         4.492187499999999,
#         4.570312499999999,
#         4.648437499999999,
#         4.726562499999999,
#         4.804687499999999,
#         4.882812499999999,
#         4.960937499999999,
#     ]
# )
# grid_x = grid_x/1000  # 将单位转换为米
# grid_y = grid_x.copy()
# z = 12/1000  # 参考点z坐标

# #取出第一个样本的电场，作为[1,1,64,64]的张量
# Ex = u[0]  # 取出第一个样本的电场
# Ex = Ex[0] + Ex[1] * 1j  # 将实部和虚部组合成复数
# Ex = Ex.reshape(1,1, 64, 64)  # 将电场转换为二维数、


# print(Ex.shape)  # 输出电场的形状


# #创建二维数组
# result = np.empty((64,64), dtype=complex)  # 创建一个64x64的复数数组
# for i in range(64):
#     for j in range(64):
#         #result[i][j] = integrate_mode_real((grid_x[i], grid_y[j], zp), , k0=2*np.pi/0.6328, mu_r=1, L=(4,4), x_grid=0, y_grid=0)
#         result[i][j] = integrate_mode_real(grid_x[i], grid_y[j], z, Ex, k0=2*np.pi/0.0345, mu_r=1, L=(0.005,0.005), x_grid = grid_x, y_grid=grid_y, zp=6/1000)

# #绘制结果
# plt.imshow(result.real, cmap='viridis')
# plt.colorbar(label='Magnitude')
# plt.title('Integrated Result')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
