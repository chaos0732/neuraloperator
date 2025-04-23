import torch
import numpy as np
import os
from glob import glob


def read_Ex(filename, num_samples, data_shape=(64, 64), skip_lines=0):
    """
    从一个 COMSOL 导出的大文件中读取多个 block 结果
    假设每个样本是连续的 [64 x 64] 行，之间用注释或空行隔开

    :param filename: COMSOL 导出的包含多个解的 txt 文件
    :param num_samples: 一共导出多少个解
    :param data_shape: 每个图像的尺寸，如 (64, 64)
    :param skip_lines: 每个样本之前跳过几行注释（如 '% real(emw.Ex)'）
    :return: 一个形状为 [num_samples, H, W] 的 numpy 数组
    """
    H, W = data_shape

    Ex = []  # 存储所有的 Ex 张量
    with open(filename, "r") as f:
        lines = f.readlines()

    # 去掉空行和注释
    content_lines = [
        line.strip() for line in lines if line.strip() and not line.startswith("%")
    ]

    # 从content_lines中读取网格坐标
    grid_x = np.array([float(x) for x in content_lines[0].split(",")])
    grid_y = np.array([float(y) for y in content_lines[1].split(",")])
    i = 2  # 当前读行位置
    for s in range(num_samples):
        # 读取实部
        data_lines = content_lines[i:i+H]
        Ex_real = np.loadtxt(data_lines, delimiter=",").reshape(data_shape)
        i += H + skip_lines  # 更新行位置

        # 读取虚部
        data_lines = content_lines[i:i+H]
        Ex_imag = np.loadtxt(data_lines, delimiter=",").reshape(data_shape)
        # 更新行位置
        i += H + skip_lines
        ex_tensor = torch.tensor(
            np.stack([Ex_real, Ex_imag]), dtype=torch.float32
        )
        Ex.append(ex_tensor)

    return grid_x, grid_y, Ex

def read_input(filename, num_samples, data_shape=(64, 64), skip_lines=0):
    """
    从一个 COMSOL 导出的大文件中读取多个 block 结果
    假设每个样本是连续的 [64 x 64] 行，之间用注释或空行隔开

    :param filename: COMSOL 导出的包含多个解的 txt 文件
    :param num_samples: 一共导出多少个解
    :param data_shape: 每个图像的尺寸，如 (64, 64)
    :param skip_lines: 每个样本之前跳过几行注释（如 '% real(emw.Ex)'）
    :return: 一个形状为 [num_samples, H, W] 的 numpy 数组
    """
    H, W = data_shape

    mask = []  # 存储所有的 Ex 张量
    with open(filename, "r") as f:
        lines = f.readlines()

    # 去掉空行和注释
    content_lines = [
        line.strip() for line in lines if line.strip() and not line.startswith("%")
    ]

    # 从content_lines中读取网格坐标
    grid_x = np.array([float(x) for x in content_lines[0].split(",")])
    grid_y = np.array([float(y) for y in content_lines[1].split(",")])
    i = 2  # 当前读行位置
    for s in range(num_samples):
        # 读取掩膜
        data_lines = content_lines[i:i+H]
        mask_tensor = np.loadtxt(data_lines, delimiter=",").reshape(data_shape)
        unique_vals = np.unique(mask_tensor)
        if len(unique_vals) == 2:
            high_val = max(unique_vals)
            mask_tensor = (mask_tensor == high_val).astype(np.float32)
        mask.append(torch.tensor(mask_tensor, dtype=torch.float32).unsqueeze(0))

        # 更新行位置
        i += H + skip_lines

    return grid_x, grid_y, mask

folder = os.getcwd()
folder = os.path.join(folder, "My_data/square_64")  # 数据文件夹路径
mask_files = glob(os.path.join(folder, "input.csv"))
ex_files = glob(os.path.join(folder, "Ex.csv"))

# 读取数据
datashape = (64, 64)  # 数据形状
num_samples = 100  # 一共导出多少个解

all_a, all_u = [], []
grid_x, grid_y,Ex = read_Ex(ex_files[0], num_samples, data_shape=datashape, skip_lines=0)
grid_x, grid_y, mask = read_input(mask_files[0], num_samples, data_shape=datashape, skip_lines=0)
all_a = torch.stack(mask, dim=0)
all_u = torch.stack(Ex, dim=0)
# 将数据保存为 PyTorch 的 .pt 文件
# 保存数据
savepath = os.path.join("./My_data/square_64", "square_all_64.pt")
torch.save({"a": all_a, "u": all_u, "gridx": grid_x, "gridy": grid_y}, savepath)

data = torch.load(savepath)
test1 = data["a"][10,0]
testoutput = data["u"][10,0]
print(data["a"].shape)  # 输出 a 的形状
print(data["u"].shape)  # 输出 u 的形状
