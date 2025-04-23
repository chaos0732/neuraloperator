import numpy as np
import torch
import os
from glob import glob


def readEx(file_path: str, data_shape=(64, 64)):
    """
    从 COMSOL 导出的 .csv 文件中提取网格坐标和数值数据
    返回 grid_x, grid_y, data_array
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # 去掉空行和注释
    content_lines = [
        line.strip() for line in lines if line.strip() and not line.startswith("%")
    ]
    # 第1行是x坐标，第2行是y坐标
    grid_x = np.array([float(x) for x in content_lines[0].split(",")])
    grid_y = np.array([float(y) for y in content_lines[1].split(",")])
    # 读取实数
    data_lines = content_lines[2:2+data_shape[0]]
    Ex_real = np.loadtxt(data_lines, delimiter=",").reshape(data_shape)
    # 读取虚部
    data_lines = content_lines[2+data_shape[0]:]
    Ex_imag = np.loadtxt(data_lines, delimiter=",").reshape(data_shape)
    return grid_x, grid_y, Ex_real, Ex_imag

def readInput(file_path: str, data_shape=(64, 64)):
    """
    从 COMSOL 导出的 .csv 文件中提取网格坐标和数值数据
    返回 grid_x, grid_y, data_array
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # 去掉空行和注释
    content_lines = [
        line.strip() for line in lines if line.strip() and not line.startswith("%")
    ]
    # 第1行是x坐标，第2行是y坐标
    grid_x = np.array([float(x) for x in content_lines[0].split(",")])
    grid_y = np.array([float(y) for y in content_lines[1].split(",")])
    # 读取输入掩膜
    data_lines = content_lines[2:2+data_shape[0]]
    mask = np.loadtxt(data_lines, delimiter=",").reshape(data_shape)
    unique_vals = np.unique(mask)
    if len(unique_vals) == 2:
        high_val = max(unique_vals)
        mask = (mask == high_val).astype(np.float32)
    return grid_x, grid_y, mask

folder = os.getcwd()
folder = os.path.join(folder, "My_data/unit_64")  # 数据文件夹路径
mask_files = sorted(glob(os.path.join(folder, "input_*.csv")))
ex_files = sorted(glob(os.path.join(folder, "Ex_*.csv")))

assert len(mask_files) == len(ex_files), "掩膜和电场文件数量不匹配！"
datashape = (64, 64)  # 数据形状
all_a, all_u = [], []

for mask_path, ex_path in zip(mask_files, ex_files):

    grid_x,grid_y, mask = readInput(mask_path, data_shape=datashape)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    grid_x, grid_y, Ex_real, Ex_imag = readEx(ex_path, data_shape=datashape)
    ex_tensor = torch.tensor(np.stack([Ex_real,Ex_imag]), dtype=torch.float32).unsqueeze(0)
    all_a.append(mask_tensor)
    all_u.append(ex_tensor)

a_tensor = torch.stack(all_a, dim=0)
u_tensor = torch.stack(all_u, dim=0)
savepath = os.path.join("./neuralop/data/datasets/data", "unit_train_64.pt")
torch.save({"a": a_tensor, "u": u_tensor}, savepath)
