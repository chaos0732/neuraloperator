import torch
from torch.utils.data import Dataset


class CrossDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.a = data["a"]  # 输入结构图
        self.u = data["u"]  # 输出电场

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return {
        'x': self.a[idx],  # 输入
        'y': self.u[idx]   # 输出
    }

