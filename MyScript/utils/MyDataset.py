import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.a = data["a"]  # 输入结构图
        self.u = data["u"]  # 输出电场

    def getgrid(self, grid_path):

        grid_data = torch.load(self.pt_path)
        x = grid_data["x"]
        y = grid_data["y"]
        return x, y

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return {"x": self.a[idx], "y": self.u[idx] }  # 输入  # 输出
