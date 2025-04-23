import numpy as np
import torch
import os
from glob import glob

folder = "."
mask_files = sorted(glob(os.path.join(folder, "input_*.csv")))
ex_files = sorted(glob(os.path.join(folder, "Ex_*.csv")))

assert len(mask_files) == len(ex_files), "掩膜和电场文件数量不匹配！"

all_a, all_u = [], []

for mask_path, ex_path in zip(mask_files, ex_files):
    mask = np.loadtxt(mask_path, delimiter=",").reshape(64, 64)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    ex = np.loadtxt(ex_path, delimiter=",").reshape(64, 64)
    ex_tensor = torch.tensor(ex, dtype=torch.float32).unsqueeze(0)

    all_a.append(mask_tensor)
    all_u.append(ex_tensor)

a_tensor = torch.stack(all_a, dim=0)
u_tensor = torch.stack(all_u, dim=0)

torch.save({"a": a_tensor, "u": u_tensor}, "maxwell_dataset.pt")
