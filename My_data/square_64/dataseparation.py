import torch
import numpy as np
from sklearn.model_selection import train_test_split

# 假设你已经有这两个变量
# all_a: [N, 1, H, W]
# all_u: [N, 2, H, W]
# 比如通过 read_comsol_multi_block 函数得到
data = torch.load("./My_data/square_64/square_all_64.pt")
all_a = data["a"]
all_u = data["u"]


# 获取样本数量
N = all_a.shape[0]
indices = np.arange(N)

# 使用 sklearn 划分
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# 划分数据集
a_train, u_train = all_a[train_idx], all_u[train_idx]
a_test, u_test = all_a[test_idx], all_u[test_idx]

# 保存数据集
train_path = "./My_data/square_64/square_train_64.pt"
test_path = "./My_data/square_64/square_test_64.pt"
torch.save({"a": a_train, "u": u_train}, train_path)
torch.save({"a": a_test, "u": u_test}, test_path)
