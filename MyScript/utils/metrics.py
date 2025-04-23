# MyScript/utils/metrics.py

import torch
import matplotlib.pyplot as plt
import numpy as np


def histogram_l2(
    pred, u, bins=50, title="Histogram of Relative L2 Error", save_path=None, threshold=0.05
):
    assert pred.shape == u.shape, "预测与真实数据维度不一致"

    errors = []
    with torch.no_grad():
        for i in range(u.shape[0]):
            pred_i = pred[i]
            u_i = u[i]
            rel_l2 = torch.norm(pred_i - u_i) / torch.norm(u_i)
            errors.append(rel_l2.item())

    mean_error = np.mean(errors)
    print(f"平均 Relative L2 Error: {mean_error:.6f}")

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=bins, alpha=0.7, color="blue", edgecolor="black",density=True)
    plt.axvline(
        threshold,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label="5% Threshold",
    )
    plt.title(title)
    plt.xlabel("Relative L2 Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    #统计误差低于阈值的样本比例
    count = np.sum(np.array(errors) < threshold)
    total = len(errors)
    ratio = count / total

    #输出低于阈值的样本数量和比例
    print(f"误差低于 {threshold} 的样本比例: {ratio:.2%}")
    print(f"误差低于 {threshold} 的样本数量: {count}/{total}")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"直方图已保存到 {save_path}")
    else:
        plt.show()
