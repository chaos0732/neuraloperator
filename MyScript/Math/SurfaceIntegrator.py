import numpy as np


class SurfaceIntegrator:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        初始化积分器对象
        x: ndarray of shape (W,)，x方向坐标点
        y: ndarray of shape (H,)，y方向坐标点
        """
        self.x = x
        self.y = y
        self.W = len(x)
        self.H = len(y)

        # 判断是否均匀网格（用于自动选择积分方式）
        self.uniform_x = np.allclose(np.diff(x), x[1] - x[0])
        self.uniform_y = np.allclose(np.diff(y), y[1] - y[0])
        self.is_uniform = self.uniform_x and self.uniform_y

        if self.is_uniform:
            self.dx = x[1] - x[0]
            self.dy = y[1] - y[0]

    def trapezoidal_integrate(self, f: np.ndarray) -> complex:
        """
        主函数：计算面积积分 ∫∫ f(x, y) dxdy
        支持复数；自动选择是否用非均匀实现
        f: shape (H, W)，二维函数值
        返回复数积分结果
        """
        assert f.shape == (self.H, self.W), "函数网格尺寸不匹配"
        if self.is_uniform:
            return self._uniform_trapezoid(f)
        else:
            return self._nonuniform_trapezoid(f)

    def _uniform_trapezoid(self, f):
        weights = np.ones_like(f, dtype=np.float32)
        weights[0, :] *= 0.5
        weights[-1, :] *= 0.5
        weights[:, 0] *= 0.5
        weights[:, -1] *= 0.5
        weights[0, 0] = weights[0, -1] = weights[-1, 0] = weights[-1, -1] = 0.25
        return np.sum(f * weights) * self.dx * self.dy

    def _nonuniform_trapezoid(self, f):
        total = 0.0 + 0j
        for i in range(self.H - 1):
            for j in range(self.W - 1):
                dx = self.x[j + 1] - self.x[j]
                dy = self.y[i + 1] - self.y[i]
                area = dx * dy
                block_avg = (
                    f[i, j] + f[i + 1, j] + f[i, j + 1] + f[i + 1, j + 1]
                ) / 4.0
                total += block_avg * area
        return total

    def integrate_mode(self, J: np.ndarray, kxp: float, kyq: float) -> complex:
        """
        根据公式：∬ J(x, y) * e^{j(kx x + ky y)} dxdy
        计算远场展开模式系数
        J: 电流网格，shape = (H, W)，可以是复数
        kx, ky: 波矢分量
        """
        X, Y = np.meshgrid(self.x, self.y)
        phase = np.exp(1j * (kxp * X + kyq * Y))
        integrand = J * phase
        return self.trapezoidal_integrate(integrand)


def integrate_mode_real(
    x,y,z, Ex, k0, mu_r, L,  x_grid, y_grid, zp
) -> complex:
    """
    计算远场外推
    公式：\vec{E}_{\text{scat}}(\vec{r}) = k_0 \mu_r \cdot \frac{1}{L_x L_y} \sum_{p} \sum_{q} \frac{1}{2k_{mz}}  e^{-j k_{mz} |z - z'|} e^{-j(k_{xp} x + k_{yq} y)} \int_{S'} \frac{\eta_0}{\eta} \vec{E}(x',y') e^{j(k_{xp} x' + k_{yq} y')} \, ds'

    输入
    x,y,z: 网格坐标，函数输入
    Ex: 电场，函数输入[Np,Nq,H,W]张量
    参数
    k0: 波数
    mu_r: 相对介电常数
    Lx,Ly: 空间周期
    N_p,N_q: 横纵单元数
    x_grid,y_grid: 网格坐标偏移量
    zp: 参考点z坐标
    """
    # 中间变量
    # kxp, kyq: 波矢分量
    # kmz: 纵向波数
    # xp, yq: 网格坐标
    Np = 100  # 横向模式数
    Nq = 100  # 纵向模式数
    Lx = L[0]  # 空间周期
    Ly = L[1]  # 空间周期
    # 循环遍历p,q
    # 计算k0*mu_r/(Lx*Ly)
    first_term = k0 * mu_r / (Lx * Ly)
    sum = 0.0 + 0j
    for p in range(-Np,Np+1):
        for q in range(-Nq,Nq+1):
            # 计算kxp, kyq
            kxp = 2 * np.pi * p / Lx
            kyq = 2 * np.pi * q / Ly

            # 计算kmz
            kmz = np.sqrt(k0**2 - kxp**2 - kyq**2)
            #kmz共轭
            kmz = np.conj(kmz)

            # 计算网格坐标
            xp = p * Lx + x_grid
            yq = q * Ly + y_grid

            # 计算积分值
            integrator = SurfaceIntegrator(xp, yq)
            integral_value = integrator.integrate_mode(Ex[0, 0].cpu().numpy(), kxp, kyq)

            # 计算相位因子
            phase = (
                1
                / (2 * kmz)
                * np.exp(-1j * kmz * np.abs(z - zp))
                * np.exp(-1j * (kxp * x + kyq * y))
            )
            # 计算总和
            sum += integral_value * phase

    result = first_term * sum
    return result


# # 测试代码
# x = np.linspace(0, 1, 64)
# y = np.linspace(0, 1, 64)
# X, Y = np.meshgrid(x, y)
# kx = np.pi
# ky = np.pi
# f = np.exp(1j * (kx * X + ky * Y))  # 示例函数
# testintegrator = SurfaceIntegrator(x, y)
# result = testintegrator.trapezoidal_integrate(f)
# print("Result of integration:", result)

# print("Shape of grid:", f.shape)
