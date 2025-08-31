import numpy as np
from mopso.MOPSO import MOPSO


if __name__ == "__main__":
    np.random.seed(42)

    # ZDT1问题
    def f1(x: np.ndarray):
        """x: (m, n)"""
        return x[:, 0]

    def f2(x: np.ndarray):
        """x: (m, n)"""
        n = x.shape[1]
        g = 1 + 9 / (n - 1) * np.sum(x[:, 1:], axis=1)
        h = 1 - np.sqrt(f1(x) / g)
        return g * h

    bounds = np.array([
        [0, 1],
        [0, 1]
    ])  # 位置边界(m, n)
    pso = MOPSO(
        objectives=[f1, f2],
        m=100,
        n=2,
        bounds=bounds,
        max_iter=100,
        archive_size=20,
    )
    pso.optimize()
    pso.visualize2d()
