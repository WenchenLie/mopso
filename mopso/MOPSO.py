from typing import Callable
import numpy as np


class _Swarm:
    """种群类"""
    def __init__(self, m: int, n: int, bounds: np.ndarray):
        """初始化种群

        Args:
            m (int): 粒子数
            n (int): 维度
            bounds (np.ndarray): 位置边界(m, n)
        """
        self.m = m  # 粒子数
        self.n = n  # 维度
        self.bounds = bounds  # 边界(m, n)
        self.x: np.ndarray = np.zeros((m, n))  # 位置(m, n)
        for i in range(n):
            self.x[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], m)  # 初始化随机位置
        self.v: np.ndarray = np.zeros((m, n))  # 初始化速度
        self.pbest = self.x.copy()  # 个体最优位置(m, n)
        self.gbest: np.ndarray | None = None  # 全局最优位置(m, n)


class MOPSO:
    """多目标粒子群优化"""
    def __init__(self,
        objectives: list[Callable],
        m: int,
        n: int,
        bounds: np.ndarray,
        max_iter: int,
        archive_size: int = None,
        w1: float = 0.9,
        w2: float = 0.4,
        c1: float = 1.0,
        c2: float = 1.0,
    ):
        """多目标粒子群优化

        Args:
            objectives (list[Callable]): 目标函数列表
            m (int): 粒子数量
            n (int): 维度
            bounds (np.ndarray): 位置边界(m, 2)
            max_iter (int): 最大迭代数
            archive_size (int, optional): 存档数量
            w1 (float, optional): 初始惯性权重
            w2 (float, optional): 最终惯性权重
            c1 (float, optional): 学习因子
            c2 (float, optional): 学习因子
        
        Note-1:
        -------
        `m`: 粒子数  
        `n`: 维度  
        `bounds`: 位置边界(m, n)
        
        Note-2:
        -------
        假设所有目标函数均为最小化问题，即目标函数值越小越好
        """
        self.objectives = objectives
        self.m = m
        self.n = n
        self.bounds = bounds
        self.max_iter = max_iter
        self.archive_size = archive_size
        self.w1 = w1
        self.w2 = w2
        self.c1 = c1
        self.c2 = c2
        self.n_obj = len(objectives)
        self.swarm = _Swarm(m, n, bounds)  # 初始化种群
        self.x = self.swarm.x  # 位置
        self.v = self.swarm.v  # 速度
        self.x_init = self.x.copy()  # 初始处置
        self.archive: np.ndarray | None = None  # 外部存档(位置)(*, n)
        self.archive_y: np.ndarray | None = None  # 外部存档(目标值)(*, n_obj)
        self.y = self._get_y(self.x)  # 目标值(m, n_obj)
        self.pbest = self.x.copy()
        self.pbest_y = self.y.copy()

    def _get_y(self, x: np.ndarray) -> np.ndarray:
        """计算目标函数值
        Args:
            x (np.ndarray): 种群位置(m, n)

        Returns:
            np.ndarray: 目标函数值(m, n_obj)
        """
        return np.vstack([f(x) for f in self.objectives]).T

    def _get_gbest(self) -> np.ndarray:
        """为每个粒子随机选择一个全局最优位置，采用轮盘赌规则，拥挤距离越大，被选择概率越大

        Returns:
            np.ndarray: 全局最优位置(m, n)
        """
        gbest = np.zeros((self.m, self.n))
        if self.archive is None or len(self.archive) == 0:
            # 若存档为空，随机生成gbest
            for i in range(self.n):
                gbest[:, i] = np.random.uniform(self.bounds[i, 0],
                                                  self.bounds[i, 1], self.m)
        else:
            cd = self._crowding_distance(self.archive_y)

            cd_without_inf = cd[cd != np.inf]
            if len(cd_without_inf) > 0:
                cd_max = max(cd_without_inf)
            else:
                cd_max = 1

            fitness = np.zeros_like(cd)
            for i, d in enumerate(cd):
                if d == np.inf:
                    fitness[i] = 10 * cd_max
                else:
                    fitness[i] = d

            if np.sum(fitness) == 0:
                prob = np.ones(len(self.archive)) / len(self.archive)
            else:
                prob = fitness / np.sum(fitness)  # 轮盘赌概率
            selected_indices = np.random.choice(len(self.archive), size=self.m,
                                                p=prob)
            gbest = self.archive[selected_indices]

        return gbest

    def _update_archive(self):
        """更新外部存档"""
        # 将种群中非支配解添加至存档
        non_dominated = []
        for i in range(self.m):
            is_dominated = False  # 当前粒子(x[i])是否被支配
            for j in range(self.m):
                if i == j:
                    continue
                if self._dominate(self.y[j], self.y[i]):
                    is_dominated = True  # 当前种群中至少有一个解支配xi，不添加至存档
                    break
            if not is_dominated:
                non_dominated.append(self.x[i])
        if self.archive is None:
            self.archive = np.array(non_dominated)
        else:
            self.archive = np.vstack([self.archive, non_dominated])
        self.archive_y = self._get_y(self.archive)

        # 检查存档，移除被支配的解
        if len(self.archive) > 0:
            non_dominated_archive = []  # 存档中的非支配解
            for i in range(len(self.archive)):
                is_dominated = False  # 存档中当前解是否被支配
                for j in range(len(self.archive)):
                    if i == j:
                        continue
                    if self._dominate(self.archive_y[j], self.archive_y[i]):
                        is_dominated = True  # 存档中至少有一个解支配xi，不保留
                        break
                if not is_dominated:
                    non_dominated_archive.append(self.archive[i])
            self.archive = np.array(non_dominated_archive)
            self.archive_y = self._get_y(self.archive)

        # 修剪存档
        if self.archive_size is not None and len(self.archive) > self.archive_size:
            cd = self._crowding_distance(self.archive_y)  # 存档的拥挤距离
            idx = np.argsort(cd)[::-1]  # 按拥挤距离排序(从大到小)
            self.archive = self.archive[idx[:self.archive_size]]
            self.archive_y = self._get_y(self.archive)

    def optimize(self):
        """开始优化"""
        for gen in range(self.max_iter):
            w = self.w1 - (self.w1 - self.w2) * gen / self.max_iter  # 自适应惯性权重
            gbest = self._get_gbest()  # 全局最优位置(m, n)
            r1, r2 = np.random.rand(self.m, self.n), np.random.rand(self.m, self.n)
            pbest = self.swarm.pbest  # 个体最优位置(m, n)

            # 更新速度和位置
            self.v = w * self.v + \
                     self.c1 * r1 * (pbest - self.x) + \
                     self.c2 * r2 * (gbest - self.x)
            self.x += self.v
            self.x = np.clip(self.x, self.bounds[:, 0], self.bounds[:, 1])
            self.y = self._get_y(self.x)  # 目标值(m, n_obj)

            # 更新个体最优位置pbest
            for i in range(self.m):
                if self._dominate(self.y[i], self.pbest_y[i]):
                    self.pbest[i] = self.x[i]
                    self.pbest_y[i] = self.y[i]
                elif not self._dominate(self.pbest_y[i], self.y[i]) and np.random.rand() < 0.5:
                    # 如果互不支配，则以一定概率更新，保持多样性
                    self.pbest[i] = self.x[i].copy()
                    self.pbest_y[i] = self.y[i].copy()

            # 更新外部存档
            self._update_archive()

            if (gen + 1) % 10 == 0 or gen + 1 == self.max_iter:
                print(f"Iter {gen + 1}/{self.max_iter}, Archive Size: {len(self.archive) if self.archive is not None else 0}")

    @staticmethod
    def _crowding_distance(y: np.ndarray) -> np.ndarray:
        """根据目标函数值计算拥挤距离

        Args:
            y (np.ndarray): 目标函数值(m, n_obj)

        Returns:
            np.ndarray: 拥挤距离(m,)
        """
        m, n_obj = y.shape
        if m <= 2:
            return np.full(m, np.inf)

        cd = np.zeros(m)  # 拥挤距离

        for i in range(n_obj):
            # 按第i个目标函数计算拥挤距离
            idx = np.argsort(y[:, i])  # 按目标函数值排序(从小到大)
            sorted_y = y[idx, i]
            y_range = sorted_y[-1] - sorted_y[0]
            if y_range < 1e-12:
                continue  # 目标函数值相同，不计算拥挤距离

            cd[idx[0]] = np.inf  # 边界点拥挤距离设为无穷大
            cd[idx[-1]] = np.inf

            for j in range(1, m - 1):
                # 计算中间点的拥挤距离
                cd[idx[j]] += (sorted_y[j + 1] - sorted_y[j - 1]) / y_range

        return cd

    def visualize2d(self):
        """可视化二维目标空间"""
        if self.n_obj != 2:
            print('只能可视化二维目标空间')
            return
        import matplotlib.pyplot as plt
        if self.archive is not None and len(self.archive) > 0:
            archive_y = self._get_y(self.archive)
            plt.scatter(archive_y[:, 0], archive_y[:, 1], c='blue', label='Pareto Front')
        plt.scatter(self.y[:, 0], self.y[:, 1], c='red', alpha=0.5, label='Particles')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('MOPSO Optimization')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def _dominate(y1: np.ndarray, y2: np.ndarray) -> bool:
        """判断y1是否支配y2

        Args:
            y1 (np.ndarray): 目标函数值1(n_obj,)
            y2 (np.ndarray): 目标函数值2(n_obj,)

        Returns:
            bool: 若y1支配y2，返回True，否则返回False

        Note:
        ----
        对于最小化问题，y1支配y2当且仅当:
        * 对于所有目标，y1 <= y2
        * 至少存在一个目标，y1 < y2
        """
        return np.all(y1 <= y2) and np.any(y1 < y2)

