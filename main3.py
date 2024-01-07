import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# X轴的取值范围
X_BOUND3 = [-513, 513]

# Y轴的取值范围
Y_BOUND3 = [-513, 513]
# 绘制3D曲面图
def plot_3d3(ax):
    # 在X_BOUND范围内生成200个等间距的数
    X = np.linspace(*X_BOUND3, 200)
    # 在Y_BOUND范围内生成200个等间距的数
    Y = np.linspace(*Y_BOUND3, 200)
    # 生成网格点坐标矩阵
    X, Y = np.meshgrid(X, Y)
    # 计算每个点的Z值
    Z = f([X, Y])
    # 绘制3D曲面图
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    # 设置x轴标签
    ax.set_xlabel('x')
    # 设置y轴标签
    ax.set_ylabel('y')
    # 设置z轴标签
    ax.set_zlabel('z')
    # 暂停3秒
    plt.pause(3)
    # # 显示图像
    # plt.show()

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([np.random.uniform(-1, 1) for _ in range(len(bounds))])
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf

    def update_velocity(self, global_best, w=0.7, c1=1.4, c2=1.4):
        r1 = np.random.uniform(0, 1, size=len(self.velocity))
        r2 = np.random.uniform(0, 1, size=len(self.velocity))
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.maximum(self.position, np.array(bounds)[:, 0])
        self.position = np.minimum(self.position, np.array(bounds)[:, 1])

class Swarm:
    def __init__(self, func, particle_count, bounds):
        self.particles = [Particle(bounds) for _ in range(particle_count)]
        self.global_best = None
        self.func = func

    def optimize(self, iterations, ax):
        global sca  # 声明sca为全局变量，以便在函数外部访问
        for _ in range(iterations):
            for particle in self.particles:
                fitness = self.func(particle.position)
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                if self.global_best is None or fitness > self.func(self.global_best):
                    self.global_best = particle.position.copy()
            for particle in self.particles:
                particle.update_velocity(self.global_best)
                particle.update_position(bounds)
                global sca  # 声明sca为全局变量，以便在函数外部访问
            if 'sca' in globals():  # 如果sca在全局变量中，移除上一次的散点图
                sca.remove()
            # 绘制新的散点图，表示种群的当前状态
            x = [particle.position[0] for particle in self.particles]
            y = [particle.position[1] for particle in self.particles]
            z = [self.func(particle.position) for particle in self.particles]
            sca = ax.scatter(x, y, z, c='black', marker='o')
            plt.draw()  # 更新图形
            plt.pause(0.1)  # 暂停0.3秒
        return self.global_best

def f(v):
    return -(v[1]+47)*np.sin(np.sqrt(np.abs(v[1]+(v[0]/2)+47))) - v[0] * np.sin(np.sqrt(np.abs(v[0]-v[1]-47)))

# 创建一个图形
fig = plt.figure()
# 添加一个3D坐标轴
ax = fig.add_axes(Axes3D(fig))
# 切换到交互模式，用于连续绘图
plt.ion()  
plot_3d3(ax)  # 绘制3D曲面图
bounds = [(-513, 513), (-513, 513)]
swarm = Swarm(f, particle_count=100, bounds=bounds)  # 传入ax作为参数
result = swarm.optimize(iterations=100, ax=ax)  # 传入ax作为参数
print("最优解：", result)
print("目标函数值：", f(result))

# 关闭交互模式
plt.ioff()  
# 显示最终的3D图形
plot_3d3(ax)  # 绘制3D曲面图
