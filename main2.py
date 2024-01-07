import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# X轴的取值范围
X_BOUND2 = [-513, 513]

# Y轴的取值范围
Y_BOUND2 = [-513, 513]
# 绘制3D曲面图
def plot_3d2(ax):
    # 在X_BOUND范围内生成200个等间距的数
    X = np.linspace(*X_BOUND2, 200)
    # 在Y_BOUND范围内生成200个等间距的数
    Y = np.linspace(*Y_BOUND2, 200)
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
   
# X轴的取值范围
X_BOUND4 = [-6, 6]

# Y轴的取值范围
Y_BOUND4 = [-6, 6]

# 绘制3D曲面图
def plot_3d4(ax):
    # 在X_BOUND范围内生成200个等间距的数
    X = np.linspace(*X_BOUND4, 200)
    # 在Y_BOUND范围内生成200个等间距的数
    Y = np.linspace(*Y_BOUND4, 200)
    # 生成网格点坐标矩阵
    X, Y = np.meshgrid(X, Y)
    # 计算每个点的Z值
    Z = f2([X, Y])
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
    # 显示图像
    # plt.show()
class Population:
    def __init__(self, min_range, max_range, dim, factor, rounds, size, object_func, CR=0.75):
        self.min_range = min_range  # 种群中个体的最小值
        self.max_range = max_range  # 种群中个体的最大值
        self.dimension = dim  # 种群的维度
        self.factor = factor  # 缩放因子
        self.rounds = rounds  # 进化的轮数
        self.size = size  # 种群的大小
        self.cur_round = 1  # 当前的轮数
        self.CR = CR  # 交叉概率
        self.get_object_function_value = object_func  # 目标函数
        # 初始化种群
        self.individuality = [np.array([random.uniform(self.min_range, self.max_range) for s in range(self.dimension)]) for tmp in range(size)]
        self.object_function_values = [self.get_object_function_value(v) for v in self.individuality]  # 计算种群中每个个体的目标函数值
        self.mutant = None  # 变异体
 
    def mutate(self):  # 变异操作
        self.mutant = []
        for i in range(self.size):
            r0, r1, r2 = 0, 0, 0
            while r0 == r1 or r1 == r2 or r0 == r2 or r0 == i:  # 随机选择三个不同的个体
                r0 = random.randint(0, self.size-1)
                r1 = random.randint(0, self.size-1)
                r2 = random.randint(0, self.size-1)
            tmp = self.individuality[r0] + (self.individuality[r1] - self.individuality[r2]) * self.factor  # 变异操作
            for t in range(self.dimension):
                if tmp[t] > self.max_range or tmp[t] < self.min_range:  # 如果变异后的值超出范围，则重新随机生成
                    tmp[t] = random.uniform(self.min_range, self.max_range)
            self.mutant.append(tmp)  # 添加到变异体列表中
 
    def crossover_and_select(self):  # 交叉和选择操作
        for i in range(self.size):
            Jrand = random.randint(0, self.dimension)
            for j in range(self.dimension):
                if random.random() > self.CR and j != Jrand:  # 交叉操作
                    self.mutant[i][j] = self.individuality[i][j]
                tmp = self.get_object_function_value(self.mutant[i])  # 计算变异体的目标函数值
                if tmp < self.object_function_values[i]:  # 如果变异体的目标函数值小于原个体的目标函数值，则替换
                    self.individuality[i] = self.mutant[i]
                    self.object_function_values[i] = tmp
 
    def print_best(self):  # 打印最优个体
        m = min(self.object_function_values)  # 找到最小的目标函数值
        i = self.object_function_values.index(m)  # 找到最小目标函数值对应的个体
        print("轮数：" + str(self.cur_round))
        print("最佳个体：" + str(self.individuality[i]))
        print("目标函数值：" + str(m))
 
    def evolution(self, ax):  # 进化操作，添加一个参数ax，用于绘图
        global sca  # 声明sca为全局变量，以便在函数外部访问
        while self.cur_round < self.rounds:  # 当前轮数小于总轮数时，继续进化
            self.mutate()  # 变异
            self.crossover_and_select()  # 交叉和选择
            self.print_best()  # 打印最优个体
            self.cur_round = self.cur_round + 1  # 轮数加1
            if 'sca' in globals():  # 如果sca在全局变量中，移除上一次的散点图
                sca.remove()
            # 绘制新的散点图，表示种群的当前状态
            x = [ind[0] for ind in self.individuality]
            y = [ind[1] for ind in self.individuality]
            z = [self.get_object_function_value(ind) for ind in self.individuality]
            sca = ax.scatter(x, y, z, c='black', marker='o')
            plt.draw()  # 更新图形
            plt.pause(0.1)  # 暂停0.3秒

def f(v):
    return -(v[1]+47)*np.sin(np.sqrt(np.abs(v[1]+(v[0]/2)+47))) - v[0] * np.sin(np.sqrt(np.abs(v[0]-v[1]-47)))
def f2(v):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (v[0]**2 + v[1]**2)))
    term2 = np.exp(0.5 * (np.cos(2*np.pi*v[0]) + np.cos(2*np.pi*v[1])))
    return 20 + np.e + term1 - term2
# 创建一个图形
fig = plt.figure()
# 添加一个3D坐标轴
ax = fig.add_axes(Axes3D(fig))
# 切换到交互模式，用于连续绘图
plt.ion()  
# plot_3d2(ax)  # 绘制3D曲面图
plot_3d4(ax)
# p = Population(min_range=-513, max_range=513, dim=2, factor=0.8, rounds=100, size=100, object_func=f)
p = Population(min_range=-5.12, max_range=5.12, dim=2, factor=0.8, rounds=100, size=100, object_func=f2)
p.evolution(ax)  # 传入ax作为参数

# 关闭交互模式
plt.ioff()  
# 显示最终的3D图形
# plot_3d2(ax)  # 绘制3D曲面图
plot_3d4(ax)