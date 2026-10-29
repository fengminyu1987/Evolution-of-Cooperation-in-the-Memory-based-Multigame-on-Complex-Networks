import networkx as nx
import numpy as np
import random
from matplotlib import pyplot as plt
import json


# 生成具有周期性边界的方格网络（N*N）
def square_lattice(N):
    G = nx.generators.lattice.grid_graph(dim=[1, N, N])
    # 构造周期性边界
    for i in range(N):
        G.add_edge((0, i, 0), (N - 1, i, 0))
        G.add_edge((i, 0, 0), (i, N - 1, 0))
    return G


n = 1000
ETime = 10000
deltalist = np.arange(0, 1.05, 0.05)
blist = np.arange(1, 2.05, 0.05)
T = 5  # 记忆长度
c = 0.2  # 记忆强度
rho = 0.8  # 进行SDG玩家的比例
# G = nx.barabasi_albert_graph(n, 3)  # 无标度网络
G = nx.watts_strogatz_graph(n, 4, 0.15)  # 小世界网络
fun_f = lambda x: (len(x) - sum(list(map(lambda x: x[-1], list(x.values()))))) / len(x)  # 根据策略字典计算合作者比例

result = {}
for delta in deltalist:
    delta_f = []
    for b in blist:
        pom_0 = [[1, delta], [b, 0]]  # SDG
        D0 = b
        pom_1 = [[1, -delta], [b, 0]]  # PDG
        D1 = b + delta
        f_c = []
        state = []  # 玩家状态字典 (SDG为0，PDG为1)
        strategy = {}  # 玩家策略字典
        # 初始化
        for i in G.nodes:
            if 0 <= random.uniform(0, 1) < rho:
                state.append(0)
            else:
                state.append(1)
            strategy[i] = [0 <= random.uniform(0, 1) < 0.5]  # True为Defector,False为Cooperator
        f_c.append(fun_f(strategy))
        tnow = 0
        while tnow < ETime:
            print("\r" + "delta={:.2f},b={:.2f},fc={:.3f},t:{:.2f}/{:.2f}".format(delta, b, f_c[-1], tnow, ETime),
                  end='')
            # 计算每个时间节点玩家的收益
            Payoff = []
            for x in G.nodes:
                Ux = 0
                x_nei = list(G.neighbors(x))
                if state[x] == 0:  # 玩家进行SDG
                    for j in x_nei:
                        Ux += pom_0[strategy[x][tnow]][strategy[j][tnow]]
                elif state[x] == 1:  # 玩家进行PDG
                    for j in x_nei:
                        Ux += pom_1[strategy[x][tnow]][strategy[j][tnow]]
                Payoff.append(Ux)
            # 更新每个玩家的策略
            for x in G.nodes:
                x_nei = list(G.neighbors(x))
                y = random.choice(x_nei)  # 随机选出一个邻居节点
                s_change = strategy[y][tnow]
                if s_change == False:  # the target is a cooperator
                    memory = (len(strategy[x][-T:]) - sum(strategy[x][-T:])) / len(strategy[x][-T:])
                else:
                    memory = sum(strategy[x][-T:]) / len(strategy[x][-T:])
                if state[x] == 0:  # 玩家进行SDG
                    ran = c * memory + (1 - c) * ((Payoff[y] - Payoff[x]) / (
                                D0 * max(len(list(G.neighbors(x))), len(list(G.neighbors(y))))))
                    if 0 <= random.uniform(0, 1) <= ran:
                        strategy[x].append(strategy[y][tnow])
                    else:
                        strategy[x].append(strategy[x][tnow])
                else:  # 玩家进行PDG
                    ran = c * memory + (1 - c) * ((Payoff[y] - Payoff[x]) / (
                                D1 * max(len(list(G.neighbors(x))), len(list(G.neighbors(y))))))
                    if 0 <= random.uniform(0, 1) <= ran:
                        strategy[x].append(strategy[y][tnow])
                    else:
                        strategy[x].append(strategy[x][tnow])
            f_c.append(fun_f(strategy))
            tnow += 1
        delta_f.append(np.mean(f_c[-500:]))
    result["delta = {:.2f}".format(delta)] = delta_f