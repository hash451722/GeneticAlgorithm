import numpy as np
import matplotlib.pyplot as plt


def spx(pop_parent, n_children):
    '''
    Simplex crossover
    pop_parent: 親個体の2次元配列
    nc: 子個体の生成数

    次元数: n
    親個体数: Np = n+1
    '''
    n = pop_parent.shape[1]  # 次元数

    g = pop_parent.sum(axis=0) / pop_parent.shape[0]  # 親個体の重心
    epsilon = np.sqrt(n+2)  # 拡張率

    for i in range(n_children):
        r = []
        for k in range(n):
            r.append( np.power( np.random.rand(), 1/(k+1) ) )

        x = []
        for k in range(n+1):
            x.append( g + epsilon * ( pop_parent[k] - g ) )

        c = [np.zeros(n)]
        for k in range(1, n+1):
            c.append( r[k-1]*(x[k-1]-x[k]+c[k-1]) )

        if i == 0:
            pop_children = x[n] + c[n]
        else:
            pop_children = np.vstack([pop_children, x[n] + c[n]])

    return pop_children


def plot_poplation(pop_parent, pop_children):
    ''' 次元数が2の場合のみ '''
    g = pop_parent.sum(axis=0) / 3  # 親個体の重心

    plt.scatter(pop_children[:, 0], pop_children[:, 1], marker='x', c="green")
    plt.scatter(pop_parent[:, 0], pop_parent[:, 1], marker='^', c="red")
    plt.scatter(g[0], g[1], marker='o', c="blue")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    dim = 2  # 次元数
    n_parent = 3

    pop_parent = (np.random.rand(n_parent, dim) -0.5) *5
    pop_children = spx(pop_parent, 100)

    print(pop_children)

    plot_poplation(pop_parent, pop_children)
