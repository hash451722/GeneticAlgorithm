import numpy as np
import matplotlib.pyplot as plt

import crossover
import functions



def plot_pop(pop, gen=-1, range=(-10,10), file_name=None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pop[:, 0], pop[:, 1], marker='o', c="black")
    ax.grid()
    ax.set_title("Generation : "+str(gen))
    ax.set_xlim(range)
    ax.set_ylim(range)

    if file_name is None:
        plt.show()
    else:
        print(file_name)


def plot_statistics(fitness_stats):
    ''' 統計量のプロット '''
    x = np.arange(fitness_stats.shape[0])

    fig, ax = plt.subplots()
    ax.plot(x, fitness_stats[:, 0])
    ax.plot(x, fitness_stats[:, 1])
    ax.plot(x, fitness_stats[:, 2])
    ax.set_yscale("log")
    ax.grid()
    plt.show()



def initialize_population(dim, n_pop, func):
    ''' 個体集団の初期化 '''
    r = func.domain
    pop = np.random.default_rng().uniform(r[0], r[1], (n_pop, dim))
    return pop


def extract(pop, n):
    '''
    個体集団popからをn個の個体をランダムに非復元抽出する
    '''
    pop_e = np.empty([pop.shape[1]])
    for _ in range(n):
        k = np.random.randint(0, pop.shape[0], (1))[0]
        ind_e = pop[k, :]
        pop_e = np.vstack([pop_e, ind_e])
        pop = np.delete(pop, k, axis=0)

    pop_e = np.delete(pop_e, 0, 0)
    return pop, pop_e


def cal_fitness(pop, func):
    ''' 適応度（評価値）の計算を求める '''
    fitness = np.empty(pop.shape[0])
    for n, individual in enumerate(pop):
        fitness[n] = func.fx(individual)

    return fitness


def select_children(pop, fitness, n_selected):
    ''' 個体集団から適用度に応じてn個の個体を抽出 '''

    # 適応度によるランキング
    rank_index = np.argsort(fitness)

    for k, sorted_i in enumerate(rank_index):
        if k> n_selected-1:
            break
        if k==0:
            pop_selected = pop[sorted_i]
        else:
            pop_selected = np.vstack([pop_selected, pop[sorted_i]])

    return pop_selected


def shift_generation(pop, n_parent, n_children, func):
    # 個体集団popから親個体をn_parent個非復元抽出
    pop, pop_parent = extract(pop, n_parent)

    # 親個体から子個体n_children個を交叉によって生成
    pop_children = crossover.spx(pop_parent, n_children)

    # 生成した子個体の適応度（評価値）を計算
    fitness = cal_fitness(pop_children, func)

    # 子個体集団からn_parent個の子個体を適応度（評価値）をもとに選択
    pop_selected_children = select_children(pop_children, fitness, n_parent)

    # 選択した子個体を個体集団popに追加
    pop = np.vstack([pop, pop_selected_children])
    
    return pop


def statistics(pop, func):
    ''' 統計量 '''
    fitness = cal_fitness(pop, func)

    max_fitness = np.amax(fitness)
    min_fitness = np.amin(fitness)
    ave_fitness = np.mean(fitness)
    return np.array([max_fitness, min_fitness, ave_fitness])


def select_function(func_name):
    if func_name == "ackley":
        return functions.ackley()
    elif func_name == "sphere":
        return functions.sphere()
    elif func_name == "rosenbrock":
        return functions.rosenbrock()
    else:
        raise Exception("Incorrect function name.")




def main():
    ''' JGG + SPX '''
    func_name = "ackley"
    dim = 2 # 次元数
    gen_max = 200 # 世代数

    # 関数の選択　funcは関数のインスタンス
    func = select_function(func_name)

    n_pop = 30 * dim  # 個体集団の個体数 (JGGでは15n～50nを推奨)
    n_parent = dim + 1  #　親個体数 (SPXではn+1)
    n_children = 10 * dim  # 生成する子個体数 (JGGでは10nを推奨)

    # 初期個体集団popの生成
    pop = initialize_population(dim, n_pop, func)
    fitness_stats = statistics(pop, func)
    # plot_pop(pop, gen=0, range=func.range)

    for gen in range(1, gen_max+1):
        pop = shift_generation(pop, n_parent, n_children, func)
        fitness_stats = np.vstack([fitness_stats, statistics(pop, func)])
        # plot_pop(pop, gen=gen, range=func.range)

    plot_statistics(fitness_stats)



if __name__ == '__main__':
    main()