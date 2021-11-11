import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class ackley():
    '''
    -32.768 < x < 32.768
    fx = 0 at x = (0, 0, ..., 0)
    '''
    def __init__(self):
        self.domain = (-32.768, 32.768)
    def fx(self, x):
        t1 = -20* np.exp( -1/5 * np.sqrt( 1/len(x) * np.sum(x*x) ) )
        t2 = -np.exp(1/len(x) * np.sum( np.cos(2*np.pi*x) ) )
        t3 = 20
        t4 = np.e
        return t1 + t2 + t3 + t4


class sphere():
    '''
    -5.12 < x < 5.12
    fx = 0 at x = (0, 0, ..., 0)
    '''
    def __init__(self):
        self.domain = (-5.12, 5.12)
    def fx(self, x):
        return np.sum(x*x)


class rosenbrock():
    '''
    -5 < x < 5
    fx = 0 at x = (1, 1, ..., 1)
    '''
    def __init__(self):
        self.domain = (-5, 5)
    def fx(self, x):
        sum = 0 
        for i in range(len(x)-1):
            sum = sum + np.power((x[i]-1), 2) + 100* np.power( x[i+1]- np.power(x[i],2) ,2)
        return sum


def select_function(func_name):
    if func_name == "ackley":
        return ackley()
    elif func_name == "sphere":
        return sphere()
    elif func_name == "rosenbrock":
        return rosenbrock()
    else:
        raise Exception("Incorrect function name.")


def plot2d(func):
    min = func.domain[0]
    max = func.domain[1]

    x = np.arange(min, max, 0.04)
    y = np.arange(min, max, 0.04)
    xx, yy = np.meshgrid(x, y)
    zz = np.empty(xx.shape)

    for i in range(zz.shape[0]):
        for j in range(zz.shape[1]):
            zz[i][j] = func.fx(np.array( [xx[i][j], yy[i][j]] ))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()


if __name__ == '__main__':
    func = select_function("sphere")
    plot2d(func)

    a = func.fx(np.array([0,0]))
    print(a)
    print(np.array([0,0]))
