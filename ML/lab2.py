import matplotlib.pyplot as plt
import numpy as np
import csv

def read_data():
    with open('ex1data.txt') as f:
        return np.loadtxt(f, delimiter = ",")

def teta1(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return np.sum((x-x_mean)*(y-y_mean) /np.sum(np.square(x-x_mean)))

def teta0(x, y):
    return  np.mean(y) - teta1(x,y)*np.mean(x)

def best_fit(x,y):
    t0 = teta0(x, y)
    t1 = teta1(x, y)
    return t0 + t1*x

def residuals(x, y):
    m = len(x)
    return 1/2*m*(x-y).T*(x-y)

def cost_function(x, y):
    m = len(x)
    r = residuals(x,y)
    return np.sum(r[1:])

if __name__ == '__main__':
    data = read_data()
    #plot(data[:,0], data[:,1])
    x = data[:,0]
    y = data[:,1]
    #plt.xlim([0, 25])
    #plt.ylim([0, 25])

    plt.plot(x, y, 'bo')
    plt.plot(x, best_fit(x,y))
    plt.plot(x, residuals(x,y))
    print(cost_function(x,y))
    plt.show()
