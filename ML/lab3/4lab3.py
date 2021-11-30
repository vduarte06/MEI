import numpy as np
import matplotlib.pyplot as plt

def h_theta(x):
    return 1/(1+np.exp(-x))
plt.figure(0)
x = np.array([-4, -3, -2, -1, 0 ,1,2,3,4])
plt.plot(h_theta(x))
plt.xlim((-4, 4))
plt.show()