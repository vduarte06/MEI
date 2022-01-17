## Initialization
import numpy as np
import scipy.io
from displayData import *

# Load the Spam Email dataset

raw=scipy.io.loadmat('digits.mat')
X = raw['X']
m = X.shape[0]  # 5000
N = 25
# Randomly select N data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[np.arange(1,N+1)],:]
displayData(sel)
