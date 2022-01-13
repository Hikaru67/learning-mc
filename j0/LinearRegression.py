import numpy as np
import sys
sys.path.append('Helpers')
from functions import *

raw = np.loadtxt('j0/data.txt', delimiter=',')

y = raw[:, 2]

X = np.zeros((np.size(y),np.size(raw,1)))
X[:, 0] = 1
X[:,1:] = raw[:,0:2]
theta = np.array([89597.909542,139.210674 ,-8738.019112]) #bộ theta chính xác là [89597.909542,139.210674 ,-8738.019112]
print(computeCost(X, y, theta), computeCost_Vec(X, y, theta))