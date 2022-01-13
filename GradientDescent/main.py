import sys
sys.path.append('Helpers')
from functions import *
import numpy as np
import matplotlib.pyplot as plt

raw = np.loadtxt('GradientDescent/data.txt', delimiter=',')
X = np.copy(raw)
X[:, 1] = X[:, 0]
X[:, 0] = 1
y = raw[:, 1]

[Theta, jHist] = gradientDescent(X, y)

predict = predict(X, Theta) * 10000

plt.figure(1)
plt.plot(X[:, 1], y, 'rx')
plt.plot(X[:, 1], predict/10000, '-b')
plt.show()

plt.figure(2)
plt.plot(jHist[:, 0], jHist[:, 1], '-r')
plt.show()