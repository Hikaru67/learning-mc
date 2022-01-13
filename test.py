import numpy as np
from functions import *
# _a = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]
# _b = [ [ 2, 3], [7, 9] ]
# a = np.array(_a) #create 2 * 3 matrix: a
# b = np.array(_b) #create 2 * 3 matrix: b
# print(‘a .* b:’, a * b) #print out a .* b

_a = [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ]
a = np.array(_a)
a_i = np.linalg.pinv(a) #Create inverse of a
print(a_i)
print(predict(a, a_i))
print('haha')