import numpy as np
import matplotlib.pyplot as plt

# Supervised Learning (Học có giám sát)

# m: Số mẫu train (số dòng)
# n: Số feature (input), chính là số cột x.
# x(i): Hàng i của các feature.
# y(i): Hàng i của output.
# xj(i): Feature j, hàng i.

X = np.loadtxt('univariate.txt', delimiter = ',')
Theta = np.loadtxt('univariate_theta.txt', delimiter = ',')
y = np.copy(X[:,-1])
X[:,1] = X[:,0]
X[:,0] = 1

#Tính lợi nhuận (đơn vị 10000$)
predict = X @ Theta
#Chuyển lợi nhuận về đơn vị $
predict = 10000 * predict
#in cặp dân số-lợi nhuận đầu tiên (đơn vị dân số: người)
print('%d nhan ma: %.2f$' %(X[0,1]*10000, predict[0]))
np.savetxt('predicted_value.txt', predict, fmt='%.6f')

#Plot giá trị thực tế (không lấy cột bias 1 đầu)
#X[:,1:] là x-axis của biểu đồ, không lấy cột đầu; y là y-axis, rx là red x, plot dữ liệu bằng dấu x màu đỏ
# plt.plot(X[:,1:],y,'rx')
#Plot dự đoán
# plt.plot(X[:,1:],predict/10000,'-b')#ta dùng đơn vị gốc là 10000$, -b là đường thẳng màu xanh
#show kết quả
# plt.show()

#Multivariate

_X = np.loadtxt('multivariate.txt', delimiter = ',')
Theta = np.loadtxt('multivariate_theta.txt')

X = np.zeros((np.size(_X, 0), np.size(_X, 1)))
X[:, 0] = 1

n = np.size(_X, 1) - 1
X[:,1:] = _X[:,0:n]
y = np.copy(X[:,1])

predict = X @ Theta
print('%.2f feet-2, %d phong` ngu?: %.1f$' %(X[0, 1], X[0, 2], predict[0]))
np.savetxt('predicted_mul_value.txt', predict, fmt = '%.2f')