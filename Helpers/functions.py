#file functions.py
import numpy as np

def printProgressBar(iteration, total, suffix = ''):
    percent = ("{0:." + str(1) + "f}").format(100 * ((iteration + 1) / float(total)))
    filledLength = int(50 * iteration // total)
    bar = '=' * filledLength + '-' * (50 - filledLength)
    print('\rTraining: |%s| %s%%' % (bar, percent), end = '\r')
    #Print new Line on Complete
    if (iteration == total):
      print()

def predict(X, Theta):
  return X @ Theta

def computeCost(X, y, Theta):
  predicted = predict(X, Theta)# h0
  sqr_error = (predicted - y) ** 2# (h0 - y) ^ 2
  sum_error = np.sum(sqr_error)# sum all (xich ma)
  m = np.size(y)
  J = (1/(2*m)) * sum_error# 1/2m * sum
  return J

def computeCost_Vec(X, y, Theta):
  error = predict(X, Theta) - y
  m = np.size(y)
  J = (1/(2*m)) * np.transpose(error) @ error# Transpose matrix là ma trận đảo hàng và cột so với ma trận gốc.
  return J

def gradientDescent(X, y, alpha=0.02, iter=5000): #giá trị mặc định của alpha là 0.02, iter (số vòng lặp tối đa) là 5000
  #Giá trị ban đầu của theta = 0
  Theta = np.zeros(np.size(X,1)) #số lượng theta bằng số cột của X
  #array lưu lại các giá trị J trong quá trình lặp
  jHist = np.zeros((iter, 2)) # kích thước là iter*2, cột đầu chỉ là các số từ 1 đến iter để tiện cho việc plot. Kích thước được truyền vào qua một tupple
  #kích thước của training set
  m = np.size(y)
  #ma trận ngược (đảo hàng và cột) của X
  X_T = np.transpose(X)
  #biến tạm để kiểm tra tiến độ Gradient Descent
  preCost = computeCost(X, y, Theta)
  for i in range(0, iter):
    printProgressBar(i,iter)
    #Tính sai số (predict - y)
    error = predict(X, Theta) - y
    #Thực hiện gradient descent để thay đổi theta
    Theta = Theta - (alpha/m)*(X_T @ error)
    cost = computeCost(X, y, Theta)
    if np.round(cost, 15) == np.round(preCost, 15):
      #in ra vòng lặp hiện tại và J
      print('Reach optima at i = %d; J = %.6f' %(i, cost))
      #thêm tất cả các index còn lại sau khi break
      jHist[i:,0] = range(i,iter)
      #giá trị J sau khi break sẽ như cũ
      jHist[i:,1] = cost
      break
    preCost = cost
    jHist[i, 0] = i
    jHist[i, 1] = cost
  yield Theta
  yield jHist