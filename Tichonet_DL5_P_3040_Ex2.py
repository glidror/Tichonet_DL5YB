import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10
from MyVector import *
import time


def calc_J(X, Y, W, b):
  m,n = len(Y), len(W)
  #dW = []
  #for j in range(n):
  #  dW.append(0)
  dW = np.zeros((n,1))
  J, db = 0, 0
  for i in range(m):
    Xi = X[:,i].reshape(n,1)
    #for j in range(n):
    #  y_hat_i += W[j]*X[j][i]
    y_hat_i = np.dot(W.T,Xi) + b
    diff = (float)(y_hat_i - Y[i])
    J += (diff**2)/m
    #for j in range(n):
    #  dW[j] += (2*diff/m)*X[j][i]
    dW += (2*diff/m)*Xi
    db += 2*diff/m;
  return J, dW, db

def main():
    X, Y = u10.load_dataB1W4_trainN()
    np.random.seed(1)
    J, dW, db = calc_J(X,Y,np.random.randn(len(X),1),3)
    print(J)
    print(dW.shape)
    print(db)


### exeution
### ========
if __name__ == '__main__':
	main()

