import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10
from MyVector import *
import time

def calc_J(X, Y, W, b):
  m,n = len(Y), len(W)
  dW = np.zeros((n,1))
  J, db = 0, 0
  Y_hat = W.T@X +b
  Diff = Y_hat-Y
  J = np.sum(Diff**2)/m
  dW = (2/m)*np.sum(Diff*X, axis=1, keepdims=True)
  db = (2/m)*np.sum(Diff)
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

