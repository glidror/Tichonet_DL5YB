import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10
from MyVector import *
import time


def calc_J_NonVector(X, Y, W, b):
  m = len(Y)
  n = len(W)
  J = 0
  dW = []
  for j in range(n):
    dW.append(0)
  db = 0
  for i in range(m):
    y_hat_i = b
    for j in range(n):
      y_hat_i += W[j]*X[j][i]
    diff = (float)(y_hat_i - Y[i])
    J += (diff**2)/m
    for j in range(n):
      dW[j] += (2*diff/m)*X[j][i]
    db += 2*diff/m;
  return J, dW, db

def train_NonVector(X, Y, alpha, num_iterations, calc_J):
    m,n = len(Y), len(X)
    costs, W, alpha_W,b = [],[],[],0
    for j in range(n):
        W.append(0)
        alpha_W.append(alpha)
    alpha_b = alpha
    
    for i in range(1,num_iterations+1):
        cost,dW,db = calc_J(X,Y,W,b)
        for j in range(n):
            alpha_W[j] *= 1.1 if dW[j]*alpha_W[j] > 0 else -0.5
        alpha_b *= 1.1 if db*alpha_b > 0 else -0.5
        for j in range(n):
            W[j] -= alpha_W[j]
        b -= alpha_b
        if i%(num_iterations//50)==0:
            print (f'Iteration {i}  cost {cost}')
            costs.append(cost)
    return costs, W, b



def calc_J(X, Y, W, b):
  m,n = len(Y), len(W)
  Y_hat = W.T@X +b
  Diff = Y_hat-Y
  J = np.sum(Diff**2)/m
  dW = (2/m)*np.sum(Diff*X, axis=1, keepdims=True)
  db = (2/m)*np.sum(Diff)
  return J, dW, db

def train(X, Y, alpha, num_iterations, calc_J):
    m,n,J,costs = len(Y),len(X),0,[]
    W,b,alpha_W,alpha_b = np.zeros((n,1)),0,np.full((n,1),alpha),alpha

    for i in range(1,num_iterations+1):
        cost, dW, db = calc_J(X, Y, W, b)
        alpha_W = np.where(dW * alpha_W > 0, alpha_W * 1.1, alpha_W * -0.5)
        alpha_b *= 1.1 if (db * alpha_b > 0) else -0.5
        W -= alpha_W
        b -= alpha_b
        if i%(num_iterations//50)==0:
            print (f'Iteration {i}  cost {cost}')
            costs.append(cost)
    return costs, W, b



def main():
    X, Y = u10.load_dataB1W4_trainN()
    np.random.seed(1)
    tic = time.time()
    costs, W, b = train_NonVector(X, Y, 0.001, 100000, calc_J_NonVector)
    print("J="+str(costs[-1])+', w1='+str(W[0])+', w2='+str(W[1])+', w3='+str(W[2])+', w4='+str(W[3])+", b="+str(b))
    toc = time.time();
    print ("Non Vectorized version: " + str(1000*(toc-tic)) + "ms")

    tic = time.time()
    costs, W, b = train(X, Y, 0.001, 100000, calc_J)
    print("J="+str(costs[-1])+', w1='+str(W[0])+', w2='+str(W[1])+', w3='+str(W[2])+', w4='+str(W[3])+", b="+str(b))
    toc = time.time();
    print ("Vectorized version: " + str(1000*(toc-tic)) + "ms")

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()


### exeution
### ========
if __name__ == '__main__':
	main()

