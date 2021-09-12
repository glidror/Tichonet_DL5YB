import numpy as np
import matplotlib.pyplot as plt
import h5py
#import scipy
#from scipy import ndimage
from unit10 import c1w2_utils as u10

import os
from PIL import Image
import time



class MyPerceptron(object):
    # ---------------------------------------------------------------
    # initialize a perceptron
    # ---------------------------------------------------------------
    def __init__ (self, X , Y):
        self.X = X
        self.Y = Y
        self.dim = X.shape[0]
        self.m = X.shape[1]
        self.W, self.b = self.initialize_with_zeros(self.dim)
        self.dW, self.db = self.initialize_with_zeros(self.dim)
    
    # --------------------------------------------------
    # Service routines
    # --------------------------------------------------

    # Sigmoid function that will work on a list of values
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))

    # initialize a list with zeros (w[]) and return the list and an additional integer (the b)
    def initialize_with_zeros(self, dim):
        w = np.zeros((dim,1), dtype = float)
        b = 0.0
        return w, b

    # --------------------------------------------------
    # linear analyzis routines
    # --------------------------------------------------

    # forward propagation - calculate the value of the avaerage cost function for the whole set of samples X
    # compared to the expected result Y, using the set of parameters W and b
    def forward_propagation(self):
        #m = self.X.shape[1]
        Z = np.dot(self.W.T, self.X)+self.b
        A = self.sigmoid(Z) 
        J= (-1/self.m)*np.sum(self.Y * np.log(A) + (1-self.Y) * np.log(1-A))
        J = np.squeeze(J)
        return A, J

    # Backword propagation - calculate the values of the difference dW and db
    # for a samples (X), with expected results Y and calculated results A
    def backward_propagation(self, A):
        #m = self.X.shape[1]
        dZ = (1/self.m)*(A-self.Y)
        dW = np.dot(self.X, dZ.T)
        db = np.sum(dZ)
        return dW, db

    # train the perceptron using a sample db X, expected results Y
    # with number of iteration and aparemeters to indicate the learning rate
    def train(self, num_iterations, learning_rate):
        ### AlphaW, self.Alphab
        for i in range(num_iterations):
            A, cost = self.forward_propagation()
            self.dW, self.db = self.backward_propagation(A)
            self.W -= learning_rate*self.dW
            self.b -= learning_rate*self.db
            if (i % 100 == 0):
              print ("Cost after iteration {} is {}".format( i, cost))
        return self.W , self.b

    # predict - get a set of pictures and predict if they are true (1) or false (0)
    # for the specific criteris (e.g. - "is it a cat")
    def predict(self, X, w, b):
        Z = np.dot(w.T,X)+b
        return (np.where(self.sigmoid(Z)>0.5, 1., 0.))        


