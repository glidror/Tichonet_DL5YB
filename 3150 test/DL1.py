import numpy as np
import matplotlib.pyplot as plt
import random

class DLmodel(object):
    def __init__(self, name="Model"):
        self.name=name
        self.layers = [None]
        self._is_compiled= False

    def add(self, layer):
        self.layers.append(layer)

    def squared_means(self, AL, Y):
        return (AL-Y)**2

    def squared_means_backward(self, AL, Y):
         return 2*(AL-1)

    def cross_entropy(self, AL, Y):
        error = np.where(Y == 0, -np.log(1 - AL), -np.log(AL))
        return error

    def cross_entropy_backward(self, AL, Y):
        return 2*(AL-1)

    def compile(self, loss, threshold = 0.5):
        self.threshold = threshold
        self.loss = loss
        self._is_compiled = True
        if loss == "squared_means":
            self.loss_forward = self.squared_means
            self.loss_backward = self.squared_means_backward
        elif loss == "cross_entropy":
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.cross_entropy_backward
        else:
            raise NotImplementedError("Unimplemented loss function: " + loss)

    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        errors = self.loss_forward(AL, Y)
        J = (1/m)*np.sum(errors)
        return J

    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []
        for i in range(num_iterations):
            Al = X
            for l in range(1,L):
                Al = self.layers[l].forward_propagation(Al,False)            
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1,L)):
                dAl = self.layers[l].backward_propagation(dAl)
                self.layers[l].update_parameters()
            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
                print("cost after ", str(i+1), "updates ("+str(i//print_ind)+"%):",str(J))
        return costs

    def predict(self, X):
        Al = X
        L = len(self.layers)
        for i in range(1,L):
            Al = self.layers[i].forward_propagation(Al,True)
        return Al > self.threshold

    def save_weights(self,path):
        for i in range(1,len(self.layers)):
            self.layers[i].save_weights(path,"Layer"+str(i))


## Targil 3150 - 2.3
print ("------------------------------------------------------------------")
print ("Targil 3150 - 2.3")
print ("------------------------------------------------------------------")
np.random.seed(1)
m1 = DLmodel()
AL = np.random.rand(4,3)
Y = np.random.rand(4,3) > 0.7
m1.compile("cross_entropy")
errors = m1.loss_forward(AL,Y)
dAL = m1.loss_backward(AL,Y)
print("cross entropy error:\n",errors)
print("cross entropy dAL:\n",dAL)
m2 = DLmodel()
m2.compile("squared_means")
errors = m2.loss_forward(AL,Y)
dAL = m2.loss_backward(AL,Y)
print("squared means error:\n",errors)
print("squared means dAL:\n",dAL)


## Targil 3150 - 2.4
print ("------------------------------------------------------------------")
print ("Targil 3150 - 2.4")
print ("------------------------------------------------------------------")
print("cost m1:", m1.compute_cost(AL,Y))
print("cost m2:", m2.compute_cost(AL,Y))

## Targil 3150 - 2.5
print ("------------------------------------------------------------------")
print ("Targil 3150 - 2.5")
print ("------------------------------------------------------------------")
np.random.seed(1)
model = DLmodel();
model.add(DLR1("Perseptrons 1", 10,(12288,)))
model.add(DLR1("Perseptrons 2", 1,(10,),"trim_sigmoid"))
model.compile("cross_entropy", 0.7)
X = np.random.randn(12288,10) * 256
print("predict:",model.predict(X))


## Targil 3150 - 2.6
print ("------------------------------------------------------------------")
print ("Targil 3150 - 2.6")
print ("------------------------------------------------------------------")
print(model)