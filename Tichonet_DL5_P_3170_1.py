import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from unit10 import c2w1_init_utils as u10
from DL1 import *
import os
import h5py

plt.rcParams['figure.figsize'] = (7.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = u10.load_dataset()
plt.show()

## Targil 3170 - 1.1
print ("------------------------------------------------------------------")
print ("Targil 3170 - 1.1")
print ("------------------------------------------------------------------")
np.random.seed(1)
hidden1 = DLLayer("Perseptrons 1", 30,(12288,),"relu",W_initialization = "Xaviar",learning_rate = 0.0075, optimization='adaptive')
hidden2 = DLLayer("Perseptrons 2", 15,(30,),"trim_sigmoid",W_initialization = "He",learning_rate = 0.1)
print(hidden1)
print(hidden2)

## Targil 3170 - 1.2
print ("------------------------------------------------------------------")
print ("Targil 3170 - 1.2")
print ("------------------------------------------------------------------")
hidden1 = DLLayer("Perseptrons 1", 10,(10,),"relu",W_initialization = "Xaviar",learning_rate = 0.0075)
hidden1.b = np.random.rand(hidden1.b.shape[0], hidden1.b.shape[1])
hidden1.save_weights("SaveDir","Hidden1")
hidden2 = DLLayer ("Perseptrons 2", 10,(10,),"trim_sigmoid",W_initialization = "SaveDir/Hidden1.h5",learning_rate = 0.1)
print(hidden1)
print(hidden2)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
dir = "model"
model.save_weights(dir)
print(os.listdir(dir))
print ('\n\n')

## Targil 3170 - 1.3
print ("------------------------------------------------------------------")
print ("Targil 3170 - 1.3")
print ("------------------------------------------------------------------")
init = "zeros"
hidden1 = DLLayer("Perseptrons 1", 10,(2,),"relu",W_initialization = init ,learning_rate = 0.01)
hidden2 = DLLayer("Perseptrons 2", 5,(10,),"relu",W_initialization = init ,learning_rate = 0.01)
hidden3 = DLLayer("Perseptrons 3", 1,(5,),"trim_sigmoid",W_initialization = init, learning_rate = 0.1)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
model.compile("cross_entropy", 0.5)

costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
axes = plt.gca()
axes.set_ylim([0.65,0.75])
plt.title("Model with " + init + " initialization")
plt.show()
predictions = model.predict(train_X)
print ('zeros- Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T))/float(train_Y.size)*100) + '%')


## Targil 3170 - 1.4
print ("------------------------------------------------------------------")
print ("Targil 3170 - 1.4")
print ("------------------------------------------------------------------")
np.random.seed(1)
init = "random"

hidden1 = DLLayer("Perseptrons 1", 10,(2,),"relu",W_initialization = init ,learning_rate = 0.01)
hidden2 = DLLayer("Perseptrons 2", 5,(10,),"relu",W_initialization = init ,learning_rate = 0.01)
hidden3 = DLLayer("Perseptrons 3", 1,(5,),"trim_sigmoid",W_initialization = init, learning_rate = 0.1)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
model.compile("cross_entropy", 0.5)

costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title(init + " initialization")
plt.show()

plt.title("Model with " + init + " initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model.predict(x.T), test_X, test_Y)

predictions = model.predict(train_X)
print ('Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T))/float(train_Y.size)*100) + '%')
predictions = model.predict(test_X)
print ('Test accuracy: %d' % float((np.dot(test_Y,predictions.T) + np.dot(1-test_Y,1-predictions.T))/float(test_Y.size)*100) + '%')

## Targil 3170 - 1.5
print ("------------------------------------------------------------------")
print ("Targil 3170 - 1.5")
print ("------------------------------------------------------------------")
np.random.seed(1)
init = "He"

hidden1 = DLLayer("Perseptrons 1", 10,(2,),"relu",W_initialization = init ,learning_rate = 0.01)
hidden2 = DLLayer("Perseptrons 2", 5,(10,),"relu",W_initialization = init ,learning_rate = 0.01)
hidden3 = DLLayer("Perseptrons 3", 1,(5,),"trim_sigmoid",W_initialization = init, learning_rate = 0.1)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
model.compile("cross_entropy", 0.5)

costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title(init + " initialization")
plt.show()

plt.title("Model with " + init + " initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model.predict(x.T), test_X, test_Y)

predictions = model.predict(train_X)
print ('Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T))/float(train_Y.size)*100) + '%')
predictions = model.predict(test_X)
print ('Test accuracy: %d' % float((np.dot(test_Y,predictions.T) + np.dot(1-test_Y,1-predictions.T))/float(test_Y.size)*100) + '%')

