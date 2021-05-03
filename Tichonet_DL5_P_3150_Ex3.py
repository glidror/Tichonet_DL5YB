import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from unit10 import c1w3_utils as u10
from DL1 import *

np.random.seed(1)

"""
## Targil 3150 - 3.0
print ("------------------------------------------------------------------")
print ("Targil 3150 - 3 - HIMUM")
print ("------------------------------------------------------------------")

X, Y = u10.load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y[0,:])
# Plot the decision boundary for logistic regression
u10.plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")



## Targil 3150 - 3.1
print ("------------------------------------------------------------------")
print ("Targil 3150 - 3.1")
print ("------------------------------------------------------------------")

model = DLModel();
model.add(DLLayer("Perseptrons 1", 4,(2,),"tanh","random", 0.1))
model.add(DLLayer("Perseptrons 2", 1,(4,),"sigmoid", "random", 0.1))
model.compile("cross_entropy", 0.5)

print(model)


## Targil 3150 - 3.2 (1)
print ("------------------------------------------------------------------")
print ("Targil 3150 - 3.2 (1)")
print ("------------------------------------------------------------------")
costs = model.train(X,Y,10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

## Targil 3150 - 3.2 (2)
print ("------------------------------------------------------------------")
print ("Targil 3150 - 3.2 (2)")
print ("------------------------------------------------------------------")

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = model.predict(X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')



## Targil 3150 - 3.3
print ("------------------------------------------------------------------")
print ("Targil 3150 - 3.3")
print ("------------------------------------------------------------------")
X, Y = u10.load_planar_dataset()

model = DLModel();
model.add(DLLayer("Perseptrons 1", 4,(2,),"tanh","random", 0.1))
model.add(DLLayer("Perseptrons 1", 3,(4,),"tanh","random", 0.1))
model.add(DLLayer("Perseptrons 2", 1,(3,),"sigmoid", "random", 0.1))
model.compile("cross_entropy", 0.5)

costs = model.train(X,Y,10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = model.predict(X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


"""

## Targil 3150 - 3.4 
print ("------------------------------------------------------------------")
print ("Targil 3150 - 3.4")
print ("------------------------------------------------------------------")

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = u10.load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
dataset = "noisy_moons"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
if dataset == "blobs":
    Y = Y%2

model = DLModel();
model.add(DLLayer("Perseptrons 1", 6,(2,),"tanh","random", 0.01, 'adaptive'))
model.add(DLLayer("Perseptrons 2", 1,(6,),"sigmoid", "random", 0.01, 'adaptive'))
model.compile("cross_entropy", 0.5)

costs = model.train(X,Y,25000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = model.predict(X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
