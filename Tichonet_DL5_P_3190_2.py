from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from unit10 import c1w5_utils as u10
from DL3 import *


## Targil 3190 - 2.1
print ("------------------------------------------------------------------")
print ("Targil 3190 - 2.1")
print ("------------------------------------------------------------------")
mnist = fetch_openml('mnist_784')
X, Y = mnist["data"], mnist["target"]
X = X / 255.0 -0.5

i = 12

img = X[i:i+1].to_numpy().reshape(28,28)

plt.imshow(img, cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print("Label is: '"+Y[i]+"'")

## Targil 3190 - 2.2
print ("------------------------------------------------------------------")
print ("Targil 3190 - 2.2")
print ("------------------------------------------------------------------")
digits = 10
examples = Y.shape[0]

Y = Y.to_numpy().reshape(1, examples)

Y_new = np.eye(digits)[Y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

print(Y_new[:,12])

## Targil 3190 - 2.3
print ("------------------------------------------------------------------")
print ("Targil 3190 - 2.3")
print ("------------------------------------------------------------------")
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()


np.random.seed(111)
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
i = 12
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print(Y_train[:,i])

## Targil 3190 - 2.4
print ("------------------------------------------------------------------")
print ("Targil 3190 - 2.4")
print ("------------------------------------------------------------------")
np.random.seed(1)

hidden_layer = DLLayer ("Softmax 1", 64,(784,),"sigmoid","He", 1)
softmax_layer = DLLayer ("Softmax 1", 10,(64,),"softmax","He", 1)
model = DLModel()
model.add(hidden_layer )
model.add(softmax_layer)
model.compile("categorical_cross_entropy")

costs = model.train(X_train,Y_train,2000)

#parameters, costs = L_layer_model_softmax(X_train, Y_train, layers_dims, 1, 2000, u10.sigmoid, u10.sigmoid_backward)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(1))
plt.show()


## Targil 3190 - 2.5
print ("------------------------------------------------------------------")
print ("Targil 3190 - 2.5")
print ("------------------------------------------------------------------")
print('Deep train accuracy')
model.confusion_matrix(X_train, Y_train)
print('Deep test accuracy')
model.confusion_matrix(X_test, Y_test)

i=4
#print('train',str(i),str(pred_train[i][i]/np.sum(pred_train[:,i])))
#print('test',str(i),str(pred_test[i][i]/np.sum(pred_test[:,i])))

## Targil 3190 - 2.6
print ("------------------------------------------------------------------")
print ("Targil 3190 - 2.6")
print ("------------------------------------------------------------------")
from PIL import Image, ImageOps
#Test your image
num_px = 28
img_path = r'data\images\three.jpg'
my_label_y = [0,0,0,1,0,0,0,0,0,0] # change the 1’s position to fit image
image = Image.open(img_path)
image28 = image.resize((num_px, num_px), Image.ANTIALIAS) # resize to 28X28 
plt.imshow(image)	   # Before scale 
plt.show();
plt.imshow(image28)  # After scale
plt.show();
gray_image = ImageOps.grayscale(image28)	# grayscale – to fit to training data   
my_image = np.reshape(gray_image,(num_px*num_px,1))
my_label_y = np.reshape(my_label_y,(10,1))	
my_image = my_image / 255.0 -0.5  # normelize
p = predict(my_image)
print (p)


