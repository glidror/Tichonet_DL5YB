import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from unit10 import c1w4_utils as u10
from DL1 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W4()
# Example of a picture
index = 87
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[0,index]) + ". It's a " + classes[train_set_y[0,index]].decode("utf-8") +  " picture.")


## Targil 3170 - 2.1
print ("------------------------------------------------------------------")
print ("Targil 3170 - 2.1")
print ("------------------------------------------------------------------")

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_set_x_orig.shape))
print ("train_y shape: " + str(train_set_y.shape))
print ("test_x_orig shape: " + str(test_set_x_orig.shape))
print ("test_y shape: " + str(test_set_y.shape))

## Targil 3170 - 2.2
print ("------------------------------------------------------------------")
print ("Targil 3170 - 2.2")
print ("------------------------------------------------------------------")

train_set_y = train_set_y.reshape(-1)
test_set_y = test_set_y.reshape(-1)

# Reshape the training and test examples 
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten  = test_set_x_orig.reshape (test_set_x_orig.shape[0], -1).T
# Standardize data to have feature values between -0.5 and 0.5.
train_set_x = train_set_x_flatten/255.0 - 0.5
test_set_x = test_set_x_flatten/255.0 - 0.5

print ("train_x's shape: " + str(train_set_x.shape))
print ("test_x's shape: " + str(test_set_x.shape))
print ("normelized train color: ", str(train_set_x[10][10]))
print ("normelized test color: ", str(test_set_x[10][10]))

## Targil 3170 - 2.3
print ("------------------------------------------------------------------")
print ("Targil 3170 - 2.3")
print ("------------------------------------------------------------------")

hidden1 = DLLayer("Hidden 1", 7,(train_set_x.shape[0],),"relu",W_initialization = "Xaviar" ,learning_rate = 0.007)
hidden2 = DLLayer("Output", 1,(7,),"sigmoid",W_initialization = "Xaviar" ,learning_rate = 0.007)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.compile("cross_entropy", 0.5)

costs = model.train(train_set_x, train_set_y,2500)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 25s)')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_set_x) == train_set_y))
print("test accuracy:", np.mean(model.predict(test_set_x) == test_set_y))



## Targil 3170 - 2.4
print ("------------------------------------------------------------------")
print ("Targil 3170 - 2.4")
print ("------------------------------------------------------------------")

hidden1 = DLLayer("Hidden 1", 30,(train_set_x.shape[0],),"relu", W_initialization = "Xaviar" ,learning_rate = 0.007)
hidden2 = DLLayer("Hidden 2", 15,(30,),"relu", W_initialization = "Xaviar" ,learning_rate = 0.007)
hidden3 = DLLayer("Hidden 3", 10,(15,),"relu", W_initialization = "Xaviar" ,learning_rate = 0.007)
hidden4 = DLLayer("Hidden 4", 10,(10,),"relu", W_initialization = "Xaviar" ,learning_rate = 0.007)
hidden5 = DLLayer("Hidden 5", 5 ,(10,),"relu", W_initialization = "Xaviar" ,learning_rate = 0.007)
hidden6 = DLLayer("Output",   1 ,(5,) ,"sigmoid", W_initialization = "Xaviar" ,learning_rate = 0.007)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
model.add(hidden4)
model.add(hidden5)
model.add(hidden6)
model.compile("cross_entropy", 0.5)

costs = model.train(train_set_x, train_set_y,2500)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 25s)')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_set_x) == train_set_y))
print("test accuracy:", np.mean(model.predict(test_set_x) == test_set_y))


## Targil 3170 - 2.5
print ("------------------------------------------------------------------")
print ("Targil 3170 - 2.5")
print ("------------------------------------------------------------------")
img_path = r'data\images\cat.4009.jpg' # full path of the image
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
img = Image.open(img_path)
image64 = img.resize((num_px, num_px), Image.ANTIALIAS)
plt.imshow(img)
plt.show()
plt.imshow(image64)
plt.show();
my_image = np.reshape(image64,(num_px*num_px*3,1))
my_image = my_image/255. - 0.5
p = model.predict(my_image)
print ("L-layer model predicts a \"" + classes[int(p),].decode("utf-8") +  "\" picture.")
