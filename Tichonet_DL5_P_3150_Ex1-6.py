import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import random
import unit10.c1w2_utils as u10
from MyPerceptron import *



## Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()

# setting parameters for the size of the sampled data
m_train, num_px, num_px, num_pxColor  =  train_set_x_orig.shape
m_test, num_pxX, num_pxY, num_pxColor =  test_set_x_orig.shape

# flatten the pictures to one dimentionsl array of values, keeping a seperated array for each picture
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten  = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# normalize the values to be between 0 and 1
train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0


# create my perceptron !!
Myperceptron = MyPerceptron(train_set_x, train_set_y)

# train my perceptron
W, b = Myperceptron.train(num_iterations=4000, learning_rate=0.005)

# Predict according to the trained perceptron
Y_prediction_test = Myperceptron.predict(test_set_x, W, b )
Y_prediction_train = Myperceptron.predict(train_set_x, W, b )


# Print the accuracy of identifying in the train and in the test
print ("Percent train = ", np.sum(train_set_y==Y_prediction_train)/train_set_y.shape[1])
print ("Percent test  = ", np.sum(test_set_y ==Y_prediction_test) /test_set_y.shape[1])

# Try to identify an image of a cat / no-cat
fname = 'data\\images\\cat.4009.jpg'  
im = Image.open(fname)
im  = im.resize((num_px, num_px), Image.ANTIALIAS)
load_image = np.array(im)

plt.imshow(im)
plt.show()

my_image = np.array(im).reshape(1, -1).T
MyNorm_pic = my_image /255.0

my_predicted_image = Myperceptron.predict(MyNorm_pic, W, b)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


