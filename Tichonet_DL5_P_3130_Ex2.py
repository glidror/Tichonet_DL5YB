import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import random
import unit10.c1w2_utils as u10
from MyVector import *

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()

train_set_y = train_set_y.reshape(-1)
test_set_y = test_set_y.reshape(-1)

m_train = train_set_y.shape[0]
m_test = test_set_y.shape[0]
num_px = train_set_x_orig.shape[1]


train_set_x_orig = reshape(train_set_x_orig.shape[0], -1)

train_set_x_flatten =   train_set_x_orig.reshape (train_set_x_orig[0],m_train)
test_set_x_flatten = train_set_x_orig.reshape (train_set_x_orig[0],m_train)

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0


