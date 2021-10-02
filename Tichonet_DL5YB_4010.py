import numpy as np
import matplotlib.pyplot as plt
from unit10 import utils as u10
import sklearn
import sklearn.datasets
import scipy.io
from DL4 import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


l1 = DLLayer("Hidden",6 , (5,) ,"relu", learning_rate = 0.1)
l2 = DLLayer("Output",1 , (6,) ,"sigmoid",learning_rate = 0.1)
print("Default:")
print(l1.is_train)  #_is_compiled
print(l2.is_train)
n = DLModel("Example")
n.add(l1)
n.add(l2)
n.set_train(True)
print("After set to True:")
print(l1.is_train)
print(l2.is_train)
