#**************************************************
# Check implementation of convolutional networks
#**************************************************
import numpy as np
import h5py
import matplotlib.pyplot as plt
from DL7 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


## Targil 4310 - 1
print ("------------------------------------------------------------------")
print ("Targil 4310 Ex. 1 - constructor")
print ("------------------------------------------------------------------")
np.random.seed(1)
linear = DLLayer("line",2,(3,),activation = 'NoActivation', W_initialization="Xaviar", learning_rate = 0.001)
linear.init_weights("Xaviar")  # Do the Xaviar twice...
print(linear)
print(linear.W)
convValid = DLConv("Valid", 3, (3,15,20), filter_size=(3,3), strides=(1,2), W_initialization = "He",
                   padding="Valid", learning_rate = 0.01)
print(convValid)
convSame = DLConv("Same", 2,(3,30,64), filter_size=(5,5), strides=(1,1),  W_initialization = "He",
                   padding="Same", learning_rate = 0.1,  optimization='adaptive', regularization="L2")
print(convSame)
conv34 = DLConv("34", 2,(3,28,28), filter_size=(2,2), strides=(1,1), W_initialization = "He", 
                   padding=(3,4), learning_rate = 0.07,  optimization='adaptive', regularization="L2")
print(conv34)
print(conv34.W)


## Targil 4310 - 2
print ("------------------------------------------------------------------")
print ("Targil 4310 Ex. 2 - forward propegation")
print ("------------------------------------------------------------------")
np.random.seed(1)
prev_A = np.random.randn(3,4,4,10)
test = DLConv("test forward", 8 ,(3,4,4), filter_size=(2,2), strides=(2,2), padding=(2,2), W_initialization = "He", activation="NoActivation")
A = test.forward_propagation(test, prev_A)
print("A's mean =", np.mean(A))
print("A.shape =", str(A.shape))
print("A[3,2,1] =", A[3,2,1])
print("W.shape =", str(test.W.shape))


## Targil 4310 - 3
print ("------------------------------------------------------------------")
print ("Targil 4310 Ex. 3 - backword propegation")
print ("------------------------------------------------------------------")#**************************************************

np.random.seed(1)
prev_A = np.random.randn(3,4,4,10)
test = DLConv("test backword", 8 ,(3,4,4), filter_size=(2,2), strides=(2,2), padding=(2,2), W_initialization = "He")
A = test.forward_propagation(test, prev_A)
dA = A * np.random.randn(8,4,4,10)
dA_prev = test.backward_propagation(test, dA)
print("dA_prev's mean =", np.mean(dA_prev))
print("dA_prev.shape =", str(dA_prev.shape))
print("dA_prev[1,2,3] =", dA_prev[1,2,3])
print("dW shape =", test.dW.shape)
print("dW[3,2,1] =", test.dW[3,2,1])
print("db = ", test.db)


## Targil 4310 - 4
print ("------------------------------------------------------------------")
print ("Targil 4310 Ex. 4 - Conv. layer buildup - summary")
print ("------------------------------------------------------------------")
conv_layer = DLConv("test conv layer", 7, (3,100,100), learning_rate =0.1, activation = "relu",filter_size=(3,3),strides=(1,1), W_initialization = "He",
                        padding = 'Same',  optimization='adaptive',regularization="L2")
print(conv_layer)

## Targil 4310 - 5
print ("------------------------------------------------------------------")
print ("Targil 4310 Ex. 5 - pooling")
print ("------------------------------------------------------------------")
np.random.seed(1)
A_prev = np.random.randn(3,100,100,10)
test = DLMaxpooling("test maxpooling",(3,100,100),filter_size=(3,3),strides=(2,2))
print(test)
Z = test.forward_propagation(test, A_prev)
print("Z.shape =", str(Z.shape))
print("Z[1,2,3] =", Z[1,2,3])
dZ = Z * np.random.randn(3,49,49,10)
dA_prev = test.backward_propagation(test, dZ)
print("dA_prev's mean =", np.mean(dA_prev))
print("dA_prev.shape =", str(dA_prev.shape))
print("dA_prev[1,2,3] =", dA_prev[1,2,3])


## Targil 4310 - 6
print ("------------------------------------------------------------------")
print ("Targil 4310 Ex. 6 - flatten and full test")
print ("------------------------------------------------------------------")
np.random.seed(1)
check_X = np.random.randn(3,28,28,3)
check_Y = np.random.rand(1,3) > 0.5
test_conv = DLConv("test conv", 12, (3,28,28), learning_rate = 0.1, filter_size=(3,3), padding='Same', strides=(1,1), activation="sigmoid", W_initialization = "He")
test_maxpooling = DLMaxpooling("test maxpool", (12,28,28) ,filter_size=(2,2), strides=(2,2))
test_flatten = DLFlatten("test flatten", (12,14,14))
test_layer1 = DLLayer("test layer1", 17, (12*14*14,) ,activation = "tanh",learning_rate = 0.1, regularization='L2', W_initialization = "He")
test_layer2 = DLLayer("test layer2", 1, (17,) ,activation = "sigmoid",learning_rate = 0.1, W_initialization = "He")
DNN = DLModel("Test div model")
DNN.add(test_conv)
DNN.add(test_maxpooling)
DNN.add(test_flatten)
DNN.add(test_layer1)
DNN.add(test_layer2)
DNN.compile("squared_means")
print (DNN)



