import numpy as np
import h5py
import matplotlib.pyplot as plt
from DL6 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# ---------------------------
# Targil - 1
# ---------------------------
np.random.seed(1)
linear = DLLayer("line",2,(3,),activation = 'NoActivation',W_initialization="Xaviar", learning_rate = 0.001)
print(linear)
print(linear.W)
convValid = DLConv("Valid", 3, (3,15,20), filter_size=(3,3), strides=(1,2), 
                                  padding="Valid",learning_rate = 0.01)
print(convValid)
convSame = DLConv("Same", 2,(3,30,64), filter_size=(5,5), strides=(1,1), 
                                  padding="Same",learning_rate = 0.1,  optimization='adaptive', regularization="L2")
##DLConv("Same", 2,(3,30,64),0.1,(5,5),(1,1),"same", optimization='adaptive', regularization="L2")
print(convSame)
conv34 = DLConv("34", 2,(3,28,28), filter_size=(2,2), strides=(1,1), 
                                  padding=(3,4),learning_rate = 0.07,  optimization='adaptive', regularization="L2")
#DLConv("34", 2,(3,28,28),0.07,(2,2),(1,1),padding=(3,4))
print(conv34)
print(conv34.W)


# ---------------------------
# Targil - 2
# ---------------------------
np.random.seed(1)
prev_A = np.random.randn(3,4,4,10)
test = DLConv("test forward", 8 ,(3,4,4), filter_size=(2,2), strides=(2,2), padding=(2,2),learning_rate = 0.1,  optimization='adaptive', regularization="L2")
##test = DLConv("test forward", 8 ,(3,4,4),0.1,filter_size=(2,2),padding=(2,2),strides=(2,2))
Z = test.forward_propagation(test, prev_A)
print("Z's mean =", np.mean(Z))
print("Z.shape =", str(Z.shape))
print("Z[3,2,1] =", Z[3,2,1])
print("W.shape =", str(test.W.shape))


# ---------------------------
# Targil - 3
# ---------------------------
np.random.seed(1)
A_prev = np.random.randn(3,4,4,10)
test = DLConv("test backword", 8 ,(3,4,4), filter_size=(2,2), strides=(2,2), padding=(2,2),learning_rate = 0.1,  optimization='adaptive', regularization="L2")
Z = test.forward_propagation(test, A_prev)
dZ = Z * np.random.randn(8,4,4,10)
dA_prev = test.backward_propagation(test, dZ)
print("dA_prev's mean =", np.mean(dA_prev))
print("dA_prev.shape =", str(dA_prev.shape))
print("dA_prev[1,2,3] =", dA_prev[1,2,3])
print("dW shape =", test.dW.shape)
print("dW[3,2,1] =", test.dW[3,2,1])
print("db = ", test.db)


# ---------------------------
# Targil - 4
# ---------------------------
#                  DLConv("test backword", 8 ,(3,4,4), filter_size=(2,2), strides=(2,2), padding=(2,2),learning_rate = 0.1,  optimization='adaptive', regularization="L2")
conv_layer = DLConv("test conv layer", 7, (3,100,100), learning_rate =0.1, activation = "relu",filter_size=(3,3),strides=(1,1),padding = 'Same',  optimization='adaptive',regularization="L2")
print(conv_layer)


# ---------------------------
# Targil - 5
# ---------------------------
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


# ---------------------------
# Targil - 6
# ---------------------------
np.random.seed(1)
check_X = np.random.randn(3,28,28,3)
check_Y = np.random.rand(1,3) > 0.5
test_conv = DLConv("test conv", 12, (3,28,28), learning_rate = 0.1, filter_size=(3,3), padding='Same', strides=(1,1), activation="sigmoid")
test_maxpooling = DLMaxpooling("test maxpool", (12,28,28) ,filter_size=(2,2), strides=(2,2))
test_flatten = DLFlatten("test flatten", (12,14,14))

test_layer1 = DLLayer("test layer1", 17, (12*14*14,) ,activation = "tanh",learning_rate = 0.1, regularization='L2')
test_layer2 = DLLayer("test layer2", 1, (17,) ,activation = "sigmoid",learning_rate = 0.1)

DNN = DLModel("Test div model")
DNN.add(test_conv)
DNN.add(test_maxpooling)
DNN.add(test_flatten)
DNN.add(test_layer1)
DNN.add(test_layer2)
DNN.compile("squared_means")

# ---------------------------
# Targil - 7
# ---------------------------
check, max_diff, max_layer = DNN.check_backward_propagation(check_X, check_Y)
print("check:",str(check), ", diff:", str(max_diff), ", layer:", str(max_layer))
