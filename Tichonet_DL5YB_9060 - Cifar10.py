#**************************************************
# Combine DLLayer and TF2  - Cifar implementation
#**************************************************

import os
import h5py
from DL7 import *
import unit10.utils as u10

dir = r'data' # change to your download directory!!
conv = DLModel("convolution")
conv.compile("categorical_cross_entropy")

conv.add(DLConv("conv1", 32, (3,32,32), learning_rate = 0.01, activation="relu", 
                filter_size=(3,3), strides=(1,1), padding='Valid'))
conv.add(DLMaxpooling("maxpool1", (32,30,30), filter_size=(2,2), strides=(2,2)))
conv.add(DLConv("conv2", 64, (32,15,15), learning_rate = 0.01, activation="relu", filter_size=(3,3), strides=(1,1), padding='Valid'))
conv.add(DLMaxpooling("maxpool2", (64,13,13), filter_size=(2,2), strides=(2,2)))
conv.add(DLConv("conv3", 64, (64,6,6), learning_rate = 0.01, activation="relu", filter_size=(3,3), strides=(1,1), padding='Valid'))
conv.add(DLFlatten("flatten", (64,4,4)))
conv.add(DLLayer("dense", 64, (1024,) , "relu", learning_rate = 0.01))
conv.add(DLLayer("softmax", 10, (64,) , "softmax", learning_rate = 0.01))

conv.restore_parameters(dir)

TF_layer_outputs = []
with h5py.File(dir + "\\"+conv.name + r'/check.h5', 'r') as hf:
    input = hf['input'][:]
    TF_layer_outputs.append(hf['conv1'][:])
    TF_layer_outputs.append(hf['poolmax1'][:])
    TF_layer_outputs.append(hf['conv2'][:])
    TF_layer_outputs.append(hf['poolmax2'][:])
    TF_layer_outputs.append(hf['conv3'][:])
    TF_layer_outputs.append(hf['flatten'][:])
    TF_layer_outputs.append(hf['dense'][:])
    TF_layer_outputs.append(hf['softmax'][:])


### Problem in layer Flatten !!!
### XXXXXXXXXXXXXXXXXXXXXXXXXXXX

A = input
for i in range(len(conv.layers)):
    A = conv.layers[i].forward_propagation(conv.layers[i], A)
    if u10.compare_A(A, TF_layer_outputs[i]):
        print("output of", conv.layers[i].name, "is the same as in TF")
    else:
        break
print(A)


