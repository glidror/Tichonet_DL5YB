#**************************************************
# Combine DLLayer and TF2
#**************************************************
import os
import h5py
from DL7 import *
import unit10.utils as u10
path = r'data' # change to your download directory!!
flatten = DLFlatten("flat",(1,28,28))
layer1  = DLLayer("layer1",256,(784,),activation = 'relu', learning_rate = 0.01)
layer2  = DLLayer("layer2",10,(256,),activation = 'softmax', learning_rate = 0.01)

model = DLModel("Restore_Parameters")

model.add(flatten)
model.add(layer1)
model.add(layer2)
model.restore_parameters(path)
print(flatten)
print(layer1)
print(layer2)

try:
    with h5py.File(path + "\\"+model.name + r'/check.h5', 'r') as hf:
        input = hf['input'][:]
        TF_flatten_output = hf['flatten_output'][:]
        TF_layer1_output = hf['layer1_output'][:]
        TF_layer2_output = hf['layer2_output'][:]
except (FileNotFoundError):
    raise NotImplementedError("Unrecognized initialization:", file)


def epsEqual(x1,x2,eps=1e-6):
    if x1 == 0 and x2 == 0:
       return True
    return (max(x1,x2)-min(x1,x2))/(abs(x1)+abs(x2))**0.5<eps

def compare_A(DL_A, TF_A):
    print(DL_A.shape)
    print(TF_A.shape)
    DL_A_flat = DL_A.reshape(-1)
    TF_A_flat = TF_A.reshape(-1)

    for i in range(len(DL_A_flat)):
        if not epsEqual(DL_A_flat[i],TF_A_flat[i],1e-4):
                print("oops",str(DL_A_flat[i]),"!=",str(TF_A_flat[i]))
                return False
    return True



DL_flatten_output = flatten.forward_propagation(flatten, input)
if u10.compare_A(DL_flatten_output, TF_flatten_output):
    print("output of flatten is the same as in TF")
DL_Layer1_output = layer1.forward_propagation(layer1, DL_flatten_output)
if u10.compare_A(DL_Layer1_output, TF_layer1_output):
    print("output of layer1 is the same as in TF")
DL_Layer2_output = layer2.forward_propagation(layer2, DL_Layer1_output)
if u10.compare_A(DL_Layer2_output, TF_layer2_output):
    print("output of layer2 is the same as in TF")
print(DL_Layer2_output)





