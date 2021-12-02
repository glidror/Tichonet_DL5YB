#**************************************************
# Check implementation of YOLO - Part 1 - Incomplete !
#**************************************************
import numpy as np
import h5py
import matplotlib.pyplot as plt
from DL7 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


print ("------------------------------------------------------------------")
print ("Targil 6010 Ex.1 - filter boxs")
print ("------------------------------------------------------------------")
# An - number of Anchorsm, (Gx, Gy) - dim of grid, C - num of categories
# box_confidence - (Gx, Gy, An, 1) of all Pc's The probability of an object in each Anchor
# boxes - (Gx, Gy, An, 4) - Size of each box surrounding an object, for each Anchore
# box_class_probs - (Gx, Gy, An, C) Probability of each object in all Anchores
# threshold - for accepting an object
# return:
#    scores - (NumOfBoxes,) The prob of each box (that passed the threshold)
#    boxes - (NumOfBoxes,4) The dimentions of each box
#    box_class_probs - (Gx, Gy, An, C) The probability of each category in each box
#    classes - (NumOfBoxes,) The serial num of the category of each box
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_scores = box_confidence * box_class_probs
    class_winner_in_box = np.argmax(box_scores, axis=-1)
