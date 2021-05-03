import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10
from Tichonet_DL5_P_3030_MyCector import *



def main():
    try:
        print(MyVector(3,fill=1) + MyVector(4,fill=2))
    except Exception as err1:
        print("Exception:",err1)
        try:
            print(MyVector(3,is_col=False, fill=1) + MyVector(3,fill=2))
        except Exception as err2:
            print("Exception:",err2)
            print(MyVector(3, fill=15) + MyVector(3,fill=21))




### exeution
### ========
if __name__ == '__main__':
	main()

