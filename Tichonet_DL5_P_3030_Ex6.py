import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10
from MyVector import *



def main():
    try:
        print(MyVector(3,is_col=False,fill=1).dot(MyVector(4,fill=2)))
    except ValueError as err1:
        print("Exception:",err1)
        try:
            print(MyVector(4,is_col=True,fill=1).dot(MyVector(4,fill=2)))
        except ValueError as err2:
            print("Exception:",err2)
            v1,v2 = MyVector(3,is_col=False,fill=4), MyVector(3,fill=2)
            print(v1.dot(v2),v1,v2)





### exeution
### ========
if __name__ == '__main__':
	main()

