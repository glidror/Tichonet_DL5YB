import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10
from MyVector import *
import time

def main():
    a = np.random.random(1000000)
    b = np.random.random(1000000)
    ####  Non Vectorized version ####
    tic = time.time()
    c = 0
    for i in range(1000000):
        c += a[i]*b[i]
    toc = time.time();
    print("c="+str(c))
    print ("Non Vectorized version: " + 
	str(1000*(toc-tic)) + "ms")
    ####  Vectorized version #######
    tic = time.time()
    c = np.dot(a,b)
    toc = time.time(); 
    print("c="+str(c))
    print ("Vectorized version: " + 
     	str(1000*(toc-tic)) + "ms")

### exeution
### ========
if __name__ == '__main__':
	main()

