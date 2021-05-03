import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10
from MyVector import *



def main():
	alpha_W = MyVector(3,init_values=[-0.7,-0.03,1.44])
	dW = MyVector(3,init_values=[100,-200,5])
	same_direction = dW * alpha_W > 0
	alpha_W = alpha_W*same_direction*1.1 + alpha_W*(1-same_direction)*-0.5
	print(alpha_W)




### exeution
### ========
if __name__ == '__main__':
	main()

