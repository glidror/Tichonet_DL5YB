import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10

class MyClass:
	def __init__(self):
		self.x = 7		# We can use __ to indicated a private variable
		self.y = 8

	def sum(self, const):
		return self.x + self.y + const

	def __str__(self):
		return 'x=' + self.__x+' , y=' + self.y



def main():
	mish = MyClass()	# Activates the constructor  __init__
	s = mish.sum(3)	# Will return 15
	print (f'The result is - {s}')



### exeution
### ========
if __name__ == '__main__':
	main()

