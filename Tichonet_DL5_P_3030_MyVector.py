import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10

class MyVector(object):
    # ---------------------------------------------------------------
    # initialize a new vector object
    # ---------------------------------------------------------------
    def __init__ (self, size, is_col=True, fill=0, init_values=None):
        self.vector = []
        self.size = size
        self.is_col = is_col
        if (init_values != None):
            l = len(init_values)
            for i in range(size):
                self.vector.append(init_values[i % l])
        else:
            for i in range (size):
                self.vector.append(fill)
                
    # ---------------------------------------------------------------
    # printing a vectore. Distinguish between line and column vector
    # ---------------------------------------------------------------
    def __str__(self):
        s = '['
        lf = "\n" if self.is_col else ""
        for item in self.vector:
            s = s + str(item) + ', ' +lf
        s+= "]"
        return  (s)        
    # ---------------------------------------------------------------
    # Get an item from a vactor, by its key (position)
    # ---------------------------------------------------------------
    def __getitem__(self, key):
        return (self.vector[key])

    # ---------------------------------------------------------------
    # Set an item of a vactor, by its key (position) using a given value
    # ---------------------------------------------------------------
    def __setitem__(self, key, value):
        self.vector[key] = value


    
    def __len__(self):
        return self.size

    # ---------------------------------------------------------------
    # Check for validity of self and other. If scalar - will broadcast to a vector
    # ---------------------------------------------------------------
    def __check_other(self, other):
        if not isinstance(other,MyVector):
            if (type(other) in [int, float]):
                other = MyVector(self.size, True, fill = other)
            else:
                raise ValueError("*** Wrong type of parameter")
        if (self.is_col == False or other.is_col == False):
            raise ValueError("*** both vectors must be column vectors")
        if (self.size != other.size):
            raise ValueError("*** vectors must be of same size")
        return other    

    # ---------------------------------------------------------------
    # ADD vectors
    # ---------------------------------------------------------------
    def __add__(self,w):
        w = self.__check_other(w)        
        res = []
        for i in range (self.size):
            res.append (self.vector[i] + w.vector[i])
        return (MyVector(self.size, True, fill = 0, init_values=res))

    # ---------------------------------------------------------------
    # Transpose a vector. flip flop between a line and a column
    # ---------------------------------------------------------------
    def transpose(self):
        NewVec = MyVector(self.size, not self.is_col, fill = 0, init_values = self.vector)
        return (NewVec)
