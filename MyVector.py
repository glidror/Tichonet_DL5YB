import math


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
    # Get an item from a vactor, by its key (position)
    # ---------------------------------------------------------------
    def __getitem__(self, key):
        return (self.vector[key])

    # ---------------------------------------------------------------
    # Set an item of a vactor, by its key (position) using a given value
    # ---------------------------------------------------------------
    def __setitem__(self, key, value):
        self.vector[key] = value


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

    def __len__(self):
        return self.size

    # ---------------------------------------------------------------
    # Check for validity of self and other. If scalar - will broadcast to a vector
    # ---------------------------------------------------------------
    def __check_other(self, other):
        if not isinstance(other,MyVector):
            if (type(other) in [int, float]):
                other = MyVector(self.size, True, fill = other)
        if (not isinstance(other,MyVector)):
            raise ValueError("*** Wrong type of parameter")
        if (self.is_col == False or other.is_col == False):
            raise ValueError("*** both vectors must be column vectors")
        if (self.size != other.size):
            raise ValueError("*** vectors must be of same size")
        return other
        
# ================================================================================================
# ARITHMETIC operations. Element wise.
# ================================================================================================
    
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
    # MULL vectors
    # ---------------------------------------------------------------
    def __mul__(self,w):        #   v1 * v2
        w = self.__check_other(w)        
        res = []
        for i in range (self.size):
            res.append (self.vector[i] * w.vector[i])
        return (MyVector(self.size, True, fill = 0, init_values=res))

    # ---------------------------------------------------------------
    # TRUE DIV vectors
    # ---------------------------------------------------------------
    def __truediv__(self,w):
        w = self.__check_other(w)        
        res = []
        try:
            for i in range (self.size):
                res.append (float(self.vector[i]) / w.vector[i])
        except ValueError as err1:
            raise ValueError ("Divide by 0 : ",err1)
            return None
        return (MyVector(self.size, True, fill = 0, init_values=res))
        

    # ---------------------------------------------------------------
    # SUB vectors
    # ---------------------------------------------------------------
    def __sub__(self,w):
        return self + (-1*w)

    # ---------------------------------------------------------------
    # Right ADD vectors
    # ---------------------------------------------------------------
    def __radd__ (self, w):
        return (self + w)

    # ---------------------------------------------------------------
    # Right SUB vectors
    # ---------------------------------------------------------------
    def __rsub__ (self, w):
        return (self - w)

    # ---------------------------------------------------------------
    # Right MUL vectors
    # ---------------------------------------------------------------
    def __rmul__ (self, w):
        return (self * w)

    # ---------------------------------------------------------------
    # Right TRUE DIV vectors
    # ---------------------------------------------------------------
    def __rtruediv__ (self, w):
        w = self.__check_other(w)        
        res = []
        try:
            for i in range (self.size):
                res.append (float( w.vector[i]) / self.vector[i])
        except ValueError as err1:
            raise ValueError ("Divide by 0 : ",err1)
            return None
        return (MyVector(self.size, True, fill = 0, init_values=res))

    # ---------------------------------------------------------------
    # Transpose a vector. flip flop between a line and a column
    # ---------------------------------------------------------------
    def transpose(self):
        NewVec = MyVector(self.size, not self.is_col, fill = 0, init_values = self.vector)
        return (NewVec)

# ================================================================================================
# BOOLEAN operations 
# ================================================================================================
    
    # ---------------------------------------------------------------
    # Boolean LOWER THEN between two vectors (by position). Result in a boolean vector
    # ---------------------------------------------------------------
    def __lt__(self, other):
        other = self.__check_other(other)        
        res = MyVector(self.size, fill = 0)
        for i in range (self.size):
            if (self.vector[i] < other.vector[i]):
                res[i] = 1
        return (res)


    # ---------------------------------------------------------------
    # Boolean LOWER EQUAL between two vectors (by position). Result in a boolean vector
    # ---------------------------------------------------------------
    def __le__(self, other):   #      V1 <= 2  
        other = self.__check_other(other)        
        res = MyVector(self.size, fill = 0)
        for i in range (self.size):
            if (self.vector[i] <= other.vector[i]):
                res[i] = 1
        return (res)   # [1, 0, 0 , 1]

    # ---------------------------------------------------------------
    # Boolean EQUAL between two vectors (by position). Result in a boolean vector
    # ---------------------------------------------------------------
    def __eq__(self, other):
        other = self.__check_other(other)        
        res = MyVector(self.size, fill = 0)
        for i in range (self.size):
            if (self.vector[i] == other.vector[i]):
                res[i] = 1
        return (res)        # return an array (list) of booleans - item-wise

    # ---------------------------------------------------------------
    # Boolean NOT EQUAL between two vectors (by position). Result in a boolean vector
    # ---------------------------------------------------------------
    def __ne__(self, other):
        other = self.__check_other(other)        
        res = MyVector(self.size, fill = 0)
        for i in range (self.size):
            if (self.vector[i] != other.vector[i]):
                res[i] = 1
        return (res)

    # ---------------------------------------------------------------
    # Boolean GREATER THEN between two vectors (by position). Result in a boolean vector
    # ---------------------------------------------------------------
    def __gt__(self, other):
        other = self.__check_other(other)        
        res = MyVector(self.size, fill = 0)
        for i in range (self.size):
            if (self.vector[i] > other.vector[i]):
                res[i] = 1
        return (res)

    # ---------------------------------------------------------------
    # Boolean GREATER EQUAL between two vectors (by position). Result in a boolean vector
    # ---------------------------------------------------------------
    def __ge__(self, other):
        other = self.__check_other(other)        
        res = MyVector(self.size, fill = 0)
        for i in range (self.size):
            if (self.vector[i] >= other.vector[i]):
                res[i] = 1
        return (res)

# ================================================================================================
# LINEAR ALGEBRIC operations 
# ================================================================================================
    
    # ---------------------------------------------------------------
    # Dot Product - will multiply and sum each vakue by position between a line and a col vectors
    # ---------------------------------------------------------------
    def dot (self, other):
        # No broadcast for dot operation. Must have two vectors
        if (not isinstance(other,MyVector)):
            raise ValueError("*** Wrong type of parameter")
        # Must have a line and then a col vectors
        if (self.is_col == True or other.is_col == False):
            raise ValueError("*** Dot Product must be line and column ")
        # Length of the line and the col vectors must be identical
        if (self.size != other.size):
            raise ValueError("*** vectors must be of same size")
        
        res = 0
        for i in range (self.size):
            res += self.vector[i]*other.vector[i]
        return res

    # ---------------------------------------------------------------
    # Norm - The size of a vector
    # ---------------------------------------------------------------
    def norm (self):
        res = 0
        for i in range (self.size):
            res += self.vector[i]**2
        return math.sqrt(res)
