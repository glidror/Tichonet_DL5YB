import math

class perceptron(object):

    def __init__ (self, X , Y):
        self.X = X
        self.Y = Y
        self.w = np.zeros((X.shape[0],1), dtype = float)
        self.b = 0.0
        self.dW = np.zeros((X.shape[0],1), dtype = float)
        self.db = 0.0
        
    # --------------------------------------------------
    # Service routines
    # --------------------------------------------------

    # Sigmoid function that will work on a list of values
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))

    # initialize a list with zeros (w[]) and return the list and an additional integer (the b)
    def initialize_with_zeros(self,dim):
        W = np.zeros((dim,1), dtype = float)
        b = 0.0
        return W,b

    # --------------------------------------------------
    # linear analyzis routines
    # --------------------------------------------------

    # forward propagation - calculate the value of the avaerage cost function for the whole set of samples X
    # compared to the expected result Y, using the set of parameters W and b
    def forward_propagation(self):
        m = self.X.shape[1]
        Z = np.dot(self.w.T, self.X)+self.b
        A = self.sigmoid(Z) 
        J= (-1/m)*np.sum(self.Y * np.log(A) + (1-self.Y) * np.log(1-A))
        J = np.squeeze(J)
        return A, J

    # Backword propagation - calculate the values of the difference dW and db
    # for a samples (X), with expected results Y and calculated results A
    def backward_propagation(self, A):
        m = self.X.shape[1]
        n = self.X.shape[0]
        ##dw, db = initialize_with_zeros(n)

        dZ = (1/m)*(A-self.Y)
        dw = np.dot(self.X, dZ.T)
        db = np.sum(dZ)
        return dw, db

    # train the perceptron using a sample db X, expected results Y
    # with number of iteration and aparemeters to indicate the learning rate
    def train(self, num_iterations, learning_rate):
        for i in range(num_iterations):
            A, cost = self.forward_propagation()
            self.dw, self.db = self.backward_propagation(A)
            self.w -= learning_rate*self.dw
            self.b -= learning_rate*self.db
            if (i % 100 == 0):
              print ("Cost after iteration {} is {}".format( i, cost))
        return self.w , self.b

    # predict - get a set of pictures and predict if they are true (1) or false (0)
    # for the specific criteris (e.g. - "is it a cat")
    def predict(self, X, w, b):
        Z = np.dot(w.T,X)+b
        return (np.where(self.sigmoid(Z)>0.5, 1., 0.))

