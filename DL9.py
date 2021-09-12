import numpy as np
import os
import h5py
from sklearn.metrics import classification_report, confusion_matrix
import math

class ITrainable:
    def __init__(self):
        self.is_train = False

    def set_train(self, is_train):
        self.is_train = is_train

    def forward_propagation(self, prev_A):
        raise NotImplementedError("forward_propagation not implemented: ITrainable is an interface")

    def backward_propagation(self, dA):
        raise NotImplementedError("backward_propagation not implemented: ITrainable is an interface")

    def update_parameters(self):
        raise NotImplementedError("update_parameters not implemented: ITrainable is an interface")
    
    def save_parameters(self, file_path):
        pass
    def restore_parameters(self, file_path):
        pass
    def parms_to_vec(self):
        pass
    def vec_to_parms(self, vec):
        pass
    def gradients_to_vec(self):
        pass
    def to_layers(self):
        return np.array([self])
    def regularization_cost(self,m):
        return 0


class DLLinearLayer(ITrainable):
    def __init__(self,name, num_units, input_size, alpha, optimization=None, regularization=None):
        ITrainable.__init__(self)
        self.regularization = regularization
        self.name = name
        self.alpha = alpha
        self.optimization = optimization
        self.num_units = num_units
        self.input_size = input_size
        self.W = np.empty((num_units,input_size))
        self.W_He_initialization()
        self.b = np.zeros((num_units,1), dtype=float)
        # optimization parameters
        if self.optimization == 'adaptive':
            self.adaptive_cont = 1.1
            self.adaptive_switch = 0.5
            self.adaptive_W = np.full(self.W.shape,alpha,dtype=float)
            self.adaptive_b = np.full(self.b.shape,alpha,dtype=float)      
        # regularization parameters
        if regularization == "L2":
            self.L2_lambda = 0.6
        if regularization == "dropout":
            self.dropout_keep_prob = 0.7
   
    def __str__(self):
        s = f"{self.name} Function:\n"
        s += f"\tlearning_rate (alpha): {self.alpha}\n"
        s += f'\tnum inputs:{self.input_size}\n' 
        s += f'\tnum units:{self.num_units}\n' 
        if self.optimization != None:
            s += f"\tOptimization: {self.optimization}\n"
            if self.optimization == "adaptive":
                s += f"\t\tadaptive parameters:\n"
                s += f"\t\t\tcont: {self.adaptive_cont}\n"
                s += f"\t\t\tswitch: {self.adaptive_switch}\n"
        s += "\tParameters shape:\n"
        s += f"\t\tW shape: {self.W.shape}\n"
        s += f"\t\tb shape: {self.b.shape}\n"
        s += self.regularization_str()
        return s;

    def regularization_str(self):
        s = ""
        if self.regularization != None:
            s += f"\tregularization:{self.regularization}\n"
            if self.regularization == "L2":
                s += f"\t\tL2 parameters:\n\t\t\tlambda:{self.L2_lambda}\n"
            if self.regularization == "dropout":
                s += f"\t\tdropout parameters:\n\t\t\tkeep prob:{self.dropout_keep_prob}\n"
        return s

    def forward_dropout(self,prev_A):
        prev_A = np.array(prev_A, copy=True)
        if self.is_train and self.regularization == "dropout":
            self._D = np.random.rand(*prev_A.shape)     
            self._D = self._D < self.dropout_keep_prob
            prev_A *= self._D                          
            prev_A /= self.dropout_keep_prob
        
        return prev_A

    def regularization_cost(self,m):
        if self.regularization == "L2":
            return (self.L2_lambda/(2*m)) * np.sum(np.square(self.W))
        return 0

    def W_He_initialization(self):
        self.W = DLLinearLayer.normal_initialization(self.W.shape,np.sqrt(2/self.input_size))
    def W_Xaviar_initialization(self):
        self.W = DLLinearLayer.normal_initialization(self.W.shape,np.sqrt(1/self.input_size))

    def forward_propagation(self, prev_A):
        self.prev_A = self.forward_dropout(prev_A)
        Z = self.W @ self.prev_A+self.b
        return Z

    def backward_dropout(self,dA_prev):    
        if self.regularization == 'dropout':
            dA_prev = dA_prev * self._D 
            dA_prev /= self.dropout_keep_prob
        return dA_prev    

    def backward_propagation(self, dZ):
        db_m_values = dZ * np.full((1,self.prev_A.shape[1]),1)
        self.db = np.sum(db_m_values, keepdims=True, axis=1)
        self.dW = dZ @ self.prev_A.T
        if self.regularization == 'L2':
            m = dZ.shape[-1]
            self.dW += (self.L2_lambda/m) * self.W
        dA_prev = self.W.T @ dZ
        if self.regularization == 'dropout':
            dA_prev = self.backward_dropout(dA_prev)
        return dA_prev 

    def update_parameters(self):
        if self.optimization == 'adaptive':
            self.adaptive_W *= np.where(self.adaptive_W * self.dW > 0, self.adaptive_cont, -self.adaptive_switch)
            self.W -= self.adaptive_W 
            self.adaptive_b *= np.where(self.adaptive_b * self.db > 0, self.adaptive_cont, -self.adaptive_switch)
            self.b -= self.adaptive_b 
        else:
            self.W -= self.alpha * self.dW
            self.b -= self.alpha * self.db
    
    def save_parameters(self, file_path):
        with h5py.File(file_path+"/"+self.name+'.h5', 'w') as hf:
            hf.create_dataset("W",  data=self.W)
            hf.create_dataset("b",  data=self.b)

    def restore_parameters(self, file_path):
        with h5py.File(file_path+"/"+self.name+'.h5', 'r') as hf:
            if self.W.shape != hf['W'][:].shape:
                raise ValueError(f"Wrong W shape: {hf['W'][:].shape} and not {self.W.shape}")
            self.W = hf['W'][:]
            if self.b.shape != hf['b'][:].shape:
                raise ValueError(f"Wrong b shape: {hf['b'][:].shape} and not {self.b.shape}")
            self.b = hf['b'][:]

    def parms_to_vec(self):
        return np.concatenate((np.reshape(self.W,(-1,)), np.reshape(self.b, (-1,))), axis=0)
    
    def vec_to_parms(self, vec):
        self.W = vec[0:self.W.size].reshape(self.W.shape)
        self.b = vec[self.W.size:].reshape(self.b.shape)
    
    def gradients_to_vec(self):
        return np.concatenate((np.reshape(self.dW,(-1,)), np.reshape(self.db, (-1,))), axis=0)

    @staticmethod
    def normal_initialization(shape,factor=0.01):
        return factor*np.random.randn(*shape)


class DLNetwork(ITrainable):
    def __init__(self,name):
        ITrainable.__init__(self)
        self.name = name
        self.layers = []

    def __str__(self):
        s = f"{self.name}:\n"
        for l in self.layers:
            s += str(l)
        return s

    def add(self,iTrainable):
        for l in self.layers:
            if l.name == iTrainable.name:
                raise ValueError(f"{iTrainable.name} already exists")
        self.layers.append(iTrainable)

    def forward_propagation(self,X):
        Al=X
        for l in self.layers:
           Al = l.forward_propagation(Al)
        return Al

    def backward_propagation(self, dY_hat):
        dAl = dY_hat
        for l in reversed(self.layers):
           dAl = l.backward_propagation(dAl)
        return dAl

    def update_parameters(self):
        for l in self.layers:
           l.update_parameters()

    def save_parameters(self, directory_path):
        directory = directory_path+"/"+self.name
        os.makedirs(directory, exist_ok=True)
        for l in self.layers:
            l.save_parameters(directory)

    def restore_parameters(self, directory_path):
        directory = directory_path+"/"+self.name
        for l in self.layers:
            l.restore_parameters(directory)
    
    def to_layers(self):
        layers = np.array([])
        for l in self.layers:
                layers = np.concatenate((layers,l.to_layers()))
        return layers
    
    def set_train(self, is_train):
        for l in self.layers:
           l.set_train(is_train)
    
    def regularization_cost(self,m):
        cost = 0
        for l in self.layers:
           cost += l.regularization_cost(m)
        return cost
    
    def print_regularization_cost(self,m):
        s = ""
        L = len(self.layers)
        for l in self.layers:
           reg_cost = l.regularization_cost(m)
           if reg_cost > 0:
                s += f"\t{self.layers[l].name}: {reg_cost}\n"
        return s

class DLActivation(ITrainable):
    def __init__(self, activation):
        ITrainable.__init__(self)
        self.name = activation
        if activation == "sigmoid":
            self.forward_propagation = self.sigmoid
            self.backward_propagation = self.sigmoid_dZ
        elif activation == "relu":
            self.forward_propagation = self.relu
            self.backward_propagation = self.relu_dZ
        elif activation == "tanh":
            self.forward_propagation = self.tanh
            self.backward_propagation = self.tanh_dZ
        elif activation == "leaky_relu":
            self.forward_propagation = self.leaky_relu
            self.backward_propagation = self.leaky_relu_dZ
            self.leaky_relu_d = 0.01
        elif activation == "softmax":
            self.forward_propagation = self.softmax
            self.backward_propagation = self.softmax_dZ
        else:
            raise NotImplementedError("Unimplemented activation:", activation)

    def __str__(self):
        s = f"Activation function: {self.name}"
        if self.name == "leaky_relu":
            s += f"\t d = {self.leaky_relu_d}"
        return s
    
    def sigmoid(self,Z):
        self.S = 1/(1+np.exp(-Z))
        return self.S

    def sigmoid_dZ(self, dS):
        dZ = dS * self.S * (1-self.S)
        return dZ
    
    def leaky_relu(self,Z):
        self.Z = np.copy(Z)
        dZ = np.where(self.Z <= 0, self.leaky_relu_d*Z, Z)
        return dZ

    def leaky_relu_dZ(self, dA):
        dZ = np.where(self.Z <= 0, self.leaky_relu_d*dA, dA)
        return dZ

    def tanh(self,Z):
        self.A = np.tanh(Z)
        return self.A

    def tanh_dZ(self, dA):
        dZ = dA *  (1-self.A * self.A)
        return dZ

    def relu(self,Z):
        self.Z = np.copy(Z)
        A = np.maximum(0,Z)
        return A
    
    def relu_dZ(self, dA):
        dZ = np.where(self.Z <= 0, 0, dA)
        return dZ

    def softmax(self,Z):
        eZ = np.exp(Z)
        A = eZ/np.sum(eZ,axis=0)
        return A

    def softmax_dZ(self, dZ):
        #an empty backward function that gets dZ and returns it
        #just to comply with the flow of the model
        return dZ
    
    def update_parameters(self):
        pass

class DLNeuronsLayer(DLNetwork):
    def __init__(self,name,num_units,input_size, activation, alpha,optimization=None, regularization=None):
        DLNetwork.__init__(self, name)
        self.linear = DLLinearLayer("linear",num_units,input_size,alpha,optimization, regularization)
        self.activation = DLActivation(activation)
        self.add(self.linear)
        self.add(self.activation)

class DLModel:
    def __init__(self,name,iTrainable, loss):
        self.name = name
        self.inject_str_func = None
        self.iTrainable = iTrainable
        self.loss = loss
        if loss == "square_dist":
            self.loss_forward = self.square_dist
            self.loss_backward = self.dSquare_dist
        elif loss == "cross_entropy":
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.dCross_entropy
        elif loss == "categorical_cross_entropy":
            self.loss_forward = self.categorical_cross_entropy
            self.loss_backward = self.dCategorical_cross_entropy        
        else:
            raise NotImplementedError("Unimplemented loss function: " + loss)

    def __str__(self):
        s = self.name + "\n"
        s += "\tLoss function: " + self.loss + "\n"
        s += "\t"+str(self.iTrainable) + "\n"
        return s


    def square_dist(self, Y_hat, Y):
        errors = (Y_hat - Y)**2
        return errors

    def dSquare_dist(self, Y_hat, Y):
        m = Y.shape[1]
        dY_hat = 2*(Y_hat - Y)/m
        return dY_hat

    def categorical_cross_entropy(self, Y_hat, Y):
        eps = 1e-10
        Y_hat = np.where(Y_hat==0,eps,Y_hat)
        Y_hat = np.where(Y_hat == 1, 1-eps,Y_hat)
        errors = -np.sum(np.multiply(Y, np.log(Y_hat)),axis=0)
        return errors

    def dCategorical_cross_entropy(self, Y_hat, Y):
        # compute dZ directly
        m = Y.shape[1]
        dZ = (Y_hat - Y)/m
        return dZ
    
    def compute_cost(self, Y_hat, Y):
        m = Y.shape[1]
        errors = self.loss_forward(Y_hat, Y)
        J = np.sum(errors)/m + self.iTrainable.regularization_cost(m)
        return J

    def cross_entropy(self, Y_hat, Y):
        eps = 1e-10
        Y_hat = np.where(Y_hat==0,eps,Y_hat)
        Y_hat = np.where(Y_hat == 1, 1-eps,Y_hat)
        logprobs = -((1 - Y)*np.log(1 - Y_hat)+Y*np.log(Y_hat))
        return logprobs

    def dCross_entropy(self, Y_hat, Y):
        eps = 1e-10
        Y_hat = np.where(Y_hat==0,eps,Y_hat)
        Y_hat = np.where(Y_hat == 1, 1-eps,Y_hat)
        m = Y_hat.shape[1]
        dY_hat =(1-Y)/(1-Y_hat)-Y/Y_hat
        return dY_hat/m

    def train(self, X, Y, num_epocs, mini_batch_size=64):
        seed = 10
        self.iTrainable.set_train(True)
        print_ind = max(num_epocs//100, 1)
        costs = []
        for i in range(num_epocs):
            minibatches = DLModel.random_mini_batches(X, Y, mini_batch_size , seed)
            seed += 1
            for minibatch in minibatches:
                Y_hat = self.forward_propagation(minibatch[0])
                self.backward_propagation(Y_hat, minibatch[1])
                self.update_parameters()
            #record progress
            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Y_hat, minibatch[1])
                costs.append(J)
                #user defined info
                inject_string = ""
                if self.inject_str_func != None:
                    inject_string = self.inject_str_func(self, X, Y, Y_hat)
                print(f"cost after {i} full updates {100*i/num_epocs}%:{J}" + inject_string)
        costs.append(self.compute_cost(Y_hat, minibatch[1]))
        self.iTrainable.set_train(False)
        return costs

    def forward_propagation(self, X):
        return self.iTrainable.forward_propagation(X)

    def backward_propagation(self, Y_hat,Y):
        dY_hat = self.loss_backward(Y_hat, Y)
        self.iTrainable.backward_propagation(dY_hat)

    def update_parameters(self):
        self.iTrainable.update_parameters()

    def check_backward_propagation(self, X, Y, epsilon=1e-4, delta=1e-7):    
        # forward propagation
        AL = self.forward_propagation(X)           
        #backward propagation
        self.backward_propagation(AL,Y)
        #un case the network 
        layers = self.iTrainable.to_layers()
        L = len(layers)
        # check gradients of each layer separatly
        for main_l in reversed(range(L)):
            layer = layers[main_l]
            parms_vec = layer.parms_to_vec()
            if parms_vec is None:
                continue
            gradients_vec = layer.gradients_to_vec()
            n = parms_vec.shape[0]
            approx = np.zeros((n,))

            for i in range(n):
                # compute J(parms[i] + delta)
                parms_plus_delta = np.copy(parms_vec)                
                parms_plus_delta[i] = parms_plus_delta[i] + delta
                layer.vec_to_parms(parms_plus_delta)
                AL = self.forward_propagation(X)   
                f_plus = self.compute_cost(AL,Y)

                # compute J(parms[i] - delta)
                parms_minus_delta = np.copy(parms_vec)                
                parms_minus_delta[i] = parms_minus_delta[i]-delta  
                layer.vec_to_parms(parms_minus_delta)
                AL = self.forward_propagation(X)   
                f_minus = self.compute_cost(AL,Y)

                approx[i] = (f_plus - f_minus)/(2*delta)
            
            layer.vec_to_parms(parms_vec)
            if (np.linalg.norm(gradients_vec) + np.linalg.norm(approx)) > 0:
                diff = (np.linalg.norm(gradients_vec - approx) ) / ( np.linalg.norm(gradients_vec)+ np.linalg.norm(approx) ) 
                if diff > epsilon:
                    return False, diff, main_l
        return True, diff, L

    def confusion_matrix(self, X, Y):
        prediction = self.forward_propagation(X)
        prediction_index = np.argmax(prediction, axis=0)
        Y_index = np.argmax(Y, axis=0)
        right = np.sum(prediction_index == Y_index)
        print("accuracy: ",str(right/len(Y[0])))
        print(confusion_matrix(prediction_index, Y_index))

    @staticmethod
    def to_one_hot(num_categories, Y):
        m = Y.shape[0]
        Y = Y.reshape(1, m)
        Y_new = np.eye(num_categories)[Y.astype('int32')]
        Y_new = Y_new.T.reshape(num_categories, m)
        return Y_new
    
    @staticmethod
    def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
        minibatchs = []
        np.random.seed(seed) 
        m = Y.shape[1]
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((-1,m))
        num_complete_minibatches = math.floor(m/mini_batch_size)
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:, mini_batch_size*k : (k+1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, mini_batch_size*k : (k+1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            minibatchs.append(mini_batch)
        if m > num_complete_minibatches * mini_batch_size:
            mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            minibatchs.append(mini_batch)
        return minibatchs