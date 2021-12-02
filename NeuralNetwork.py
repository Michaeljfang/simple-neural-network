import numpy as np
from copy import deepcopy

"""
The simplest fully neural network with the most default options you'll find anywhere.
Activation: logistic, capped or uncapped Rectified Linear Unit (ReLU), or hyperbolic tangent (tanh).
Weights initialization: Zeros, random, He, or Xavier.

Ctrl-F list:
**kwargs                            **options
learning rate, lambda               learnrate
input, x, data, training            X
output, y, y hat, y_hat             y
bias                                b
weights                             W
hypothesis function                 h
activation                          activate
regularization function             regularizer
add data, initialize                receive
"""


class NeuralNetwork:

    def __init__(self, list_of_hidden_dimensions, learnrate=0.01, **options):
        """
        list_of_hidden_dimensions   LIST.INT+           Length of list is depth, value at each index is the dimension*/size/number of nodes of layer.
        learnrate                   NUMBER              Rate of learning, "lambda"
        **options                   KEYWORD.ARGS        Options below:
            |-activation            STRING/LIST         Activation function for each or all layers. If type list, specify for layer; if string, specify for all.
            |   types: logistic, relu, relu1, tanh
            |   relu1 is relu but with a cropped range 0~1
            |-activation_steepness  NUMBER              How "steep" the activation function is. See "activate()"for the exact parameter changed.
            |-W_initials            STRING              How to initialize weights. At the init stage, this only saves to a variable because not all dimensions are known yet.
            |-basis                 STRING              Data transformation. E.g. include squares or other functions of the data to learn.
            |   none, poly[power].
            |   e.g.: poly3 to use x +x^2 +x^3. Constant term is always included even when not using poly basis.
            |-regularizer           STRING              How to limit the values of parameters
            |  sum, squaresum
            |-error_method          STRING              How to calculate error. crossentropy: for classification; mse: for regression.
            |-dynamiclearn          STRING              How to control the learning rate.
            |   none/off/false, momentum
        """

        # *dimension means the size of the ordered tuple used to represent one data point.
        # e.g. for points on a 2D x-y space, or a data point that has 2 features, the dimension is 2.


        # This block deals with common errors in input.
        assert type(list_of_hidden_dimensions)==list, "List of hidden dimensions must be a python or numpy list"
        assert len(list_of_hidden_dimensions)>=1, "There must be at least 1 hidden layer"
        assert min(list_of_hidden_dimensions)>=2, "Minimum hidden layer(s) dimension(s) must be 2 or higher"
        assert np.product(list_of_hidden_dimensions)<=1000000, "Exceeded arbitrary network size limit. Change the limit in source code if confident"
        
        try:
            list_of_hidden_dimensions=[int(j) for j in list_of_hidden_dimensions] # convert all values to ints. if fails, list has wrong types.
        except:
            raise TypeError("list dimension must be a number. wrong types detected.")


        # Here are ALL the instance variables #######################################
        # housekeeping: saving the inputs
        self.activation=options.get("activation", "logistic")       # STRING    default activation is logistic
        self.list_of_hidden_dimensions=list_of_hidden_dimensions    # LIST      temp variable to store list of hidden dimensions
        self.output_dimension=None                                  # INT       output dimension.
        self.layers=len(list_of_hidden_dimensions)+2                # INT       the layers count include input and output layers.
        self.dimensions=None                                        # LIST      list of dimensions including input and output dimensions.

        self.initialized=False                                      # BOOL      if there is data to train.
        self.iterations=0                                           # INT       how many cycles it's been trained.

        # Parameters relating directly to data and calculation results.
        self.X=None                                                 # NP.NDARRAY    training_data, "X"
        self.tX=None                                                # NP.NDARRAY    transformed training data according to a basis.
        self.y=None                                                 # NP.NDARRAY    reference_data, "y"
        self.W=[0]                                                  # LIST          weights, a list of weights for each layer.
        # X-> a->h --> a->h --> a-> h ...
        self.h=[None]*self.layers                                   # LIST          partial results after a particular layer. This will include X as the first entry.
        self.a=[None]*self.layers                                   # LIST          partial results after xW+b but before activation.
        self.a[0]=0                                                 #               the first value of "a" is not used. to keep it the same length as "h".
        self.datasize=0                                             # INT           calculated. number of data points calculated from X.
        self.W_initials=options.get("W_initials", "zeros")          # how to initialize weights.

        self.error=[]
        self.error_method=options.get("error_method", "crossentropy")

        # Parameters relating to transformations and analysis including methods and the methods' parameters
        self.regularizer=options.get("regularizer", None)           # How to regularize parameters

        self.basis=options.get("basis", None)                       # How to transform input data.
        
        self.activation_steepness=options.get("activation_steepness", 1)
        assert self.activation_steepness>0
        self.base_learnrate=learnrate                               # base learning rate. If dynamiclearn is off, this will be the actual learning rate.
        self.learn_rate=None                                        # actual learning rate used during gradient descent.
        self.dynamic_learn=options.get("dynamiclearn", False)
        self.descent_history=[]                                      # previous descent step. Used in dynamiclearn: e.g. when learn rate is controlled by previous desent step in gradient descent with momentum
        self.momentum_decay=options.get("momentum_decay", 0.6)
        # Above are ALL the instance variables ######################################
        return None

    def __repr__(self):
        """Self-representation when printed"""
        if self.initialized==False:
            return "Simple Neural Network at %s. [Not initialized] Number of hidden layers: %s. \nUse .receive() to provide data." % (\
                hex(id(self)), str(len(self.list_of_hidden_dimensions))\
                )
        else:
            return "Simple Neural Network at %s. Number of hidden layers: %s. Input-output dims: %s->%s. Iterations: %s. \nUse .reset() to reset, .train(... resume=True) to resume training." % (\
                hex(id(self)), str(len(self.list_of_hidden_dimensions)), str(self.dimensions[0]), str(self.dimensions[-1]),\
                str(self.iterations)
                )


    def reset(self, list_of_hidden_dimensions=None):
        """Resets weights and erases data by calling init. Optionally pass in new parameters"""
        self.layers=0
        if list_of_hidden_dimensions==None: hidden=self.list_of_hidden_dimensions
        else: hidden=list_of_hidden_dimensions
        self.__init__(hidden)
        return None

    #########################################################################

#█▄█ ██▀ █   █▀▄ ██▀ █▀▄ ▄▀▀
#█ █ █▄▄ █▄▄ █▀  █▄▄ █▀▄ ▄██

    def basis_convert(self, input):
        """converts the input data using a basis. Returns deep copy."""
        if self.basis==None:
            transformed=deepcopy(input)
            transformed=np.insert(transformed, 0, 1, axis=1)
            return transformed
        elif self.basis[:4]=="poly":
            power=int(self.basis[4:])
            assert power>0, "Polynomial transformation power must be positive. E.g. poly3"
            if power==1: return deepcopy(input)
            else:
                build=deepcopy(input)
                for p in range(2, power+1):
                    build=np.concatenate((build, np.power(input, p)), axis=1)
                return build


    def regularize(self, params=None):
        """gets regularization term of one parameter"""
        if self.regularizer==None or params==None:
            return 0
        else:
            raise NotImplementedError

    def activate(self, input, activation_override=None):
        """
        Helper.
        activation functions with a steepness parameter.
        steepness controls the coefficient of input, i.e. the "s" term below:
        1/(1+exp(-sx))          - logistic
        tanh(sx)                - hyperbolic tangent
        sx                      - ReLU
        none, identity, i       - no "activation"
        """
        if activation_override==None:
            activation=self.activation
        else:
            activation=activation_override
        steepness=self.activation_steepness
        if activation=="tanh":
            return np.tanh(steepness*input)
        elif activation=="logistic" or self.activation=="sigmoid":
            return np.divide(1.0, 1+np.exp(-1*steepness*input))
        elif activation=="relu":
            return np.maximum(0, steepness*input)
        elif activation=="relu1":
            return np.minimum(1, np.maximum(0, steepness*input))
        elif activation=="none" or activation=="identity" or activation=="i":
            return steepness*input
    
    def activation_derivative(self, input, activation_override=None):
        """
        Helper.
        Derivatives of activation functions
        """
        if activation_override==None:
            activation=self.activation
        else:
            activation=activation_override
        steepness=self.activation_steepness
        if activation=="tanh":
            return steepness*(1-np.power(np.tanh(steepness*input), 2))
        elif activation=="logistic" or self.activation=="sigmoid":
            f=np.divide(1.0, 1+np.exp(-1*steepness*input))
            return steepness*(1-f)*f
        elif activation=="relu":
            return steepness*(input>=0)
        elif activation=="relu1":
            return steepness*((1*input>=0)*(1*input<=1))
        elif activation=="none" or activation=="identity" or activation=="i":
            return steepness

    #########################################################################

#               
#      ▀███           ██           
#        ██           ██           
#   ▄█▀▀███  ▄█▀██▄ ██████ ▄█▀██▄  
# ▄██    ██ ██   ██   ██  ██   ██  
# ███    ██  ▄█████   ██   ▄█████  
# ▀██    ██ ██   ██   ██  ██   ██  
#  ▀████▀███▄████▀██▄ ▀████████▀██▄


    def receive(self, training_data, reference_data):
        """
        saves data and resets weights. figures out the sizes/dimensions for later access, sets up for training

        """
        # first convert to numpy.ndarray. Don't want matrices.f.shape
        training_data=np.asarray(training_data)
        reference_data=np.asarray(reference_data)

        # The transponse confusion:
        # The number of rows represents the number of data points, and the number of columns represent the features of one data point.
        # This would require the h(x) function at each layer to be xW+b

        assert len(training_data.shape)==2 and len(reference_data.shape)==2, "Wrong formatting for training or testing data. Must be 2D arrays"
        self.output_dimension=int(reference_data.shape[1])

        self.X=training_data
        self.tX=self.basis_convert(self.X)
        self.h[0]=self.tX
        self.y=reference_data
        self.datasize=self.tX.shape[0]

        # get the dimensions for each layer. self.dimensions will now contain dimensions of all layers, including input and output layers.
        # the length of self.dimensions is one more than the number of weight matrices.
        self.dimensions=[self.tX.shape[1]]+self.list_of_hidden_dimensions+[self.output_dimension]

        for layer in range(1, self.layers):
            # options for weights initialization
            # formula: xW+b
            if self.W_initials=="zeros":
                self.W.append(np.zeros((self.dimensions[layer-1], self.dimensions[layer]))+0.0000001)
            elif self.W_initials=="random":
                self.W.append(np.random.rand( self.dimensions[layer-1], self.dimensions[layer]) )
            elif self.W_initials=="he":
                self.W.append(np.random.rand(self.dimensions[layer-1], self.dimensions[layer])*np.sqrt(2/self.dimensions[layer-1]))
            elif self.W_initials=="xavier":
                self.W.append(np.random.rand(self.dimensions[layer-1], self.dimensions[layer])*np.sqrt(1/self.dimensions[layer-1]))
            elif self.W_initials=="ones":
                self.W.append(np.ones((self.dimensions[layer-1], self.dimensions[layer])))
            self.descent_history.append(0)

            # adding small value to biases to activate ReLU. Got this off the internet.
            #self.b.append(np.zeros((self.datasize, self.dimensions[layer]))+0.0001)
        self.initialized=True

        return self.W

#########################################
#  ██                    ██             
#  ██                                   
#██████▀███▄███ ▄█▀██▄ ▀███ ▀████████▄  
#  ██    ██▀ ▀▀██   ██   ██   ██    ██  
#  ██    ██     ▄█████   ██   ██    ██  
#  ██    ██    ██   ██   ██   ██    ██  
#  ▀████████▄  ▀████▀██▄████▄████  ████▄
#########################################

    def train(self, iterations):
        """Start or resume the training process. If interrupted, intermediate results should be available."""
        if self.X is None or self.y is None or len(self.W)==1 or self.dimensions==None:
            raise ValueError("Not initialized for training. Use .receive() to provide training data and reference data.")
        
        for i in range(iterations):
            
            # training cycle

            #█▀ ▄▀▄ █▀▄ █   █ ▄▀▄ █▀▄ █▀▄
            #█▀ ▀▄▀ █▀▄ ▀▄▀▄▀ █▀█ █▀▄ █▄▀

            for l in range(1, self.layers): # "l" for "layer"
                self.a[l]=np.matmul(self.h[l-1], self.W[l])
                self.h[l]=self.activate(self.a[l])


            # K(W, b) error with respect to weights and biases.
            r=self.h[-1] # r: result
            self.learn_rate=self.base_learnrate
 

            #############################################################################################################
            # error and starting propagation (with error's gradient)
            propagated_gradient=0

            if self.error_method=="crossentropy":
                main_term=-1*np.mean(np.sum(np.multiply(np.log(r), self.y)-np.multiply((1-self.y), np.log(1-r)), axis=1, keepdims=1), axis=0)[0]
                propagated_gradient=-1*np.divide(self.y, r)+np.divide((1-self.y), (1-r))
            elif self.error_method=="mse":
                main_term=np.sum(np.square(np.subtract(r, self.y)))/self.datasize
                #print("main_term", main_term)
                propagated_gradient=2*(r-self.y)
            regularization_term=self.regularize()*self.learn_rate
            self.error.append(main_term+regularization_term)

            
            #############################################################################################################

            #██▄ ▄▀▄ ▄▀▀ █▄▀ █   █ ▄▀▄ █▀▄ █▀▄
            #█▄█ █▀█ ▀▄▄ █ █ ▀▄▀▄▀ █▀█ █▀▄ █▄▀


            for l in range(len(self.W)-1, 0, -1):
                assert propagated_gradient.shape==self.a[l].shape # sanity check
                propagated_gradient=np.multiply(propagated_gradient, self.activation_derivative(self.a[l]))
                error_derivative_wrt_W_l=np.matmul(self.h[l-1].T, propagated_gradient)#+self.learn_rate*self.regularize() ################################################################# TODO Backprop not implemented
                if self.dynamic_learn=="momentum":
                    error_derivative_wrt_W_l-=self.descent_history[l-1]*self.momentum_decay
                    self.descent_history[l-1]=error_derivative_wrt_W_l

                self.W[l]=np.subtract(self.W[l], self.learn_rate*error_derivative_wrt_W_l) # update weights
                propagated_gradient=np.matmul(propagated_gradient, self.W[l].T)

                
            self.iterations+=1
        # CHECK ACTIVATION VARIABLE TYPE. LIST OR STRING

        return None

#█▀▄ █▀▄ ██▀ █▀▄ █ ▄▀▀ ▀█▀
#█▀  █▀▄ █▄▄ █▄▀ █ ▀▄▄  █ 

    def predict(self, input, reference=None):
        result=self.basis_convert(input)
        for l in range(1, self.layers): # "l" for "layer"
            result=np.matmul(result, self.W[l])
            result=self.activate(result)
        if reference !=None:
            raise NotImplementedError
        
        return result

    ###################################################################################################
    # useless functions
    def __eq__(self, other):
        """lmao"""
        if type(self)==type(other) and self.__dict__==other.__dict__:
            return True
        else: return False

    def params(self):
        """prints the weights and biases. default python printing leaves the last row of a numpy array in the same line as the first row of the next numpy array. annoying"""
        print("weights ##############")
        for pweights in self.W:
            print("\n",pweights)

        return None

    def partialresults(self):
        """prints the partial results from each layer for debugging"""
        for n in range(1, self.layers):
            print("\n########################\n###################### layer", n, "\na")
            print(self.a[n])
            print("h")
            print(self.h[n])
        
        print("\n######\n##### prediction")
        print(self.h[-1])
        return None
    
    def errorhistory(self, historylength=None, by=1):
        """prints error history"""
        print("\n############################################# error history")
        if historylength==None or historylength>len(self.error): historylength=len(self.error)
        for v in range(-historylength, -1, by):
            if -self.error[v] >1000:
                print(str(-self.error[v]).split(".")[0]+",")
            else:
                print(str(-self.error[v])[:8]+",")


def binary():
    b=NeuralNetwork([3], learnrate=1, W_initials="he")
    b.receive([[1, 3], [2, 2], [1, 1]], [[0, 1, 1], [1, 1, 0], [1, 0, 0]])
    b.train(100)
    print("predict")
    print(b.predict([[1.1, 3.2], [1.9, 2.1], [0.8, 1.3]]))
    #b.errorhistory()

def regression():
    b=NeuralNetwork([3], learnrate=0.001, W_initials="he", activation="i", error_method="mse", dynamic_learn="momentum")
    b.receive([[0.5], [1.1], [2]], [[2.75], [3.97], [4]])
    b.train(2000)
    #b.params()
    print("predict")
    print(b.predict([[0], [1], [3]]))
    b.errorhistory(100, 10)

#regression()
#binary()

# gradient descent with momentum broken
