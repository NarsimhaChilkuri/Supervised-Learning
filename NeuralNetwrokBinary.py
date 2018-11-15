import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """

    Classification (multi-class) usign linear discriminant analysis.

    NOTE: This implementation has high variance -- try running
          the alogorithm a few times using the same hyperparameters.

    Args:

        X : numpy array of dimension pxN, where p is the number of predictors
            and N is the number of trainig examples.
        Y : numpy array of dimension Nx1, where N is the number of training examples.
        nn_structure: list of ints - number of nodes in hidden-layers 1 to L-1
                      (the output layer size should be 1 for binary NN.)
        activatoin: string, activation function to be used in hidden layers. Should
                    be either "sigmoid" or "relu"
        alpha: real > 0, step-size.
        iterations: int > 0, number of iterations of gradient descent.
        init_const: real > 0, number that multiplies initialization of weights - W = np.random.randn(,) * init_cost

    """

    def __init__(self, X, Y, nn_structure, activation, alpha, iterations, init_const):

        self.X = X
        self.Y = Y

        # Useful quantities
        self.N = self.X.shape[1]
        self.p = self.X.shape[0]
        self.nn_structure =  [self.p] + nn_structure + [1]
        self.L = len(self.nn_structure)
        self.dW = {}
        self.db = {}
        self.dA = {}
        self.dZ = {}
        self.A = {0: self.X}
        self.Z = {}

        # Hyper-parameters
        self.activation = activation
        self.alpha = alpha
        self.iterations = iterations
        self.init_const = init_const

        # Parameters
        self.W = {}
        self.b = {}

        self.predictions = 0

    def initialize_weights(self):
        for l in range(1, self.L):
            m = self.nn_structure[l]
            n = self.nn_structure[l - 1]
            self.W[l] = np.random.randn(m, n) * self.init_const
            self.b[l] = np.zeros((m, 1))

    def activation_function(self, f_name, Z):
        """
        Takes in an array and applies an activation function (sigmolid or relu) to it.

        f_name: string -> name of the actiovation function. Should
                          be either sigmoid or relu (rectified linear unit).
        Z: numpy array.
        """
        if f_name == "sigmoid":
            return 1/(1 + np.exp(-Z))
        elif f_name == "relu":
            return np.maximum(0, Z)

    def activation_derivatives(self, f_name, A):
        if f_name == "sigmoid":
            return np.multiply(A, (1 - A))
        elif f_name == "relu":
            sign_A = np.sign(A)
            return np.maximum(0, sign_A)


    def forward_prop(self):

        act_fnc = self.activation
        for l in range(1, self.L):
            act_fnc = self.activation
            if l == self.L - 1:
                act_fnc = "sigmoid"
            self.Z[l] = np.dot(self.W[l], self.A[l-1]) + self.b[l]
            self.A[l] = self.activation_function(act_fnc, self.Z[l])


    def backward_prop(self):

        # first figure out dL/dA[L-1] and the rest of derivtives at L -1 or the output layer
        Y_reshaped = np.reshape(self.Y, self.A[self.L - 1].shape)
        self.dA[self.L - 1] = - np.divide(Y_reshaped, self.A[self.L - 1]) + np.divide((1 - Y_reshaped), (1 - self.A[self.L - 1]))

        activation_back = self.activation_derivatives("sigmoid", self.A[self.L - 1])
        self.dZ[self.L - 1] = np.multiply(self.dA[self.L - 1], activation_back)

        self.dW[self.L - 1] = 1/self.N * np.dot(self.dZ[self.L - 1], self.A[self.L - 2].T)
        self.db[self.L - 1] = 1/self.N * np.sum(self.dZ[self.L - 1], axis=1, keepdims=True)


        # Computing the derivatives for the rest of the layers 1, 2, 3, ...., L-2.
        for l in reversed(range(1, self.L - 1)):

            self.dA[l] = np.dot(self.W[l + 1].T, self.dZ[l + 1])

            activation_back = self.activation_derivatives(self.activation, self.A[l])
            self.dZ[l] = np.multiply(self.dA[l], activation_back)

            self.dW[l] = 1/self.N * np.dot(self.dZ[l], self.A[l - 1].T)
            self.db[l] = 1/self.N * np.sum(self.dZ[l], axis=1, keepdims=True)

    def gradient_descent(self):

        for l in range(1, self.L):

            self.W[l] = self.W[l] - self.alpha * self.dW[l]
            self.b[l] = self.b[l] - self.alpha * self.db[l]

    def predict(self):

        self.initialize_weights()

        for i in range(0, self.iterations):
            #print(i)
            self.forward_prop()
            self.backward_prop()
            self.gradient_descent()

        self.forward_prop()
        self.predictions = np.round(self.A[self.L - 1])
