import numpy as np

class LogisticReg:

    """
    Logistic classification where weights are obtained my minimizing the
    log-likelihood loss. Minimization is done using gradient descent.

    Args:

        X : numpy array of dimension Nxp, where N is the number of trainig examples
            and p is the number of predictors.
        Y : numpy array of dimension Nx1, where N is the number of training examples.
       alpha : real greated than 0; this is the step-size.
       iterations : int - number of gradient descent iterations.

    """

    def __init__(self, X, Y, alpha, iterations):
        self.X = self.add_ones_column(X)
        self.Y = Y

        # Parameters
        self.W = np.random.rand(self.X.shape[1], 1)

        #Hyper-parameters
        self.alpha = alpha
        self.iterations = iterations

        self.predictions = 0

    def add_ones_column(self, A):
        ones_column = np.ones((A.shape[0], 1))
        B = np.append(A, ones_column, axis=1)
        return B


    def compute_gradient(self):

        exp_W_X = np.exp(-np.matmul(self.X, self.W))
        p1 = np.multiply(self.X, exp_W_X)
        p2 = (1 - self.Y)/(exp_W_X) - 1/(1 + exp_W_X)
        grad =  np.sum(np.multiply(p1, p2), axis=0, keepdims=True)
        grad = grad.T
        return grad

    def gradient_descent(self):
        gradient = self.compute_gradient()
        self.W = self.W - self.alpha * gradient

    def predict(self):

        for i in range(0, self.iterations):
            self.gradient_descent()

        self.predictions =  np.round(1/ (1+ np.exp(-np.matmul(self.X, self.W))))
