import numpy as np

class LassoReg:
    """
    Regression weights are obtained by minimizing RSS + L1_norm(weights)

    Args:

        X : matrix of dimension NxM, where N is the number of trainig examples
            and M is the number of predictors.
        Y : vector of dimension Nx1, where N is the number of training examples.
        alpha : real number that is greater than or equal to zero. This works
                as the regularization coefficient. alpha = 0 results in basic
                linear regression.

    """

    def __init__(self, X, Y, alpha):


        self.X = np.column_stack((np.ones(len(X)),X))
        self.Y = Y

        # Useful quantities
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        # Parameters
        self.W = np.zeros((self.p, 1))

        # Hyper-parameters
        self.alpha = alpha

        self.predictions = 0

    def compute_weights(self):

        for j in range(0, 500):
            for k in range(0, self.p):

                z = np.sum(np.power(self.X[:, k], 2))

                X_modified = np.copy(self.X)
                X_modified[:, k] = 0
                diff = -np.dot(X_modified, self.W) + self.Y
                rho = np.dot(self.X[:, k], diff)

                if rho < -self.alpha/2:
                    self.W[k] = (rho + self.alpha/2)/z
                elif rho > self.alpha:
                    self.W[k] = (rho - self.alpha/2)/z
                else:
                    self.W[k] = 0

    def predict(self):

        self.compute_weights()
        self.predictions = np.matmul(self.X, self.W)
