import numpy as np

class RidgeReg:
    """
    Regression weights are obtained by minimizing RSS + L2_norm(weights)

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

        # Parameters
        self.W = 0

        # Hyper-parameters
        self.alpha = alpha

        self.predictions = 0

    def ridge_regression(self):
        """ Takes in as inputs matrix X and the vector Y along with a positive
            real number alpha and outputs the regression coeffiecients W."""

        self.W = np.dot(np.linalg.inv(np.dot(self.X.T, self.X) + self.alpha*np.identity(self.X.shape[1])), np.dot(self.X.T, self.Y))


    def predict(self):
        """This function first calculates the weights of regularized linear
           regression with the help of the above function and then outputs
           the predicitions y_hat for all x_hat."""

        self.ridge_regression()
        self.predictions = np.matmul(self.X, self.W)
