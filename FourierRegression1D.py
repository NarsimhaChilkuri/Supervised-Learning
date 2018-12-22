import numpy as np

class FourierReg:
    """
    Regression using Fourier Basis in 1D. Using a naive fourier transform rather than DFT.

    Args:

        X : numpy array of dimension Nxp, where N is the number of trainig examples
            and p is the number of predictors.
        Y : numpy array of dimension Nx1, where N is the number of training examples.
        k : inger greater than 0; this defines the number of sine
            and cosine pairs we use (sum_0^k sin(..) + cos(...)).

    """

    def __init__(self, X, Y, k):

        self.X = X
        self.Y = Y

        # Useful quantities
        self.X_fbasis = 0
        self.L = np.ceil(np.max(X, axis=0)[0] - np.min(X, axis=0)[0]) + 1

        # Parameters
        self.W = 0

        # Hyper-parameters
        self.k = k

        self.predictions = 0


    def fbasis_matrix(self):

        fbasis = []
        for x in self.X:
            row = []
            for k in range(0, self.k + 1):
                if k == 0:
                    row.append(1)
                else:
                    row.append(np.sin(2*np.pi*k*x/self.L))
                    row.append(np.cos(2*np.pi*k*x/self.L))
            fbasis.append(row)

        self.X_fbasis =  np.array(fbasis)


    def fourier_reg(self):

        # Computing the matrix that has the fourier basis applied to inputs.
        self.fbasis_matrix()

        # Computing the fourier coeffiecients using the above matrix
        self.W = np.matmul(np.linalg.inv(np.matmul(self.X_fbasis.T, self.X_fbasis)), np.matmul(self.X_fbasis.T, self.Y))

    def predict(self):
        self.predictions = np.matmul(self.X_fbasis, self.W)

    def error(self, string):

        if string == "RMS":
            return np.asscalar(np.mean((self.Y - self.predictions)**2, axis=0))
        elif string == "R2":
            RSS = np.sum((self.Y - self.predictions)**2, axis=0)
            TSS = np.sum((self.Y - np.mean(self.Y, axis=0))**2, axis=0)
            return np.asscalar(1 - RSS/TSS)
        else:
            print("Please choose from RMS and R2.")
