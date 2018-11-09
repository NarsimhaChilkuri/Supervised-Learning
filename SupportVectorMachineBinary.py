import numpy as np

class SVM:
    """
    Classification using the support vector machine approach.

    NOTE: This implementation has high variance -- try running
          the alogorithm a few times using the same hyperparameters.

    Args:

        X: numpy array of size m x p, 'm' being the number of samples and 'p' being the
           number of predictors.
        Y: numpy array of size m x 1. Elements of Y can be are either -1 or +1.
        C: integer, servers as a regularizing parameter
        tol: real number, serves as numerical tolerance
        max_passes: int > 0, affects the number of iterations of SMO.
        kernel: string - choose one from {lin, poly, exp}
        degree: int > 0 - degree of the polynomial        
        gamma: real > 0, constant of exponential kernel
    """

    def __init__(self, X, Y, C, tol, max_passes, kernel, degree, gamma):

        self.X = X
        self.Y = Y

        # Useful quantites
        self.m = self.X.shape[0]
        self.p = self.X.shape[1]

        # Parameters
        self.alpha = np.zeros((self.m, 1))
        self.b = 0

        # Hyper-parameters
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel = kernel
        self.d = degree
        self.g = gamma

        self.predictions = []


    def K(self, x_i, x_j):
        if self.kernel == "lin":
            return np.dot(x_i, x_j)
        elif self.kernel == "poly":
            return np.power(np.dot(x_i, x_j) + 1, self.d)
        elif self.kernel == "exp":
            return np.exp(-self.g * np.sum(np.square(x_i-x_j)))

    def K_bulk(self, x_i):

        x_reshaped = np.reshape(x_i, (self.p, 1))

        if self.kernel == "lin":
            return np.matmul(self.X, x_reshaped)
        elif self.kernel == "poly":
            return np.power(np.reshape(np.matmul(self.X, x_reshaped) + 0.5, (self.m, 1)), self.d)
        elif self.kernel == "exp":
            return np.exp(np.sum(np.square(self.X - x_reshaped.T), axis=1, keepdims=True) * -self.g)


    def compute_f(self, x):
        """
        Computes the classifier function for SVM using alphas.
        x: numpy array of shape (m,).
        """
        x_reshaped = np.reshape(x, (self.p, 1))
        return np.sum(np.multiply(np.multiply(self.alpha, self.Y), self.K_bulk(x_reshaped))) + self.b

    def compute_E(self, i):
        """
        Computes f(x_i) - y_i
        i: int in [0, m-1]
        """
        return self.compute_f(self.X[i,:]) - self.Y[i]

    def compute_L_H(self, i, j):
        """
        Computes lower and upper bounds for alphas to be used in SMO.
        i, j: integers in [0, m-1] and i != j
        """
        if self.Y[i] != self.Y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        elif self.Y[i] == self.Y[j]:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])

        return L,H

    def compute_eta(self, i, j):
        """
        Computes 2*K(x_i, x_j) -  K(x_i, x_i) - K(x_j, x_j)
        i, j: integers in [0, m-1] and i != j
        """
        return 2 * self.K(self.X[i,:], self.X[j,:]) - \
                   self.K(self.X[i,:], self.X[i,:]) - \
                   self.K(self.X[j,:], self.X[j,:])

    def clip_alpha(self, j, L, H):
        if self.alpha[j] > H:
            return H
        elif self.alpha[j] < L:
            return L
        else:
            return self.alpha[j]

    def compute_b(self, i, j, E_i, E_j, alpha_i_old, alpha_j_old):

        b_1 = self.b - E_i - self.Y[i] * (self.alpha[i] - alpha_i_old) * self.K(X[i,:], X[i,:]) - \
                            - self.Y[j] * (self.alpha[j] - alpha_j_old) * self.K(X[i,:], X[j,:])

        b_2 = self.b - E_j - self.Y[i] * (self.alpha[i] - alpha_i_old) * self.K(X[i,:], X[j,:]) - \
                            - self.Y[j] * (self.alpha[j] - alpha_j_old) * self.K(X[j,:], X[j,:])

        if self.alpha[i] > 0 and self.alpha[i] < self.C:
            return b_1
        elif self.alpha[j] > 0 and self.alpha[j] < self.C:
            return b_2
        else:
            return (b_1 + b_2) / 2

    def SMO(self):
        """
        This fucntion uses simplified SMO(sequential minimal optimization) to compute the
        parameters alphas of the dual SVM problem.
        """
        passes = 0
        while(passes < self.max_passes):
            num_changed_alphas = 0

            for i in range(0, self.m):
                E_i = self.compute_E(i)


                if((self.Y[i]*E_i < -self.tol and self.alpha[i] < self.C) or
                   (self.Y[i]*E_i > self.tol and self.alpha[i] > 0)):
                    j = i
                    while(j == i):
                        j = np.asscalar(np.random.randint(0, self.m, 1))
                    E_j = self.compute_E(j)
                    # storing current alpha values before they are updated
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    # Computing lower and upper bounds
                    L,H = self.compute_L_H(i, j)
                    eta = self.compute_eta(i, j)

                    if eta >= 0 or L == H:
                        continue

                    self.alpha[j] = self.alpha[j] - (self.Y[j] * ( E_i - E_j)) / eta
                    self.alpha[j] = self.clip_alpha(j, L, H)

                    if (abs(self.alpha[j] - alpha_j_old) < 1e-5):
                        continue

                    self.alpha[i] = self.alpha[i] + self.Y[i] * self.Y[j] (alpha_j_old - self.alpha[j])

                    self.b = self.compute_b(i, j, E_i, E_j, alpha_i_old, alpha_j_old)

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def predict(self):

        self.SMO()
        for x in self.X:
            self.predictions.append(self.compute_f(x))
        self.predictions = np.sign(self.predictions)
