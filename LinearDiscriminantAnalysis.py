import numpy as np

class LDA:

    """
    Classification (multi-class) usign linear discriminant analysis.

    Args:

        X : numpy array of dimension Nxp, where N is the number of trainig examples
            and p is the number of predictors.
        Y : numpy array of dimension Nx1, where N is the number of training examples.

    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        # Useful quantities
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.mean = {}
        self.counter = {}
        self.covar = {}
        self.pi = {}
        self.X_sorted = sorted(self.X.tolist(), key=lambda x: self.Y[self.X.tolist().index(x)])
        self.X_sorted = np.array(self.X_sorted)

        self.predictions = 0

    def estimate_mean_covar_prior(self):

        self.classes = sorted(list(set(self.Y)))
        self.K = len(self.classes)
        self.Sigma = np.zeros((self.p, self.p))

        for i in self.classes:
            self.counter[i] = 0
            self.mean[i] = 0

        for y in self.Y:
            self.counter[y] += 1

        f = 0
        for y in self.classes:
            m = self.X_sorted[f: f+ self.counter[y]]
            self.mean[y] = np.mean(m, axis=0)
            self.covar[y] = np.cov(m.T)
            self.pi[y] = self.counter[y] / self.X.shape[0]
            f = f + self.counter[y]

        for y in self.classes:
            self.Sigma += self.covar[y]
        self.Sigma = self.Sigma * 1/(self.n - self.K)

    def predict_one(self, x):

        probs = []
        for y in self.classes:
            delta = np.matmul(x.T, np.matmul(np.linalg.inv(self.Sigma), self.mean[y])) -\
                    1/2 * np.matmul(self.mean[y].T, np.matmul(np.linalg.inv(self.Sigma), self.mean[y])) + \
                    np.log(self.pi[y])
            probs.append(delta)
        return self.classes[probs.index(max(probs))]

    def predict(self):

        self.estimate_mean_covar_prior()
        predictions = []
        for x in self.X:
            predictions.append(self.predict_one(x))
        self.predictions = np.array(predictions)
