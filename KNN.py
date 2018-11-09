import numpy as np

class KNN:
    """
    Regression or Classification using the K-Nearest Neighbours approach.

    Args:

        X : numpy array of dimension Nxp, where N is the number of trainig examples
                and p is the number of predictors.
        Y : numpy array of dimension Nx1, where N is the number of training examples.
        P : integer greater than zero. This defines the type of metric we use.
                P = 1 gives the manhattan metric and P = 2 gives the euclidean distance.
        K : integer greater than zero. This is the number of nearest neighbours that
                we use when generalizing.
        type_ : string, which is either "regression" or "classification"
    """

    def __init__(self, X, Y,  K, P, type_):

        self.X = X
        self.Y = Y
        self.type = type_

        # Hyper-parameters
        self.K = K
        self.P = P

        # Useful quantities
        self.N = self.X.shape[0]
        self.p = self.X.shape[1]
        self.neighbours = np.zeros((self.N, self.K))

        self.predictions = np.zeros((self.N, 1))

    def compute_distance(self, x_1, x_2):
        return np.power(np.sum(np.power(np.abs(x_1 - x_2), self.P)), 1/self.P)

    # Finding only the "first-found" K points, irrespective of the number
    # of points at the same distance.
    def find_nearest_points(self):
        for i,x_1 in enumerate(self.X):
            list_temp = []
            for j,x_2 in enumerate(self.X):
                list_temp.append(self.compute_distance(x_1, x_2))

            sorting_indices = np.argsort(list_temp)
            sorted_y_values = self.Y[sorting_indices]
            self.neighbours[i,:] = np.reshape(sorted_y_values[1:self.K+1], (self.K))

    def predict(self):
        self.find_nearest_points()
        self.predictions = np.mean(self.neighbours, axis=1)

        if self.type == "classification":
            # Breaking a tie randomly
            if np.random.randint(0, 2):
                truth_values = (self.predictions - np.round(self.predictions)) == 0.5
                self.predictions[truth_values] += 0.1

            self.predictions = np.round(self.predictions)
