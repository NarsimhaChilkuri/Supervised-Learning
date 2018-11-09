import numpy as np

class DecisionTree:
    """
    Classification and Regression using decision trees.

    Args:

        X : numpy array of dimension Nxp, where N is the number of trainig examples
            and p is the number of predictors.
        Y : numpy array of dimension Nx1, where N is the number of training examples.
        stop : int > 0, number of regions to divide the predictor space into.
        type : string, either "regression" or "classification"               
    """

    def __init__(self, X, Y, stop, type_):
        self.X = X
        self.Y = Y

        # Useful quantities
        self.N = self.X.shape[0]
        self.p = self.X.shape[1]
        self.L = np.min(self.X, axis=0).tolist()
        self.H = np.max(self.X, axis=0).tolist()
        self.regions = [ self.L + self.H ]
        self.rss_regions = {}
        self.means = {}
        self.type = type_

        # Hyper-parameter
        self.stop = stop

        self.predictions = np.zeros((self.N, 1))

    def build_tree(self):

        for n in range(0, self.stop):
            current_best = (0, 0, np.inf, 0)
            for i in range(0, self.p):

                for region in self.regions:
                    lower = region[0:self.p]
                    upper = region[self.p:]
                    truth_val_region = (self.X >= lower).all(axis=1) * (self.X <= upper).all(axis=1)
                    X_region = self.X[truth_val_region]
                    Y_region = self.Y[truth_val_region]

                    ind = np.argsort(X_region[:,i])
                    X_region_sorted = X_region[ind]
                    Y_region_sorted = Y_region[ind]

                    for j, s in enumerate(X_region_sorted):
                        mean1 = np.asscalar(np.mean(Y_region_sorted[0:j]))
                        mean2 = np.asscalar(np.mean(Y_region_sorted[j:]))
                        rss1 = np.sum((Y_region_sorted[0:j] - mean1)**2)
                        rss2 = np.sum((Y_region_sorted[j:] - mean2)**2)
                        rss = rss1 + rss2
                        for reg in self.rss_regions.keys():
                            if reg != str(region):
                                rss += self.rss_regions[reg]

                        if rss < current_best[2]:
                            current_best = (i, s[i], rss, region, (rss1, rss2))

            best_X, best_val, best_reg, best_rss_region = current_best[0], current_best[1], current_best[3], current_best[4]

            new_reg1 = best_reg[:]
            new_reg2 = best_reg[:]
            new_reg2[self.p + best_X] = best_val
            new_reg1[best_X] = best_val

            self.rss_regions[str(new_reg1)] = rss1
            self.rss_regions[str(new_reg2)] = rss2

            self.regions.remove(best_reg)
            self.regions.append(new_reg2)
            self.regions.append(new_reg1)


    def compute_means(self):
        for region in self.regions:
                lower = region[0:self.p]
                upper = region[self.p:]

                truth_val_region = (self.X >= lower).all(axis=1) * (self.X <= upper).all(axis=1)
                X_region = self.X[truth_val_region]
                Y_region = self.Y[truth_val_region]
                self.means[str(region)] = np.mean(Y_region)
                self.predictions[truth_val_region] = self.means[str(region)]



    def predict(self):
        self.build_tree()
        self.compute_means()

        if self.type == "classification":
            if np.random.randint(0, 2):
                truth_values = (self.predictions - np.round(self.predictions)) == 0.5
                self.predictions[truth_values] += 0.1

            self.predictions = np.round(self.predictions)
