import numpy as np
import gudhi as gd
import gudhi.representations as gdr

class clasSILfier():
    '''
    A topological classifier based on euclidean distances of silhouettes
    of zero dimensional homology
    '''
    def __init__(self, resolution = 1000, p = 1):
        self.silhouettes = []
        self.data = []
        self.classes = []
        self.resolution = resolution
        self.p = p
    
    def _get_persistence(self, block):
        Rips_complex_sample = gd.RipsComplex(block)
        Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=1)
        persistence = Rips_simplex_tree_sample.persistence()
        return [np.array(persistence[i][1]) for i in range(len(persistence)) if persistence[i][0] == 0 and persistence[i][1][1] != float('inf')]
        

    def _get_distances(self, sample):
        distances = []
        for i in range(len(self.classes)):
            mtv = self.data[i]
            pers = self._get_persistence(np.vstack([mtv, sample]))
            sil = gdr.Silhouette(weight = lambda x: np.power(x[1]-x[0], self.p), resolution=self.resolution)
            res = sil.fit_transform([np.array(pers)])[0]
            distances.append(np.sum(np.square(self.silhouettes[i] - res)))
        return distances
    
    def get_trained_silhouettes(self):
        return self.silhouettes
    
    def get_new_silhouettes(self, sample):
        distances = []
        sils = []
        for i in range(len(self.classes)):
            mtv = self.data[i]
            pers = self._get_persistence(np.vstack([mtv, sample]))
            sil = gdr.Silhouette(weight = lambda x: np.power(x[1]-x[0], self.p), resolution=self.resolution)
            res = sil.fit_transform([np.array(pers)])[0]
            distances.append(np.sum(np.square(self.silhouettes[i] - res)))
            sils.append(res)
        return sils, distances
        
    def fit(self, X, y):
        class_values = np.unique(y)
        self.classes = class_values
        for v in class_values:
            mtv = X[y == v, :]
            persistence = self._get_persistence(mtv)
            sil = gdr.Silhouette(weight = lambda x: np.power(x[1]-x[0], self.p), resolution=self.resolution)
            self.data.append(mtv)
            self.silhouettes.append(sil.fit_transform([np.array(persistence)])[0])

    def predict(self, X):
        y_hat = []
        for i in range(len(X)):
            sample = X[i, :]
            dist = self._get_distances(sample)
            y_hat.append(np.argmin(dist))
        return y_hat
        
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.sum(y_hat == y) / len(y_hat)