from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


class KMeansClassifier(object):

    def __init__(self, clusters_cnt):
        self.clusters_cnt = clusters_cnt
        self.kmeans = KMeans()
        pass

    def load_data(self):
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        return X

    def train(self):
        X = self.load_data()
        self.kmeans.fit(X)

        labels = pd.Series(data=self.kmeans.labels_, name='cluster')
        labels.to_csv('processed_data/kmeans_label.csv')
        pass

    def test(self):
        pass


if __name__ == '__main__':
    clf = KMeansClassifier(2)
