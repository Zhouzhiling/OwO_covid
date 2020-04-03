from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from preprocesss import PreProcess


class KMeansClassifier(object):

    def __init__(self, clusters_cnt):
        self.kmeans = KMeans(clusters_cnt)

    def load_data(self):
        preprocess = PreProcess('./data/us/covid/deaths.csv')
        data = preprocess.getData()
        death = data.iloc[:, 4:].diff(axis=1).iloc[:, 1:]
        FIPS = data['countyFIPS']

        # normalization
        min_max_scale = MinMaxScaler()
        death = min_max_scale.fit_transform(death.to_numpy().T).T

        return FIPS, death

    def train(self):
        FIPS, death = self.load_data()
        self.kmeans.fit(death)

        labels = pd.Series(data=self.kmeans.labels_, name='cluster')
        county_clusters = pd.concat([FIPS, labels], axis=1)
        county_clusters.to_csv(path_or_buf='processed_data/kmeans_label.csv', index=False)

    def test(self):
        pass


if __name__ == '__main__':
    clf = KMeansClassifier(2)
    clf.train()
