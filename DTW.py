import numpy as np
import preprocesss
import pandas as pd
from dtw import *
from collections import defaultdict


class DTW(object):

    def __init__(self):
        self.distances = defaultdict(float)
        self.average = 0.0
        self.std = 0.0
        self.threshold = 0.0
        self.county_numbers = 0
        self.clusters = defaultdict(list)
        self.data = None

    def load_data(self):
        processor = preprocesss.PreProcess('./data/us/covid/')
        self.data = processor.generate_for_time_series()

    def calculate_distance(self):
        self.load_data()
        labels = pd.Series(data=self.data['countyFIPS'], name='label')
        self.data = pd.concat([self.data, labels], axis=1)

        self.county_numbers = len(self.data)

        for i in range(self.county_numbers):
            for j in range(i + 1, self.county_numbers):
                county_i = self.data['death_list'][i]
                county_j = self.data['death_list'][j]

                series_length = min(len(county_i), len(county_j))
                self.distances[(i, j)] = dtw(county_i[:series_length], county_j[:series_length]).distance
                print('Distance between county %d and %d is %d.\n' % (i, j, self.distances[(i, j)]))

    def calculate_statistics(self):
        distances = np.fromiter(self.distances.values(), dtype=float)
        self.average = np.mean(distances)
        self.std = np.std(distances)
        # self.threshold = max(0.0, 2 * self.average / self.std)
        self.threshold = max(0.0, self.average)

    def compare_similarity(self):
        self.calculate_distance()
        self.calculate_statistics()

        label = 0
        for i in range(self.county_numbers):
            if not self.clusters:
                self.clusters[label].append(i)
                label += 1
            else:
                included = False
                for key, val in self.clusters.items():
                    should_include = True
                    for county in val:
                        if self.distances[(county, i)] > self.threshold:
                            should_include = False
                            break
                    if should_include:
                        self.clusters[key].append(i)
                        included = True
                        break
                if not included:
                    self.clusters[label].append(i)
                    label += 1

        print('There are %d clusters.' % label)

        for key, val in self.clusters.items():
            for county in val:
                self.data['label'][county] = key

    def output(self):
        self.data.to_pickle(path='processed_data/DTW/DTW_label_deaths.plk')

    def get_DTW_label_data(self):
        return self.data


if __name__ == '__main__':
    DTWClass = DTW()
    DTWClass.compare_similarity()
    DTWClass.output()
