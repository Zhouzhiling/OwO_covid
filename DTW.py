import numpy as np
from dtw import *


class DTW(object):

    def __init__(self):
        pass

    def load_data(self):
        idx = np.linspace(0, 6.28, num=100)
        query = np.sin(idx) + np.random.uniform(size=100) / 10.0
        template = np.cos(idx)
        return query, template

    def compare_similarity(self):
        query, template = self.load_data()

        alignment = dtw(query, template, keep_internals=True)


if __name__ == '__main__':
    DTWClass = DTW()
    DTWClass.compare_similarity()
    pass
