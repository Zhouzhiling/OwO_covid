import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class PreProcess(object):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def removeStatewideUnallocated(self):
        indexes = []
        for idx, item in enumerate(self.data.values):
            if item[0] == 0:
                indexes.append(idx)
        self.data = self.data.drop(indexes, axis=0).reset_index()
        # self.data.reset_index()

    def process(self):
        self.removeStatewideUnallocated()

    def getData(self):
        return self.data

    def visualization(self):
        tmp = self.data.drop(['countyFIPS'], axis=1)
        tmp = tmp.drop(['stateFIPS'], axis=1)  # County Name
        count_zero = 0
        cumulative_death = tmp['4/2/20']
        plt.hist(cumulative_death)
        plt.title('cumulative # of deaths until 4.2')

        plt.show()
        for item in tmp.values:
            countyName = item[0]
            deaths = item[2:]
            if deaths[-1] >= 50:
                fig1 = plt.gcf()
                plt.scatter(range(0, len(deaths)), deaths)
                plt.title(countyName)
                plt.show()
                # plt.draw()
                fig1.savefig('./img/%s_4.2.png' % countyName, dpi=400)

    # def fitExponential(self):



if __name__ == '__main__':
    path = './data/us/covid/deaths.csv'
    preprocess = PreProcess(path)
    preprocess.process()
    preprocess.visualization()

