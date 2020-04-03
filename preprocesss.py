import pandas as pd
import numpy as np
import seaborn as sns

class PreProcess(object):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def process(self):
        # print(self.data.head())
        print(self.data.values)
        tmp = self.data.drop(['countyFIPS'], axis=1)
        tmp = tmp.drop(['stateFIPS'], axis=1)  # County Name
        # for item in tmp.values:
        #     deaths = item[2:]
        #     sns.relplot(x="timepoint", y="signal", col="region",
        #                 hue="event", style="event",
        #                 kind="line", data=fmri);
        #     print(item)


    def getData(self):
        return self.data


    def visualization(self):
        pass


if __name__ == '__main__':
    path = './data/us/covid/deaths.csv'
    preprocess = PreProcess(path)
    preprocess.process()

