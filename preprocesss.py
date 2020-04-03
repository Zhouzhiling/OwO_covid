import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def findFirstNonZero(countyList):
    length = len(countyList)
    st = 0
    ed = len(countyList)-1
    while st < ed:
        mid = (st + ed) // 2
        if countyList[mid] == 0:
            st = mid + 1
        else:
            ed = mid
    return st


class PreProcess(object):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.process()

    def removeStatewideUnallocated(self):
        indexes = []
        for idx, item in enumerate(self.data.values):
            if item[0] == 0:
                indexes.append(idx)
        self.data = self.data.drop(indexes, axis=0)
        self.data = self.data.reset_index(drop=True)
        # self.data.reset_index()

    def process(self):
        self.removeStatewideUnallocated()

    def getData(self):
        return self.data

    def getNdarray(self):
        labels = self.data['countyFIPS']
        features = self.data.values[:, 4:]
        return features, np.array(labels)

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

    def checkExponentialFit(self):
        countyName = 'Orleans Parish'
        countyData = self.data.loc[self.data['County Name'] == 'Orleans Parish']
        countyList = countyData.values[0][5:]

        startIdx = findFirstNonZero(countyList)
        x_data = np.array(countyList[startIdx:])
        y_data = np.array(range(0, len(x_data)))

        log_x_data = np.log(list(x_data))

        curve_fit = np.polyfit(log_x_data, y_data, 1)
        print(curve_fit)

        y = curve_fit[0] * log_x_data + curve_fit[1]
        plt.plot(x_data, y_data, "o")
        plt.plot(x_data, y)
        plt.title(countyName)
        plt.ylabel('date')
        plt.xlabel('death count')
        plt.show()

    def storeNoneZeroData(self):
        last_date = self.data.columns[-1]
        countyNoneZero = self.data.loc[self.data[last_date] != 0]
        countyZero = self.data.loc[self.data[last_date] == 0]
        countyNoneZero.reset_index(drop=True)
        # countyZero.reset_index()
        # countyNoneZero.drop('index')

        countyZero.reset_index(drop=True)
        # countyZero.drop('index')

        countyNoneZero.to_csv('./processed_data/death_nonzero.csv', index=False)
        countyZero.to_csv('./processed_data/death_zero.csv', index=False)

    def getNoneZeroData(self):
        last_date = self.data.columns[-1]
        countyNoneZero = self.data.loc[self.data[last_date] != 0]
        countyNoneZero = countyNoneZero.reset_index(drop=True)
        return countyNoneZero


if __name__ == '__main__':
    path = './data/us/covid/deaths.csv'
    preprocess = PreProcess(path)
    # preprocess.process()
    # preprocess.visualization()
    # preprocess.checkExponentialFit()
    # preprocess.getNdarray()
    tmp = preprocess.getNoneZeroData()
    print(tmp.head())
