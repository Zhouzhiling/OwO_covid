import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def findFirstNonZero(countyList):
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
        self.deathData = pd.read_csv(path + 'deaths.csv')
        self.confirmedData = pd.read_csv(path + 'confirmed_cases.csv')
        self.process()

    def removeStatewideUnallocated(self):
        indexes = []
        for idx, item in enumerate(self.deathData.values):
            if item[0] == 0:
                indexes.append(idx)
        self.deathData = self.deathData.drop(indexes, axis=0)
        self.deathData = self.deathData.reset_index(drop=True)

        indexes = []
        for idx, item in enumerate(self.confirmedData.values):
            if item[0] == 0:
                indexes.append(idx)
        self.confirmedData = self.confirmedData.drop(indexes, axis=0)
        self.confirmedData = self.confirmedData.reset_index(drop=True)

    def process(self):
        self.removeStatewideUnallocated()

    def getData(self):
        return self.deathData

    def getConfirmedData(self):
        return self.confirmedData

    def getNdarray(self):
        labels = self.deathData['countyFIPS']
        features = self.deathData.values[:, 4:]
        return features, np.array(labels)

    def getConfirmedDataNdarray(self):
        labels = self.confirmedData['countyFIPS']
        features = self.confirmedData.values[:, 4:]
        return features, np.array(labels)

    def visualization(self):
        death = self.deathData.drop(['countyFIPS'], axis=1)
        death = death.drop(['stateFIPS'], axis=1)  # County Name
        cumulative_death = death['4/3/20']
        # plt.hist(cumulative_death)
        # plt.title('cumulative # of deaths until 4.2')

        confirmed = self.confirmedData.drop(['countyFIPS'], axis=1)
        confirmed = confirmed.drop(['stateFIPS'], axis=1)  # County Name
        cumulative_confirmed = confirmed['4/3/20']

        for d, c in zip(death.values, confirmed.values):
            countyName = d[0]
            deaths = d[2:]
            confirmeds = c[2:]
            X = range(0, len(deaths))
            if deaths[-1] >= 50:
                fig1 = plt.gcf()
                plt.plot(X, deaths, X, confirmeds)
                plt.legend(['death', 'confirmed'])
                plt.title(countyName)
                plt.show()
                # plt.draw()
                fig1.savefig('./img/%s_4.2.png' % countyName, dpi=400)

    def checkExponentialFit(self):
        countyName = 'Orleans Parish'
        countyData = self.deathData.loc[self.deathData['County Name'] == 'Orleans Parish']
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
        last_date = self.deathData.columns[-1]
        countyNoneZero = self.deathData.loc[self.deathData[last_date] != 0]
        countyZero = self.deathData.loc[self.deathData[last_date] == 0]
        countyNoneZero.reset_index(drop=True)
        # countyZero.reset_index()
        # countyNoneZero.drop('index')

        countyZero.reset_index(drop=True)
        # countyZero.drop('index')

        countyNoneZero.to_csv('./processed_data/death_nonzero.csv', index=False)
        countyZero.to_csv('./processed_data/death_zero.csv', index=False)

    def getNoneZeroData(self):
        last_date = self.deathData.columns[-1]
        countyNoneZero = self.deathData.loc[self.deathData[last_date] != 0]
        countyNoneZero = countyNoneZero.reset_index(drop=True)
        return countyNoneZero


if __name__ == '__main__':
    path = './data/us/covid/'
    preprocess = PreProcess(path)
    # preprocess.process()
    preprocess.visualization()
    # preprocess.checkExponentialFit()
    # preprocess.getNdarray()
    # tmp = preprocess.getConfirmedDataNdarray()
    # print(tmp.head())
