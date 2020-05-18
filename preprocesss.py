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


def printProportion(deaths, confirmeds):
    res = np.ndarray(len(deaths))
    for i in range(len(deaths)):
        d = deaths[i]
        c = confirmeds[i]
        if c == 0:
            continue
        else:
            res[i] = 1.0 * d/c
    return res


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
        # death = self.deathData.drop(['County Name'], axis=1)
        death = self.deathData  # County Name
        # cumulative_death = death['4/3/20']
        # plt.hist(cumulative_death)
        # plt.title('cumulative # of deaths until 4.2')

        # confirmed = self.confirmedData.drop(['County Name'], axis=1)
        # confirmed = self.confirmedData  # County Name
        # cumulative_confirmed = confirmed['4/3/20']
        cumulative_proportion1 = []
        cumulative_proportion2 = []

        for d in death.values:
            countyFIPS = d[0]
            print(countyFIPS)
            deaths = d[4:]
            c = self.confirmedData.loc[self.confirmedData['countyFIPS'] == countyFIPS]
            if c.empty:
                continue
            confirmeds = c.values[0][4:]
            X = range(0, len(deaths))
            if deaths[-1] >= 50:
                # fig1 = plt.gcf()
                # plt.plot(X, deaths, X, confirmeds)
                # plt.legend(['death', 'confirmed'])
                # plt.title(countyName)
                # plt.show()
                cumulative_proportion1.append(1.0*deaths[-1]/confirmeds[-1] if confirmeds[-1]!=0 else 0)
                # fig1.savefig('./img/4.4_death_and_confirmed/%s.png' % countyName, dpi=400)
                # print(printProportion(deaths, confirmeds))
            elif deaths[-1] > 0:
                cumulative_proportion2.append(1.0 * deaths[-1] / confirmeds[-1] if confirmeds[-1]!=0 else 0)

        print(cumulative_proportion1)
        print(cumulative_proportion2)
        print('std of > 50 death is %s' % np.std(cumulative_proportion1))
        print('mean of > 50 death is %s' % np.mean(cumulative_proportion1))

        print('std of all county is %s' % np.std(cumulative_proportion2))
        print('mean of all county is %s' % np.mean(cumulative_proportion2))

    def exponentialFit(self):
        data = self.getNoneZeroData()
        res = []
        for item in data.values:
            countyFIPS = item[0]
            deaths = item[4:]

            startIdx = findFirstNonZero(deaths)
            x_data = np.array(deaths[startIdx:])
            print(x_data)
            if len(x_data) <= 2 or x_data[0] == x_data[-1] or any(pd.DataFrame(x_data).diff(axis=0).values < 0):
                continue
            y_data = np.array(range(0, len(x_data)))
            y_predict = np.array(range(len(x_data), len(x_data) + 14))

            log_x_data = np.log(list(x_data))
            # if len(log_x_data) <= 1:
            #     continue
            curve_fit = np.polyfit(log_x_data, y_data, 1)
            a, b = curve_fit[0], curve_fit[1]

            x_predict = np.exp((y_predict-b)/a)
            res.append(np.asarray([countyFIPS] + [int(round(tmp)) for tmp in x_predict]))
        pd.DataFrame(res).to_csv('exponential_output.csv', index=False)

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
        countyZero.reset_index(drop=True)

        countyNoneZero.to_csv('./processed_data/death_nonzero.csv', index=False)
        countyZero.to_csv('./processed_data/death_zero.csv', index=False)

    def getNoneZeroData(self):
        last_date = self.deathData.columns[-1]
        countyNoneZero = self.deathData.loc[self.deathData[last_date] != 0]
        countyNoneZero = countyNoneZero.reset_index(drop=True)
        return countyNoneZero

    def generate_for_time_series(self):
        data = self.getNoneZeroData()
        list_to_append = []
        for item in data.values:
            countyFIPS = item[0]
            deaths = item[4:]
            startIdx = findFirstNonZero(deaths)
            x_data = np.array(deaths[startIdx:])
            list_to_append.append(x_data)
        data['death_list'] = list_to_append
        return data[['countyFIPS', 'death_list']]

    def generate_for_time_series_deleteme(self):
        data = self.getNoneZeroData()
        list_to_append = []
        random = []
        for item in data.values:
            countyFIPS = item[0]
            deaths = item[4:]
            startIdx = findFirstNonZero(deaths)
            x_data = np.array(deaths[startIdx:])
            list_to_append.append(x_data)
            random.append(countyFIPS % 4)
        data['death_list'] = list_to_append
        data['class'] = random
        return data[['countyFIPS', 'death_list', 'class']]



if __name__ == '__main__':
    path = './data/us/covid/'
    preprocess = PreProcess(path)
    # preprocess.process()
    # preprocess.visualization()

    # for check and visualization
    # preprocess.checkExponentialFit()
    # preprocess.exponentialFit()
    # preprocess.getNdarray()
    # tmp = preprocess.getConfirmedDataNdarray()
    # print(tmp.head())

    # 4.16 time series
    preprocess.generate_for_time_series()
