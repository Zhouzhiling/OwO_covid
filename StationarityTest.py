from statsmodels.tsa.stattools import adfuller
import pandas as pd
import math


class StationarityTest(object):

    def __init__(self):
        self.data = None
        self.adfs = []
        self.county_number = 0
        self.significance = 0.05
        self.is_stationary = []
        pass

    def load_data(self):
        self.data = pd.read_pickle('processed_data/DTW/DTW_label_deaths.plk')
        self.county_number = len(self.data)

    def test(self):
        self.load_data()

        for i in range(self.county_number):
            if len(self.data['death_list'][i]) <= 3:
                self.is_stationary.append(True)
                continue
            adf = adfuller(self.data['death_list'][i], autolag='AIC')
            if math.isnan(adf[0]):
                self.is_stationary.append(True)
            elif adf[0] < min(adf[4].values()):
                self.is_stationary.append(True)
            else:
                self.is_stationary.append(False)
            self.adfs.append(adf)

        self.data = pd.concat([self.data, pd.Series(data=self.is_stationary, name='stationary')], axis=1)

    def output(self):
        self.data.to_pickle(path='processed_data/Stationary/stationary_label_deaths.plk')


if __name__ == '__main__':
    stationarity = StationarityTest()
    stationarity.test()
    pass
