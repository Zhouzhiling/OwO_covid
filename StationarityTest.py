from statsmodels.tsa.stattools import adfuller
import regression_model
import numpy as np
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

    def differentiate(self):
        # diffs: second order difference
        # origin: death on first day
        # origin_diff_one: first order different on first day
        diffs = []
        origin = []
        origin_diff_one = []
        for i in range(self.county_number):
            if self.is_stationary[i]:
                diffs.append([])
                origin.append([])
                origin_diff_one.append([])
                continue
            death = self.data['death_list'][i]
            diff_one = [j - i for i, j in zip(death[: -1], death[1:])]
            diff_two = [j - i for i, j in zip(diff_one[: -1], diff_one[1:])]
            origin.append(death[0])
            origin_diff_one.append(diff_one[0])
            diffs.append(diff_two)

        self.data = pd.concat([self.data, pd.Series(data=diffs, name='diff'), pd.Series(data=origin, name='origin'),
                               pd.Series(data=origin_diff_one, name='origin_diff_one')], axis=1)

    def output(self):
        self.data.to_pickle(path='processed_data/Stationary/stationary_label_deaths.plk')

    def generate_output(self, path):
        result = None
        predicted = pd.read_csv(path)
        for i in range(self.county_number):
            FIPS = self.data['countyFIPS'][i]
            increases = self.data.loc[self.data['countyFIPS'] == FIPS].values[1:]
            origin_one = self.data.loc[self.data['countyFIPS'] == FIPS]['origin_diff_one'].values
            origin = self.data.loc[self.data['countyFIPS'] == FIPS]['origin'].values
            predicted_list = self.reverse(increases, origin_one, origin)

            result_array = pd.concat([pd.Series(data=FIPS, name='countyFIPS'), pd.DataFrame(data=predicted_list)],
                                     axis=1)
            if result is not None:
                result = pd.concat([result, pd.DataFrame(result_array)])
            else:
                result = pd.DataFrame(result_array)

        path = './processed_data/regression_submission.csv'
        self.save_output(result, path)


    def save_output(self, data, path):
        data.to_csv(path, index=False)


    def reverse(self, increases, origin_one, origin):
        output = np.zeros(1, len(increases)+2)
        output[0] = origin
        output[1] = origin_one
        for i, n in enumerate(increases):
            output[i+2] = output[i+1] + increases[i]
            output[i+1] = output[i] + output[i+1]
        return output







if __name__ == '__main__':
    stationarity = StationarityTest()
    stationarity.test()
    stationarity.differentiate()
    stationarity.output()

    # regression = regression_model.Regression()
    # regression.load_data()
    # regression.process_data()
    # regression.train('LinearRegression')
    # path = regression.predict(100)

    path = './processed_data/regression_predictions.csv'
    # stationarity.generate_output(path)

    pass
