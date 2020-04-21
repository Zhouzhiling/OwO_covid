import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

import preprocesss
import output as op


class Regression(object):
    def __init__(self):
        self.data = None
        self.window_size = 5
        self.class_label = None     # set with unique labels
        self.model_selection_threshold = 0
        # first item is stationary, second is non-stationary
        self.coeff_dict = defaultdict(lambda: [0, 0])
        self.intercept_dict = defaultdict(lambda: [0, 0])
        self.result = None

    def load_data(self, path='processed_data/Stationary/stationary_label_deaths.plk'):
        self.data = pd.read_pickle(path)
        self.class_label = set(self.data['label'].values)

    def generate_input(self, deaths):
        length = len(deaths)
        X, Y = [], []
        for i in range(length - self.window_size):
            X.append(deaths[i:i + self.window_size])
            Y.append(deaths[i + self.window_size])

        return np.asarray(X), np.asarray(Y).reshape(-1, 1)

    @staticmethod
    def train_specific_model(X, Y, model_type='LinearRegression'):
        if model_type == 'LinearRegression':
            reg = LinearRegression().fit(X, Y)
        elif model_type == 'Lasso':
            reg = Lasso().fit(X, Y)
        else:
            reg = Ridge().fit(X, Y)

        score = r2_score(Y, reg.predict(X))

        return reg.coef_, reg.intercept_, score
        # cur_coeff = reg.coef_
        # cur_score = reg.score(X, Y)

    def get_seed(self, input):
        result, FIPS = [], []
        for i, row in input.iterrows():
            death = row['death_list']
            index = row['countyFIPS']
            if len(death) < self.window_size:
                result.append(np.concatenate((np.zeros(self.window_size - len(death)), death[:])))
            else:
                result.append(np.asarray(death[-self.window_size:]))
            FIPS.append(index)
        return np.asarray(result), np.asarray(FIPS)

    def train(self, model_type):
        # tmp load data
        for label in self.class_label:
            print("Training for label %d ..." % label)
            coeff_list, intercept_list = [[], []], [[], []]
            item_info = self.data.loc[self.data['label'] == label]

            # go through all data
            for idx in range(len(item_info)):
                stationary = item_info['stationary'].values[idx]

                # check whether current item is stationary or not
                if stationary:
                    data = item_info['death_list'].values[idx]
                else:
                    data = item_info['diff'].values[idx]

                if len(data) <= self.window_size:
                    continue

                X, Y = self.generate_input(data)
                coeff, intercept, score = self.train_specific_model(X, Y, model_type)

                # only keep the model with high score
                if score > self.model_selection_threshold:
                    coeff_list[stationary].append(coeff)
                    intercept_list[stationary].append(intercept)

            self.store_into_dict(label, coeff_list, intercept_list)

    def store_into_dict(self, label, coeff_list, intercept_list):
        for stationary in range(2):
            coeff_val, intercept_val = np.asarray(coeff_list[stationary]), np.asarray(intercept_list[stationary])
            self.coeff_dict[label][stationary] = coeff_val.mean(axis=0)
            self.intercept_dict[label][stationary] = intercept_val.mean(axis=0)

    def predict(self, predict_days=100):
        for label in self.class_label:
            print("Predicting for label %d ..." % label)

            # coeff and intercept ars list with 2 values
            coeff = self.coeff_dict[label]
            intercept = self.intercept_dict[label]
            data_this_label = self.data.loc[self.data['label'] == label]

            station_data = data_this_label.loc[data_this_label['stationary'] == True]
            non_station_data = data_this_label.loc[data_this_label['stationary'] == False]

            deaths = [station_data, non_station_data]

            for i in range(2):
                if deaths[i].empty:
                    continue
                X, FIPS = self.get_seed(deaths[i])
                predicted_list = np.zeros((len(X), predict_days))

                for day in range(predict_days):
                    predicted = np.sum(X * coeff[i], axis=1) + intercept[i]
                    predicted_list[:, day] = predicted
                    X = np.concatenate((np.delete(X, 0, axis=1), predicted.reshape(-1, 1)), axis=1)

                item = pd.concat([pd.Series(data=FIPS, name='countyFIPS'), pd.DataFrame(data=predicted_list)], axis=1)
                self.result = pd.concat([self.result, pd.DataFrame(item)])

        self.result = self.result.reset_index()

    def generate_output(self, path):
        result = None
        diff_value = pd.read_pickle(path)
        for i in range(len(self.result)):
            FIPS = self.result['countyFIPS'][i]
            increases = self.result.loc[self.result['countyFIPS'] == FIPS].values[0][2:]
            station_info = diff_value.loc[diff_value['countyFIPS'] == FIPS]

            item = self.calculate_integral(FIPS, increases, station_info)

            result = pd.concat([result, pd.DataFrame(item)])

        output = op.Output()
        output.save_submission(result)

    def calculate_integral(self, FIPS, increases, station_info):
        stationary = station_info['stationary'].values[0]
        origin_one = station_info['origin_diff_one'].values[0]
        origin = station_info['origin'].values[0]

        predicted_list = self.integral(stationary, increases, origin_one, origin)
        item = pd.concat([pd.Series(data=FIPS, name='countyFIPS'), pd.DataFrame(data=predicted_list)], axis=1)
        return item

    @staticmethod
    def integral(stationary, increases, origin_one, origin):
        output = [0 for _ in range(len(increases) + 2)]
        if not stationary:
            output[0] = origin
            output[1] = origin_one
            for i, n in enumerate(increases):
                output[i+2] = output[i+1] + increases[i]
                output[i+1] = output[i] + output[i+1]
        else:
            for i, n in enumerate(increases):
                output[i+2] = increases[i]

        return np.asarray(output[2:]).reshape((1, -1))

    @staticmethod
    def save_output(data, path):
        data.to_csv(path, index=False)


if __name__ == "__main__":
    regression = Regression()
    regression.load_data()
    regression.train('LinearRegression')
    regression.predict(100)
    regression.generate_output('processed_data/Stationary/stationary_label_deaths.plk')
