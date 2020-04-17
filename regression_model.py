import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import preprocesss


class Regression(object):
    def __init__(self):
        self.data = None
        self.window_size = 5
        self.class_label = None
        self.model_selection_threshold = 0.5
        self.coeff_dict = dict()
        self.intercept_dict = dict()

    def load_data(self):
        path = './data/us/covid/'
        preprocess = preprocesss.PreProcess(path)
        self.data = preprocess.generate_for_time_series_deleteme()

    def process_data(self):
        # self.data['class']
        self.class_label = set(self.data['class'].values)

    def generate_input(self, deaths):
        length = len(deaths)
        X, Y = [], []
        for i in range(length - self.window_size):
            X.append(deaths[i:i + self.window_size])
            Y.append(deaths[i + self.window_size])

        return np.asarray(X), np.asarray(Y).reshape(-1, 1)

    def train_specific_model(self, X, Y, model_type='LinearRegression'):
        if model_type == 'LinearRegression':
            reg = LinearRegression().fit(X, Y)
        elif model_type == 'Lasso':
            reg = Lasso().fit(X, Y)
        else:
            reg = Ridge().fit(X, Y)
        return reg.coef_, reg.intercept_, reg.score(X, Y)
        # cur_coeff = reg.coef_
        # cur_score = reg.score(X, Y)

    def get_initial_data(self, input):
        result, FIPS = [], []
        for i, row in input.iterrows():
            death = row['death_list']
            index = row['countyFIPS']
            if len(death) <= self.window_size:
                continue
            result.append(np.asarray(death[-self.window_size:]))
            FIPS.append(index)
        return np.asarray(result), np.asarray(FIPS)

    def train(self, model_type):
        # tmp load data
        for label in self.class_label:
            print("Training for label %d ..." % label)
            coeff_list, intercept_list = [], []
            deaths = self.data.loc[self.data['class'] == label]['death_list'].values
            for idx in range(len(deaths)):
                data = deaths[idx]
                if len(data) <= self.window_size:
                    continue

                X, Y = self.generate_input(data)
                coeff, intercept, score = self.train_specific_model(X, Y, model_type)
                if score > self.model_selection_threshold:
                    coeff_list.append(coeff)
                    intercept_list.append(intercept)
            coeff_list, intercept_list = np.asarray(coeff_list), np.asarray(intercept_list)
            self.coeff_dict[label] = coeff_list.mean(axis=0)
            self.intercept_dict[label] = intercept_list.mean(axis=0)

    def predict(self, predict_days = 100):
        # for index, row in self.data.iterrows():
        #     predicted = []
        #     coeff = self.coeff_dict[row['class']]
        #     initial_data = self.get_initial_data(row['death_list'])
        #     for i in range(predict_days):
        #         sum(coeff[0] * initial_data)

        result = None
        for label in self.class_label:
            print("Predicting for label %d ..." % label)
            coeff = self.coeff_dict[label]
            intercept = self.intercept_dict[label]
            deaths = self.data.loc[self.data['class'] == label]
            X, FIPS = self.get_initial_data(deaths)
            predicted_list = np.zeros((len(X), predict_days))
            for day in range(predict_days):
                predicted = np.sum(X * coeff, axis=1) + intercept
                predicted_list[:, day] = predicted
                X = np.concatenate((np.delete(X, 0, axis=1), predicted.reshape(-1, 1)), axis=1)

            result_array = np.concatenate((FIPS.reshape(-1,1), predicted_list), axis=1)
            if result is not None:
                result = pd.concat([result, pd.DataFrame(result_array)])
            else:
                result = pd.DataFrame(result_array)
        return result





if __name__ == "__main__":
    regression = Regression()
    regression.load_data()
    regression.process_data()
    regression.train('LinearRegression')
    regression.predict(100)
