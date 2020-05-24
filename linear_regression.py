from sklearn.linear_model import LinearRegression
from preprocessForNN import PreprocessForNN
import pandas as pd
import numpy as np


class LinearRegressor(object):

    def __init__(self):
        self.clfs = []
        self.preprocess = PreprocessForNN()

    def train(self):

        feature, label = self.preprocess.generate_training_data(mode='outbreak')
        print("Finish data process!")
        for i in range(14):
            clf = LinearRegression()
            clf.fit(feature[:, :14], label[:, i])

            acc = round(clf.score(feature[:, :14], label[:, i]) * 100, 2)

            print('Date: %d. Training acc: %f' % (i, acc))

            self.clfs.append(clf)

    def test(self):

        feature, FIPS, base = self.preprocess.generate_testing_data(mode='outbreak')

        predictions = []

        std = self.preprocess.get_std()
        average = self.preprocess.get_average()

        for i in range(14):
            pre = self.clfs[i].predict(feature[:, :14])
            pre = np.round(pre * std[i] + average[i])

            # for j in range(len(pre)):
            #     pre[j] = min(pre[j], round(base[j] * (1.5 - ((i + 5) % 7) / 7)))

            predictions.append(pre)

        predictions = np.array(predictions)
        predictions = np.transpose(predictions)

        prediction = pd.DataFrame(predictions, index=None)

        result = pd.concat([FIPS, prediction], axis=1, ignore_index=True)
        result = result.rename(
            columns={
                0: 'countyFIPS',
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                9: 8,
                10: 9,
                11: 10,
                12: 11,
                13: 12,
                14: 13
            }
        )

        result.to_csv('models/LR/lr_outbreak.csv', index=False)


if __name__ == '__main__':
    lr = LinearRegressor()
    lr.train()
    lr.test()
