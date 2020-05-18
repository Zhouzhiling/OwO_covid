import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from preprocessForNN import PreprocessForNN
import output


class DecisionTree(object):

    def __init__(self):
        self.clf = []
        for i in range(14):
            self.clf.append(DecisionTreeRegressor())
        self.preprocess = PreprocessForNN()
        self.mode = 'outbreak'
        pass

    def train(self):
        feature, labels = self.preprocess.generate_training_data(mode=self.mode)

        for day in range(14):
            label = np.array([labels[k][0] for k in range(len(labels))])
            print(label)
            self.clf[day].fit(feature, label)
            acc = round(self.clf[day].score(feature, label) * 100, 2)
            print('Day: %d acc: %f' % (day, acc))

    def test(self):
        feature, FIPS = self.preprocess.generate_testing_data(mode=self.mode)
        predictions = []
        for day in range(14):
            prediction = self.clf[day].predict(feature)
            std = self.preprocess.get_std()
            average = self.preprocess.get_average()
            prediction = np.round(prediction * std[day] + average[day])
            predictions.append(prediction)

        predictions = np.array(predictions)
        predictions = np.reshape(predictions, (len(predictions[0]), len(predictions)))

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

        result.to_csv('models/DT/dt_outbreak.csv', index=False)
    #
    # def generate_output(self):
    #     source = 'models/DT/dt.csv'
    #     dst = 'submissions/dt.csv'
    #     Output = output.Output()
    #     Output.save_submission(source, dst)


if __name__ == '__main__':
    ti = DecisionTree()
    ti.train()
    ti.test()
    # ti.generate_output()