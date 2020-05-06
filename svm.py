from sklearn import svm
from preprocessForNN import PreprocessForNN
import pandas as pd
import numpy as np


class SVM(object):

    def __init__(self):
        self.clfs = []
        self.preprocess = PreprocessForNN()

    def train(self):

        feature, label = self.preprocess.generate_training_data()

        for i in range(14):
            clf = svm.SVR()
            clf.fit(feature, label[:, i])

            acc = round(clf.score(feature, label[:, i]) * 100, 2)

            print('Date: %d. Training acc: %f' % (i, acc))

            self.clfs.append(clf)

    def test(self):

        feature, FIPS = self.preprocess.generate_testing_data()

        predictions = []

        for i in range(14):
            pre = self.clfs[i].predict(feature)
            std = self.preprocess.get_std()
            average = self.preprocess.get_average()
            pre = np.round(pre * std[i] + average[i])
            predictions.append(pre)

        predictions = np.array(predictions)
        predictions = np.reshape(predictions, (len(predictions[0]), len(predictions)))

        prediction = pd.DataFrame(predictions, index=None)

        result = pd.concat([FIPS, prediction], axis=1, ignore_index=True)

        result.to_csv('models/SVM/svm.csv', index=False)


if __name__ == '__main__':
    s = SVM()
    s.train()
    s.test()
