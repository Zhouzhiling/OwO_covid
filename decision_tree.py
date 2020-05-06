import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from preprocessForNN import PreprocessForNN


class DecisionTree(object):

    def __init__(self):
        self.clf = DecisionTreeRegressor()
        self.preprocess = PreprocessForNN()
        self.mode = 'outbreak'
        pass

    def train(self):
        feature, labels = self.preprocess.generate_training_data(mode=self.mode)

        for day in range(14):
            label = np.array([labels[k][0] for k in range(len(labels))])
            print(label)
            self.clf.fit(feature, label)
            acc = round(self.clf.score(feature, label) * 100, 2)
            print('Day: %d acc: %f' % (day, acc))

    def test(self):
        feature, FIPS = self.preprocess.generate_testing_data(mode=self.mode)
        pre = self.clf.predict(feature)
        print(pre)

        # probability = pd.Series([i[0] for i in probability], name='DT_probability').to_frame()
        # probability.to_csv(path_or_buf='probability/DT_probability.csv', index=False)

        # prediction = pd.Series((i for i in pre), name='Survived')
        # submission = pd.concat([FIPS, prediction], axis=1)
        # submission.to_csv(path_or_buf='data/DT_submission.csv', index=False)


if __name__ == '__main__':
    ti = DecisionTree()
    ti.train()
    ti.test()
