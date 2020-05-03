from sklearn import svm
from preprocessForNN import PreprocessForNN


class SVM(object):

    def __init__(self):
        self.clf = svm.SVC(probability=True)
        self.preprocess = PreprocessForNN()

    def train(self):

        feature, label = self.preprocess.generate_training_data()

        self.clf.fit(feature, label)

        acc = round(self.clf.score(feature, label) * 100, 2)

        print('Training acc: %f' % acc)

    def test(self):

        feature, FIPS = self.preprocess.generate_testing_data()

        pre = self.clf.predict(feature)

        # todo save results

        return pre


if __name__ == '__main__':
    svm = SVM()
    svm.train()
    svm.test()
