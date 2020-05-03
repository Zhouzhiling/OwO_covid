from sklearn import svm
from preprocessForNN import PreprocessForNN


class SVM(object):

    def __init__(self):
        self.clf = svm.SVC(probability=True)

    @staticmethod
    def load_data():

        preprocess = PreprocessForNN()

        return preprocess.generate_training_data()

    def train(self):

        feature, label = self.load_data()

        self.clf.fit(feature, label)

        acc = round(self.clf.score(feature, label) * 100, 2)

        print('Training acc: %f' % acc)

    def test(self):

        feature = []

        pre = self.clf.predict(feature)

        return pre


if __name__ == '__main__':
    svm = SVM()
    svm.train()
    svm.test()
