import pandas as pd


class Output(object):

    def __init__(self):
        self.sample = self.read_sample()
        pass

    @staticmethod
    def read_sample():
        return pd.read_csv('sample_submission.csv')

    def modify_submission(self):
        pass

    def save_submission(self):
        pass


if __name__ == '__main__':
    output = Output()
    output.save_submission()
