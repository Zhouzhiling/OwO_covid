import pandas as pd


class Output(object):

    def __init__(self):
        self.sample = self.read_sample()

    @staticmethod
    def read_sample():
        return pd.read_csv('sample_submission.csv')

    def modify_submission(self):
        pass

    def format_key(self, key):
        strs = key.split('-')
        year = strs[0]
        month = strs[1][1]
        if strs[2][0] == '0':
            day = strs[2][1]
        else:
            day = strs[2]

        date = month + '/' + day + '/' + year

        FIPS = strs[3]

        return date, FIPS

    def save_submission(self):
        self.modify_submission()
        pass


if __name__ == '__main__':
    output = Output()
    output.save_submission()
