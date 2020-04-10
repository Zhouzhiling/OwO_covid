import pandas as pd


class Output(object):

    def __init__(self):
        self.sample = self.read_sample()
        self.last_day = '4/6/2020'

    @staticmethod
    def read_sample():
        return pd.read_csv('sample_submission.csv')

    def modify_submission(self):
        # predicted part
        predicted = pd.read_csv('processed_data/SEIRS_predictions.csv')
        for deaths in predicted.values:
            FIPS, deaths = deaths[0], deaths[1:]
        
    def generate_percentile(self, mid):
        length = 9
        output = []
        unit = mid / 5.0
        for i in range(9):
            output.append(unit * (i+1))
        return output


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
    # output.save_submission()
    output.modify_submission()
    # print(output.generate_percentile(5))
