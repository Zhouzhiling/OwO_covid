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

        # ground truth part
        ground_truth = pd.read_csv('data/us/covid/deaths.csv')
        for i in range(len(self.sample)):
            date, FIPS = self.format_key(self.sample['id'][i])
            if date not in ground_truth or int(FIPS) not in ground_truth['countyFIPS']:
                continue
            if len(ground_truth.loc[ground_truth['countyFIPS'] == int(FIPS)][date].values) == 0:
                continue
            average = ground_truth.loc[ground_truth['countyFIPS'] == int(FIPS)][date].values[0]
            percentile = self.generate_percentile(average)
            for j in range(9):
                self.sample.iloc[i, j + 1] = percentile[j]
        
    @staticmethod
    def generate_percentile(mid):
        percentile = []
        unit = mid / 5.0
        for i in range(9):
            percentile.append(unit * (i+1))
        return percentile

    @staticmethod
    def format_key(key):
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
        self.sample.to_csv('submission.csv')


if __name__ == '__main__':
    output = Output()
    output.save_submission()
    output.modify_submission()
    # print(output.generate_percentile(5))
