import pandas as pd
import numpy as np
import datetime
from scipy import stats


class Output(object):

    def __init__(self, flag_calculate_diff=False, submission='sample_submission.csv'):
        self.flag_calculate_diff = flag_calculate_diff
        self.last_day = '5/17/2020'
        self.submission = submission
        self.sample = self.read_sample()

    def read_sample(self):
        return pd.read_csv(self.submission)

    @staticmethod
    def generate_key(cur_date, FIPS):
        # e.g. 2020-04-01-10001
        return cur_date.strftime('%Y-%m-%d') + '-' + str(int(FIPS))

    @staticmethod
    def calculate_diff(predicted):
        FIPS = predicted['countyFIPS']
        diff_value = predicted.iloc[:,1:].diff(axis=1)
        return pd.concat([FIPS, diff_value.iloc[:, 1:]], axis=1)

    def modify_submission(self, source):
        # predicted part
        if isinstance(source, str):
            predicted = pd.read_csv(source)
        else:
            predicted = source

        date_time = datetime.datetime.strptime(self.last_day, '%m/%d/%Y')
        key_value = dict()

        if self.flag_calculate_diff:
            predicted = self.calculate_diff(predicted)

        for infos in predicted.values:
            FIPS, deaths = infos[0], infos[1:]
            for day_idx, average in enumerate(deaths):
                if day_idx == 0:
                    continue
                cur_date = date_time + datetime.timedelta(days=day_idx)
                key = self.generate_key(cur_date, FIPS)
                value = average
                key_value[key] = value

        # pre = np.zeros((len(self.sample), 9))
        pre = self.sample.values[:, 1:]

        for i in range(len(self.sample)):
            if i % 100000 == 0:
                print("%d/%d" % (i, len(self.sample)))

            key = self.sample['id'][i]
            if key not in key_value or key_value[key]<0:
                continue
            percentiles = self.generate_percentile(key_value[key])
            pre[i][:] = percentiles

        # # # ground truth part
        # ground_truth = pd.read_csv('data/us/covid/deaths.csv')
        # for i in range(len(self.sample)):
        #     date, FIPS = self.format_key(self.sample['id'][i])
        #     if date not in ground_truth or int(FIPS) not in ground_truth['countyFIPS']:
        #         continue
        #     if len(ground_truth.loc[ground_truth['countyFIPS'] == int(FIPS)][date].values) == 0:
        #         continue
        #     average = ground_truth.loc[ground_truth['countyFIPS'] == int(FIPS)][date].values[0]
        #     percentiles = self.generate_percentile(average)
        #     pre[i][:] = percentiles
        #     # for j in range(9):
        #     #     self.sample.iloc[i, j + 1] = percentile[j]

        percentile_keys = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
        for col in range(9):
            self.sample[percentile_keys[col]] = pre[:, col]

    @staticmethod
    def generate_percentile(mid, mode='Norm', std=1.0):
        if mode == 'Norm':
            percentile = list(np.round(stats.norm.ppf(np.linspace(0.1, 0.9, 9)) * std + mid))
            percentile = [max(0, val) for val in percentile]
        else:
            percentile = []
            unit = mid / 5.0
            for i in range(9):
                percentile.append(unit * (i+1))
        return percentile

    @staticmethod
    def format_key(key):
        strs = key.split('-')
        year = strs[0][-2:]
        month = strs[1][1]
        if strs[2][0] == '0':
            day = strs[2][1]
        else:
            day = strs[2]

        date = month + '/' + day + '/' + year

        FIPS = strs[3]

        return date, FIPS

    def save_submission(self, source, dst):
        self.modify_submission(source)
        self.sample.to_csv(dst, index=False)
        print('Successfully saved!')


if __name__ == '__main__':
    # source = 'processed_data/SEIRS_predictions.csv'
    # source = 'models/SVM/svm_mid_2.csv'
    source = 'models/LR/lr_burning.csv'
    dst = 'submissions/submission_final_checkpoint.csv'
    output = Output()
    output.save_submission(source, dst)
