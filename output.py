import pandas as pd
import datetime

class Output(object):

    def __init__(self):
        self.sample = self.read_sample()
        self.last_day = '4/6/2020'

    @staticmethod
    def read_sample():
        return pd.read_csv('sample_submission.csv')

    def generate_key(self, cur_date, FIPS):
        # e.g. 2020-04-01-10001
        return cur_date.strftime('%Y-%m-%d') + '-' + str(int(FIPS))
        

    def modify_submission(self):
        # predicted part
        predicted = pd.read_csv('processed_data/SEIRS_predictions.csv')
        date_time = datetime.datetime.strptime(self.last_day, '%m/%d/%Y')
        key_value = dict()

        for infos in predicted.values:
            FIPS, deaths = infos[0], infos[1:]
            for day_idx, average in enumerate(deaths):
                if day_idx == 0:
                    continue
                cur_date = date_time + datetime.timedelta(days=day_idx)
                key = self.generate_key(cur_date, FIPS)
                value = average
                key_value[key] = value
                # for i, v in enumerate(value):
                # index = self.sample['id'] == key
                # self.sample.loc[][str((i+1)*10)] = v
                # print(key)
                # print(self.sample.loc[self.sample['id'] == key][str((i+1)*10)])

        # for i in range(len(self.sample.values)):
        #     key = self.sample.values[i][0]
        #     if key not in key_value:
        #         continue
        #     percentiles = key_value[key]
        #     for j in range(len(percentiles)):
        #         self.sample.iloc[i, j] = percentiles[j]
        print(1)
        for i in range(len(self.sample)):
            if i % 1000 == 0:
                print("%d/%d" % (i, len(self.sample)))

            key = self.sample['id'][i]
            if key not in key_value:
                continue
            percentiles = self.generate_percentile(key_value[key])
            # for j in range(len(percentiles)):
            self.sample.iloc[i, 1:10] = percentiles

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
    # print(output.generate_percentile(5))
