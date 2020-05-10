import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import math
import os.path
from sklearn.preprocessing import StandardScaler


def feature_engineering(feature):
    # confirmed, death, confirmed_diff, death_diff, confirmed_square, death_square
    diff = [0 for _ in range(12)]
    squared = [0 for _ in range(14)]
    for idx in range(2):
        for i in range(1, 7):
            diff[idx * 6 + i - 1] = feature[idx * 7 + i] - feature[idx * 7 + i - 1]
    feature.extend(diff)

    for i in range(14):
        squared[i] = feature[i] * feature[i]
    feature.extend(squared)

    return feature

class PreprocessForNN(object):

    def __init__(self):
        self.deathData = None
        self.confirmedData = None
        self.features = []
        self.icu_beds = defaultdict(int)
        self.staffed_beds = defaultdict(int)
        self.licensed_beds = defaultdict(int)
        self.total_population = defaultdict(int)
        self.population_over_sixty = defaultdict(int)
        self.policies = defaultdict(lambda: [0 for _ in range(8)])
        self.scaler_feature = StandardScaler()
        self.scaler_label = StandardScaler()

    def load_policies(self):
        '''
        dict[key][0] = stay at home
        dict[key][1] = >50 gathering
        dict[key][2] = >500 gathering
        dict[key][3] = public schools
        dict[key][4] = restaurant dine-in
        dict[key][5] = entertainment/gym
        dict[key][6] = federal guidelines
        dict[key][7] = foreign travel ban
        '''
        policy = pd.read_csv(filepath_or_buffer='data/us/other/policies.csv')
        label_names = ['stay at home', '>50 gatherings', '>500 gatherings',
                       'public schools', 'restaurant dine-in', 'entertainment/gym']
        mean_times = [0 for _ in range(len(label_names))]
        ranges = [0 for _ in range(len(label_names))]
        min_times = [0 for _ in range(len(label_names))]

        for idx, label in enumerate(label_names):
            times = policy[label].values[~np.isnan(policy[label])]
            mean_times[idx] = np.mean(times)
            ranges[idx] = max(1, np.max(times) - np.min(times))
            min_times[idx] = np.min(times)

        for item in policy.iterrows():
            fips = item[1]['FIPS']
            for idx, label in enumerate(label_names):
                if not math.isnan(item[1][label]):
                    scaled = 1 - (item[1][label] - min_times[idx]) / ranges[idx]
                    self.policies[fips][idx] = scaled

    def load_data(self):
        self.load_beds_dict()
        self.load_population_dict()
        self.load_policies()

    def fetch_none_zero_data(self):
        last_date = self.deathData.columns[-1]
        countyNoneZero = self.deathData.loc[self.deathData[last_date] != 0]
        countyNoneZero.reset_index(drop=True)
        self.valid_FIPS = set(self.deathData['countyFIPS'])
        self.deathData = countyNoneZero

        keep_flag = []
        for FIPS in self.confirmedData['countyFIPS']:
            keep_flag.append(int(FIPS) in self.valid_FIPS)
        self.confimedData = self.confirmedData.loc[keep_flag, :]

    @staticmethod
    def add_window(FIPS, death_list, confirmed_list, mode):
        window_size_7 = 7
        window_size_14 = 14
        count = len(death_list) - window_size_7 - window_size_14 + 1
        output = []
        FIPS_list = []
        label_list = []
        feature_list = []
        if mode == 'train':
            for i in range(count):
                feature = confirmed_list[i:i+window_size_7] + death_list[i:i+window_size_7]
                feature = feature_engineering(feature)
                label = death_list[i+window_size_7:i+window_size_7+window_size_14]
                if sum(label) == 0:
                    continue
                FIPS_list.append(FIPS)
                label_list.append(label)
                feature_list.append(feature)
            dict = {'FIPS': FIPS_list, 'label': label_list, 'feature': feature_list}
        else:
            feature = confirmed_list[-window_size_7:] + death_list[-window_size_7:]
            feature = feature_engineering(feature)
            FIPS_list.append(FIPS)
            feature_list.append(feature)
            dict = {'FIPS': FIPS_list, 'feature': feature_list}

        output = pd.DataFrame(dict)
        return output

    def load_death_and_confirmed(self, mode='train'):
        # return dataframe ['FIPS', 'label', 'feature']
        path = './processed_data/'
        death_path = path + 'daily_death_from_nyu.csv'
        confirmed_path = path + 'daily_confirmed_from_nyu.csv'

        if not os.path.isfile(death_path):
            self.transform_format('death')
            self.transform_format('confirmed')

        self.deathData = pd.read_csv(death_path)
        self.confirmedData = pd.read_csv(confirmed_path)
        self.valid_FIPS = set(self.deathData['countyFIPS'])
        # self.fetch_none_zero_data()
        output = None
        for FIPS in self.valid_FIPS:
            if FIPS == 0:
                continue
            death_list = list(self.deathData.loc[self.deathData['countyFIPS'] == FIPS].values[0][4:])
            confirmed_list = list(self.confirmedData.loc[self.confirmedData['countyFIPS'] == FIPS].values[0][4:])
            cur_res = self.add_window(FIPS, death_list, confirmed_list, mode)
            if output is None:
                output = cur_res
            else:
                output = pd.concat([output, cur_res])

        return output


    def load_beds_dict(self):
        # load number of beds from 'beds_by_county.csv'
        beds = pd.read_csv(filepath_or_buffer='data/us/hospitals/beds_by_county.csv')
        for item in beds.iterrows():
            self.icu_beds[item[1]['FIPS']] = item[1]['icu_beds']
            self.staffed_beds[item[1]['FIPS']] = item[1]['staffed_beds']
            self.licensed_beds[item[1]['FIPS']] = item[1]['licensed_beds']

    def load_population_dict(self):
        # load population from 'county_populations.csv'
        population = pd.read_csv(filepath_or_buffer='data/us/demographics/county_populations.csv')
        for item in population.iterrows():
            self.total_population[item[1]['FIPS']] = item[1]['total_pop']
            self.population_over_sixty[item[1]['FIPS']] = item[1]['60plus']

    def generate_training_data(self, mode='outbreak'):
        self.load_data()
        data = self.load_death_and_confirmed()

        confirmed_death_feature_key = 'feature'
        death_key = 'label'

        feature, label = [], []

        for item in data.iterrows():
            FIPS = item[1]['FIPS']
            point = item[1][confirmed_death_feature_key]
            point.append(self.icu_beds[FIPS])
            point.append(self.staffed_beds[FIPS])
            point.append(self.licensed_beds[FIPS])
            point.append(self.total_population[FIPS])
            point.append(self.population_over_sixty[FIPS])
            point.extend(self.policies[FIPS])

            feature.append(point)
            label.append(item[1][death_key])

        threshold = 100

        outbreak_feature, outbreak_label = [], []
        burning_feature, burning_label = [], []

        for i in range(len(feature)):
            if np.max(label[i]) > threshold:
                outbreak_feature.append(feature[i])
                outbreak_label.append(label[i])
            else:
                burning_feature.append(feature[i])
                burning_label.append(label[i])

        if mode == 'outbreak':
            scalered_feature = np.array(outbreak_feature)
            scalered_label = np.array(outbreak_label)
            scalered_feature = self.scaler_feature.fit_transform(scalered_feature)
            scalered_label = self.scaler_label.fit_transform(scalered_label)
            return scalered_feature, scalered_label
        else:
            scalered_feature = np.array(burning_feature)
            scalered_label = np.array(burning_label)
            scalered_feature = self.scaler_feature.fit_transform(scalered_feature)
            scalered_label = self.scaler_label.fit_transform(scalered_label)
            return scalered_feature, scalered_label

    def generate_testing_data(self, mode='outbreak'):

        self.load_data()
        data = self.load_death_and_confirmed('test')
        confirmed_death_feature_key = 'feature'
        feature, FIPS_list = [], []

        for item in data.iterrows():
            FIPS = item[1]['FIPS']
            point = item[1][confirmed_death_feature_key]
            point.append(self.icu_beds[FIPS])
            point.append(self.staffed_beds[FIPS])
            point.append(self.licensed_beds[FIPS])
            point.append(self.total_population[FIPS])
            point.append(self.population_over_sixty[FIPS])
            point.extend(self.policies[FIPS])

            feature.append(point)
            FIPS_list.append(FIPS)

        threshold = 10

        outbreak_feature, outbreak_FIPS = [], []
        burning_feature, burning_FIPS = [], []

        for i in range(len(feature)):
            if np.max(feature[i][13]) > threshold:
                outbreak_feature.append(feature[i])
                outbreak_FIPS.append(FIPS_list[i])
            else:
                burning_feature.append(feature[i])
                burning_FIPS.append(FIPS_list[i])

        if mode == 'outbreak':
            outbreak_base = [outbreak_feature[i][13] for i in range(len(outbreak_feature))]
            scalered_feature = np.array(outbreak_feature)
            scalered_feature = self.scaler_feature.fit_transform(scalered_feature)
            return scalered_feature, pd.Series(outbreak_FIPS), outbreak_base
        else:
            burning_base = [burning_feature[i][13] for i in range(len(burning_feature))]
            scalered_feature = np.array(burning_feature)
            scalered_feature = self.scaler_feature.fit_transform(scalered_feature)
            return scalered_feature, pd.Series(burning_FIPS), burning_base

    @staticmethod
    def transform_format(mode='death'):
        '''
        The evaluation data is from ./data/us/covid/nyt_us_counties_daily
        The data we employed for training is ./data/us/covid/deaths
        This function transform nyt_us_counties_daily into the foramt of deaths
        Store the result in csv named ./data/us/covid/nyt_deaths
        :return:
        '''

        if mode == 'death':
            out_path = './processed_data/daily_death_from_nyu.csv'
        else:
            out_path = './processed_data/daily_confirmed_from_nyu.csv'

        template_path = './data/us/covid/deaths.csv'
        source_path = './data/us/covid/nyt_us_counties_daily.csv'

        template = pd.read_csv(template_path)
        source = pd.read_csv(source_path)

        pre = np.zeros((len(template), len(template.columns)-4))

        first_day = datetime.datetime.strptime(template.columns[4], '%m/%d/%Y')
        fips_idx = dict()

        for i in range(len(template)):
            fips_idx[template['countyFIPS'][i]] = i

        for infos in source.values:
            FIPS = infos[0]
            date = datetime.datetime.strptime(infos[1], '%Y-%m-%d')
            if mode == 'death':
                value = infos[5]
            else:
                value = infos[4]

            date_index = (date - first_day).days
            if date_index >= 0 and str(FIPS) in fips_idx:
                fips_index = fips_idx[str(FIPS)]
                pre[fips_index][date_index] = value

        for day_diff in range(len(template.columns)-4):
            day_to_add = datetime.timedelta(days=day_diff)
            label_name = (day_to_add + first_day).strftime('%#m/%#d/%Y')
            template[label_name] = pre[:, day_diff]

        template.to_csv(out_path, index=False)

    def get_average(self):
        return self.scaler_label.mean_

    def get_std(self):
        return np.sqrt(self.scaler_label.var_)


if __name__ == "__main__":
    preprocess = PreprocessForNN()
    preprocess.transform_format('death')
    preprocess.transform_format('confirmed')

    # preprocess.load_data()
    # preprocess.generate_training_data()
    # preprocess.generate_testing_data()