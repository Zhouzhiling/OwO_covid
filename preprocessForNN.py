import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import os.path


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

    def load_data(self):
        self.load_beds_dict()
        self.load_population_dict()

    def fetch_none_zero_data(self):
        last_date = self.deathData.columns[-1]
        countyNoneZero = self.deathData.loc[self.deathData[last_date] != 0]
        countyNoneZero.reset_index(drop=True)
        self.valid_FIPS = set(countyNoneZero['countyFIPS'])
        self.deathData = countyNoneZero

        keep_flag = []
        for FIPS in self.confirmedData['countyFIPS']:
            keep_flag.append(int(FIPS) in self.valid_FIPS)
        self.confimedData = self.confirmedData.loc[keep_flag, :]

    @staticmethod
    def add_window(FIPS, death_list, confirmed_list):
        window_size_7 = 7
        window_size_14 = 14
        count = len(death_list) - window_size_7 - window_size_14 + 1
        output = []
        FIPS_list = []
        label_list = []
        feature_list = []
        for i in range(count):
            feature = confirmed_list[i:i+window_size_7] + death_list[i:i+window_size_7]
            label = death_list[i+window_size_7:i+window_size_7+window_size_14]
            if sum(label) == 0:
                continue
            FIPS_list.append(FIPS)
            label_list.append(label)
            feature_list.append(feature)

        dict = {'FIPS': FIPS_list, 'label': label_list, 'feature': feature_list}
        output = pd.DataFrame(dict)
        return output

    def load_death_and_confirmed(self):
        # return dataframe ['FIPS', 'label', 'feature']
        path = './processed_data/'
        death_path = path + 'daily_death_from_nyu.csv'
        confirmed_path = path + 'daily_confirmed_from_nyu.csv'

        if not os.path.isfile(death_path):
            self.transform_format('death')
            self.transform_format('confirmed')

        self.deathData = pd.read_csv(death_path)
        self.confirmedData = pd.read_csv(confirmed_path)
        self.fetch_none_zero_data()
        output = None
        for FIPS in self.valid_FIPS:
            if FIPS == 0:
                continue
            death_list = list(self.deathData.loc[self.deathData['countyFIPS'] == FIPS].values[0][4:])
            confirmed_list = list(self.confirmedData.loc[self.confirmedData['countyFIPS'] == FIPS].values[0][4:])
            cur_res = self.add_window(FIPS, death_list, confirmed_list)
            # output.append(cur_res)
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

    def generate_training_data(self):
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

            feature.append(point)
            label.append(item[1][death_key])

        return np.array(feature), np.array(label)

    def generate_testing_data(self):

        # todo add implementation

        feature, FIPS = [], []

        return feature, FIPS

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
            if date_index >= 0 and FIPS in fips_idx:
                fips_index = fips_idx[FIPS]
                pre[fips_index][date_index] = value

        for day_diff in range(len(template.columns)-4):
            day_to_add = datetime.timedelta(days=day_diff)
            label_name = (day_to_add + first_day).strftime('%#m/%#d/%Y')
            template[label_name] = pre[:, day_diff]

        template.to_csv(out_path, index=False)


if __name__ == "__main__":
    preprocess = PreprocessForNN()
    # preprocess.transform_format('death')
    # preprocess.transform_format('confirmed')

    preprocess.load_data()
    preprocess.generate_training_data()