import numpy as np
import pandas as pd


class PreprocessForNN(object):
    def __init__(self):
        self.deathData = None
        self.confirmedData = None
        self.features = []
        pass

    def load_data(self):
        self.load_population_dict()
        self.load_icu_dict()
        self.load_death_and_confirmed()

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
            feature = np.append(confirmed_list[i:i+window_size_7], death_list[i:i+window_size_7])
            label = death_list[i+window_size_7:i+window_size_7+window_size_14]
            FIPS_list.append(FIPS)
            label_list.append(label)
            feature_list.append(feature)

        dict = {'FIPS': FIPS_list, 'label': label_list, 'feature': feature_list}
        output = pd.DataFrame(dict)
        return output

    def load_death_and_confirmed(self):
        # return dataframe ['FIPS', 'label', 'feature']
        path = './data/us/covid/'
        self.deathData = pd.read_csv(path + 'deaths.csv')
        self.confirmedData = pd.read_csv(path + 'confirmed_cases.csv')
        self.fetch_none_zero_data()
        output = None
        for FIPS in self.valid_FIPS:
            if FIPS == 0:
                continue
            death_list = self.deathData.loc[self.deathData['countyFIPS'] == FIPS].values[0][4:]
            confimed_list = self.confirmedData.loc[self.confirmedData['countyFIPS'] == FIPS].values[0][4:]
            cur_res = self.add_window(FIPS, death_list, confimed_list)
            # output.append(cur_res)
            if output is None:
                output = cur_res
            else:
                output = pd.concat([output, cur_res])

        return output


    def load_icu_dict(self):
        # TODO:
        # generate a dict, key = countyFIPS, value = icu_bed_count
        pass

    def load_population_dict(self):
        # TODO:
        # generate a dict, key = countyFIPS, value = population
        pass


if __name__ == "__main__":
    preprocess = PreprocessForNN()
    preprocess.load_data()
