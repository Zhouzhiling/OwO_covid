import numpy as np
import pandas as pd
from collections import defaultdict


class PreprocessForNN(object):

    def __init__(self):
        self.icu_beds = defaultdict(int)
        self.total_population = defaultdict(int)
        pass

    def load_data(self):
        self.load_death_and_confirmed()
        self.load_icu_dict()
        self.load_death_and_confirmed()
        self.load_population_dict()

    def load_death_and_confirmed(self):
        pass

    def load_icu_dict(self):
        beds = pd.read_csv(filepath_or_buffer='data/us/hospitals/beds_by_county.csv')
        for item in beds.iterrows():
            self.icu_beds[item[1]['FIPS']] = item[1]['icu_beds']

    def load_population_dict(self):
        # TODO:
        # generate a dict, key = countyFIPS, value = population
        population = pd.read_csv(filepath_or_buffer='data/us/demographics/county_populations.csv')
        for item in population.iterrows():
            self.total_population[item[1]['FIPS']] = item[1]['total_pop']
        pass


if __name__ == "__main__":
    preprocess = PreprocessForNN()
    preprocess.load_data()
