import numpy as np
import pandas as pd
from collections import defaultdict


class PreprocessForNN(object):

    def __init__(self):
        self.icu_beds = defaultdict(int)
        self.staffed_beds = defaultdict(int)
        self.licensed_beds = defaultdict(int)
        self.total_population = defaultdict(int)
        self.population_over_sixty = defaultdict(int)

    def load_data(self):
        self.load_death_and_confirmed()
        self.load_beds_dict()
        self.load_death_and_confirmed()
        self.load_population_dict()

    def load_death_and_confirmed(self):
        pass

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


if __name__ == "__main__":
    preprocess = PreprocessForNN()
    preprocess.load_data()
