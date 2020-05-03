import numpy as np
import pandas as pd


class PreprocessForNN(object):
    def __init__(self):
        pass

    def load_data(self):
        self.load_confirmed_dict()
        self.load_icu_dict()
        self.load_death_and_confirmed()

    def load_death_and_confirmed(self):
        pass

    def load_icu_dict(self):
        pass

    def load_confirmed_dict(self):
        pass
