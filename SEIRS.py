from seirsplus.models import *
from preprocesss import PreProcess
import pandas as pd


class SEIRSModelClass(object):
    def __init__(self):
        self.models = []
        self.data = None
        self.checkpoints = {
            't': [50, 100],
            'beta': [0.12, 0.147],
            'theta_E': [0.02, 0.02],
            'theta_I': [0.02, 0.02]
        }
        self.initial_parameters = []
        self.initial_death = []
        self.period = 300

    def preprocess(self):
        last_day = '4/6/2020'
        preprocess = PreProcess('./data/us/covid/')
        countyInfo = preprocess.deathData[['countyFIPS', 'County Name', 'State', 'stateFIPS', last_day]]
        county_population = pd.read_csv('data/us/demographics/county_populations.csv')
        countyInfo = countyInfo.rename(columns={last_day: "death_%s" % last_day})

        county_initial_parameters = county_population.merge(countyInfo, left_on='FIPS', right_on='countyFIPS')
        county_numbers = len(county_initial_parameters)
        beta = pd.Series([0.147 for _ in range(county_numbers)], name='beta')
        sigma = pd.Series([1 / 5.2 for _ in range(county_numbers)], name='sigma')
        gamma = pd.Series([1 / 12.39 for _ in range(county_numbers)], name='gamma')
        mu_I = pd.Series([0.001 for _ in range(county_numbers)], name='mu_I')
        xi = pd.Series([0 for _ in range(county_numbers)], name='xi')
        sigma_D = pd.Series([1 / 5.2 for _ in range(county_numbers)], name='sigma_D')
        theta_E = pd.Series([0 for _ in range(county_numbers)], name='theta_E')
        theta_I = pd.Series([0 for _ in range(county_numbers)], name='theta_I')
        psi_E = pd.Series([0 for _ in range(county_numbers)], name='psi_E')
        psi_I = pd.Series([0 for _ in range(county_numbers)], name='psi_I')
        initF = pd.Series(countyInfo["death_%s" % last_day], name='initF')

        self.initial_parameters = pd.concat([county_initial_parameters, beta, sigma, gamma, mu_I, xi, sigma_D, theta_E, theta_I, psi_E, psi_I, initF], axis=1)
        self.initial_parameters.to_csv('processed_data/county_initial_parameters.csv', index=False)

    def train(self):
        for i in range(len(self.initial_parameters)):

            model = SEIRSModel(
                initN=self.initial_parameters['total_pop'][i],
                beta=self.initial_parameters['beta'][i],         # transmissions per S-I contact per time
                sigma=self.initial_parameters['sigma'][i],      # inverse of incubation period
                gamma=self.initial_parameters['gamma'][i],    # inverse of infectious period
                mu_I=self.initial_parameters['mu_I'][i],         # deaths per infectious individual per time
                mu_0=0,
                nu=0,
                xi=self.initial_parameters['xi'][i],               # 0: inverse of temporary immunity period
                beta_D=0.147,
                sigma_D=self.initial_parameters['sigma_D'][i],    # rate of progression for detected cases
                gamma_D=1 / 12.39,
                mu_D=0.0004,
                theta_E=self.initial_parameters['theta_E'][i],          # rate of testing for exposed individuals
                theta_I=self.initial_parameters['theta_I'][i],          # rate of testing for infectious individuals
                psi_E=self.initial_parameters['psi_E'][i],          # rate of positive test results for exposed individuals
                psi_I=self.initial_parameters['psi_I'][i],          # rate of positive test results for infectious individuals
                initI=1,
                initE=0,        # initI * 0.15
                initD_E=0,
                initD_I=0,
                initR=self.initial_parameters['initF'][i] * 2,        # initF * 2
                initF=self.initial_parameters['initF'][i]
            )

            model.run(T=self.period, checkpoints=self.checkpoints)

            self.models.append(model)

    def getDeath(self):
        # output = pd.DataFrame()
        death_count = []
        for i in range(len(self.models)):
            death_count.append([round(death) for death in self.models[i].numF[-14:]])
        death_count = pd.DataFrame(death_count)
        county_info = self.initial_parameters['countyFIPS']
        output = pd.concat([county_info, death_count], axis=1)
        output.to_csv('./processed_data/SEIRS_predictions.csv', index=False)

    def visualization(self):
        self.models[0].figure_infections(vlines=self.checkpoints['t'], plot_F='line', ylim=0.2)


if __name__ == '__main__':
    seirsModel = SEIRSModelClass()
    seirsModel.preprocess()
    seirsModel.train()
    # seirsModel.visualization()
    # seirsModel.getDeath()
