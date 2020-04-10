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
        self.period = 85
        self.last_day = '4/6/2020'

    def preprocess(self):

        # initiate parameters
        preprocess = PreProcess('./data/us/covid/')
        countyInfo = preprocess.deathData[['countyFIPS', 'County Name', 'State', 'stateFIPS', self.last_day]]
        county_population = pd.read_csv('data/us/demographics/county_populations.csv')
        countyInfo = countyInfo.rename(columns={self.last_day: "death_%s" % self.last_day})

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
        initF = pd.Series(county_initial_parameters["death_%s" % self.last_day], name='initF')

        self.initial_parameters = pd.concat([county_initial_parameters, beta, sigma, gamma, mu_I, xi, sigma_D, theta_E, theta_I, psi_E, psi_I, initF], axis=1)
        self.initial_parameters.to_csv('processed_data/county_initial_parameters.csv', index=False)

        # initiate infectious population
        county_confirmed_cases = pd.read_csv('data/us/covid/confirmed_cases.csv')
        county_initial_confirmed_case = pd.Series([0 for _ in range(county_numbers)], name='init_infected')

        self.initial_parameters = pd.concat([self.initial_parameters, county_initial_confirmed_case], axis=1)

        for i in range(county_numbers):
            self.initial_parameters['init_infected'][i] = county_confirmed_cases.loc[county_confirmed_cases['countyFIPS'] == self.initial_parameters['countyFIPS'][i]]['4/6/20']

        # TA's bug
        self.initial_parameters = self.initial_parameters.drop(county_numbers - 1)

    def train(self):
        for i in range(len(self.initial_parameters)):

            infected = round(self.initial_parameters['init_infected'][i] / 0.85)
            exposed = round(infected * 0.15)

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
                initI=infected,
                initE=exposed,
                initD_E=round(self.initial_parameters['init_infected'][i] * 0.15),
                initD_I=round(self.initial_parameters['init_infected'][i] * 0.85),
                initR=self.initial_parameters['initF'][i] * 2,  # initF * 2
                initF=self.initial_parameters['initF'][i]
            )

            model.run(T=self.period)

            self.models.append(model)

    def getDeath(self):
        # output = pd.DataFrame()
        death_count = []
        for i in range(len(self.models)):
            curr_county_death = []
            for j in range(self.period):
                curr_county_death.append(self.models[i].numF[(j + 1) * 10])
            death_count.append(curr_county_death)
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
    seirsModel.getDeath()
