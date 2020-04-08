from seirsplus.models import *
from preprocesss import PreProcess


class SEIRSModelClass(object):
    def __init__(self):
        self.model = None
        self.data = None
        self.checkpoints = {
            't': [50, 100],
            'beta': [0.12, 0.147],
            'theta_E': [0.02, 0.02],
            'theta_I': [0.02, 0.02]
        }

    def preprocess(self):
        preprocess = PreProcess('./data/us/covid/')
        countyInfo = preprocess.deathData[['countyFIPS', 'County Name', 'State', 'stateFIPS']]
        print(countyInfo['countyFIPS'])


    def train(self):
        # self.model = SEIRSModel(beta=0.155, sigma=1/5.2, gamma=1/12.39, initN=100000, initI=100)
        self.model = SEIRSModel(
            initN=1000000,
            beta=0.147,         # transmissions per S-I contact per time
            sigma=1 / 5.2,      # inverse of incubation period
            gamma=1 / 12.39,    # inverse of infectious period
            mu_I=0.001,         # deaths per infectious individual per time
            mu_0=0,
            nu=0,
            xi=0,               # 0: inverse of temporary immunity period
            beta_D=0.147,
            sigma_D=1 / 5.2,    # rate of progression for detected cases
            gamma_D=1 / 12.39,
            mu_D=0.0004,
            theta_E=0,          # rate of testing for exposed individuals
            theta_I=0,          # rate of testing for infectious individuals
            psi_E=1.0,          # rate of positive test results for exposed individuals
            psi_I=1.0,          # rate of positive test results for infectious individuals
            initI=10000,
            initE=0,
            initD_E=0,
            initD_I=0,
            initR=0,
            initF=0
        )

        self.model.run(T=300, checkpoints=self.checkpoints)
        # self.model.run(T=300)

    def visualization(self):
        self.model.figure_infections(vlines=self.checkpoints['t'], plot_F='line', ylim=0.2)


if __name__ == '__main__':
    seirsModel = SEIRSModelClass()
    seirsModel.preprocess()
    seirsModel.train()
    seirsModel.visualization()
