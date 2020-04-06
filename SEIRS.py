from seirsplus.models import *


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
        pass

    def train(self):
        # self.model = SEIRSModel(beta=0.155, sigma=1/5.2, gamma=1/12.39, initN=100000, initI=100)
        self.model = SEIRSModel(initN=1000000,
                                beta=0.147,
                                sigma=1 / 5.2,
                                gamma=1 / 12.39,
                                mu_I=0.001,
                                mu_0=0,
                                nu=0,
                                xi=0,
                                beta_D=0.147,
                                sigma_D=1 / 5.2,
                                gamma_D=1 / 12.39,
                                mu_D=0.0004,
                                theta_E=0,
                                theta_I=0,
                                psi_E=1.0,
                                psi_I=1.0,
                                initI=10000,
                                initE=0,
                                initD_E=0,
                                initD_I=0,
                                initR=0,
                                initF=0
                                )

        # self.model.run(T=300, checkpoints=self.checkpoints)
        self.model.run(T=300)

    def visualization(self):
        self.model.figure_infections(ylim=0.2)


if __name__ == '__main__':
    seirsModel = SEIRSModelClass()
    # seirsModel.preprocess()
    seirsModel.train()
    seirsModel.visualization()
