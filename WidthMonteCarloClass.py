import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate
import scipy.stats
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.ndimage
import scipy.spatial
import AuxFunctions

class WidthDistributionGenerator:
    def __init__(self, Npsrs):
        self.Npsrs = Npsrs
        self.chis_dist = lambda t: 1 + 0 * t
        self.dzetas_dist = lambda t: np.sin(t)
        self.chis = np.zeros(self.Npsrs)
        self.dzetas = np.zeros(self.Npsrs)
        self.betas = np.zeros(self.Npsrs)
        self.Ps = np.zeros(self.Npsrs)
        self.Pdots = np.zeros(self.Npsrs) #*1e-15
        self.visibility_mask = np.zeros(self.Npsrs)
        self.rho0 = 8.5 * np.pi / 180
        self.sigma = 0.3
        self.n_x = 0.5
        self.n_chi = 0
        self.rho0s = np.random.lognormal(np.log(self.rho0) + self.sigma**2, self.sigma, self.Npsrs)
        self.Ws = []

    def set_rho0(self, rho0):
        self.rho0 = rho0 * np.pi / 180
        self.rho0s = np.random.lognormal(np.log(self.rho0) + self.sigma**2, self.sigma, self.Npsrs)

    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def set_n_x(self, n_x):
        self.n_x = n_x

    def set_n_chi(self, n_chi):
        self.n_chi = n_chi
        self.chis_dist = lambda t: np.sin(t)**self.n_chi

    def generate_PPdot_sample(self):
        distribution = scipy.stats.multivariate_normal([-0.24, -14.52+15], [[0.13,  -0.02], [-0.02, 0.98]])
        PPdot_arr = 10**distribution.rvs(self.Npsrs).T
        self.Ps, self.Pdots = PPdot_arr[0], PPdot_arr[1]
        # self.Ps = 10**(0.65 - 0.30 * PC1 - 0.22 * PC2)
        # self.Pdots = 10**(3.9 - 0.78 * PC1 + 0.31 * PC2)


    def generate_psr_sample(self):
        self.rho0s = np.random.lognormal(np.log(self.rho0) + self.sigma**2, self.sigma, self.Npsrs)
        self.chis = AuxFunctions.inverse_sample_function(self.chis_dist, self.Npsrs, 0, np.pi)
        self.dzetas = AuxFunctions.inverse_sample_function(self.dzetas_dist, self.Npsrs, 0, np.pi)
        self.betas = self.dzetas - self.chis
        self.generate_PPdot_sample()
        # self.Ps = scipy.stats.gamma.rvs(2.5, size=self.Npsrs) * 0.3
        # self.Ps = np.ones(self.Npsrs) * 0.5
        self.rhos = self.rho0s * self.Ps**(-0.5)
    
    def generate_width_sample(self):
        self.generate_psr_sample()
        for i in range(self.Npsrs):
            x = np.abs(self.betas[i]) / self.rhos[i]
            if x < 1 and self.Pdots[i] >1e-5 * self.Ps[i]**(11/4) * np.abs(np.cos(self.chis[i]))**(-7/4) and np.random.uniform() < (1 - x**2)**self.n_x:
                W = AuxFunctions.calculate_width(self.chis[i], self.betas[i], self.rhos[i]) * 180 / np.pi
                self.Ws.append(W)
            else:
                self.visibility_mask[i] = 1
        self.Ws = np.array(self.Ws)
    
    def get_Ws(self):
        return self.Ws

    def get_chis(self):
        return np.ma.array(self.chis, mask=self.visibility_mask).compressed()

    def get_Ps(self):
        return np.ma.array(self.Ps, mask=self.visibility_mask).compressed()
    
    def get_dzetas(self):
        return np.ma.array(self.dzetas, mask=self.visibility_mask).compressed()
    
    def get_betas(self):
        return np.ma.array(self.betas, mask=self.visibility_mask).compressed()
        
    def get_rhos(self):
        return np.ma.array(self.rhos, mask=self.visibility_mask).compressed()

    def get_rho0s(self):
        return np.ma.array(self.rho0s, mask=self.visibility_mask).compressed()


    
    

