import DataLoaders
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import urllib
from bs4 import BeautifulSoup
import requests
from astropy.io import fits
from astropy.table import Table
import glob
import scipy.integrate
import scipy.stats
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.ndimage
import scipy.spatial
import AuxFunctions

class PulsarProfile:
    def load_profile_data(self):
        """
        loads profile data
        data[0] = I, data[1] = Q, data[2] = U, data[3] = V, data[4] = L, data[5] = PA
        returns: data 
        """
        if self.source == 'FAST':
            data = DataLoaders.load_FAST_data(self.file_name)
        elif self.source == 'MeerKAT':
            data = DataLoaders.load_MeerKAT_mean_data(self.file_name)
        elif self.source == 'EPN':
            data = DataLoaders.read_EPN_data(self.file_name)
        else:
            print('Unknown source!!!') #TODO raise error
        return data

    def __init__(self, file_name, source):
        self.file_name = file_name
        self.source = source
        data = self.load_profile_data()
        self.I = data[0]
        self.Q = data[1]
        self.U = data[2]
        self.V = data[3]
        self.L = data[4]
        self.PA = data[5]
        self.Ncounts = self.I.shape[0]
        self.peaks_arr = self.find_peaks()
        self.profile_type = self.find_profile_type()

    def plot_profile(self, plot_fit=False, zoom=False):
        # TODO zoom to profile
        phase = np.linspace(0, 1, self.Ncounts)
        fig, axs = plt.subplots(2, height_ratios=[1, 4])
        noise = AuxFunctions.noise_estimation(self.I)
        good_L_array = ((self.L / (np.abs(self.I) + 0.01)) > 0.1) & (self.I > 4 * noise)
        axs[0].set_xlim(0, 1)
        axs[0].scatter(phase[good_L_array], self.PA[good_L_array], c='black', s=3)
        axs[1].plot(phase, self.I, c='black', label='I')
        axs[1].plot(phase, self.V, c='blue', label='V')
        axs[1].plot(phase, self.L, c='red', label='L')
        if plot_fit == True:
            noise = AuxFunctions.noise_estimation(self.I)
            spl = scipy.interpolate.splrep(phase, self.I, s=self.Ncounts*noise**2)
            I_func = scipy.interpolate.BSpline(*spl)
            axs[1].plot(phase, I_func(phase), c='yellow')
        fig.legend()
        fig.show()
        
    def find_peaks(self):
        noise = AuxFunctions.noise_estimation(self.I)
        phase = np.linspace(0, 1, self.Ncounts)
        spl = scipy.interpolate.splrep(phase, self.I, s=self.Ncounts*noise**2)
        I_func = scipy.interpolate.BSpline(*spl)
        peaks, info = scipy.signal.find_peaks(I_func(phase), prominence=np.max(I_func(phase) * 0.05), height=2 * noise) #before here was max(self.I * 0.05) and height=4*noise
        return peaks

    def find_profile_type(self):
        if self.peaks_arr.shape[0] == 1:
            return 'single'
        elif self.peaks_arr.shape[0] >= 2:
            dists = scipy.spatial.distance.cdist(self.peaks_arr.reshape((-1, 1)), self.peaks_arr.reshape((-1, 1)), lambda u, v: min((u-v)%self.Ncounts, (v-u)%self.Ncounts))
            maxdist = np.max(dists)
            if maxdist > 0.9 * self.Ncounts / 2:
                return 'orthogonal'
            else:
                if self.peaks_arr.shape[0] == 2:
                    return 'double'
                elif self.peaks_arr.shape[0] == 3:
                    return 'tripple'
                else:
                    return 'complex'
        else:
            return 'error'
        
    def get_level_bounds(self, level):
        """
        params: level - from 0 to 100
        returns: leftmost and right most indicies correspoding to level
        """
        height = np.max(self.I) * (level/100)
        # oversampling-------
        phase = np.linspace(0, 1, self.Ncounts)
        phase_x10 = np.linspace(0, 1, 10 * self.Ncounts)
        Is = scipy.interpolate.interp1d(phase, self.I)(phase_x10)
        #--------------------
        left_ind = np.where(np.isclose(Is, height, rtol=1e-1))[0][0] // 10
        right_ind = np.where(np.isclose(Is, height, rtol=1e-1))[0][-1] // 10
        if right_ind - left_ind < 3:
            left_ind = max(0, left_ind - 3)
            right_ind = min(self.Ncounts-1, right_ind + 3)

        return (left_ind, right_ind)

    def get_Wa(self, a):
        """
        params: a - level (from 0 to 100)
        returns: W_a - width on level a
        """
        height_a = np.max(self.I) * (a/100)
        # oversampling-------
        phase = np.linspace(0, 1, self.Ncounts)
        phase_x10 = np.linspace(0, 1, 10 * self.Ncounts)
        Is = scipy.interpolate.interp1d(phase, self.I)(phase_x10)
        #--------------------
        left_ind = np.where(np.isclose(Is, height_a, rtol=1e-1))[0][0]
        right_ind = np.where(np.isclose(Is, height_a, rtol=1e-1))[0][-1]
        # print(left_ind/self.Ncounts, right_ind/self.Ncounts, self.I[left_ind] / np.max(self.I), self.I[right_ind] / np.max(self.I))
        return 360 * (right_ind - left_ind) / 10 / self.Ncounts

        # iil = np.where(np.isclose(pulsar_data[0][:peaks[0]], min(pulsar_data[0][:peaks[0]], key=lambda x:abs(x-height50))))[0][0]
        # iir = np.where(np.isclose(pulsar_data[0][peaks[0]:], min(pulsar_data[0][peaks[0]:], key=lambda x:abs(x-height50))))[0][0] + peaks[0]
        # if iir <= peaks[1]:
        #     height50 = np.max(pulsar_data[0][peaks[1]])/2
        #     iir = np.where(np.isclose(pulsar_data[0][peaks[1]:], min(pulsar_data[0][peaks[1]:], key=lambda x:abs(x-height50))))[0][0] + peaks[1]

    def get_mode(self):
        left_ind, right_ind = self.get_level_bounds(10)
        # print(left_ind, right_ind)
        Varray = self.V[left_ind:right_ind]
        PAarray = self.PA[left_ind:right_ind]
        N = Varray.shape[0]
        xs = np.arange(0, N, 1)
        spl = scipy.interpolate.splrep(xs, PAarray, s=N*5**2)
        PA_func = scipy.interpolate.BSpline(*spl)
        ders = PA_func(xs, 1) 
        count = 0
        for i in range(N):
            count += ((Varray[i]>0) * 2 - 1) * ((ders[i]>0) * 2 - 1)
        count = count / N
        if count > 0.4:
            return 'X'
        elif count < -0.4:
            return 'O'
        else:
            return '?'