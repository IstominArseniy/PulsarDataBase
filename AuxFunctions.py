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


def noise_estimation(data):
    # N = data.shape[0]
    # n5 = N // 10
    # mean = np.max(np.abs(data))
    # noise = 0
    # for i in range(10):
    #     if np.mean(np.abs(data[i * n5: (i+1) * n5])) < mean:
    #         noise = np.sqrt(np.var(data[i * n5: (i+1) * n5]))
    #         mean = np.mean(np.abs(data[i * n5: (i+1) * n5]))
    # return noise
    N = data.shape[0]
    n8 = N // 8
    mean = np.max(np.abs(data))
    sigmas = []
    for i in range(8):
        sigmas.append(np.var(data[i * n8: (i+1) * n8]))
    return np.sqrt(np.min(sigmas))
    
def noise_mean(data):
    N = data.shape[0]
    n5 = N // 5
    std = np.max(data) - np.min(data)
    mean = 0
    for i in range(5):
        if np.sqrt(np.var(data[i * n5: (i+1) * n5])) < std:
            std = np.sqrt(np.var(data[i * n5: (i+1) * n5]))
            mean = np.mean(data[i * n5: (i+1) * n5])
    return mean

def find_loglikelihood(ys, func, xs, noise):
    ans = 0
    for i in range(ys.shape[0]):
        ans += -(ys[i] - func(xs[i]))**2 / 2 / noise**2
    ans += np.log(1/np.sqrt(2 * np.pi) / noise) * ys.shape[0]
    return ans

def find_BIC(ys, func, xs, noise, n_params):
    return n_params * np.log(ys.shape[0]) - 2 * find_loglikelihood(ys, func, xs, noise)

        

def multi_gaus_scale(amps, disps):
    amps = np.array(amps)
    disps = np.array(disps)
    return 1/np.sqrt(2*np.pi)/np.sum(disps * amps)

def gauss_function(x, scale=1, shift=0, disp=1):
    return scale * np.exp(-(x - shift)**2 / 2 / disp**2)

def dbl_gauss(x, scale1, shift1, disp1, scale2, shift2, disp2):            
    return gauss_function(x, scale1, shift1, disp1) + gauss_function(x, scale2, shift2, disp2)

def gaussian_fit(data_array, x_array, peak_number=1):
    scale = 1/np.max(data_array)
    data_array *= scale
    if peak_number == 1:
        popt, pcov = scipy.optimize.curve_fit(gauss_function, x_array, data_array, p0=(1, 0, 1), bounds=([0, np.min(x_array), 0], [2, np.max(x_array), 2]))
        popt[0] /= scale
    elif peak_number == 2:
        popt, pcov = scipy.optimize.curve_fit(dbl_gauss, x_array, data_array, p0=(1, 0, 1, 1, 0, 1), \
                                              bounds=([0, np.min(x_array), 0, 0, np.min(x_array), 0], [2, np.max(x_array), 2, 2, np.max(x_array), 2]))
        popt[0] /=scale
        popt[3] /= scale
    return popt


def profile_fit(data_array, x_array, peak_number=1):
    if peak_number==1:
        return gauss_function(x_array, *gaussian_fit(data_array, x_array, peak_number=1))
    elif peak_number==2:
        return dbl_gauss(x_array, *gaussian_fit(data_array, x_array, peak_number=2))

def inverse_sample_function(dist, Npnts, x_min=-100, x_max=100, n=1e6, **kwargs):
    x = np.linspace(x_min, x_max, int(n))
    cumulative = np.cumsum(dist(x, **kwargs)) 
    cumulative -= cumulative.min()
    f = scipy.interpolate.interp1d(cumulative/cumulative.max(), x)
    return f(np.random.random(int(Npnts)))

def calculate_width(chi, beta, rho):
    dzeta = beta + chi
    if np.sqrt(np.sin((rho + beta)/2) * np.sin((rho - beta)/2) / np.sin(chi) / np.sin(dzeta)) >= 1:
        return np.nan
    return 4 * np.arcsin(np.sqrt(np.sin((rho + beta)/2) * np.sin((rho - beta)/2) \
    / np.sin(chi) / np.sin(dzeta)))

def find_pl_ind(xs, ys):
    log_xs = np.log(xs)
    log_ys = np.log(ys)
    res = scipy.stats.linregress(log_xs, log_ys)
    return res.slope, res.rvalue