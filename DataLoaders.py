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
import AuxFunctions as aux


def load_FAST_data(file_name):
    profile_data = np.array(0)
    with open(file_name, 'r') as file:
        lines = file.readlines()
        profile_data = np.zeros((6, len(lines) - 1))
        for i in range(1, len(lines)):
            ILVPA = list(map(float, lines[i].split(' ')))
            profile_data[0][i-1] = ILVPA[1]
            profile_data[4][i-1] = ILVPA[2]
            profile_data[5][i-1] = ILVPA[4]
            profile_data[3][i-1] = ILVPA[3]
            profile_data[1][i-1] = profile_data[4][i-1] * np.cos(profile_data[5][i-1])
            profile_data[2][i-1] = profile_data[4][i-1] * np.sin(profile_data[5][i-1])
    return profile_data

def load_MeerKAT_mean_data(file_name):
    hdul = fits.open(file_name)
    DM = hdul[4].header['DM']
    DC = DM / 2.41e-16
    Jname = hdul[0].header['SRC_NAME']
    P = hdul[4].header['PERIOD']
    freqs = hdul[4].data[0]['DAT_FREQ'] * 1e6
    N = hdul[4].data[0]['DATA'][0][0].shape[0]
    profile_data = np.zeros((6, N))
    add_shift = (np.argmax(hdul[4].data[0]['DATA'][0][0]) - N//4)
    for channel in range(8):
        for stocks in range(4):
            hdul[4].data[0]['DATA'][stocks][channel] = np.roll(hdul[4].data[0]['DATA'][stocks][channel], int(DC * (1/freqs[0]**2 - 1/freqs[channel]**2) * N / P) - add_shift)
    # TEST global profile shift
    for channel in range(8):
        profile_data[0] += (hdul[4].data[0]['DATA'][0][channel].astype('int64') - aux.noise_mean(hdul[4].data[0]['DATA'][0][channel].astype('int64'))) * hdul[4].data[0]['DAT_SCL'][8*0+channel] * (hdul[4].data[0]['DAT_WTS'][channel] / np.sum(hdul[4].data[0]['DAT_WTS']))
        profile_data[1] += (hdul[4].data[0]['DATA'][1][channel].astype('int64') - aux.noise_mean(hdul[4].data[0]['DATA'][1][channel].astype('int64'))) * hdul[4].data[0]['DAT_SCL'][8*1+channel] * (hdul[4].data[0]['DAT_WTS'][channel] / np.sum(hdul[4].data[0]['DAT_WTS']))
        profile_data[2] += (hdul[4].data[0]['DATA'][2][channel].astype('int64') - aux.noise_mean(hdul[4].data[0]['DATA'][2][channel].astype('int64'))) * hdul[4].data[0]['DAT_SCL'][8*2+channel] * (hdul[4].data[0]['DAT_WTS'][channel] / np.sum(hdul[4].data[0]['DAT_WTS']))
        profile_data[3] += (hdul[4].data[0]['DATA'][3][channel].astype('int64') - aux.noise_mean(hdul[4].data[0]['DATA'][3][channel].astype('int64'))) * hdul[4].data[0]['DAT_SCL'][8*3+channel] * (hdul[4].data[0]['DAT_WTS'][channel] / np.sum(hdul[4].data[0]['DAT_WTS']))
    profile_data[4] = np.sqrt(profile_data[1]**2 + profile_data[2]**2)
    profile_data[4] -= aux.noise_mean(profile_data[4])
    profile_data[5] = 0.5 * np.arctan2(profile_data[2], profile_data[1]) * 180 / np.pi
    # TODO profile_data[5] <-> PA
    return profile_data
    

def load_MeerKAT_channel_data(file_name, channel):
    pass

def load_EPN_data(file_name):
    pass


