#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 05:49:01 2023

@author: shiva
"""

# %% Constants


from scipy import signal, stats
from numpy.fft import rfft, fft, fftfreq
from scipy import optimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
from Oscillation_module import *
import seaborn as sns
import pandas as pd
import sys
import pickle
from scipy.ndimage import gaussian_filter1d
from matplotlib import cm, patches
from mpl_toolkits.mplot3d import Axes3D
from tempfile import TemporaryFile
import timeit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from itertools import chain
import xlsxwriter
import matplotlib.animation as animation
import math
from pygifsicle import optimize
import itertools, random

root = '/home/shiva/BG_Oscillations'
# root = '/Users/apple/BG_Oscillations'
path_lacie = '/media/shiva/LaCie/Membrane_pot_dists'

path = os.path.join(root, 'Outputs_SNN')
path_rate = os.path.join(root, 'Outputs_rate_model')
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
tau_Proto = 12.94
neuronal_consts = {
    
    'Proto': { 
        '12.94' :{
        
        'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_initial': {'min': -65, 'max': -54.84},

          'u_rest': {'mean': -65, 'var': 1. , 'truncmin': -85, 'truncmax': -60}, # fitted to tau_m = 13
         # 'u_rest': {'mean': -65, 'var': .1 , 'truncmin': -85, 'truncmax': -60}, # fitted to tau_m = 4
        # Also similar to traces of Fig 7. Abdi et al. 2015 ~ -65        
        # Abdi et al 2015 Table. 1 : average of "AHP" PV+ and PV- proto is -67.08 ±  6.45 n = 19 juvenile rat (P20-P35) in vitro electric stim temp = 35-37
        # 'u_rest': {'mean': -53, 'var': 4 , 'truncmin': -1000, 'truncmax': -45}, 
        # Bugaysen et al. 2010 [type B] Vm (rest?) -53 ±  4 n= 24 temp = 34 (P12-P21)
        # 'u_rest': {'mean': -66.3, 'var': 3 , 'truncmin': -80, 'truncmax': -50}, 
        # Stanford & cooper 2000 [type A] -66.3 ±  9.19 sem = 0.8, n = 132 Table 2.  temp = 32-34 in vitro whole cell patch


        # 'membrane_time_constant':{'mean': 4, 'var': 0.1, 'truncmin': 0.2, 'truncmax': 20}, 
        # 'membrane_time_constant':{'mean': 25.4,'var': 6, 'truncmin': 3, 'truncmax': 50}, 
        # Stanford & cooper 2000 [type A] 25.4 ±  11.49 sem = 1.0, n = 132 Table 2. temp = 32-34 in vitro whole cell patch
        # 'membrane_time_constant': {'mean': 43, 'var': 10, 'truncmin': 3, 'truncmax': 100}, 
        # Jerome's measurements temp = 32
        'membrane_time_constant':{'mean':12.94,'var': 1.3, 'truncmin': 2.25, 'truncmax': 45},
          # Karube et al. 2019 Table 1 12.94 ±  10.23 range = (2.25, 79.75) n = 108 in vitro whole cell patch (P30-P65) temp = 32

        'spike_thresh': {'mean': -54.84, 'var': 1.}},   # fitted to tau_m = 13
        # 'spike_thresh': {'mean': -54.84, 'var': .1}},   # fitted to tau_m = 4
        # Abdi et al 2015 Table. 1 : -54.84 ±  7.11 average of PV+ and PV- proto n = 19 juvenile rat (P20-P35) in vitro electric stim temp = 35-37 
        # 'spike_thresh': {'mean': -44, 'var': 5}},  
        # Bugaysen et al. 2010 [type B] -44 ±  5 n = 24 temp = 34 (P12-P21)
        # 'spike_thresh': {'mean': -42.1, 'var': 8}},  
        # Stanford & cooper 2000 [type A] -42.1 ±  8 sem = 0.7, n = 132 Table 2. temp = 32-34 in vitro whole cell patch
        # 'spike_thresh': {'mean':-37.57,'var':4.79}}, 
        #  Karube et al. 2019 Table 1 range = (-24.09, -48.11) n = 108 in vitro whole cell patch (P30-P65) temp = 34

        '43' :{
        
        'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_initial': {'min': -65, 'max': -54.84},
        'u_rest': {'mean': -65, 'var': 2. , 'truncmin': -85, 'truncmax': -60}, # fitted to tau_m = 43
        # Abdi et al 2015 Table. 1 : average of "AHP" PV+ and PV- proto is -67.08 ±  6.45 n = 19 juvenile rat (P20-P35) in vitro electric stim temp = 35-37
        'membrane_time_constant': {'mean': 43, 'var': 10, 'truncmin': 3, 'truncmax': 100}, 
        # Jerome's measurements temp = 32
        'spike_thresh': {'mean': -54.84, 'var': 6.}},   # fitted to tau_m = 13
        # Abdi et al 2015 Table. 1 : -54.84 ±  7.11 average of PV+ and PV- proto n = 19 juvenile rat (P20-P35) in vitro electric stim temp = 35-37 

        '25.4' :{
        
        'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_initial': {'min': -65, 'max': -54.84},
          'u_rest': {'mean': -65, 'var': 1.2 , 'truncmin': -85, 'truncmax': -60}, # fitted to tau_m = 25.4
        # Abdi et al 2015 Table. 1 : average of "AHP" PV+ and PV- proto is -67.08 ±  6.45 n = 19 juvenile rat (P20-P35) in vitro electric stim temp = 35-37
        'membrane_time_constant':{'mean': 25.4,'var': 6, 'truncmin': 3, 'truncmax': 50}, 
        # Jerome's measurements temp = 32
        'spike_thresh': {'mean': -54.84, 'var': 7.11}}   # fitted to tau_m = 25.4
        # Abdi et al 2015 Table. 1 : -54.84 ±  7.11 average of PV+ and PV- proto n = 19 juvenile rat (P20-P35) in vitro electric stim temp = 35-37 
        },
    
    'Arky': {
        
        'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_initial': {'min': -54, 'max': -43},


        'u_rest': {'mean': -70, 'var': 1, 'truncmin': -90, 'truncmax': -60},  
        # Abdi et al 2015 from trace juvenile rat (P20-P35) in vitro electric stim temp = 35-37
        # 'u_rest': {'mean': -54, 'var': 4 , 'truncmin': -1000, 'truncmax': -44}, 
        # Bugaysen et al. 2010 [type C] Vm (rest?) -54 ±  4 n=38 temp = 34 (P12-P21) Table 1
        # 'u_rest': {'mean': -58.1, 'var': 8.93, 'truncmin': -75, 'truncmax': -48},
        # Stanford & cooper 2000 [type B] -58.1, ±  8.93 sem = 1.1, n = 66 Table 2. temp = 32-34 in vitro whole cell patch
         
        # 'membrane_time_constant': {'mean': 36.5, 'var': 10, 'truncmin': 0.5, 'truncmax': 100}, 
        # Jerome's measurements temp = 32
        'membrane_time_constant':{'mean': 19.9,'var': 3, 'truncmin': 2, 'truncmax': 100}, 
        # Stanford & cooper 2000 [type B] 19.9 ±  13 sem = 1.6, n = 66 Table 2. temp = 32-34 in vitro whole cell patch


        'spike_thresh': {'mean': -55.0, 'var': 2}},  
        # Abdi et al 2015 Table. 1 -55 ± 7.64 sem = 1.8 n = 18  (P20-P35) rat in vitro electric stim temp = 35-37
        # 'spike_thresh': {'mean': -43, 'var': 4.5}},  
        # Bugaysen et al. 2010 [type C] -43 ±  4.5 n = 38 temp = 34 (P12-P21) Table 1
        # 'spike_thresh': {'mean':-42.9,'var': 6.5}}, 
        # Stanford & cooper 2000 [type B] -42.9 ±  6.5 sem = 0.8, n = 66 Table 2. temp = 32-34 in vitro whole cell patch

        
    'D2': {
        
        'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_initial': {'min': -76.8, 'max': -50},
        # 'u_rest': {'mean': -78.5 , 'var': 6.26, 'truncmin': -1000, 'truncmax': -52}, 
        # Mahon et al. 2003  -78.5 ± 6.26 sem = 1 n = 20 rats in vitro electrical stim in methods
        # 'u_rest': {'mean': -64.47, 'var': 1.5, 'truncmin': -1000, 'truncmax': -42}, 
        # Planert et al. 2013 Table. 1 sd = 14.25 n = 25 rats in vitro electrical stim temp = 34-35
        'u_rest': {'mean': -76.8 , 'var': 3, 'truncmin': -100, 'truncmax': -55}, 
        # Slaght et al. 2004  -76.8 ± 4.43 sem = 1.4 n = 10 rats in vitro electrical stim GAERS

        # 'u_rest': {'mean': -80.9 , 'var': 4.59, 'truncmin': -1000, 'truncmax': -42}, 
        # Mahon et al. 2001  -80.9 ±  4.59 n = 10 normal rats in vivo electrical stim Table 1
        # 'u_rest': {'mean': -94.1, 'var': 1.6, 'truncmin': -1000, 'truncmax': -42}, 
        # Koos and Tepper 1999  -94.1 ±  1.6 n = 7 rats in vitro electrical stim       


        'membrane_time_constant': {'mean': 4.9, 'var': 0.5, 'truncmin': 2, 'truncmax': 12},  
        # Slaght et al. 2004  4.9 ±  1.58 sem = 0.5 n = 10 rats in vitro electrical stim GAERS
        # 'membrane_time_constant': {'mean': 13.85, 'var': 2, 'truncmin': 0.5, 'truncmax': 100},  
        # Planert et al. 2013 Table. 1 sd = 6.25 n = 25 rats in vitro electrical stim temp = 34-35
        # 'membrane_time_constant': {'mean': 3.8, 'var': 1.5, 'truncmin': 0.5, 'truncmax': 100},  
        # Schulz et al. 2011. 3.8 ±  1.5 rat in vivo temp = 36 electric stim n = 90-74 Table 1. 
        
        # 'spike_thresh': {'mean': -51, 'var': 1.8}}, 
        # Mahon et al. 2003  -51 ±  1.8 sem = 0.6 n = 24 rats in vitro electrical stim
        # 'spike_thresh': {'mean': -41.94, 'var': 1.5}}, 
        # Planert et al. 2013 Table. 1 sd = 3.19 rats in vitro electrical stim temp = 34-35 
        'spike_thresh': {'mean': -50, 'var': 0.63}}, 
        # Slaght et al. 2004  -50 ±  0.63 sem = 0.2 n = 10 rats in vitro electrical stim GAERS
        # 'spike_thresh': {'mean': -51, 'var': 1.8}}, 
        # Koos and Tepper 1999  -47.3 ±  1.3 n = 8 rats in vitro electrical stim  
        
    'FSI': {
        
        'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_initial': {'min': -78.2, 'max': -52.4},
        'u_rest': {'mean': -78.2, 'var': 0.5, 'truncmin': -85, 'truncmax': -60},  
        # Schulz et al. 2011. -78.2 ±  6.2 rat in vivo temp = 36 electric stim n = 9 Table 1. 
        # 'u_rest': {'mean': -79.9, 'var': 3.2, 'truncmin': -1000, 'truncmax': -50},  
        # Kawaguchi 1993. -79.9 ±  3.2 rat in vitro electric stim n = 13 Table 1.
        # 'u_rest': {'mean': -76, 'var': 2.71, 'truncmin': -1000, 'truncmax': -50},  
        # Russo et al. 2013. -76 ±  2.71 sem = 0.4 mice in vitro temp = 32 electric stim n = 46 Table 1. Regular
        'membrane_time_constant': {'mean': 3.1, 'var': 0.3, 'truncmin': 1, 'truncmax': 6},  
        # Schulz et al. 2011. 3.1 ±  1.1 rat in vivo temp = 36 electric stim n = 9 Table 1. 
        # 'membrane_time_constant': {'mean': 9.2, 'var': 1.35, 'truncmin': 0.5, 'truncmax': 100},  
        # Russo et al 2013 9.2 ±  1.35 sem = 0.2 mice in vitro temp = 32 electric stim . n = 46 Table 1. Regular
        'spike_thresh': {'mean': -52.4, 'var': 0.5 }},  
        # Schulz et al. 2011. -52.4 ±  4.6 rat in vivo temp = 36 electric stim n = 9 Table 1. 
        # 'spike_thresh': {'mean': -46, 'var': 2.71 }},  
        # Russo et al 2013 -46 ±  2.71 sem = 0.4 mice in vitro temp = 32 electric stim, n = 46 Table 1. Regular

    'STN': {
        
        'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_initial': {'min': -59, 'max': -50.8},
        'u_rest': {'mean': -59, 'var': 0.5, 'truncmin': -75, 'truncmax': -55},  
        # Paz et al. 2005 GAERS -59 ±  1.2 sem = 0.4 in-vivo intracellular electric stim Fig.6B n = 9 temp = (36.5–37.5°C) sd decreased to match FR dist of De la Crompe 2020/ 
        # 'u_rest': {'mean': -60.3, 'var': 5.12, 'truncmin': -1000, 'truncmax': -46}, 
        # Beurrier et al. 1999 Table 1. n = 41 AHP range = (-50, -76) rat in vitro electric stim temp = 30 ±  2 
        'membrane_time_constant': {'mean': 5.13, 'var': 0.6 , 'truncmin': 2, 'truncmax': 10},  
        # Paz et al. 2005 GAERS 5.13 ±  2.38 sem = 0.97  in-vivo intracellular electric stim Fig.6B n = 6 temp = (36.5–37.5°C) sd decreased to match FR dist of De la Crompe 2020/ 
        # Kita et al. 1983 reports 6 ± 2 n=7  range = (4,9)
        'spike_thresh': {'mean': -50.8, 'var': 0.5}}  
        # Paz et al. 2005 GAERS -50.8 ±  1.5  sem = 0.5 in-vivo intracellular electric stim Fig.6B n = 9 temp = (36.5–37.5°C) sd decreased to match FR dist of De la Crompe 2020/ 
        # 'spike_thresh': {'mean': -41.4, 'var': 4.48}}  
        # Beurrier et al. 1999 Table 1.  n = 41 range = (-34, -54) rat in vitro electric stim temp = 30 ±  2 only single spikes
        }

neuronal_consts['Proto'] = neuronal_consts['Proto'][str(tau_Proto)]
N_sim = 1000
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}

N_Str = 2.79*10**6  # Oorschot 1998
N_real = {'STN': 13560, 'Proto': int( 46000 * 0.7 ),
          'Arky': int( 46000 * 0.25 ), 'GPi': 3200, 'Str': N_Str,
          'D2': int( 0.475 * N_Str), 'D1': int( 0.475 * N_Str), # MSNs make up at least 95% of all striatal cells (Kemp and Powell, 1971)
          'FSI': int( 0.02 * N_Str) }  # Oorschot 1998 , FSI-MSN: (Gerfen et al., 2010; Tepper, 2010)


# Dec 2021 FR rest set to anesthetized
FSI_DD = np.average( np.array([8.7, 10.56, 8, 11.2, 10.3]) ) / 19.78 * 10

A_anesthetized = {'STN': 7,  # De la Crompe (2020) averaged  / Mallet et al. 2008 (Sharott) from Fig 1I ~ 14 ± 3 Hz
                  'Proto': 39.84,  # De la Crompe (2020) averaged
                  'Arky': 14,  # De la Crompe (2020) averaged
                  'FSI': 3.67, #+/- 3.22 Mallet et al. 2005 Table 2 (SW-) sem = ±  1.14 n = 8/ # Sharott et al. 2012 Fig 5A (SW-)
                  'D2': 0.5} #  ± 0.14  Sharott et al. 2017 Fig 8E  (SW-) n=12 / Sharott et al. 2012 Fig 5A  (SW-) n=16 /  1.41 +/- 6.23 Slaght et al. 2004 n = 10 intra cellular/

A_DD_anesthetized = {'STN': 24.54,   # De la Crompe (2020) averaged / Mallet et al. 2008 (Sharrot) from Fig 1I  ~ 32.5 ±  5 Hz
                     'Proto': 21.6, # De la Crompe (2020) averaged / Mallet et al. 2008 Fig 6J reports 13                     
                     'Arky': 12.2, # De la Crompe (2020) averaged / Mallet et al. 2008 Fig 6J reports 19               
                     'FSI': 4.93,  # Xiao et al. 2020 Fig 7B state unknown average of all FSI pre in Fig 7B
                     'D2': 2.8}   # ±  0.47 Sharott et al. 2017 Fig 8E n=46 (SW-)

A_awake = {
            # 'STN': 15,  # Mirzaei et al. 2017 Fig 1C (Before Jan 2022)
           'STN': 10.12, # ± 12.17 inattentive rest, Delaville et al. 2015 Figure 6A n=28, 8 rats
           'Proto': 46,  # Mirzaei et al. 2017 Fig 1F / Dodson et al. 2015 awake mice at rest reports 48.3± 3.4 Hz
           'Arky': 3.6,  # ±  2.8 extrapolated from Mallet et al. 2016 Fig 2A rat in vivo head-fixed n = 6/ Dodson et al. 2015 awake mice at rest reports 9.8± 2.3 Hz
           'FSI': 15.2,  # ± 11.46 Berke et al. 2004 n = 67 rat Fig 1B /  Perk et al. 2015 Fig 4A reports 6 Hz
           'D2': 1.41}  # +/- 6.23 Sandstorm & Rebec 2003 n = 364 intracellular/ 1.1 Mirzaei et al. 2017 Fig 1E n = 100 peak at Nose-out/ Perk et al. 2015 Fig 3A. reports 0.5/ Berke et al. 2004 rat Fig 1B reports 0.6 ± 0.83 for all MSN n= 273

A_DD_awake = {'STN': 10.1 , # ±  2.3   inattentive rest, Delaville et al. 2015 Figure 6A   n=32, 9 rats
              'Proto': 0,
              'Arky': 0,
              'FSI': 0,
              'D2': 0}

A_mvt = {
        # 'STN': 22,  # Mirzaei et al. 2017 Fig 1C (Before Jan 2022)
         'STN': 10.37, # ± 7.93 Delaville et al. 2015 Figure 6A  n=28, 8 rats
         'Proto': 22,  # Mirzaei et al. 2017 Fig 1F
         'Arky': 10,  # Mallet et al. 2016 Fig. 2B
         'FSI': 24.7,  # Berke et al. 2008 Fig. 1B rat n = 36 / Yael et al. 2013 in freely moving rats reports 17.5 ±  10.3 n = 15
         'D2':  1.63}  # Berke et al. 2008  n = 109 Fig. 1B rat # 4 from Mirzaei et al. 2017 Fig 1E /1.9 +/- 1.9 Yael et al. 2013 n = 30 Fig 2B

A_DD_mvt = {
         'STN': 10.1 # ±  1.2 Delaville et al. 2015 Figure 6A   n=32, 9 rats
             }

A_trans_Nico_mice = {'STN': 3.7, 
                     'Proto': 26.48, 
                     'D2': 0.37, 
                     'Arky': 3.5}  

A_trans_Kita_rat = {'STN': 9.38, # Kita & Kita 2011
                    'Proto': 31.16, 
                    'D2': 0.67} 
 
A_induction = {'STN-excitation': {'STN': 1,
                                  'Proto': 10,
                                  'Arky' : 10,
                                  'D2': A_anesthetized['D2'],
                                  'FSI': A_anesthetized['FSI']},
               'Proto-inhibition': {'STN': 2,
                                    'Proto': 17,
                                    'Arky': 3.6,
                                    'D2': A_anesthetized['D2'],
                                    'FSI': A_anesthetized['FSI']}}

Arky_A_mean = np.average([3.5, .66, 1.6, 1.0, .4, .85]) / 37.32 * 100
Arky_A_sd = np.std(np.array([3.5, .66, 1.6, 1.0, .4, .85]) / 37.32 * 100)


A = A_anesthetized

Act = {'rest': A_anesthetized, 'awake_rest': A_awake, 'mvt': A_mvt,
       'DD_anesth': A_DD_anesthetized, 'trans_Nico_mice': A_trans_Nico_mice,
       'trans_Kita_rat': A_trans_Kita_rat, 
       'induction_STN_excitation': A_induction['STN-excitation'],
       'induction_Proto_inhibition': A_induction['Proto-inhibition']}

# print( 'Proto thresholds average of PV± ', mean_std_multiple_sets(np.array( [-56.6, -49.9]) , 
#                                                                    np.array( [1.8, 2.8]), 
#                                                                    np.array( [14, 5]), 
#                                                                   sem_instead_of_std = True) )
# print( 'Proto mean membrane pot average of PV± ', mean_std_multiple_sets(np.array( [-67.3, -66.5]) , 
#                                                                np.array( [1.8, 2.8]), 
#                                                                np.array( [14, 5]) , 
#                                                                sem_instead_of_std = True) )
# # Abdli et al 2015 Table. 1 : average of PV+ and PV- proto
# print( 'STN thresholds average of bursting and single spikes', mean_std_multiple_sets(np.array( [-50.4, -49.1, -41.4]) , 
#                                                                                       np.array( [1, 1.2, 0.7]), 
#                                                                                       np.array( [20, 12, 41]), 
#                                                                                       sem_instead_of_std = True) )                                                                                      
# print( 'STN u rest average of bursting and single spikes', mean_std_multiple_sets(np.array( [-61.8, -60.8, -60.3]) , 
#                                                                                   np.array( [0.8, 0.9, 0.8]), 
#                                                                                   np.array( [20, 12, 41]), 
#                                                                                   sem_instead_of_std = True) )

# print( 'Mean std D2 FR awake rest from Slaght et al. 2004 = ',mean_std_multiple_sets(np.array( [0, 4.85]), 
#                                                                                      np.array( [0, 8.5]), 
#                                                                                      np.array( [264, 100])))
tau = {
    ('D2', 'FSI'): {'rise': {'mean': [1.5], 'sd' : [2.9], 'truncmin': [0.6], 'truncmax': [3]}, 
                    'decay': {'mean' : [11.4], 'sd': [2.1], 'truncmin': [0.5], 'truncmax': [30]}},  
    # Koos et al. 2004 rat in vitro electric stm temp= RT or 35. rise: 1.5 ±  2.9 range = (0.6, 3) and decay: 11.4 ±  2.1 .
    # Straub et al. 2016 Fig 1E. mice in vitro optogenetic temp = 33-34 14.4 ±  0.5 n = 41/15  Before Dec 2021
    # Gittis et. al 2010 decay = 7.6 +/ 2.3 Fig 7E range estimated from fig mice in vitro electric stim temp = 31-33
    ('STN', 'Proto'): {'rise': {'mean' : [1.1], 'sd' : [0.4], 'truncmin': [0.8], 'truncmax': [1.6]},
                       'decay' : {'mean' : [7.8], 'sd' : [4.4], 'truncmin': [4.6], 'truncmax': [18.4]}},
    # Baufreton et al. 2009 rise = 1.1 ±  0.4 range = (0.8, 1,6), decay = 7.8 ±  4.4 range = (4.6, 18.4) n = 8 temp = 37
    # Fan et. al 2012 decay = 6.48 ±  1.92 n = 26 temp not mentioned (possibly RT)
    # ('STN','Proto'): {'rise':[1.1, 40],'decay':[7.8, 200]},  # Baufreton et al. 2009, decay=6.48 ±  1.92 n = 26 temp not mentioned Fan et. al 2012
    ('Proto', 'STN'): {'rise': {'mean' : [0.6], 'sd' : [0.2], 'truncmin': [0.1], 'truncmax': [10]},
                        'decay' : {'mean' : [1.81], 'sd' : [2.5], 'truncmin': [0.43], 'truncmax': [6.86]}},
    # decay Jerome measurements in Asier et al. 2021
                           # 'decay' : {'mean' : [6], 'sd' : [1], 'truncmin': [0.1], 'truncmax': [1000]}}, # Glut estimate
    ('Proto', 'Proto'): {'rise': {'mean' : [0.5], 'sd' : [0.15], 'truncmin': [0.1], 'truncmax': [10]},
                          'decay' : {'mean' : [4.91], 'sd' : [1.08], 'truncmin': [0.1], 'truncmax': [30]}},
    # Sims et al. 2008 rat in vitro electric stim (KCl-based electrode) temp = 32 n = 14 ( in thesis it was 2 and 10)

    ('Proto', 'D2'): {'rise': {'mean' : [0.8], 'sd' : [0.22], 'truncmin': [0.2], 'truncmax': [10]},
                      'decay' : {'mean' :[6.13], 'sd' : [1.42], 'truncmin': [0.5], 'truncmax': [30]}},
    # no distiction in GP. GABA-a Sims et al. 2008 rat in vitro electric stim (KCl-based electrode) temp = 32 n = 14,
    # Asier 2021  extrapolated from one trace rise = 0.43, deacy = 6.77
    ('FSI', 'Proto'): {'rise': {'mean' : [1.1], 'sd' : [0.4], 'truncmin': [0.2], 'truncmax': [10]},
                       'decay' : {'mean' : [7], 'sd' : [1], 'truncmin': [0.5], 'truncmax': [30]}},
    # UPDATE 17 Jan: similar to Proto-STN connection according to Jerome/Nico so changed to that
    # (Was 1, 15 between Sep to Dec) Saunders et al. 2016 extrapolated from trace in Fig 4G mice in vitro optogenetic temp = room temerature 
    # (estimate was 6 before Sep 2021)
    ('FSI', 'STN'): {'rise': {'mean' : [0.6], 'sd' : [0.2], 'truncmin': [0.1], 'truncmax': [10]},
                        'decay' : {'mean' : [1.81], 'sd' : [2.5], 'truncmin': [0.43], 'truncmax': [6.86]}},
    # decay Jerome measurements in Asier et al. 2021
    ('Arky', 'Proto'): {'rise': {'mean' : [0.5], 'sd' : [0.15], 'truncmin': [0.1], 'truncmax': [10]},
                          'decay' : {'mean' : [4.91], 'sd' : [1.08], 'truncmin': [0.1], 'truncmax': [30]}},
    # no distiction in GP. GABA-a Sims et al. 2008 rat in vitro electric stim (KCl-based electrode) temp = 32
    ('D2', 'Arky'): {'rise': {'mean' : [1.], 'sd' : [0.5], 'truncmin': [0.2], 'truncmax': [10]},
                     'decay' : {'mean': [28], 'sd' : [5], 'truncmin': [0.5], 'truncmax': [30]}},
    # in vitro Jerome now. Before: 65 was measured from Glajch et al. 2016 [Fig. 2]. They report >200ms
    # ('D1','D2'):{'rise':[3],'decay':[35]}, # Straub et al. 2016
}

T = {
    # ('STN', 'Proto'): 4, # Fujimoto & Kita (1993) - [firing rate] Before Dec 2021
    ('STN', 'Proto'): {'mean': 1.3, 'sd' : 0.3, 'truncmin': 0.8, 'truncmax': 2.5},
    # 1.3 ±  0.3 range = (0.8,2.5) Kita et al. (1983) Fig 5G. rat n=102 in vivo electric stim
    ('Proto', 'STN'): {'mean': 2.8, 'sd' : 0.6, 'truncmin': 2, 'truncmax': 4.4} ,
    # kita & Kitai (1991) rat in vivo electrric stim. temp = 37, n = 18 range =(2, 4.4) Fig 7 sd calculated as range/4
    # Before Dec 2021: Ketzef & Silberberg 2020 mice in vivo optogenetic temp = 36.5 reports 4.75 ±  0.14
    # Fujimoto & Kita (1983) Fig 3C. reports 1.2 ms antidromic response with GP stim rat in vivo n=72 
    ('Proto', 'Proto'): {'mean': 4.76, 'sd' : 0.88, 'truncmin': 3.05, 'truncmax': 7.55}, 
    # Ketzef & Silberberg (2020) Fig 3H 5.12 ±  0.88 range=(3.5,8) n = 5 mice in vivo optogenetic temp = 36.5 Proto ChR2 onset = 0.42 ±  0.05 n = 20 Fig 3H
    # Bugaysen et al. 2013 reports 0.96 ±  0.27 n=4 in vitro electric stim temp = 34
    ('Arky', 'Proto'): {'mean': 4.55, 'sd' : 0.54, 'truncmin': 2.55, 'truncmax': 7.05},     
    # Ketzef & Silberberg (2020) Fig 3H 5 ±  0.54 range=(3,7.5) n = 7 range from fig mice in vivo optogenetic temp = 36.5 Proto ChR2 onset = 0.42 ±  0.05 n = 20 Fig 3H
    ('D2', 'Arky'): {'mean': 4.9, 'sd' : 0.6, 'truncmin': 3.8, 'truncmax': 7.7},     
    # Glajch et al. 2016 Fig. 2/ Table 2 4.9 ±  0.6 n= 12 mice in vitro optogenetic temp: 20-22 . estimate was 7 before Sep 2021.
    ('FSI', 'Proto'): {'mean': 4.3, 'sd' : 0.7, 'truncmin': 3.2, 'truncmax': 7.},     
    # Glajch et al. 2016 Fig. 2/ Table 2 mice  4.3 ±  0.7 n= 17 range from fig in vitro optogenetic temp: 20-22. estimate was 6 before Sep 2021.
    ('FSI', 'STN'): {'mean': 3.9, 'sd' : 0.21, 'truncmin': 2, 'truncmax': 10.}, 
    # Kondabolu et al. 2020
    # ('Proto', 'D2'): {'mean': 7.4, 'sd' : 1.3, 'truncmin': 5.8, 'truncmax': 9.9},
    # Park et al. (1982) 7.4 ± 1.3 range = (5.8, 9.9) n = 22 rats in vivo
    ('Proto', 'D2'): {'mean': 6.89, 'sd' : 0.6, 'truncmin': 5.8, 'truncmax': 9.9},
    # Ketzef & Silberberg (2020) Fig 6K 6.89 ±  0.35 range=(4.3, 11.3) n = 27 range from fig mice in vivo optogenetic temp = 36.5/ corrected for D2-ChR2 onset delay
    # Kita & Kitai (1991) range = (2.2, 11.8), n = 33 in-vivo electric stim temp = 37    
    ('D2', 'FSI'): {'mean': 0.93, 'sd' : 0.29, 'truncmin': 0.8, 'truncmax': 2},  
    # Gittis et al 2010 mice in vitro electric stim temp = 31-33 sd = 0.29, range = (0.8, 2)
    ('STN', 'Ctx'): {'mean': 3, 'sd' : 1.3, 'truncmin': 1, 'truncmax': 10}, 
    # Fujimoto & Kita 1993 3 ±  1.3 ipsi Sensorimotor Ctx stim
    ('D2', 'Ctx'): {'mean': 3, 'sd' : 1.3, 'truncmin': 1, 'truncmax': 10}} 
    # Jaeger & Kita 2016 assumes it's similar to Ctx-STN
# #       ('D2', 'Ctx'): 10.5, # excitation of MC--> Str Kita & Kita (2011) - [firing rate]
# #       # ('FSI', 'Ctx'): 8/12.5 * 10.5 ,# Kita & Kita (2011) x FSI/MSN latency in SW- Mallet et al. 2005
# #       ('FSI', 'Ctx') : 7.5, # Based on Fig. 2A of Mallet et. al 2005 (average of MC-stim (80-400 micA))
# #       ('GPi', 'D1'): 7.2, #  Kita et al. 2001 - [IPSP] / 13.5 (MC-GPi) early inhibition - 10.5 = 3? Kita et al. 2011
# #       ('GPi', 'STN'): 1.7, #  STN-EP Nakanishi et al. 1991 [EPSP] /1ms # STN-SNr Nakanishi et al 1987 / 6 - 5.5  (early excitaion latency of MC--> GPi Kita & Kita (2011) - Ctx-STN) - [firing rate]
# #       ('Arky', 'STN'):  4.35, # Ketzef & Silberberg 2020 #  kita & Kitai (1991) - [firing rate]  reports 2ms
# #       ('GPi', 'Proto'): 3, # Kita et al 2001 --> short latency of 2.8 and long latency 5.9 ms [IPSP]/ (4 - 2) ms Nakanishi et al. 1991: the IPSP following the EPSP with STN activation in EP, supposedly being due to STN-Proto-GPi circuit?
# #       ('Th', 'GPi'): 5, # Xu et al. (2008)
# #       ('D1' , 'FSI'): 1, #0.84 ms mice Gittis et al 2010
# #       ('FSI' , 'FSI'): 1, # estimate based on proximity
# #       ('Ctx','Th'): 5.6, # Walker et al. (2012)
# #       ('D1', 'D2'): 1,
# #       ('D2', 'D2'): 1}

# Gerstner. synaptic time scale for excitation and inhibition
decay_time_scale = {'GABA-A': 6, 'GABA-B': 200,
                    'Glut': 5, 'AMPA': 1.8, 'NMDA': 51}

synaptic_time_constant = {}




bins = {'D2': {'rest' : {'max': 10,
                         'step': 0.3},
               'awake_rest' : {'max': 10,
                               'step': 0.1},
               'DD_anesth' : {'max': 16,
                               'step': 0.5},
               'mvt' : {'max': 16,
                        'step': 0.5}},
        
        'Proto': { 'rest' : {'max': 140,
                            'step': 5},
                    'awake_rest' : {'max': 140,
                                    'step': 5},
                    'DD_anesth' : {'max': 100,
                                    'step': 4},
                    'mvt' : {'max': 100,
                             'step': 3}},
        
        'FSI': { 'rest' : {'max': 15,
                           'step': 0.5},
                 'awake_rest' : {'max': 60,
                                 'step': 2},
                 'DD_anesth' : {'max': 15,
                                 'step': 0.5},
                 'mvt' : {'max': 80,
                          'step': 2}},
        
        'STN': { 'rest' : {'max': 60,
                           'step': 2},
                'awake_rest': {'max': 60,
                               'step': 2},
                'DD_anesth': {'max': 60,
                               'step': 2},
                'mvt': {'max': 60,
                        'step': 2}},
        
        'Arky': { 'rest' : {'max': 60,
                           'step': 2},
                'awake_rest': {'max': 20,
                               'step': 0.5},
                'DD_anesth': {'max': 60,
                               'step': 2},
                'mvt': {'max': 60,
                        'step': 2}},
        
        }


noise_variance = {
    
     'rest' : {
                # 'Proto': 23, # tau_m = 43
                # 'Proto': 16, # tau_m = 25.4
                # 'Proto': 9, # tau_m = 4
                'Proto': 12, # tau_m = 12.94
               'STN': 5., 
               'FSI':  18, 
               # 'D2': 16, # FR = 1.4
               'D2': 15, # FR = 0.5
               'Arky': 14} , # tau_m = 19.9
     
      'awake_rest' : {
                      # 'Proto': 18, # tau_m = 25.4
                      'Proto': 13, # tau_m = 12.94
                      'STN': 5, 
                      'FSI': 19, 
                      'D2': 16, # tau_m = 4.9
                      'Arky': 15},
      
      'DD_anesth' : {
                      # 'Proto': 12, # tau_m = 25.4
                      'Proto': 10, # tau_m = 12.94
                      'STN': 6, 
                      'FSI': 18, 
                      'D2': 17, 
                      'Arky': 14} ,# tau_m = 19.9
      
      'mvt' : {
              # 'Proto': 12,  # tau_m = 25.4
              'Proto': 10,  # tau_m = 12.94
               'STN': 6, 
               'FSI': 21, 
               'D2': 17, 
               'Arky': 14}, # tau_m = 19.9
      
      # Nico's in mice
      'trans_Nico_mice' : {
                          # 'Proto': 21, # tau_m = 25.4
                          'Proto': 10, # tau_m = 12.94
                          # 'Proto': 6, # tau_m = 4
                          'STN': 5, 
                           # 'FSI': 29.5, 
                           'D2': 14.5, 
                           'Arky': 15},
      
      # Kia & Kita 2011 rats
     'trans_Kita_rat' : {
                          # 'Proto': 14, # tau_m = 25.4
                         # 'Proto': 11, # tau_m = 12.94
                          'Proto': 8, # tau_m = 4
                          'STN': 5, 
                         'D2': 15, 
                         'Arky': 15},
     'induction_STN_excitation' : {
                              'Proto': 8, # tau_m = 4
                              'STN': 4, 
                              'FSI':  18, 
                             'D2': 15, 
                             'Arky': 14},
     'induction_Proto_inhibition' : {
                              'Proto': 9, # tau_m = 4
                              'STN': 4, 
                              'FSI':  18, 
                             'D2': 15, 
                             'Arky': 15}
                             }
FR_ext_range = {
 
    ### tau_m = 4
    # 'Proto': {'rest': np.array([18/1000, 22/1000]), 'awake_rest': np.array([5/1000, 10/1000]), 
    #           'DD_anesth': [1./300, 3./300], 'mvt': [1/300, 3/300],  
    #           'trans_Nico_mice': [3/300, 6/300],  'trans_Kita_rat': [18/1000, 22/1000]},
    
    # ### tau_m = 13
    'Proto': {'rest': np.array([7/1000, 12/1000]), 'awake_rest': np.array([7/1000, 12/1000]), 
              'DD_anesth': [6/1000, 11/1000], 'mvt': [6/1000, 11/1000],  
              'trans_Nico_mice': [2/300, 4/300],  'trans_Kita_rat': [5/1000, 10/1000],
              'induction_STN_excitation' : np.array([4/1000, 8/1000]),
              'induction_Proto_inhibition' : np.array([5/1000, 8/1000])},
    
    ### tau_m = 25
    # 'Proto': {'rest': np.array([5/1000, 10/1000]), 'awake_rest': np.array([5/1000, 10/1000]), 
    #           'DD_anesth': [1./300, 3./300], 'mvt': [1/300, 3/300],  
    #           'trans_Nico_mice': [1/300, 2/300],  'trans_Kita_rat': [4/1000, 9/1000]},
    
    ## tau_m = 43
    # 'Proto': {'rest': np.array([3/1000, 8/1000]), 'awake_rest': np.array([5/1000, 10/1000]), 
    #           'DD_anesth': [1/300, 2/300], 'mvt': [1/300, 2/300],  
    #           'trans_Nico_mice': [1/300, 2/300],  'trans_Kita_rat': [1/300, 2/300]},
    
    'STN': {'rest': np.array([8/1000, 11/1000]), 'awake_rest': np.array([9.5/1000, 11.5/1000]), 
            'DD_anesth': [5/1000, 20/1000], 'mvt':  [9/1000, 12/1000], 
            'trans_Nico_mice': np.array([8.5/1000, 12/1000]), 'trans_Kita_rat': np.array([8/1000, 11/1000]),
              'induction_STN_excitation' : np.array([6/1000, 9/1000]),
              'induction_Proto_inhibition' : np.array([6/1000, 9/1000])}, # high FR sigma
   
    ### tau_m = 3
    'FSI': {'rest': np.array([48/1000, 65/1000]), 'awake_rest': np.array([61/1000, 68/1000]), 
            'DD_anesth': [50/1000, 68/1000], 'mvt':[63/1000, 75/1000],
              'induction_STN_excitation' : np.array([6/1000, 9/1000]),
              'induction_Proto_inhibition' : np.array([6/1000, 9/1000])},
    
    ### tau_m = 9
    # 'FSI': {'rest': np.array([22/1000, 30/1000]), 'awake_rest': np.array([28/1000, 32/1000]), 
    #         'DD_anesth': [22/1000, 31.5/1000], 'mvt':[30/1000, 35/1000]},

    ### tau_m = 4.9
    
    'D2': {
            # 'rest': np.array([25/1000, 38/1000]), # FR = 1.4
            'rest': np.array([22/1000, 34/1000]), # FR = 0.5
           'awake_rest': np.array([25/1000, 38/1000]), 
           'DD_anesth': [34/1000, 38/1000], 'mvt': [25/1000, 38/1000], 
           'trans_Nico_mice': np.array([20/1000, 38/1000]), 'trans_Kita_rat': np.array([22/1000, 38/1000]),
              'induction_STN_excitation' : np.array([6/1000, 9/1000]),
              'induction_Proto_inhibition' : np.array([6/1000, 9/1000])},
    
    ### tau_m = 13 ms
    # 'D2': {'rest': np.array([10/1000, 20/1000]), 'awake_rest': np.array([7/1000, 15/1000]), 
    #        'DD_anesth': [10/1000, 15/1000], 'mvt': [10/1000, 15.5/1000], 
    #        'trans_Nico_mice': np.array([6/1000, 13/1000]), 'trans_Kita_rat': np.array([6/1000, 13/1000])},
    
    ### tau_m = 19.9
    
    'Arky': {'rest': np.array([7/1000, 9/1000]), 'awake_rest': np.array([4/1000, 7/1000]), 
             'DD_anesth': [7/1000, 9/1000], 'mvt': [6/1000, 8/1000], 
             'trans_Nico_mice': np.array([3/1000, 5/1000]),
              'induction_STN_excitation' : np.array([6/1000, 9/1000]),
              'induction_Proto_inhibition' : np.array([6/1000, 9/1000])}
    
    ### tau_m = 36
    # 'Arky': {'rest': np.array([5/1000, 7/1000]), 'awake_rest': np.array([3/1000, 5/1000]), 
    #      'DD_anesth': [5/1000, 7/1000], 'mvt': [4/1000, 7/1000], 
    #      'trans_Nico_mice': np.array([3/1000, 5/1000])}
                }


FR_ext_sd_dict = {   
    # 'Proto': {'rest' : 0.0005, 'awake_rest' : 0, 'DD_anesth' : 0.0007, 'mvt' : 0.00001, 'trans_Nico_mice' : 0,'trans_Kita_rat' : 0.001}, # tau_m = 25
    'Proto': {'rest' : 0, 'awake_rest' : 0, 'DD_anesth' : 0.0007, 'mvt' : 0.00001, 
              'trans_Nico_mice' : 0, 'trans_Kita_rat' : 0.001,
              'induction_STN_excitation': 0, 'induction_Proto_inhibition' : 0}, # tau_m = 13
    'STN': {'rest' : 0.0008, 'awake_rest' : 0.0018, 'DD_anesth' : 0, 'mvt' : 0.001,
            'trans_Nico_mice' : 0,'trans_Kita_rat' : 0.001,
            'induction_STN_excitation': 0.0001, 'induction_Proto_inhibition' : 0.0001},
    'FSI': {'rest' : 0, 'awake_rest' : 0, 'DD_anesth' : 0, 'mvt' : 0, 
            'trans_Nico_mice' : 0,'trans_Kita_rat' : 0,
            'induction_STN_excitation': 0, 'induction_Proto_inhibition' : 0},
    'D2': {
            # 'rest' : 0.002, # FR = 1.4
            'rest' : 0.00001, # FR = 0.5
           'awake_rest' : 0.00001, 'DD_anesth' : 0.0001, 'mvt' : 0.0001, 
           'trans_Nico_mice' : 0,'trans_Kita_rat' : 0.001,
           'induction_STN_excitation': 0.00001, 'induction_Proto_inhibition' : 0.00001},
    'Arky': {'rest' : 0.0008, 'awake_rest' : 0.001, 'DD_anesth' : 0.000001, 'mvt' : 0.0005, 
             'trans_Nico_mice' : 0,'trans_Kita_rat' : 0,
             'induction_STN_excitation': 0.00001, 'induction_Proto_inhibition' : 0.0001}
                }
    
FR_ext_specs = { name : { state : {'mean': 0, 'sd': sd, 'truncmin': 0, 'truncmax': 100}
                         for state, sd in sd_dict.items()} 
                for name, sd_dict in FR_ext_sd_dict.items() }
noise_amplitude = {'Proto': 1, 'STN': 1, 'D2': 1, 'FSI': 1, 'Arky': 1}

end_of_nonlinearity = {
    
    'FSI': {'rest': 10, 'awake_rest': 10, 'mvt': 20, 'DD_anesth': 10, 'trans_Nico_mice': 20, 'trans_Kita_rat': 20,
            'induction_STN_excitation': 10, 'induction_Proto_inhibition': 10},
    'D2':  {'rest': 10, 'awake_rest': 10, 'mvt': 10, 'DD_anesth': 10, 'trans_Nico_mice': 10, 'trans_Kita_rat': 10,
            'induction_STN_excitation': 10, 'induction_Proto_inhibition': 10},
    'Proto':  {'rest': 20, 'awake_rest': 20, 'mvt': 20, 'DD_anesth': 20, 'trans_Nico_mice': 20, 'trans_Kita_rat': 20,
               'induction_STN_excitation': 20, 'induction_Proto_inhibition': 10},
    'STN': {'rest': 20, 'awake_rest': 15, 'mvt': 20, 'DD_anesth': 20, 'trans_Nico_mice': 20, 'trans_Kita_rat': 20,
            'induction_STN_excitation': 10, 'induction_Proto_inhibition': 10},
    'Arky':  {'rest':  10, 'awake_rest': 15, 'mvt': 25, 'DD_anesth': 10, 'trans_Nico_mice': 15, 'trans_Kita_rat': 15,
              'induction_STN_excitation': 10, 'induction_Proto_inhibition': 10}
                        }
threshold = {'STN': .1, 'Proto': .1, 'D2': .1, 'FSI': .1, 'Arky': 0.1}
neuron_type = {'STN': 'Glut', 'Proto': 'GABA', 'D2': 'GABA', 'FSI': 'GABA'}
gain = {'STN': 1, 'Proto': 1, 'D2': 1, 'FSI': 1, 'Arky': 1}

syn_per_ter = {('STN', 'Proto'): round(12/10.6),  # number of synapses per bouton = 12/10.6, #Baufreton et al. (2009)
               ('Proto', 'Proto'): 10,  # Sadek et al. 2006
               ('FSI', 'Proto'): 1,
               ('D1', 'FSI'): 1,
               ('D2', 'FSI'): 1,
               ('FSI', 'FSI'): 1,
               ('Proto', 'D2'): 1,
               ('GPi', 'D1'): 1,
               ('GPi', 'STN'): 1,
               ('GPi', 'Proto'): 1,
               ('D1', 'D2'): 1,
               ('D2', 'D2'): 1}

Str_connec = {('D1', 'D2'): .28, ('D2', 'D2'): .36,  ('D1', 'D1'): .26,
              ('D2', 'D1'): .05, ('MSN', 'MSN'): 1350}  # Taverna et al 2008

# Before June 2022
K_real_bouton = {
          # 243 bouton per GP. number of synapses per bouton = 12/10.6  Baufreton et al. 2009 & Sadek et al. (2006).
          ('STN', 'Proto'): round(243 * N_real['Proto'] / N_real['STN']), 
          # boutons Kita & Jaeger (2016) based on Koshimizu et al. (2013)
          ('Proto', 'STN'): 135,
          # Sadek et al. 2006 --> lateral 264, medial = 581 boutons mean 10 syn per bouton. 650 boutons Hegeman et al. (2017)
          ('Proto', 'Proto'): round((264 + 581) / 2),
          # averaging the FSI contacting of Proto boutons Bevan 1998
          ('FSI', 'Proto'): 360,
          # Guzman et al  (2003): 240 from one class interneuron to each MSI # 53% (FSI-D2) Gittis et al.2010
          ('FSI', 'STN'): 150, # estimate
          ('D2', 'FSI'): round(53 * 2/ (36 + 53) * 240),
          #           ('Proto', 'D2'): int(N_Str/2*(1-np.power(0.13,1/(.1*N_Str/2)))), # Chuhma et al. 2011 --> 10% MSN activation leads to 87% proto activation
          # each Proto 226 from iSPN Kawaguchi et al. (1990)
          ('Proto', 'D2'): round(N_real['D2'] * 226 / N_real['Proto']),
          # each Proto 123 from dSPN Kawaguchi et al. (1990)
          ('Proto', 'D1'): round(N_real['D1'] * 123/ N_real['Proto']),          
          ('GPi', 'STN'): 457, # boutons Kita & Jaeger (2016) based on Koshimizu et al. (2013)
          # ('D1', 'D2'): int(Str_connec[('MSN','MSN')]*Str_connec[('D1', 'D2')]/(Str_connec[('D1', 'D2')]+Str_connec[('D2', 'D2')])), #Guzman et al (2003) based on Taverna et al (2008)
          # ('D2', 'D2'): int(Str_connec[('MSN','MSN')]*Str_connec[('D2', 'D2')]/(Str_connec[('D1', 'D2')]+Str_connec[('D2', 'D2')])),
          ('D2', 'Arky'): 100,  # estimate
          ('Arky', 'Proto'): round((264 + 581) / 2)}  
          # averaging the FSI contacting of Proto boutons Bevan 1998

#           ('D1', 'D1'): Str_connec[('MSN','MSN')]*Str_connec[('D1', 'D1')]/(Str_connec[('D1', 'D1')]+Str_connec[('D2', 'D1')]),
#           ('D2', 'D1'): Str_connec[('MSN','MSN')]*Str_connec[('D2', 'D1')]/(Str_connec[('D1', 'D1')]+Str_connec[('D2', 'D1')]), #Guzman et al (2003) based on Taverna et al (2008)
#           ('D1', 'Proto'): int(N_real['Proto']*(1-np.power(64/81, 1/N_real['Proto'])))} # Klug et al 2018
# n_bouton_per_neuron = 10
# K_real = {
#     ('STN', 'Proto'): round(0.02 * N_real['Proto'] / n_bouton_per_neuron),
#     #= 65 Baufreton et al. 2009 single STN neuron receives input maximally from 2%  
#     ('Proto', 'STN'): round(188 / n_bouton_per_neuron), 
#     # boutons Kita & Jaeger (2016) based on Koshimizu et al. (2013)
#     ('Proto', 'Proto'): 35, 
#     # Sadek et al. 2006
#     ('FSI', 'Proto'): round(360/ n_bouton_per_neuron),
#     # averaging the FSI contacting of Proto boutons Bevan 1998
#     ('FSI', 'STN'): 15, 
#     # estimate
#     ('D2', 'FSI'): 16,
#     # ~ 16 FSI converge to a single MSN Rat (Koos & Tepper 1999)
#     ('Proto', 'D2'): round(N_real['D2'] * 226 / N_real['Proto']/ n_bouton_per_neuron),
#     # number of boutons in the GPe formed by
#     # an axon of an indirect projection neuron is 226 (Koshimizu et al., 2013)
#     ('D2', 'Arky'): round(100/ n_bouton_per_neuron),  # estimate
#     ('Arky', 'Proto'): 35}  # estimate
#     # Sadek et al. 2006
n_bouton_per_neuron = 3

K_real = {
    ('STN', 'Proto'): round(275 * N_real['Proto'] / N_real['STN'] / n_bouton_per_neuron),
    #= 218, Baufreton et al. 2009 An individual GP neuron forms about 275 synapses in the STN \cite{Baufreton2009}
    ('Proto', 'STN'): round(558 * 0.8 * N_real['STN'] / N_real['Proto'] / n_bouton_per_neuron), 
    # = 63, 8/10 neurons have on average 558 boutons Koshimizu et al. (2013)
    ('Proto', 'Proto'): round(442 * N_real['Proto'] / (N_real['Proto'] + N_real['Arky']) / n_bouton_per_neuron), 
    # = 104 Sadek et al. 2006
    ('FSI', 'Proto'): round(791 * 0.44 * N_real['Proto'] / N_real['FSI'] / n_bouton_per_neuron),
    # = 67, averaging the FSI contacting of Proto boutons Bevan 1998
    ('FSI', 'STN'): round(157 * 0.8 * N_real['STN'] / N_real['FSI'] / n_bouton_per_neuron), 
    # = 10, Koshimizu et al., 2013 8/10 STN neurons send projections to the CPu and each made 31 to 460 presynaptic boutons (on average 157).
    ('D2', 'FSI'): round(2840 * 0.05 * 0.36),
    # = 51, There are 2840 striatal neurons inside the volume of a spiny dendritic tree
    # (Oorschot, 1996; Kincaid et al., 1998). 5% of which are FSIs n = 140 (Koos and Tepper, 1999), P_connection = 36% Gitties et al. 2011
    ('Proto', 'D2'): round(226 * N_real['D2'] / N_real['Proto']/ n_bouton_per_neuron),
    # = 3100, Koshimizu et al., 2013 an axon of an indirect projection neuron is 226.
    ('D2', 'Arky'): 10,
    # estimate
    ('Arky', 'Proto'): round(442 * N_real['Proto'] / (N_real['Proto'] + N_real['Arky']) / n_bouton_per_neuron)}
    # = 104 Sadek et al. 2006
    
K_real_DD = {
    ('D2', 'FSI'): 2 * K_real[('D2', 'FSI')],
    ('Proto', 'D2'): K_real[('Proto', 'D2')],
    ('FSI', 'Proto'): K_real[('FSI', 'Proto')], 
    ('Proto', 'STN'): K_real[('Proto', 'STN')],
    ('FSI', 'STN'): K_real[('FSI', 'STN')],
    ('STN', 'Proto'): K_real[('STN', 'Proto')],
    ('D2', 'Arky'): K_real[('D2', 'Arky')], 
    ('Arky', 'Proto'): K_real[('Arky', 'Proto')],
    ('Proto', 'Proto'): K_real[('Proto', 'Proto')]}

K_all = {'rest': K_real, 'DD_anesth': K_real_DD, 
         'awake_rest': K_real, 'mvt': K_real,
         'trans_Kita_rat': K_real, 'trans_Nico_mice': K_real}

# K_real_STN_Proto_diverse = K_real.copy()
# K_real_STN_Proto_diverse[('Proto', 'STN')] = K_real_STN_Proto_diverse[('Proto', 'STN')] / N_sub_pop # because one subpop in STN contacts all subpop in Proto

oscil_peak_threshold = {'Proto': 0.1, 'STN': 0.1,
                        'D2': 0.1, 'FSI': 0.1, 'Arky': 0.1}

smooth_kern_window = {key: value * 30 for key, value in noise_variance['rest'].items()}
#oscil_peak_threshold = {key: (gain[key]*noise_amplitude[key]*noise_variance[key]-threshold[key])/5 for key in noise_variance.keys()}

syn_coef_GABA_b = 1

syn_component_weight = {    # the relative weight of the GABA-a and GABA-b components
    ('D2', 'FSI'): [1],
    ('STN', 'Proto'): [1, syn_coef_GABA_b],
    ('Proto', 'STN'): [1],

    ('Proto', 'Proto'): [1, syn_coef_GABA_b],
    ('Proto', 'D2'): [1],
    ('FSI', 'Proto'): [1],
    ('Arky', 'Proto'): [1, syn_coef_GABA_b],
    ('D2', 'Arky'): [1]}

syn_component_weight = {key: [1] for key in list(tau.keys())}

tau_DD = {('STN', 'Proto'): {'rise': [0.1], 'decay': [7.78]}} # Fan et. al 2012

color_dict = {'Proto': 'r', 'STN': 'k',
              'D2': 'b', 'FSI': 'g', 'Arky': 'darkorange'}
axvspan_color = {'DD_anesth': 'darkseagreen', 'mvt': 'lightskyblue', 
                 'induction': 'plum'}

I_ext_range = {'Proto': [0.175*100, 0.185*100],
               'STN': [0.012009 * 100, 0.02 * 100],
               'D2': [0.0595*100 / 3, 0.0605*100 / 3],
               'FSI': [0.05948*100, 0.0605*100],
               'Arky': []}
ext_inp_delay = 0

# %% save_FR_ext_to_pkl


path = os.getcwd()

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 25800 
t_list = np.arange(int(t_sim/dt))
plot_start = 200
t_transition = plot_start + 0# int(t_sim / 5)
duration_base = np.array( [int(200/dt), int(t_transition/dt)] )
duration_DD = np.array( [int(t_transition / dt) + int(300/dt) , int(t_sim / dt)] ) 
t_mvt = t_sim
D_mvt = t_sim - t_mvt

duration = duration_DD - int( t_transition /dt )
t_sim = t_sim - t_transition
t_list = np.arange(int(t_sim/dt))
# end_phase = t_sim
n_phase_bins = 72
n_windows = int(t_sim - 800) / 1000

name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'
name_list = [name1, name2, name3, name4, name5]

state_1 = 'rest'
state_2 = 'DD_anesth'
G = {}

print( " transition from " , state_1, ' to ', state_2)



g = -0.0025 ## log-normal syn weight dist F = 18.5 Hz
G = { (name2, name1) :{'mean': g * K[name2, name1] * 0},#}, ## free
      (name3, name2) :{'mean': g * K[name3, name2] * 0},#11.}, ## free
      (name1, name3) :{'mean': g * K[name1, name3] * 0},#30 * 66/63}, ## free
      (name2, name4) :{'mean': g * K[name2, name4] * 0},#0.01}, ## free
      (name4, name3) :{'mean': g * K[name4, name3] * 0},
      (name3, name5) :{'mean': -g * K[name3, name5] * 0},
      (name5, name3) :{'mean': g * K[name5, name3] * 0},# 4.7},
      (name3, name3) :{'mean': g * K[name3, name3] * 0}}#2.}}#, 



G = set_G_dist_specs(G,  order_mag_sigma = 1)
G_dict = {k: {'mean': [v['mean'] ]} for k, v in G.items()}

# G_dict = {k: {'mean': np.full(4, v['mean'])} for k, v in G.items()}
# G_dict[(name3, name2)]['mean'] = G_dict[(name3, name2)]['mean'] * np.array([1, 0.7, 0.4, 0.1])

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1'), (name4, '1')],                      
                      (name3, '1'): [(name2, '1'), (name3, '1'), (name5, '1')],
                      (name4, '1'): [(name3, '1')],
                      (name5, '1'): [(name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
mem_pot_init_method = 'uniform'

keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method,Act = Act, state = state_1) for i in pop_list] for name in name_list}

save_FR_ext_to_pkl(nuclei_dict, path, dt = dt)

#%%

path = os.getcwd()
path_lacie = path


plt.close('all')
name_list = ['D2', 'FSI', 'STN', 'Proto', 'Arky']

for name in name_list:

    
    state = 'DD_anesth'
    
    print('desired activity =', Act[state][name])
    save_mem_pot_dist = True
    # save_mem_pot_dist = False
    
    FSI_on_log = False
    N_sim = 1000
    N = dict.fromkeys(N, N_sim)
    dt = 0.1    
    t_sim = 4000
    t_list = np.arange(int(t_sim/dt))
    duration = [int(t_sim/dt/2), int(t_sim/dt)]
    t_mvt = t_sim
    D_mvt = t_sim - t_mvt
    
    G = {}
    receiving_pop_list = {(name, '1'): []}
    
    pop_list = [1]
    
    g = -0.01
    init_method = 'heterogeneous'
    syn_input_integ_method = 'exp_rise_and_decay'
    ext_input_integ_method = 'dirac_delta_input'
    ext_inp_method = 'const+noise'
    mem_pot_init_method = 'draw_from_data'
    mem_pot_init_method = 'uniform'
    keep_mem_pot_all_t = True
    keep_noise_all_t = True
    set_FR_range_from_theory = False
    set_input_from_response_curve = True
    save_init = True
    der_ext_I_from_curve = True
    if_plot = True
    noise_method = 'Gaussian'
    noise_method = 'Ornstein-Uhlenbeck'
    use_saved_FR_ext = False
    use_saved_FR_ext = True
    
    poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
        'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 'g': 0.01}}
    
    class Nuc_keep_V_m(Nucleus):
    
        def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):
            
            self.cal_ext_inp_method_dict [self.external_input_bool](dt, t)
            synaptic_inputs = self.sum_synaptic_input(receiving_from_class_list, dt, t)
            self.update_potential(synaptic_inputs, dt, t, receiving_from_class_list)
            spiking_ind = self.find_and_save_new_spikes(t)
            # self.reset_potential(spiking_ind)
            self.reset_potential_with_interpolation(spiking_ind,dt)
            self.all_mem_pot[:, t] = self.mem_potential
    
        # def cal_ext_inp(self, dt, t):
    
        #     # choose method of exerting external input from dictionary of methods
        #     I_ext = self.ext_inp_method_dict[self.ext_inp_method](dt)
    
        #     self.noise_all_t[:, t] = self.noise.reshape(-1,)
        #     # print(self.rest_ext_input.shape, self.noise.shape)
        #     self.I_syn['ext_pop', '1'], self.I_rise['ext_pop', '1'] = self.input_integ_method_dict[self. ext_input_integ_method](
        #         I_ext, dt,
        #         I_rise=self.I_rise['ext_pop', '1'],
        #         I=self.I_syn['ext_pop', '1'],
        #         tau_rise=self.tau_ext_pop['rise'],
        #         tau_decay=self.tau_ext_pop['decay'])
    
        # def constant_ext_input_with_noise(self, dt):
    
    
        #     self.noise =  self.noise_generator_dict [self.noise_method] (self.noise_amplitude,
        #                                                                  self.noise_std,
        #                                                                  self.n,
        #                                                                  dt,
        #                                                                  self.sqrt_dt,
        #                                                                  tau=self.noise_tau,
        #                                                                  noise_dt_before=self.noise
        #                                                                  )
    
        #     return self.rest_ext_input + self.noise.reshape(-1,)
    
    
    nuc = [Nuc_keep_V_m(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, 
                        N, Act[state], A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                        smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve, 
                        state = state, poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, 
                        mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t, 
                        ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method,
                        path=path_lacie, save_init=save_init, noise_method=noise_method, keep_noise_all_t=keep_noise_all_t,
                        Act  = Act, FR_ext_specs = FR_ext_specs[name][state],
                        plot_spike_thresh_hist= False, plot_RMP_to_APth = False) for i in pop_list]
    
    
     # {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
     #               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
     #               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
     #               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
     #               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state, 
     #               hetero_trans_delay = hetero_trans_delays, hetero_tau = hetero_tau, Act  = Act) for i in pop_list] for name in name_list}
    
    nuclei_dict = {name: nuc}
    nucleus = nuc[0]
    
    # plot_action_potentials(nucleus, n_neuron=1, t_end=5000)
    
    
    n_FR = 20
    all_FR_list = {name: FR_ext_range[name][state]
                   for name in list(nuclei_dict.keys())}
    
    receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, 
                                                           receiving_pop_list, nuclei_dict, t_list, all_FR_list=all_FR_list,
                                                            n_FR=n_FR, if_plot=if_plot, end_of_nonlinearity=end_of_nonlinearity,
                                                           set_FR_range_from_theory=False, method='collective', save_FR_ext=True,
                                                           use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)
    
    nuclei_dict = run(receiving_class_dict, t_list, dt,  {name: nuc})
    if save_mem_pot_dist:
        save_all_mem_potential(nuclei_dict, path, state)
        