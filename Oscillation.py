# %% Constants
from scipy import signal, stats
from numpy.fft import rfft, fft, fftfreq
from scipy import optimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Oscillation_module import *
import seaborn as sns
import pandas as pd
import sys
import pickle
from scipy.ndimage import gaussian_filter1d
from matplotlib import cm
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

# %% De La Crompe FR distributions

xls = pd.ExcelFile(os.path.join(root, 'Modeling_Data_Nico/Brice_paper',  'FR_Brice_data.xlsx'))
name_list = ['STN', 'Proto', 'Arky']
state_list = ['CTRL', 'Park']

state_list = ['CTRL']
name_list = ['Proto']
plot_exper_FR_distribution(xls, name_list, state_list,
                           color_dict, bins=np.arange(0, 100, 5))
# %% beta induction schematic

filename = 'excitation'
filename = 'inhibition'
tick_length = 10
c_d= {'excitation': 'b',
      'inhibition': 'yellow'}
n = 256
X = np.linspace(0, 720, n, endpoint=True)
Y =(np.sin(X/180 * np.pi))
Y[Y<0] = 0
fig, ax = plt.subplots()
ax.plot(X, Y, color='k', alpha=1.00, lw = 4)
ax.fill_between(X, Y, 0, color=c_d[filename], alpha=1)
ax.set_xticks([0, 360, 720])
ax.tick_params(which = 'major', axis='both', pad=1, length = tick_length)
ax.tick_params(which = 'minor', axis='both', pad=1, length = tick_length/2)
set_minor_locator(ax, n = 2, axis = 'x')
ax.set_yticks([])
ax.xaxis.set_ticklabels([])
remove_whole_frame(ax)
ax.set_ylim([-0.02,1.05])
ax.set_xlim([0,720])

ax.spines['bottom'].set_visible(True)

save_pdf_png(fig, os.path.join(path, filename),
             size=(6,1))
# %% log-normal test

size = 10 ** 5
order = 2
mu = 3
sigma = (10**order - 1) / (10**order +1) * mu


x  = truncated_lognormal_distributed(mu, sigma, 1_000_000, scale_bound=scale_bound_with_mean, 
                                  scale=None, lower_bound_perc=0.8, upper_bound_perc=1.2, 
                                  truncmin=0.0001, truncmax = 10000)
# x_norm = truncated_normal_distributed(mu_norm, sigma_norm, 1_000_000, scale_bound=scale_bound_with_mean, 
#                                   scale=None, lower_bound_perc=0.8, upper_bound_perc=1.2, 
#                                   truncmin=0.01, truncmax=1000)

print(np.average(x), np.std(x), np.min(x), np.max(x))

bins = 100
fig, ax = plt.subplots()
ax.hist(x, bins = 1000)
ax.set_xscale('log')
print((mu + sigma)/ (mu - sigma))
ax.axvline(mu - sigma, ymax = 5000)
ax.axvline(mu + sigma, ymax = 5000)

# %% De la crompe Phase distributions EcoG aligned (Published)

plt.close('all')
filename = os.path.join(root, 'Modeling_Data_Nico', 'Brice_paper', 'De_La_Crompe_2020_data.xlsx') 
scale_count_to_FR = False

#################### DD
FR_header = 'Fig 3F: Firing rate during spontaneous and induced Beta oscillations'
fig_ind_hist = 'Fig 3D (1): Phase Histogram during Spontaneous Beta oscillations'
fig_ind_phase = 'Fig 3D (2): Phase Angle during Spontaneous Beta oscillations' 
angle_header = 'Unnamed: 13_level_1'
recording_spec = 'Spont_beta'
sheet_name = 'Fig 3'
state = 'DD'
y_max_series = {'STN': 5.0, 'Arky': 3.0, 'Proto': 3.2}
name_list = ['STN', 'Arky', 'Proto']
phase_text_x_shift = 150
ylabel_fontsize = 12;  xlabel_fontsize = 13;  xlabel_y = 0.01
n_decimal = 1
scale_count_to_FR = True
y_max_series = {'STN': 30, 'Arky': 21, 'Proto': 26}
n_decimal = 0



# #################### Proto beta inhibition
# FR_header = 'Fig 6F: Firing rate during spontaneous and induced Beta oscillations'
# fig_ind_hist = 'Fig 6E (1): Phase Histogram during Spontaneous Beta oscillations'
# fig_ind_phase = 'Fig 6E (2): Phase Angle during Spontaneous Beta oscillations'      
# angle_header = 'Unnamed: 35_level_1'
# recording_spec = 'Proto_inhibition_induced_beta'
# sheet_name = 'Fig 6'
# state = 'OFF'
# y_max_series = {'STN': 3.0, 'Arky': 1.8, 'Proto': 3.5}
# name_list = ['STN',  'Arky', 'Proto']
# phase_text_x_shift = 100
# ylabel_fontsize = 10;  xlabel_fontsize = 10; xlabel_y = -.01
# n_decimal = 1
# scale_count_to_FR = True
# y_max_series = {'STN': 27, 'Arky': 14, 'Proto': 50}
# n_decimal = 0

# #################### STN beta excitation
# FR_header = 'Fig 3F: Firing rate during spontaneous and induced Beta oscillations'
# fig_ind_hist = 'Fig 3E (1): Phase Histogram during Spontaneous Beta oscillations'
# fig_ind_phase = 'Fig 3E (2): Phase Angle during Spontaneous Beta oscillations'      
# angle_header = 'Unnamed: 35_level_1'
# recording_spec = 'STN_excitation_induced_beta'
# sheet_name = 'Fig 3'
# y_max_series = {'STN': 5, 'Arky': 3, 'Proto': 7} # spike count
# name_list = ['STN',  'Arky', 'Proto']
# phase_text_x_shift = 100
# ylabel_fontsize = 10;  xlabel_fontsize = 10; xlabel_y = -.05
# n_decimal = 0
# scale_count_to_FR = True
# y_max_series = {'STN': 40, 'Arky': 24, 'Proto': 50}
# state = 'OFF'

# # # ################### STN beta inhibtiion aligned to ECog
# FR_header = 'SupFig 5H: Firing rate during spontaneous and induced Beta oscillations'

# fig_ind_hist = 'SupFig 5F (1): Phase Histogram during Induced Beta oscillations peak EcoG'
# fig_ind_phase = 'SupFig 5F (2): Phase Angle during Spontaneous Beta oscillations'      
# angle_header = 'Unnamed: 14_level_1'
# recording_spec = 'STN_inhibition_induced_beta_ECoG'
# sheet_name = 'SupFig 5'
# y_max_series = {'STN': 1, 'Arky': 1, 'Proto': 2}
# name_list = ['STN',  'Arky', 'Proto']
# phase_text_x_shift = 100
# ylabel_fontsize = 10;  xlabel_fontsize = 10; xlabel_y = -.05
# n_decimal = 0
# scale_count_to_FR = True
# y_max_series = {'STN': 11, 'Arky': 8, 'Proto': 39}
# state = 'OFF'


# # #################### STN beta inhibtiion aligned to laser
# FR_header = 'SupFig 5H: Firing rate during spontaneous and induced Beta oscillations'

# fig_ind_hist = 'SupFig 5G (1): Phase Histogram during Beta patterning peak Laser'
# fig_ind_phase = 'SupFig 5G (2): Phase Histogram during Beta patterning peak Laser'      
# angle_header = 'Unnamed: 36_level_1'
# recording_spec = 'STN_inhibition_induced_beta_laser'
# sheet_name = 'SupFig 5'
# y_max_series = {'STN': 2, 'Arky': 1, 'Proto': 3}
# name_list = ['STN',  'Arky', 'Proto']
# phase_text_x_shift = 100
# ylabel_fontsize = 10;  xlabel_fontsize = 10; xlabel_y = -.05
# n_decimal = 0
# scale_count_to_FR = True
# y_max_series = {'STN': 11, 'Arky': 8, 'Proto': 39}
# state = 'OFF'



coef = 100
plot_FR = True

angles, phase_dict = read_Brice_EcoG_aligned_phase_hist(filename,  ['STN', 'Arky', 'Proto'], fig_ind_hist, 
                                           fig_ind_phase, angle_header, coef = coef, sheet_name = sheet_name)


FR_dict = read_Brice_FR_states(filename, name_list, FR_header, sheet_name = sheet_name)    
   

fig = phase_plot_Brice_EcoG_aligned(phase_dict, angles, name_list, color_dict, total_phase = 720, 
                                    set_ylim = True, shift_phase = None, y_max_series = y_max_series,
                                    ylabel_fontsize = ylabel_fontsize,  xlabel_fontsize = xlabel_fontsize, 
                                    tick_label_fontsize = 10, n_decimal = n_decimal, coef = coef,
                                    FR_dict = FR_dict, title = recording_spec.replace('_', ' '),
                                    xlabel = 'phase (deg)', lw = 1, name_fontsize = 12, 
                                    name_ylabel_pad = [0, 0, 0], name_place = 'ylabel', alpha = 0.1,
                                    phase_text_x_shift = phase_text_x_shift, state = state,
                                    xlabel_y = xlabel_y , ylabel_x = -0.1, plot_FR = plot_FR,
                                    box_plot = True, strip_plot = False, 
                                    scale_count_to_FR = scale_count_to_FR)

FR_or_not = ''
if scale_count_to_FR:

    FR_or_not = '_FR'

save_pdf_png(fig, filename.split('.')[0] + '_' + recording_spec + FR_or_not + '_Phase',
             size = (1.8, len(name_list) * 1))

# %% De la Crompe Power spec
plt.close('all')
filename = os.path.join(root, 'Modeling_Data_Nico', 'Brice_paper', 'De_La_Crompe_2020_data.xlsx') 

    
xls = pd.ExcelFile(filename)
data = pd.read_excel(xls, 'Fig 2', header = [0, 1, 2])#, skiprows = [0])
    
f = data['Fig 2I (1): Coherence GP-LFP vs mCx EcoG', 'Unnamed: 46_level_1', 'Frequency (in Hz)'].values
mean_OFF = data['Fig 2I (1): Coherence GP-LFP vs mCx EcoG', 'OFF', 'MEAN '].values
sem_OFF = data['Fig 2I (1): Coherence GP-LFP vs mCx EcoG', 'OFF', 'SEM'].values

mean_ON = data['Fig 2I (1): Coherence GP-LFP vs mCx EcoG', 'ON', 'MEAN '].values
sem_ON = data['Fig 2I (1): Coherence GP-LFP vs mCx EcoG', 'ON', 'SEM'].values

fig, ax = plt.subplots()    
ax.errorbar(f[f < 70], mean_OFF[f < 70], yerr = sem_OFF[f < 70], label = 'OFF', c = 'k')
ax.errorbar(f[f < 70], mean_ON[f < 70], yerr = sem_ON[f < 70],  label = 'ON', c = 'r')
ax.legend(frameon = False)
ind_beta = (f > 10) & (f < 70)
f_beta = f[ind_beta]
print(f_beta[np.argmax(mean_OFF[ind_beta])])
print(f_beta[np.argmax(mean_ON[ind_beta])])


# %% De la Crompe Phase distributions Laser aligned (Unpublished)


experiment_protocol = 'ChR2_STN_CTL'
y_max_series = {'STN': 320, 'Arky': 45, 'Proto': 160}
sheet_name = 'Fig 3'
FR_header = 'Fig 3F: Firing rate during spontaneous and induced Beta oscillations'
y_max_series = {'STN': 36, 'Arky': 7, 'Proto': 20}

# experiment_protocol = 'ArchT_STN_CTL'
# y_max_series = {'STN': 50, 'Arky': 20, 'Proto': 35}
# FR_header = 'SupFig 5H: Firing rate during spontaneous and induced Beta oscillations'
# sheet_name = 'SupFig 5'

# experiment_protocol = 'ArchT_GP_CTL'
# y_max_series = {'STN': 100, 'Arky': 40, 'Proto': 125}
# FR_header = 'Fig 6F: Firing rate during spontaneous and induced Beta oscillations'
# sheet_name = 'Fig 6'
# y_max_series = {'STN': 17, 'Arky': 8, 'Proto': 14}


n_bins = 36
path_Brice = os.path.join(root, 'Modeling_Data_Nico', 'Laser_beta_induction', experiment_protocol)

name_list = ['STN', 'Arky', 'Proto']


FR_dict = read_Brice_FR_states(os.path.join(root, 'Modeling_Data_Nico', 'Brice_paper', 'De_La_Crompe_2020_data.xlsx'),
                               name_list, FR_header, sheet_name = sheet_name)    

shift_phase = 'forward'
# shift_phase = 'backward'
# shift_phase = None
# scale_count_to_FR = True
scale_count_to_FR = False

fig = phase_plot_experiment_laser_aligned(experiment_protocol,  ['Proto', 'STN', 'Arky'], color_dict, path_Brice, y_max_series, 
                                           n_bins = n_bins, f_stim = 20, 
                                           scale_count_to_FR = scale_count_to_FR, title_fontsize = 10, FR_dict = FR_dict, 
                                           total_phase = 720,  box_plot = True, set_ylim= True,
                                           print_stat_phase = True, coef = 1000, phase_text_x_shift = 150, 
                                           phase_txt_fontsize = 8, lw = 0.5,
                                           phase_txt_yshift_coef = 1.4, lw_single_neuron = 1, name_fontsize = 8,  plot_FR = True, 
                                           name_ylabel_pad = [0,0,0], name_place = 'ylabel', alpha_single_neuron = 0.12, title = '',
                                           xlabel_y = 0.01, ylabel_x = -0.1, n_fontsize = 8, state = 'OFF',
                                           plot_single_neuron_hist = True, hist_smoothing_wind = 5,n_neurons_to_plot = 10,
                                           smooth_hist = False, shift_phase = shift_phase, random_seed = 1)#, shift_phase_deg = -90)

save_pdf_png(fig,os.path.join(path_Brice, experiment_protocol + '_Phase'),
             size = (1.8, len(name_list) * 1))


# filepath = '/home/shiva/BG_Oscillations/Modeling_Data_Nico/Laser_beta_induction/ChR2_STN_CTL/STN_Phase/BC_314_10052017_b_02_N1_ON_PhaseStimLaser.txt'

# df = pd.read_table(filepath, skiprows=[0], header = [0])
# df = df [ df ['Times'].notna() ].reset_index()

# n_sweeps = int(np.max(df [ df ['Sweep'].notna() ] ['Sweep'].values))


# nuc_folder = name + '_Phase'
# path_Brice = os.path.join(root, 'Shiva_Modeling_Data', experiment_folder, nuc_folder)                                    xlabel_y = 0.05, ylabel_x = -0.1, n_fontsize = 8)      
# filepath_list = list_files_of_interest_in_path(path_Brice, extensions = ['txt'])
# look_at_one_neuron_laser_Brice(filepath_list[0], total_phase, n_bins, name, color_dict)

# %% Asier Phase distributions Laser aligned (Unpublished)

A_mouse = {'D2' : 0.066, 'Proto' : 25, 'Arky' : 3, 'STN' : 4}

stim_name = 'STN'
name_list = [ 'STN', 'Arky', 'Proto']
align_to= 'Laser'; y_max_series = {'STN': 330, 'Arky': 12, 'Proto': 280}

# stim_name = 'D2'
# name_list = ['D2', 'STN', 'Arky', 'Proto']
# align_to= 'Laser'; y_max_series = {'D2': 180, 'STN': 35, 'Arky': 50, 'Proto': 40}
# # align_to= 'EcoG'; y_max_series = {'D2': 15, 'STN': 13, 'Arky': 25, 'Proto': 11}

experiment_protocol = 'ChR2_' + stim_name + '_CTL_mouse'
path_Asier = os.path.join(root, 'Modeling_Data_Nico', 'Laser_beta_induction', experiment_protocol)

shift_phase = 'forward'
# shift_phase = 'backward'
# shift_phase = 'both'
# shift_phase = None

plot_single_neuron_hist = False
plot_single_neuron_hist = True

# color_dict = {'Proto': 'r', 'STN': 'deepskyblue', Nico's color codes
#               'D2': 'slateblue', 'FSI': 'g', 'Arky': 'yellowgreen'}
fig = phase_plot_experiment_laser_aligned( experiment_protocol, name_list, color_dict, path_Asier, y_max_series, 
                                          n_bins =None, f_stim = 20, 
                                          scale_count_to_FR = True, title_fontsize = 10, FR_dict = A_mouse, 
                                          total_phase = 720, alpha_sem = 0.2, box_plot = True, set_ylim= True,
                                          print_stat_phase = True, coef = 1, phase_text_x_shift = 150, phase_txt_fontsize = 8, 
                                          phase_txt_yshift_coef = 1.4, lw = 0.5, name_fontsize = 8,  plot_FR = True, 
                                          name_ylabel_pad = [0,0,0,0], name_place = 'ylabel', alpha = 0.15, title = '',
                                          xlabel_y = 0.01, ylabel_x = -0.1, n_fontsize = 8, state = 'OFF',
                                          plot_single_neuron_hist = plot_single_neuron_hist, hist_smoothing_wind = 5,
                                          smooth_hist = False, plot_mean_FR = True,
                                          align_to= align_to, shift_phase = shift_phase)
    
save_pdf_png(fig, os.path.join(path_Asier, experiment_protocol + '_Phase_aligned_to_' + align_to),
              size = (1.5, len(name_list) * 1))
             # size = (1.5, len(name_list) * 1.5))

# %% Nico's transient D2 and STN stim



def merge_labeled_non_labeled_data(filepath_list, sheet_name_extra_list, merged_filepath):
    
    FR_df = {}
    
    for i, filepath in enumerate(filepath_list):
        xls = pd.ExcelFile(filepath)
        sheet_name_list = xls.sheet_names

        for sheet_name in sheet_name_list:
            
            name = sheet_name.split('_')[-1].replace( sheet_name_extra_list[i], '')

            if i == 0: # to create the df template to later be concatenated
                
                FR_df[name] = pd.read_excel(xls, sheet_name, header = [0])
                
            else:
                
                FR_df[name] = pd.concat( [ FR_df[name], pd.read_excel(xls, sheet_name, header = [0]).drop(columns = 'Time')], 
                                        axis = 1)

    save_df_dict_to_excel_sheets(FR_df, merged_filepath)
    
    return 0

def decide_what_to_plot(plot_what, filename_labeled, filename_non_labeled, path):
    
    if plot_what == 'non-labeled':
        sheet_name_extra = 'Response'
        filename = filename_non_labeled
        filepath = os.path.join(path, filename)
    
    elif plot_what == 'labeled':
        sheet_name_extra = 'LabelResponse'
        filename = filename_labeled
        filepath = os.path.join(path, filename)
    ###################### labeled and non-labeled merged
    elif plot_what == 'merged':
        filename_list = [filename_non_labeled, filename_labeled] 
        
        filepath_list = [os.path.join(path, filename) for filename in filename_list]
        
        merged_filepath= os.path.join(path, longestSubstringFinder(*filename_list) + 'merged.xlsx')
        filepath = merged_filepath
        sheet_name_extra = ''
        merge_labeled_non_labeled_data(filepath_list, 
                                        ['Response', 'LabelResponse'], 
                                        merged_filepath = merged_filepath)
        
    return filepath, sheet_name_extra


color_dict['MSNs'] = color_dict['D2']
plot_what = 'labeled'
# plot_what = 'non-labeled'
plot_what = 'merged'

##################### Labeled/or non-labeld
filename_non_labeled = 'D2-10ms_OptoStimData_RecMSN-Proto-Arky-STN_NotLabelled.xlsx'
filename_labeled = 'D2-10ms_OptoStimData_RecMSN-Proto-Arky-STN_OnlyLabelled.xlsx'
ylim = (-5, 60)

filename_non_labeled = 'STN-10ms_OptoStimData_RecSTN-Proto-Arky_NotLabelled.xlsx'
filename_labeled = 'STN-10ms_OptoStimData_RecSTN-Proto-Arky_OnlyLabelled.xlsx'
ylim = (-10, 160)
##################### Read and plot
exp_path = os.path.join(root, 'Modeling_Data_Nico', 'Transient_stim',)
filepath, sheet_name_extra = decide_what_to_plot(plot_what, filename_labeled, filename_non_labeled, exp_path)
FR_df = read_sheets_of_xls_data(filepath, sheet_name_extra = sheet_name_extra)
# fig, ax = plot_fr_response_from_experiment(FR_df, filepath, color_dict, ylim = ylim)
# save_pdf_png(fig, filepath.split('.')[0],
#               size=(5, 3))


def plot_individual_traces( FR_df, plot_what, color_dict, xlim = None, ylim = None,  stim_duration = 10, ax = None, 
                           time_shift_dict = None, sheet_name_extra = '', legend_loc = 'upper right'):
    
    fig, ax = get_axes( ax )
    
    if time_shift_dict == None:
        time_shift_dict = { key: 0 for key in list(FR_df.keys())}

    for name in list(FR_df.keys()):
        
        name_adj = name.split('_')[-1].replace( sheet_name_extra, '')
        time = FR_df[name]['Time'] * 1000 - time_shift_dict[name_adj]
        fr = FR_df[name].drop(columns = ['Time'])
        fr_mean = fr.mean(axis=1)
        
        # if name == 'MSNs':
            
        for trace in fr.columns[1:]:
            
            ax.plot(time, fr[trace], c = color_dict[name_adj], alpha = 0.1)
            
        
        n_cells = len(FR_df[name].columns) - 1
        ax.plot(time, fr_mean, c = color_dict[name_adj], label = name_adj + ' n =' + str(n_cells))
        # fr_std = fr.std(axis=1)
        # ax.fill_between(time, fr_mean - fr_std/ np.sqrt(n_cells), fr_mean + fr_std/ np.sqrt(n_cells), 
        #                 color = color_dict[name_adj], alpha = 0.1)
        
    
    ax.set_title(plot_what, fontsize = 15)
    ax.legend(fontsize = 10, frameon = False, loc = legend_loc)
    ax.set_xlabel('Time (ms)', fontsize = 14)
    ax.set_ylabel('Firing rate (spk/s)', fontsize = 14)
    ax.axvspan(0, stim_duration, alpha=0.2, color='yellow')
    ax.axvline(0, *ax.get_ylim(), c= 'grey', ls = '--', lw = 1)
    ax.axvline(stim_duration, *ax.get_ylim(), c= 'grey', ls = '--', lw = 1)
    ax.axhline(0, c= 'grey', ls = ':')

    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)

    remove_frame(ax)
    return fig, ax
    
plot_individual_traces( FR_df, plot_what, color_dict, xlim = None, ylim = None,  stim_duration = 10, ax = None, 
                           time_shift_dict = None, sheet_name_extra = '', legend_loc = 'upper right')
# %% create gif


create_gif_from_images(os.path.join(path, 'Loop_rest_freq'), 
                        'Loop_rest_freq', image_ext = 'png', fps = 2)#, loop = 1)
create_gif_from_images(os.path.join(path, 'Loop_state_freq'), 
                        'Loop_state_freq', image_ext = 'png', fps = 2)#, loop = 1)
# create_gif_from_images(os.path.join(path_rate, 'STN-GP_anim'), 
#                         'STN-GP_anim', image_ext = 'png', fps = 12, 
#                         optimize_gif = True, loop = 1, ext = '.mp4')
# create_gif_from_images(os.path.join(path_rate, 'FSI-Loop_anim'), 
#                         'FSI-Loop_anim', image_ext = 'png', fps = 12, 
#                         optimize_gif = True, loop = 1, ext = '.mp4')
# create_gif_from_images(os.path.join(path_rate, 'FSI-Loop_anim_g_6'), 
#                         'FSI-Loop_anim_g_6', image_ext = 'png', fps = 8, 
#                         optimize_gif = True, loop = 1, ext = '.mp4')
# filepath = os.path.join(path_rate, 'STN-GP_anim', 'STN-GP_anim.gif')
# filepath = os.path.join(path_rate, 'STN-GP_anim', 'STN-Proto_amplitude_ratio.gif')
# img = Image.open(filepath)
# print(find_duration_gif(img))
# %% D2-FSI- Proto with derived external inputs

N_sim = 1000
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt


g = -0.01
G = {}
G[('D2', 'FSI')], G[('FSI', 'Proto')], G[('Proto', 'D2')] = g, g, g*0.5
noise_variance = {'FSI': 0.025003981672404058,
                  'D2': 0.0018347080766611102, 'Proto': 10**-10}
g_ext = -g
name1 = 'D2'
name2 = 'Proto'
name3 = 'FSI'


poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name3: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')
nuclei_dict = run(receiving_class_dict, t_list, dt,
                  nuclei_dict, neuronal_model='spiking')
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=15, plot_start=100,
           title='')  # r"$G_{SP}="+str(round(G[('Proto', 'STN')],2))+"$ "+", $G_{PS}=G_{PP}="+str(round(G[('STN', 'Proto')],2))+'$')

fig, axs = plt.subplots(len(nuclei_dict), 1, sharex=True, sharey=True)
count = 0
for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        count += 1
        nucleus.pop_act = moving_average_array(nucleus.pop_act, 50)
        print(nucleus.name, np.average(nucleus.pop_act[int(
            len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]), 2))
        spikes_sparse = [np.where(nucleus.spikes[i, :] == 1)[
            0]*dt for i in range(nucleus.n)]

        axs[count-1].eventplot(spikes_sparse, colors='k',
                               linelengths=2, lw=2, orientation='horizontal')
        axs[count-1].tick_params(axis='both', labelsize=10)
        axs[count-1].set_title(nucleus.name,
                               c=color_dict[nucleus.name], fontsize=15)
        find_freq_of_pop_act_spec_window_spiking(
            nucleus, 0, t_list[-1], dt, cut_plateau_epsilon=0.1, peak_threshold=0.1, smooth_kern_window=3, check_stability=False)

fig.text(0.5, 0.02, 'time (ms)', ha='center', va='center', fontsize=15)
fig.text(0.02, 0.5, 'neuron', ha='center',
         va='center', rotation='vertical', fontsize=15)

fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
fig2, ax2 = plt.subplots(1, 1, sharex=True, sharey=True)
fig3, ax3 = plt.subplots(1, 1, sharex=True, sharey=True)

for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        ax1.plot(t_list*dt, nucleus.voltage_trace,
                 c=color_dict[nucleus.name], label=nucleus.name)
        ax2.plot(t_list*dt, nucleus.representative_inp['ext_pop',
                 '1'], c=color_dict[nucleus.name], label=nucleus.name)
        ax3.plot(t_list*dt, np.sum([nucleus.representative_inp[key].reshape(-1,) for key in nucleus.representative_inp.keys()], axis=0)-nucleus.representative_inp['ext_pop', '1'],
                 c=color_dict[nucleus.name], label=nucleus.name)
ax1.set_title('membrane potential', fontsize=15)
ax2.set_title('external input', fontsize=15)
ax3.set_title('synaptic input', fontsize=15)
ax1.legend()
ax2.legend()
ax3.legend()

# start=0.002/10; end=0.02/10;
# list_1 = np.linspace(start,end,n)
#loss,ext_firing,firing_prop = find_ext_input_reproduce_nat_firing_relative(tuning_param, list_1, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict)
# loss, ext_firing, firing_prop = find_ext_input_reproduce_nat_firing_3_pop(tuning_param, list_1,list_2, list_3, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict)
# best = np.where(loss == np.min(loss))
# print(ext_firing[best],loss[best],firing_prop['Proto']['firing_mean'][best],firing_prop['D2']['firing_mean'][best],firing_prop['FSI']['firing_mean'][best])
# data_temp = {  'D2' : D2,'Proto': Proto,'FSI':FSI}
# data ={'Proto': Proto[0].pop_act, 'FSI':FSI[0].pop_act, 'D2':D2[0].pop_act, 'G_P_D': Proto[0].synaptic_weight['Proto','D2'],'G_D_F': D2[0].synaptic_weight['D2','FSI'],
#        'G_F_P': FSI[0].synaptic_weight['FSI','Proto']}
# filename = 'FSI_D2_Proto_spiking_N10000_nuclei_dict.pkl'
# output = open(filename, 'wb')
# pickle.dump(data, output)
# output.close()

# pkl_file = open(filename, 'rb')
# data = pickle.load(pkl_file)
# pkl_file.close()

# %% FR simulation vs FR_expected (I = cte + Gaussian noise)


def run_FR_sim_vs_FR_expected_with_I_cte_and_noise(name, FR_list, variance, amplitude):
    N_sim = 1000
    N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
         'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
    dt = 0.25
    t_sim = 1000
    t_list = np.arange(int(t_sim/dt))
    t_mvt = t_sim
    D_mvt = t_sim - t_mvt

    G = {}
    g = -0.01
    g_ext = -g
    poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
        'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}
    noise_variance = {name: variance}
    noise_amplitude = {name: amplitude}
    receiving_pop_list = {(name, '1'): []}

    pop_list = [1]
    nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
    nuclei_dict = {name: nuc}
    receiving_class_dict = set_connec_ext_inp(
        A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')

    firing_prop_hetero = find_FR_sim_vs_FR_expected(
        FR_list, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict, A, A_mvt, D_mvt, t_mvt)

    return firing_prop_hetero


name = 'Proto'
init_method = 'homogeneous'
# init_method = 'heterogeneous'
n = 10
start = 1
end = 10
start = .04
end = .08
FR_list = np.linspace(start, end, n)

n_samples = 4
colormap = plt.cm.viridis  # LinearSegmentedColormap
Ncolors = min(colormap.N, n_samples)
mapcolors = [colormap(int(x*colormap.N/Ncolors)) for x in range(Ncolors)]
amplitude_list = np.full(n_samples, 1)
variance_list = np.logspace(start=-10, stop=1.5, num=n_samples, base=10)
plt.figure()
for i in range(n_samples):
    firing_prop_hetero = run_FR_sim_vs_FR_expected_with_I_cte_and_noise(
        name, FR_list, variance_list[i], amplitude_list[i])
    plt.plot(FR_list, firing_prop_hetero[name]['firing_mean'][:, 0], '-o',
             label=r'$\sigma=$'+"{:e}".format(variance_list[i]), c=mapcolors[i], markersize=4)
    plt.fill_between(FR_list, firing_prop_hetero[name]['firing_mean'][:, 0]-firing_prop_hetero[name]['firing_var'][:, 0],
                     firing_prop_hetero[name]['firing_mean'][:, 0]+firing_prop_hetero[name]['firing_var'][:, 0], alpha=0.1, color=mapcolors[i])
plt.plot(FR_list, FR_list, '--', label='y=x', c='k')
plt.xlabel(r'$FR_{expected}$', fontsize=15)
plt.ylabel(r'$FR_{simulation}$', fontsize=15)
plt.title(name + ' ' + init_method, fontsize=20)
plt.legend()
# %% FR simulation vs FR_ext (I = cte + Gaussian noise)
plt.close('all')


def run_FR_sim_vs_FR_ext_with_I_cte_and_noise(name, g_ext, poisson_prop, FR_list, variance, amplitude):

    N_sim = 1000
    N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
         'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
    dt = 0.25
    t_sim = 1000
    t_list = np.arange(int(t_sim/dt))
    t_mvt = t_sim
    D_mvt = t_sim - t_mvt
    G = {}
    receiving_pop_list = {(name, '1'): []}

    pop_list = [1]

    noise_variance = {name: variance}
    noise_amplitude = {name: amplitude}
    nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
    nuclei_dict = {name: nuc}
    receiving_class_dict = set_connec_ext_inp(
        A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)
    firing_prop = find_FR_sim_vs_FR_ext(
        FR_list, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict, A, A_mvt, D_mvt, t_mvt)

    return firing_prop


init_method = 'heterogeneous'
# init_method = 'homogeneous'
name = 'FSI'
n = 100

g = -0.01
g_ext = -g
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}
# FR_list = spacing_with_high_resolution_in_the_middle(n, *I_ext_range[name]).reshape(-1,) / poisson_prop[name]['g'] / poisson_prop [name ]['n'] / 3
FR_list = np.linspace(*I_ext_range[name], n).reshape(-1,) / \
    poisson_prop[name]['g'] / poisson_prop[name]['n'] / 3

n_samples = 1
mapcolors = create_color_map(n_samples, colormap=plt.get_cmap('viridis'))
mapcolors = create_color_map(n_samples, colormap=plt.get_cmap('Set1'))
mapcolors = ['#354b61', '#2c8bc8', '#e95849', '#3c7c24', '#854a45']
amplitude_list = np.full(n_samples, 1)
variance_list = np.logspace(start=-11.8, stop=-1, num=n_samples, base=10)
variance_list = [0, .1, 1, 5, 15]
variance_list = [5]
fig, ax = plt.subplots(1, 1, figsize=(6, 5))


for i in range(n_samples):
    print(i+1, 'from', n_samples)
    # label_str = fmt(variance_list[i])
    label_str = (variance_list[i])
    # x_series  = FR_list * 1000
    x_series = FR_list * g_ext * \
        poisson_prop[name]['tau']['decay']['mean'] * poisson_prop[name]['n']
    firing_prop_hetero = run_FR_sim_vs_FR_ext_with_I_cte_and_noise(
        name, g_ext, poisson_prop, FR_list, variance_list[i], amplitude_list[i])
    plt.plot(x_series, firing_prop_hetero[name]['firing_mean'][:, 0], '-o',
             label=r'$\sigma=$'+"{}".format(label_str), c=mapcolors[i], markersize=4)
    plt.fill_between(x_series,
                     (firing_prop_hetero[name]['firing_mean'][:, 0] -
                      firing_prop_hetero[name]['firing_var'][:, 0]),
                     (firing_prop_hetero[name]['firing_mean'][:, 0] +
                      firing_prop_hetero[name]['firing_var'][:, 0]),
                     alpha=0.1, color=mapcolors[i])
    # plt.plot(x_series, x_series*i ,'-o',label = r'$\sigma=$'+"{}".format(label_str), c = mapcolors[i], markersize = 4)

# plt.xlabel(r'$FR_{external}$',fontsize = 15)

plt.title(name + ' ' + init_method, fontsize=20)
# plt.ylim(min(FR_list * 1000), max(FR_list * 1000))
plt.legend()
# plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, I_ext_range[name], neuronal_consts, x_val = 'I_ext', ax = ax)
plt.xlabel(r'$I_{external} \; (mV)$', fontsize=15)
plt.ylabel(r'$FR_{simulation} \; (Hz)$', fontsize=15)
remove_frame(ax)
filename = name + \
    '_N_1000_response_curve_with_different_noise_[0_0-1_1_5_15].png'
fig.savefig(os.path.join(path, filename), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)  # extrapolate with the average firing rate ofthe  population
# filename = name + '_1000_neuron_response_curve_with_different_noise.pdf'
# fig.savefig(os.path.join(path, filename), dpi = 300, facecolor='w', edgecolor='w',
#         orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# FR_ext,_ = extrapolate_FR_ext_from_neuronal_response_curve ( FR_list * 1000, firing_prop_hetero[name]['firing_mean'][:,0] ,A[name],
# if_plot = True, end_of_nonlinearity = 25)

# N_sim = 1000
# N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
# dt = 0.25
# t_sim = 1000; t_list = np.arange(int(t_sim/dt))
# t_mvt = t_sim ; D_mvt = t_sim - t_mvt
# G = {}
# receiving_pop_list = {(name,'1') : []}

# pop_list = [1]
# # init_method = 'heterogeneous'
# init_method = 'homogeneous'
# noise_variance = {name : .1}
# noise_amplitude = {name : 1}
# nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
#            synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop, init_method = init_method) for i in pop_list]
# nuclei_dict = {name: nuc}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')
# nuc[0].clear_history(neuronal_model = 'spiking')
# nuc[0].rest_ext_input = FR_ext * nuc[0].syn_weight_ext_pop * nuc[0].n_ext_population * nuc[0].membrane_time_constant

# nuclei_dict = run(receiving_class_dict,t_list, dt,  {name: nuc},neuronal_model = 'spiking')

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = 100, title = '')
# fig, axs = plt.subplots(len(nuclei_dict), 1, sharex=True, sharey=True)
# count = 0
# for nuclei_list in nuclei_dict.values():
#     for nucleus in nuclei_list:
#         count +=1
#         nucleus.smooth_pop_activity(dt, window_ms = 5)
#         FR_mean, FR_std = nucleus. average_pop_activity( t_list, last_fraction = 1/2)
#         print(nucleus.name, 'average ={}, std = {}'.format(FR_mean, FR_std  ) )
#         spikes_sparse = create_sparse_matrix (nucleus.spikes) * dt
#         raster_plot(axs, spikes_sparse, nucleus.name, color_dict, labelsize=10, title_fontsize = 15)
#         find_freq_of_pop_act_spec_window_spiking(nucleus, 0,t_list[-1], dt, cut_plateau_epsilon =0.1, peak_threshold = 0.1, smooth_kern_window= 3 , check_stability = False)


# firing_prop = find_FR_sim_vs_FR_ext([FR_ext],poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt)


# %% Deriving F_ext from response curve of collective behavior in heterogeneous mode Demo

plt.close('all')


def run_FR_sim_vs_FR_ext_with_I_cte_and_noise(name, g_ext, poisson_prop, FR_list, variance, amplitude):

    N_sim = 1000
    N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
         'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
    dt = 0.25
    t_sim = 1000
    t_list = np.arange(int(t_sim/dt))
    t_mvt = t_sim
    D_mvt = t_sim - t_mvt
    G = {}
    receiving_pop_list = {(name, '1'): []}

    pop_list = [1]

    noise_variance = {name: variance}
    noise_amplitude = {name: amplitude}
    nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
    print("noise variance = ", nuc[0].noise_variance)
    nuclei_dict = {name: nuc}
    receiving_class_dict = set_connec_ext_inp(
        A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)
    firing_prop = find_FR_sim_vs_FR_ext(
        FR_list, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict, A, A_mvt, D_mvt, t_mvt)

    return firing_prop


init_method = 'heterogeneous'
# init_method = 'homogeneous'
name = 'D2'
n = 10

g = -0.01
g_ext = -g
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}
# FR_list = spacing_with_high_resolution_in_the_middle(n, *I_ext_range[name]).reshape(-1,) / poisson_prop[name]['g'] / poisson_prop [name ]['n'] / 3
FR_list = np.linspace(*I_ext_range[name], n).reshape(-1,)


fig, ax = plt.subplots(1, 1, figsize=(6, 5))
x_series = FR_list * g_ext * \
    poisson_prop[name]['tau']['decay']['mean'] * poisson_prop[name]['n']
firing_prop_hetero = run_FR_sim_vs_FR_ext_with_I_cte_and_noise(
    name, g_ext, poisson_prop, FR_list, noise_variance[name], 1)

plt.plot(x_series, firing_prop_hetero[name]['firing_mean'][:, 0], '-o', label=r'$\sigma=$'+"{}".format(noise_variance[name]),
         c=color_dict[name], markersize=4)

plt.fill_between(x_series,
                 (firing_prop_hetero[name]['firing_mean'][:, 0] -
                  firing_prop_hetero[name]['firing_var'][:, 0]),
                 (firing_prop_hetero[name]['firing_mean'][:, 0] +
                  firing_prop_hetero[name]['firing_var'][:, 0]),
                 alpha=0.1, color=color_dict[name])


plt.title(name + ' ' + init_method, fontsize=20)
plt.legend()
# plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, I_ext_range[name], neuronal_consts, x_val = 'I_ext', ax = ax)
plt.xlabel(r'$I_{external} \; (mV)$', fontsize=15)
plt.ylabel(r'$FR_{simulation} \; (Hz)$', fontsize=15)
remove_frame(ax)
filename = name + \
    '_N_1000_response_curve_with_different_noise_[0_0-1_1_5_15].png'
fig.savefig(os.path.join(path, filename), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)  # extrapolate with the average firing rate ofthe  population


# %% Deriving F_ext from response curve of collective behavior in heterogeneous mode



plt.close('all')
# name = 'D2'
# name = 'FSI'
# name = 'STN'
name = 'Proto'
# name = 'Arky'

# state = 'rest'
# state = 'awake_rest'
# state = 'DD_anesth'
# state = 'mvt'
# state = 'trans_Nico_mice'
# state = 'trans_Kita_rat'
state = 'induction_STN_excitation'
state = 'induction_Proto_inhibition'

print('desirec activity =', Act[state][name])
save_mem_pot_dist = True
# save_mem_pot_dist = False

FSI_on_log = False
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.1    
t_sim = 2000
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
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = True
keep_noise_all_t = True
set_FR_range_from_theory = False
set_input_from_response_curve = True
save_init = False
der_ext_I_from_curve = True
if_plot = True
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = False
# use_saved_FR_ext = True

poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 'g': 0.01}}

class Nuc_keep_V_m(Nucleus):

    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = self.sum_synaptic_input(
            receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, t,
                              receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        self.reset_potential_with_interpolation(spiking_ind, dt)
        self.all_mem_pot[:, t] = self.mem_potential

    def cal_ext_inp(self, dt, t):

        # choose method of exerting external input from dictionary of methods
        I_ext = self.ext_inp_method_dict[self.ext_inp_method](dt)

        self.noise_all_t[:, t] = self.noise.reshape(-1,)
        # print(self.rest_ext_input.shape, self.noise.shape)
        self.I_syn['ext_pop', '1'], self.I_rise['ext_pop', '1'] = self.input_integ_method_dict[self. ext_input_integ_method](
            I_ext, dt,
            I_rise=self.I_rise['ext_pop', '1'],
            I=self.I_syn['ext_pop', '1'],
            tau_rise=self.tau_ext_pop['rise'],
            tau_decay=self.tau_ext_pop['decay'])

    def constant_ext_input_with_noise(self, dt):


        self.noise =  self.noise_generator_dict [self.noise_method] (self.noise_amplitude,
                                                                     self.noise_std,
                                                                     self.n,
                                                                     dt,
                                                                     self.sqrt_dt,
                                                                     tau=self.noise_tau,
                                                                     noise_dt_before=self.noise
                                                                     )

        return self.rest_ext_input + self.noise.reshape(-1,)


nuc = [Nuc_keep_V_m(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, 
                    N, Act[state], A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                    smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve, 
                    state = state, poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, 
                    mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t, 
                    ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method,
                    path=path_lacie, save_init=save_init, noise_method=noise_method, keep_noise_all_t=keep_noise_all_t,
                    FR_ext_specs=FR_ext_specs[name][state], plot_spike_thresh_hist= False, plot_RMP_to_APth = False) for i in pop_list]

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
    
    
for name in list(nuclei_dict.keys()):
    print('mean I0 ', name, np.round( np.average( nuclei_dict[name][0].rest_ext_input)) , 2) 
    print('mean Noise ', name, np.average(
        abs(nuclei_dict[name][0].noise_all_t)))
    print('std Noise ', name, np.std(nuclei_dict[name][0].noise_all_t))
    print('mean firing =', np.round( np.average(nuclei_dict[name][0].pop_act[int(t_sim / 2):]), 2 ) )
          # , '± ', np.round( np.std(nuclei_dict[name][0].pop_act[int(t_sim / 2):]), 2 ) ) 
    print('coherence = ', nuclei_dict[name][0].cal_coherence(
        dt, sampling_t_distance_ms=1))
    
if name in ['STN', 'Arky', 'Proto'] and state in ['rest', 'DD_anesth']:
    state_dict = {'rest': 'CTRL', 'DD_anesth': 'Park'}
    xls = pd.ExcelFile(os.path.join(root, 'FR_Brice_data.xlsx'))
    if name == 'STN':
        col = 'w'
    else:
        col = 'k'
    figs = plot_exper_FR_distribution(xls, [name], [state_dict[state]],
                                      color_dict, bins=np.arange(
                                          0, bins[name][state]['max'], bins[name][state]['step']),
                                      hatch='/', edgecolor=col, alpha=0.2, zorder=1,
                                      annotate_fontsize = 20)
    

    fig_FR_dist = plot_FR_distribution(nuclei_dict, dt, color_dict, bins=np.arange(0, bins[name][state]['max'], bins[name][state]['step']),
                                    ax=figs[name].gca(), alpha=1, zorder=0, start=int(t_sim / dt / 2),
                                     legend_fontsize = 18, label_fontsize = 20, ticklabel_fontsize = 20,
                                     annotate_fontsize = 20, nbins = 4, state = state)

elif state in list (bins[name].keys() ):
    if name == '':
        
        only_non_zero= True; box_plot =False
    
    else:
        
        only_non_zero= False; box_plot =False
        
    if name == 'FSI' and FSI_on_log:
        log_hist = True
        _bins = np.logspace(-2, 2, 50)
    
    else:
        log_hist = False
        _bins = np.arange(0, bins[name][state]['max'], bins[name][state]['step'])
        
    fig_FR_dist = plot_FR_distribution(nuclei_dict, dt, color_dict,
                                        bins = _bins,
                                        ax = None, alpha = 1, zorder = 0, start = int(t_sim / dt / 2),
                                        log_hist = log_hist, only_non_zero= only_non_zero, box_plot =box_plot,
                                        legend_fontsize = 18, label_fontsize = 20, ticklabel_fontsize = 20,
                                        annotate_fontsize = 20, nbins = 4, state = state)
try:
    save_pdf_png(fig_FR_dist, os.path.join(path, name + '_FR_dist_' + state + '_'),
              size=(6, 5))
except NameError:
    pass
# fig_ISI_dist = plot_ISI_distribution(nuclei_dict, dt, color_dict, bins=np.logspace(0, 4, 50),
#                                      ax=None, alpha=1, zorder=0, start=int(t_sim / dt / 2), log_hist=True)

# save_pdf_png(fig_ISI_dist, os.path.join(path, name + '_ISI_dist_' + state + '_'),
#              size=(6, 5))

# plot_spike_amp_distribution(nuclei_dict, dt, color_dict, bins = 50)


status = 'set_FR'

    
# fig, ax = plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)

# nucleus.smooth_pop_activity(dt, window_ms=5)
# fig = plot(nuclei_dict, color_dict, dt,  t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
#            title_fontsize=15, plot_start=int(t_sim / 2), title=str(dt),
#            include_FR=False, include_std=False, plt_mvt=False,
#            legend_loc='upper right', ylim=None)

# # save_pdf_png(fig, os.path.join(path, name + '_Firing_'),
# #              size=(12, 4))

# peak_threshold = 0.1
# smooth_window_ms = 3
# smooth_window_ms = 5
# cut_plateau_epsilon = 0.1
# lim_oscil_perc = 10
# low_pass_filter = False

# fig_spec, ax = plt.subplots(1, 1)
# _, f, pxx = find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
#                                       smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
#                                       low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
#                                       spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=3,
#                                       include_beta_band_in_legend=False)
# ax.set_xlim(0, 70)
# save_pdf_png(fig_spec, os.path.join(path, name + '_spec_' + state + '_'),
#              size=(4, 4))

# fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
#                                     plot_start=int(t_sim/2), plot_end=t_sim, tick_label_fontsize=12,
#                                     title_fontsize=25, lw=1, linelengths=1, n_neuron=50,
#                                     include_nuc_name=True, set_xlim=True,
#                                     remove_ax_frame=False, y_tick_length=2, x_tick_length=3)
# save_pdf_png(fig_raster, os.path.join(path, name + '_Raster_' + state + '_'),
#              size=(12, 4))

# %% check noise variance

n = 10
dt = 0.1
sqrt_dt = np.sqrt(dt)
t_sim = 10000
t_list = np.arange(int(t_sim/dt))
noise = np.zeros((n, len(t_list)))
variance = 1
for t in t_list[1:]:
    noise[:, t] = OU_noise_generator(
        1, variance, n, dt, sqrt_dt, tau=20,  noise_dt_before=noise[0, t-1]).reshape(-1,)
    # noise[0, t] = noise_generator(1, variance, n, dt, sqrt_dt, tau = 0, noise_dt_before = 0)
fig, ax = plt.subplots()
ax.plot(t_list, noise[0, :])
print(np.std(noise))
# %% Find noise sigma

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_sim = 5000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'Proto'
name_list = [name1]
state = 'rest'
g_ext = 0.01
g = 0
G = {}
firing_ylim = [30, 70]
noise_dict = {'Proto': [5000, 10000, 25000, 50000],
              'STN': [10, 500, 1000]}

poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}

receiving_pop_list = {(name1, '1'):  []}  # ,


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True
keep_mem_pot_all_t = True
keep_noise_all_t = True
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True


class Nuc_keep_V_m(Nucleus):
    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = self.sum_synaptic_input(
            receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, t,
                              receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        self.reset_potential_with_interpolation(spiking_ind, dt)
        self.all_mem_pot[:, t] = self.mem_potential

    def cal_ext_inp(self, dt, t):

        # choose method of exerting external input from dictionary of methods
        I_ext, noise = self.ext_inp_method_dict[self.ext_inp_method](dt)

        self.noise_all_t[:, t] = noise

        self.I_syn['ext_pop', '1'], self.I_rise['ext_pop', '1'] = self.input_integ_method_dict[self. ext_input_integ_method](I_ext, dt,
                                                                                                                             I_rise=self.I_rise[
                                                                                                                                 'ext_pop', '1'],
                                                                                                                             I=self.I_syn[
                                                                                                                                 'ext_pop', '1'],
                                                                                                                             tau_rise=self.tau_ext_pop[
                                                                                                                                 'rise'],
                                                                                                                             tau_decay=self.tau_ext_pop['decay'])

    def constant_ext_input_with_noise(self, dt):
        noise = self.noise_generator_dict[self.noise_method](self.noise_amplitude,
                                                             self.noise_variance,
                                                             self.n,
                                                             dt,
                                                             self.sqrt_dt,
                                                             tau=self.noise_tau,
                                                             noise_dt_before=self.noise
                                                             ).reshape(-1,)

        return self.rest_ext_input + noise, noise


nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, keep_noise_all_t=keep_noise_all_t) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective', save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


n_run = 1
plot_firing = True
plot_spectrum = True
plot_raster = True
plot_phase = True
low_pass_filter = False
save_pkl = False
save_figures = True
save_pxx = False
# n_run = 8; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True

# save_figures = True ; save_pkl = True
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
check_peak_significance = False
G_dict = {}
G_dict = {k: v * K[k] for k, v in G_dict.items()}
n = len(list(noise_dict.values())[0])
filename = name1 + '_N_1000_T_5000_' + str(n) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') + '_' + noise_method + '.pkl'

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
phase_ref = name1


figs, title, data = Coherence_single_pop_exploration_SNN(noise_dict, path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                         noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                         lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=firing_ylim, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                         plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                         find_beta_band_power=True, fft_method=fft_method, n_windows=3, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                         reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                         state=state, K_real=K_real, N_real=N_real, N=N, divide_beta_band_in_power=False,
                                                         receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                         use_saved_FR_ext=True, check_peak_significance=check_peak_significance,
                                                         find_phase=True, phase_thresh_h=0, filter_order=6, low_f=8, high_f=40,
                                                         n_phase_bins=70, start_phase=int(t_sim/4), phase_ref=phase_ref, save_pxx=save_pxx,
                                                         plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                         len_f_pxx=250, title_pad=-8)


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    filename = 'noise_level'
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.2)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 6), (5, 5), (10, 6), (4, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=[name1 + '_']*4, s=s)


# %% Deriving F_ext from the response curve


plt.close('all')
name = 'Proto'
N_sim = 1
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
G = {}
receiving_pop_list = {(name, '1'): []}

pop_list = [1]
noise_variance = {name: 0}
noise_amplitude = {name: 1}
g = -0.01
g_ext = -g
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
set_input_from_response_curve = True
save_init = False
der_ext_I_from_curve = True
if_plot = True
# bound_to_mean_ratio = [0.5, 20]
# spike_thresh_bound_ratio = [1/5, 1/5]
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': .2}}, 'g': g_ext}}

Act = A
# Act = A_DD
nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, Act, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuclei_dict = {name: nuc}
nucleus = nuc[0]
n = 70
pad = [0.001, 0.001]
all_FR_list = {'FSI': np.linspace(0.045, 0.08, 250).reshape(-1, 1),
               'D2': np.linspace(0.045, 0.08, 250).reshape(-1, 1),
               'Proto':   (0.04, 0.07)}  # [0.02, 0.05]}
all_FR_list = {name: all_FR_list[name]}
name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
filepaths = {name1: name1 + '_N_'+str(N_sim) + '_T_2000.pkl',
             name2: name2 + '_N_'+str(N_sim) + '_T_2000.pkl',
             name3: name3 + '_N_'+str(N_sim) + '_T_2000_noise_var_15.pkl'}
# nuc[0].set_init_from_pickle( os.path.join( path, filepaths[name]))
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
mapcolors = ['#354b61', '#2c8bc8', '#e95849', '#3c7c24', '#854a45']
color = mapcolors[0]
nuc[0].noise_variance = 0
# pad = [0.003, 0.0033] # FSI mvt
receiving_class_dict = set_connec_ext_inp(Act, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n, if_plot=if_plot, end_of_nonlinearity=30, c=color, ax=ax,
                                          left_pad=pad[0], right_pad=pad[1])
plot_theory_FR_sim_vs_FR_ext(
    name, poisson_prop, I_ext_range[name], neuronal_consts, x_val='I_ext', ax=ax)

# ax.set_xlim(29, 29.9) ### indiviual fitted curves for noise =0, 0.1, 1  of FSI
# ax.set_xlim(90.2, 94) ### indiviual fitted curves for noise =0, 5, 15  of Proto
# ax.set_ylim(20, 70) ### indiviual fitted curves for noise =0, 5, 15  of Proto

# fig.savefig(os.path.join(path, 'low_FR_fitted_curve_'+name+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#         orientation='portrait',
#         transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'low_FR_fitted_curve_'+name+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#         orientation='portrait',
#         transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# nuclei_dict = run(receiving_class_dict,t_list, dt,  {name: nuc})
# nucleus.smooth_pop_activity(dt, window_ms = 5)
# fig = plot(nuclei_dict,color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, title = init_method)


# Check behavior
# nuclei_dict = run(receiving_class_dict,t_list, dt,  {name: nuc})


# fig, axs = plt.subplots(len(nuclei_dict), 1, sharex=True, sharey=True)
# count = 0
# for nuclei_list in nuclei_dict.values():
#     for nucleus in nuclei_list:
#         count +=1
#         nucleus.smooth_pop_activity(dt, window_ms = 5)
#         FR_mean, FR_std = nucleus. average_pop_activity( t_list, last_fraction = 1/2)
#         print(nucleus.name, 'average ={}, std = {}'.format(FR_mean, FR_std  ) )
#         spikes_sparse = create_sparse_matrix (nucleus.spikes) * dt
#         raster_plot(spikes_sparse, nucleus.name, color_dict, ax = axs[count -1], labelsize=10, title_fontsize = 15, xlim = [1000, 1500])
#         find_freq_of_pop_act_spec_window_spiking(nucleus, 0,t_list[-1], dt, cut_plateau_epsilon =0.1, peak_threshold = 0.1, smooth_kern_window= 3 , check_stability = False)

# fig.text(0.5, 0.02, 'time (ms)', ha='center', va='center',fontsize= 15)
# fig.text(0.02, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 15)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, title = init_method)

# filename = ( 'Smoothed_average_FR_' +  name + '_g=' + str(g) + '_' + init_method + '_' + ext_inp_method + '_noise=' +
#             str(noise_variance[name])  #+ 'N_ext_Proto='  + str(poisson_prop[name]['n'])
#             + '_N=' + str(N_sim)  + '.png' )

# plt.savefig(os.path.join(path, filename), dpi = 300, facecolor='w', edgecolor='w',
#         orientation='portrait',
#         transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
# fig2, ax2 = plt.subplots(1, 1, sharex=True, sharey=True)
# fig3, ax3 = plt.subplots(1, 1, sharex=True, sharey=True)

# for nuclei_list in nuclei_dict.values():
#     for nucleus in nuclei_list:
#         ax1.plot(t_list*dt,nucleus.voltage_trace,c = color_dict[nucleus.name],label = nucleus.name)
#         ax2.plot(t_list*dt,nucleus.representative_inp['ext_pop','1'],c = color_dict[nucleus.name],label = nucleus.name)
#         ax3.plot(t_list*dt,np.average(nucleus.ext_input_all,axis = 0),c = color_dict[nucleus.name],label = nucleus.name)
#         ax3.fill_between(t_list*dt, np.average(nucleus.ext_input_all,axis = 0) - np.std(nucleus.ext_input_all,axis = 0),
#                           np.average(nucleus.ext_input_all,axis = 0) + np.std(nucleus.ext_input_all,axis = 0), alpha = 0.3)
#         # ax3.plot(t_list*dt,np.sum([nucleus.representative_inp[key].reshape(-1,) for key in nucleus.representative_inp.keys()],axis =0)-nucleus.representative_inp['ext_pop','1'],
#                   # c = color_dict[nucleus.name],label = nucleus.name)
# ax1.set_title('membrane potential',fontsize = 15)
# ax2.set_title('external input one neuron',fontsize = 15)
# ax3.set_title('Mean external input',fontsize = 15)
# ax1.legend();ax2.legend() ; ax3.legend()
# plt.legend()

# see the individual spikes and firing rates with instantenous and exponential input methods.
# nucleus = nuc[0]
# plt.figure()
# for i in range(nucleus.n):
#     plt.plot(t_list * dt, nucleus.all_mem_pot[i,:])

# # nucleus.smooth_pop_activity(dt, window_ms = 10)
# # plt.plot(t_list * dt, nucleus.pop_act,  label = 'instanteneous')
# plt.xlabel("time (ms)", fontsize = 15)
# plt.ylabel(r"$V_{m}$", fontsize = 15)
# plt.legend()
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, title = init_method)
# %% Deriving F_ext from the response curve for each neuron individually
# np.random.seed(1006)
plt.close('all')
name = 'FSI'
state = 'rest'
N_sim = 1
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
G = {}
receiving_pop_list = {(name, '1'): []}

pop_list = [1]

# noise_variance_tau_real = {'rest': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*7, 'STN':4} , # Proto tau_m = 20
#                            'mvt': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*7, 'STN':4} ,
#                            'DD': {'FSI' : 10 , 'D2': 3 , 'Proto': 15*7, 'STN':4} ,
#                           }
# noise_variance_tau_real = {'rest': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*8, 'STN':4} , # Proto tau_m = 25
#                            'mvt': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*8, 'STN':4} ,
#                            'DD': {'FSI' : 10 , 'D2': 3 , 'Proto': 15*8, 'STN':4} ,
#                          }
noise_variance_tau_real = {'rest': {'FSI': 8, 'D2': 3, 'Proto': 15*2, 'STN': 4},  # Proto tau_m = 13
                           'mvt': {'FSI': 8, 'D2': 3, 'Proto': 15*2, 'STN': 4},
                           'DD': {'FSI': 15, 'D2': 3, 'Proto': 15*2, 'STN': 6},
                           'trans': {'STN': 5, 'D2': 12}
                           }
noise_variance = noise_variance_tau_real[state]

noise_amplitude = {name: 1}
g = -0.01
g_ext = -g
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
set_FR_range_from_theory = False
set_input_from_response_curve = True
save_init = False
der_ext_I_from_curve = True
if_plot = False
keep_mem_pot_all_t = True
# bound_to_mean_ratio = [0.5, 20]
# spike_thresh_bound_ratio = [1/5, 1/5]
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 'g': g_ext}}

nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuclei_dict = {name: nuc}
nucleus = nuc[0]
n = 50

pad_tau_real = {'FSI': {'rest': [0.005, 0.0057], 'mvt': [0.005, 0.006], 'DD': [0.005, 0.01]},
                'D2': {'rest': [0.002, 0.002], 'mvt': [0.001, 0.001], 'DD': [0.002, 0.002], 'trans': [0.003, 0.005]},
                'Proto': {'rest': [0.001, 0.001], 'mvt': [0.01, 0.01], 'DD': []},
                'STN': {'rest': [0.004, 0.006], 'mvt': [0.003, 0.0033], 'DD': [0.003, 0.0055], 'trans': [0.003, 0.003]}}

pad = pad_tau_real
# all_FR_list_tau_m_5 ={
#                       'FSI': { 'rest' : np.linspace ( 0.045, 0.08 , 250).reshape(-1,1) ,
#                                'mvt': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1),
#                                'DD': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1) / 2 } ,
#                       'D2': { 'rest' : np.linspace ( 0.045, 0.08 , 250).reshape(-1,1),
#                               'mvt': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1) ,
#                               'DD': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1) } ,
#                       'Proto': { 'rest' : np.array([0.04, 0.07]) ,
#                                  'mvt' : np.linspace ( 0.13, 0.3 , 250).reshape(-1,1) ,
#                                  'DD' : np.array([0.04, 0.07])}
#                       }

all_FR_list_tau_real = {
    'FSI': {'rest': np.linspace(0.020, 0.05, 250).reshape(-1, 1),
            'mvt': np.linspace(0.02, 0.05, 250).reshape(-1, 1),
            'DD': np.linspace(0.02,  0.05, 250).reshape(-1, 1)},
    'D2': {'rest': np.linspace(0.001, 0.005, 250).reshape(-1, 1),
           'mvt': np.linspace(0.015, 0.05, 250).reshape(-1, 1),
           'DD': np.linspace(0.015, 0.05, 250).reshape(-1, 1),
           'trans': np.linspace(0.01, 0.04, 250).reshape(-1, 1)},
    # 'Proto': { 'rest' : np.array([0.02,0.05]), # tau_m = 20
    #            'mvt' : np.linspace ( 0.13, 0.3 , 250).reshape(-1,1) ,
    #            'DD' : []},
    'Proto': {'rest': np.array([0.01, 0.02]),  # tau_m = 43
              'mvt': np.linspace(0.13, 0.3, 250).reshape(-1, 1),
              'DD': []},
    # 'Proto': { 'rest' : np.array([0.01,0.03]), # tau_m = 13
    #            'mvt' : np.linspace (  0.015, 0.022 , 250).reshape(-1,1),
    #            'DD' : []},
    'STN': {'rest': np.linspace(0.008, 0.03, 250).reshape(-1, 1),
            'mvt': np.linspace(0.045, 0.08, 250).reshape(-1, 1),
            'DD': np.linspace(0.01, 0.02, 250).reshape(-1, 1),
            'trans': np.linspace(0.0135, 0.018, 250).reshape(-1, 1)}
}
end_of_nonlinearity = {
    'FSI': {'rest': 35, 'mvt': 40, 'DD': 40},
    'D2':  {'rest': 10, 'mvt': 10, 'DD': 20, 'trans': 40},
    'Proto':  {'rest':  25, 'mvt': 40, 'DD': 35},
    'STN': {'rest': 35, 'mvt': 25, 'DD': 40, 'trans': 35}
}
all_FR_list = {name: all_FR_list_tau_real[name][state]}

filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl',
             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl',
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
             'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl'}

# nuc[0].set_init_from_pickle( os.path.join( path,filepaths[name]), set_noise = False)
# receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
#                                           all_FR_list = all_FR_list , n_FR =n, if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity[name][state],
#                                           left_pad =pad[name][state][0], right_pad=pad[name][state][1], set_FR_range_from_theory=set_FR_range_from_theory)
# ########## Collective

receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=20, if_plot=if_plot, end_of_nonlinearity=end_of_nonlinearity[name][state],
                                          left_pad=pad[name][state][0], right_pad=pad[name][state][
                                              1], set_FR_range_from_theory=set_FR_range_from_theory,
                                          method='single_neuron')
print("rest ext inp mean = ", np.average(nuc[0].rest_ext_input))
print("FR_ext mean = ", np.average(nuc[0].FR_ext))
print("number of nans = ", np.sum(np.isnan(nuc[0].FR_ext)))

# save_all_mem_potential(nuclei_dict, path)

nuclei_dict = run(receiving_class_dict, t_list, dt,  {name: nuc})
plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
# nucleus.smooth_pop_activity(dt, window_ms = 5)
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt,
           D_mvt, ax=None, title_fontsize=15, title=init_method)


# %% plot fitted response curve different noise
plt.close('all')
name = 'Proto'
fit = 'linear'
N_sim = 1
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
G = {}
receiving_pop_list = {(name, '1'): []}

pop_list = [1]
# init_method = 'heterogeneous'
init_method = 'homogeneous'
noise_variance = {name: 0.1}
noise_amplitude = {name: 1}
g = -0.01
g_ext = -g
ext_inp_method = 'const+noise'
ext_input_integ_method = 'dirac_delta_input'
syn_input_integ_method = 'exp_rise_and_decay'
bound_to_mean_ratio = [0.5, 20]
spike_thresh_bound_ratio = [1/5, 1/5]
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 'g': g_ext}}
Act = A_DD
nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, Act, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking',
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=True, ext_inp_method=ext_inp_method,
               syn_input_integ_method=syn_input_integ_method, ext_input_integ_method=ext_input_integ_method, path=path) for i in pop_list]
nuclei_dict = {name: nuc}
nucleus = nuc[0]
n = 50
pad = [0.001, 0.001]
all_FR_list = {'FSI': np.linspace(0.05, 0.07, 250).reshape(-1, 1),
               'D2': np.linspace(0.05, 0.07, 250).reshape(-1, 1),
               'Proto': [0.04, 0.07]}
all_FR_list = {name: all_FR_list[name]}
n_samples = 3
# mapcolors = create_color_map(n_samples, colormap = plt.get_cmap('viridis'))
mapcolors = ['#354b61', '#3c7c24', '#854a45']
if_plot = True
fig, ax = plt.subplots(1, 1)
variance_list = [0, 5, 15]
count = 0
for count, noise in enumerate(variance_list):
    c = mapcolors[count]
    nucleus.noise_variance = noise
    receiving_class_dict = set_connec_ext_inp(A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                              all_FR_list=all_FR_list, n_FR=n, if_plot=if_plot, end_of_nonlinearity=30,
                                              left_pad=pad[0], right_pad=pad[1], ax=ax, c=c)
    count += 1
# Proto
FR_str = FR_ext_of_given_FR_theory(nucleus.spike_thresh, nucleus.u_rest, nucleus.membrane_time_constant,
                                   nucleus.syn_weight_ext_pop, all_FR_list[name][0], nucleus.n_ext_population)
FR_end = FR_ext_of_given_FR_theory(nucleus.spike_thresh, nucleus.u_rest, nucleus.membrane_time_constant,
                                   nucleus.syn_weight_ext_pop, all_FR_list[name][1], nucleus.n_ext_population)
FR_list = np.linspace(FR_str, FR_end, 100)
ax.plot(FR_list * nucleus.membrane_time_constant * nucleus.syn_weight_ext_pop * nucleus.n_ext_population,
        FR_ext_theory(nucleus.spike_thresh, nucleus.u_rest, nucleus.membrane_time_constant, nucleus.syn_weight_ext_pop,
                      FR_list, nucleus.n_ext_population) * 1000, c='lightcoral', label='theory', lw=2.5)
# FSI
plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, [
                             I_ext_range[name][0], 6.01], neuronal_consts, x_val='I_ext', ax=ax)

# mutual
ax.legend()
ax_label_adjust(ax, fontsize=20)
filename = name + '_one_neuron_' + fit + '_fit_to_response_curve.png'
fig.savefig(os.path.join(path, filename), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, filename.replace('png', 'pdf')), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
# %% two connected nuclei with derived I_ext from response curve
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 200
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

name1 = 'Proto'  # projecting
name2 = 'FSI'  # recieving
g = -10**-5
g_ext = 0.01
G = {}
G[(name2, name1)] = g

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'): [],
                      (name2, '1'): [(name1, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'

save_init = False
if_plot = False
noise_variance = {name1: 3, name2: .1}
noise_amplitude = {name1: 1, name2: 1}
# Act = A_DD
Act = A
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, Act, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking',
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init, set_input_from_response_curve=True) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, Act, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking',
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init, set_input_from_response_curve=True) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
n = 50
pad = [0.001, 0.001]
all_FR_list = {'FSI': np.linspace(0.045, 0.08, 250).reshape(-1, 1),
               'D2': np.linspace(0.045, 0.08, 250).reshape(-1, 1),
               'Proto': [0.04, 0.07]}
if_plot = False
filepaths = {'FSI': 'FSI_A_12-5_N_1000_T_2000_noise_var_0-1.pkl',
             'D2': 'D2_A_1-1_N_1000_T_2000_noise_var_0-1.pkl',
             'Proto': 'Proto_A_45_N_1000_T_2000_noise_var_3.pkl'}
DD_init_filepaths = {'Proto': 'Proto_A_38_N_1000_T_2000_noise_var_10.pkl',
                     'FSI': 'FSI_A_24_N_1000_T_2000_noise_var_1.pkl',
                     'D2': 'D2_A_6-6_N_1000_T_2000_noise_var_0-1.pkl'}
set_init_all_nuclei(nuclei_dict, filepaths=filepaths)
receiving_class_dict = set_connec_ext_inp(
    Act, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)

# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
#                                          all_FR_list = all_FR_list , n_FR =n, if_plot = if_plot, end_of_nonlinearity = 25, left_pad =pad[0], right_pad=pad[1])

# Check behavior
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, Act, A_mvt, D_mvt, t_mvt, t_list, dt,
                                      mem_pot_init_method=mem_pot_init_method, set_noise=False)
nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)

for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        nucleus.smooth_pop_activity(dt, window_ms=5)


fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=0, plot_end=t_sim, ax_label=True,
                              tick_label_fontsize=18, labelsize=20, title_fontsize=30, lw=2, linelengths=2, n_neuron=100, include_title=True, set_xlim=True)
fig_.set_size_inches((10, 3.5), forward=False)
filename = ('Raster_plot_' + mem_pot_init_method + '_' + name1 + '_' + name2 + '_' + init_method + '_' + ext_inp_method + '_noise=' +
            str(noise_variance[name1]) + '_' + str(noise_variance[name2])
            + '_N=' + str(N_sim) + '_N_ext=' + str(poisson_prop[name1]['n']) + '.png')

plt.savefig(os.path.join(path, filename), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait',
            transparent=True, bbox_inches="tight", pad_inches=0.1)
plt.savefig(os.path.join(path, filename.replace('png', 'pdf')), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait',
            transparent=True, bbox_inches="tight", pad_inches=0.1)
fig = plot(nuclei_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=15,
           plot_start=0, title="", tick_label_fontsize=18, plt_mvt=False, ylim=(-10, 110))
fig.set_size_inches((6, 5), forward=False)
# fig.tight_layout()
filename = ('Smoothed_average_FR_' + mem_pot_init_method + '_' + name1 + '_' + name2 + '_' + init_method + '_' + ext_inp_method + '_noise=' +
            str(noise_variance[name1]) + '_' + str(noise_variance[name2])
            + '_N=' + str(N_sim) + '_N_ext=' + str(poisson_prop[name1]['n']) + '.png')

plt.savefig(os.path.join(path, filename), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait',
            transparent=True, bbox_inches="tight", pad_inches=0.1)

plt.savefig(os.path.join(path, filename.replace('png', 'pdf')), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait',
            transparent=True, bbox_inches="tight", pad_inches=0.1)
# %% Run three connected nuclei save membrane potential distribution
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [0, int(t_mvt/dt)]
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
g = 0
g_ext = 0.01
G = {}
G[(name2, name1)], G[(name3, name2)], G[(name1, name3)] = g, 0.5 * g, g

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name3: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
save_init = False
if_plot = False
noise_variance = {name1: 0.1, name2: 0.1, name3: 15}
noise_amplitude = {name1: 1, name2: 1, name3: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking',
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=True,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking',
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=True,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking',
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=True,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)

filepaths = {'FSI': 'tau_m_9.5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl',
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
             'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_15.pkl'}
# 'Proto':'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl'
set_init_all_nuclei(nuclei_dict, filepaths=filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise=False)
nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,
           ax=None, title_fontsize=15, plot_start=0, title=init_method)

# json_name = os.path.join(path, 'FSI_D2_Proto_N_' + str( N_sim) +'_T_'+ str(t_sim) +  '_ms.json')
# write_obj_to_json(nuclei_dict, json_name)
save_all_mem_potential(nuclei_dict, path)
plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
# %% Effect of proto tau_m on STN-GPe-GPe frequency

# plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [400, int(t_sim/dt)]
name1 = 'STN'
name2 = 'Proto'
state = 'rest'
name_list = [name1, name2]

g_ext = 0.01
G = {}
plot_start = 1000
plot_start_raster = 1000

# g = -0.004
# G[(name1, name2)] , G[(name2, name1)] , G[(name2, name2)]= g , -g, g
# G = { k: v * K[k] for k, v in G.items()}

# relative_G =  G[(name2, name2)]**2 / ( G[(name2, name1)] * G[(name1, name2)] )
# print('relative G tau 43 = ', relative_G)

# neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
#                     'membrane_time_constant':{'mean':43,'var':10,  'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
# noise_variance['Proto'] = 100

g = - 0.006
G[(name1, name2)], G[(name2, name1)], G[(name2, name2)] = g, -g, g
G = {k: v * K[k] for k, v in G.items()}
relative_G = G[(name2, name2)]**2 / (G[(name2, name1)] * G[(name1, name2)])
print('relative G tau 13 = ', relative_G)
neuronal_consts['Proto'] = {'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial': {'min': -65, 'max': -37},  # Bogacz et al. 2016
                            'membrane_time_constant': {'mean': 12.94, 'var': 10, 'truncmin': 2, 'truncmax': 100}, 'spike_thresh': {'mean': -37, 'var': 5}}  # tau_m :#Projecting to STN from Karube et al 2019
noise_variance['Proto'] = 50


poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}

receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name1, '1'), (name2, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys())}

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True,
#                                           use_saved_FR_ext= False, normalize_G_by_N= True)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

# Run on previously saved data
FR_ext_all_nuclei = load_pickle(os.path.join(path,  'FR_ext_STN-Proto-Proto_tau_' + str(
    neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-') + '.pkl'))
receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                          set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=False,
                                          use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)


n_run = 1
n_iter = 1

peak_threshold = 0.1
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
check_stability = False
plot_sig = False
lower_freq_cut = 1
upper_freq_cut = 2000
freq_method = 'fft'
fft_method = 'Welch'
fig, ax = plt.subplots()
data = {}
for nucleus_list in nuclei_dict.values():
    nucleus = nucleus_list[0]  # get only on class from each population
    data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'peak_significance')] = np.zeros(
        (n_iter, n_run), dtype=bool)


for i in range(n_run):
    if i > 0:

        nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A,
                                              A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False,
                                              reset_init_dist=True, poisson_prop=poisson_prop, normalize_G_by_N=True)  # , mem_pot_init_method = 'uniform')
        n_FR = 20
        all_FR_list = {name: FR_ext_range[name]
                       for name in list(nuclei_dict.keys())}
        # receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
        #                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
        #                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True,
        #                                           use_saved_FR_ext= False, normalize_G_by_N= False)
        # pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
        FR_ext_all_nuclei = load_pickle(os.path.join(path,  'FR_ext_STN-Proto-Proto_tau_' + str(
            neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-') + '.pkl'))
        receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                                  all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                                  set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=False,
                                                  use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=False)

    nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
    smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)

    data = find_freq_all_nuclei(data, 0, i, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 3, smooth_window_ms, cut_plateau_epsilon,
                         check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=True, ax=ax,
                         c_spec=color_dict, n_windows=6, fft_method=fft_method, find_beta_band_power=True,
                         include_beta_band_in_legend=True, half_peak_range=5,
                         n_std=2, cut_off_freq=100, check_peak_significance=True)

    fig.set_size_inches((6, 5), forward=False)
    ax.set_xlim(0, 100)

# pickle_obj(data, os.path.join(path, 'STN-GP-GP-frequency-tau' +  str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

status = 'STN-GPe-GPe_tau_m_' + \
    str(neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-')

# fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title='',
           include_FR=True, include_std=True, plt_mvt=False, legend_loc='upper right', ylim=None)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.set_size_inches((10, 6), forward=False)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

# %% Effect of proto tau_m on STN-GPe frequency
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [400, int(t_sim/dt)]
name1 = 'STN'
name2 = 'Proto'
state = 'rest'
name_list = [name1, name2]

g_ext = 0.01
G = {}
plot_start = 1000
plot_start_raster = 1000

g = - 0.01
G[(name1, name2)], G[(name2, name1)] = g, -g

# neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
#                     'membrane_time_constant':{'mean':43,'var':10,  'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
# noise_variance['Proto'] = 100

neuronal_consts['Proto'] = {'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial': {'min': -65, 'max': -37},  # Bogacz et al. 2016
                            'membrane_time_constant': {'mean': 12.94, 'var': 10, 'truncmin': 2, 'truncmax': 100}, 'spike_thresh': {'mean': -37, 'var': 5}}  # tau_m :#Projecting to STN from Karube et al 2019
noise_variance['Proto'] = 50


poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}
G = {k: v * K[k] for k, v in G.items()}

receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name1, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False

n_run = 10
n_iter = 1

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}

peak_threshold = 0.1
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
check_stability = False
plot_sig = False
lower_freq_cut = 1
upper_freq_cut = 2000
freq_method = 'fft'
fft_method = 'Welch'
fig, ax = plt.subplots()
data = {}

for nucleus_list in nuclei_dict.values():
    nucleus = nucleus_list[0]  # get only on class from each population
    data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'peak_significance')] = np.zeros(
        (n_iter, n_run), dtype=bool)


for i in range(n_run):

    nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                   poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                   ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}

    n_FR = 20
    all_FR_list = {name: FR_ext_range[name]
                   for name in list(nuclei_dict.keys())}

    # receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
    #                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
    #                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True,
    #                                           use_saved_FR_ext= False, normalize_G_by_N= True)
    # pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

    # Run on previously saved data
    FR_ext_all_nuclei = load_pickle(os.path.join(path,  'FR_ext_STN-Proto_tau_' + str(
        neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-') + '.pkl'))
    receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                              all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                              set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=False,
                                              use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

    nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
    # smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)

    data = find_freq_all_nuclei(data, 0, i, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 3, smooth_window_ms, cut_plateau_epsilon,
                         check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=True, ax=ax,
                         c_spec=color_dict, n_windows=6, fft_method=fft_method, find_beta_band_power=True,
                         include_beta_band_in_legend=True, half_peak_range=5,
                         n_std=2, cut_off_freq=100, check_peak_significance=True)

    fig.set_size_inches((6, 5), forward=False)
    ax.set_xlim(0, 100)

pickle_obj(data, os.path.join(path, 'STN-GP-frequency-tau' +
           str(neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-') + '.pkl'))

status = 'STN-GPe_tau_m_' + \
    str(neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-')

fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title='',
           include_FR=True, include_std=True, plt_mvt=False, legend_loc='upper right', ylim=None)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.set_size_inches((10, 6), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
# %% Effect of proto tau_m on STN-GPe-FSI-D2 frequency

# plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name4 = 'STN'
state = 'rest'
name_list = [name1, name2, name3, name4]

g_ext = 0.01
G = {}
plot_start = int(t_sim / 2)
plot_start_raster = int(t_sim / 2)
g = -0.0035
G[(name2, name1)], G[(name3, name2)], G[(name1, name3)], G[(
    name3, name4)], G[(name4, name3)] = g, g, g, -g * 3, g * 3
G[(name2, name1)], G[(name3, name2)], G[(name1, name3)], G[(
    name3, name4)], G[(name4, name3)] = 0, 0, 0, -g * 3, g * 3

neuronal_consts['Proto'] = {'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial': {'min': -65, 'max': -37},  # Bogacz et al. 2016
                            'membrane_time_constant': {'mean': 43, 'var': 10,  'truncmin': 2, 'truncmax': 100}, 'spike_thresh': {'mean': -37, 'var': 5}}  # tau_m :#Projecting to STN from Karube et al 2019
noise_variance['Proto'] = 100

# neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
#                     'membrane_time_constant':{'mean':12.94,'var':10, 'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
# noise_variance['Proto'] = 50

G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}

receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name4, '1')],
                      (name4, '1'): [(name3, '1')]}
# (name3, '1'): [(name2,'1'), (name3, '1')]} # with GP-GP


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}

n_run = 1
n_iter = 1

peak_threshold = 0.1
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
check_stability = False
plot_sig = False
lower_freq_cut = 1
upper_freq_cut = 2000
freq_method = 'fft'
fft_method = 'Welch'
fig, ax = plt.subplots()
data = {}
for nucleus_list in nuclei_dict.values():
    nucleus = nucleus_list[0]  # get only on class from each population
    data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'peak_significance')] = np.zeros(
        (n_iter, n_run), dtype=bool)


for i in range(n_run):

    nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                   poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                   ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}
    n_FR = 20
    all_FR_list = {name: FR_ext_range[name]
                   for name in list(nuclei_dict.keys())}

    # receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
    #                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
    #                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True,
    #                                           use_saved_FR_ext= False, normalize_G_by_N= True)
    # pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-GPe-FSI-D2_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

    # ### Run on previously saved data
    FR_ext_all_nuclei = load_pickle(os.path.join(path,  'FR_ext_STN-GPe-FSI-D2_tau_' + str(
        neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-') + '.pkl'))
    receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                              all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                              set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=False,
                                              use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

    nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
    smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)

    data = find_freq_all_nuclei(data, 0, i, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 3, smooth_window_ms, cut_plateau_epsilon,
                         check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=True, ax=ax,
                         c_spec=color_dict, n_windows=6, fft_method=fft_method, find_beta_band_power=True,
                         include_beta_band_in_legend=True, half_peak_range=5,
                         n_std=2, cut_off_freq=100, check_peak_significance=True)

    fig.set_size_inches((6, 5), forward=False)
    ax.set_xlim(0, 100)


status = 'STN-GPe-FSI-D2_tau_m_' + \
    str(neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-')
pickle_obj(data, os.path.join(path, status + '.pkl'))

fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title='',
           include_FR=True, include_std=True, plt_mvt=False, legend_loc='upper right', ylim=None)

fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
# fig.set_size_inches((10, 6), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)

# %% Effect of Proto tau_m on GPe-GPe frequency

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]
name1 = 'Proto'
name_list = [name1]
state = 'rest'
g_ext = 0.01
G = {}
plot_start = int(t_sim/2)
plot_start_raster = int(t_sim/2)

# G[(name1, name1)] = -0.005#-0.025
# neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
#                     'membrane_time_constant':{'mean':43,'var':10,  'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
# noise_variance['Proto'] = 100


G[(name1, name1)] = -0.005
neuronal_consts['Proto'] = {'nonlin_thresh': -20, 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial': {'min': -65, 'max': -37},  # Bogacz et al. 2016
                            'membrane_time_constant': {'mean': 12.94, 'var': 10,  'truncmin': 2, 'truncmax': 100}, 'spike_thresh': {'mean': -37, 'var': 5}}  # tau_m :#Projecting to STN from Karube et al 2019
noise_variance['Proto'] = 50

G = {k: v * K[k] for k, v in G.items()}
poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}
receiving_pop_list = {(name1, '1'):  [(name1, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}

# n_FR = 20
# all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) }

# # receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
# #                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
# #                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True,
# #                                           use_saved_FR_ext= False, normalize_G_by_N= True)
# # pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

# #### Run on previously saved data
# FR_ext_all_nuclei  = load_pickle( os.path.join(path,  'FR_ext_Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
# receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False,
#                                           use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N= True)


n_run = 10
n_iter = 1

peak_threshold = 0.1
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
check_stability = False
plot_sig = False
lower_freq_cut = 1
upper_freq_cut = 2000
freq_method = 'fft'
fft_method = 'Welch'
fig, ax = plt.subplots()
data = {}
for nucleus_list in nuclei_dict.values():
    nucleus = nucleus_list[0]  # get only on class from each population
    data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'peak_significance')] = np.zeros(
        (n_iter, n_run), dtype=bool)


for i in range(n_run):

    # nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A,
    #                                       A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False,
    #                                       reset_init_dist= True, poisson_prop = poisson_prop, normalize_G_by_N= True)  # , mem_pot_init_method = 'uniform')
    nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                                   synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                                   poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                                   ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}

    n_FR = 20
    all_FR_list = {name: FR_ext_range[name]
                   for name in list(nuclei_dict.keys())}
    receiving_class_dict, FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                                                 all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                                                 set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=True,
                                                                 use_saved_FR_ext=False, normalize_G_by_N=True)
    pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Proto-Proto_tau_' +
               str(neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-') + '.pkl'))
    # FR_ext_all_nuclei  = load_pickle( os.path.join(path,  'FR_ext_Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
    # receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
    #                                   all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
    #                                   set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False,
    #                                   use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N= True)

    nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
    # smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)

    data = find_freq_all_nuclei(data, 0, i, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 3, smooth_window_ms, cut_plateau_epsilon,
                         check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=True, ax=ax,
                         c_spec=color_dict, n_windows=6, fft_method=fft_method, find_beta_band_power=True,
                         include_beta_band_in_legend=True, half_peak_range=5,
                         n_std=2, cut_off_freq=100, check_peak_significance=True)

    fig.set_size_inches((6, 5), forward=False)
    ax.set_xlim(0, 100)

pickle_obj(data, os.path.join(path, 'GP-GP-frequency-tau' +
           str(neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-') + '.pkl'))

status = 'GPe-GPe_tau_m_' + \
    str(neuronal_consts['Proto']['membrane_time_constant']).replace('.', '-')

# fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title='',
           include_FR=True, include_std=True, plt_mvt=False, legend_loc='upper right', ylim=None)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.set_size_inches((10, 6), forward=False)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

# %% plot freq vs. tau_m of Proto

# Proto-Proto
status = "GPe-GPe"
data_13 = load_pickle(os.path.join(
    path, "GP-GP-frequency-tau{'mean': 12-94, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))
data_43 = load_pickle(os.path.join(
    path, "GP-GP-frequency-tau{'mean': 43, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))

# STN-GPe-GPe
# status = "STN-GPe-GPe"
# data_13 = load_pickle(os.path.join(path, "STN-GP-GP-frequency-tau{'mean': 12-94, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))
# data_43 = load_pickle(os.path.join(path,"STN-GP-GP-frequency-tau{'mean': 43, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))

# STN-GPe
status = "STN-GPe"
data_13 = load_pickle(os.path.join(
    path, "STN-GP-frequency-tau{'mean': 12-94, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))
data_43 = load_pickle(os.path.join(
    path, "STN-GP-frequency-tau{'mean': 43, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))


name = 'Proto'
n_run = data_13[name, 'base_freq'].shape[1]
df_13_Proto = pd.DataFrame(
    {'f': data_13[name, 'base_freq'].reshape(n_run,), 'tau': [13] * n_run})
df_43_Proto = pd.DataFrame(
    {'f': data_43[name, 'base_freq'].reshape(n_run,), 'tau': [43] * n_run})
df = pd.concat([df_13_Proto, df_43_Proto], ignore_index=True)


my_pal = {13: "white", 43: "white"}
my_pal_pts = {13: "g", 43: "b"}
fig, ax = plt.subplots()
ax = sns.boxplot(x="tau", y="f", data=df,  palette=my_pal, width=0.3)
ax = sns.swarmplot(x="tau", y="f", data=df, palette=my_pal_pts)
ax.set_xlabel(r'$\tau_{m}$', fontsize=15)
ax.set_ylabel('Frequency (Hz)', fontsize=15)
ax.set_title(status, fontsize=18)

fig.savefig(os.path.join(path, 'Frequency_vs_Proto_tau_m_'+status+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


# %% phase histogram

## Polar phase histogram

low_f = 8
high_f = 20
plot_phase_histogram_all_nuclei(nuclei_dict, dt, color_dict, low_f, high_f, filter_order=6, height=0, density=False, n_bins=20,
                                start=int(t_sim/dt/2), projection='polar', total_phase=360, phase_ref='self')

## Linear phase histogram
low_f = 8
high_f = 20
plot_phase_histogram_all_nuclei(nuclei_dict, dt, color_dict, low_f, high_f, filter_order=6, height=0, density=False, n_bins=20,
                                start=int(t_sim/dt/2), projection=None, total_phase=720, phase_ref='Proto')
# %% Coherence

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_sim = 6000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'STN'
name2 = 'Proto'
state = 'rest'
name_list = [name1, name2]
g = -0.01

G = {}

G[(name1, name2)], G[(name2, name1)] = g, -g
G = {k: v * K[k] for k, v in G.items()}


poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': 0.01} for name in name_list}


receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name1, '1')]}
# (name2, '1'): [(name1,'1'), (name2,'1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = True
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
noise_tau = 20


class Nuc_keep_V_m(Nucleus):
    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = self.sum_synaptic_input(
            receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, t,
                              receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        self.reset_potential_with_interpolation(spiking_ind, dt)
        self.all_mem_pot[:, t] = self.mem_potential


nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}


receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


# x = np.array([0.0, 0.01, 0.015])
x = np.linspace(0, 0.015 ** 2, num=10)
x = np.sqrt(x)
n = len(x)

G_dict = {(name1, name2): -x,
          (name2, name1): x
          }
G_dict = {k: v * K[k] for k, v in G_dict.items()}

coherence = coherence_exploration(path, nuclei_dict, G_dict.copy(), noise_amplitude, noise_variance, A, N, N_real, K_real, receiving_pop_list,
                                  A_mvt, D_mvt, t_mvt, t_list, dt,  all_FR_list, n_FR, receiving_class_dict,
                                  end_of_nonlinearity, color_dict,
                                  poisson_prop, reset_init_dist=True, if_plot=False)

pickle_obj(coherence, os.path.join(path, 'coherence_' + str(dt).replace('.', '-') +
                                   '_T_' + str(t_sim) +
                                   '_N_' + str(N_sim) +
                                   '.pkl'))

# %% Plot Coherence

filename_list = ['coherence_0-1_T_6000_N_500.pkl',
                 ]
filename_list = [os.path.join(path, file) for file in filename_list]

name_list = ['STN', 'Proto']
for name in name_list:
    fig, ax = plt.subplots()
    for i, filename in enumerate(filename_list):
        coh = load_pickle(filename)
        ax.plot(multiply_values_of_dict(coh['G']), coh[name], '-o', label=name + ' (dt = ' + os.path.basename(filename).split('_')[1] + ')',
                )  # alpha = 1 - 0.2 * i)
    ax.set_ylabel('Coherence', fontsize=15)
    ax.set_xlabel(r'$G_{Loop}$')
    ax.legend(fontsize=10)


# %% Proto-Proto

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_sim = 3000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(1000/dt), int(t_sim/dt)]
name1 = 'Proto'
name_list = [name1]
n_windows = 2

state = 'rest' # set
g = -0.015 # rest

# state = 'DD_anesth' # set
# g = -0.007 # 'DD_anesth'

# state = 'awake_rest' # set
# g = -0.005 # 'awake_rest'

# state = 'mvt' # set
# g = -0.007 # 'mvt'

G = {}
plot_start = t_sim - 600
plot_start_raster = plot_start


G = {(name1, name1) :{'mean': g * K[name1, name1] }
      }
G = set_G_dist_specs(G, order_mag_sigma = 1)


poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}

receiving_pop_list = {(name1, '1'):  [(name1, '1')]}
pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 50; high_f = 70

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
low_f = 8
high_f = 80
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'GPe-GPe_' + state #'_G_PP_' + str(round(abs(G[('Proto', 'Proto')]), 1))
fig_sizes = {'firing': (10, 6),
             'raster': (11, 7),
             'spectrum': (6, 5)}

n_neuron = 25
firing_fig_ylims_dict = {'rest': [20, 65],
                         'awake_rest': [20, 70],
                         'DD_anesth': [0, 50],
                         'mvt': [0, 50]}

firing_fig_ylims = firing_fig_ylims_dict[state]
three_nuc_raster_y = (60 + 5) * 0.025

fig_sizes = {'firing': (5, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.025),
             'raster': (5, three_nuc_raster_y/3),
             'spectrum': (3, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.025)}


fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=plot_start, title='',  y_ticks = firing_fig_ylims,
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=firing_fig_ylims, label_fontsize = 8)

fig = remove_all_x_labels(fig)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
             size=fig_sizes['firing'])

include_nuc_name = False
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, tick_label_fontsize=12,
                                    title_fontsize=25, lw=0.6, linelengths=2, n_neuron=n_neuron, remove_whole_ax_frame = True,
                                    include_nuc_name=include_nuc_name, set_xlim=True, 
                                    remove_ax_frame=True, y_tick_length=0, x_tick_length=3)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
             size=fig_sizes['raster'])

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig_spec, ax = plt.subplots(1, 1)
freq, f, pxx = find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                                        smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                                        low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                                        spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=n_windows,
                                        include_beta_band_in_legend=False)

fig_spec = remove_all_x_labels(fig_spec)
# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0, 100)
# ax.yaxis.set_major_locator(MaxNLocator(2))
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status),
             size=fig_sizes['spectrum'])

# fig, ax = plt.subplots()
# check_significance_of_PSD_peak(f, pxx['Proto'],  n_std_thresh=2, min_f=0,
#                                max_f=250, n_pts_above_thresh=3, ax=ax, legend='Proto', c=color_dict['Proto'])


# phase_ref = 'Proto'
# find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
#                                   height = 0, phase_ref = phase_ref, start = 0, total_phase = 720,
#                                   only_entrained_neurons =False)
# fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt,
#                                     density = False, phase_ref = phase_ref, total_phase = 720, projection = None,
#                                     outer=None, fig=None,  title='', tick_label_fontsize=18,
#                                     labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

# %% STN-Proto + Proto-Proto
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto'  # projecting
name2 = 'STN'  # recieving
g = -0.003
g_ext = 0.01
G = {}
plot_start = 0
plot_start_raster = 500
G[(name2, name1)], G[(name1, name2)], G[(name1, name1)] = g, -g, g

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [(name2, '1'), (name1, '1')],
                      (name2, '1'): [(name1, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False

set_input_from_response_curve = True
save_init = False
noise_variance = {name1: 0.1, name2: 0.1, name3: 15}
noise_amplitude = {name1: 1, name2: 1, name3: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)

# filepaths = {name1: name1+ '_N_'+str(N_sim) +'_T_2000.pkl',
#              name2:name2 + '_N_'+str(N_sim) +'_T_2000.pkl',
#             name3: name3 + '_N_'+str(N_sim) +'_T_2000_noise_var_15.pkl'}
# filepaths = {'FSI': 'FSI_A_12-5_N_1000_T_2000_noise_var_0-1.pkl' ,
#              'D2': 'D2_A_1-1_N_1000_T_2000_noise_var_0-1.pkl' ,
#             'Proto': 'Proto_A_45_N_1000_T_2000_noise_var_15.pkl'}
filepaths = {'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl',
             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

set_init_all_nuclei(nuclei_dict, filepaths=filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise=False)

nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
# save_all_mem_potential(nuclei_dict, path)
# fig, ax = plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
# fig.savefig(os.path.join(path, 'V_m_Distribution_STN_GPe_'+mem_pot_init_method+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'V_m_Distribution_STN_GPe_'+mem_pot_init_method+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
state = 'STN_GPe_GPe-GPe_Real_tau_Proto_13_'

fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=plot_start, title='',
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=None)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
             size=(10, 6))

fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, labelsize=20,
                                    title_fontsize=25, lw=2, linelengths=2, n_neuron=40,
                                    include_title=True, set_xlim=True)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
             size=(11, 6))

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                         smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                         spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=3,
                         include_beta_band_in_legend=False)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0, 70)
save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_' + status),
             size=(6, 5))
# %% STN-Proto

plt.close('all')

N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim 
D_mvt = t_sim - t_mvt
duration = [int(1000/dt), int(t_sim/dt)]

n_windows = 3
name1 = 'STN'
name2 = 'Proto'
name_list = [name1, name2]


np.random.seed(1)
state = 'rest' # set
g = -0.029 # rest

# state = 'DD_anesth' # set
# g = -0.01 # 'DD_anesth'

# state = 'awake_rest' # set
# g = -0.014 # 'awake_rest'

# state = 'mvt' # set
# g = -0.015 # 'mvt'

G = {(name1, name2) :{'mean': g * K[name1, name2] },
      (name2, name1) :{'mean': -g * K[name2, name1]}
      }
G = set_G_dist_specs(G, order_mag_sigma = 1)

### homogeneous
# G[(name1, name2)], G[(name2, name1)] = g, -g
# G = {k: v * K[k] for k, v in G.items()}

plot_start = t_sim - 600
plot_start_raster = plot_start


poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}


receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name1, '1')]}

# print_G_items(G)
pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
hetero_trans_delays = True
hetero_tau = True
nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state, 
               hetero_trans_delay = hetero_trans_delays, hetero_tau = hetero_tau, Act  = Act) for i in pop_list] for name in name_list}

# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}
# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}


receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
# for name in list(nuclei_dict.keys()):

#     print('mean firing =', np.average(
#         nuclei_dict[name][0].pop_act[int(t_sim / 2):]))
#     figs = plot_exper_FR_distribution(xls, [name], [state_dict[state]],
#                                   color_dict, bins=np.arange(
#                                       0, bins[name][state]['max'], bins[name][state]['step']),
#                                   hatch='/', edgecolor='w', alpha=0.2, zorder=1)


#     fig_FR_dist = plot_FR_distribution({name: nuclei_dict[name]}, dt, color_dict, bins=np.arange(0, bins[name][state]['max'], bins[name][state]['step']),
#                                         ax=figs[name].gca(), alpha=1, zorder=0, start=int(t_sim / dt / 2))
#     save_pdf_png(fig_FR_dist, os.path.join(path, name + '_FR_dist_'),
#                   size=(6, 5))
nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)

# # fig, ax = plt.subplots()
# ax.plot(t_list * dt, nuclei_dict['STN'][0].voltage_trace[:-1], label = "dt=" + str(dt))
# # ax.plot(t_list * dt, nuclei_dict['STN'][0].representative_inp['ext_pop', '1'])
# ax.legend(fontsize = 15, loc= 'upper left')
# ax.set_title('Voltage', fontsize = 20)

status = 'STN-GPe_'  + state #+ '_slow'# '_G_SP_' + str(round(abs(G[('STN', 'Proto')]),1)) + '_G_PS_' + str(round(abs(G[('Proto', 'STN')]),1))

fig_sizes = {'firing': (10,6),
              'raster': (11,7),
              'spectrum': (6, 5)}

firing_fig_ylims_dict = {'rest': [0, 65],
                         'awake_rest': [0, 65],
                         'DD_anesth': [10, 55],
                         'mvt': [0, 55]}
firing_fig_ylims = firing_fig_ylims_dict[state]
three_nuc_raster_y = (60 + 5) * 0.025
fig_sizes = {'firing': (5, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.025),
             'raster': (5, three_nuc_raster_y/3 * 2),
             'spectrum': (3, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.025)}

fig = plot(nuclei_dict, color_dict, dt,  t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
            title_fontsize=15, plot_start=plot_start, title='', y_ticks = firing_fig_ylims,
            include_FR=False, include_std=False, plt_mvt=False,
            legend_loc='upper right', ylim=firing_fig_ylims)

fig = remove_all_x_labels(fig)
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#             title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
              size = fig_sizes['firing'])

n_neuron = 25
include_nuc_name = False
raster_order = ['Proto', 'STN']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, tick_label_fontsize=12,
                                    title_fontsize=25, lw=0.6, linelengths=2, n_neuron=n_neuron, remove_whole_ax_frame = True,
                                    include_nuc_name=include_nuc_name, set_xlim=True, name_list=raster_order,
                                    remove_ax_frame=True, y_tick_length=0, x_tick_length=3)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
             size=fig_sizes['raster'])

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig_spec, ax = plt.subplots(1, 1)
_, f, pxx = find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                                     smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                                     low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                                     spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=n_windows,
                                     include_beta_band_in_legend=False)

# fig_spec = remove_all_x_labels(fig_spec)

# # x_l = 0.75
# # ax.axhline(x_l, ls = '--', c = 'grey')
# # ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')
ax.set_xlim(0, 100)
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status ),
              size = fig_sizes['spectrum'])




# ax.set_xlim(0, 40)
# fig, ax = plt.subplots()
# check_significance_of_PSD_peak(f, pxx['STN'],  n_std_thresh = 2, min_f = 0, max_f = 250, n_pts_above_thresh = 3, ax = ax, legend = 'STN', c = color_dict['STN'])
# check_significance_of_PSD_peak(f, pxx['Proto'],  n_std_thresh = 2, min_f = 0, max_f = 250, n_pts_above_thresh = 3, ax = ax, legend = 'Proto', c = color_dict['Proto'])

# To see how removing the so-called non-entrained neurons will change the population mean firing rate
# for nuclei_list in nuclei_dict.values():
#     for nucleus in nuclei_list:
#         entrained_ind = significance_of_oscil_all_neurons( nucleus, dt, window_mov_avg = 10, max_f = 250,
#                                                           n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2)
#         print(nucleus.name, len(entrained_ind))
#         nucleus.pop_act = np.average(nucleus.spikes[entrained_ind,:], axis = 0)/(dt/1000)
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = '',
#            include_FR = False, include_std=False, plt_mvt=False, legend_loc='upper right', ylim =None)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(), title_fontsize=15, plot_start = plot_start, title = '',
#            include_FR = False, include_std=False, plt_mvt=False, legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

# phase_ref = 'Proto'
# low_f, high_f = 30, 60
# find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins=100,
#                                   height=0, phase_ref=phase_ref, start=0, total_phase=720,
#                                   only_entrained_neurons=False)
# fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt,
#                                     density=False, phase_ref=phase_ref, total_phase=720, projection=None,
#                                     outer=None, fig=None,  title='', tick_label_fontsize=18,
#                                     labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

# %% STN-Proto-Arky

plt.close('all')

N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.2
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim 
D_mvt = t_sim - t_mvt
duration = [int(300/dt), int(t_sim/dt)]

n_windows = 2
name1 = 'STN'
name2 = 'Proto'
name3 = 'Arky'
name_list = [name1, name2, name3]



state = 'rest' # set
g = -0.015 # rest

# state = 'DD_anesth' # set
# g = -0.01 # 'DD_anesth'

# state = 'awake_rest' # set
# g = -0.014 # 'awake_rest'

# state = 'mvt' # set
# g = -0.015 # 'mvt'

G = {(name1, name2) :{'mean': g * K[name1, name2] },
      (name2, name1) :{'mean': -g * K[name2, name1]},
      (name3, name2) :{'mean': g * K[name3, name2]}
      }

G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2)

### homogeneous
# G[(name1, name2)], G[(name2, name1)] = g, -g
# G = {k: v * K[k] for k, v in G.items()}

plot_start = t_sim - 600
plot_start_raster = plot_start


poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}


receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}
# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}


receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
# for name in list(nuclei_dict.keys()):

#     print('mean firing =', np.average(
#         nuclei_dict[name][0].pop_act[int(t_sim / 2):]))
#     figs = plot_exper_FR_distribution(xls, [name], [state_dict[state]],
#                                   color_dict, bins=np.arange(
#                                       0, bins[name][state]['max'], bins[name][state]['step']),
#                                   hatch='/', edgecolor='w', alpha=0.2, zorder=1)


#     fig_FR_dist = plot_FR_distribution({name: nuclei_dict[name]}, dt, color_dict, bins=np.arange(0, bins[name][state]['max'], bins[name][state]['step']),
#                                         ax=figs[name].gca(), alpha=1, zorder=0, start=int(t_sim / dt / 2))
#     save_pdf_png(fig_FR_dist, os.path.join(path, name + '_FR_dist_'),
#                   size=(6, 5))
nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)

# # fig, ax = plt.subplots()
# ax.plot(t_list * dt, nuclei_dict['STN'][0].voltage_trace[:-1], label = "dt=" + str(dt))
# # ax.plot(t_list * dt, nuclei_dict['STN'][0].representative_inp['ext_pop', '1'])
# ax.legend(fontsize = 15, loc= 'upper left')
# ax.set_title('Voltage', fontsize = 20)

status = 'STN-GPe_'  + state #+ '_slow'# '_G_SP_' + str(round(abs(G[('STN', 'Proto')]),1)) + '_G_PS_' + str(round(abs(G[('Proto', 'STN')]),1))

fig_sizes = {'firing': (10,6),
              'raster': (11,7),
              'spectrum': (6, 5)}

firing_fig_ylims_dict = {'rest': [0, 65],
                    'awake_rest': [0, 65],
                     'DD_anesth': [10, 55],
                      'mvt': [0, 55]}
firing_fig_ylims = firing_fig_ylims_dict[state]
three_nuc_raster_y = (60 + 5) * 0.05
fig_sizes = {'firing': (5, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05),
              'raster': (5, three_nuc_raster_y/3 * 2),
              'spectrum': (3, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05)}

fig = plot(nuclei_dict, color_dict, dt,  t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
            title_fontsize=15, plot_start=plot_start, title='',
            include_FR=False, include_std=False, plt_mvt=False,
            legend_loc='upper right', ylim=None)

# fig = remove_all_x_labels(fig)
# fig.axes[0].set_ylim(firing_fig_ylims)
# fig = set_y_ticks(fig, firing_fig_ylims)
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#             title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
              size = fig_sizes['firing'])

n_neuron = 50
include_nuc_name = False
raster_order = ['Proto', 'STN', 'Arky']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, tick_label_fontsize=12,
                                    title_fontsize=25, lw=1, linelengths=1, n_neuron=n_neuron,
                                    include_nuc_name=include_nuc_name, set_xlim=True, name_list=raster_order,
                                    remove_ax_frame=False, y_tick_length=2, x_tick_length=3)

fig_raster = remove_all_x_labels(fig_raster)
fig_raster = set_y_ticks(fig_raster, [0, n_neuron])
save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status ),
              size = fig_sizes['raster'])

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig_spec, ax = plt.subplots(1, 1)
_, f, pxx = find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                                     smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                                     low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                                     spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=n_windows,
                                     include_beta_band_in_legend=False)

# fig_spec = remove_all_x_labels(fig_spec)

# # x_l = 0.75
# # ax.axhline(x_l, ls = '--', c = 'grey')
# # ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')
ax.set_xlim(0, 100)
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status ),
              size = fig_sizes['spectrum'])


# %% FSI-D2-Proto

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 4000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(1000), int(t_sim/dt)]
# duration = [int(1000/dt), int(t_sim/dt)]
n_windows = 3
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name_list = [name1, name2, name3]
G = {}

state = 'rest' # set
g = -0.045 # G(DF) = g x 1.6

# state = 'DD_anesth' # set
# g = -0.03 # anesthetized G(DF) = g x 1.6

# state = 'awake_rest' # set
# g = -0.0075  # awake G(DF) = g x 1.6

# state = 'mvt' # set
# g = -0.0068 # 'mvt' G(DF) = g x 1.6

plot_start =  t_sim - 600
plot_start_raster = plot_start

G = {(name2, name1) :{'mean': g * K[name2, name1] * 2},
      (name1, name3) :{'mean': g * K[name1, name3]},
      (name3, name2) :{'mean': g * K[name3, name2]}
      }
G = set_G_dist_specs(G, order_mag_sigma = 1)

# G[(name2, name1)], G[(name3, name2)], G[(name1, name3)] = g * 1.6,  g , g
# G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                       (name2, '1'): [(name1, '1')],
                       (name3, '1'): [(name2, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True


nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
# for name in list(nuclei_dict.keys()):
#     print('I_0', np.average(nuclei_dict[name][0].rest_ext_input))
#     plot_FR_distribution({name: nuclei_dict[name]}, dt, color_dict, bins=np.arange(0, bins[name][state]['max'], 
#                                                                                    bins[name][state]['step']),  # ) bins = np.logspace(0,np.log10(50), 50),
#                          ax=None, alpha=1, zorder=0, start=int(t_sim / dt / 2),
#                          log_hist=False)  # , box_plot =True)

nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
low_f = 5 ; high_f = 20
status = 'FSI-D2-Proto_' +  state #('_G_DF_' + str(round(abs(G[('D2', 'FSI')]), 1)) +
                           #'_G_PD_' + str(round(abs(G[('Proto', 'D2')]), 1)) +
                           #'_G_FP_' + str(round(abs(G[('FSI', 'Proto')]), 1)))
n_neuron = 25
fig_sizes = {'firing': (10, 6),
             'raster': (11, 7),
             'spectrum': (6, 5)}

firing_fig_ylims_dict = {'rest': [-5, 60],
                         'awake_rest': [-5, 60],
                         'DD_anesth': [-5, 60],
                         'mvt': [-5, 60]}
firing_fig_ylims = firing_fig_ylims_dict[state]
fig_sizes = {'firing': (5, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.025),
             'raster': (5, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.025),
             'spectrum': (3, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.025)}

fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=plot_start, title='',  y_ticks = [0, firing_fig_ylims[1]],
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=firing_fig_ylims)

fig = remove_all_x_labels(fig)
# fig = plot(nuclei_dict,color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#             title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = low_f, high_f = high_f)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
             size=fig_sizes['firing'])

include_nuc_name = False
raster_order = ['Proto', 'FSI', 'D2']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, tick_label_fontsize=12,
                                    title_fontsize=25, lw=0.6, linelengths=2, n_neuron=n_neuron, remove_whole_ax_frame = True,
                                    include_nuc_name=include_nuc_name, set_xlim=True, name_list=raster_order,
                                    remove_ax_frame=True, y_tick_length=0, x_tick_length=3)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
             size=fig_sizes['raster'])

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig_spec, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                         smooth_window_ms, cut_plateau_epsilon, False, 'fft', False, 
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                         spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=n_windows,
                         include_beta_band_in_legend=False, normalize_spec=True)

# fig_spec = remove_all_x_labels(fig_spec)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')


ax.set_xlim(0, 100)
# ax.yaxis.set_major_locator(MaxNLocator(2))
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status),
             size=fig_sizes['spectrum'])


phase_ref = 'Proto'
nuclei_dict = find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins=36,
                                                peak_threshold=0, phase_ref= phase_ref, start=duration[0], total_phase=360,
                                                only_PSD_entrained_neurons= False, troughs = False,
                                                only_rtest_entrained = True, threshold_by_percentile = 50, 
                                                plot = False)
fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt,
                                    density=False, phase_ref=phase_ref, total_phase=720, projection=None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18,
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)
# plot_spike_amp_distribution(nuclei_dict, dt, color_dict, bins = 50

# %% Arky-D2-Proto

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 3000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(1000), int(t_sim/dt)]
name1 = 'Arky'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name_list = [name1, name2, name3]



state = 'rest' # set 
g = -0.03 # rest G(DA) = g x 1.6 hetero

# state = 'DD_anesth' # set 
# g = -0.03 # 'DD_anesth'  G(DA) = g x 1.6 hetero

# state = 'awake_rest' # set
# g = -0.0117 # 'awake_rest G(DA) = g x 1.6 hetero

# state = 'mvt' # set
# g = -0.01 # 'mvt' G(DA) = g x 1.6 hetero

G = {}
plot_start =  t_sim - 600
plot_start_raster = plot_start



G = {(name2, name1) :{'mean': g * K[name2, name1] * 5.},
      (name1, name3) :{'mean': g * K[name1, name3] },
      (name3, name2) :{'mean': g * K[name3, name2]}
      }
G = set_G_dist_specs(G, order_mag_sigma = 1)


poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}

receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 40

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
# save_all_mem_potential(nuclei_dict, path)

# fig, ax = plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
# fig.savefig(os.path.join(path, 'V_m_Distribution_all_nuclei_'+mem_pot_init_method+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'V_m_Distribution_all_nuclei_'+mem_pot_init_method+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'Arky-D2-Proto_' + \
        state
        # ('G_DA_' + str(round(abs(G[('D2', 'Arky')]), 1)) +
        #  'G_PD_' + str(round(abs(G[('Proto', 'D2')]), 1)) +
        #  'G_AP_' + str(round(abs(G[('Arky', 'Proto')]), 1)))
n_neuron = 25
fig_sizes = {'firing': (10, 6),
             'raster': (11, 7),
             'spectrum': (6, 5)}
firing_fig_ylims_dict = {'rest': [-5, 60],
                         'awake_rest': [-5, 60],
                         'DD_anesth': [-5, 60],
                         'mvt': [-5, 60]}

firing_fig_ylims = firing_fig_ylims_dict[state]
fig_sizes = {'firing': (5, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.025),
             'raster': (5, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.025),
             'spectrum': (3, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.025)}

fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=plot_start, title='',  y_ticks = [0, firing_fig_ylims[1]],
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=firing_fig_ylims)#, label_fontsize = 8)

fig = remove_all_x_labels(fig)
# fig = plot(nuclei_dict,color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#             title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = low_f, high_f = high_f)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
             size=fig_sizes['firing'])

include_nuc_name = False
raster_order = ['Proto', 'Arky', 'D2']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, tick_label_fontsize=12,
                                    title_fontsize=25, lw=0.6, linelengths=2, n_neuron=n_neuron, remove_whole_ax_frame = True,
                                    include_nuc_name=include_nuc_name, set_xlim=True, name_list=raster_order,
                                    remove_ax_frame=True, y_tick_length=0, x_tick_length=3)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
             size=fig_sizes['raster'])


peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig_spec, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                         smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                         spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=n_windows,
                         include_beta_band_in_legend=False)

fig_spec = remove_all_x_labels(fig_spec)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0, 100)
# ax.yaxis.set_major_locator(MaxNLocator(2))
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status),
             size=fig_sizes['spectrum'])
# %% FSI-D2-Proto + STN-GPe

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
# K = calculate_number_of_connections(N, N_real, K_all['DD_anesth'])

dt = 0.2
t_sim = 5300 
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(300/dt), int(t_sim/dt)]
n_windows = 4
name1 = 'FSI' 
name2 = 'D2' 
name3 = 'Proto'
name4 = 'STN'
state = 'awake_rest'
# state = 'DD_anesth'

name_list = [name1, name2, name3, name4]

# g_FSI_list = - np.linspace(0.15, 2., 18, endpoint=True)
# g_STN_list = - np.linspace(0.15, 2.2, 18, endpoint=True)

# examples_ind = {'A' : (4, 6), 'B': (16, 5),
#                 'C': (10,10), 'D': (14,11),
#                 'E' : (0, 17), 'F': (6, 17),
#                 'G':(14,17), 'H': (17, 17)

                # }
n = 14
g_STN_list = - np.linspace(1, 2.3, n, endpoint=True)
g_FSI_list = - np.linspace(1, 2.2, n, endpoint=True)

examples_ind = {'A' : (0, 0), 'B': (13, 0),
                'C': (0,1), 'D': (7,0),
                'E' : (0, 13), 'F': (1, 13),
                'G':(6,13), 'H': (13, 13)
                }
key = 'H'
g_FSI  = g_FSI_list[examples_ind[key][1]]
g_STN = g_STN_list[examples_ind[key][0]]
G = {}
plot_start = t_sim - 600
plot_start_raster = t_sim - 600

g = -.0015
# G[(name2, name1)], G[(name3, name2)], G[(name1, name3)], G[(name4, name3)], G[(name3, name4)] = g_FSI , g_FSI, g_FSI , g_STN, - g_STN
# G[(name2, name1)], G[(name3, name2)], G[(name1, name3)], G[(name3, name4)], G[(name4, name3)] = 3* g * 2 * 4/15, 3.5 * g* 2.8/1.1, 2.5 * g *21/46, 3. * -g *24/10 , 3. * g *2 * 21/46

( G[(name2, name1)], G[(name3, name2)], G[(name1, name3)], 
 G[(name3, name4)], G[(name4, name3)] ) = 3 * g * 1.9, 3.5 * g, 2.5 * g , 3. * -g * 1.6 , 3. * g * 2.

G = {k: v * K[k] for k, v in G.items()}

print_G_items(G)
# 3 * g * 1.9, 3.5 * g, 2.5 * g , 3. * -g * 1.6 , 3. * g * 2. --> 19.2 Hz /20 Hz
# 3 * g * 1.8, 3.5 * g, 2.5 * g , 3. * -g * 1.4 , 3. * g * 2. --> 18.4 Hz
# 3 * g * 1.8, 3.5 * g, 2.5 * g , 3. * -g * 1.3 , 3. * g * 2. --> 17.6 Hz
# 3 * g * 1.8, 3.5 * g, 2.5 * g , 3. * -g * 1.2 , 3. * g * 2 --> 16.8 Hz

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name4, '1')],
                      (name4, '1'): [(name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 40

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'Pallidostratal_plus_STN_GPe'

fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=plot_start, title= G_as_txt(G, display='normal', decimal=1) ,
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=None)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
              size=(10, 6))

raster_order = ['Proto', 'FSI', 'D2']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, tick_label_fontsize=12,
                                    title_fontsize=25, lw=1, linelengths=1, n_neuron=40,
                                    include_nuc_name=True, set_xlim=True, name_list=raster_order,
                                    remove_ax_frame=False, y_tick_length=2, x_tick_length=3)
save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
              size=(11, 6))

peak_threshold = 0.1; smooth_window_ms = 3; smooth_window_ms = 5; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig, ax = plt.subplots(1, 1)
_, f, pxx = find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                         smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                         spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=n_windows,
                         include_beta_band_in_legend=False, check_significance= True, plot_peak_sig = False,
                         min_f=  100, max_f = 300, plot_sig_thresh= True)
print('fft size = ', len(f))
# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0, 60)
ax.set_title(dt)
save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_' + status),
              size=(6, 5))

# pickle_obj({'time_series': nuclei_dict['Proto'][0].pop_act}, os.path.join(root, 'test_pop_act.pkl'))
# %% FSI-D2-Proto + GPe-GPe

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 5000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(1000/dt), int(t_sim/dt)]
name1 = 'FSI' 
name2 = 'D2' 
name3 = 'Proto'
state = 'awake_rest'
name_list = [name1, name2, name3]

g_FSI_list = - np.linspace(0.15, 2, 12, endpoint=True)
g_GPe_list = - np.linspace(0.15, 1.7, 12, endpoint=True)

g_FSI_list = - np.linspace(0.05, 2, 15, endpoint=True)
g_GPe_list = - np.linspace(0.05, 2, 15, endpoint=True)

examples_ind_02 = {'A' : (3, 4), 'B': (11, 4),
                 'C' : (0, 11), 'D': (6, 11),
                 'E': (11, 11), 'F': (1,11)}


examples_ind_01 = {'A' : (3, 5), 'B': (14, 5),
                     'C' : (0, 14), 'D': (7, 14),
                     'E': (14, 14)}

examples_ind = examples_ind_01
key = 'D'
g_FSI  = g_FSI_list[examples_ind[key][1]]
g_GPe = g_GPe_list[examples_ind[key][0]]
G = {}

plot_start = t_sim - 600
plot_start_raster = t_sim - 600

G[(name2, name1)], G[(name3, name2)], G[(name1, name3)], G[(name3, name3)] = g_FSI , g_FSI , g_FSI , g_GPe
# G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 40

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'FSI_Loop_plus_STN_GPe' + '_' + key + '_equal_g'
include_nuc_name = False
fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=0, title='',
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=None)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
# #             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
              size=(7, 3))

raster_order = ['Proto', 'FSI', 'D2', 'STN']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, tick_label_fontsize=12,
                                    title_fontsize=25, lw=1, linelengths=1, n_neuron=40,
                                    include_nuc_name=include_nuc_name, set_xlim=True, name_list=raster_order,
                                    remove_ax_frame=False, y_tick_length=2, x_tick_length=3)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
              size=(11, 6))

peak_threshold = 0.1; smooth_window_ms = 3; smooth_window_ms = 5; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig_spec, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                         smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                         spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=3,
                         include_beta_band_in_legend=False, check_significance= True, plot_peak_sig = False, 
                         min_f=  100, max_f = 300, plot_sig_thresh= True)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(0, 80)
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status),
              size=(6, 3))


# %% FSI-D2-Proto + Arky-D2-Proto + STN-GPe + GPe-GPe

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.2
t_sim = 3000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(400/dt), int(t_sim/dt)]
plot_start = t_sim - 400
plot_start_raster = plot_start

name1 = 'FSI' 
name2 = 'D2'  
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'

state = 'rest'
name_list = [name1, name2, name3, name4, name5]

g_ext = 0.01
G = {}

# g = -0.004
# (G[(name2, name1)], G[(name3, name2)],
#  G[(name1, name3)], G[(name2, name4)],
#  G[(name4, name3)], G[(name3, name5)],
#  G[(name5, name3)], G[(name3, name3)]) = 2*g, 0.5 * g, 0.5*g, g, g, -g * 3, g * 3, g * 0.1




g = -0.004
(G[(name2, name1)], G[(name3, name2)],
 G[(name1, name3)], G[(name2, name4)],
 G[(name4, name3)], G[(name3, name5)],
 G[(name5, name3)], G[(name3, name3)]) = 1.5* g, 0.5 * g, 1 * g, 1. * g, 1. * g, 3 * -g , 2 * g , g * 0.1 
 # G[(name5, name3)], G[(name3, name3)]) = g, 1.5*g, g, g, g, -g * 3, g*2.5, g * 0.1 


### Dec 2021 5 second, D2 Fr = 1.1/ state = 'rest'
# 1.5 * g, 1.5 * g, 1.5 * g, 1.5 * g, 1.5 * g, -g ,g , g * 0.1 --> 12.4 Hz g = 0.0045
# 2* g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 3 * -g , 1.5 * g , g * 0.1 --> 13.7 Hz g= 0.004
# g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 3 * -g , 1.5 * g , g * 0.1 --> 14.3 Hz g = 0.0045
# 2* g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 3.5 * -g , 2 * g , g * 0.1 -- > 15 Hz g = 0.004
# 2* g, 2.5 * g, 2 * g, 2. * g, 2 * g, 4 * -g , 2 * g , g * 0.1 -- > 15 Hz g = 0.004
# 2* g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 3 * -g , 2 * g , g * 0.1 -- > 16 Hz g = 0.004

### Dec 2021 5 second, D2 Fr = 1.1/ state = 'rest'


### Dec 2021 5 second, D2 Fr = 0.5, g = 0.006
# 1.5 * g, 1.5 * g, 1.5 * g, 1.5 * g, 1.5 * g, -g ,g , g * 0.1 --> 11.7 Hz 
# 1.4 * g, 1.4 * g, 1.4 * g, 1.1 * g, 1.1 * g, -g ,g , g * 0.1 --> 13 Hz 

# D2 RMP var = 1 t = 5s
# 2*g, 0.5*g, 0.5*g, g, g, -g * 3, g * 3, g * 0.1 --> 30 Hz
# g, 2*g, g, g, g, -g * 3, g*2.5 , g * 0.1 --> 18.8 Hz double peak with 35
# g, 1.5*g, g, g, g, -g * 3, g*2.5, g * 0.1 --> 18.8 Hz almost single peak
# g, 2*g, g, g, g, -g * 3, g*2, g * 0.1 --> 17.6 Hz single peak
# g, 1.8*g, g, g, g, -g * 3, g*1.5, g * 0.1 --> 16.9 Hz
# g, 2*g, g, g, g, -g * 3, g, g * 0.1 --> 15 Hz  works

G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}

receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1'), (name4, '1')],
                      # (name3, '1'): [(name2,'1'), (name5,'1')],
                      # with GP-GP
                      (name3, '1'): [(name2, '1'), (name3, '1'), (name5, '1')],
                      (name4, '1'): [(name3, '1')],
                      (name5, '1'): [(name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 40

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)

nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'All_loops'

fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=plot_start, title='',
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=None)

# fig = plot(nuclei_dict,color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
             size=(10, 6))

fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, labelsize=20,
                                    title_fontsize=25, lw=2, linelengths=2, n_neuron=40,
                                    include_title=True, set_xlim=True)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
             size=(11, 6))

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                         smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                         spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=3,
                         include_beta_band_in_legend=False)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0, 70)
save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_' + status),
             size=(6, 5))
phase_ref = 'D2'
find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins=100,
                                  height=0, phase_ref=phase_ref, start=0, total_phase=720,
                                  only_entrained_neurons=False, troughs=True)
fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, nuc_order=['D2', 'STN', 'Arky', 'Proto', 'FSI'],
                                    density=False, phase_ref=phase_ref, total_phase=720, projection=None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18,
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

# to see the heterogenity of D2 spikes
# nucleus = nuclei_dict['D2'][0]
# neurons = np.random.choice(nucleus.n, 1000, replace=False)
# spikes_sparse = create_sparse_matrix(nucleus.spikes[neurons, :], end=(
#     t_sim / dt), start=(plot_start / dt)) * dt
# ax = raster_plot(spikes_sparse, nucleus.name, color_dict,  ax=None, labelsize=10, title_fontsize = 15, lw =3 , linelengths = 2, orientation = 'vertical')

# %% Tune Gs from Brice (De la Crompe Supplementary Figure 6)
# 'ChR2_STN' : {'STN': [6, 26], 'Proto' : [34, 60], 'Arky' : [11, 7]
# plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.2
plot_start = 20
t_sim = plot_start + 500 + 2000 + 500
t_list = np.arange(int(t_sim/dt))
t_transient = 500 + plot_start
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(400/dt), int(t_sim/dt)]
plot_start_raster = plot_start

name1 = 'FSI' 
name2 = 'D2'  
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'

state = 'rest'
name_list = [name1, name2, name3, name4, name5]


G = {}
g = -0.0025 
G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.5}, ## free
      (name3, name2) :{'mean': g * K[name3, name2] * 3}, ## free
      (name1, name3) :{'mean': g * K[name1, name3] * 3.5}, ## free
      (name2, name4) :{'mean': g * K[name2, name4] * 3.5}, ## free
      (name4, name3) :{'mean': g * K[name4, name3] * 1.3},
      (name3, name5) :{'mean': -g * K[name3, name5] * 2.},
      (name5, name3) :{'mean': g * K[name5, name3] * 3.2},
      (name3, name3) :{'mean': g * K[name3, name3] * 0.2}
      }

G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2)

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
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 40

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state, external_input_bool = True) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                                       all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                       set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                                       use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)

### OFF vs. ON FRs
mod_dict = {'ChR2_STN' : {'STN': [6, 26], 'Proto' : [34, 55], 'Arky' : [11, 7]}} # DE la Crompe supp fig 6
            # 'ArchT_Proto': {'STN': [9, 52], 'Proto' : [57, 5], 'Arky' : [12, 44]},
            # 'ArchT_STN': {'STN': [96, 2], 'Proto' : [38, 22], 'Arky' : [0, 0]}}

duration = 2000
n_run = 2

# stim_nuc = 'Proto'
# stim = 'inhibition'

stim_nuc = 'STN'
stim = 'excitation'

ext_inp_dict = {stim_nuc: {'mean' : 0.29, 'sigma': .5 * .1, 'tau_rise': 0, 'tau_decay': 0} } # exponetial tau exit 6 only labeled

syn_trans_delay_dict = {stim_nuc : {'mean': 0, 'sd': 0, 'truncmin': 1, 'truncmax': 10}}
syn_trans_delay_dict = syn_trans_delay_homogeneous( syn_trans_delay_dict, dt, N_sim )

stim_method = 'ChR2'
ext_inp_method_trans = 'step'
# ext_inp_method_trans = 'exponential'

avg_act = average_multi_run_collective(path, tau, receiving_pop_list, receiving_class_dict, t_list, dt, nuclei_dict,  
                                       Act, G, N, N_real, K_real, K_all, syn_trans_delay_dict, poisson_prop,
                                        n_FR, all_FR_list, end_of_nonlinearity, t_transient=int(t_transient/dt), 
                                        duration=int(duration/dt), n_run=n_run, A_mvt=None, D_mvt=0, t_mvt=t_mvt, 
                                        ext_inp_dict=ext_inp_dict, noise_amplitude=noise_amplitude, noise_variance=noise_variance, 
                                        reset_init_dist=True, color_dict=color_dict, state = state,
                                        ext_inp_method = ext_inp_method_trans, stim_method = stim_method,
                                        homogeneous= True)



for nuclei_list in nuclei_dict.values():
    for k, nucleus in enumerate(nuclei_list):
        nucleus.pop_act = avg_act[nucleus.name][:, k]
        print(nucleus.name, np.average(nucleus.pop_act[ int(t_transient / dt):
                                                        int((t_transient + duration) / dt)]))
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=1)

status = 'All_loops_' + stim_nuc + '_' + stim + '_with_PP'

fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=plot_start, title='',
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=None)


save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
             size=(10, 6))

# fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
#                                     plot_start=plot_start_raster, plot_end=t_sim, labelsize=20,
#                                     title_fontsize=25, lw=2, linelengths=2, n_neuron=40,
#                                     include_title=True, set_xlim=True)

# save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
#              size=(11, 6))

# peak_threshold = 0.1
# smooth_window_ms = 3
# smooth_window_ms = 5
# cut_plateau_epsilon = 0.1
# lim_oscil_perc = 10
# low_pass_filter = False

# fig, ax = plt.subplots(1, 1)
# find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
#                          smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
#                          low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
#                          spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=3,
#                          include_beta_band_in_legend=False)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

# ax.set_xlim(0, 70)
# save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_' + status),
#              size=(6, 5))
# phase_ref = 'D2'min_in_dict(color_dict)
# find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins=100,
#                                   height=0, phase_ref=phase_ref, start=0, total_phase=720,
#                                   only_entrained_neurons=False, troughs=True)
# fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, nuc_order=['D2', 'STN', 'Arky', 'Proto', 'FSI'],
#                                     density=False, phase_ref=phase_ref, total_phase=720, projection=None,
#                                     outer=None, fig=None,  title='', tick_label_fontsize=18,
#                                     labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

# %% Beta induction FSI loop


plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration = [int(t_sim/dt/2), int(t_sim/dt)]
# duration = [int(1000/dt), int(t_sim/dt)]

name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name_list = [name1, name2, name3]
G = {}

state = 'rest' # set
g = -0.005 # G(DF) = g x 1.6

# state = 'DD_anesth' # set
# g = -0.008 # anesthetized G(DF) = g x 1.6

# state = 'awake_rest' # set
# g = -0.0075  # awake G(DF) = g x 1.6

# state = 'mvt' # set
# g = -0.0068 # 'mvt' G(DF) = g x 1.6

plot_start =  0
plot_start_raster = plot_start

G = {(name2, name1) :{'mean': g * K[name2, name1]  * 1.6},
      (name1, name3) :{'mean': g * K[name1, name3] },
      (name3, name2) :{'mean': g * K[name3, name2]}
      }

G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2)

# G[(name2, name1)], G[(name3, name2)], G[(name1, name3)] = g * 1.6,  g , g
# G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                       (name2, '1'): [(name1, '1')],
                       (name3, '1'): [(name2, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True


nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)

nuclei_dict['D2'][0].add_beta_ext_input(3, dt, freq = 20, start = int(500 / dt), end = None, mean = 0)
nuclei_dict = run(receiving_class_dict, t_list, dt,  nuclei_dict)

nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
low_f = 5 ; high_f = 20
status = 'FSI-D2-Proto_' +  state #('_G_DF_' + str(round(abs(G[('D2', 'FSI')]), 1)) +
                           #'_G_PD_' + str(round(abs(G[('Proto', 'D2')]), 1)) +
                           #'_G_FP_' + str(round(abs(G[('FSI', 'Proto')]), 1)))
n_neuron = 50
fig_sizes = {'firing': (10, 6),
             'raster': (11, 7),
             'spectrum': (6, 5)}

firing_fig_ylims_dict = {'rest': [-5, 60],
                    'awake_rest': [-5, 60],
                    'DD_anesth': [-5, 60],
                    'mvt': [-5, 60]}
firing_fig_ylims = firing_fig_ylims_dict[state]
fig_sizes = {'firing': (5, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.05),
             'raster': (5, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.05),
             'spectrum': (3, (firing_fig_ylims[1] - firing_fig_ylims[0]) * 0.05)}

fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
           title_fontsize=15, plot_start=plot_start, title='',
           include_FR=False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim=None)

# fig = remove_all_x_labels(fig)
fig.axes[0].set_ylim(firing_fig_ylims)
fig = set_y_ticks(fig, [0, 30, 60])
# fig = plot(nuclei_dict,color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax = plt.gca(),
#             title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False,
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = low_f, high_f = high_f)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status),
             size=fig_sizes['firing'])

include_nuc_name = False
raster_order = ['Proto', 'FSI', 'D2']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                    plot_start=plot_start_raster, plot_end=t_sim, tick_label_fontsize=12,
                                    title_fontsize=25, lw=1, linelengths=1, n_neuron=n_neuron,
                                    include_nuc_name=include_nuc_name, set_xlim=True, name_list=raster_order,
                                    remove_ax_frame=False, y_tick_length=2, x_tick_length=3)

# fig_raster = remove_all_x_labels(fig_raster)
fig_raster = set_y_ticks(fig_raster, [0, n_neuron])
save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status),
             size=fig_sizes['raster'])

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig_spec, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold, smooth_kern_window,
                         smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                         spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=3,
                         include_beta_band_in_legend=False, normalize_spec=True)

# fig_spec = remove_all_x_labels(fig_spec)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0, 100)
# ax.yaxis.set_major_locator(MaxNLocator(2))
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status),
             size=fig_sizes['spectrum'])


phase_ref = 'Proto'
find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins=180,
                                  height=0, phase_ref=phase_ref, start=duration[0], total_phase=720,
                                  only_entrained_neurons=False, troughs= True)
fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt,
                                    density=False, phase_ref=phase_ref, total_phase=720, projection=None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18,
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)
# plot_spike_amp_distribution(nuclei_dict, dt, color_dict, bins = 50

# %% Autocorrelation of individual neurons Demo
plt.close('all')
np.random.seed(10)
name = 'D2'
n_neuron = 4
window_ms = 10
t_lag = 200
nucleus = nuclei_dict[name][0]
n = nucleus.n
entrained_ind = significance_of_oscil_all_neurons(nuclei_dict[name][0], dt, window_mov_avg=10, max_f=250,
                                                  n_window_welch=6, n_sd_thresh=2, n_pts_above_thresh=2, min_f_sig_thres=0)

# entrained_ind_dble_thresh = significance_of_oscil_all_neurons( nuclei_dict[name][0], dt, window_mov_avg = 10, max_f = 250,
#                                       n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2, min_f_sig_thres = 0,
#                                       min_f_AUC_thres = 7,  PSD_AUC_thresh = 0.8*10**-5, filter_based_on_AUC_of_PSD = True)


# neuron_list = create_a_list_of_entrianed_and_not(entrained_ind, nucleus.n, n_entrained = 2, n_not_entrained = 2)
neuron_list = np.random.choice(get_complement_ind(entrained_ind, nucleus.n), 4)
# neurons = np.random.choice(nucleus.n, n_neuron)
spks = nucleus.spikes

spks = moving_average_array_2d(spks, int(window_ms / dt))
autc = autocorr_2d(spks)
f, pxx, peak_f = freq_from_welch_2d(autc, dt/1000, n_windows=6)
f, pxx = cut_PSD(f, pxx, max_f=250)


t_series = np.arange(int(t_lag / dt)) * dt
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
c_list = ['g', 'g', 'r', 'r']
line_style_list = ['-', '--', '-', '--']
n_sd_thresh = 2
signif_thresh = cal_sig_thresh(f, pxx, n_sd_thresh=2, min_f=7, max_f=250)
for n, i in enumerate(neuron_list):
    ax.plot(f, pxx[i, :], color=c_list[n], ls=line_style_list[n], marker='o')
    # ax.axhline(np.average(pxx[i,:]), f[0], f[-1], c = c_list[n])
    ax.axhline(signif_thresh[i], f[0], f[1], ls='--', c=c_list[n])
    ax1.plot(t_series, autc[i, :int(t_lag / dt)], color=c_list[n])

ax.set_xlim(0, 100)
ax.set_ylabel('PSD', fontsize=15)
ax.set_xlabel('Frequency (Hz)', fontsize=15)
ax1.set_ylabel('Autocorrelation', fontsize=15)
ax1.set_xlabel('lag (ms)', fontsize=15)
# %% Check effect of G=1 from pre to post synaptic FR

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'STN'
name2 = 'Proto'
state = 'rest'
g = -0.01
g_ext = 0.01
G = {}
plot_start = 0
plot_start_raster = 0

G[(name2, name1)] = 1

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [],
                      (name2, '1'): [(name1, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_amplitude = {name1: 1, name2: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}

n_FR = 20
all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys())}

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True,
#                                           use_saved_FR_ext= False)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Proto-STN.pkl'))


# Run on previously saved data
FR_ext_all_nuclei = load_pickle(os.path.join(path, 'FR_ext_Proto-STN.pkl'))
receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                          set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=False,
                                          use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

# nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
#                                       t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False, normalize_G_by_N=True)

t_transition = int(400 / dt)
name = 'STN'
A_new = 25
FR_list = [2.9/300, 3.8/300]
nuclei_dict[name][0].change_basal_firing(A_new)
FR_ext_new = nuclei_dict[name][0].set_ext_inp_const_plus_noise_collective(
    FR_list, t_list, dt, receiving_class_dict, end_of_nonlinearity=20, n_FR=n_FR)
nuclei_dict[name][0].change_basal_firing(A[name])
FR_ext_all_nuclei = load_pickle(os.path.join(path, 'FR_ext_Proto-STN.pkl'))
receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                          set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=False,
                                          use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=False)


def run(receiving_class_dict, t_list, dt, nuclei_dict, nuc_name, FR_ext_new, A, t_transition, A_mvt, D_mvt, t_mvt):

    start = timeit.default_timer()

    for t in t_list:
        if t == t_transition:
            # nuclei_dict[nuc_name][0].change_basal_firing(FR_ext_new)
            nuclei_dict[nuc_name][0].change_pop_firing_rate(FR_ext_new, A)
        for nuclei_list in nuclei_dict.values():
            for k, nucleus in enumerate(nuclei_list):
                mvt_ext_inp = np.ones((nucleus.n, 1)) * \
                    nucleus.external_inp_t_series[t]  # movement added
                nucleus.solve_IF(t, dt, receiving_class_dict[(
                    nucleus.name, str(k + 1))], mvt_ext_inp)

    stop = timeit.default_timer()
    print("t = ", stop - start)
    return nuclei_dict


nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict,
                  name, FR_ext_new, A, t_transition, A_mvt, D_mvt, t_mvt)

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)

status = 'STN-GPe'
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title='',
           include_FR=False, include_std=False, plt_mvt=False, legend_loc='upper right', ylim=None)
round_dec = 1
for k, nuclei_list in enumerate(nuclei_dict.values()):
    for j, nucleus in enumerate(nuclei_list):
        FR_mean, FR_std = nucleus. average_pop_activity(0, t_transition)
        txt = r"$\overline{{FR_{{{0}}}}}$ ={1} ".format(
            nucleus.name,  round(FR_mean, round_dec))

        fig.gca().text(0.05, 0.48 * (k+1), txt, ha='left', va='center', rotation='horizontal', fontsize=15, color=color_dict[nucleus.name],
                       transform=fig.gca().transAxes)

        FR_mean, FR_std = nucleus. average_pop_activity(
            t_transition, len(t_list))
        txt = r"$\overline{{FR_{{{0}}}}}$ ={1} ".format(
            nucleus.name,  round(FR_mean, round_dec))

        fig.gca().text(0.5, 0.48 * (k+1), txt, ha='left', va='center', rotation='horizontal', fontsize=15, color=color_dict[nucleus.name],
                       transform=fig.gca().transAxes)
plt.axvline(t_transition * dt, linestyle='--')
fig.savefig(os.path.join(path, 'G_STN_Proto_equal_to_1_'+status+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
# %% effect of transient increase in STN activity onto GPe (with/without GABA-B)
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
sdt = 0.25
t_sim = 800
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto'  # projecting
name2 = 'STN'  # recieving
g = -0.008
g_ext = 0.01
G = {}
plot_start = 600
plot_start_raster = 600
G[(name2, name1)], G[(name1, name2)] = g, -g
# G[(name2, name1)] , G[(name1, name2)]  = 0,0
# Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner
tau[('STN', 'Proto')] = {'rise': [40], 'decay': [200]}
# tau[('STN','Proto')] =  {'rise':[1.1],'decay':[7.8]} # Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name1, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False

set_input_from_response_curve = True
save_init = False
noise_variance = {name1: 0.1,  name2: 15}
noise_amplitude = {name1: 1,  name2: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)

rest_init_filepaths = {'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl',
                       # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
                       'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}
# 'Proto': 'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl'}

trans_init_filepaths = {'STN': 'tau_m_5-13_STN_A_100_N_1000_T_2000_noise_var_10.pkl',
                        'Proto': rest_init_filepaths['Proto']}
t_transient = 600  # ms
duration = 5
n_run = 10
syn_trans_delay_dict = {'STN': 0}
set_init_all_nuclei(nuclei_dict, filepaths=rest_init_filepaths)

nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise=False)

# run_with_transient_external_input(receiving_class_dict,t_list, dt, nuclei_dict, rest_init_filepaths, trans_init_filepaths, Act['rest'],
# 										Act['trans'],list_of_nuc_with_trans_inp, t_transient = int( t_transient / dt), duration = int( duration / dt))

# nuc1[0].low_pass_filter( dt, 1,200, order = 6)
# nuc2[0].low_pass_filter( dt, 1,200, order = 6)
# # smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)

avg_act = average_multi_run(receiving_class_dict, t_list, dt, nuclei_dict, rest_init_filepaths, Act['rest'], Act['trans'],
                            syn_trans_delay_dict, t_transient=int(t_transient / dt), transient_init_filepaths=trans_init_filepaths,
                            duration=int(duration / dt), n_run=n_run)
for nuclei_list in nuclei_dict.values():
    for k, nucleus in enumerate(nuclei_list):
        nucleus.pop_act = avg_act[nucleus.name][:, k]
state = 'Only_STN_GPe_Real_tau_Proto_13_ms_only_STN-Proto_trans_Ctx_' + \
    str(n_run) + '_run'
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=20, plot_start=plot_start,
           title=r'$\tau_{{m}}^{{Proto}} = 13\;ms\; , \; G={0}$'.format(g), plt_mvt=False, include_FR=False, ylim=[0, 150])
# fig.set_size_inches((15, 7), forward=False)
plt.axvspan(t_transient, (t_transient + duration), alpha=0.2, color='yellow')
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi=500, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)


# fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start_raster, plot_end = t_sim,
#                             labelsize = 20, title_fontsize = 25, lw  = 2, linelengths = 2, n_neuron = 60, include_title = True, set_xlim=True)
# fig_.set_size_inches((11, 7), forward=False)
# fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig, ax = plt.subplots(1,1)
# peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
# find_freq_all_nuclei(dt, nuclei_dict, duration_2, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False ,
#                 low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False,
#                 fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
# # x_l = 0.75
# # ax.axhline(x_l, ls = '--', c = 'grey')
# ax.set_xlim(0,55)
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')
# fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

# %% effect of transient increase in STN activity onto GPe (with/without GABA-B) collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_transient = 350
t_sim = t_transient + 150
plot_start = t_transient - 50
plot_start_raster = plot_start
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
name1 = 'Proto'
name2 = 'STN'
name3 = 'Arky'
state = 'trans_Nico_mice'
name_list = [name1, name2]#, name3]
G = {}

# syn_coef_GABA_b = 1
# syn_component_weight = {
#     ('D2', 'FSI'): [1],
#     # the relative weight of the GABA-a and GABA-b components
#     ('STN', 'Proto'): [1, syn_coef_GABA_b],
#     ('Proto', 'STN'): [1],
#     ('Proto', 'Proto'): [1, syn_coef_GABA_b],
#     ('Proto', 'D2'): [1],
#     ('FSI', 'Proto'): [1],
#     ('Arky', 'Proto'): [1, syn_coef_GABA_b],
#     ('D2', 'Arky'): [1]
# }

peak_jiggle_dict = {'STN': 10,
                    'Proto':0.5}

note = '_tau_PS_1-8_G_diverse' 
STN_shift = 2
proto_shift = -5 + STN_shift 
proto_shift =  STN_shift 

G[(name2, name1)], G[(name1, name2)], G[(name3, name1)],  G[(name1, name1)] = -0.00, 0.01, -0.00, -0.000
ext_inp_dict = {'STN': {'mean' : 70, 'sigma': .5 * .1, 'tau_rise': 5, 'tau_decay': 4} } # exponetial tau 2 merged


G = {(name2, name1) :{'mean': 0 * K[name2, name1] },
      (name1, name2) :{'mean': 0.01 * K[name1, name2] },
      (name3, name1) :{'mean': 0 * K[name3, name1]},
      (name1, name1) :{'mean': 0 * K[name1, name1]}

      }
G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2)


# note = '_tau_PS_1-8' 
# STN_shift = 2
# proto_shift = -5 + STN_shift 

# G[(name2, name1)], G[(name1, name2)], G[(name3, name1)],  G[(name1, name1)] = -0.00, 0.020, -0.00, -0.000
# ext_inp_dict = {'STN': {'mean' : 60., 'sigma': .5 * .1, 'tau_rise': 1000000, 'tau_decay': 4} } # exponetial tau 2 merged
# G = {k: v * K[k] for k, v in G.items()}

# note = '_tau_PS_1-8_with_PS' 
# STN_shift = -1
# tau[('Proto', 'STN')]['decay'] = {'mean' : [1.81], 'sd' : [2.5], 'truncmin': [0.43], 'truncmax': [6.86]}
# # G[(name2, name1)], G[(name1, name2)], G[(name3, name1)],  G[(name1, name1)] = -0.0019, 0.022, -0.001, -0.000
# G[(name2, name1)], G[(name1, name2)], G[(name3, name1)],  G[(name1, name1)] = -0.0019, 0.017, -0.001, -0.000
# ext_inp_dict = {'STN': {'mean' : 130., 'sigma': .5 * .1, 'tau_rise': 1000, 'tau_decay': 10} } # exponetial tau 2 merged


# note = '_tau_PS_1-8_with_PS_PP' 
# STN_shift = -1
# tau[('Proto', 'STN')]['decay'] = {'mean' : [1.81], 'sd' : [2.5], 'truncmin': [0.43], 'truncmax': [6.86]}
# G[(name2, name1)], G[(name1, name2)], G[(name3, name1)],  G[(name1, name1)] = -0.0019, 0.022, -0.001, -0.0004
# ext_inp_dict = {'STN': {'mean' : 130., 'sigma': .5 * .1, 'tau_rise': 1000, 'tau_decay': 10} } # exponetial tau 2 merged

# note = '_tau_PS_1-8_with_PP' 
# STN_shift = 0
# tau[('Proto', 'STN')]['decay'] = {'mean' : [1.81], 'sd' : [2.5], 'truncmin': [0.43], 'truncmax': [6.86]}
# G[(name2, name1)], G[(name1, name2)], G[(name3, name1)],  G[(name1, name1)] = -0.00, 0.02, -0.001, -0.001
# ext_inp_dict = {'STN': {'mean' : 70., 'sigma': .5 * .1, 'tau_rise': 1000, 'tau_decay': 5} } # exponetial tau 2 merged



poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'): [(name2, '1')],# (name1, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name1, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True


nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state, external_input_bool = True) for i in pop_list] for name in name_list}


n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)



duration = 10
n_run =  1

# ext_inp_dict = {'STN': {'mean' : .5, 'sigma': .5 * .1, 'tau_rise': 0, 'tau_decay': 0} } # step input
# ext_inp_dict = {'STN': {'mean' : 70., 'sigma': .5 * .1, 'tau_rise': 100, 'tau_decay': 5} } # exponetial tau 6 merged
# ext_inp_dict = {'STN': {'mean' : 70., 'sigma': .5 * .1, 'tau_rise': 250, 'tau_decay': 4} } # exponetial tau 6 only labeled




syn_trans_delay_dict = {'STN' : {'mean': 0, 'sd': 0, 'truncmin': 1, 'truncmax': 10}}
syn_trans_delay_dict = syn_trans_delay_homogeneous( syn_trans_delay_dict, dt, N_sim )

stim_method = 'ChR2'
ext_inp_method = 'step'
ext_inp_method = 'exponential'

avg_act = average_multi_run_collective(path, tau, receiving_pop_list, receiving_class_dict, t_list, dt, nuclei_dict,  
                                       Act, G, N, N_real, K_real, K_all, syn_trans_delay_dict, poisson_prop,
                                        n_FR, all_FR_list, end_of_nonlinearity, t_transient=int(t_transient/dt), 
                                        duration=int(duration/dt), n_run=n_run, A_mvt=None, D_mvt=0, t_mvt=t_mvt, 
                                        ext_inp_dict=ext_inp_dict, noise_amplitude=noise_amplitude, noise_variance=noise_variance, 
                                        reset_init_dist=True, color_dict=color_dict, state = state,
                                        ext_inp_method = ext_inp_method, stim_method = stim_method)




for nuclei_list in nuclei_dict.values():
    for k, nucleus in enumerate(nuclei_list):
        nucleus.pop_act = avg_act[nucleus.name][:, k]

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=1)
fig = plot(nuclei_dict, color_dict, dt, t_list - np.full_like(t_list, t_transient / dt), Act[state], 
           A_mvt, t_mvt, D_mvt, title_fontsize=20, plot_start= plot_start, 
           title="", plt_mvt=False, include_FR=False , ylim = [-10,160], alpha = 0.2)

fig = plot_extermums_FR(nuclei_dict, peak_jiggle_dict, t_transient, dt, color_dict, fig, alpha = 0.5, smooth_kern_window=5)


ax = fig.gca()
# ax.axvspan(t_transient, (t_transient + duration), alpha=0.2, color='yellow')
# ax.axvline(t_transient, *ax.get_ylim(), c= 'grey', ls = '--', lw = 1)
# ax.axvline(t_transient + duration, *ax.get_ylim(), c= 'grey', ls = '--', lw = 1)
# ax.axhline(0, c= 'grey', ls = ':')
# save_pdf_png(fig, os.path.join(path, 'STN_trans_stim_' + str(duration) +'ms_onto_Proto' + str(n_run) + '_run'), size = (5,3))


######## Plot experimental on top
filename = 'STN-10ms_OptoStimData_RecSTN-Proto-Arky_merged.xlsx'
sheet_name_extra = ''
# filename = 'STN-10ms_OptoStimData_RecSTN-Proto-Arky_OnlyLabelled.xlsx'
# sheet_name_extra = 'Label'
FR_df = read_sheets_of_xls_data(filepath = os.path.join( root, 'Modeling_Data_Nico', 
                                                        'Transient_stim', filename))

fig, ax, title = plot_fr_response_from_experiment(FR_df, filename, color_dict, xlim = None, ylim = None, stim_duration = 10, 
                                           ax = None, time_shift_dict = {'STN': STN_shift, 'Arky': 0, 'Proto': proto_shift}, 
                                           sheet_name_extra = sheet_name_extra)
ax.set_title( title + ' ' + note, fontsize = 15)
ax.set_xlim(-15, 90)
save_pdf_png(fig, os.path.join(path, 'STN_trans_stim_onto_Proto_' + str(n_run) + 
                               '_run_compared_with_experiment_' + 
                               filename.split('.')[0].split('_')[-1] + note), size = (5,3))
# %% exp rise and decay test

x = np.linspace(0, 5000, num = 5000)

y =  (1 -np.exp(-(x) / 10000000)) * (np.exp(- (x) / 200))

# fig, ax = plt.subplots()
ax.plot(x * .1, y/ np.trapz(y, x))

# ax.plot(500- x * np.log(x/ ( 100 + x)))


# %% effect of MC-induced transient input on a STR-GPe-STN network single neuron

# plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 400
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto'  # projecting
name2 = 'STN'  # recieving
name3 = 'D2'
g = -0.004
g_ext = 0.01
G = {}

# Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner
tau[('STN', 'Proto')] = {'rise': [1.1, 40], 'decay': [7.8, 200]}

plot_start = 150
plot_start_raster = 500
G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)] = -.001, 0.002, -0.001
# G[(name2, name1)] , G[(name1, name2)] ,  G[(name1, name3)]  = 0,0, 0

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name3: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [(name2, '1'), (name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): []
                      }

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_variance = {name1: 0.1,  name2: 15, name3: .1}
noise_amplitude = {name1: 1,  name2: 1, name3: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)

rest_init_filepaths = {
    'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl',
    'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
    # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
    'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}
# 'Proto': 'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl'}

trans_init_filepaths = {
    # 'STN':'tau_m_5-13_STN_A_46_N_1000_T_2000_noise_var_5.pkl',
    'STN': 'tau_m_5-13_STN_A_65_N_1000_T_2000_noise_var_5.pkl',
    'Proto': rest_init_filepaths['Proto'],
    # 'D2':'tau_m_13_D2_A_30_N_1000_T_2000_noise_var_20.pkl',
    'D2': 'tau_m_13_D2_A_23_N_1000_T_2000_noise_var_12.pkl',
    # 'D2' : 'tau_m_13_D2_A_20_N_1000_T_2000_noise_var_10.pkl',
}
t_transient = 200  # ms
duration = 5
n_run = 1
list_of_nuc_with_trans_inp = ['STN', 'D2']


syn_trans_delay_dict_STN = {k[0]: v for k,
                            v in T.items() if k[0] == 'STN' and k[1] == 'Ctx'}
syn_trans_delay_dict_STR = {k[0]: v for k,
                            v in T.items() if k[0] == 'D2' and k[1] == 'Ctx'}
syn_trans_delay_dict = {**syn_trans_delay_dict_STN, **syn_trans_delay_dict_STR}
syn_trans_delay_dict = {k: v / dt for k, v in syn_trans_delay_dict.items()}


set_init_all_nuclei(nuclei_dict, filepaths=rest_init_filepaths)

nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise=False)


# nuc1[0].low_pass_filter( dt, 1,200, order = 6)
# nuc2[0].low_pass_filter( dt, 1,200, order = 6)

avg_act = average_multi_run(receiving_class_dict, t_list, dt, nuclei_dict, rest_init_filepaths, Act['rest'], Act['trans'],
                            syn_trans_delay_dict, t_transient=int(t_transient / dt),  transient_init_filepaths=trans_init_filepaths,
                            duration=int(duration / dt), n_run=n_run)

for nuclei_list in nuclei_dict.values():
    for k, nucleus in enumerate(nuclei_list):
        nucleus.pop_act = avg_act[nucleus.name][:, k]
state = 'STN_GPe_D2_Real_tau_Proto_13_ms_trans_Ctx_' + \
    str(n_run) + '_trans_delay_not_included_run_tau_SP_6'
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=20, plot_start=plot_start,
           title=r'$\tau_{{m}}^{{Proto}} = 13\;ms\; , \; G={0}, \; \tau_{{SP}}=12$'.format(g), plt_mvt=False, include_FR=False)  # , ylim = [0,150])
fig.set_size_inches((15, 7), forward=False)
plt.axvspan(t_transient, (t_transient + duration), alpha=0.2, color='yellow')
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 500, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

# %% effect of MC-induced transient input on a STR-GPe-STN network collective


plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_transient = 200
t_sim = t_transient + 500
plot_start = t_transient - 50
plot_start_raster = plot_start
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

name1 = 'Proto' 
name2 = 'STN' 
name3 = 'D2'
state = 'trans_Kita_rat'
name_list = [name1, name2, name3]
G = {}

# stim_method = 'ChR2'
stim_method = 'Projection'
# ext_inp_method = 'step'
ext_inp_method_trans = 'exponential'

duration = 10
n_run = 1
duration_fft = [int((t_transient+100)/dt), int(t_sim/dt)]

list_of_nuc_with_trans_inp = ['STN', 'D2']

ext_inp_dict = {'STN': {'mean' : 0.4, 'sigma': 0.4 * .1, 'tau_rise': 100, 'tau_decay': 5}, #### step
                'D2': {'mean' : 2, 'sigma': 2 * .1, 'tau_rise': 100, 'tau_decay': 5}}

ext_inp_dict = {'STN': {'mean' : 40, 'sigma': 0.4 * .1, 'tau_rise': 100, 'tau_decay': 5}, #### exponential
                'D2': {'mean' : 150, 'sigma': 2 * .1, 'tau_rise': 100, 'tau_decay': 4}}

# ext_inp_dict = {'STN': {'mean' : 60, 'sigma': 0.4 * .1, 'tau_rise': 100, 'tau_decay': 5}, #### exponential high FR ext sigma
#                 'D2': {'mean' : 230, 'sigma': 2 * .1, 'tau_rise': 100, 'tau_decay': 4}}

T[('D2', 'Ctx')]['mean'] = 11

peak_jiggle_dict = {'D2': 15, 
               'STN': 0.5,
               'Proto': 0.4}

note = '_tau_PS_1-8'
# tau[('Proto', 'D2')] =  {'rise': {'mean' : [0.2], 'sd' : [0.06], 'truncmin': [0.1], 'truncmax': [100]},
#                       'decay' : {'mean' :[1000], 'sd' : [.38], 'truncmin': [0.1], 'truncmax': [100]}}
T[('Proto', 'D2')] ={'mean': 2, 'sd' : 0.35, 'truncmin': 0.1, 'truncmax': 11.3}     

G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)], G[(name1, name1)] = -0.008, 0.006, -0.0002, -0.000  # tau 25
G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)], G[(name1, name1)] = -0.008, 0.006, -0.0004, -0.000  # tau 4

# G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)], G[(name1, name1)] = -0.0095, 0.006, -0.0002, -0.000  # high FR ext sigma
# G = {key: {'mean': v * K[key], 'sd': 2 , 'truncmin': 0, 'truncmax': 10}
#      for key, v in G.items()}
G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name2, '1'), (name3, '1'), (name1, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): []
                      }

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True


nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state, external_input_bool = True) for i in pop_list] for name in name_list}


n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)

syn_trans_delay_dict = filter_transmission_delay_for_downstream_projection(T, list_of_nuc_with_trans_inp, 
                                                                           projecting = 'Ctx')

if stim_method == 'Projection':
    syn_trans_delay_dict = syn_trans_delay_heterogeneous( syn_trans_delay_dict, dt, N_sim )
    
elif stim_method == 'ChR2':
    syn_trans_delay_dict = syn_trans_delay_homogeneous( syn_trans_delay_dict, dt, N_sim )
    
avg_act = average_multi_run_collective(path, tau, receiving_pop_list, receiving_class_dict, t_list, dt, nuclei_dict,  
                                       Act[state], G, N, N_real, K_real, syn_trans_delay_dict, poisson_prop,
                                        n_FR, all_FR_list, end_of_nonlinearity, t_transient=int(t_transient/dt), 
                                        duration=int(duration/dt), n_run=n_run, A_mvt=None, D_mvt=0, t_mvt=t_mvt, 
                                        ext_inp_dict=ext_inp_dict, noise_amplitude=None, noise_variance=None, 
                                        reset_init_dist=True, color_dict=color_dict, state = state, plot = False, 
                                        ext_inp_method =ext_inp_method_trans, stim_method = stim_method)


for nuclei_list in nuclei_dict.values():
    for k, nucleus in enumerate(nuclei_list):
        nucleus.pop_act = avg_act[nucleus.name][:, k]

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'trans_Ctx_to_STN_and_D2_' + str(n_run) + '_run_'

fig = plot(nuclei_dict, color_dict, dt, t_list - np.full_like(t_list, t_transient / dt), 
           Act[state], A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=20, plot_start=plot_start, figsize=(20,4),
           title="", plt_mvt=False, include_FR=False, xlim = (-50, t_sim - t_transient))  # , ylim = [0,150])]

fig = plot_extermums_FR(nuclei_dict, peak_jiggle_dict, t_transient, dt, color_dict, fig )

fig = set_x_ticks(fig, np.arange(0, 600, 100))
fig = set_y_ticks(fig, np.arange(0, 175, 25))
fig.gca().set_ylim(-10, 150)
fig.gca().axvspan(0, duration, alpha=0.2, color='yellow')
save_pdf_png(fig, os.path.join(path, 'MC_trans_stim_' + str(duration) +'ms_onto_D2-STN-Proto' + note), size = (8,3.5))

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig_spec, ax = plt.subplots(1, 1)
_, f, pxx = find_freq_all_nuclei(dt, nuclei_dict, duration_fft, lim_oscil_perc, peak_threshold, smooth_kern_window,
                                     smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                                     low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict,
                                     spec_figsize=(6, 5), find_beta_band_power=False, fft_method='Welch', n_windows=2,
                                     include_beta_band_in_legend=False)
ax.set_xlim(0, 100)

# %% effect of MC-induced transient input on a STR-GPe-STN network collective varying taus

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_transient = 200
t_sim = t_transient + 500
plot_start = t_transient - 50
plot_start_raster = plot_start
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

name1 = 'Proto' 
name2 = 'STN' 
name3 = 'D2'
state = 'trans_Kita_rat'
name_list = [name1, name2, name3]
G = {}

# stim_method = 'ChR2'
stim_method = 'Projection'
# ext_inp_method = 'step'
ext_inp_method_trans = 'exponential'

duration = 10
duration_fft = [int((t_transient+100)/dt), int(t_sim/dt)]

list_of_nuc_with_trans_inp = ['STN', 'D2']

# ext_inp_dict = {'STN': {'mean' : 0.4, 'sigma': 0.4 * .1, 'tau_rise': 100, 'tau_decay': 5}, #### step
#                 'D2': {'mean' : 2, 'sigma': 2 * .1, 'tau_rise': 100, 'tau_decay': 5}}

# ext_inp_dict = {'STN': {'mean' : 60, 'sigma': 0.4 * .1, 'tau_rise': 100, 'tau_decay': 5}, #### exponential high FR ext sigma
#                 'D2': {'mean' : 230, 'sigma': 2 * .1, 'tau_rise': 100, 'tau_decay': 4}}

ext_inp_dict = {'STN': {'mean' : 40, 'sigma': 0.4 * .1, 'tau_rise': 100, 'tau_decay': 5}, #### exponential
                'D2': {'mean' : 150, 'sigma': 2 * .1, 'tau_rise': 100, 'tau_decay': 4}}


T[('D2', 'Ctx')]['mean'] = 11

peak_jiggle_dict = {'D2': 15, 
                    'STN': 0.5,
                    'Proto': 0.2}

note = '_tau_PS_1-8'

# T[('Proto', 'D2')] ={'mean': 2, 'sd' : 0.35, 'truncmin': 0.1, 'truncmax': 11.3}     

G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)], G[(name1, name1)] = -0.008, 0.006, -0.0002, -0.000  # tau 25
G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)], G[(name1, name1)] = -0.008, 0.006, -0.0006, -0.000  # tau 4

G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name2, '1'), (name3, '1'), (name1, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): []
                      }

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True


tau_list = np.array([1, 2, 5, 30])
alpha_list = np.linspace(0.1, 1, endpoint = True, num = len(tau_list))
n_run = 5
fig, ax = plt.subplots()

for ( tau_changing, alpha ) in zip ( tau_list, alpha_list ):
    
    tau[('Proto', 'D2')] =  {'rise': {'mean' : [0.1], 'sd' : [0.06], 'truncmin': [0.1], 'truncmax': [100]},
                             'decay' : {'mean' :[tau_changing], 'sd' : [.5], 'truncmin': [0.1], 'truncmax': [100]}}
    
    nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                   poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                   ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
                   syn_component_weight=syn_component_weight, noise_method=noise_method, state = state, external_input_bool = True) for i in pop_list] for name in name_list}
    
    
    n_FR = 20
    all_FR_list = {name: FR_ext_range[name][state]
                   for name in list(nuclei_dict.keys())}
    
    receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                              all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                              set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                              use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)
    
    syn_trans_delay_dict = filter_transmission_delay_for_downstream_projection(T, list_of_nuc_with_trans_inp, 
                                                                               projecting = 'Ctx')
    
    if stim_method == 'Projection':
        syn_trans_delay_dict = syn_trans_delay_heterogeneous( syn_trans_delay_dict, dt, N_sim )
        
    elif stim_method == 'ChR2':
        syn_trans_delay_dict = syn_trans_delay_homogeneous( syn_trans_delay_dict, dt, N_sim )
        
    avg_act = average_multi_run_collective(path, tau, receiving_pop_list, receiving_class_dict, t_list, dt, nuclei_dict,  
                                           Act[state], G, N, N_real, K_real, syn_trans_delay_dict, poisson_prop,
                                            n_FR, all_FR_list, end_of_nonlinearity, t_transient=int(t_transient/dt), 
                                            duration=int(duration/dt), n_run=n_run, A_mvt=None, D_mvt=0, t_mvt=t_mvt, 
                                            ext_inp_dict=ext_inp_dict, noise_amplitude=None, noise_variance=None, 
                                            reset_init_dist=True, color_dict=color_dict, state = state, plot = False, 
                                            ext_inp_method = ext_inp_method_trans, stim_method = stim_method)
    
    
    for nuclei_list in nuclei_dict.values():
        for k, nucleus in enumerate(nuclei_list):
            nucleus.pop_act = avg_act[nucleus.name][:, k]
    
    smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
    status = 'trans_Ctx_to_STN_and_D2_' + str(n_run) + '_run_'
    
    fig = plot(nuclei_dict, color_dict, dt, t_list - np.full_like(t_list, t_transient / dt), 
               Act[state], A_mvt, t_mvt, D_mvt, ax= ax, title_fontsize=20, plot_start=plot_start, figsize=(20,4),
               title="", plt_mvt=False, include_FR=False, xlim = (-50, t_sim - t_transient), lw = 0.5, alpha = alpha)  # , ylim = [0,150])]
    
    fig = plot_extermums_FR(nuclei_dict, peak_jiggle_dict, t_transient, dt, color_dict, fig, alpha = alpha)

# fig = set_x_ticks(fig, np.arange(0, 600, 100))
fig = set_x_ticks(fig, np.arange(0, 100, 10))
fig.gca().get_legend().remove()
# fig = set_y_ticks(fig, np.arange(0, 175, 25))
fig.gca().set_ylim(-10, 160)
fig.gca().set_xlim(-5, 60)
fig.gca().axvspan(0, duration, alpha=0.2, color='yellow')
fig.gca().annotate(r'$\tau_{D2-Proto} \in [1, \; 2, \; 5, \; 30] \; ms$', 
                   (0.45,.7), color = 'k', xycoords='axes fraction', fontsize = 15)
fig.gca().annotate(r'$\tau_{m}^{Proto} = ' + str(nuclei_dict['Proto'][0].neuronal_consts['membrane_time_constant']['mean']) + 
                                               '\; ms$', 
                   (0.45,.5), color = 'k', xycoords='axes fraction', fontsize = 15)
save_pdf_png(fig, os.path.join(path, 'MC_trans_stim_' + str(duration) +'ms_onto_D2-STN-Proto' + note +'_ann_tau_m_' +
                                str(nuclei_dict['Proto'][0].neuronal_consts['membrane_time_constant']['mean']) ), size = (8,3.5))


# %% effect of D2-induced transient input on a STR-GPe-STN network collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_transient = 350
t_sim = t_transient + 150
plot_start = t_transient - 50
plot_start_raster = plot_start
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
name1 = 'Proto'  # projecting
name2 = 'STN'  # recieving
name3 = 'D2'
name4 = 'Arky'
state = 'trans_Nico_mice'
name_list = [name1, name2, name3]#, name4]
g = -0.005
G = {}


( G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)], 
 G[(name4, name1)], G[(name1, name1)] ) = -0.006, 0.002, -0.0007, -0.002, -0.002 ## real tau

( G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)], 
 G[(name4, name1)], G[(name1, name1)] ) = -0.008, 0.002, -0.002, -0.002, -0.002 ##  tau exit 1.8 tau_m Proto 4
# G[(name2, name1)], G[(name1, name2)],  G[(name1, name3)], G[(name4, name1)] = -0.005, 0.0008, -0.0022, -0.002 

G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name2, '1'), (name3, '1'), (name1, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): []}#,
                      # (name4, '1'): [(name1, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True



nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state, external_input_bool = True) for i in pop_list] for name in name_list}


n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


duration = 10
n_run = 5


ext_inp_dict = {'D2': {'mean' : 150., 'sigma': .5 * .1, 'tau_rise': 100, 'tau_decay': 5} } # exponetial tau exit 6 merged
ext_inp_dict = {'D2': {'mean' : 118., 'sigma': .5 * .1, 'tau_rise': 100, 'tau_decay': 5} } # exponetial tau exit 1.8 tau_m Proto 4 merged

# ext_inp_dict = {'D2': {'mean' : 77., 'sigma': .5 * .1, 'tau_rise': 100, 'tau_decay': 5} } # exponetial tau exit 6 only labeled


syn_trans_delay_dict = {'D2' : {'mean': 0, 'sd': 0, 'truncmin': 1, 'truncmax': 10}}
syn_trans_delay_dict = syn_trans_delay_homogeneous( syn_trans_delay_dict, dt, N_sim )

stim_method = 'ChR2'
ext_inp_method_trans = 'step'
ext_inp_method_trans = 'exponential'

avg_act = average_multi_run_collective(path, tau, receiving_pop_list, receiving_class_dict, t_list, dt, nuclei_dict,  
                                       Act[state], G, N, N_real, K_real, syn_trans_delay_dict, poisson_prop,
                                        n_FR, all_FR_list, end_of_nonlinearity, t_transient=int(t_transient/dt), 
                                        duration=int(duration/dt), n_run=n_run, A_mvt=None, D_mvt=0, t_mvt=t_mvt, 
                                        ext_inp_dict=ext_inp_dict, noise_amplitude=None, noise_variance=None, 
                                        reset_init_dist=True, color_dict=color_dict, state = state,
                                        ext_inp_method = ext_inp_method_trans, stim_method = stim_method)


for nuclei_list in nuclei_dict.values():
    for k, nucleus in enumerate(nuclei_list):
        nucleus.pop_act = avg_act[nucleus.name][:, k]

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=1)
fig = plot(nuclei_dict, color_dict, dt, t_list - np.full_like(t_list, t_transient / dt), Act[state], 
           A_mvt, t_mvt, D_mvt, title_fontsize=20, plot_start= plot_start, 
           title="", plt_mvt=False, include_FR=False , ylim = [-5,60], alpha = 0.2)


ax = fig.gca()
# ax.axvspan(t_transient, (t_transient + duration), alpha=0.2, color='yellow')
# ax.axvline(t_transient, *ax.get_ylim(), c= 'grey', ls = '--', lw = 1)
# ax.axvline(t_transient + duration, *ax.get_ylim(), c= 'grey', ls = '--', lw = 1)
# ax.axhline(0, c= 'grey', ls = ':')
# save_pdf_png(fig, os.path.join(path, 'STN_trans_stim_' + str(duration) +'ms_onto_Proto' + str(n_run) + '_run'), size = (5,3))


color_dict['MSNs'] = color_dict['D2']

######## Plot experimental on top
filename = 'D2-10ms_OptoStimData_RecMSN-Proto-Arky-STN_merged.xlsx'
proto_shift = 3
sheet_name_extra = ''
time_shift_dict = {'MSNs': 2.6 ,'STN': 2.6 - 10.1 , 'Arky': 2.6 - 10.1, 'Proto': 2.6 - 10.1+ proto_shift}
# time_shift_dict = {'MSNs': 0 ,'STN': 0, 'Arky': 0, 'Proto': 0}

# filename = 'D2-10ms_OptoStimData_RecMSN-Proto-Arky-STN_OnlyLabelled.xlsx'
# proto_shift = -3
# sheet_name_extra = 'Label'
FR_df = read_sheets_of_xls_data(filepath = os.path.join( root, 'Exp_Stim_data', filename ))

fig, ax, title = plot_fr_response_from_experiment(FR_df, filename, color_dict, xlim = None, ylim = None, stim_duration = 10, 
                           ax = ax, time_shift_dict = time_shift_dict, sheet_name_extra = sheet_name_extra)

ax.set_xlim(-15, 150)
save_pdf_png(fig, os.path.join(path, 'Real_D2_trans_stim_onto_Proto' + str(n_run) + '_run_compared_with_experiment_' + 
                               filename.split('.')[0].split('_')[-1] + '_tau_m_Proto_' + 
                               str(nuclei_dict['Proto'][0].neuronal_consts['membrane_time_constant']['mean'])), size = (5,3))



# %% effect of MC-induced transient input on FSI and D2
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 300
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
state = 'rest'
g = -0.005
g_ext = 0.01
G = {}


plot_start = 150
plot_start_raster = 500
G[(name2, name1)] = g
G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [],
                      (name2, '1'): [(name1, '1')]
                      }

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
der_ext_I_from_curve = False
der_ext_I_from_curve = True

# noise_variance = {name1 : 1,  name2 : 0.1}
noise_amplitude = {name1: 1,  name2: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# rest_init_filepaths = {
#                     'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
#                     'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl'}
# set_init_all_nuclei(nuclei_dict, filepaths = rest_init_filepaths)

n_FR = 20
all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys())}


receiving_class_dict, FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                                             all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                                             set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=True,
                                                             use_saved_FR_ext=False)

pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_FSI-D2.pkl'))


# Run on previously saved data
# FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_FSI-D2.pkl'))
# receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False,
#                                           use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

t_transient = 200  # ms
duration = 5
n_run = 5
list_of_nuc_with_trans_inp = ['STN', 'D2']
inp = 2  # external input in mV to FSI and D2 from Cortex
g_rel_MC = 1.5  # relative gain of MC-FSI to MC-D2
g_rel_MC_series = np.linspace(1, 4, 4)
inp_series = np.linspace(0.5, 5, 4)
n_subplots = int(len(g_rel_MC_series) * len(inp_series))


syn_trans_delay_dict_STN = {k[0]: v for k,
                            v in T.items() if k[0] == 'FSI' and k[1] == 'Ctx'}
syn_trans_delay_dict_STR = {k[0]: v for k,
                            v in T.items() if k[0] == 'D2' and k[1] == 'Ctx'}
syn_trans_delay_dict = {**syn_trans_delay_dict_STN, **syn_trans_delay_dict_STR}
syn_trans_delay_dict = {k: v / dt for k, v in syn_trans_delay_dict.items()}


count = 0
fig = plt.figure()
# fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(6, 6))

for inp in inp_series:

    for g_rel_MC in g_rel_MC_series:
        print(count, "from ", n_subplots)
        ext_inp_dict = {'FSI': {'mean': g_rel_MC * inp, 'sigma': 5},
                        'D2': {'mean': inp, 'sigma': 5}
                        }
        nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                              t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise=False)

        avg_act = average_multi_run(receiving_class_dict, t_list, dt, nuclei_dict, rest_init_filepaths, Act['rest'],
                                    Act['trans'], syn_trans_delay_dict, t_transient=int(
                                        t_transient / dt),
                                    duration=int(duration / dt), n_run=n_run, inp_method='add', ext_inp_dict=ext_inp_dict)

        for nuclei_list in nuclei_dict.values():
            for k, nucleus in enumerate(nuclei_list):
                nucleus.pop_act = avg_act[nucleus.name][:, k]
        state = 'FSI_D2_trans_Ctx_'+str(n_run) + '_runs_delay_included'
        ax = fig.add_subplot(len(inp_series), len(g_rel_MC_series), count+1)
        plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title_fontsize=10, plot_start=plot_start, n_subplots=n_subplots,
             title=r'$\frac{{G_{{MC-FSI}}}}{{G_{{MC-D2}}}} = {0} \; I_{{MC}}={1}$'.format(
                 g_rel_MC, inp),
             plt_mvt=False, include_FR=False, tick_label_fontsize=10)  # , ylim = [0,150])
        plt.axvspan(t_transient, (t_transient + duration),
                    alpha=0.2, color='yellow')
        plt.xlabel("")
        plt.ylabel("")
        count += 1
        if count < (len(inp_series) - 1) * len(g_rel_MC_series) - 1:
            ax.axes.xaxis.set_ticklabels([])


fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False,
                bottom=False, left=False, right=False)
plt.xlabel("time (ms)")
plt.ylabel("Firing rate (Hz)")
fig.set_size_inches((15, 7), forward=False)
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 500, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)


# %% Transition to activated state FSI-D2-GPe + STN-GPe single neuron

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
t_transition = 1000
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto'  # projecting
name2 = 'STN'  # recieving
name3 = 'D2'
name4 = 'FSI'
g = -0.0012
g_ext = 0.01
G = {}
plot_start = 500
plot_start_raster = 600

G[(name2, name1)], G[(name1, name2)],  = g, -g
G[(name3, name4)], G[(name4, name1)], G[(name1, name3)] = g, g, g
# Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner
tau[('STN', 'Proto')] = {'rise': [1.1, 40], 'decay': [7.8,  200]}
# tau[('STN','Proto')] =  {'rise':[1.1],'decay':[7.8]} # Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name3: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name4: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [(name2, '1'), (name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name4, '1')],
                      (name4, '1'): [(name1, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False

set_input_from_response_curve = True
save_init = False
noise_variance = {name1: 0.1,  name2: 15,  name3: 1,  name4: 1}
noise_amplitude = {name1: 1,  name2: 1,  name3: 1,  name4: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc4 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name4, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]


nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3, name4: nuc4}


filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl',
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl',
             # 'Proto': 'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl',
             'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl'
             }

mvt_init_filepaths = {'Proto': 'tau_m_12-94_Proto_A_22_N_1000_T_2000_noise_var_30.pkl',
                      # 'FSI': 'FSI_A_70_N_1000_T_2000_noise_var_10.pkl',
                      'FSI': 'tau_m_9-2_FSI_A_32_N_1000_T_2000_noise_var_8.pkl',
                      'D2': 'tau_m_13_D2_A_4_N_1000_T_2000_noise_var_3.pkl',
                      'STN': "tau_m_5-13_STN_A_50_N_1000_T_2000_noise_var_3.pkl"
                      }


receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)
set_init_all_nuclei(nuclei_dict, filepaths=filepaths)  # filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method='draw_from_data', set_noise=False)


nuclei_dict = run_transition_to_movement(receiving_class_dict, t_list, dt, nuclei_dict, mvt_init_filepaths, N, N_real,
                                         A_mvt, D_mvt, t_mvt, t_transition=int(t_transition/dt))

# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
state = 'transition_to_mvt_transient'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_transition, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title=init_method,
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color='lightskyblue', ylim=(-10, 80))
fig.set_size_inches((15, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=t_sim,
                              labelsize=20, title_fontsize=25, lw=1.5, linelengths=2, n_neuron=40, include_title=True, set_xlim=True,
                              axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color='lightskyblue')
fig.set_size_inches((15, 5), forward=False)
fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center', fontsize=18)
fig_.text(0.03, 0.5, 'neuron', ha='center',
          va='center', rotation='vertical', fontsize=18)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
             orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
             orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig, ax = plt.subplots(1, 1)
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
find_freq_all_nuclei(dt, nuclei_dict, duration_2, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)
x_l = 0.75
ax.axhline(x_l, ls='--', c='grey')
ax.set_xlim(0, 55)
ax.axvspan(0, 55, alpha=0.2, color='lightskyblue')
fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig, ax = plt.subplots(1, 1)
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)
ax.axhline(x_l, ls='--', c='grey')
ax.set_xlim(0, 55)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)

# %% Transition to DD FSI-D2-GPe + STN-GPe single neuron

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

plot_start = int(t_sim / 5)
t_transition = plot_start + int(t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim / dt) - length, int(t_sim / dt)]

name1 = 'Proto'  # projecting
name2 = 'STN'  # recieving
name3 = 'D2'
name4 = 'FSI'
g = -0.0009
g_ext = 0.01
G = {}
G[(name2, name1)], G[(name1, name2)],  = g, -g
G[(name3, name4)], G[(name4, name1)], G[(name1, name3)] = g, g, g

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name3: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name4: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [(name2, '1'), (name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name4, '1')],
                      (name4, '1'): [(name1, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False

set_input_from_response_curve = True
save_init = False


nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc4 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name4, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]


nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3, name4: nuc4}


filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl',
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl',
             # 'Proto': 'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl',
             'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl'
             }

DD_init_filepaths = {'Proto': 'tau_m_12-94_Proto_A_22_N_1000_T_2000_noise_var_30.pkl',
                     # 'FSI': 'FSI_A_70_N_1000_T_2000_noise_var_10.pkl',
                     'FSI': 'tau_m_9-2_FSI_A_24_N_1000_T_2000_noise_var_15.pkl',
                     'D2': 'tau_m_13_D2_A_6-6_N_1000_T_2000_noise_var_3.pkl',
                     'STN': "tau_m_5-13_STN_A_24_N_1000_T_2000_noise_var_6.pkl"
                     }


receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)
set_init_all_nuclei(nuclei_dict, filepaths=filepaths)  # filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method='draw_from_data', set_noise=False)


nuclei_dict = run_transition_to_DA_depletion(receiving_class_dict, t_list, dt, nuclei_dict, DD_init_filepaths,
                                             K_real_DD, N, N_real, A_DD, A_mvt, D_mvt, t_mvt, t_transition=int(t_transition/dt))

# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
state = 'transition_to_DD_with_GABA-b'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_DD, t_transition, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title=init_method,
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color='darkseagreen')
fig.set_size_inches((15, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=t_sim,
                              labelsize=20, title_fontsize=25, lw=1.5, linelengths=2, n_neuron=40, include_title=True, set_xlim=True,
                              axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color='darkseagreen')
fig.set_size_inches((15, 5), forward=False)
fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center', fontsize=18)
fig_.text(0.03, 0.5, 'neuron', ha='center',
          va='center', rotation='vertical', fontsize=18)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
             orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
             orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig, ax = plt.subplots(1, 1)
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)
x_l = 5
ax.axhline(x_l, ls='--', c='grey')
ax.set_xlim(5, 55)
ax.axvspan(5, 55, alpha=0.2, color='darkseagreen')
fig.savefig(os.path.join(path, 'SNN_spectrum_DD_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_DD_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig, ax = plt.subplots(1, 1)
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)
x_l = 5
ax.axhline(x_l, ls='--', c='grey')
ax.set_xlim(5, 55)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)

# fig,ax = plt.subplots(1,1)
# f, t, Sxx = spectrogram(nuc2[0].pop_act[int(plot_start/dt):], 1/(dt/1000))
# img = ax.pcolormesh(t, f, 10*np.log(Sxx), cmap = plt.get_cmap('YlOrBr'),shading='gouraud', vmin=-30, vmax=0)
# ax.axvline( (t_transition - plot_start)/1000, ls = '--', c = 'grey')
# ax.set_ylabel('Frequency (Hz)',  fontsize = 15)
# ax.set_xlabel('Time (sec)', fontsize = 15)
# ax.set_ylim(0,70)
# fig.set_size_inches((15, 5), forward=False)
# clb = fig.colorbar(img)
# clb.set_label(r'$10Log(PSD)$', labelpad=15, y=0.5, rotation=-90, fontsize = 15)
# fig.savefig(os.path.join(path, 'SNN_temporal_spectrum_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_temporal_spectrum_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# %% Transition to activated state FSI-D2-GPe single neuron
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
duration_base = [0, int(t_mvt/dt)]
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
g = -0.5
g_ext = 0.01
G = {}
plot_start = int(t_sim / 5)
t_transition = plot_start + int(t_sim / 4)
# plot_start = 300
# t_transition = 800

duration_base = [int(100/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_2 = [int(t_sim / dt) - length, int(t_sim / dt)]

# G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] =  0,0,0

G[(name2, name1)], G[(name3, name2)], G[(name1, name3)] = - \
    1.8 * 10**-4, -3.5 * 10**-4, -12 * 10**-4  # FSI 70
G[(name2, name1)], G[(name3, name2)], G[(name1, name3)] = -1.1 * \
    10**-4, -3.2 * 10**-4, -3.2 * 10**-4  # close to oscillatory regime

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name3: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
set_input_from_response_curve = True
save_init = False
noise_variance = {name1: 0.1, name2: 0.1, name3: 15}
noise_amplitude = {name1: 1, name2: 1, name3: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}

filepaths = {name1: 'FSI_A_18-5_N_1000_T_2000_noise_var_1.pkl',
             name2: 'D2_A_1-1_N_1000_T_2000_noise_var_0-1.pkl',
             name3: name3 + '_N_'+str(N_sim) + '_T_2000_noise_var_15.pkl'}

mvt_init_filepaths = {'Proto': 'Proto_A_22_N_1000_T_2000_noise_var_15.pkl',
                      # 'FSI': 'FSI_A_70_N_1000_T_2000_noise_var_10.pkl',
                      'FSI': 'FSI_A_32_N_1000_T_2000_noise_var_1.pkl',
                      'D2': 'D2_A_4_N_1000_T_2000_noise_var_0-1.pkl'}


receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)
set_init_all_nuclei(nuclei_dict, filepaths=filepaths)  # filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method='draw_from_data', set_noise=False)


nuclei_dict = run_transition_to_movement(receiving_class_dict, t_list, dt, nuclei_dict,
                                         mvt_init_filepaths, N, N_real, A_mvt, D_mvt, t_mvt, t_transition=int(t_transition/dt))

# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
state = 'transition_to_mvt_transient'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_transition, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title=init_method,
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color='lightskyblue', ylim=(-10, 80))
# fig.set_size_inches((15, 5), forward=False)
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start, plot_end = t_sim,
#                             labelsize = 20, title_fontsize = 25, lw  = 1.5, linelengths = 2, n_neuron = 40, include_title = True, set_xlim=True,
#                             axvspan = True, span_start = t_transition, span_end = t_sim, axvspan_color = 'lightskyblue')
# fig.set_size_inches((15, 5), forward=False)
# fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center',fontsize= 18)
# fig_.text(0.03, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 18)
# fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig, ax = plt.subplots(1,1)
# peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
# find_freq_all_nuclei(dt, nuclei_dict, duration_2, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False ,
#                 low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False,
#                 fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.set_xlim(0,55)
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')
# fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig, ax = plt.subplots(1,1)
# peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
# find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False ,
#                 low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False,
#                 fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.set_xlim(0,55)
# fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)


# fig,ax = plt.subplots(1,1)
# f, t, Sxx = spectrogram(nuc2[0].pop_act[int(plot_start/dt):], 1/(dt/1000))
# img = ax.pcolormesh(t, f, 10*np.log(Sxx), cmap = plt.get_cmap('YlOrBr'),shading='gouraud', vmin=-30, vmax=0)
# ax.axvline( (t_transition - plot_start)/1000, ls = '--', c = 'grey')
# ax.set_ylabel('Frequency (Hz)',  fontsize = 15)
# ax.set_xlabel('Time (sec)', fontsize = 15)
# ax.set_ylim(0,70)
# fig.set_size_inches((10, 5), forward=False)
# clb = fig.colorbar(img)
# clb.set_label(r'$10Log(PSD)$', labelpad=15, y=0.5, rotation=-90, fontsize = 15)
# fig.savefig(os.path.join(path, 'SNN_temporal_spectrum_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_temporal_spectrum_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# %% Transition to DD FSI-D2-GPe single neuron
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
# A['D2'] = 3
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [0, int(t_mvt/dt)]
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
g = -0.5
g_ext = 0.01
G = {}
plot_start = int(t_sim / 5)
t_transition = plot_start + int(t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim / dt) - length, int(t_sim / dt)]

# G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] =  0,0,0

G[(name2, name1)], G[(name3, name2)], G[(name1, name3)] = -1.2 * \
    10**-4, -3.8 * 10**-4, -1. * 10**-4  # close to oscillatory regime

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name3: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
set_input_from_response_curve = True
save_init = False
noise_variance = {name1: 0.1, name2: 0.1, name3: 15}
noise_amplitude = {name1: 1, name2: 1, name3: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
                poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, mem_pot_init_method=mem_pot_init_method,
                ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list]
nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}

filepaths = {name1: name1 + '_N_'+str(N_sim) + '_T_2000.pkl',
             # name2:name2 + '_N_'+str(N_sim) +'_T_2000.pkl', ## A = 3
             name2: 'D2_A_1-1_N_1000_T_2000_noise_var_0-1.pkl',
             name3: name3 + '_N_'+str(N_sim) + '_T_2000_noise_var_15.pkl'}
DD_init_filepaths = {'Proto': 'Proto_A_38_N_1000_T_2000_noise_var_10.pkl',
                     'FSI': 'FSI_A_24_N_1000_T_2000_noise_var_1.pkl',
                     'D2': 'D2_A_6-6_N_1000_T_2000_noise_var_0-1.pkl'}


receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)
set_init_all_nuclei(nuclei_dict, filepaths=filepaths)  # filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method='draw_from_data', set_noise=False)


nuclei_dict = run_transition_to_DA_depletion(receiving_class_dict, t_list, dt, nuclei_dict, DD_init_filepaths,
                                             K_real_DD, N, N_real, A_DD, A_mvt, D_mvt, t_mvt, t_transition=int(t_transition/dt))

# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
state = 'transition_to_DD'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_DD, t_transition, D_mvt, ax=None, title_fontsize=15, plot_start=plot_start, title=init_method,
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color='darkseagreen')
fig.set_size_inches((15, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=t_sim,
                              labelsize=20, title_fontsize=25, lw=1.5, linelengths=2, n_neuron=40, include_title=True, set_xlim=True,
                              axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color='darkseagreen')
fig.set_size_inches((15, 5), forward=False)
fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center', fontsize=18)
fig_.text(0.03, 0.5, 'neuron', ha='center',
          va='center', rotation='vertical', fontsize=18)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
             orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
             orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig, ax = plt.subplots(1, 1)
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)
x_l = 8
ax.axhline(x_l, ls='--', c='grey')
ax.set_xlim(5, 55)
ax.axvspan(5, 55, alpha=0.2, color='darkseagreen')
fig.savefig(os.path.join(path, 'SNN_spectrum_DD_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_DD_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig, ax = plt.subplots(1, 1)
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)
x_l = 8
ax.axhline(x_l, ls='--', c='grey')
ax.set_xlim(5, 55)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)

fig, ax = plt.subplots(1, 1)
f, t, Sxx = spectrogram(nuc2[0].pop_act[int(plot_start/dt):], 1/(dt/1000))
img = ax.pcolormesh(t, f, 10*np.log(Sxx), cmap=plt.get_cmap('YlOrBr'),
                    shading='gouraud', vmin=-30, vmax=0)
ax.axvline((t_transition - plot_start)/1000, ls='--', c='grey')
ax.set_ylabel('Frequency (Hz)',  fontsize=15)
ax.set_xlabel('Time (sec)', fontsize=15)
ax.set_ylim(0, 70)
fig.set_size_inches((15, 5), forward=False)
clb = fig.colorbar(img)
clb.set_label(r'$10Log(PSD)$', labelpad=15, y=0.5, rotation=-90, fontsize=15)
fig.savefig(os.path.join(path, 'SNN_temporal_spectrum_'+state+'.png'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_temporal_spectrum_'+state+'.pdf'), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)

# %% Transition FSI-D2-GPe collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 1600

t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 5)
t_transition = 1000
duration_base = [int(100/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim / dt) - length +
               int(t_transition / 4 / dt), int(t_sim / dt)]


plot_start = t_transition - 500
plot_end = t_transition + 600

name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'

state_1 = 'awake_rest'
state_2 = 'mvt'


name_list = [name1, name2, name3]

G = {}

g = -0.0045 # rest to DD anesthetized
g = -0.0032 # awake to mvt


(G[(name2, name1)], 
 G[(name3, name2)],
 G[(name1, name3)])= g, g, g



G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}

receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}
                      # (name3, '1'): [(name2,'1'), (name5,'1')]}
                      # with GP-GP


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 20

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state_1) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict,nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)

nuclei_dict = run_transition_state_collective_setting(G, noise_variance, noise_amplitude, path, receiving_class_dict, receiving_pop_list, 
                                                      t_list, dt, nuclei_dict, Act, state_1, state_2, K_all, N, N_real,
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity, t_transition = int(t_transition/dt))


nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'transition_to_' + state_2

D_mvt = t_sim - t_transition

if 'DD' in state_2:
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, title='', legend_loc='upper left', plot_end=t_transition-10, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8)
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_' + state_1),
                 size=(8, 5))
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start = t_sim - (t_transition-plot_start), title='',
               legend_loc='upper left', plot_end= t_sim, vspan=True, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color = axvspan_color[state_2])
    
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_plot_' + state_2),
                 size=(8, 5))
    fig_state_1, fig_state_2 = raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start,
                                             labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=False, set_xlim=True,
                                             axvspan_color=axvspan_color[state_2], n=N_sim,  ylabel_x=0.01,
                                             t_transition=t_transition, t_sim=t_sim, tick_label_fontsize=12, include_nuc_name=False)

    save_pdf_png(fig_state_1, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_1), size=(3, 5))
    save_pdf_png(fig_state_2, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_2), size=(3, 5))
    
elif 'mvt' in state_2:
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, plot_end = plot_end, title='', legend_loc='upper left',  ylim=(-4, 80), vspan=True,
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color=axvspan_color[state_2])
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_from_' + state_1),
                 size=(10, 5))
    
    fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=plot_end, ax_label = True,
                                  labelsize=10, title_fontsize=15, lw=1., linelengths=1.5, n_neuron=40, include_title=True, set_xlim=True, ylabel_x = 0.01,
                                  axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color=axvspan_color[state_2], include_nuc_name = False,
                                   tick_label_fontsize=12)
    save_pdf_png(fig_raster, os.path.join(
        path, 'SNN_raster_' + status ), size=(5, 5))


peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)

ax.set_xlim(5, 55)
ax.axvspan(5, 55, alpha=0.2, color=axvspan_color[state_2])
ax.set_ylim(-0.01, 0.1)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(axis='both', labelsize=15)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_1),
             size=(5, 3))

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False, include_peak_f_in_legend = False)

ax.set_xlim(5, 55)
ax.set_ylim(-0.01, 0.1)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=10, frameon=False)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_2),
             size=(5, 3))


# %% Transition to MVT FSI-D2-GPe + Arky-D2-GPe + STN-GPe + GPe-GPe collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.2
t_sim = 4000
t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 5)
t_transition = plot_start + int(t_sim / 4)
duration_base = [int(500/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim / dt) - length +
               int(t_transition / 4 / dt), int(t_sim / dt)]


plot_start = t_transition - 500
plot_end = t_transition + 1000



name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'


state_1 = 'awake_rest'
state_2 = 'mvt'

print( " transition from " , state_1, ' to ', state_2)
name_list = [name1, name2, name3, name4, name5]

G = {}

g = -0.002

(G[(name2, name1)], G[(name3, name2)],
 G[(name1, name3)], G[(name2, name4)],
 G[(name4, name3)], G[(name3, name5)],
 G[(name5, name3)], G[(name3, name3)]) = 2* g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 3.5 * -g , 3 * g , g * 0.1

# [2, 2.5, 1, 1, 1, 3, 1, 0.1] --> 15 Hz g = 0.002
# 2* g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 3 * -g , 2 * g , g * 0.1 --> 16.4 Hz g = 0.002
# 2* g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 3.5 * -g , 3 * g , g * 0.1 --> 17 Hz g = 0.002

# 2* g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 4 * -g , 3 * g , g * 0.1 --> 17 Hz g = 0.0019
# 2* g, 2.5 * g, 1.5 * g, 2. * g, 1.5 * g, 4 * -g , 3 * g , g * 0.1 --> 18.9 Hz g = 0.002 but small peak at rest at ~40 Hz

G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1'), (name4, '1')],
                      # (name3, '1'): [(name2,'1'), (name5,'1')],
                      # with GP-GP
                      (name3, '1'): [(name2, '1'), (name3, '1'), (name5, '1')],
                      (name4, '1'): [(name3, '1')],
                      (name5, '1'): [(name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 20

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state_1) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)

nuclei_dict = run_transition_state_collective_setting(G, noise_variance, noise_amplitude, path, receiving_class_dict, receiving_pop_list, 
                                                      t_list, dt, nuclei_dict, Act, state_1, state_2, K_all, N, N_real,
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity, t_transition = int(t_transition/dt))


nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'transition_to_' + state_2

D_mvt = t_sim - t_transition
if 'DD' in state_2:
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, title='', legend_loc='upper left', plot_end=t_transition-10, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8)
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_' + state_1),
                 size=(8, 5))
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start = t_sim - (t_transition-plot_start), title='',
               legend_loc='upper left', plot_end= t_sim, vspan=True, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color = axvspan_color[state_2])
    
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_plot_' + state_2),
                 size=(8, 5))
    fig_state_1, fig_state_2 = raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start,
                                             labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=False, set_xlim=True,
                                             axvspan_color=axvspan_color[state_2], n=N_sim,  ylabel_x=0.01,
                                             t_transition=t_transition, t_sim=t_sim, tick_label_fontsize=12, include_nuc_name=False)

    save_pdf_png(fig_state_1, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_1), size=(3, 5))
    save_pdf_png(fig_state_2, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_2), size=(3, 5))
    
elif 'mvt' in state_2:
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, plot_end = plot_end, title='', legend_loc='upper left',  ylim=(-4, 80), vspan=True,
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color=axvspan_color[state_2])
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_from_' + state_1),
                 size=(15, 5))
    
    fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=plot_end, ax_label = True,
                                  labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=True, set_xlim=True, ylabel_x = 0.01,
                                  axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color=axvspan_color[state_2], include_nuc_name = False,
                                   tick_label_fontsize=12)
    save_pdf_png(fig_raster, os.path.join(
        path, 'SNN_raster_' + status ), size=(5, 5))


peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows= n_windows, include_beta_band_in_legend=False)

ax.set_xlim(5, 55)
ax.axvspan(5, 55, alpha=0.2, color=axvspan_color[state_2])
ax.set_ylim(-0.01, 0.1)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(axis='both', labelsize=15)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_1),
             size=(5, 3))

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows= n_windows, include_beta_band_in_legend=False, include_peak_f_in_legend = False)

ax.set_xlim(5, 55)
ax.set_ylim(-0.01, 0.1)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=10, frameon=False)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_2),
             size=(5, 3))


# %% Transition to DD FSI-D2-GPe + Arky-D2-GPe + STN-GPe + GPe-GPe collective

mod_dict = {'DD' : {'STN': [21, 28], 'Proto' : [17, 20], 'Arky' : [9, 16]}}
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
K_small = calculate_number_of_connections(dict.fromkeys(N, 1000), N_real, K_real)
K_ratio = {key :v/K[key] for key, v in K_small.items()}
dt = 0.1
t_sim = 7600 
t_list = np.arange(int(t_sim/dt))
plot_start = 0
t_transition = plot_start + 1300# int(t_sim / 5)3
duration_base = np.array( [int(300/dt), int(t_transition/dt)] )
duration_DD = np.array( [int(t_transition / dt) + int(300/dt) , int(t_sim / dt)] ) 
t_mvt = t_sim
D_mvt = t_sim - t_mvt
n_windows_DD = int((t_sim - 600) / 1000)
n_windows_base = (t_transition - 300) / 1000


name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'

state_1 = 'rest'
state_2 = 'DD_anesth'


print( " transition from " , state_1, ' to ', state_2)

name_list = [name1, name2, name3, name4, name5]

G = {}


### Uniform Gs
# g = -0.0035
# (G[(name2, name1)], G[(name3, name2)],
#   G[(name1, name3)], G[(name2, name4)],
#   G[(name4, name3)], G[(name3, name5)],
#   G[(name5, name3)], G[(name3, name3)]) = 3* g, 4 * g, 2.5 * g, 2.5 * g, 2.5 * g, 2. * -g , 3. * g , g * 0.1
# G = {k: v * K[k] for k, v in G.items()}



g = -0.0025 ## log-normal syn weight dist F = 18.5 Hz
G = { (name2, name1) :{'mean': g * K[name2, name1] * 11},#}, ## free
      (name3, name2) :{'mean': g * K[name3, name2] * 11},#11.}, ## free
      (name1, name3) :{'mean': g * K[name1, name3] * 11},#30 * 66/63}, ## free
      (name2, name4) :{'mean': g * K[name2, name4] * 4},#0.01}, ## free
      (name4, name3) :{'mean': g * K[name4, name3] * 3},
      (name3, name5) :{'mean': -g * K[name3, name5] * 2.4},
      (name5, name3) :{'mean': g * K[name5, name3] * 4.7},
      (name3, name3) :{'mean': g * K[name3, name3] * 1.25}}#2.}}#, 
      # (name1, name5) :{'mean': g * K[name1, name5] * 1}}
      
# g = -0.0025 ## log-normal syn weight dist F = 17.5 Hz
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 13.5},#6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 13.5},#11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 13.5},#30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 5},#0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.4},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4.7},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.9}}#2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}


# g = -0.0025 ## log-normal syn weight dist F = 17.3 Hz SD = 10**2
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 13.5},#6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 13.5},#11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 13.5},#30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 5},#0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 3.2},
#       (name5, name3) :{'mean': g * K[name5, name3] * 6.3},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.2}}#2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}

g = -0.0025 ## log-normal syn weight dist F = 17.3 Hz SD = 10**2
G = { (name2, name1) :{'mean': g * K[name2, name1]  * 0},#6}, ## free
      (name3, name2) :{'mean': g * K[name3, name2] * 0},#11.}, ## free
      (name1, name3) :{'mean': g * K[name1, name3] * 0},#30 * 66/63}, ## free
      (name2, name4) :{'mean': g * K[name2, name4] * 0},#0.01}, ## free
      (name4, name3) :{'mean': g * K[name4, name3] * 0},
      (name3, name5) :{'mean': -g * K[name3, name5] * 2.4},
      (name5, name3) :{'mean': g * K[name5, name3] * 4.7},
      (name3, name3) :{'mean': g * K[name3, name3] * 0}}#2.}}#, 
      # (name1, name5) :{'mean': g * K[name1, name5] * 1}}
            
     
# g = -0.0025 ## log-normal syn weight dist F = 17.3 Hz scaled for large network
# G = { (name2, name1) :{'mean': g * K[name2, name1]* K_ratio[name2, name1] * 13.5}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2]* K_ratio[name3, name2] * 13.5}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3]* K_ratio[name1, name3] * 13.5}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4]* K_ratio[name2, name4] * 5}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3]* K_ratio[name4, name3] * 3},
#       (name3, name5) :{'mean': -g * K[name3, name5]* K_ratio[name3, name5]* 3.2},
#       (name5, name3) :{'mean': g * K[name5, name3]* K_ratio[name5, name3] * 6.3},
#       (name3, name3) :{'mean': g * K[name3, name3]* K_ratio[name3, name3] * 1.2}}



# g = -0.0025 ## K_connections tuned July 2022 N = 1000 f = 17 Hz, returned with brice
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.5},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.45 * 62/60},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4.3 * 205/180},
#       (name3, name3) :{'mean': g * K[name3, name3] * 2.2}}
#         # (name1, name5) :{'mean': g * K[name1, name5] * 3}}


# g = -0.0025 ## FENS poster
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 10},#6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 10},#11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 10},#30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 10},#0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.45 * 62/60},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4.3 * 205/180},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.6}}#2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}



# g = -0.0025 ## K_connections tuned July 2022 N = 1000  f = 18 Hz
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 8}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 0}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.2*62/60},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4*205/180},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.5}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}

# g = -0.01 ### for N = 3500
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 9.63}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 1.53}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 6.3}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 5}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.62},
#       (name3, name5) :{'mean': -g * K[name3, name5]* 2.65},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.5},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.2}
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}
#       }

# g = -0.01
# G = { (name2, name1) :{'mean': g * K[name2, name1]* K_ratio[name2, name1] * 9.63}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2]* K_ratio[name3, name2] * 1.53}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3]* K_ratio[name1, name3] * 6.3}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4]* K_ratio[name2, name4] * 5}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3]* K_ratio[name4, name3] * 2.62},
#       (name3, name5) :{'mean': -g * K[name3, name5]* K_ratio[name3, name5]* 2.65},
#       (name5, name3) :{'mean': g * K[name5, name3]* K_ratio[name5, name3] * 3.5},
#       (name3, name3) :{'mean': g * K[name3, name3]* K_ratio[name3, name3] * 1.2},
#       (name1, name5) :{'mean': g * K[name1, name5] * K_ratio[name1, name5] * 4}}


# g = -0.0025 
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.7}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 3.8}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 3.6}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 3.5}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.62},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.65},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.2},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.005}
#       }
# g = -0.0025 
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.7}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 3.8}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 3.6}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 3.5}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 1.3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.2},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.005}
#       }

# D2-Proto tuned
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 5}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 0.45},
#       (name1, name3) :{'mean': g * K[name1, name3] * 5}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 3}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 1.2},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 0.75},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.2},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.1}
#       }


# g = -0.01 ## no tuning to experiments 
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.1},
#       (name3, name2) :{'mean': g * K[name3, name2] * 4},
#       (name1, name3) :{'mean': g * K[name1, name3] * 2.5},
#       (name2, name4) :{'mean': g * K[name2, name4] * 2.5},
#       (name4, name3) :{'mean': g * K[name4, name3] * 1.3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 1.7},
#       (name5, name3) :{'mean': g * K[name5, name3] * 2},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.1}
#       }


G = set_G_dist_specs(G,  order_mag_sigma = 2)




poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 
                 'tau': { 'rise': {'mean': 1, 'var': .5}, 
                         'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'): [(name3, '1')],#, (name5, '1')],
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
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 12; high_f = 30
spike_history = 'long-term'
set_random_seed = False

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state_1,
               spike_history = spike_history, Act = Act) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                                       all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                       set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                                       use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)


nuclei_dict = run_transition_state_collective_setting(G, noise_variance, noise_amplitude, path, receiving_class_dict, receiving_pop_list, 
                                                      t_list, dt, nuclei_dict, Act, state_1, state_2, K_all, N, N_real,
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity, t_transition = int(t_transition/dt))


nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'transition_to_' + state_2 + '_tuned'

ylim = (-2, 60)

D_mvt = t_sim - t_transition
plot_end_rest = t_transition - 10
plot_start_rest =  t_transition - 700
plot_start_DD = t_sim - 700
plot_end_DD = t_sim

if 'DD' in state_2:
    
    fig = plot(nuclei_dict, color_dict, dt, (t_list - np.full_like(t_list, plot_start_rest / dt)), 
               Act[state_1], Act[state_2], 
               t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start = plot_start_rest, title='', legend_loc='upper left', plot_end= plot_end_rest, ylim=ylim,
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, ncol_legend = 1,
               xlim = (0, (plot_end_rest - plot_start_rest )), 
               legend_fontsize = 18, label_fontsize = 25, legend = False)

    set_x_ticks_one_ax(fig.gca(), [0, 200, 400])
    set_x_ticks_one_ax(fig.gca(), [0, 350, 700])
    set_y_ticks_one_ax(fig.gca(), [0, 30, 60])
    set_minor_locator(fig.gca(), n = 2, axis = 'both')
    
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_' + state_1),
                 size=(8, 5))
    
    fig = plot(nuclei_dict, color_dict, dt, t_list- np.full_like(t_list, plot_start_DD / dt), 
               Act[state_1], Act[state_2], t_transition - plot_start_DD, D_mvt, ax=None, title_fontsize=15, 
               plot_start = plot_start_DD, title='',  legend_fontsize = 15, label_fontsize = 25,
               legend_loc='upper left', plot_end= plot_end_DD, vspan=True, ylim=ylim,
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, 
               axvspan_color = axvspan_color[state_2], ncol_legend = 1, legend = False,
               xlim = (0, (plot_end_DD - plot_start_DD )))
    # fig_filtered = plot(nuclei_dict, color_dict, dt, t_list, 
    #            Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
    #             title='',  legend_fontsize = 15, label_fontsize = 25,
    #            legend_loc='upper left', vspan=True, ylim=ylim,
    #            include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, 
    #            axvspan_color = axvspan_color[state_2], ncol_legend = 1, legend = False,
    #            plot_filtered=True, low_f = low_f, high_f = high_f, threshold_peak_by_max = 0.5)
    set_y_ticks_one_ax(fig.gca(), [0, 30, 60])
    
    set_x_ticks_one_ax(fig.gca(), [0, 350, 700])
    set_minor_locator(fig.gca(), n = 2, axis = 'both')
    
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_plot_' + state_2),
                  size=(8, 5))
    
    fig_state_1, fig_state_2 = raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',
                                                                 labelsize=25, title_fontsize=15, lw=1., linelengths=2, 
                                                                 n_neuron=40, include_title=False, set_xlim= True,
                                                                 axvspan_color=axvspan_color[state_2], n=N_sim,  ylabel_x=0.005,
                                                                 t_transition=t_transition, t_sim=t_sim, tick_label_fontsize=20, 
                                                                 include_nuc_name=False,
                                                                 plot_start_state_1=t_transition - 710, plot_end_state_1= plot_end_rest, 
                                                                 plot_start_state_2=plot_start_DD, plot_end_state_2=t_sim)

    set_x_ticks(fig_state_1, [0, int( (plot_end_DD - plot_start_DD)/2 ), plot_end_DD - plot_start_DD])
    set_x_ticks(fig_state_2, [0,int( (plot_end_DD - plot_start_DD)/2 ), plot_end_DD - plot_start_DD])
    rm_ax_unnecessary_labels_in_fig(fig_state_1)
    rm_ax_unnecessary_labels_in_fig(fig_state_2)
    save_pdf_png(fig_state_1, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_1), size=(5., 6))
    save_pdf_png(fig_state_2, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_2), size=(5., 6))
    
elif 'mvt' in state_2:
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, plot_end = plot_end, title='', legend_loc='upper left',  ylim=(-4, 80), vspan=True,
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color=axvspan_color[state_2])
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_from_' + state_1),
                 size=(15, 5))
    
    fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=plot_end, ax_label = True,
                                  labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=True, set_xlim=True, ylabel_x = 0.01,
                                  axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color=axvspan_color[state_2], include_nuc_name = False,
                                   tick_label_fontsize=12)
    save_pdf_png(fig_raster, os.path.join(
        path, 'SNN_raster_' + status ), size=(5, 5))
    

peak_threshold = 0.1
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                          low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                          fft_method='Welch', n_windows=n_windows_DD, include_beta_band_in_legend=False)

ax.set_xlim(5, 70)
ax.axvspan(5, 70, alpha=0.2, color=axvspan_color[state_2])
# ax.set_ylim(-0.01, 40)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(axis='both', labelsize=15)
ylim = ax.get_ylim()
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_2),
              size=(5, 3))

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                          low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                          fft_method='Welch', n_windows=n_windows_base, include_beta_band_in_legend=False, include_peak_f_in_legend = False)

ax.set_xlim(5, 70)
ax.set_ylim(ylim)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=10, frameon=False)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_1),
              size=(5, 3))

phase_ref = 'Proto'
nuclei_dict = find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins=36,
                                                peak_threshold=None, phase_ref= phase_ref, start=duration_DD[0], total_phase=360,
                                                only_PSD_entrained_neurons= False, troughs = False,
                                                only_rtest_entrained = True, threshold_by_percentile = 50, 
                                                plot = False, shift_phase_deg = 45)
y_max_series = {'D2': 10, 'STN': 35, 'Arky': 21, 'Proto': 26, 'FSI': 10} # with single neuron traces

fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, coef = 1, scale_count_to_FR = True,
                                    density=False, phase_ref= phase_ref, total_phase=720, projection=None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18, n_decimal = 0,
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, 
                                    ax_label=False, nuc_order = [ 'FSI', 'D2', 'STN', 'Arky', 'Proto'], 
                                    y_max_series = y_max_series, set_ylim = True)
save_pdf_png(fig, os.path.join(path, 'SNN_phase_' + status + '_plot_' + state_1),
              size=(3, 1.5 * len (name_list)))

# %% Transition to Beta induction, all nuclei

# K_real = K_real_bouton

#### modulation with laser beta inductions of de la crompe 2020, and Asier unpublished
mod_dict = {'ChR2_STN' : {'STN': [1.6, 93], 'Proto' : [36, 72], 'Arky' : [25, 3]},
            'ArchT_Proto': {'STN': [3.5, 34], 'Proto' : [44, 5], 'Arky' : [7, 23]},
            'ArchT_STN': {'STN': [16, 0], 'Proto' : [15, 9], 'Arky' : [0, 0]},
            'ChR2_D2_mouse': {'D2': [0, 36.6], 'STN': [3.8, 8], 'Proto' : [11.4, 1.1], 'Arky' : [4.8, 17.5]},
            'ChR2_STN_mouse' : {'STN': [0.7, 46.7], 'Proto' : [20, 57.5], 'Arky' : [0.4, 3.7]}}

## baseline mouse:
final_act_rat = {}
A_mouse = {'D2' : 0.066, 'Proto' : 25, 'Arky' : 3, 'STN' : 4}

for pop in ['STN', 'Proto', 'Arky']:
    
    rel_change_Proto_mouse = (mod_dict['ChR2_D2_mouse'][pop][0] - mod_dict['ChR2_D2_mouse'][pop][1]) \
                            / mod_dict['ChR2_D2_mouse'][pop][0]
    final_act_rat[pop] = Act['rest'][pop] * (1 - rel_change_Proto_mouse)



plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
K_small = calculate_number_of_connections(dict.fromkeys(N, 1000), N_real, K_real)
K_ratio = {key : K[key] / v for key, v in K_small.items()}
dt = 0.1


t_sim = 6900 
# t_sim = 3000
t_list = np.arange(int(t_sim/dt))
plot_start = 300
t_transition = plot_start + 0# int(t_sim / 5)
duration_base = np.array( [int(0/dt), int(t_transition/dt)] )
duration_induction = np.array( [int(t_transition / dt) + int(300/dt) , int(t_sim / dt)] ) 
end_phase = t_sim 
t_mvt = t_sim
D_mvt = t_sim - t_mvt
n_windows_DD = 3
n_windows_base = 3


name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'

state_1 = 'rest'
state_2 = 'induction'

induction_nuc_name = 'D2'
beta_induction_method = 'excitation'

induction_nuc_name = 'Proto'
beta_induction_method = 'inhibition'

induction_nuc_name = 'STN'
beta_induction_method = 'excitation'
# beta_induction_method = 'inhibition'
# state_1 = '_'.join(['induction', induction_nuc_name, beta_induction_method])

# neuronal_consts['Proto']['membrane_time_constant'] = {'mean': 43, 'var': 10, 'truncmin': 3, 'truncmax': 100}
beta_induc_name_list = [induction_nuc_name]


amplitude_dict = {'inhibition':{'Proto': 6.4, 'STN': 2.28}, 
                  'excitation': {'D2': 15, 'STN': 4.9}} ### N = 1000 log-normal G SD = 10**2

amplitude_dict = {'inhibition':{'Proto': 7.1, 'STN': 2.30}, 
                  'excitation': {'D2': 15, 'STN': 4.9}} ### N = 1000 log-normal G SD = 10**1, 17.5 Hz

amplitude_dict = {'inhibition':{'Proto': 6.5, 'STN': 2.30}, 
                  'excitation': {'D2': 15, 'STN': 4.9}} ### N = 1000 log-normal G SD = 10**1

freq_dict = {induction_nuc_name: 20} 
start_dict = {induction_nuc_name : int(t_transition / dt) }
end_dict = {induction_nuc_name: int(t_sim / dt)}
mean_dict = {induction_nuc_name : 0 }

print( " transition from " , state_1, ' to ',  induction_nuc_name, ' ', beta_induction_method, ' beta ', state_2)

name_list = [name1, name2, name3, name4, name5]


g = -0.0025 ## log-normal syn weight dist F = 18.5 Hz
G = { (name2, name1) :{'mean': g * K[name2, name1] * 11},#}, ## free
      (name3, name2) :{'mean': g * K[name3, name2] * 11},#11.}, ## free
      (name1, name3) :{'mean': g * K[name1, name3] * 11},#30 * 66/63}, ## free
      (name2, name4) :{'mean': g * K[name2, name4] * 4},#0.01}, ## free
      (name4, name3) :{'mean': g * K[name4, name3] * 3},
      (name3, name5) :{'mean': -g * K[name3, name5] * 2.4},
      (name5, name3) :{'mean': g * K[name5, name3] * 4.7},# 4.7},
      (name3, name3) :{'mean': g * K[name3, name3] * 1.25}}#2.}}#, 
      # (name1, name5) :{'mean': g * K[name1, name5] * 1}}


# g = -0.0025 ## log-normal syn weight dist F = 17.5 Hz not tuned in DD?
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 13.5},#6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 13.5},#11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 13.5},#30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 5},#0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.4},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4.7},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.9}}#2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}



# g = -0.0025 ## log-normal syn weight dist F = 17.3 Hz SD = 10**2
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 13.5},#6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 13.5},#11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 13.5},#30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 5},#0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 3.2},
#       (name5, name3) :{'mean': g * K[name5, name3] * 6.3},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.9}}#2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}


# g = -0.0025 ## K_connections tuned July 2022 N = 1000 f = 18 Hz
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 11}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.5},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.45 * 62/60},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4.3 * 205/180},
#       (name3, name3) :{'mean': g * K[name3, name3] * 2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}

# g = -0.0025 ## FENS poster
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 10},#6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 10},#11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 11},#30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 10},#0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.5},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.45 * 62/60},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4.3 * 205/180},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.5}}#2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}


# g = -0.0025 ## K_connections tuned July 2022
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 30}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 5}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 8}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 10}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.5},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.5},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1}
#       }




# g = -0.01 ### for N = 3500
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 9.63}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 1.53}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 6.3}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 7.965}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.62},
#       (name3, name5) :{'mean': -g * K[name3, name5]* 2.65},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.5},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.2}
#       }

# g = -0.01 ### 16 Hz for N = 2000
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 13.7}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 1.78}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 7.0}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 8.85}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.62},
#       (name3, name5) :{'mean': -g * K[name3, name5]* 2.65},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.1},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.13}
#       }



# g = -0.0025 ## bouton connection tuned
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 3.7}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 3.8}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 3.6}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 3.5}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 1.3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.2},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.005}
#       }

G = set_G_dist_specs(G, order_mag_sigma = 1)


poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 
                 'tau': { 'rise': {'mean': 1, 'var': .5}, 
                         'decay': {'mean': 5, 'var': 2.3}}, 
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
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 12; high_f = 30
spike_history = 'long-term'
set_random_seed = False
plot_syn_weight_hist = False
nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state_1,
               spike_history = spike_history, Act = Act,  external_input_bool = True) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                                       all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                       set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                                       use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1, plot_syn_weight_hist = plot_syn_weight_hist)

run_transition_to_beta_induction( receiving_class_dict,  beta_induc_name_list, dt, amplitude_dict[beta_induction_method], freq_dict, 
                                     start_dict, end_dict, mean_dict,
                                     t_list, nuclei_dict, t_transition= t_transition, method = beta_induction_method)

nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'transition_to_'  + state_2 + '_with_' + beta_induction_method + '_at_'+ induction_nuc_name

ylim = (-2, 75)
plot_start = t_transition - 100
plot_end = t_transition + 400



fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_1], t_transition, D_mvt, ax=None, 
            title_fontsize=15, plot_start= plot_start, plot_end = plot_end, title='', legend_loc='upper left',  
            ylim=(-4, 80), vspan=True, include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, 
            alpha_mvt=0.8, axvspan_color=axvspan_color[state_2])

# fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_1], t_transition, D_mvt, ax=fig.gca(), 
#        title_fontsize=15, plot_start= plot_start, plot_end = plot_end, title='', legend_loc='upper left',  
#        ylim=(-4, 80), vspan=True, include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, 
#        alpha_mvt=0.8, axvspan_color=axvspan_color[state_2], plot_filtered=True, low_f = 15, high_f = 25)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_from_' + state_1 ),
             size=(6, 5))

# fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, 
#                                     plot_end = plot_end, ax_label = True, labelsize=10, title_fontsize=15, lw=1., 
#                                     linelengths=2, n_neuron=40, include_title=True, set_xlim=True, ylabel_x = 0.01,
#                                     axvspan=True, span_start=t_transition, span_end=t_sim, 
#                                     axvspan_color=axvspan_color[state_2], include_nuc_name = False,
#                                     tick_label_fontsize=12)

# save_pdf_png(fig_raster, os.path.join(
#     path, 'SNN_raster_' + status ), size=(10, 5))
    
# peak_threshold = 0.1
# smooth_window_ms = 5
# cut_plateau_epsilon = 0.1
# lim_oscil_perc = 10
# low_pass_filter = False

# fig, ax = plt.subplots(1, 1)
# find_freq_all_nuclei(dt, nuclei_dict, duration_induction, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
#                          low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
#                          fft_method='Welch', n_windows=n_windows_DD, include_beta_band_in_legend=False)

# ax.set_xlim(5, 70)
# ax.axvspan(5, 70, alpha=0.2, color=axvspan_color[state_2])
# ax.set_ylim(-0.01, 40)
# ax.legend(fontsize=10, frameon=False)
# ax.tick_params(axis='both', labelsize=15)
# save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_2),
#               size=(5, 3))

# fig, ax = plt.subplots(1, 1)
# find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
#                          low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
#                          fft_method='Welch', n_windows=n_windows_base, include_beta_band_in_legend=False, include_peak_f_in_legend = False)

# ax.set_xlim(5, 70)
# ax.set_ylim(-0.01, 40)
# ax.tick_params(axis='both', labelsize=15)
# ax.legend(fontsize=10, frameon=False)
# save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_1),
#               size=(5, 3))

phase_ref = 'stimulation'

# phase_ref = 'D2'
find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins = 120,
                                  peak_threshold=0, phase_ref = phase_ref, start=duration_induction[0], 
                                  end = end_phase, total_phase= 720, plot = False, 
                                  troughs = False, align_to_stim_onset = True, only_rtest_entrained = False)

FR_dict = create_FR_dict_from_sim(nuclei_dict, 
                                  int( plot_start / dt), 
                                  int (t_transition / dt), 
                                  state = 'rest')

fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, coef = 1000,
                                    density=False, phase_ref= phase_ref, total_phase=720, projection=None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18,
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, 
                                    ax_label=False, nuc_order = ['FSI', 'D2', 'STN', 'Arky', 'Proto'],
                                    scale_count_to_FR = True, FR_dict = FR_dict, plot_FR = True,
                                    f_stim = freq_dict [induction_nuc_name])



fig.suptitle(' '.join([induction_nuc_name, beta_induction_method]), fontsize = 15)
save_pdf_png(fig, os.path.join(path, 'SNN_Phase_' + status + '_plot_' + state_1),
              size=(3, len(name_list) *1.5))

# %% Transition to DD FSI-D2-GPe + STN-GPe collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.2
t_sim = 5000
t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 5)
t_transition = plot_start + int(t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim / dt) - length +
               int(t_transition / 4 / dt), int(t_sim / dt)]


plot_start = t_transition - 1000
plot_end = t_transition + 1000

name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
name5 = 'STN'

state_1 = 'rest'
state_2 = 'DD_anesth'


print( " transition from " , state_1, ' to ', state_2)

name_list = [name1, name2, name3,  name5]

G = {}

### 17 Hz
# g_STN_Loop = 1.5
# g_FSI_Loop = 10

g_STN_Loop = 1.7
g_FSI_Loop = 11
g_FSI = -g_FSI_Loop ** (1/3)
g_STN = -g_STN_Loop ** (1/2)

(G[(name2, name1)], G[(name3, name2)],
 G[(name1, name3)], G[(name3, name5)],
 G[(name5, name3)]) = g_FSI, g_FSI, g_FSI, -g_STN, g_STN


g_STN_Loop = 1.7
g_FSI_Loop = 11
g_FSI = -g_FSI_Loop ** (1/3)
g_STN = -g_STN_Loop ** (1/2)
g_FSI = -2.3
g_STN = -1.6
(G[(name2, name1)], G[(name3, name2)],
 G[(name1, name3)], G[(name3, name5)],
 G[(name5, name3)]) = g_FSI, g_FSI, g_FSI, -g_STN, g_STN

# 3* g, 3 * g, 2.5 * g, 2.5 * g, 2.5 * g, 3. * -g , 3 * g , g * 0.1 --> g = 0.002 single run 17 Hz, average 15.4
# 2.5* g, 3.5 * g, 2.5 * g, 2.5 * g, 2.5 * g, 3.5 * -g , 3. * g , g * 0.1 --> 16.8 Hz g = 0.002
# [2, 2.5, 1, 1, 1, 3, 1, 0.1] --> 17 Hz g = 0.002
# x = [1.5, 2.5, 1, 1, 1, 3, 1, 0.1] --> 17 Hz g = 0.003 double peak 19 and 40
# 2.8* g, 3.5 * g, 2.5 * g, 2.5 * g, 2.5 * g, 4 * -g , 3. * g , g * 0.1--> 19 Hz g = 0.002 Noooo

# G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name5, '1')],
                      (name5, '1'): [(name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 20

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state_1) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)

nuclei_dict = run_transition_state_collective_setting(G, noise_variance, noise_amplitude, path, receiving_class_dict, receiving_pop_list, 
                                                      t_list, dt, nuclei_dict, Act, state_1, state_2, K_all, N, N_real,
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity, t_transition = int(t_transition/dt))


nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'transition_to_' + state_2

D_mvt = t_sim - t_transition
if 'DD' in state_2:
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, title='', legend_loc='upper left', plot_end=t_transition-10, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8)
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_' + state_1),
                 size=(8, 5))
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start = t_sim - (t_transition-plot_start), title='', 
               legend_loc='upper left', plot_end= t_sim, vspan=True, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color = axvspan_color[state_2])
    
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_plot_' + state_2),
                 size=(8, 5))
    fig_state_1, fig_state_2 = raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start,
                                             labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=False, set_xlim=True,
                                             axvspan_color=axvspan_color[state_2], n=N_sim,  ylabel_x=0.01,
                                             t_transition=t_transition, t_sim=t_sim, tick_label_fontsize=12, include_nuc_name=False)

    save_pdf_png(fig_state_1, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_1), size=(3, 5))
    save_pdf_png(fig_state_2, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_2), size=(3, 5))
    
elif 'mvt' in state_2:
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, plot_end = plot_end, title='', legend_loc='upper left',  ylim=(-4, 80), vspan=True,
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color=axvspan_color[state_2])
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_from_' + state_1),
                 size=(15, 5))
    
    fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=plot_end, ax_label = True,
                                  labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=True, set_xlim=True, ylabel_x = 0.01,
                                  axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color=axvspan_color[state_2], include_nuc_name = False,
                                   tick_label_fontsize=12)
    save_pdf_png(fig_raster, os.path.join(
        path, 'SNN_raster_' + status ), size=(5, 5))
    
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)

ax.set_xlim(5, 70)
ax.axvspan(5, 70, alpha=0.2, color=axvspan_color[state_2])
ax.set_ylim(-0.01, 0.1)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(axis='both', labelsize=15)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_1),
             size=(5, 3))

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False, include_peak_f_in_legend = False)

ax.set_xlim(5, 70)
ax.set_ylim(-0.01, 0.1)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=10, frameon=False)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_2),
             size=(5, 3))

fig, ax = plt.subplots(1, 1)
ax.annotate([g_FSI_Loop, g_STN_Loop], xy=(0.2,0.5),xycoords='axes fraction', 
 fontsize= 20)
# %% Transition to DD FSI-D2-GPe + GPe-GPe collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.2
t_sim = 5000
t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 10)
t_transition = plot_start + int(t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim / dt) - length +
               int(t_transition / 4 / dt), int(t_sim / dt)]


plot_start = t_transition - 1000
plot_end = t_transition + 1000

name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'

state_1 = 'rest'
state_2 = 'DD_anesth'


print( " transition from " , state_1, ' to ', state_2)

name_list = [name1, name2, name3]

G = {}

g_GPe_Loop = 0.6
g_FSI_Loop = 7
g_FSI = -g_FSI_Loop ** (1/3)
g_GPe = -g_GPe_Loop

print(g_FSI, g_GPe)
(G[(name2, name1)], G[(name3, name2)],
 G[(name1, name3)],  G[(name3, name3)]) = g_FSI, g_FSI, g_FSI, g_GPe,

# (G[(name2, name1)], G[(name3, name2)],
#  G[(name1, name3)],  G[(name3, name3)]) = -1, -2.5, -3.2, -0.62


# 3* g, 3 * g, 2.5 * g, 2.5 * g, 2.5 * g, 3. * -g , 3 * g , g * 0.1 --> g = 0.002 single run 17 Hz, average 15.4
# 2.5* g, 3.5 * g, 2.5 * g, 2.5 * g, 2.5 * g, 3.5 * -g , 3. * g , g * 0.1 --> 16.8 Hz g = 0.002
# [2, 2.5, 1, 1, 1, 3, 1, 0.1] --> 17 Hz g = 0.002
# x = [1.5, 2.5, 1, 1, 1, 3, 1, 0.1] --> 17 Hz g = 0.003 double peak 19 and 40
# 2.8* g, 3.5 * g, 2.5 * g, 2.5 * g, 2.5 * g, 4 * -g , 3. * g , g * 0.1--> 19 Hz g = 0.002 Noooo

# G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 20

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state_1) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)

nuclei_dict = run_transition_state_collective_setting(G, noise_variance, noise_amplitude, path, receiving_class_dict, receiving_pop_list, 
                                                      t_list, dt, nuclei_dict, Act, state_1, state_2, K_all, N, N_real,
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity, t_transition = int(t_transition/dt))


nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'transition_to_' + state_2

D_mvt = t_sim - t_transition
if 'DD' in state_2:
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, title='', legend_loc='upper left', plot_end=t_transition-10, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8)
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_' + state_1),
                 size=(8, 5))
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start = t_sim - (t_transition-plot_start), title='', 
               legend_loc='upper left', plot_end= t_sim, vspan=True, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color = axvspan_color[state_2])
    
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_plot_' + state_2),
                 size=(8, 5))
    fig_state_1, fig_state_2 = raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start,
                                             labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=False, set_xlim=True,
                                             axvspan_color=axvspan_color[state_2], n=N_sim,  ylabel_x=0.01,
                                             t_transition=t_transition, t_sim=t_sim, tick_label_fontsize=12, include_nuc_name=False)

    save_pdf_png(fig_state_1, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_1), size=(3, 5))
    save_pdf_png(fig_state_2, os.path.join(
        path, 'SNN_raster_' + status + '_plot_' + state_2), size=(3, 5))
    
elif 'mvt' in state_2:
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, plot_end = plot_end, title='', legend_loc='upper left',  ylim=(-4, 80), vspan=True,
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color=axvspan_color[state_2])
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_from_' + state_1),
                 size=(15, 5))
    
    fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=plot_end, ax_label = True,
                                  labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=True, set_xlim=True, ylabel_x = 0.01,
                                  axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color=axvspan_color[state_2], include_nuc_name = False,
                                   tick_label_fontsize=12)
    save_pdf_png(fig_raster, os.path.join(
        path, 'SNN_raster_' + status ), size=(5, 5))
    
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)

ax.set_xlim(5, 70)
ax.axvspan(5, 70, alpha=0.2, color=axvspan_color[state_2])
# ax.set_ylim(-0.01, 0.1)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(axis='both', labelsize=15)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_1),
             size=(5, 3))

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False, include_peak_f_in_legend = False)

ax.set_xlim(5, 70)
# ax.set_ylim(-0.01, 0.1)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=10, frameon=False)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_2),
             size=(5, 3))

fig, ax = plt.subplots(1, 1)
ax.annotate([g_FSI_Loop, g_GPe_Loop], xy=(0.2,0.5),xycoords='axes fraction', 
 fontsize= 20)

# %% Transition to DD FSI-D2-GPe + STN-GPe + GPe-GPe collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.2
t_sim = 5000
t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 5)
t_transition = plot_start + int(t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim / dt) - length +
               int(t_transition / 4 / dt), int(t_sim / dt)]


plot_start = t_transition - 1000
plot_end = t_transition + 1000

name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
name5 = 'STN'

state_1 = 'rest'
state_2 = 'DD_anesth'


print( " transition from " , state_1, ' to ', state_2)

name_list = [name1, name2, name3,  name5]

G = {}

### 17 Hz
# g_STN_Loop = 1.5
# g_FSI_Loop = 10

g_STN_Loop = 1.7
g_FSI_Loop = 11
g_FSI = -g_FSI_Loop ** (1/3)
g_STN = -g_STN_Loop ** (1/2)

(G[(name2, name1)], G[(name3, name2)],
 G[(name1, name3)], G[(name3, name5)],
 G[(name5, name3)]) = g_FSI, g_FSI, g_FSI, -g_STN, g_STN


g_STN_Loop = 0.5
g_FSI_Loop = 12
g_FSI = -g_FSI_Loop ** (1/3)
g_STN = -g_STN_Loop ** (1/2)
g_STN = -1.6 
g_FSI = -2.3
g_GPe = -0.1

g = -0.0018
(G[(name2, name1)], G[(name3, name2)],
 G[(name1, name3)], G[(name3, name5)],
  G[(name5, name3)], G[(name3, name3)]) = g_FSI, g_FSI, g_FSI, -g_STN, g_STN, g_GPe
# 3* g, 3 * g, 2.5 * g, 2.5 * g, 2.5 * g, 3. * -g , 3 * g , g * 0.1 --> g = 0.002 single run 17 Hz, average 15.4
# 2.5* g, 3.5 * g, 2.5 * g, 2.5 * g, 2.5 * g, 3.5 * -g , 3. * g , g * 0.1 --> 16.8 Hz g = 0.002
# [2, 2.5, 1, 1, 1, 3, 1, 0.1] --> 17 Hz g = 0.002
# x = [1.5, 2.5, 1, 1, 1, 3, 1, 0.1] --> 17 Hz g = 0.003 double peak 19 and 40
# 2.8* g, 3.5 * g, 2.5 * g, 2.5 * g, 2.5 * g, 4 * -g , 3. * g , g * 0.1--> 19 Hz g = 0.002 Noooo

G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name5, '1'), (name3, '1')],
                      (name5, '1'): [(name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 20

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state_1) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)

nuclei_dict = run_transition_state_collective_setting(G, noise_variance, noise_amplitude, path, receiving_class_dict, receiving_pop_list, 
                                                      t_list, dt, nuclei_dict, Act, state_1, state_2, K_all, N, N_real,
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity, t_transition = int(t_transition/dt))


nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'transition_to_' + state_2

D_mvt = t_sim - t_transition
if 'DD' in state_2:
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, title='', legend_loc='upper left', plot_end=t_transition-10, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8)
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_' + state_1),
                 size=(8, 5))
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start = t_sim - (t_transition-plot_start), title='', 
               legend_loc='upper left', plot_end= t_sim, vspan=True, ylim=(-4, 80),
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color = axvspan_color[state_2])
    
    # save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_plot_' + state_2),
    #              size=(8, 5))
    fig_state_1, fig_state_2 = raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start,
                                             labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=False, set_xlim=True,
                                             axvspan_color=axvspan_color[state_2], n=N_sim,  ylabel_x=0.01,
                                             t_transition=t_transition, t_sim=t_sim, tick_label_fontsize=12, include_nuc_name=False)

    # save_pdf_png(fig_state_1, os.path.join(
    #     path, 'SNN_raster_' + status + '_plot_' + state_1), size=(3, 5))
    # save_pdf_png(fig_state_2, os.path.join(
    #     path, 'SNN_raster_' + status + '_plot_' + state_2), size=(3, 5))
    
elif 'mvt' in state_2:
    
    fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
               plot_start= plot_start, plot_end = plot_end, title='', legend_loc='upper left',  ylim=(-4, 80), vspan=True,
               include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color=axvspan_color[state_2])
    save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_from_' + state_1),
                 size=(15, 5))
    
    fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start, plot_end=plot_end, ax_label = True,
                                  labelsize=10, title_fontsize=15, lw=1., linelengths=2, n_neuron=40, include_title=True, set_xlim=True, ylabel_x = 0.01,
                                  axvspan=True, span_start=t_transition, span_end=t_sim, axvspan_color=axvspan_color[state_2], include_nuc_name = False,
                                   tick_label_fontsize=12)
    save_pdf_png(fig_raster, os.path.join(
        path, 'SNN_raster_' + status ), size=(5, 5))
    
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)

ax.set_xlim(5, 70)
ax.axvspan(5, 70, alpha=0.2, color=axvspan_color[state_2])
# ax.set_ylim(-0.01, 0.1)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(axis='both', labelsize=15)
# save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_1),
#              size=(5, 3))

fig, ax = plt.subplots(1, 1)
find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False, include_peak_f_in_legend = False)

ax.set_xlim(5, 70)
ax.set_ylim(-0.01, 0.1)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=10, frameon=False)
# save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_2),
#              size=(5, 3))

fig, ax = plt.subplots(1, 1)
ax.annotate([g_FSI_Loop, g_STN_Loop], xy=(0.2,0.5),xycoords='axes fraction', 
 fontsize= 20)
# %% Transition FSI-D2-GPe + Arky-D2-GPe + STN-GPe + GPe-GPe collective multi run


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


# g = -0.0018
# (G[(name2, name1)], G[(name3, name2)],
#  G[(name1, name3)], G[(name2, name4)],
#  G[(name4, name3)], G[(name3, name5)],
#  G[(name5, name3)], G[(name3, name3)]) = 3* g, 3.5 * g, 2.5 * g, 2.5 * g, 2.5 * g, 3. * -g , 3.8 * g , g * 0.1
# G = {k: v * K[k] for k, v in G.items()}
# print_G_items(G)


g = -0.0025 ## log-normal syn weight dist F = 17.3 Hz
G = { (name2, name1) :{'mean': g * K[name2, name1] * 11}, ## free
      (name3, name2) :{'mean': g * K[name3, name2] * 12.2}, ## free
      (name1, name3) :{'mean': g * K[name1, name3] * 11}, ## free
      (name2, name4) :{'mean': g * K[name2, name4] * 3}, ## free
      (name4, name3) :{'mean': g * K[name4, name3] * 3},
      (name3, name5) :{'mean': -g * K[name3, name5] * 2.4},
      (name5, name3) :{'mean': g * K[name5, name3] * 4.7},
      (name3, name3) :{'mean': g * K[name3, name3] * 1.35}} # further increasing it comes at the cost of reducing the 17 hz peak and further separating the high-low frequencies.
      # (name1, name5) :{'mean': g * K[name1, name5] * 1}}

# g = -0.0025 ## log-normal syn weight dist F = 17.6 Hz
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 11}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 12.15}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 11}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 3}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.4},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4.7},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.3}} # further increasing it comes at the cost of reducing the 17 hz peak and further separating the high-low frequencies.
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}

# g = -0.0025 ## log-normal syn weight dist F = 18 Hz SD = 10**2
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 13.5},#6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 13.5},#11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 13.5},#30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 5},#0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 3.2},
#       (name5, name3) :{'mean': g * K[name5, name3] * 6.3},
#       (name3, name3) :{'mean': g * K[name3, name3] * 1.2}}#2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}
      
      
# g = -0.0025 ## log-normal syn weight dist F = 17.3 Hz
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 0},#13.5},#6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 0},#11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 0},#30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 0},#0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 0},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 3.2},
#       (name5, name3) :{'mean': g * K[name5, name3] * 6.3},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0}}#2.}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}


# g = -0.0025 ## K_connections tuned July 2022 N = 1000 f = 17-18 Hz
# G = { (name2, name1) :{'mean': g * K[name2, name1]  * 6}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 11.}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 30 * 66/63}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 0.01}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 2.5},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.45 * 62/60},
#       (name5, name3) :{'mean': g * K[name5, name3] * 4.3 * 205/180},
#       (name3, name3) :{'mean': g * K[name3, name3] * 2.2}}#, 
#       # (name1, name5) :{'mean': g * K[name1, name5] * 1}}

# g = -0.0025 
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.5}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 3}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 3.5}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 3.5}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 1.3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 2.},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.2},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.2}
#       }

# g = -0.0025 # Brice FR wasn't measured right
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.5}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 3}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 3.5}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 4.}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 1.3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 0.9},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.2},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.4}
#       }

# G = { (name2, name1) :{'mean': g * K[name2, name1] * 5}, ## free
#       (name3, name2) :{'mean': g * K[name3, name2] * 5.5}, ## free
#       (name1, name3) :{'mean': g * K[name1, name3] * 4.5}, ## free
#       (name2, name4) :{'mean': g * K[name2, name4] * 4.5}, ## free
#       (name4, name3) :{'mean': g * K[name4, name3] * 0.9},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 3.6},
#       (name5, name3) :{'mean': g * K[name5, name3] * 3.2},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.1}
#       }




# g = -0.0025 ## before exact tuning
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.9},
#       (name3, name2) :{'mean': g * K[name3, name2] * 3},
#       (name1, name3) :{'mean': g * K[name1, name3] * 3.8},
#       (name2, name4) :{'mean': g * K[name2, name4] * 3.9},
#       (name4, name3) :{'mean': g * K[name4, name3] * 0.9},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 1},
#       (name5, name3) :{'mean': g * K[name5, name3] * 2.},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.1}
#       }



G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2, order_mag_sigma = 2)
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
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_all[state_1], receiving_pop_list, nuclei_dict, t_list,
                                                       all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                       set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                                       use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)



# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 1; plot_firing = False; plot_spectrum = False; plot_raster = False; plot_phase = False; low_pass_filter = False; save_pkl = True; save_figures = False; save_pxx = True
save_pop_act = True
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
check_peak_significance = False
low_f = 12; high_f = 30
phase_ref = 'Proto'

filename = ('All_nuc_from_' + state_1 + '_to_' + state_2 + '_N_1000_T_' + str( int(( duration[1] -duration[0]) * dt) ) +
             '_n_' + str(n_run) + '_runs_aligned_to_' + phase_ref + '_tuned_to_Brice_G_lognormal.pkl')

filepath = os.path.join(path, 'Beta_power', filename)
fft_method = 'Welch'
nuc_order = ['D2', 'STN', 'Arky', 'Proto', 'FSI']


data = multi_run_transition(path, nuclei_dict, filepath, duration, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                            noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                            lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                            plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                            find_beta_band_power=True, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                            reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                            K_real=K_all[state_2], N_real=N_real, N=N, divide_beta_band_in_power=True,
                            receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                            use_saved_FR_ext=True, check_peak_significance=check_peak_significance,
                            find_phase=True, phase_thresh_h=None, filter_order=6, low_f= low_f, high_f= high_f,
                            n_phase_bins=n_phase_bins, start_phase= duration[0], phase_ref=phase_ref, save_pxx=save_pxx,
                            plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=False,
                            nuc_order=nuc_order, len_f_pxx=150, state_1 = state_1, state_2 = state_2, K_all = K_all, 
                            state_change_func = change_network_states, end_phase = duration[1], save_pop_act = save_pop_act,
                            only_rtest_entrained = True, threshold_by_percentile = 50, shift_phase_deg = 45)
pickle_obj(data, filepath)
plt.figure()

# %% Transition to Beta induction, All nuclei collective multi run


plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 21000 
t_list = np.arange(int(t_sim/dt))
plot_start = 1000
t_transition = plot_start + 0# int(t_sim / 5)
duration_base = np.array( [int(200/dt), int(t_transition/dt)] )
duration_DD = np.array( [int(t_transition / dt) + int(300/dt) , int(t_sim / dt)] ) 
t_mvt = t_sim
D_mvt = t_sim - t_mvt

duration = duration_DD - int( t_transition /dt )
t_sim = t_sim - t_transition
t_list = np.arange(int(t_sim/dt))
end_phase = t_sim - int(300 / dt)
n_phase_bins = 120
n_windows = 10

name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'
name_list = [name1, name2, name3, name4, name5]

state_1 = 'rest'
state_2 = 'induction'
G = {}

print( " transition from " , state_1, ' to ', state_2)




induction_nuc_name = 'D2'
beta_induction_method = 'excitation'

induction_nuc_name = 'Proto'
beta_induction_method = 'inhibition'

# induction_nuc_name = 'STN'
# beta_induction_method = 'excitation'
# beta_induction_method = 'inhibition'

# neuronal_consts['Proto']['membrane_time_constant'] = {'mean': 43, 'var': 10, 'truncmin': 3, 'truncmax': 100}
beta_induc_name_list = [induction_nuc_name]

amplitude_dict = {'inhibition':{'Proto': 6, 'STN': 2.28}, 
                  'excitation': {'D2': 15, 'STN': 6.2}} 

freq_dict = {induction_nuc_name: 20} 
start_dict = {induction_nuc_name : int(t_transition / dt) }
end_dict = {induction_nuc_name: int(t_sim / dt)}
mean_dict = {induction_nuc_name : 0 }


print( " transition from " , state_1, ' to ',  induction_nuc_name, ' ', beta_induction_method, ' beta ', state_2)

name_list = [name1, name2, name3, name4, name5]

G = {}


g = -0.0025 ## log-normal syn weight dist F = 18.5 Hz
G = { (name2, name1) :{'mean': g * K[name2, name1] * 11},#}, ## free
      (name3, name2) :{'mean': g * K[name3, name2] * 11},#11.}, ## free
      (name1, name3) :{'mean': g * K[name1, name3] * 11},#30 * 66/63}, ## free
      (name2, name4) :{'mean': g * K[name2, name4] * 4},#0.01}, ## free
      (name4, name3) :{'mean': g * K[name4, name3] * 3},
      (name3, name5) :{'mean': -g * K[name3, name5] * 2.4},
      (name5, name3) :{'mean': g * K[name5, name3] * 4.7},# 4.7},
      (name3, name3) :{'mean': g * K[name3, name3] * 1.25}}#2.}}#, 
      # (name1, name5) :{'mean': g * K[name1, name5] * 1}}




G = set_G_dist_specs(G, order_mag_sigma = 2)
G_dict = {k: {'mean': [v['mean']]} for k, v in G.items()}

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
               syn_component_weight=syn_component_weight, noise_method=noise_method,Act = Act, state = state_1, external_input_bool = True) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_all[state_1], receiving_pop_list, nuclei_dict, t_list,
                                                       all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                       set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                                       use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)



# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 1; plot_firing = False; plot_spectrum = False; plot_raster = False; plot_phase = False; low_pass_filter = False; save_pkl = True; save_figures = False; save_pxx = True
 
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
check_peak_significance = False
low_f = 12; high_f = 30
phase_ref = 'stimulation'
# phase_ref = 'D2'

filename = ('All_nuc_from_' + state_1 + '_to_'  + state_2 + 
            '_with_' + beta_induction_method + '_at_'+ induction_nuc_name + 
            '_N_1000_T_' + str( int(( duration[1] -duration[0]) * dt) ) +
             '_n_' + str(n_run) + '_runs_aligned_to_' + phase_ref + '.pkl')

filepath = os.path.join(path, 'Beta_power', filename)
fft_method = 'Welch'
nuc_order = ['D2', 'STN', 'Arky', 'Proto', 'FSI']

data = multi_run_transition(path, nuclei_dict, filepath, duration, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                            noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                            lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                            plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                            find_beta_band_power=True, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                            reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                            K_real=K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                            receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                            use_saved_FR_ext=True, check_peak_significance=check_peak_significance,
                            find_phase=True, phase_thresh_h=0, filter_order=6, low_f= low_f, high_f= high_f,
                            n_phase_bins=n_phase_bins, start_phase=int(t_sim/4), phase_ref=phase_ref, save_pxx=save_pxx,
                            plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                            nuc_order=nuc_order, len_f_pxx=150, state_1 = state_1, state_2 = state_2, K_all = K_all, 
                            beta_induc_name_list = beta_induc_name_list, amplitude_dict = amplitude_dict[beta_induction_method] , end_phase = end_phase,
                            freq_dict = freq_dict, start_dict = start_dict, end_dict = end_dict, mean_dict = mean_dict,
                            induction_method = beta_induction_method, only_rtest_entrained = True)

# %% Transition to activated state FSI-D2-GPe + Arky-D2-GPe + STN-GPe collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 5)
# plot_end = int((t_transition + 1000)/dt)
t_transition = plot_start + int(t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)]
length = duration_base[1] - duration_base[0]
duration_mvt = [int(t_sim / dt) - length, int(t_sim / dt)]
duration_last = [int(t_sim / dt) - length, int(t_sim / dt)]
name1 = 'FSI'
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'

state_1 = 'rest'
state_2 = 'mvt'
name_list = [name1, name2, name3, name4, name5]

g_ext = 0.01
G = {}

g = -0.0022
(G[(name2, name1)], G[(name3, name2)],
 G[(name1, name3)], G[(name2, name4)],
 G[(name4, name3)], G[(name3, name5)],
 G[(name5, name3)], G[(name3, name3)]) = g, g*1.1, g, g, g * 0.8, -g*2.5, g*2.5, g * 0.1
G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}

receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1'), (name4, '1')],
                      # (name3, '1'): [(name2,'1'), (name5,'1')],
                      # with GP-GP
                      (name3, '1'): [(name2, '1'), (name3, '1'), (name5, '1')],
                      (name4, '1'): [(name3, '1')],
                      (name5, '1'): [(name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True
low_f = 8; high_f = 20

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude, N, Act[state_1], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state_1) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict,nuclei_dict = set_connec_ext_inp(path, Act[state_1], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_1)

nuclei_dict = run_transition_state_collective_setting(G, noise_variance, noise_amplitude, path, receiving_class_dict, receiving_pop_list, 
                                                      t_list, dt, nuclei_dict, Act, state_1, state_2, K_all, N, N_real,
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity, t_transition = int(t_transition/dt))


nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)
status = 'transition_to_' + state_2

D_mvt = t_sim - t_transition
fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
           plot_start= plot_start, title='', legend_loc='upper left', plot_end=t_transition-10, ylim=(-4, 80),
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color='darkseagreen')
save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_' + state_1),
             size=(8, 5))

fig = plot(nuclei_dict, color_dict, dt, t_list, Act[state_1], Act[state_2], t_transition, D_mvt, ax=None, title_fontsize=15, 
           plot_start = t_sim - (t_transition-plot_start), title='',
           legend_loc='upper left', plot_end=t_sim, vspan=True, ylim=(-4, 80),
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt=0.8, axvspan_color='darkseagreen')

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status + '_plot_' + state_2),
             size=(8, 5))

fig_state_1, fig_state_2 = raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=plot_start,
                                             labelsize=10, title_fontsize=15, lw=1.5, linelengths=2, n_neuron=40, include_title=True, set_xlim=True,
                                             axvspan_color='darkseagreen', n=1000,  ylabel_x=0.01,
                                             t_transition=t_transition, t_sim=t_sim, tick_label_fontsize=12)

save_pdf_png(fig_state_1, os.path.join(
    path, 'SNN_raster_' + status + '_plot_' + state_1), size=(3, 5))
save_pdf_png(fig_state_2, os.path.join(
    path, 'SNN_raster_' + status + '_plot_' + state_2), size=(3, 5))

fig, ax = plt.subplots(1, 1)
peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
find_freq_all_nuclei(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)

ax.set_xlim(5, 55)
ax.axvspan(5, 55, alpha=0.2, color='darkseagreen')
ax.set_ylim(-0.01, 0.1)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(axis='both', labelsize=15)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_1),
             size=(5, 3))
fig, ax = plt.subplots(1, 1)

find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, False, 'fft', False,
                         low_pass_filter, 0, 2000, plot_spectrum=True, ax=ax, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                         fft_method='Welch', n_windows=3, include_beta_band_in_legend=False)

ax.set_xlim(5, 55)
ax.set_ylim(-0.01, 0.1)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=10, frameon=False)
save_pdf_png(fig, os.path.join(path, 'SNN_spec_' + status + '_plot_' + state_2),
             size=(5, 3))

# %% Parameterscape SNN

        
filename =  'D2_Proto_FSI_N_1000_T_3000_1_pts_1_runs_dt_0-25_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46.pkl'
filename =  'D2_Proto_FSI_N_1000_T_3000_15_pts_1_runs_dt_0-25_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46.pkl'
filename =  'D2_Proto_FSI_N_1000_T_3000_15_pts_1_runs_dt_0-25_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46.pkl' ## equal Gs

examples_ind = {'A' : (3, 5), 'B': (14, 5),
                'C' : (0, 14), 'D': (7, 14),
                'E': (14, 14)}

filename =  'D2_Proto_FSI_N_1000_T_5000_12_pts_1_runs_dt_0-25_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46.pkl' ## FSI-D2 double
examples_ind = {'A' : (3, 4), 'B': (11, 4),
                'C' : (0, 11), 'D': (6, 11),
                'E': (11, 11)}

key_x = ('Proto', 'Proto')
key_y = ('FSI', 'Proto')
name_list = ['D2', 'FSI', 'Proto']
xlabel = r'$G_{Proto-Loop}$'
markerstyle_list = ['o', 'o', 's']
size_list = np.array([4, 2, 1]) * 1000


filename = 'STN_D2_Proto_FSI_N_1000_T_2300_2_pts_3_runs_dt_0-15_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46_STN_15_all_avg.pkl'
examples_ind = {'A' : (2, 3), 'B': (13, 0),
                'C': (4, 8), 'D': (1, 10),
                'E' : (0, 13), 'F': (2, 13),
                'G':(5, 13), 'H': (13, 13), 'L':(2,11)
                }
# filename = 'STN_D2_Proto_FSI_G_DF_&_PS_changing_N_1000_T_5300_10_pts_3_runs_dt_0-2_awake_rest_4_wind_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46_STN_10-12_all_avg.pkl'

# examples_ind = {'A' : (0, 0), 'B': (11, 2),
#                 'C': (0, 4), 'D': (6, 4),
#                 'E' : (0, 9), 'F': (8, 6),
#                 'G':(6, 9), 'H': (13, 9)
#                 }
filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_5_pts_3_runs.pkl'
examples_ind = {'A' :0, 'B': 1,
                'C': 2, 'D': 3,
                'E' : 4
                }
dt = 0.15
key_x = ('Proto', 'STN')
key_y = ('D2', 'FSI')
name_list = ['D2', 'FSI', 'Proto', 'STN']
xlabel = r'$G_{STN-Proto}$'
markerstyle_list = ['o', 'o', 's', 'o']
size_list = np.array([4, 2, 1, 0.5]) * 1000

filepath = os.path.join(path, 'Beta_power', filename)



data  = load_pickle(filepath)
x_list = data['g'][key_x]
y_list = data['g'][key_y]    
n_x = len(x_list)
n_y = len(y_list)

ylabel = r'$G_{FSI-Proto}$'

power_dict = {name: np.zeros((len(x_list), len(y_list))) for name in name_list}
freq_dict = {name: np.zeros((len(x_list), len(y_list))) for name in name_list}
f_peak_sig_dict = {name: np.zeros((len(x_list), len(y_list))) for name in name_list}

p_ind = {'low beta (12-20 Hz)': 0, 'high beta (20-30 Hz)':1, 'low gamma (30-70 Hz)': 2}
f_range = 'low beta (12-20 Hz)'
# f_range = 'high beta (20-30 Hz)'
# f_range = 'low gamma (30-70 Hz)'
param = f_range + ' ' + 'power'
param = 'frequency (Hz)'

# for name in name_list:  ####### one single run (hence the squeeze)
#     power_dict[name] = np.squeeze( data[name, 'base_beta_power'] )[:,:,p_ind[f_range]]
#     freq_dict[name] = np.squeeze( data[name, 'base_freq'] )
    # f_peak_sig_dict[name] =  np.zeros_like(freq_dict[name])

for name in name_list:   ####### averaged multiprocessing results
    power_dict[name] = data[name, 'power_all_runs'] [:,:,p_ind[f_range]]
    freq_dict[name] = data[name, 'peak_freq_all_runs'] 
    f_peak_sig_dict[name] =  data[name, 'peak_significance_all_runs'] 

# for name in name_list:   ####### average sigle multiprocessing file 
#     power_dict[name] = np.average( data[name, 'base_beta_power'] [:,:,:, p_ind[f_range]], axis = 2)
#     freq_dict[name] = np.average( data[name, 'base_freq'], axis = 2)
#     f_peak_sig_dict[name] =  data[name, 'peak_significance_all_runs'] 

# run = 1
# for name in name_list: ####### single multi run file 
#     power_dict[name] = data[name, 'base_beta_power'] [:, :, run , p_ind[f_range]]
#     freq_dict[name] = np.average (data[name, 'base_freq'], axis = 2)
#     f_peak_sig_dict[name] =  np.zeros_like(freq_dict[name])
    
    



# f_list = data['D2', 'f'][0,0,0,:]
# fig_PSD_surf,_ = plot_spec_as_surf(data['g']['Proto','STN'], 
#                   f_list[f_list < 80], 
#                   np.average(
#                       data['Proto', 'pxx'][-1,:,:,:], axis = 1).T[f_list < 80,:],
#                   xlabel = r'$G_{STN \; Loop}$', ylabel = 'Frequency (Hz)', 
#                   zlabel = 'Normalized Power' + r'$(\times 10^{-2})$')
# save_pdf_png(fig, filepath.split('.')[0] + '_' + param + '_PSD_surf', size = (.8 * n_x + 1, .8 * n_y))


# fig = parameterscape(x_list, y_list, name_list, markerstyle_list, freq_dict, freq_dict, f_peak_sig_dict, 
#                     size_list, xlabel, ylabel, label_fontsize = 22, title = param, 
#                     annotate = False, ann_name='Proto',  tick_size = 18, only_significant = False)


# fig_exmp = plot_pop_act_and_PSD_of_example_pts(data, name_list, examples_ind, x_list, y_list, dt, color_dict, Act, plt_duration = 600, run_no = 0)

save_pdf_png(fig_exmp, filepath.split('.')[0] + '_' + param + '_details', size = (10, 15))


fig = highlight_example_pts(fig, examples_ind, x_list, y_list, size_list, highlight_color = 'w')
save_pdf_png(fig, filepath.split('.')[0] + '_' + param, size = (.8 * n_x + 1, .8 * n_y))

# %% only changing STN to Proto weight

filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_5_pts_3_runs.pkl'
examples_ind = {'A' :0, 'B': 1,
                'C': 2, 'D': 3,
                'E' : 4
                }
dt = 0.15
key_x = ('Proto', 'STN')
key_y = ('D2', 'FSI')
name_list = ['D2', 'FSI', 'Proto', 'STN']
xlabel = r'$G_{STN-Proto}$'
markerstyle_list = ['o', 'o', 's', 'o']
size_list = np.array([4, 2, 1, 0.5]) * 1000

filepath = os.path.join(path, 'Beta_power', filename)



data  = load_pickle(filepath)
x_list = data['g'][key_x]
y_list = data['g'][key_y]    
n_x = len(x_list)
n_y = len(y_list)
fig_exmp = plot_pop_act_and_PSD_of_example_pts_1d(data, name_list, examples_ind, x_list, y_list, dt, color_dict, Act, plt_duration = 600, run_no = 0)
save_pdf_png(fig_exmp, filepath.split('.')[0] +  '_details', size = (10, 10))


# f_list = data['D2', 'f'][0,0,:]
# fig_PSD_surf,_ = plot_spec_as_surf(data['g']['Proto','STN'], 
#                   f_list[f_list < 80], 
#                   np.average(
#                       data['Proto', 'pxx'], axis = 1).T[f_list < 80,:],
#                   xlabel = r'$G_{STN \; Loop}$', ylabel = 'Frequency (Hz)', 
#                   zlabel = 'Normalized Power' + r'$(\times 10^{-2})$')
# save_pdf_png(fig_PSD_surf, filepath.split('.')[0] + '_PSD_surf', size = (.8 * n_x + 1, .8 * n_y))


# %% Merge data from multiprocessing


def get_specs_of_mp(ex_filename, path, name = 'FSI'):
    
    filepath = os.path.join(path, 'Beta_power', ex_filename + '0.pkl')
    data  = load_pickle(filepath)
    n_iter_l_1, n_iter_l_2, n_run, len_f_pxx = data[(name, 'pxx')].shape
    n_iter_l_1, n_iter_l_2, n_run, duration = data[(name, 'pop_act')].shape
    
    return n_iter_l_1, n_iter_l_2, duration, n_run, len_f_pxx

def create_df_for_merging_all_mp_results(name_list, n_run, n_phase_bins, len_f_pxx, duration, 
                                         n_iter_each_loop, loop_key_lists):
    
    data = {}
    n_iter_l_1, n_iter_l_2 = n_iter_each_loop
    
    for name in name_list:
        data[(name, 'base_freq')] = np.zeros((n_iter_l_1, n_iter_l_2, n_run))
        data[(name, 'pop_act')] = np.zeros((n_iter_l_1, n_iter_l_2, n_run,  duration))
        data[(name, 'peak_significance')] = np.zeros((n_iter_l_1, n_iter_l_2, n_run), dtype = bool) # stores the value of the PSD at the peak and the mean of the PSD elsewhere
        
        if find_phase:
            data[(name, 'rel_phase_hist')] = np.zeros((n_iter_l_1, n_iter_l_2, n_run, 2, n_phase_bins-1))
            data[(name, 'rel_phase')] = np.zeros((n_iter_l_1, n_iter_l_2, n_run))
    
        data[(name, 'base_beta_power')] = np.zeros((n_iter_l_1, n_iter_l_2, n_run, 3))
        data[(name, 'f')] = np.zeros((n_iter_l_1, n_iter_l_2, n_run, len_f_pxx))
        data[(name, 'pxx')] = np.zeros((n_iter_l_1, n_iter_l_2, n_run, len_f_pxx))
        data[(name, 'peak_significance_all_runs')] = np.zeros((n_iter_l_1, n_iter_l_2))
        data[(name, 'peak_freq_all_runs')] = np.zeros((n_iter_l_1, n_iter_l_2))
        data[(name, 'power_all_runs')] = np.zeros((n_iter_l_1, n_iter_l_2, 3))
        
    data = create_g_in_data(data, loop_key_lists, n_iter_each_loop)

    return data

def create_g_in_data(data, loop_key_lists, n_iter_each_loop):
 
    data['g'] = {}
    n_loops = len(loop_key_lists)
    
    for loop in range(n_loops):
        
        loop_keys = loop_key_lists[loop]
        for key in loop_keys:
            data['g'][key] = np.zeros(n_iter_each_loop[loop])
    return data

def get_the_other_loop_ind(changing_loop_ind):
    if changing_loop_ind == 1:
        return 0
    else:
        return 1
    
def save_g_to_data_all(loop_keys, n_iter, this_iter, data, changing_loop_ind = 0):
    
    for key in loop_keys[changing_loop_ind]:
        data_all['g'][key][this_iter: this_iter + n_iter]  = data['g'][key]
    
    for key in loop_keys[get_the_other_loop_ind(changing_loop_ind)]:
        data_all['g'] [key] = data['g'][key]
        
    return data_all

def merge_mp_data(data_all, ex_filename, path, n_iter_mp, n_mp, loop_keys, find_phase = False, 
                  changing_loop_ind = 0):
    
    for mp in range(n_mp):

        filename = ex_filename + str(mp) + '.pkl'
        print(filename)
        filepath = os.path.join(path, 'Beta_power', filename)
        data  = load_pickle(filepath)
        
        this_iter = n_iter_mp * mp
        
        for name in name_list:
            if changing_loop_ind == 0:
                data_all = fill_data_all_first_loop_mp(this_iter, n_iter_mp, data_all, data, name)
            if changing_loop_ind == 1:
                data_all = fill_data_all_second_loop_mp(this_iter, n_iter_mp, data_all, data, name)

        data_all = save_g_to_data_all(loop_keys, n_iter_mp, this_iter, data, changing_loop_ind = changing_loop_ind )
        
    return data_all

def fill_data_all_first_loop_mp(this_iter, n_iter, data_all, data, name):
    
    if (name, 'peak_significance') in data:
        
        data_all[(name, 'peak_significance')][this_iter: this_iter + n_iter, :, :] =data [(name, 'peak_significance')]
    
    if (name, 'rel_phase_hist') in data:
        
        data_all[(name, 'rel_phase_hist')][this_iter: this_iter + n_iter, :, :, :] =  data[(name, 'rel_phase_hist')]  
        data_all[(name, 'rel_phase')][this_iter: this_iter + n_iter, : ] =  data[(name, 'rel_phase')] 
    
    data_all[(name, 'pop_act')][this_iter: this_iter + n_iter, :, :] = data[(name, 'pop_act')] 
    data_all[(name, 'base_freq')][this_iter: this_iter + n_iter, :] = data[(name, 'base_freq')] 

    data_all[(name, 'base_beta_power')][this_iter: this_iter + n_iter, :, :, :]  = data[(name, 'base_beta_power')]
    data_all[(name, 'f')][this_iter: this_iter + n_iter, :, :, :]  = data[(name, 'f')] 
    data_all[(name, 'pxx')][this_iter: this_iter + n_iter, :, :, :]  = data[(name, 'pxx')] 
    
    return data_all


def fill_data_all_second_loop_mp(this_iter, n_iter, data_all, data, name):
    
    if (name, 'peak_significance') in data:
        
        data_all[(name, 'peak_significance')][:, this_iter: this_iter + n_iter, :] = data [(name, 'peak_significance')]
    
    if (name, 'rel_phase_hist') in data:
        
        data_all[(name, 'rel_phase_hist')][:, this_iter: this_iter + n_iter, :, :] =  data[(name, 'rel_phase_hist')]  
        data_all[(name, 'rel_phase')][:, this_iter: this_iter + n_iter] =  data[(name, 'rel_phase')] 
    

    data_all[(name, 'pop_act')][:, this_iter: this_iter + n_iter, :] = data[(name, 'pop_act')] 
    data_all[(name, 'base_freq')][:, this_iter: this_iter + n_iter] = data[(name, 'base_freq')] 

    data_all[(name, 'base_beta_power')][:, this_iter: this_iter + n_iter, :, :]  = data[(name, 'base_beta_power')]
    data_all[(name, 'f')][:, this_iter: this_iter + n_iter, :, :]  = data[(name, 'f')] 
    data_all[(name, 'pxx')][:, this_iter: this_iter + n_iter, :, :]  = data[(name, 'pxx')] 
    
    return data_all

name1 = 'FSI'  
name2 = 'D2'  
name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]
n_mp = 7
find_phase = False ; n_phase_bins = 180

loop_key_lists = [ [(name2, name1),
                    (name3, name2),
                    (name1, name3)], 
                  [(name4, name3), (name3, name4)]]

ex_filename = 'STN_D2_Proto_FSI_N_1000_T_2300_2_pts_3_runs_dt_0-15_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46_STN_15_'
t_sim = 2300
dt = 0.15
changing_loop_ind = 0
n_iter_l_1, n_iter_l_2, duration, n_run, len_f_pxx = get_specs_of_mp(ex_filename, path, name = 'FSI')

ex_filename = 'STN_D2_Proto_FSI_G_DF_&_PS_changing_N_1000_T_5300_10_pts_3_runs_dt_0-2_awake_rest_4_wind_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46_STN_10-12_'
t_sim = 5300
dt = 0.2
changing_loop_ind = 1
n_iter_l_1, n_iter_l_2, duration, n_run, len_f_pxx = get_specs_of_mp(ex_filename, path, name = 'FSI')

n_iter_each_loop_each_file = [n_iter_l_1, n_iter_l_2 ]
mp_to_multiply = np.ones(2); mp_to_multiply[changing_loop_ind] = n_mp
n_iter_each_loop = (n_iter_each_loop_each_file * mp_to_multiply).astype(int)

data_all = create_df_for_merging_all_mp_results(name_list, n_run, n_phase_bins, len_f_pxx, duration, 
                                                n_iter_each_loop, loop_key_lists)

data_all = merge_mp_data(data_all, ex_filename, path, n_iter_each_loop_each_file[changing_loop_ind], n_mp, loop_key_lists, 
                         find_phase = False,changing_loop_ind  = changing_loop_ind)
pickle_obj(data_all,os.path.join(path, 'Beta_power', ex_filename + 'all.pkl'))
 
# %% average data from multiprocessing

name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]

filename = 'STN_D2_Proto_FSI_N_1000_T_2300_2_pts_3_runs_dt_0-15_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46_STN_15_all.pkl'
filename = 'STN_D2_Proto_FSI_G_DF_&_PS_changing_N_1000_T_5300_10_pts_3_runs_dt_0-2_awake_rest_4_wind_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46_STN_10-12_all.pkl'

filepath = os.path.join(path, 'Beta_power', filename)
data_all  = load_pickle(filepath)
n_run = data_all[(name1, 'f')].shape[2]
n_iter_1 = data_all[(name1, 'f')].shape[0]
n_iter_2 = data_all[(name1, 'f')].shape[1]

def check_for_nans_in_freq_analysis(name_list, data_all, n_iter_1, n_iter_2, n_run):
    for iter_1 in range(n_iter_1):
        for iter_2 in range(n_iter_2):
            for n in range(n_run):
                for name in name_list:
                
                    ind =   np.isnan(data_all[(name, 'f')][iter_1, iter_2, n, : ])
                    # print(name, 'iter 1', iter_1, 'iter 2', iter_2, 'run', n )
                    print( 'number of nans = ', sum(ind))
                    
            
def average_over_runs_and_update_dataframe(filepath, data_all, n_iter_1, n_iter_2, n_run, name_list , save_gamma = True,
                                           low_beta_range = [12,20], high_beta_range = [20, 30], low_gamma_range = [30, 70],
                                           AUC_ratio_thresh = 0.15):
    e = 0
    # for iter_1 in range(e, e + 1):
    #     for iter_2 in range(e, e + 1):
    for iter_1 in range(n_iter_1):
        for iter_2 in range(n_iter_2):

            # fig, ax = plt.subplots()
            for name in name_list:
                f = data_all[(name, 'f')][iter_1, iter_2, 0, : ]
                pxx = np.average(data_all[(name, 'pxx')][iter_1, iter_2, :, : ], axis = 0)

                
                data_all[( name, 'peak_significance_all_runs')][iter_1, iter_2 ] = check_significance_of_PSD_peak(f, pxx,  
                                                                                                                 n_std_thresh = 2, 
                                                                                                                 min_f = 0, max_f = 250,
                                                                                                                 n_pts_above_thresh = 3,
                                                                                                                 if_plot = False, 
                                                                                                                 name = name, 
                                                                                                                 AUC_ratio_thresh = AUC_ratio_thresh)
                
                data_all[(name, 'peak_freq_all_runs')][iter_1, iter_2 ] = f[ np.argmax( pxx ) ]
                low_beta_band_power = beta_bandpower(f, pxx, *low_beta_range)
                high_beta_band_power = beta_bandpower(f, pxx, *high_beta_range)
                low_gamma_band_power = beta_bandpower(f, pxx, * low_gamma_range)
                data_all[(name, 'power_all_runs')][iter_1, iter_2,: ] = low_beta_band_power, high_beta_band_power, low_gamma_band_power
                # ax.plot(f, pxx, c=color_dict[name], label=name + str(data_all[(name, 'peak_freq_all_runs')][iter_1, iter_2 ]), lw=1.5)
                # ax.set_xlim(0,80)
                # ax.legend()
    pickle_obj(data_all, filepath.replace('.pkl', '_avg.pkl'))
    
# check_for_nans_in_freq_analysis(name_list, data_all, n_iter_1, n_iter_2, n_run)
average_over_runs_and_update_dataframe(filepath, data_all, n_iter_1, n_iter_2, n_run, name_list , save_gamma = True)
# %% Synapric weight exploraion (Proto-Proto)  resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_sim = 10000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(1000/dt), int(t_sim/dt)]

n_windows = 9
name1 = 'Proto'
name_list = [name1]
G = {}

# run rest and DD again with 8 pts
state = 'rest' # set
g = -0.015 # rest

# state = 'DD_anesth' # set
# g = -0.007 # 'DD_anesth'

# state = 'awake_rest' # set
# g = -0.005 # 'awake_rest'

# state = 'mvt' # set
# g = -0.007 # 'mvt'

G = {}
plot_start = t_sim - 600
plot_start_raster = plot_start


G = {(name1, name1) :{'mean': g * K[name1, name1] }
      }
G = set_G_dist_specs(G, order_mag_sigma = 1)

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
receiving_pop_list = {(name1, '1'):  [(name1, '1')]
                      }

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 8; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True


# save_figures = True ; save_pkl = True
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
low_f = 50 ; high_f = 70

x = np.array([1])

n = len(x)

G_dict = {(name1, name1): { 'mean' : g * x *  K[name1, name1]}

          }

filename = 'Proto_Proto_N_1000_T_' + str(t_sim) + '_' + str(n) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') +   \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'

# G_dict = {k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['Proto']
phase_ref = 'Proto'
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance=False, K_all = K_all,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase=int(t_sim/4), phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150)

# pickle_obj(data, filepath)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [
        str(round(G[('Proto', 'Proto')][0], 1)) + '_' + str(round(G[('Proto', 'Proto')][-1]*scale, 1))]

    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]
    filename = (names[0] + '_G(PP)= ' + gs[0] +
                '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method +
                '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[names[0]])
                + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 6), (5, 5), (10, 6), (4, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

# %% Synapric weight exploraion (STN-Proto)  resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_sim = 10000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(1000/dt), int(t_sim/dt)]

n_windows = 9
name2 = 'Proto'
name1 = 'STN'
name_list = [name1, name2]
G = {}




state = 'rest' # set
g = -0.029 # rest

# state = 'DD_anesth' # set
# g = -0.01 # 'DD_anesth'

# state = 'awake_rest' # set
# g = -0.014 # 'awake_rest'

# state = 'mvt' # set
# g = -0.015 # 'mvt'

G = {}
plot_start = t_sim - 600
plot_start_raster = plot_start


G = {(name2, name1) :{'mean': -g * K[name2, name1] },
     (name1, name2) :{'mean': g * K[name1, name2]  }}
G = set_G_dist_specs(G, order_mag_sigma = 1)

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'):  [(name1, '1')],
                      }

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 8; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True


# save_figures = True ; save_pkl = True
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
low_f = 30 ; high_f = 60

x = np.array([1])

n = len(x)

G_dict = {(name2, name1): { 'mean' : -g * x *  K[name2, name1]},
          (name1, name2): { 'mean' : g * x *  K[name1, name2]}
          }

filename = 'STN_Proto_N_1000_T_' + str(t_sim) + '_' + str(n) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') +  \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'

# G_dict = {k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['Proto', 'STN']
phase_ref = 'Proto'
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance=False,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase=int(t_sim/4), phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150, save_pop_act = True)



def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [
        str(round(G[('Proto', 'Proto')][0], 1)) + '_' + str(round(G[('Proto', 'Proto')][-1]*scale, 1))]

    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]
    filename = (names[0] + '_G(PP)= ' + gs[0] +
                '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method +
                '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[names[0]])
                + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 6), (5, 5), (10, 6), (4, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# %% synaptic weight exploration (FSI-D2-Proto) resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 10000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(1000/dt), int(t_sim/dt)]
n_windows = 9
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'

state = 'rest' # set
g = -0.045 # anesthetized all equal G

# state = 'DD_anesth' # set
# g = -0.005 # anesthetized all equal G
# g = -0.0048 # anesthetized G(DF) = g x 1.6

# state = 'awake_rest' # set
# g = -0.0055  # awake all equal G
# g = -0.00472  # awake G(DF) = g x 1.6

# state = 'mvt' # set
# g = -0.0035 # 'mvt'  all equal G
# g = -0.0032 # 'mvt' G(DF) = g x 1.6

name_list = [name1, name2, name3]

G = {(name2, name1) :{'mean': g * K[name2, name1] * 2},
      (name1, name3) :{'mean': g * K[name1, name3]},
      (name3, name2) :{'mean': g * K[name3, name2]}
      }
G = set_G_dist_specs(G, order_mag_sigma = 1)

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}
# (name3, '1'): [(name2,'1'), (name3, '1')]} # with GP-GP


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


# n_run = 1; plot_firing = True; plot_spectrum = True; plot_raster = True; plot_phase = True; low_pass_filter = False; save_pkl = False; save_figures = True; save_pxx = False
n_run = 8; plot_firing = False; plot_spectrum= False; plot_raster = False; plot_phase = False; low_pass_filter= False ;save_pkl = True ; save_figures = False; save_pxx = True
round_dec = 1
include_std = False
plot_start = 1500  # int(t_sim/2)
plot_raster_start = 1500  # int(t_sim/2)
n_neuron = 30
legend_loc = 'center right'
low_f = 5; high_f = 20
# x = np.flip(np.geomspace(-40, -0.1, n))

x = np.array([1])

# x = np.linspace(.1, 1.5, 20)

n = len(x)


# 0.0045, 2.5, 0.5, 0.5 ,x = 1.5
# G_dict = {(name2, name1) : np.array([g * 2]* (n)) ,
#           (name3, name2): g * x ,
#           (name1, name3) :  np.array( [g * 0.5]* (n)) }
# filename = 'D2_Proto_FSI_N_1000_T_5000_G_D2_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) :  g * 2 * x ,
#           (name3, name2): np.array([g ]* (n)) ,
#           (name1, name3) :  np.array( [g * 0.5]* (n)) }
# filename = 'D2_Proto_FSI_N_1000_T_5000_G_FSI_D2_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array([g*2]* (n)) ,
#           (name3, name2): np.array( [g]* (n)) ,
#           (name1, name3) : g * 0.5 * x  }
# filename = 'D2_Proto_FSI_N_1000_T_5000_G_Proto_FSI_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = {(name2, name1): {'mean' : g * 2 * x * K[(name2, name1)]},
          (name3, name2): {'mean' : g * x * K[name3, name2]},
          (name1, name3): {'mean' :g * x * K[(name1, name3)]}}


filename = 'D2_Proto_FSI_N_1000_T_' + str(t_sim) + '_' + str(n) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') + \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'
       


fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
phase_ref = 'Proto'
nuc_order = ['Proto', 'D2', 'FSI']
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows = n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance=False,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase=duration_base[0], phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, state, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '--' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale, 3))]
    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2] + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] +
                '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[state][names[0]]) + '_' + str(
        noise_variance[state][names[1]]) + '_' + str(noise_variance[state][names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, state, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, state, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 13), (5, 10), (6, 12), (6, 12)]
s = [(8, 4), (5, 5), (6, 3), (4, len(name_list))]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, state, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()


# %% synaptic tau explofignameration (FSI-D2-Proto) resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 4000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int( 300/dt), int(t_sim/dt)]
n_windows = 3
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name_list = [name1, name2, name3]


state = 'rest' # set
g = -0.01075 # G(DF) = g x 1.6 homogeneous
g = -0.015 # G(DF) = g x 1.6 heterogenous

# state = 'DD_anesth' # set
# g = -0.008 # anesthetized G(DF) = g x 1.6 homogeneous
# g = -0.008 # anesthetized G(DF) = g x 1.6 heterogenous

# state = 'awake_rest' # set
# g = -0.007  # awake G(DF) = g x 1.6 homogeneous
# g = -0.0075  # awake G(DF) = g x 1.6 heterogenous

# state = 'mvt' # set
# g = -0.007 # 'mvt' G(DF) = g x 1.6 homogeneous
# g = -0.007 # 'mvt' G(DF) = g x 1.6 heterogenous


G = {}

# G[(name2, name1)], G[(name3, name2)], G[(name1, name3)] = g * 1.6 , g, g
# G = {k: v * K[k] for k, v in G.items()}
np.linspace(5, 20, num = 8, endpoint = True)
G = {(name2, name1) :{'mean': g * K[name2, name1] * 1.6},
      (name1, name3) :{'mean': g * K[name1, name3]},
      (name3, name2) :{'mean': g * K[name3, name2]}
      }
G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2)

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


# n_run = 1; plot_firing = True; plot_spectrum = True; plot_raster = True; plot_phase = True; low_pass_filter = False; save_pkl = False; save_figures = False; save_pxx = False
n_run = 8; plot_firing = False; plot_spectrum= False; plot_raster = False; plot_phase = False; low_pass_filter= False ;save_pkl = True ; save_figures = False; save_pxx = True
round_dec = 1
include_std = False
plot_start = t_sim - 1000
plot_raster_start = plot_start
n_neuron = 30
legend_loc = 'center right'
low_f = 5; high_f = 20

x = np.linspace(5, 20, num = 8, endpoint = True)

tau_dict = {('FSI', 'Proto'):{ 'mean': x, 'sd': x/5, 
                              'truncmin': x - 2 * (x / 5), 
                              'truncmax': x + 2 * (x / 5)}}

n = len(x)
filename = 'D2_Proto_FSI_tau_sweep_N_1000_T_' + str(t_sim) + '_' + str(n) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') +'_' + state + '_' + noise_method +  \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'
       


fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
phase_ref = 'Proto'
nuc_order = ['Proto', 'D2', 'FSI']
figs, title, data = synaptic_tau_exploration_SNN(path, tau, nuclei_dict, filepath, duration_base, G, tau_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows= n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance=False,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase=duration_base[0], phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150, save_pop_act = True, normalize_spec = False)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, state, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '--' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale, 3))]
    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2] + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] +
                '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[state][names[0]]) + '_' + str(
        noise_variance[state][names[1]]) + '_' + str(noise_variance[state][names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, state, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, state, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 13), (5, 10), (6, 12), (6, 12)]
s = [(8, 4), (5, 5), (6, 3), (4, len(name_list))]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, state, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

# %% tau exploration average

low_beta_range = [12,20]; high_beta_range = [20, 30]; low_gamma_range = [30, 70]
filename = 'D2_Proto_FSI_tau_sweep_N_1000_T_3300_15_pts_3_runs_dt_0-15_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46.pkl'
filepath = os.path.join(path, 'Beta_power', filename)
data  = load_pickle(filepath)
n_iter =  data_all[(name, 'f')].shape[0]

data_avg = {}
name_list = ['D2', 'FSI', 'Proto']
for name in name_list:
    data_avg[(name, 'peak_significance_all_runs')] = np.zeros((n_iter))
    data_avg[(name, 'peak_freq_all_runs')] = np.zeros((n_iter))
    data_avg[(name, 'power_all_runs')] = np.zeros((n_iter, 3))
        
for m in range(n_iter):

    for name in name_list:
        f = data[(name, 'f')][ m ,  0, : ]
        pxx = np.average(data[(name, 'pxx')][m, :, : ], axis = 0)

        
        # data_avg[( name, 'peak_significance_all_runs')][m] = check_significance_of_PSD_peak(f, pxx,  
#                                                                                                  n_std_thresh = 2, 
#                                                                                                  min_f = 0, max_f = 250,
#                                                                                                  n_pts_above_thresh = 3,
#                                                                                                  if_plot = False, 
#                                                                                                  name = name, 
#                                                                                                  AUC_ratio_thresh = 0.2)

        data_avg[(name, 'peak_freq_all_runs')][m ] = f[ np.argmax( pxx ) ]
        low_beta_band_power = beta_bandpower(f, pxx, *low_beta_range)
        high_beta_band_power = beta_bandpower(f, pxx, *high_beta_range)
        low_gamma_band_power = beta_bandpower(f, pxx, * low_gamma_range)
        data_avg[(name, 'power_all_runs')][m, : ] = low_beta_band_power, high_beta_band_power, low_gamma_band_power
        # ax.plot(f, pxx, c=color_dict[name], label=name + str(data_all[(name, 'peak_freq_all_runs')][iter_1, iter_2 ]), lw=1.5)
        # ax.set_xlim(0,80)
        # ax.legend()
    pickle_obj(data_avg, filepath.replace('.pkl', '_avg.pkl'))
    
    
fig, ax = plt.subplots()

for name in name_list:  
    ax.plot(data['tau'][('FSI', 'Proto')][:-1], data[(name, 'base_freq')][:-1,0], '-o', c = color_dict[name], label = name)

    # ax.plot(data['tau'][('FSI', 'Proto')][:-1], data_avg[(name, 'peak_freq_all_runs')], '-o', c = color_dict[name], label = name)
ax.legend()
# %% FSI and STN loop freq for uncertain tau and Ts
filename = 'D2_Proto_FSI_tau_sweep_N_1000_T_3300_15_pts_3_runs_dt_0-15_awake_rest_Ornstein-Uhlenbeck_A_FSI_15-2_D2_1-1_Proto_46_3.pkl'
filename = 'STN_Proto_T_sweep_N_1000_T_3300_10_pts_3_runs_dt_0-15_rest_Ornstein-Uhlenbeck_A_STN_7_Proto_39-84.pkl'
filepath = os.path.join(path, 'Beta_power', filename)
data  = load_pickle(filepath)
fig, ax = plt.subplots()
name = 'FSI'
name_list = ['FSI', 'D2', 'Proto']
param = 'tau'
key = ('FSI', 'Proto')
name = 'STN'
name_list = ['STN']#, 'Proto']
param = 'T'
key = ('STN', 'Proto')
x = np.linspace(5, 15, num = 15)
n_iter =  data[(name, 'f')].shape[0]
n_run =  data[(name, 'f')].shape[1]
for name in name_list:  
    
    # for i in range(n_run):
    #     # print(data[param][('FSI', 'Proto')].shape, data[(name, 'base_freq')][:,i].shape)
    #     # ax.scatter(data[param][('FSI', 'Proto')], data[(name, 'base_freq')][:,i],  c = color_dict[name])

    ax.boxplot(data[(name, 'base_freq')].T , labels = np.round(data[param][key],1),  patch_artist=True,
                whis=(0, 100), zorder=0)
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

# for label in ax.xaxis.get_ticklabels()[::2]:
#     label.set_visible(True)
# plt.xticks(np.arange(5, 15, 5))
# ax.locator_params(nbins=5)
# fig = set_x_ticks(fig, [5, 10, 15])

ax.set_ylabel('Frequency of ' + name + ' Loop (Hz)', fontsize = 15)
ax.set_xlabel(r'$\tau_{Proto-FSI}, \; \tau_{FSI-D2}$' , fontsize = 15)

save_pdf_png(fig, os.path.join(path, name + '_tau_exploration'  ),
              size = (6,5))

# %% synaptic T exploration (FSI-D2-Proto) resetting inti dists and setting ext input collectively


plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.15
t_sim = 3300
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(300/dt), int(t_sim/dt)]

name1 = 'STN'
name2 = 'Proto'
name_list = [name1, name2]



state = 'rest' # set
g = -0.013 # rest

# state = 'DD_anesth' # set
# g = -0.008 # 'DD_anesth'

# state = 'awake_rest' # set
# g = -0.01 # 'awake_rest'

# state = 'mvt' # set
# g = -0.008 # 'mvt'

# g = 0
G = {}

G[(name1, name2)], G[(name2, name1)] = g, -g
G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name1, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


# n_run = 1; plot_firing = True; plot_spectrum = True; plot_raster = True; plot_phase = True; low_pass_filter = False; save_pkl = False; save_figures = False; save_pxx = False
n_run = 3; plot_firing = False; plot_spectrum= False; plot_raster = False; plot_phase = False; low_pass_filter= False ;save_pkl = True ; save_figures = False; save_pxx = True
round_dec = 1
include_std = False
plot_start = t_sim - 600
plot_raster_start = plot_start
n_neuron = 30
legend_loc = 'center right'
low_f = 5; high_f = 20


T_dict = {('Proto', 'STN'):   np.linspace(2.8, 4.75, num = 10),
            ('STN', 'Proto'):   np.linspace(1.3, 4, num = 10)}

n = len( T_dict[('Proto', 'STN')] ) 
filename = 'STN_Proto_T_sweep_N_1000_T_' + str(t_sim) + '_' + str(n) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') +'_' + state + '_' + noise_method +  \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'
       


fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
phase_ref = 'Proto'
nuc_order = ['Proto', 'STN']
figs, title, data = synaptic_T_exploration_SNN(path, nuclei_dict, filepath, duration_base, G, T_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=3, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance=False,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase=duration_base[0], phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150, save_pop_act = True, normalize_spec = False)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, state, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '--' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale, 3))]
    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2] + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] +
                '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[state][names[0]]) + '_' + str(
        noise_variance[state][names[1]]) + '_' + str(noise_variance[state][names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, state, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, state, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 13), (5, 10), (6, 12), (6, 12)]
s = [(8, 4), (5, 5), (6, 3), (4, len(name_list))]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, state, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# %% synaptic weight exploration (Arky-D2-Proto) resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 10000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(1000/dt), int(t_sim/dt)]

name1 = 'Arky'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'

name_list = [name1, name2, name3]
n_windows = 9

state = 'rest' # set 
g = -0.03 # rest G(DA) = g x 1.6 hetero

# state = 'DD_anesth' # set 
# g = -0.0075 # 'DD_anesth'  G(DA) = g x 1.6 hetero

# state = 'awake_rest' # set
# g = -0.0117 # 'awake_rest G(DA) = g x 1.6 hetero

# state = 'mvt' # set
# g = -0.01 # 'mvt' G(DA) = g x 1.6 hetero

G = {}

G = {(name2, name1) :{'mean': g * K[name2, name1] * 5.},
      (name1, name3) :{'mean': g * K[name1, name3] },
      (name3, name2) :{'mean': g * K[name3, name2]}
      }
G = set_G_dist_specs(G, order_mag_sigma = 1)



poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

# (name3, '1'): [(name2,'1'), (name3, '1')]} # with GP-GP


pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)

# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True; plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = True
n_run = 8; plot_firing = False; plot_spectrum= False; plot_raster = False; plot_phase = False; low_pass_filter= False ;save_pkl = True ; save_figures = False; save_pxx = True


round_dec = 1
include_std = False
plot_start = 1500  # int(t_sim/2)
plot_raster_start = 1500  # int(t_sim/2)
n_neuron = 30
legend_loc = 'center right'
low_f = 5; high_f = 20
# x = np.flip(np.geomspace(-40, -0.1, n))

x = np.array([1])

# x = np.linspace(.1, 1.5, 20)

n = len(x)


# 0.0045, 2.5, 0.5, 0.5 ,x = 1.5
# G_dict = {(name2, name1) : np.array([g * 2]* (n)) ,
#           (name3, name2): g * x ,
#           (name1, name3) :  np.array( [g * 0.5]* (n)) }
# filename = 'D2_Proto_Arky_N_1000_T_5000_G_D2_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) :  g * 2 * x ,
#           (name3, name2): np.array([g ]* (n)) ,
#           (name1, name3) :  np.array( [g * 0.5]* (n)) }
# filename = 'D2_Proto_Arky_N_1000_T_5000_G_Arky_D2_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array([g*2]* (n)) ,
#           (name3, name2): np.array( [g]* (n)) ,
#           (name1, name3) : g * 0.5 * x  }
# filename = 'D2_Proto_Arky_N_1000_T_5000_G_Proto_Arky_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = {(name2, name1): {'mean' : g * 5 * x * K[(name2, name1)]},
          (name3, name2): {'mean' : g * x * K[name3, name2]},
          (name1, name3): {'mean' :g * x * K[(name1, name3)]}}

filename = 'D2_Proto_Arky_N_1000_T_' + str(t_sim) + '_' + str(n) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') +  \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'
     
# G_dict = {k: v * K[k] for k, v in G_dict.items()}


fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['D2', 'Proto', 'Arky']
phase_ref = 'Proto'
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance=False,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase=duration_base[0], phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150)

def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'Arky')][0], 3)) + '--' + str(round(G[('D2', 'Arky')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '--' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('Arky', 'Proto')][0], 3)) + '--' + str(round(G[('Arky', 'Proto')][-1]*scale, 3))]
    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2] + '_G(AD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PA)= ' + gs[2] +
                '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[names[0]]) + '_' + str(
        noise_variance[names[1]]) + '_' + str(noise_variance[names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)
    


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 13), (5, 10), (6, 12), (6, 12)]
s = [(8, 4), (5, 5), (6, 3), (4, len(name_list))]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()


# %% Synapric weight exploraion STN-Proto + FSI-D2-Proto individual neurons

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]

g_ext = 0.01
g = 0
G = {}

G[(name2, name1)], G[(name3, name2)], G[(name1, name3)
                                        ], G[(name3, name4)], G[(name4, name3)] = g, g, g, -g, g

poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}

receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name4, '1')],
                      (name4, '1'): [(name3, '1')]}
# (name3, '1'): [(name2,'1'), (name3, '1')]} # with GP-GP


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'


keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3, name4: nuc4}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)


filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl',
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl',
             'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl'}

set_init_all_nuclei(nuclei_dict, filepaths=filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt,
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise=False)

n = 3
n_run = 1
plot_firing = True
plot_spectrum = True
plot_raster = True
low_pass_filter = False
save_pkl = False
# plot_firing = False; plot_spectrum= False; plot_raster = False; save_pkl = True ; save_figures = False
save_figures = True
save_pkl = True
round_dec = 1
include_std = False
plot_start = 500
plot_raster_start = 0
n_neuron = 25
legend_loc = 'center right'
# x = np.flip(np.geomspace(-40, -0.1, n))

# x = np.linspace(5, 1.1, n)
x = np.array([4, 12, 20])
g = -0.001  # start

# G_dict = {(name2, name1) : np.array([g]* (n)) ,
#           (name3, name2): g * x ,
#           (name1, name3) :  np.array( [g]* (n)),
# 			 (name3, name4) :  np.array( [-g]* (n)) ,
# 			  (name4, name3) :  np.array( [g]* (n))  }
# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_D2_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) :  g  * x ,
#           (name3, name2): np.array([g]* (n)) ,
#           (name1, name3) :  np.array( [g]* (n)),
# 			 (name3, name4) :  np.array( [-g]* (n)) ,
# 			  (name4, name3) :  np.array( [g]* (n))   }
# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_FSI_D2_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array([g]* (n)) ,
#           (name3, name2): np.array( [g]* (n)) ,
#           (name1, name3) : g * x  ,
# 		  (name3, name4) :  np.array( [-g]* (n)) ,
# 		  (name4, name3) :  np.array( [g]* (n))  }
# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_Proto_FSI_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = {(name2, name1): np.array([g*1.5] * (n)),
          (name3, name2): np.array([g*1.5] * (n)),
          (name1, name3): np.array([g*1.5] * (n)),
          (name3, name4): -g * x,
          (name4, name3): np.array([g*1.5] * (n))}
filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_and_Proto_STN_changing_' + \
    str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)

# figs, title, data = synaptic_weight_exploration_SNN_all_changing(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict,
#                                                     noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter, legend_loc = legend_loc,
#                                                     lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = [-10,70], n_run = n_run,  plot_start_raster= plot_raster_start,
#                                                     plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
#                                                     find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=False, save_pkl = save_pkl)

# Note: "synaptic_weight_exploration_SNN" when signal is all plateau it has problem saving empty f array to a designated 200 array.
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=[-10, 70], n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=3, include_beta_band_in_legend=True, save_pkl=save_pkl)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '--' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale, 3))]
    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2] + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] +
                '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[names[0]]) + '_' + str(
        noise_variance[names[1]]) + '_' + str(noise_variance[names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(15, 15), (5, 15), (6, 11)]
s = [(8, 13), (5, 10), (6, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*3, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# %% Synapric weight exploraion STN-Proto + FSI-D2-Proto resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.2
t_sim = 5300
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(300/dt), int(t_sim/dt)]
n_windows = 4
name1 = 'FSI'  
name2 = 'D2'  
name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]
state = 'awake_rest'
g = 0
G = {}

G[(name2, name1)], G[(name3, name2)], G[(name1, name3)
                                        ], G[(name3, name4)], G[(name4, name3)] = g, g, g, -g, g

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name4, '1')],
                      (name4, '1'): [(name3, '1')]}
# (name3, '1'): [(name2,'1'), (name3, '1')]} # with GP-GP

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init,
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}


n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)



# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = False; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 2; plot_firing = False; plot_spectrum = False; plot_raster = False; plot_phase = False; low_pass_filter = False; save_pkl = True; save_figures = False; save_pxx = True

include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
round_dec = 1


# coef = 2
# x = np.array([ 1/4 , 4.5/4 , 6/4, 7./4])
# g = -0.005  # start
# n = len(x)

# x0 = np.array([1.5, 1.5, 1.5, 1.5])
# x1 = np.array([1.5, 1.5, 1.5, 1.5])
# x2 = np.array([1.5, 1.5, 1.3, 1])
# x3 = np.array([0.25, 3, 3.5, 4])
# x4 = np.array([0.5, 1, 3, 3])

# x0 = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
# x1 = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
# x2 = np.array([1.5, 1.5, 1.4, 1.45, 1.3, 1.2, 1.1, 1])
# x3 = np.array([0.25, 3, 3.1, 3.2, 3.5, 3.6, 3.7, 4])
# x4 = np.array([0.5, 1, 1.8, 2.6, 3, 3, 3, 3])
# g = -0.0035  # start
# n = len(x0)

# g = -.0045
# coef = 2
# for k,v in data['g'].items():
#     print(k)
#     print(v/K[k])

# x = np.array([0, 1.7, 3.8])

# x = np.linspace(0, 3.8, 15)

# n = len(x)
# 13, 14.3, 28, 30
# m = 2
# x0 =np.array( [x0[m]]); x1 = np.array([x1[m]]) ; x2 = np.array([x2[m]]); x3 = np.array([x3[m]]); x4 = np.array([x4[m]]);


# x0, x1 = np.ones(n) * 1.5
# G_dict = {(name2, name1) : np.array([g]* (n)) ,
#           (name3, name2) : g * x ,
#           (name1, name3) :  np.array( [g]* (n)),
# 		  (name3, name4) :  np.array( [-g * coef]* (n)) ,
#   		  (name4, name3) :  np.array( [g * coef]* (n))  }
# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_D2_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) :  g  * x ,
#           (name3, name2) : np.array([g]* (n)) ,
#           (name1, name3) :  np.array( [g]* (n)),
#   		  (name3, name4) :  np.array( [-g * coef]* (n)) ,
#   		  (name4, name3) :  np.array( [g * coef]* (n))   }
# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_FSI_D2_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array([g]* (n)) ,
#           (name3, name2) : np.array( [g]* (n)) ,
#           (name1, name3) : g * x  ,
# 		  (name3, name4) :  np.array( [-g * coef]* (n)) ,
# 		  (name4, name3) :  np.array( [g * coef]* (n))  }
# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_Proto_FSI_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array([g]* (n)) ,
#           (name3, name2) : np.array( [g]* (n)) ,
#           (name1, name3) : np.array( [g]* (n)) ,
# 		  (name3, name4) :  np.array( [-g * coef]* (n)) ,
# 		  (name4, name3) :  g * coef * x    }
# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_Proto_STN_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1): np.array([g] * (n)),
#           (name3, name2): np.array([g] * (n)) * 1.5,
#           (name1, name3): np.array([g] * (n)),
#           (name3, name4): -g * x,
#           (name4, name3):   np.array([g * coef] * (n))}



n = 5
g_STN_list = - np.linspace(0.5, 2.5, n, endpoint=True)
g_FSI_list = - np.array([2.2] * (n))



G_dict = {(name2, name1): g_FSI_list ,
          (name3, name2): g_FSI_list,
          (name1, name3): g_FSI_list,
          (name3, name4): -g_STN_list,
          (name4, name3): g_STN_list}
filename = 'D2_Proto_FSI_STN_N_1000_T_' + str(t_sim) + '_G_STN_Proto_changing_' + \
    str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : g *  x0,
#           (name3, name2) : g * x1,
#           (name1, name3) : g * x2,
# 		  (name3, name4) : -g  * x3,
# 		  (name4, name3) : g * x4}

# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_all_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['D2', 'STN', 'Proto', 'FSI']
phase_ref = 'D2'
low_f = 8; high_f = 60
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                        noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                        lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                        plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                        find_beta_band_power=True, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=False, save_pkl=save_pkl,
                                                        reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                        state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                        receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                        use_saved_FR_ext=True, check_peak_significance= False, 
                                                        find_phase=False, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                        n_phase_bins=180, start_phase= duration_base[0], phase_ref=phase_ref, save_pxx=save_pxx,
                                                        plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True, normalize_spec = False,
                                                        nuc_order=nuc_order, len_f_pxx=200, min_f = 100, max_f = 300, AUC_ratio_thresh = .65, 
                                                        save_pop_act= True)

# pickle_obj(data, filepath)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '_' + str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '_' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '_' +
          str(round(G[('FSI', 'Proto')][-1]*scale, 3)),
          str(round(G[('Proto', 'STN')][0], 3)) + '_' + str(round(G[('Proto', 'STN')][-1]*scale, 3))]

    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2] + names[3] + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] + '_G(SP)= ' + gs[3] +
                '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method +
                '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[names[0]]) + '_' + str(
        noise_variance[names[1]]) + '_' + str(noise_variance[names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 13), (5, 10), (6, 12), (6, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

# %% Synapric weight exploraion 2d FSI-D2-Proto + Proto-Proto resetting inti dists and setting ext input collectively


plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 5000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(1000/dt), int(t_sim/dt)]

name1 = 'FSI'  
name2 = 'D2'  
name3 = 'Proto'
name_list = [name1, name2, name3]

state = 'awake_rest'

g = 0
G = {}

(G[(name2, name1)], G[(name3, name2)], 
 G[(name1, name3)], G[(name3, name3)]) = g, g, g, g

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name3, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)

# n_run = 1;plot_firing = True;plot_spectrum = True; plot_raster = True; plot_phase = True; low_pass_filter = False; save_pkl = False; save_figures = True; save_pxx = False
n_run = 1; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True
# save_pkl = True ; save_pxx = True
round_dec = 1
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
low_f = 8; high_f = 40

g_FSI_list = - np.linspace(0.15, 2, 12, endpoint=True)
g_GPe_list = - np.linspace(0.15, 1.7, 12, endpoint=True)


G_dict = {(name2, name1): g_FSI_list * 2,
          (name3, name2): g_FSI_list,
          (name1, name3): g_FSI_list,
          (name3, name3): g_GPe_list}

loop_key_lists = [ [(name2, name1),
                    (name3, name2),
                    (name1, name3)], [(name3, name3)]]

filename = 'D2_Proto_FSI_N_1000_T_' + str(t_sim) + '_' + str(len(g_FSI_list)) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') + '_' + state + '_' + noise_method +  \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'


# G_dict = {k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['D2', 'Proto', 'FSI']
phase_ref = 'D2'
figs, title, data = synaptic_weight_exploration_SNN_2d(loop_key_lists, path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=3, include_beta_band_in_legend=False, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance= True,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase= duration_base[0], phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150)


# pickle_obj(data, filepath)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '_' + 
          str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '_' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '_' +
          str(round(G[('FSI', 'Proto')][-1]*scale, 3)),
          str(round(G[('Proto', 'Proto')][0], 3)) + '_' + 
          str(round(G[('Proto', 'Proto')][-1]*scale, 3))]

    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2]  + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] + '_G(PP)= ' + gs[3] +
                '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method +
                '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[state][names[0]]) + '_' + str(
        noise_variance[state][names[1]]) + '_' + str(noise_variance[state][names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        # figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
        #                 orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 13), (5, 10), (6, 12), (6, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*4, s=s)

set_y_lim_all_axis(figs[1], (0,10))
set_x_lim_all_axis(figs[1], (0,70))

remove_legend_all_axis(figs[0])
remove_title_all_axis(figs[0])
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()


# %% Synapric weight exploraion 2d FSI-D2-Proto + STN-Proto resetting inti dists and setting ext input collectively


plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(1000/dt), int(t_sim/dt)]

name1 = 'FSI'  
name2 = 'D2'  
name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]
state = 'awake_rest'
g = 0
G = {}

G[(name2, name1)], G[(name3, name2)], G[(name1, name3)], G[(name4, name3)], G[(name3, name4)] = 0,0,0,0,0
# G = {k: v * K[k] for k, v in G.items()}

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}
    
receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name4, '1')],
                      (name4, '1'): [(name3, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)

# n_run = 1;plot_firing = True;plot_spectrum = True; plot_raster = True; plot_phase = True; low_pass_filter = False; save_pkl = False; save_figures = True; save_pxx = False1
n_run = 1; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True
# save_pkl = True ; save_pxx = True
round_dec = 1
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
low_f = 8; high_f = 40

g_FSI_list = - np.linspace(0.15, 2., 2, endpoint=True)
g_STN_list = - np.linspace(0.15, 2.2, 4, endpoint=True)


G_dict = {(name2, name1): g_FSI_list ,
          (name3, name2): g_FSI_list,
          (name1, name3): g_FSI_list,
          (name4, name3): g_STN_list,
          (name3, name4): -g_STN_list}

loop_key_lists = [ [(name2, name1),
                    (name3, name2),
                    (name1, name3)], [(name4, name3), (name3, name4)]]
filename = 'STN_D2_Proto_FSI_N_1000_T_' + str(t_sim) + '_' + str(len(g_FSI_list)) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') + '_' + state + '_' + noise_method +  \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'


# G_dict = {k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['D2', 'Proto', 'FSI', 'STN']
phase_ref = 'D2'
figs, title, data = synaptic_weight_exploration_SNN_2d(loop_key_lists, path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=3, include_beta_band_in_legend=False, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance= True,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase= duration_base[0], phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=200, min_f = 100, max_f = 300, AUC_ratio_thresh = .65)


# pickle_obj(data, filepath)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '_' + 
          str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '_' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '_' +
          str(round(G[('FSI', 'Proto')][-1]*scale, 3)),
          str(round(G[('Proto', 'Proto')][0], 3)) + '_' + 
          str(round(G[('Proto', 'Proto')][-1]*scale, 3))]

    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2]  + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] + '_G(PP)= ' + gs[3] +
                '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method +
                '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[state][names[0]]) + '_' + str(
        noise_variance[state][names[1]]) + '_' + str(noise_variance[state][names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        # figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
        #                 orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 13), (5, 10), (6, 12), (6, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*4, s=s)

set_y_lim_all_axis(figs[1], (0,10))
set_x_lim_all_axis(figs[1], (0,70))

remove_legend_all_axis(figs[0])
remove_title_all_axis(figs[0])
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

# %% Synapric weight exploraion STN-Proto + Arky resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_sim = 4000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(300/dt), int(t_sim/dt)]

n_windows = 3
name1 = 'STN'
name2 = 'Proto'
name3 = 'Arky'
name_list = [name1, name2, name3]
G = {}



state = 'rest' # set
g = -0.015 # rest

# state = 'DD_anesth' # set
# g = -0.01 # 'DD_anesth'

# state = 'awake_rest' # set
# g = -0.014 # 'awake_rest'

# state = 'mvt' # set
# g = -0.015 # 'mvt'

G = {(name1, name2) :{'mean': g * K[name1, name2] },
      (name2, name1) :{'mean': -g * K[name2, name1]},
      (name3, name2) :{'mean': g * K[name3, name2]}
      }

G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2)

poisson_prop = {name: 
                {'n': 10000, 'firing': 0.0475, 'tau': {
                'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 
                'g': 0.01} 
                for name in name_list}

    
receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path_lacie,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective',  save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 8; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True


# save_figures = True ; save_pkl = True
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
low_f = 30 ; high_f = 60

x = np.array([1])

n = len(x)

G_dict = {(name2, name1): { 'mean' : -g * x *  K[name2, name1]},
          (name1, name2): { 'mean' : g * x *  K[name1, name2]},
          (name3, name2) :{'mean': g * x * K[name3, name2] }
          }

filename = 'STN_Proto_Arky_N_1000_T_' + str(t_sim) + '_' + str(n) + '_pts_' + str(
    n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') + '_' + state + '_' + noise_method +  \
     '_A_' + get_str_of_nuclei_FR(nuclei_dict, name_list) + '.pkl'

# G_dict = {k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['Proto', 'STN']
phase_ref = 'Proto'
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance=False,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=180, start_phase=int(t_sim/4), phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150)




def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('Arky', 'Proto')][0], 3)) + '_' + str(round(G[('Arky', 'Proto')][-1]*scale, 3)),
          str(round(G[('STN', 'Proto')][0], 3)) + '_' +
          str(round(G[('STN', 'Proto')][-1]*scale, 3)),
          str(round(G[('Proto', 'STN')][0], 3)) + '_' + str(round(G[('Proto', 'STN')][-1]*scale, 3))]

    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2])

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 6), (5, 5), (10, 6), (4, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# %% Synapric weight exploraion STN-GPe-GPe + Arky resetting inti dists and setting ext input collectively
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 5000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'Arky'  # projecting
name2 = 'Proto'
name3 = 'STN'
name_list = [name1, name2, name3]
state = 'rest'
g_ext = 0.01
g = 0
G = {}

G[(name1, name2)], G[(name3, name2)], G[(
    name2, name3)], G[(name2, name2)] = g, g, -g, g*.2

poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}

receiving_pop_list = {(name1, '1'):  [(name2, '1')],
                      (name2, '1'): [(name3, '1'), (name3, '1')],
                      (name3, '1'): [(name2, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True
keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective', save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 20
plot_firing = False
plot_spectrum = False
plot_raster = False
plot_phase = False
low_pass_filter = False
save_pkl = True
save_figures = False
save_pxx = True

# save_figures = True ; save_pkl = True
round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'

x = np.array([1])

n = len(x)
g = -0.005  # start


G_dict = {(name1, name2): np.array([g] * (n)),
          (name3, name2): np.array([g] * (n)),
          (name2, name3): np.array([-g] * (n)),
          (name2, name2): np.array([g] * (n)) * 0.2}

filename = 'STN_Proto_Proto_Arky_N_1000_T_2000_' + \
    str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = {k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['STN', 'Proto', 'Arky']
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=3, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, check_peak_significance=False,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=8, high_f=70,
                                                    n_phase_bins=70, start_phase=int(t_sim/4), phase_ref='STN', save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True, nuc_order=nuc_order)

# pickle_obj(data, filepath)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('Arky', 'Proto')][0], 3)) + '_' + str(round(G[('Arky', 'Proto')][-1]*scale, 3)),
          str(round(G[('STN', 'Proto')][0], 3)) + '_' +
          str(round(G[('STN', 'Proto')][-1]*scale, 3)),
          str(round(G[('Proto', 'STN')][0], 3)) + '_' +
          str(round(G[('Proto', 'STN')][-1]*scale, 3)),
          str(round(G[('Proto', 'Proto')][0], 3)) + '_' + str(round(G[('Proto', 'Proto')][-1]*scale, 3))]

    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]
    filename = (names[0] + '_' + names[1] + '_' + names[2] + '_G(PA)=' + gs[0] + '_G(PS)=' + gs[1] + '_G(SP)= ' + gs[2] + '_G(PP)= ' + gs[3] +
                '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method +
                '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[names[0]]) + '_' + str(
        noise_variance[names[1]]) + '_' + str(noise_variance[names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_', 'Phase_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 6), (5, 5), (10, 6), (4, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*4, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()


# %% Synapric weight exploraion STN-GPe-GPe + FSI-D2-GPe-GPe resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]

state = 'rest'
g_ext = 0.01
g = 0
G = {}

G[(name2, name1)], G[(name3, name2)], G[(name1, name3)], G[(name3, name4)
                                                           ], G[(name4, name3)], G[(name3, name3)], = g, g, g, -g, g, g
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}

receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1'), (name4, '1'), (name3, '1')],
                      (name4, '1'): [(name3, '1')]}


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True

keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path,
               save_init=save_init, syn_component_weight=syn_component_weight, noise_method=noise_method) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                          set_FR_range_from_theory=False, method='collective', save_FR_ext=True,
                                          use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state)


n_run = 1
plot_firing = True
plot_spectrum = True
plot_raster = True
low_pass_filter = False
save_pkl = False
save_figures = True
# plot_firing = False; plot_spectrum= False; plot_raster = False; save_pkl = True ; save_figures = False

round_dec = 1
include_std = False
plot_start = int(t_sim/2)
plot_raster_start = int(t_sim/2)
n_neuron = 25
legend_loc = 'center right'

# x = np.flip(np.geomspace(-40, -0.1, n))
# x = np.linspace(0, 12, 15)
coef = 1
# 0.2, 0.7, 0.8
x = np.array([0.75])
n = len(x)
g = -0.005  # start

# G_dict = {(name2, name1) : g  * x ,
#           (name3, name2) : np.array([g]* (n)) ,
#           (name1, name3) : np.array( [g]* (n)),
#           (name3, name3) : np.array( [g]* (n)),
# 		  (name3, name4) : np.array( [-g]* (n)) ,
#  		  (name4, name3) : np.array( [g]* (n))   }
# filename = 'D2_Proto_Proto_FSI_STN_N_1000_T_2000_G_FSI_D2_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# K = calculate_number_of_connections(N, N_real, K_real)
# G_norm = {k: v * K[k]
#           for k, v in G_dict.items()}
# G_dict = {(name2, name1) : np.array([g]* (n)),
#           (name3, name2) : g * x,
#           (name1, name3) : np.array( [g]* (n)),
#           (name3, name3) : np.array( [g]* (n)),
# 		  (name3, name4) : np.array( [-g]* (n)),
#  		  (name4, name3) : np.array( [g]* (n))  }
# filename = 'D2_Proto_Proto_FSI_STN_N_1000_T_2000_G_D2_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array([g]* (n)),
#           (name3, name2) : np.array( [g]* (n)),
#           (name1, name3) : g * x,
#           (name3, name3) : np.array( [g]* (n)),
# 		  (name3, name4) : np.array( [-g]* (n)),
# 		  (name4, name3) : np.array( [g]* (n))  }
# filename = 'D2_Proto_Proto_FSI_STN_N_1000_T_2000_G_Proto_FSI_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array( [g]* (n)),
#           (name3, name2) : np.array( [g]* (n)),
#           (name1, name3) : np.array( [g]* (n)),
#           (name3, name3) : g * x,
# 		  (name3, name4) : np.array( [-g]* (n)),
# 		  (name4, name3) : np.array( [g]* (n))}
# filename = 'D2_Proto_Proto_FSI_STN_N_1000_T_1000_G_Proto_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array( [g]* (n)),
#           (name3, name2) : np.array( [g]* (n)),
#           (name1, name3) : np.array( [g]* (n)),
#           (name3, name3) : np.array( [g]* (n)),
# 		  (name3, name4) :  -g * x,
# 		  (name4, name3) : np.array( [g]* (n))}
# filename = 'D2_Proto_Proto_FSI_STN_N_1000_T_1000_G_STN_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array( [g]* (n)),
#           (name3, name2) : np.array( [g]* (n)),
#           (name1, name3) : np.array( [g]* (n)),
#           (name3, name3) : np.array( [g]* (n)),
# 		  (name3, name4) : np.array( [-g]* (n)),
# 		  (name4, name3) : g * x }
# filename = 'D2_Proto_Proto_FSI_STN_N_1000_T_1000_G_Proto_STN_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = {(name2, name1): g * x,
          (name3, name2): g * x,
          (name1, name3): g * x,
          (name3, name3): g * x,
          (name3, name4): -g * x,
          (name4, name3): g * x}
filename = 'D2_Proto_Proto_FSI_STN_N_1000_T_1000_G_all_changing_' + \
    str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = {k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)

# figs, title, data = synaptic_weight_exploration_SNN_all_changing(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict,
#                                                     noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter, legend_loc = legend_loc,
#                                                     lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = [-10,70], n_run = n_run,  plot_start_raster= plot_raster_start,
#                                                     plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
#                                                     find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=False, save_pkl = save_pkl)

# Note: "synaptic_weight_exploration_SNN" when signal is all plateau it has problem saving empty f array to a designated 200 array.
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec,
                                                    include_std=include_std, find_beta_band_power=True, fft_method=fft_method, n_windows=3, include_beta_band_in_legend=True,
                                                    save_pkl=save_pkl, reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, decimal=1, divide_beta_band_in_power=True,
                                                    spec_lim=[0, 65], include_FR=False)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '_' + str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '_' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '_' +
          str(round(G[('FSI', 'Proto')][-1]*scale, 3)),
          str(round(G[('Proto', 'STN')][0], 3)) + '_' + str(round(G[('Proto', 'STN')][-1]*scale, 3))]

    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2] + names[3] + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] + '_G(SP)= ' + gs[3] +
                '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method +
                '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[names[0]]) + '_' + str(
        noise_variance[names[1]]) + '_' + str(noise_variance[names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(15, 15), (5, 15), (6, 11)]
s = [(8, 13), (5, 10), (6, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*3, s=s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

# %% Synapric weight exploraion Arky-D2-Proto + FSI-D2-GPe resetting inti dists and setting ext input collectively

# sys.modules[__name__].__dict__.clear()
runcell('Constants', '/home/shiva/BG_Oscillations/Oscillation.py')
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name4 = 'Arky'
state = 'rest'
name_list = [name1, name2, name3, name4]

g_ext = 0.01
g = 0
G = {}

G[(name2, name1)], G[(name3, name2)], G[(name1, name3)
                                        ], G[(name2, name4)], G[(name4, name3)] = g, g, g, g, g

poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {
    'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext} for name in name_list}

receiving_pop_list = {(name1, '1'):  [(name3, '1')],
                      (name2, '1'): [(name1, '1'), (name4, '1')],
                      (name3, '1'): [(name2, '1')],
                      (name4, '1'): [(name3, '1')]}
# (name3, '1'): [(name2,'1'), (name3, '1')]} # with GP-GP


pop_list = [1]
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
der_ext_I_from_curve = True

keep_mem_pot_all_t = False
set_input_from_response_curve = True
save_init = False
# noise_variance = {name1 : 8, name2: 3, name3 : 30, name4: 4}
noise_amplitude = {name1: 1, name2: 1, name3: 1, name4: 1}

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve,
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t=keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method, path=path, save_init=save_init) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys())}


# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35,
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True,
#                                           use_saved_FR_ext= False)

# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Arky-D2-Proto-FSI.pkl'))


# Run on previously saved data
FR_ext_all_nuclei = load_pickle(
    os.path.join(path, 'FR_ext_Arky-D2-Proto-FSI.pkl'))
receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list,
                                          all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=35,
                                          set_FR_range_from_theory=False, method='collective', return_saved_FR_ext=False,
                                          use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

n_run = 1
plot_firing = True
plot_spectrum = True
plot_raster = True
low_pass_filter = False
save_pkl = False
save_figures = True
# n_run = 5; plot_firing = False; plot_spectrum= False; plot_raster = False;  low_pass_filter= False ;save_pkl = True ; save_figures = False
# save_figures = True
# save_pkl = False
round_dec = 1
include_std = False
plot_start = 1500  # int(t_sim/2)
plot_raster_start = 1500  # int(t_sim/2)
n_neuron = 30
legend_loc = 'center right'

# x = np.linspace(0.3, 12, 20)
coef = 1
# x = np.array([0.1, 0.694, .91])
# x = np.array([ .91])

x = np.linspace(0.1, 0.91, 16)
n = len(x)
g = -0.005  # start


# G_dict = {(name2, name1) : np.array([g]* (n)) ,
#           (name3, name2): g * x ,
#           (name1, name3) :  np.array( [g]* (n)),
# 			 (name2, name4) :  np.array( [g]* (n)) ,
# 			  (name4, name3) :  np.array( [g]* (n))  }
# filename = 'D2_Proto_FSI_Arky_N_1000_T_2000_G_D2_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) :  g  * x ,
#           (name3, name2): np.array([g]* (n)) ,
#           (name1, name3) :  np.array( [g]* (n)),
# 			 (name2, name4) :  np.array( [g]* (n)) ,
# 			  (name4, name3) :  np.array( [g]* (n))   }
# filename = 'D2_Proto_FSI_Arky_N_1000_T_2000_G_FSI_D2_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array([g]* (n)) ,
#           (name3, name2): np.array( [g]* (n)) ,
#           (name1, name3) : g * x  ,
# 		  (name2, name4) :  np.array( [g]* (n)) ,
# 		  (name4, name3) :  np.array( [g]* (n))  }
# filename = 'D2_Proto_FSI_Arky_N_1000_T_2000_G_Proto_FSI_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array([g]* (n)) ,
#           (name3, name2): np.array( [g]* (n)) ,
#           (name1, name3) : np.array( [g]* (n)) ,
# 		  (name2, name4) :  np.array( [g]* (n)) ,
# 		  (name4, name3) :  g * x   }
# filename = 'D2_Proto_FSI_Arky_N_1000_T_2000_G_Proto_Akry_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : np.array( [g*coef]* (n)) ,
#           (name3, name2) : np.array( [g*coef]* (n)) ,
#           (name1, name3) : np.array( [g*coef]* (n)),
# 		  (name2, name4) :  g * x   ,
# 		  (name4, name3) : np.array( [g*coef]* (n))}
# filename = 'D2_Proto_FSI_Arky_N_1000_T_1000_G_Arky_D2_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = {(name2, name1): g * x,
          (name3, name2): g * x,
          (name1, name3): g * x,
          (name2, name4): g * x,
          (name4, name3): g * x}
filename = 'D2_Proto_FSI_Arky_N_1000_T_1000_G_all_changing_' + \
    str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name4 = 'Arky'
G_dict = {k: v * K[k] for k, v in G_dict.items()}
G_FSI_loop = G_dict[(name1, name3)] * \
    G_dict[(name2, name1)] * G_dict[(name3, name2)]
G_Arky_loop = G_dict[(name2, name4)] * \
    G_dict[(name4, name3)] * G_dict[(name3, name2)]
G_loop = G_FSI_loop + G_Arky_loop

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)

figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=3, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity, state=state, K_real=K_real, 
                                                    K_all = K_all, N_real=N_real, N=N,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale=1):
    G = G_dict
    names = [list(nuclei_dict.values())[i]
             [0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0], 3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale, 3)),
          str(round(G[('Proto', 'D2')][0], 3)) + '--' +
          str(round(G[('Proto', 'D2')][-1]*scale, 3)),
          str(round(G[('FSI', 'Proto')][0], 3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale, 3))]
    gs = [gs[i].replace('.', '-') for i in range(len(gs))]
    nucleus = nuclei_dict[names[0]][0]

    filename = (names[0] + '_' + names[1] + '_' + names[2] + '_G(FD)=' + gs[0] + '_G(DP)=' + gs[1] + '_G(PF)= ' + gs[2] +
                '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method + '_' +
                str(noise_variance[names[0]]) + '_' + str(
        noise_variance[names[1]]) + '_' + str(noise_variance[names[2]])
        + '_N=' + str(nucleus.n) + '_T' + str(nucleus.t_sim) + '_' + fft_method)

    return filename


def save_figs(figs, nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix=['']*3, s=[(15, 15)]*3, scale=1):
    prefix = ['Firing_rate_', 'Power_spectrum_', 'Raster_']
    prefix = [pre_prefix[i] + prefix[i] for i in range(len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(
        nuclei_dict, G, noise_variance, fft_method, scale=scale)
    for i in range(len(figs)):
        figs[i].set_size_inches(s[i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.pdf'), dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)


s = [(8, 13), (5, 10), (6, 12)]
s = [(13, 8), (5, 10), (6, 12)]

if save_figures:
    save_figs(figs, nuclei_dict, G_dict, noise_variance, path,
              fft_method, pre_prefix=['Dem_norm_']*3, s=s)

# manager = plt.get_current_fig_manager()

# %% Synapric weight exploraion Arky-D2-Proto + FSI-D2-GPe + STN-GPe resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 20300
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
duration_base = [int(300/dt), int(t_sim/dt)]
n_windows = int(t_sim - 300) / 1000
name1 = 'FSI'  # projecting
name2 = 'D2'  # recieving
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'
state = 'rest'
state_1 = state
state_2 = 'DD_anesth'
name_list = [name1, name2, name3, name4, name5]

G = {}

# (G[(name2, name1)], G[(name3, name2)],
#  G[(name1, name3)], G[(name2, name4)],
#  G[(name4, name3)], G[(name3, name5)],
#  G[(name5, name3)], G[(name3, name3)]) = g, g, g, g, g, -g * 3, g * 3, g * 0.1

# G = {k: v * K[k] for k, v in G.items()}
# g = -0.00345  # significant peak at low beta
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.1},
#       (name3, name2) :{'mean': g * K[name3, name2] * 4},
#       (name1, name3) :{'mean': g * K[name1, name3] * 2.5},
#       (name2, name4) :{'mean': g * K[name2, name4] * 2.5},
#       (name4, name3) :{'mean': g * K[name4, name3] * 1.6},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 1.7},
#       (name5, name3) :{'mean': g * K[name5, name3] * 2.7},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.1}
#       }


# g = -0.00348 # Proto Arky tau_s sems not corrected for SD
# G = { (name2, name1) :{'mean': g * K[name2, name1] * 3.1},
#       (name3, name2) :{'mean': g * K[name3, name2] * 4},
#       (name1, name3) :{'mean': g * K[name1, name3] * 2.5},
#       (name2, name4) :{'mean': g * K[name2, name4] * 2.5},
#       (name4, name3) :{'mean': g * K[name4, name3] * 1.3},
#       (name3, name5) :{'mean': -g * K[name3, name5] * 1.7},
#       (name5, name3) :{'mean': g * K[name5, name3] * 2.8},
#       (name3, name3) :{'mean': g * K[name3, name3] * 0.1}
#       }


g = -0.0025 ## K_connections tuned July 2022 N = 1000 f = 17 Hz, returned with brice
G = { (name2, name1) :{'mean': g * K[name2, name1]  * 6}, ## free
      (name3, name2) :{'mean': g * K[name3, name2] * 11.}, ## free
      (name1, name3) :{'mean': g * K[name1, name3] * 30 * 66/63}, ## free
      (name2, name4) :{'mean': g * K[name2, name4] * 0.01}, ## free
      (name4, name3) :{'mean': g * K[name4, name3] * 2.5},
      (name3, name5) :{'mean': -g * K[name3, name5] * 2.45 * 62/60},
      (name5, name3) :{'mean': g * K[name5, name3] * 4.3 * 205/180},
      (name3, name3) :{'mean': g * K[name3, name3] * 2.2}}#, 
      # (name1, name5) :{'mean': g * K[name1, name5] * 1}}
G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2)
G_dict = {k: {'mean': [v['mean']]} for k, v in G.items()}


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
               syn_component_weight=syn_component_weight, noise_method=noise_method, state = state) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state_1]
               for name in list(nuclei_dict.keys())}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_all[state], receiving_pop_list, nuclei_dict, t_list,
                                                       all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                       set_FR_range_from_theory=False, method='collective',  save_FR_ext= False,
                                                       use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state_2)



# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 2; plot_firing = False; plot_spectrum = False; plot_raster = False; plot_phase = False; low_pass_filter = False; save_pkl = True; save_figures = False; save_pxx = True


filename = ('All_nuc_' + state + '_N_1000_T_' + str( int(( duration_base[1] -duration_base[0]) * dt) ) +
             '_n_' + str(n_run) + '_runs' + '_tuned.pkl')

round_dec = 1
include_std = False
plot_start = int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
check_peak_significance = False
low_f = 12; high_f = 30


fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['D2', 'STN', 'Arky', 'Proto', 'FSI']
low_f, high_f = 12, 30
phase_ref = 'Proto'
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict,
                                                    noise_amplitude, noise_variance, lim_oscil_perc=10, plot_firing=plot_firing, low_pass_filter=low_pass_filter, legend_loc=legend_loc,
                                                    lower_freq_cut=8, upper_freq_cut=40, set_seed=False, firing_ylim=None, n_run=n_run,  plot_start_raster=plot_raster_start,
                                                    plot_spectrum=plot_spectrum, plot_raster=plot_raster, plot_start=plot_start, plot_end=t_sim, n_neuron=n_neuron, round_dec=round_dec, include_std=include_std,
                                                    find_beta_band_power=True, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=True, save_pkl=save_pkl,
                                                    reset_init_dist=True, all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, end_of_nonlinearity=end_of_nonlinearity,
                                                    state=state, K_real=K_real, K_all = K_all, N_real=N_real, N=N, divide_beta_band_in_power=True,
                                                    receiving_pop_list=receiving_pop_list, poisson_prop=poisson_prop, return_saved_FR_ext=False,
                                                    use_saved_FR_ext=True, check_peak_significance=False,
                                                    find_phase=True, phase_thresh_h=0, filter_order=6, low_f=low_f, high_f=high_f,
                                                    n_phase_bins=72, start_phase=int(t_sim/4), phase_ref=phase_ref, save_pxx=save_pxx,
                                                    plot_phase=plot_phase, total_phase=720, phase_projection=None, troughs=True,
                                                    nuc_order=nuc_order, len_f_pxx=150)


# %% Beta power vs. G low and high beta separate
plt.close('all')


def _str_G_with_key(key):
    return r'$G_{' + list(key)[1] + '-' + list(key)[0] + r'}$'

# title = (r"$G_{"+list(G_dict.keys())[0][0]+"-"+list(G_dict.keys())[0][1]+"}$ = "+ str(round(list(G_dict.values())[0][0],2)) +
#         r"  $G_{"+list(G_dict.keys())[2][0]+"-"+list(G_dict.keys())[2][1]+"}$ ="+str(round(list(G_dict.values())[2][0],2)))


title = ""
n_nuclei = 4
g_cte_ind = [0, 0, 0]
g_ch_ind = [1, 1, 1]
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_N_1000_T_2000_G_D2_Proto_changing_15_pts_2_runs.pkl')]; key = ('Proto','D2')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_N_1000_T_2000_G_FSI_D2_changing_15_pts_2_runs.pkl')]; key = ('D2','FSI')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2 _Proto_FSI_N_1000_T_2000_G_Proto_FSI_changing_15_pts_2_runs.pkl')]; key = ('FSI', 'Proto')

filename_list = n_nuclei * \
    [os.path.join(path, 'Beta_power',
                  'D2_Proto_FSI_STN_N_1000_T_2000_G_FSI_D2_changing_20_pts_5_runs.pkl')]
key = ('D2', 'FSI')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_Proto_FSI_changing_20_pts_5_runs.pkl')]; key = ('FSI', 'Proto')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_Proto_STN_changing_20_pts_5_runs.pkl')]; key = ('STN', 'Proto')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_20_pts_5_runs.pkl')]; key = ('Proto', 'STN')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_D2_Proto_changing_20_pts_5_runs.pkl')]; key = ('Proto', 'D2')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_20_pts_10_runs.pkl')]; key = ('Proto', 'STN')
nucleus_name_list = ['FSI', 'Proto', 'D2', 'STN']

# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_Arky_N_1000_T_1000_G_Arky_D2_changing_20_pts_10_runs.pkl')]; key = ('D2', 'Arky')
# nucleus_name_list = ['FSI', 'Proto','D2', 'Arky']

# filename_list = [os.path.join(path, filename) for filename in filename_list]
# x_axis = 'one'
x_axis = 'multiply'
legend_list = nucleus_name_list
color_list = [color_dict[name] for name in nucleus_name_list]
param_list = n_nuclei * ['base_beta_power']
color_param_list = n_nuclei * ['base_freq']
# param = 'low'
param = 'high'
# param = 'all'
y_line_fix = 2
legend_loc = 'center'
clb_loc = 'upper right'
clb_borderpad = 5
y_line_fix = 2
legend_loc = 'upper left'
clb_loc = 'lower left'
clb_borderpad = 2

fig = synaptic_weight_transition_multiple_circuit_SNN(filename_list, nucleus_name_list, legend_list, color_list, g_cte_ind, g_ch_ind, param_list,
                                                      color_param_list, 'YlOrBr', x_axis=x_axis, param=param,  key=key, y_line_fix=y_line_fix,
                                                      clb_higher_lim=30, clb_lower_lim=5, legend_loc=legend_loc, clb_loc=clb_loc, clb_borderpad=clb_borderpad)
fig.savefig(os.path.join(path, 'Beta_power', 'abs_norm_G_' + param + '_beta_' + os.path.basename(filename_list[0]).replace('.pkl', '.png')), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)

# %% Beta power vs. G low and high beta separate (frequency as inset)
# plt.close('all')


################
# nucleus_name_list = ['FSI', 'Proto','D2']
# # filename_list = os.path.join(path, 'Beta_power', 'D2_Proto_FSI_N_1000_T_2000_G_all_changing_16_pts_5_runs.pkl')
# filename_list = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_FSI_D2_changing_20_pts_10_runs.pkl' )
# filename_list = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_Proto_FSI_changing_20_pts_10_runs.pkl' )
# filename_list = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_D2_Proto_changing_20_pts_10_runs.pkl' )

# key = [('D2', 'FSI'),
#        ('FSI', 'Proto'),
#        ('Proto', 'D2')]
# nuc_loop_lists = [['Proto', 'FSI', 'D2']]
# new_tick_locations = np.array([0.2, 0.5, 0.8]) ; second_axis_label = r"$G_{FSI-D2-P}$";  inset_props = [0.6, 0.6, 0.35, 0.35];
# key_sec_ax = [('D2', 'FSI'),('FSI', 'Proto'),('Proto', 'D2')]
# y_line_fix = None ; legend_loc = 'upper left'
################
nucleus_name_list = ['FSI', 'Proto', 'D2', 'STN']
filename_list = os.path.join(
    path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_16_pts_6_runs.pkl')
filename_list = os.path.join(
    path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_15_pts_5_runs.pkl')

# filename_list = os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_20_pts_5_runs.pkl')
# filename_list = os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_all_changing_8_pts_1_runs.pkl')

data = load_pickle(filename_list)
key = [('D2', 'FSI'),
       ('FSI', 'Proto'),
       ('Proto', 'D2'),
       ('STN', 'Proto'),
       ('Proto', 'STN')]
nuc_loop_lists = [['Proto', 'FSI', 'D2'], ['Proto', 'STN']]
new_tick_locations = np.array([0.2, 0.5, 0.8])
second_axis_label = r"$G_{FSI-D2-P}$"
inset_props = [0.6, 0.6, 0.35, 0.35]
key_sec_ax = [('STN', 'Proto'), ('Proto', 'STN')]
second_axis_label = r"$G_{Proto-STN}$"
y_line_fix = None
legend_loc = 'upper left'
################
# nucleus_name_list = ['FSI', 'Proto','D2', 'Arky']
# n_nuclei = len(nucleus_name_list)
# filename_list = os.path.join(path, 'Beta_power', 'D2_Proto_FSI_Arky_N_1000_T_1000_G_all_changing_16_pts_5_runs.pkl'); key = [('D2', 'Arky'),
#                                                                                                                                           ('D2', 'FSI'),
#                                                                                                                                           ('FSI', 'Proto'),
#                                                                                                                                           ('Proto', 'D2'),
#                                                                                                                                           ('Arky', 'Proto')]
# nuc_loop_lists = [['Proto', 'FSI', 'D2'],['Proto', 'Arky', 'D2']]
# new_tick_locations = np.array([0.2, 0.5, 0.8]) ; second_axis_label = r"$G_{FSI-D2-P}$";  inset_props = [0.02, 0.3, 0.35, 0.35]
# key_sec_ax = [('D2', 'FSI'),('FSI', 'Proto'),('Proto', 'D2')]
# y_line_fix = 2 ; legend_loc = 'upper left'
################
n_nuclei = len(nucleus_name_list)
# x_axis = 'one'
x_axis = 'multiply'
legend_list = nucleus_name_list
color_list = [color_dict[name] for name in nucleus_name_list]
param_list = n_nuclei * ['base_beta_power']
freq_list = n_nuclei * ['base_freq']
# param = 'low'
# param = 'high'
# param = 'all'
param = 'high_low'
include_Gs = False
double_xaxis = False  # True
# ylim = [-0.5, 1.5]
ylim = None
plot_inset = True
fig = synaptic_weight_transition_multiple_circuit_SNN_Fr_inset(filename_list, nucleus_name_list, legend_list, color_list,
                                                               param_list, freq_list, 'YlOrBr', include_Gs=include_Gs,
                                                               x_axis=x_axis, param=param, key=key, y_line_fix=y_line_fix, ylim=ylim,
                                                               legend_loc=legend_loc, double_xaxis=double_xaxis,
                                                               new_tick_locations=new_tick_locations, second_axis_label=second_axis_label,
                                                               key_sec_ax=key_sec_ax, loops='multi', nuc_loop_lists=nuc_loop_lists,
                                                               plot_inset=plot_inset, markersize=5,
                                                               inset_props=[0.3, 0.6, 0.35, 0.35])


save_pdf_png(fig,
             os.path.join(path, 'Beta_power', 'abs_norm_G_single_loop_derivative_' + param + '_beta_' +
                          os.path.basename(filename_list).split('.')[0]),
             size=(7, 5))

# %% Phase summary


state_list = ['rest']#, 'DD_anesth', 'awake_rest', 'mvt']
# state_list = ['awake_rest']
n = 8
t = 5000
filename_dict = {
    'Proto-Proto': [('Proto_Proto_N_1000_T_'+ str(t) + '_1_pts_8_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Proto'], Act, state)) for state in state_list],
    'STN-Proto': [('STN_Proto_N_1000_T_'+ str(t) + '_1_pts_8_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Proto', 'STN'], Act, state)) for state in state_list],
    'FSI Loop': [('D2_Proto_FSI_N_1000_T_'+ str(t) + '_1_pts_8_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['FSI', 'D2', 'Proto'], Act, state)) for state in state_list],
    'Arky Loop': [('D2_Proto_Arky_N_1000_T_'+ str(t) + '_1_pts_8_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Arky', 'D2', 'Proto'], Act, state)) for state in state_list]
}
filename_dict = { key : [os.path.join(path, 'Beta_power', file + '.pkl')
                         for file in filename_list] for key, filename_list in filename_dict.items()}

# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_20_pts_10_runs.pkl' )
# n_g_list = np.linspace(0, 19, endpoint = True, num = 4).astype(int)
# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_4_pts_12_runs.pkl' )
# n_g_list = np.arange(4)
# name_list = ['D2', 'STN', 'Proto', 'FSI']



# filename = os.path.join(path, 'Beta_power','STN_Proto_Arky_N_1000_T_4000_1_pts_8_runs_dt_0-1_rest_Ornstein-Uhlenbeck_A_STN_7_Proto_39-84_Arky_14.pkl' )

# coef = 10
# name_list = [ 'STN', 'Arky', 'Proto']
# name_list = [ 'STN',  'Proto']
# n_g_list = np.array([0])
# y_max_series = {'STN': 6,  'Arky': 18, 'Proto': 25}
# name_ylabel_pad = [0,0,0,0,0]
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (1.8, three_nuc_raster_y)
# ylabel = r'$ Mean \; spike \; count\;/ \;degree (.10 ^{-1})$'
# ylabel_fontsize, xlabel_fontsize = 10, 10
# xlabel_y = -0.05
# phase_text_x_shift = 100
# phase_ref = 'Arky'

# filename = os.path.join(path, 'Beta_power','D2_Proto_Arky_N_1000_T_5000_G_all_changing_1_pts_10_runs.pkl' )
# name_list = ['Proto',  'Arky', 'D2']
# loop = 'Arky Loop'
# filename = filename_dict[loop][0]
# n_g_list = np.array([0])
# y_max_series_dict = {'rest' : {'D2':0.08,  'Arky': 1.5, 'Proto': 2.},
#                       'awake_rest': {'D2':0.1,  'Arky': .8, 'Proto': 2.5}}
# y_max_series = y_max_series_dict[state_list[0]]
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (1.8, three_nuc_raster_y)


# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_FSI_D2_changing_16_pts_8_runs.pkl' )
# n_g_list = np.linspace(0, 15, endpoint = True, num = 4).astype(int)
# name_list = ['D2', 'Proto', 'FSI']
# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_all_changing_3_pts_5_runs_dt_0-1_Ornstein-Uhlenbeck.pkl' )
# loop = 'FSI Loop'
# filename = filename_dict[loop][0]
# name_list = ['Proto', 'FSI', 'D2']
# n_g_list = np.array([0])
# y_max_series_dict = {'rest' : {'D2':0.06,'Proto': 2., 'FSI': .4},
#                       'awake_rest': {'D2':0.13,'Proto': 2.3, 'FSI':1.}}
# y_max_series = y_max_series_dict[state_list[0]]
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (1.8, three_nuc_raster_y )



# filename = os.path.join(path, 'Beta_power', 'STN_Proto_N_1000_T_2000_3_pts_5_runs_dt_0-1_Ornstein-Uhlenbeck.pkl')
# loop = 'STN-Proto'
# filename = filename_dict[loop][0]
# n_g_list = np.array([0])
# name_list = ['Proto', 'STN']
# y_max_series_dict = {'rest' : {'Proto': 2.4, 'STN': .8},
#                       'awake_rest': {'Proto': 2.6, 'STN': 1.2}}
# y_max_series = y_max_series_dict[state_list[0]]
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (1.8, three_nuc_raster_y * 2/3)



# filename = os.path.join(path, 'Beta_power','Proto_Proto_N_1000_T_5000_1_pts_10_runs.pkl' )
# loop = 'Proto-Proto'
# filename = filename_dict[loop][0]
# n_g_list = np.array([0])
# name_list = ['Proto']
# y_max_series_dict = {'rest' :{'Proto': 3},
#                       'awake_rest': {'Proto': 3.4}}
# y_max_series = y_max_series_dict[state_list[0]]
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (1.8, three_nuc_raster_y * 1/3)


filename = os.path.join(path, 'Beta_power','All_nuc_from_rest_to_DD_anesth_N_1000_T_22300_n_1_runs_aligned_to_Proto_tuned_to_Brice_G_lognormal.pkl' )

y_max_series = {'D2': 3, 'STN': 9, 'Arky': 6, 'Proto': 13, 'FSI': 6}
y_max_series = {'D2': 3, 'STN': 1, 'Arky': 1, 'Proto': 3, 'FSI': 1}

n_decimal = 0
phase_ref = 'Proto'
plot_FR = False; 
scale_count_to_FR = False
# y_max_series = {'D2': 6, 'STN': 36, 'Arky': 17, 'Proto': 30, 'FSI': 9}
# y_max_series = {'D2': 15, 'STN': 75, 'Arky': 31, 'Proto': 70, 'FSI': 25} # with single neuron traces



# filename = os.path.join(path, 'Beta_power','All_nuc_from_rest_to_induction_with_excitation_at_D2_N_1000_T_20300_n_1_runs_aligned_to_stimulation.pkl' )
# n_decimal = 0
# phase_ref = 'stimulation'
# scale_count_to_FR = False

# y_max_series = {'D2': 0.3, 'STN': 1.5, 'Arky': 3, 'Proto': 7, 'FSI': 1}
# plot_FR = True; scale_count_to_FR = True
# y_max_series = {'D2': 6., 'STN': 26, 'Arky': 30, 'Proto': 65, 'FSI': 20}
# y_max_series = {'D2': 22., 'STN': 66, 'Arky': 55, 'Proto': 90, 'FSI': 35}  # with single neuron traces
# y_max_series = {'D2': 60., 'STN': 80, 'Arky': 55, 'Proto': 100, 'FSI': 50}  # with single neuron traces

filename = os.path.join(path, 'Beta_power','All_nuc_from_rest_to_induction_with_excitation_at_STN_N_1000_T_19700_n_1_runs_aligned_to_stimulation.pkl' )
n_decimal = 1
phase_ref = 'stimulation'
random_seed = 6
scale_count_to_FR = False
y_max_series = {'D2': 0.5, 'STN': 36, 'Arky': 7, 'Proto': 20, 'FSI': 4}

# scale_count_to_FR = Falseplot_FR = True; 
# y_max_series = {'D2': 1., 'STN': 40, 'Arky': 17, 'Proto': 84, 'FSI': 8}
# y_max_series = {'D2': 13., 'STN': 40, 'Arky': 40, 'Proto': 84, 'FSI': 20}  # with single neuron traces
# y_max_series = {'D2': 8., 'STN': 160, 'Arky': 40, 'Proto': 84, 'FSI': 11}  # with single neuron traces



# filename = os.path.join(path, 'Beta_power','All_nuc_from_rest_to_induction_with_inhibition_at_Proto_N_1000_T_19700_n_1_runs_aligned_to_stimulation.pkl' )
# n_decimal = 0
# phase_ref = 'stimulation'
# random_seed = 1
# scale_count_to_FR = False
# y_max_series = {'D2': 1, 'STN': 17, 'Arky': 8, 'Proto': 14, 'FSI': 6}
# # scale_count_to_FR = True; plot_FR = True; 
# # y_max_series = {'D2': 5, 'STN':90, 'Arky': 60, 'Proto': 120, 'FSI': 68}  # with single neuron traces
# # y_max_series = {'D2': 5, 'STN':110, 'Arky': 60, 'Proto': 120, 'FSI': 68}  # with single neuron traces


coef = 1000
name_list = [ 'Proto','STN', 'Arky', 'FSI', 'D2' ]
n_g_list = np.array([0])
name_ylabel_pad = [-10,-10,-15,-15,-15] # left side
name_ylabel_pad = [-1] * 5 # left side

state = 'rest'
ylabel_fontsize, xlabel_fontsize = 8, 8
xlabel_y = 0.01 ; phase_text_x_shift = 150
FR_dict =  {name: {state: {'mean' : Act[state][name], 'SEM' :0}
                          for state in ['rest']}
                  for  name  in name_list}
# n_decimal = 0

shift_phase = 'backward'
# shift_phase = 'forward'
shift_phase = None
# shift_phase = 'both'

plot_single_neuron_hist = True
smooth_hist = True
box_plot = True
fig = phase_plot(filename, name_list, color_dict, n_g_list, phase_ref=phase_ref,
                    shift_phase=shift_phase, set_ylim=True, y_max_series=y_max_series, coef = coef,
                    ylabel_fontsize = ylabel_fontsize,  xlabel_fontsize = xlabel_fontsize, 
                    tick_label_fontsize = 10, lw = 0.5, name_fontsize = 8,  lw_single_neuron = 1,
                    name_ylabel_pad = name_ylabel_pad, name_place = 'ylabel', alpha = 0.1,
                    alpha_single_neuron = 0.12,
                    xlabel_y = xlabel_y, ylabel_x = -0.1 * (n_decimal + 1), phase_txt_yshift_coef = 1.5,
                    phase_text_x_shift = phase_text_x_shift, n_decimal = n_decimal, state = 'rest',
                    plot_FR = plot_FR, scale_count_to_FR = scale_count_to_FR, FR_dict = FR_dict,
                    plot_single_neuron_hist = plot_single_neuron_hist, n_neuron_hist = 10,
                    hist_smoothing_wind = 5, smooth_hist = smooth_hist, shift_phase_deg = 40, random_seed = random_seed)#,
                    # box_plot = False, strip_plot = True)



# fig = remove_all_x_labels(fig)
# fig = set_y_ticks(fig, [0, n_neuron])

# fig.gca().xaxis.set_minor_locator([180, 540])

figname = filename.split('.')[0] + '_Phase'
if scale_count_to_FR:
    figname += '_FR'
    
if plot_single_neuron_hist:
    filename += '_with_single_neuron_traces'
save_pdf_png(fig, figname, size=(1.8, len(name_list) * 1))
# save_pdf_png(fig, figname, size = figsize)
# %% PSD summary 

state_list = ['rest']#, 'DD_anesth', 'awake_rest', 'mvt']
# state_list = ['awake_rest']
n_runs = 5
t = 10000
state = 'rest'
filename_dict = {
    'Proto-Proto': [('Proto_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_runs) + '_runs_dt_0-1' + 
                      '_A_' + get_str_of_A_with_state(['Proto'], Act, state)) for state in state_list],
    'STN-Proto': [('STN_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_runs) + '_runs_dt_0-1' +
                      '_A_' + get_str_of_A_with_state(['STN','Proto'], Act, state)) for state in state_list],
    'FSI Loop': [('D2_Proto_FSI_N_1000_T_'+ str(t) + '_1_pts_' + str(n_runs) + '_runs_dt_0-1' +
                      '_A_' + get_str_of_A_with_state(['FSI', 'D2', 'Proto'], Act, state)) for state in state_list],
    'Arky Loop': [('D2_Proto_Arky_N_1000_T_'+ str(t) + '_1_pts_' + str(n_runs) + '_runs_dt_0-1' +
                      '_A_' + get_str_of_A_with_state(['Arky', 'D2', 'Proto'], Act, state)) for state in state_list]
                }

filename_dict = { key : [os.path.join(path, 'Beta_power', file + '.pkl')
                         for file in filename_list] for key, filename_list in filename_dict.items()}




# filename = os.path.join(path, 'Beta_power','All_nuc_from_rest_to_DD_anesth_N_1000_T_25300_n_1_runs_aligned_to_Proto_tuned_to_Brice_G_lognormal_17_Hz.pkl' )
f_in_leg = True
vspan = False
# filename = os.path.join(path, 'Beta_power','All_nuc_rest_N_1000_T_3000_n_1_runs_tuned.pkl' )
# f_in_leg = False
axvspan_c = axvspan_color['DD_anesth']

name_list = ['D2', 'STN', 'Arky', 'Proto', 'FSI']
n_g_list = np.array([0])
# n_g_list = np.arange(4)
ylabel = 'Norm. PSD ' + r'$(\times 10^{-2})$'
xlabel = 'Frequency (Hz)'

loop = 'Arky Loop'
filename = filename_dict[loop][0]
name_list = ['D2', 'Arky', 'Proto']
n_g_list = np.array([0])
three_nuc_raster_y = (60 + 5) * 0.05
figsize = (2.5, three_nuc_raster_y )
ylim = [0, 15]

# loop = 'FSI Loop'
# filename = filename_dict[loop][0]
# name_list = ['D2', 'FSI', 'Proto']
# n_g_list = np.array([0])
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (2.5, three_nuc_raster_y )
# ylim = [0, 10]


loop = 'STN-Proto'
filename = filename_dict[loop][0]
name_list = ['STN', 'Proto']
n_g_list = np.array([0])
three_nuc_raster_y = (60 + 5) * 0.05
figsize = (2.5, three_nuc_raster_y * 2/3)
ylim = [0, 8]


# loop = 'Proto-Proto'
# filename = filename_dict[loop][0]
# n_g_list = np.array([0])
# name_list = ['Proto']
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (2.5, three_nuc_raster_y * 1/3)
# ylim = [0, 5]

# filename = os.path.join(path, 'Beta_power','STN_Proto_Proto_Arky_N_1000_T_2000_1_pts_20_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_Proto_Arky_N_1000_T_2000_1_pts_20_runs.pkl' )
# name_list = [ 'STN', 'Arky', 'Proto']

# filename = os.path.join(
#     path, 'Beta_power', 'Proto_N_1000_T_5000_4_pts_8_runs_dt_0-1_Ornstein-Uhlenbeck.pkl')
# inset_props = [0.6, 0.3, 0.35, 0.35]
# name_list = ['Proto']
# n_g_list = np.arange(4)

# filename = os.path.join(
#     path, 'Beta_power', 'STN_N_1000_T_5000_3_pts_8_runs_dt_0-1_Ornstein-Uhlenbeck.pkl')
# inset_props = [0.6, 0.3, 0.35, 0.35]
# name_list = ['STN']
# n_g_list = np.arange(3)

# n_g_list = np.linspace(0, 19, endpoint = True, num = 4).astype(int)
# n_g_list = np.array([0])


xlabel_y = -0.05
# xlabel_y = 0.05
vspan = False
span_beta = False
fig = PSD_summary(filename, name_list, color_dict, n_g_list, xlim=(0, 80), # inset_props=inset_props,
                  # err_plot = 'errorbar', inset_name=None)#, inset_yaxis_loc = 'left')
                  err_plot='fill_between', inset_name=None, plot_lines=False, legend_font_size = 9, 
                  legend_loc='upper right', x_y_label_size = 15, tick_label_fontsize = 12, tick_length = 6,
                   f_in_leg = True, legend = False, xlabel_y = xlabel_y, log_scale = 2, span_beta = span_beta,
                  axvspan_color = axvspan_c, vspan = vspan, normalize_PSD = True, f_decimal = 1,
                  xlabel = '', ylabel_norm ='',  x_ticks = [], xaxis_invert = True, ylim = ylim)
                  # y_ticks = [0, 15]),  x_ticks = [0,20,40,60,80]
                  # xlabel = xlabel, ylabel_norm =ylabel)


# fig = remove_all_x_labels(fig)
figsize = (2.5, 2.5 * len(n_g_list) / 2.5)
save_pdf_png(fig, filename.split('.')[0] + '_PSD',
             size=figsize)


# %% Boxplot frequency vs loop

state_list = ['rest'] #, 'DD_anesth', 'awake_rest', 'mvt']
T_list = [10000] * 4
n_run = 8


filename_dict = {
    
    'Proto-Proto': [('Proto_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1' + #'_' + state + 
                      '_A_' + get_str_of_A_with_state(['Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'STN-Proto': [('STN_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1' + #'_' + state + 
                      '_A_' + get_str_of_A_with_state(['STN', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'FSI Loop': [('D2_Proto_FSI_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1' + #'_' + state + 
                      '_A_' + get_str_of_A_with_state(['FSI', 'D2', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'Arky Loop': [('D2_Proto_Arky_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1' + #'_' + state +  
                      '_A_' + get_str_of_A_with_state(['Arky', 'D2', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)]
}

filename_dict = { key : [os.path.join(path, 'Beta_power', file + '.pkl')
                         for file in filename_list] 
                 for key, filename_list in filename_dict.items()}

loop_list = list( filename_dict.keys() )

color_list = [color_dict['Proto'], color_dict['STN'],
              color_dict['FSI'], color_dict['Arky']]

color_dict_loops = {'Proto-Proto': color_dict['Proto'], 'STN-Proto' :color_dict['STN'],
                    'FSI Loop': color_dict['FSI'], 'Arky Loop': color_dict['Arky']}


xlabels = loop_list    
# size = (2, 4)
size = (4, 2.5)

def plot_loop_state_freq(filename_dict, loop_list, state_list,  n_run, 
                          color_dict_loops, xlabels):
    
    fig, ax = plt.subplots()
    state = state_list[0]
    xs = []
    vals = []
    freq = np.zeros((n_run, len(list( filename_dict. keys()))))
    
    for i, (loop, filename_list) in enumerate( filename_dict.items() ):
        data = load_pickle(filename_list[0])
        base_freq = data[('Proto', 'base_freq')]
        freq[:, i] = base_freq
        xs.append(np.random.normal(i+1, 0.04, n_run))
        vals.append(base_freq)

    bp = ax.boxplot(freq, labels=loop_list, patch_artist=True,
                    whis=(0, 100), zorder=0, widths = 0.3)
    
    bp = set_boxplot_prop(bp, color_list, linewidths = {'box': 0.5, 'whiskers': 0.5,
                                                        'caps': 0.5, 'median': .5})
    
    for x, val, c in zip(xs, vals, color_list):
        ax.scatter(x, val, c=c, alpha=0.4, s=10, ec='k', zorder=1, lw = 0.5)
        
        set_minor_locator(ax, n = 4, axis = 'y')

    ax.tick_params(axis='x', labelsize=10, rotation=40, length = 8)
    ax.tick_params(axis='y', labelsize=12, length = 8)
    ax.tick_params(axis='y', which = 'minor', labelsize=12, length = 4)

    ax.axhspan(12, 30, color='lightgrey', alpha=0.5, zorder=0)
    ax.set_ylabel('Frequency (Hz)', fontsize = 18)
    set_y_ticks(fig, [0, 40 ,80])
    remove_frame(ax)
    ax.set_ylim(0, 80)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
    # set_boxplot_prop(bp, color_list)
    save_pdf_png(fig, os.path.join(path, 'mean_F_all_loops_' + state),
                  size=size)
    # ax.set_xlabel('')
    # ax.set_ylabel('')

    


plot_loop_state_freq(filename_dict, loop_list, state_list,  n_run, 
                         color_dict_loops, xlabels)

# %% Boxplot frequency vs loop (tau explore FSI loop)

state_list = ['rest'] #, 'DD_anesth', 'awake_rest', 'mvt']
T_list = [4000] * 4
n_run = 8
n_tau_PF = 8

filename_dict = {
    
    'Proto-Proto': [('Proto_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'STN-Proto': [('STN_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['STN', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'FSI Loop': [('D2_Proto_FSI_tau_sweep_N_1000_T_'+ str(t) + '_8_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['FSI', 'D2', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'Arky Loop': [('D2_Proto_Arky_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Arky', 'D2', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)]
}

filename_dict = { key : [os.path.join(path, 'Beta_power', file + '.pkl')
                         for file in filename_list] 
                 for key, filename_list in filename_dict.items()}

loop_list = list( filename_dict.keys() )

color_list = [color_dict['Proto'], color_dict['STN'],
              color_dict['FSI'], color_dict['Arky']]

color_dict_loops = {'Proto-Proto': color_dict['Proto'], 'STN-Proto' :color_dict['STN'],
                    'FSI Loop': color_dict['FSI'], 'Arky Loop': color_dict['Arky']}


xlabels = loop_list    

def plot_loop_state_freq(filename_dict, loop_list, state_list,  n_run, 
                          color_dict_loops, xlabels, n_tau_PF = 0, annotate = True):
    plt.rcParams.update({'font.family':'sans-serif'})

    np.random.seed(10)
    fig, ax = plt.subplots()
    state = state_list[0]
    xs = []
    vals = []
    freq = np.zeros((n_run, len(list( filename_dict. keys()))))
    
    for i, (loop, filename_list) in enumerate( filename_dict.items() ):
        data = load_pickle(filename_list[0])
        if loop == 'FSI Loop':
        
            base_f = data[('Proto', 'base_freq')][n_tau_PF, :]
            if annotate:
                tau_Proto_FSI = str(round( data['tau']['FSI','Proto']['mean'][n_tau_PF], 2))
                ax.annotate(r'$\tau_{decay}^ {Proto-FSI}$' + r'$ \; =' + tau_Proto_FSI + '\; ms$', 
                            xy=( 0.5, np.average(base_f) + 10),
                            fontsize= 11)#, xycoords='axes fraction')
            
        else:
            
            base_f = data[('Proto', 'base_freq')]
        freq[:, i] = base_f
        xs.append(np.random.normal(i+1, 0.04, n_run))
        vals.append(base_f)

    bp = ax.boxplot(freq, labels=loop_list,
                    whis=(0, 100), zorder=0)
    bp = set_boxplot_prop(bp, color_list, linewidths = {'box': 0.5, 'whiskers': 0.5,
                                                        'caps': 0.5, 'median': .5})

    for x, val, c in zip(xs, vals, color_list):
        plt.scatter(x, val, c=c, alpha=0.4, s=10, ec='k', zorder=1, linewidth = 0.5)
        

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=12)
    ax.axhspan(13, 30, color='lightgrey', alpha=0.5, zorder=0)
    ax.set_ylabel('Frequency (Hz)', fontsize = 18)
    plt.xticks(rotation=40)
    remove_frame(ax)
    ax.set_ylim(0, 70)
    set_x_tick_colors(ax, color_list)

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
    
    if not annotate:
        
        save_pdf(fig, os.path.join(path, 'Loop_rest_freq', 
                                    'mean_F_all_loops_' + state + '_tau_Proto_FSI_' + str(n_tau_PF)),
                  size=(2, 4))
    save_png(fig, os.path.join(path, 'Loop_rest_freq', 
                                    'mean_F_all_loops_' + state + '_tau_Proto_FSI_' + str(n_tau_PF)),
                  size=(2, 4))
    plt.close('all')

plot_loop_state_freq(filename_dict, loop_list, state_list,  n_run, 
                              color_dict_loops, xlabels, n_tau_PF = 1,
                              annotate = False)
# for tau_i in range(n_tau_PF):
#     plot_loop_state_freq(filename_dict, loop_list, state_list,  n_run, 
#                               color_dict_loops, xlabels, n_tau_PF = tau_i)

# %% Boxplot frequency vs loop vs. state


state_list = ['rest', 'DD_anesth', 'awake_rest', 'mvt']
T_list = [4000] * 4
n_run = 8

filename_dict = {
    
    'Proto-Proto': [('Proto_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'STN-Proto': [('STN_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['STN', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'FSI Loop': [('D2_Proto_FSI_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['FSI', 'D2', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'Arky Loop': [('D2_Proto_Arky_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Arky', 'D2', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)]
}

filename_dict = { key : [os.path.join(path, 'Beta_power', file + '.pkl')
                         for file in filename_list] 
                 for key, filename_list in filename_dict.items()}

loop_list = list( filename_dict.keys() )

color_list = [color_dict['Proto'], color_dict['STN'],
              color_dict['FSI'], color_dict['Arky']]

color_dict_Loops = {'Proto-Proto': color_dict['Proto'], 'STN-Proto' :color_dict['STN'],
              'FSI Loop': color_dict['FSI'], 'Arky Loop': color_dict['Arky']}

xlabels = {'rest': 'Anesthesia', 'DD_anesth': 'Anesthesia DD', 'awake_rest': 'Awake rest', 'mvt': 'Movement'}
xlabels = {'rest': 'Anesth', 'DD_anesth': '  DD ', 'awake_rest': 'Awake', 'mvt': 'Mvmt'}

def plot_loop_state_freq(filename_dict, loop_list, state_list, n_run, 
                         color_dict_loops, xlabels, n_tau_PF = 0):
    
    fig, axes = plt.subplots(1, len(loop_list) , figsize = (2 * len(loop_list), 6))
    
    for count, loop in enumerate(loop_list):
        
        filename_list = filename_dict[loop]
        ax = axes[count] 
        xs = []
        vals = []
        freq = np.zeros((n_run, len(filename_list)))
        
        for i, filename in enumerate(filename_list):
            data = load_pickle(filename)
            if loop == 'FSI Loop':
            
                base_f = data[('Proto', 'base_freq')][n_tau_PF, :]
                tau_Proto_FSI = str(round( data['tau']['FSI','Proto']['mean'][n_tau_PF], 2)).replace('.', '-')
            else:
                
                base_f = data[('Proto', 'base_freq')]
            freq[:, i] = base_freq
            xs.append(np.random.normal(i+1, 0.04, n_run))
            vals.append( base_freq )
            
        color = color_dict_Loops[loop]
        
        bp = ax.boxplot(freq, labels= list(xlabels.values()),
                        whis=(0, 100), zorder=0)
        
        bp = set_boxplot_prop(bp, color_list, linewidths = {'box': 0.5, 'whiskers': 0.5,
                                                            'caps': 0.5, 'median': .5})
        
        for x, val, c in zip(xs, vals, [color] * len(state_list)):
            ax.scatter(x, val, c=c, alpha=0.4, s=10, ec='grey', zorder=1, lw = 0.5)
            
        # ax.set_xlabel(list(xlabels.values()), fontdict = font_label)
    
        ax.tick_params(axis='x', labelsize=15, rotation = 40)
        ax.tick_params(axis='y', labelsize=15)
        
        
        font = {'family': 'serif',
                'color': color,
                'weight': 'normal',
                'size': 16,
                }
        ax.set_title(loop, fontdict = font)
        # set_y_ticks(fig, [10,30,60])
        remove_frame(ax)
        ax.set_ylim(0, 65)
        ax.axhspan(12, 30, color='lightgrey', alpha=0.5, zorder=0)
        
        shift_x_ticks(ax, shift_to ='right')
            
        if count == 0:
            ax.set_ylabel('Frequency (Hz)', fontsize = 18)
            
    save_pdf_png(fig, os.path.join(path, 'mean_F_all_loops_all_states'),
                 size=(8, 6))

plot_loop_state_freq(filename_dict, loop_list, state_list,  n_run, 
                         color_dict_loops, xlabels)

# %% Boxplot frequency vs loop vs. state (tau explore FSI loop)



state_list = ['rest', 'DD_anesth', 'awake_rest', 'mvt']
T_list = [4000] * 4
n_run = 8
n_tau_PF = 8
filename_dict = {
    
    'Proto-Proto': [('Proto_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'STN-Proto': [('STN_Proto_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['STN', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'FSI Loop': [('D2_Proto_FSI_tau_sweep_N_1000_T_'+ str(t) + '_8_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['FSI', 'D2', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)],

    'Arky Loop': [('D2_Proto_Arky_N_1000_T_'+ str(t) + '_1_pts_' + str(n_run) + '_runs_dt_0-1_' + state + 
                      '_Ornstein-Uhlenbeck_A_' + get_str_of_A_with_state(['Arky', 'D2', 'Proto'], Act, state)) for state,t in zip(state_list,T_list)]
}

filename_dict = { key : [os.path.join(path, 'Beta_power', file + '.pkl')
                         for file in filename_list] 
                 for key, filename_list in filename_dict.items()}

loop_list = list( filename_dict.keys() )

color_list = [color_dict['Proto'], color_dict['STN'],
              color_dict['FSI'], color_dict['Arky']]

color_dict_loops = {'Proto-Proto': color_dict['Proto'], 'STN-Proto': color_dict['STN'],
                    'FSI Loop': color_dict['FSI'], 'Arky Loop': color_dict['Arky']}

xlabels = {'rest': 'Anesthesia', 'DD_anesth': 'Anesthesia DD', 'awake_rest': 'Awake rest', 'mvt': 'Movement'}
xlabels = {'rest': 'Anesth', 'DD_anesth': '  DD ', 'awake_rest': 'Awake', 'mvt': 'Mvmt'}

def plot_loop_state_freq(filename_dict, loop_list, state_list,  n_run, 
                         color_dict_loops, xlabels, n_tau_PF = 0):
    
    np.random.seed(10)
    fig, axes = plt.subplots(1, len(loop_list) , figsize = (2 * len(loop_list), 6))

    for count, loop in enumerate(loop_list):
        
        filename_list = filename_dict[loop]
        ax = axes[count] 
        xs = []
        vals = []
        freq = np.zeros((n_run, len(filename_list)))
        annotate= False
        
        for i, filename in enumerate(filename_list):
            
            data = load_pickle(filename)
            
            if loop == 'FSI Loop':
            
                base_f = data[('Proto', 'base_freq')][n_tau_PF, :]
                tau_Proto_FSI = str(round( data['tau']['FSI','Proto']['mean'][n_tau_PF], 2))
                if not annotate :
                    annotate = True
                    ax.annotate(r'$\tau_{decay}^ {Proto-FSI}$' + '\n ' + r'$=' + tau_Proto_FSI + '\; ms$', 
                                xy=( i+ 0.5, np.average(base_f) + 10),
                                fontsize= 15)#, xycoords='axes fraction')
            else:
                
                base_f = data[('Proto', 'base_freq')]
            
            freq[:, i] = base_f
            xs.append(np.random.normal(i+1, 0.04, n_run))
            vals.append( base_f )
            
        color = color_dict_loops[loop]
        
        bp = ax.boxplot(freq, labels= list(xlabels.values()), 
                        whis=(0, 100), zorder=0)
        bp = set_boxplot_prop(bp, [color] * len(freq), linewidths = {'box': 0.5, 'whiskers': 0.5,
                                                            'caps': 0.5, 'median': .5})
        for x, val, c in zip(xs, vals, [color] * len(state_list)):
            ax.scatter(x, val, c=c, alpha=0.4, s=10, ec='grey', zorder=1, lw = 0.5)
                
        ax.tick_params(axis='x', labelsize=15, rotation = 40)
        ax.tick_params(axis='y', labelsize=15)

        font = {'family': 'serif',
                'color': color,
                'weight': 'normal',
                'size': 16,
                }

        ax.set_title(loop, fontdict = font)
        remove_frame(ax)
        ax.set_ylim(0, 70)
        ax.axhspan(12, 30, color='lightgrey', alpha=0.5, zorder=0)

        shift_x_ticks(ax, shift_to ='right')

        if count == 0:
            ax.set_ylabel('Frequency (Hz)', fontsize = 18)
            
    save_png(fig, os.path.join(path, 'Loop_state_freq', 'mean_F_all_loops_all_states_tau_Proto_FSI_' + str(n_tau_PF)),
                 size=(8, 6))
    plt.close('all')

for tau_i in range(n_tau_PF):
    
    plot_loop_state_freq(filename_dict, loop_list, state_list, n_run, 
                         color_dict_loops, xlabels, n_tau_PF = tau_i)
# %% Phase summary only entrained

phase_ref = 'D2'
entr_nuc_name = 'D2'
low_f, high_f = 8, 70
filter_based_on_AUC_of_PSD = False
only_entrained_neurons = False
c_dict = color_dict.copy()
find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins=100,
                                  height=0, phase_ref=phase_ref, start=0, total_phase=720,
                                  only_entrained_neurons=only_entrained_neurons)
c_dict[entr_nuc_name] = color_dict[entr_nuc_name]
fig = phase_plot_all_nuclei_in_grid(nuclei_dict, c_dict, dt,
                                    density=False, phase_ref=phase_ref, total_phase=720, projection=None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18, plot_mode='hist',
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)
c_dict[entr_nuc_name] = 'g'
only_entrained_neurons = True
find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order=6, n_bins=100,
                                  height=0, phase_ref=phase_ref, start=0, total_phase=720,
                                  only_entrained_neurons=only_entrained_neurons, min_f_sig_thres=0, window_mov_avg=10, max_f=250,
                                  n_window_welch=6, n_sd_thresh=2, n_pts_above_thresh=2,
                                  min_f_AUC_thres=7,  PSD_AUC_thresh=10**-6, filter_based_on_AUC_of_PSD=filter_based_on_AUC_of_PSD)

fig = phase_plot_all_nuclei_in_grid(nuclei_dict, c_dict, dt,
                                    density=False, phase_ref=phase_ref, total_phase=720, projection=None,
                                    outer=None, fig=fig,  title='', tick_label_fontsize=18, plot_mode='hist',
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)
# c_dict[entr_nuc_name] = 'g'
# only_entrained_neurons = True
# find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
#                                   height = 0, phase_ref = phase_ref, start = 0, total_phase = 720,
#                                   only_entrained_neurons =only_entrained_neurons, min_f_sig_thres = 0,window_mov_avg = 10, max_f = 250,
#                                   n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2,
#                                   min_f_AUC_thres = 7,  PSD_AUC_thresh = 10**-4.5, filter_based_on_AUC_of_PSD = filter_based_on_AUC_of_PSD)

# fig = phase_plot_all_nuclei_in_grid(nuclei_dict, c_dict, dt,
#                           density = False, phase_ref = phase_ref, total_phase = 720, projection = None,
#                           outer=None, fig=fig,  title='', tick_label_fontsize=18,
#                            labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

# %% FR simulation vs FR_expected ( heterogeneous vs. homogeneous initialization)

N_sim = 20
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

G = {}
name = 'D2'
g = -0.01
g_ext = -g
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name, '1'): []}

pop_list = [1]

tuning_param = 'firing'
n = 10
# p = np.arange(1,5,dtype=float)
# FR_list = np.ones(len(p))*np.power(10,-p)
start = 1
end = 200
FR_list = np.linspace(start, end, n)


init_method = 'homogeneous'
nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuclei_dict = {name: nuc}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')

firing_prop = find_FR_sim_vs_FR_expected(
    FR_list, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict, A, A_mvt, D_mvt, t_mvt)

init_method = 'heterogeneous'
nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuclei_dict = {name: nuc}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')

firing_prop_hetero = find_FR_sim_vs_FR_expected(
    FR_list, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict, A, A_mvt, D_mvt, t_mvt)

plt.figure()

plt.plot(FR_list, firing_prop[name]['firing_mean'][:, 0],
         '-o', label='simulation_homogeneous', c='darkred')
plt.fill_between(FR_list, firing_prop[name]['firing_mean'][:, 0]-firing_prop[name]['firing_var'][:, 0],
                 firing_prop[name]['firing_mean'][:, 0]+firing_prop[name]['firing_var'][:, 0], alpha=0.2, color='darkred')

plt.plot(FR_list, firing_prop_hetero[name]['firing_mean']
         [:, 0], '-o', label='simulation_heterogeneous', c='teal')
plt.fill_between(FR_list, firing_prop_hetero[name]['firing_mean'][:, 0]-firing_prop_hetero[name]['firing_var'][:, 0],
                 firing_prop_hetero[name]['firing_mean'][:, 0]+firing_prop_hetero[name]['firing_var'][:, 0], alpha=0.2, color='teal')

plt.plot(FR_list, FR_list, '--', label='y=x', c='k')
plt.xlabel(r'$FR_{expected}$', fontsize=10)
plt.ylabel(r'$FR_{simulation}$', fontsize=10)
plt.legend()


# %% FR vs FR_ext theory vs. simulation

N_sim = 100
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

G = {}
name = 'Proto'

g = -0.01
g_ext = -g
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}
receiving_pop_list = {(name, '1'): []}

pop_list = [1]
init_method = 'homogeneous'
# ext_inp_method = 'Poisson'
ext_inp_method = 'const+noise'

noise_variance = {name: 3}
noise_amplitude = {name: 1}

label = ext_inp_method + r' $\sigma = ' + str(noise_variance[name]) + 'mV$'
# label = ext_inp_method


nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking',
               poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=False, ext_inp_method=ext_inp_method) for i in pop_list]

nuclei_dict = {name: nuc}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')

start = 0.175
end = 0.185
n = 20
FR_list = spacing_with_high_resolution_in_the_middle(
    n, start, end).reshape(-1,)


firing_prop = find_FR_sim_vs_FR_ext(
    FR_list, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict, A, A_mvt, D_mvt, t_mvt)
plt.figure()
plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, I_ext_range, neuronal_consts)
# + ' N=' + str(poisson_prop[name]['n']) , )
plt.plot(FR_list * 1000,
         firing_prop[name]['firing_mean'][:, 0], '-o', c='teal', label=label)

plt.fill_between(FR_list * 1000, firing_prop[name]['firing_mean'][:, 0]-firing_prop[name]['firing_var'][:, 0],
                 firing_prop[name]['firing_mean'][:, 0] + firing_prop[name]['firing_var'][:, 0], alpha=0.2, color='teal')
plt.title(name + ' ' + init_method, fontsize=18)
plt.legend()
if ext_inp_method == 'Poisson':
    filename = ('FR_sim_vs_FR_ext_' + name + '_' + init_method + '_' + ext_inp_method +
                '_N=' + str(N_sim) + '_N_ext=' + str(poisson_prop[name]['n']) + '.png')
else:
    filename = ('FR_sim_vs_FR_ext_' + name + '_' + init_method + '_' + ext_inp_method + '_noise=' +
                str(noise_variance[name]) + '_N=' + str(N_sim) + '.png')
plt.savefig(os.path.join(path, filename), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait',
            transparent=True, bbox_inches="tight", pad_inches=0.1)


# %% two nuclei with given I_ext + noise

N_sim = 1000
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

name1 = 'D2'
name2 = 'Proto'
G = {}
g = -0.01
g_ext = -g
G[('Proto', 'D2')] = g

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'): [],
                      (name2, '1'): [(name1, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
noise_variance = {name1: D2_noise, name2: D2_noise}
noise_amplitude = {name1: 1, name2: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')

noise_std_min = 10**-17
noise_std_max = 10**-1  # interval of search for noise variance

nuclei_dict = run(receiving_class_dict, t_list, dt,
                  nuclei_dict, neuronal_model='spiking')
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax=None, title_fontsize=15, plot_start=100,
           title='')  # r"$G_{SP}="+str(round(G[('Proto', 'STN')],2))+"$ "+", $G_{PS}=G_{PP}="+str(round(G[('STN', 'Proto')],2))+'$')

fig, axs = plt.subplots(len(nuclei_dict), 1, sharex=True, sharey=True)
count = 0
for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        count += 1
        nucleus.smooth_pop_activity(dt, window_ms=5)
        FR_mean, FR_std = nucleus. average_pop_activity(
            t_list, last_fraction=1/2)
        print(nucleus.name, 'average ={}, std = {}'.format(FR_mean, FR_std))
        spikes_sparse = create_sparse_matrix(nucleus.spikes) * dt
        raster_plot(axs[count - 1], spikes_sparse, nucleus.name,
                    color_dict, labelsize=10, title_fontsize=15)
        find_freq_of_pop_act_spec_window_spiking(
            nucleus, 0, t_list[-1], dt, cut_plateau_epsilon=0.1, peak_threshold=0.1, smooth_kern_window=3, check_stability=False)

fig.text(0.5, 0.02, 'time (ms)', ha='center', va='center', fontsize=15)
fig.text(0.02, 0.5, 'neuron', ha='center',
         va='center', rotation='vertical', fontsize=15)
# %% Proto and FSI binary search


def find_D2_I_ext_in_D2_FSI(x):
    print(x)
    noise_variance = {name1: x, name2: 10**-10}
    noise_amplitude = {name1: 1, name2: 1}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(neuronal_model='spiking')
            nucleus.set_noise_param(noise_variance, noise_amplitude)
            # nucleus.reset_ext_pop_properties(poisson_prop,dt)
    print('I_ext', np.average(nuc1[0].rest_ext_input))
    nuclei_dict_p = run(receiving_class_dict, t_list, dt,
                        nuclei_dict, neuronal_model='spiking')
    for nuclei_list in nuclei_dict_p.values():
        for nucleus in nuclei_list:
            nucleus.smooth_pop_activity(dt, window_ms=5)
            print(nucleus.name, np.average(nucleus.pop_act[int(
                len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]), 2))
    return np.average(nuc1[0].pop_act[int(len(t_list)/2):]) - A[name1]


N_sim = 1000
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

name1 = 'FSI'
name2 = 'Proto'
G = {}
g = -0.01
g_ext = -g
G[('FSI', 'Proto')] = g


poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'): [(name2, '1')],
                      (name2, '1'): []}

pop_list = [1]
init_method = 'heterogeneous'
noise_variance = {name1: 0, name2: 0}
noise_amplitude = {name1: 1, name2: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')
ratio_list = np.linspace(0.9, 1.5, 1)
noise_std_min = 10**-17
noise_std_max = 10**-1  # interval of search for noise variance
for ratio in ratio_list:
    try:
        noise = optimize.bisect(find_D2_I_ext_in_D2_FSI,
                                noise_std_min, noise_std_max, xtol=10**-10)
        print('noise = ', noise)
        break
    except ValueError as err:
        print(err)
# find_D2_I_ext( 0.025003981672404058)

noise_std_min = 10**-17
noise_std_max = 10**-10  # interval of search for noise variance
# D2_noise = optimize.bisect(find_D2_I_ext, noise_std_min, noise_std_max, xtol = 10**-20)
# print('D2 noise = ',D2_noise)
# %% D2 and FSI binary search


def find_D2_I_ext_in_D2_FSI(x):
    print(x)
    noise_variance = {name1: x, name2: 10**-10}
    noise_amplitude = {name1: 1, name2: 1}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(neuronal_model='spiking')
            nucleus.set_noise_param(noise_variance, noise_amplitude)
            # nucleus.reset_ext_pop_properties(poisson_prop,dt)
    print('I_ext', np.average(nuc1[0].rest_ext_input))
    nuclei_dict_p = run(receiving_class_dict, t_list, dt,
                        nuclei_dict, neuronal_model='spiking')
    for nuclei_list in nuclei_dict_p.values():
        for nucleus in nuclei_list:
            nucleus.smooth_pop_activity(dt, window_ms=5)
            print(nucleus.name, np.average(nucleus.pop_act[int(
                len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]), 2))
    return np.average(nuc1[0].pop_act[int(len(t_list)/2):]) - A[name1]


N_sim = 1000
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

name1 = 'D2'
name2 = 'FSI'
G = {}
g = -0.01
g_ext = -g
G[('D2', 'FSI')] = g

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'): [],  # [(name2, '1')],
                      (name2, '1'): []}

pop_list = [1]
init_method = 'heterogeneous'
noise_variance = {name1: 0, name2: 0}
noise_amplitude = {name1: 1, name2: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')
ratio_list = np.linspace(1, 1.5, 1)
noise_std_min = 10**-17
noise_std_max = 10**-1  # interval of search for noise variance
for ratio in ratio_list:
    try:
        D2_noise = optimize.bisect(
            find_D2_I_ext_in_D2_FSI, noise_std_min, noise_std_max, xtol=10**-10)
        print('D2 noise = ', D2_noise)
        break
    except ValueError as err:
        print(err)
# find_D2_I_ext( 3.0766320304347586e-13)

# D2_noise = optimize.bisect(find_D2_I_ext, noise_std_min, noise_std_max, xtol = 10**-20)
# print('D2 noise = ',D2_noise)
# %% D2- FSI - Proto binary search


def find_D2_I_ext_in_D2_FSI_Proto(x):
    print('x = ', x)
    noise_variance = {name1: x, name2: 10**-10, name3: 0.025003981672404058}
    noise_amplitude = {name1: 1, name2: 1, name3: 1}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(neuronal_model='spiking')
            nucleus.set_noise_param(noise_variance, noise_amplitude)
            # nucleus.reset_ext_pop_properties(poisson_prop,dt)
    nuclei_dict_p = run(receiving_class_dict, t_list, dt,
                        nuclei_dict, neuronal_model='spiking')
    for nuclei_list in nuclei_dict_p.values():
        for nucleus in nuclei_list:
            nucleus.smooth_pop_activity(dt, window_ms=5)
            print(nucleus.name, np.average(nucleus.pop_act[int(
                len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]), 2))
    return np.average(nuc1[0].pop_act[int(len(t_list)/2):]) - A[name1]


N_sim = 100
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

name1 = 'D2'
name2 = 'Proto'
name3 = 'FSI'
G = {}
g = -0.1
g_ext = -g
G[('D2', 'FSI')], G[('FSI', 'Proto')], G[('Proto', 'D2')] = g, g, g*0.5

poisson_prop = {name1: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name2: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext},
                name3: {'n': 10000, 'firing': 0.0475, 'tau': {'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name1, '1'): [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]
init_method = 'heterogeneous'
noise_variance = {name1: 0, name2: 0, name3: 0}
noise_amplitude = {name1: 1, name2: 1, name3: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
                synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')
I_ext_list = np.linspace(0.055, 0.06, 10)
noise_std_min = 10**-9
noise_std_max = 10**-2  # interval of search for noise variance
for I_ext in I_ext_list:
    try:
        # nuc3[0].rest_ext_input = nuc3[0].rest_ext_input * 0.9
        D2_noise = optimize.bisect(
            find_D2_I_ext_in_D2_FSI_Proto, noise_std_min, noise_std_max, xtol=10**-10)
        print('D2 noise = ', D2_noise)
        break
    except ValueError as err:
        print(err)


# %% Check single population firing with external poisson spikes

N_sim = 1000
N = {'STN': N_sim, 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim,
     'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt
G = {}
receiving_pop_list = {(name, '1'): []}
g = -0.01
g_ext = -g
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

pop_list = [1]
# init_method = 'heterogeneous'
init_method = 'homogeneous'
noise_variance = {name: 0.1}
noise_amplitude = {name: 1}
nuc = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop, init_method=init_method) for i in pop_list]
nuclei_dict = {name: nuc}
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')
nuc[0].rest_ext_input = I_ext / 1000
nuclei_dict = run(receiving_class_dict, t_list, dt,
                  nuclei_dict, neuronal_model='spiking')
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt,
           D_mvt, ax=None, title_fontsize=15, plot_start=100, title='')

fig, axs = plt.subplots(len(nuclei_dict), 1, sharex=True, sharey=True)
count = 0
for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        count += 1
        nucleus.smooth_pop_activity(dt, window_ms=5)
        print(nucleus.name, np.average(nucleus.pop_act[int(
            len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]), 2))
        spikes_sparse = [np.where(nucleus.spikes[i, :] == 1)[
            0]*dt for i in range(nucleus.n)]

        axs.eventplot(spikes_sparse, colors='k', linelengths=2,
                      lw=2, orientation='horizontal')
        axs.tick_params(axis='both', labelsize=10)
        axs.set_title(nucleus.name, c=color_dict[nucleus.name], fontsize=15)
        find_freq_of_pop_act_spec_window_spiking(
            nucleus, 0, t_list[-1], dt, cut_plateau_epsilon=0.1, peak_threshold=0.1, smooth_kern_window=3, check_stability=False)

fig.text(0.5, 0.02, 'time (ms)', ha='center', va='center', fontsize=15)
fig.text(0.02, 0.5, 'neuron', ha='center',
         va='center', rotation='vertical', fontsize=15)
# fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
# fig2, ax2 = plt.subplots(1, 1, sharex=True, sharey=True)
# fig3, ax3 = plt.subplots(1, 1, sharex=True, sharey=True)

# for nuclei_list in nuclei_dict.values():
#     for nucleus in nuclei_list:
#         ax1.plot(t_list*dt,nucleus.voltage_trace,c = color_dict[nucleus.name],label = nucleus.name)
#         ax2.plot(t_list*dt,nucleus.representative_inp['ext_pop','1'],c = color_dict[nucleus.name],label = nucleus.name)
#         ax3.plot(t_list*dt,np.average(nucleus.ext_input_all,axis = 0),c = color_dict[nucleus.name],label = nucleus.name)
#         ax3.fill_between(t_list*dt, np.average(nucleus.ext_input_all,axis = 0) - np.std(nucleus.ext_input_all,axis = 0),
#                          np.average(nucleus.ext_input_all,axis = 0) + np.std(nucleus.ext_input_all,axis = 0), alpha = 0.3)
#         # ax3.plot(t_list*dt,np.sum([nucleus.representative_inp[key].reshape(-1,) for key in nucleus.representative_inp.keys()],axis =0)-nucleus.representative_inp['ext_pop','1'],
#                   # c = color_dict[nucleus.name],label = nucleus.name)
# ax1.set_title('membrane potential',fontsize = 15)
# ax2.set_title('external input one neuron',fontsize = 15)
# ax3.set_title('Mean external input',fontsize = 15)
# ax1.legend();ax2.legend() ; ax3.legend()
# plt.legend()

# %% Measuring AUC of the input of one spike (what constant external input reproduces the same firing with poisson spike inputs)

N_sim = 10
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 100
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

g = -1
G = {}
g_ext = 1
name = 'Proto'
poisson_prop = {name: {'n': 1000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name, '1'): []}
pop_list = [1]
find_AUC_of_input(name, path, poisson_prop, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                  N, A, A_mvt, D_mvt, t_mvt, N_real, K_real, t_list, color_dict, G, T, t_sim, dt, synaptic_time_constant,
                  receiving_pop_list, smooth_kern_window, oscil_peak_threshold, syn_component_weight, end_of_nonlinearity, if_plot=True)
# %% Compare FR of Possion ext inp with Const+noise
np.random.seed(19090)
# N_sim = 1000
N_sim = 10
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 100
t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim
D_mvt = t_sim - t_mvt

g = -1
G = {}
g_ext = 1
name = 'Proto'
poisson_prop = {name: {'n': 1000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .1}, 'decay': {'mean': 5, 'var': 0.5}}, 'g': g_ext}}

receiving_pop_list = {(name, '1'): []}
pop_list = [1]


class Nuc_poisson(Nucleus):
    def cal_ext_inp(self, dt, t):
        poisson_spikes = possion_spike_generator(
            self.n, self.n_ext_population, self.firing_of_ext_pop, dt)
        self.syn_inputs['ext_pop', '1'] = (np.sum(
            poisson_spikes, axis=1)*self.membrane_time_constant*self.syn_weight_ext_pop).reshape(-1,)
        # self.I_syn['ext_pop','1'] += np.true_divide((-self.I_syn['ext_pop','1'] + self.syn_inputs['ext_pop','1']),self.tau_ext_pop['decay']) # without rise
        self.I_rise['ext_pop', '1'] += ((-self.I_rise['ext_pop', '1'] +
                                        self.syn_inputs['ext_pop', '1'])/self.tau_ext_pop['rise'])
        self.I_syn['ext_pop', '1'] += np.true_divide(
            (-self.I_syn['ext_pop', '1'] + self.I_rise['ext_pop', '1']), self.tau_ext_pop['decay'])

    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp):

        self.cal_ext_inp(dt, t)
        inputs = self.I_syn['ext_pop', '1']*10
        self.mem_potential = self.mem_potential + \
            np.true_divide((inputs - self.mem_potential+self.u_rest)
                           * dt, self.membrane_time_constant)
        # gaussian distributed spike thresholds
        spiking_ind = np.where(self.mem_potential > self.spike_thresh)
        self.spikes[spiking_ind, t] = 1
        self.mem_potential[spiking_ind] = self.neuronal_consts['u_rest']
        self.pop_act[t] = np.average(self.spikes[:, t], axis=0)/(dt/1000)


nuc = [Nuc_poisson(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop) for i in pop_list]
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')

print(Proto[0].mem_potential[5])
nuclei_dict = {name: nuc}
nuclei_names = list(nuclei_dict.keys())

tuning_param = 'firing'
start = 0.05
end = 0.3
n = 12

list_1 = np.linspace(start, end, n)

nucleus_name = list(nuclei_dict.keys())
m = len(list_1)
firing_prop = {k: {'firing_mean': np.zeros((m, len(nuclei_dict[nucleus_name[0]]))), 'firing_var': np.zeros(
    (m, len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
ext_firing = np.zeros((m, len(nucleus_name)))
i = 0
for g in list_1:
    for j in range(len(nucleus_name)):
        poisson_prop[nucleus_name[j]][tuning_param] = g
        ext_firing[i, j] = g
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history()
            nucleus.reset_ext_pop_properties(poisson_prop, dt)
    nuclei_dict = run(receiving_class_dict, t_list, dt,
                      nuclei_dict, neuronal_model='spiking')
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            # firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = np.average(nucleus.pop_act[int(len(t_list)/2):]) # population activity
            firing_prop[nucleus.name]['firing_mean'][i, nucleus.population_num-1] = np.average(
                nucleus.spikes[5, int(len(t_list)/2):])/(dt/1000)  # single neuron activity
            firing_prop[nucleus.name]['firing_var'][i, nucleus.population_num -
                                                    1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
            print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop, 3),
                  'FR=', firing_prop[nucleus.name]['firing_mean'][i, nucleus.population_num-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, nucleus.population_num-1], 2))
    i += 1

''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''


class Nuc_I_ext_cte(Nucleus):
    def solve_EIF(self, t, dt, receiving_from_class_list, mvt_ext_inp):
        inputs = self.rest_ext_input*self.membrane_time_constant*self.n_ext_population
        self.mem_potential = self.mem_potential + \
            np.true_divide((inputs - self.mem_potential+self.u_rest)
                           * dt, self.membrane_time_constant)
        # gaussian distributed spike thresholds
        spiking_ind = np.where(self.mem_potential > self.spike_thresh)
        self.spikes[spiking_ind, t] = 1
        self.mem_potential[spiking_ind] = self.neuronal_consts['u_rest']
        self.pop_act[t] = np.average(self.spikes[:, t], axis=0)/(dt/1000)


np.random.seed(19090)
Proto = [Nuc_I_ext_cte(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt,
                       synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', poisson_prop=poisson_prop) for i in pop_list]
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, neuronal_model='spiking')
print(Proto[0].mem_potential[5])
nuclei_dict = {'Proto': Proto}
nuclei_names = list(nuclei_dict.keys())

tuning_param = 'firing'
start = .05
end = 0.30
n = 20
list_cte = np.linspace(start, end, n)

nucleus_name = list(nuclei_dict.keys())
m = len(list_cte)
firing_prop_cte = {k: {'firing_mean': np.zeros((m, len(nuclei_dict[nucleus_name[0]]))), 'firing_var': np.zeros(
    (m, len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
ext_firing = np.zeros((m, len(nucleus_name)))
i = 0
for g in list_cte:
    for j in range(len(nucleus_name)):
        poisson_prop[nucleus_name[j]][tuning_param] = g
        ext_firing[i, j] = g
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(neuronal_model='spiking')
            nucleus.reset_ext_pop_properties(poisson_prop, dt)
            nucleus.rest_ext_input = g*np.ones(nucleus.n)
    nuclei_dict = run(receiving_class_dict, t_list, dt,
                      nuclei_dict, neuronal_model='spiking')
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            # firing_prop_cte[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
            firing_prop_cte[nucleus.name]['firing_mean'][i, nucleus.population_num -
                                                         1] = np.average(nucleus.spikes[5, int(len(t_list)/2):])/(dt/1000)
            firing_prop_cte[nucleus.name]['firing_var'][i, nucleus.population_num -
                                                        1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
            print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop, 3),
                  'FR=', firing_prop_cte[nucleus.name]['firing_mean'][i, nucleus.population_num-1], 'std=', round(firing_prop_cte[nucleus.name]['firing_var'][i, nucleus.population_num-1], 2))
    i += 1

plt.figure()
plt.plot(list_1, firing_prop['Proto']['firing_mean'],
         '-o', label=r'$G\times FR (decay=5)$',)
plt.plot(list_cte, firing_prop_cte['Proto']
         ['firing_mean'], '-o', label=r'$I_{ext}*10$')
plt.legend()
plt.ylabel('FR(Hz)')
plt.xlabel(r'$G\times FR (Spk/ms)$')

# %% RATE MODEL : STN-Proto network

plt.close('all')

N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = 700
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]
plot_start = t_mvt - 200
plot_duration = 600
plot_start_stable = 0

plot_start = 0
plot_duration = t_sim
n_windows = 1
name1 = 'Proto'
name2 = 'STN'
name_list = [name1, name2]
state = 'rest'
(synaptic_time_constant[(name2, name1)],
 synaptic_time_constant[(name1, name2)]) = [10], [6]

g = -1.7
G = {('STN', 'Proto'): g,
     ('Proto', 'STN'): -g}  # synaptic weight

receiving_pop_list = {('STN', '1'): [('Proto', '1')],
                      ('Proto', '1'): [('STN', '1')]}


lim_n_cycle = [6, 10]
pop_list = [1]

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, change_states= False)

nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,  plot_start=plot_start) 
figname = 'STN-GPe loop with Proto-Proto_no_ocsill'

peak_threshold = 0.1
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
check_stability = False
freq, f, pxx = find_freq_all_nuclei(dt, nuclei_dict, duration_mvt, lim_oscil_perc, peak_threshold,
                             smooth_kern_window, smooth_window_ms,
                             cut_plateau_epsilon, check_stability, 'fft', False,
                             low_pass_filter, 0, 100, plot_spectrum=True, 
                             ax=None, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                             fft_method='Welch', n_windows=n_windows, include_beta_band_in_legend=True, smooth = False, 
                             normalize_spec = False, include_peak_f_in_legend = True,
                             check_significance = False, plot_sig_thresh = False, plot_peak_sig = False,
                             min_f = 100, max_f = 300, n_std_thresh = 2,AUC_ratio_thresh = 0.8)


# %% RATE MODEL : GPe-GPe G-sweep

N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 20000
t_list = np.arange(int(t_sim/dt))
t_mvt = 700
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]
plot_start_trans = t_mvt - 200
plot_duration = 600
plot_start_stable = 0

name1 = 'Proto'

state_1 = 'awake_rest'
state_2 = 'mvt'

name_list = [name1]


G_list = np.linspace(-7, 0, n, endpoint=True)

transition_range = [4.25 - (4.27 - 4.245), 4.27]  # dt = 0.5
transition_range = [3.8778 - (3.913 - 3.8778), 3.913]  # dt = 0.1

G_list = -pad_high_res_spacing_with_arange(
    1, transition_range[0], 1/10, transition_range[1], 5.5,  1/10, 20)
n = len(G_list)


synaptic_time_constant[('Proto', 'Proto')] = [10]

G = {
    (name1, name1): -1
}

G_ratio_dict = {
    (name1, name1): 1
}
receiving_pop_list = {('Proto', '1'): [('Proto', '1')]}


lim_n_cycle = [6, 10]
pop_list = [1]

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)

# run(receiving_class_dict,t_list, dt, nuclei_dict)
# plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,
#                                       include_FR = False, plot_start = 0, legend_loc = 'upper left',
#                                       title_fontsize = 15, ax = None)

filename = ('G_sweep_Proto-Proto' + '_tau_' +
            str(synaptic_time_constant[(name1, name1)][0]) +
            '_n_' + str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.', '-'))
filename = os.path.join(path_rate,
                        filename.replace('.', '-') + '.pkl'
                        )
fig_trans, fig_stable = synaptic_weight_exploration_RM(G.copy(), Act[state_1], Act[state_2], D_mvt, t_mvt, t_list, dt,  filename,
                                                          lim_n_cycle, G_list, nuclei_dict, duration_mvt, duration_base,
                                                          receiving_class_dict, color_dict, G_ratio_dict=G_ratio_dict,
                                                          if_plot=if_plot, legend_loc='upper right', plot_start_trans=plot_start_trans,
                                                          plot_start_stable=plot_start_stable, plot_duration=plot_duration)

# fig_trans , fig_stable = set_ylim_trans_stable_figs(fig_trans, fig_stable, ymax = [100, 100], ymin = [-4. -4])
save_trans_stable_figs(fig_trans, fig_stable, path_rate, filename.split('.')[0], figsize=(5, 3),
                       ymax=[55, 55], ymin=[-4, -4])
pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto'
color = 'perc_t_oscil_mvt'  # mvt_f'
param = 'perc_t_oscil_mvt'

xlabel = r'$G_{Loop}$'


g = Product_G(data)
fig = scatter_2d_plot(abs(g),
                      np.squeeze(data[(name, param)]),
                      np.squeeze(data[(name, color)]),
                      name + ' in GPe-GPe circuit',
                      [xlabel, param, color])

plt.axvline(data['g_loop_stable'], c='k')
plt.axvline(data['g_loop_transient'], c='grey', linestyle='--')
save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size=(8, 6))
# %% RATE MODEL : STN-GPe without GPe-GPe G-sweep
N_sim = 100
N = dict.fromkeys(N, N_sim)

dt = 0.2
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = 700
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]
plot_start_trans = t_mvt - 200
plot_duration = 600
plot_start_stable = 0

name1 = 'Proto'
name2 = 'STN'

state_1 = 'awake_rest'
state_2 = 'mvt'
name_list = [name1, name2]


# transition_range = [4.55-(4.6-4.55), 4.6] # dt = 0.5
transition_range = [3.55-(3.57 - 3.55), 3.57]  # dt = 0.1 SP = 4, PS = 4.75
transition_range = [5.08 - (5.139 - 5.08), 5.139]  # dt = 0.1 SP = 2.8, PS = 1.3
transition_range = [5.124, 5.1446]  # dt = 0.1 SP = 2.8, PS = 1.3

G_list = pad_high_res_spacing_with_arange(  
    1, transition_range[0], 1/10, transition_range[1], 6.2,  1/10, 22, base = 1.1)





G_list = -np.power(G_list, 1/2)

n = len(G_list)
(synaptic_time_constant[(name2, name1)],
 synaptic_time_constant[(name1, name2)]) = [10], [6]

G = {
    (name2, name1): -1,
    (name1, name2): 1
}

receiving_pop_list = {('STN', '1'): [('Proto', '1')],
                      ('Proto', '1'): [('STN', '1')]}

G_ratio_dict = {
    (name2, name1): 1,
    (name1, name2): -1
}

lim_n_cycle = [6, 10]
pop_list = [1]

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, Act[state_1], Act[state_2], name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    Act[state_1], Act[state_2], D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)



# run(receiving_class_dict,t_list, dt, nuclei_dict)
# plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,
#                                       include_FR = False, plot_start = 0, legend_loc = 'upper left',
#                                       title_fontsize = 15, ax = None)


filename = ('G_sweep_STN-GPe' + '_tau_' + name1[0] + name2[0] + '_' +
            str(synaptic_time_constant[(name2, name1)][0]) + '_' + name2[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name1, name2)][0]) +
            '_n_' + str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.', '-') + '_SP_2-8')

filename = os.path.join(path_rate,
                        filename.replace('.', '-') + '.pkl'
                        )
fig_unconnected, fig_trans, fig_stable = synaptic_weight_exploration_RM( N, N_real, K_real,
                                                                           G.copy(), Act[state_1], Act[state_2], D_mvt, t_mvt, t_list, dt, filename,
                                                                           lim_n_cycle, G_list, nuclei_dict, duration_mvt, duration_base,
                                                                           receiving_class_dict, color_dict, G_ratio_dict=G_ratio_dict,
                                                                           legend_loc='upper right', plot_start_trans=plot_start_trans,
                                                                           plot_start_stable=plot_start_stable, plot_duration=plot_duration,
                                                                           path = os.path.join(path_rate, 'STN-GP_anim'), 
                                                                           if_plot = True, legend_fontsize=18)

fig_stable.gca().get_legend().remove()
fig_stable.gca().legend(fontsize = 20, loc = 'upper right', frameon = False)
fig_trans.gca().legend(fontsize = 20, loc = 'upper right', frameon = False)
fig_unconnected.gca().legend(fontsize = 20, loc = 'upper right', frameon = False)

# # fig_trans , fig_stable = set_ylim_trans_stable_figs(fig_trans, fig_stable, ymax = [100, 100], ymin = [-4, -4])
# save_trans_stable_figs([fig_unconnected, fig_trans, fig_stable], 
#                        ['unconnected', 'transient', 'stable'], 
#                        path_rate, filename.split('.')[
#                        0], figsize=(5, 3), ymax=[75, 75, 75], ymin=[-4, -4, -4])

# data = load_pickle(filename)

# name = 'Proto'
# color = 'last_first_peak_ratio_mvt'  # mvt_f'
# param = 'last_first_peak_ratio_mvt'

# xlabel = r'$G_{Loop}$'


# g = Product_G(data)
# fig = scatter_2d_plot(abs(g),
#                       np.squeeze(data[(name, param)]),
#                       np.squeeze(data[(name, color)]),
#                       name + ' in STN-GPe circuit',
#                       [xlabel, param, color])

# plt.axvline(data['g_loop_stable'], c='k')
# plt.axvline(data['g_loop_transient'], c='grey', linestyle='--')
# save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0],     )

# %% RATE MODEL : STN-GPe without GPe-GPe tau-sweep

N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.2
t_sim = 20500
t_list = np.arange(int(t_sim/dt))
t_mvt = 500
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]

name1 = 'Proto'
name2 = 'STN'

state_1 = 'awake_rest'
state_2 = 'mvt'

# state = 'rest'

name_list = [name1, name2]

g_list = np.linspace(-5, -0.5, 150)


(synaptic_time_constant[(name2, name1)],
 synaptic_time_constant[(name1, name2)]) = [10], [6]

G = {
    (name2, name1): -1,
    (name1, name2): 1
}

receiving_pop_list = {('STN', '1'): [('Proto', '1')],
                      ('Proto', '1'): [('STN', '1')]}

G_ratio_dict = {
    ('STN', 'Proto'): 1,
    ('Proto', 'STN'): -1
}

lim_n_cycle = [6, 10]
pop_list = [1]

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

n = 20
syn_decay_dict = {'tau_1':
                  {
                      'tau_ratio': {('STN', 'Proto'): 1
                                    },
                      'tau_list': np.linspace(5, 25, n)
                  },
                  'tau_2': {
                      'tau_ratio': {('Proto', 'STN'): 1},
                      'tau_list': [6]}  # np.linspace(1,15,n)}}
                  }

filename = ('Tau_sweep_STN-GPe_tau_ratio_' + name1[0] + name2[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name2, name1)]) + '_' + name2[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_2']['tau_ratio'][(name1, name2)]) + '_G_ratio_' + name1[0] + name2[0] + '_' +
            str(abs(G_ratio_dict[(name2, name1)])) + '_' + name2[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name1, name2)])) + '_n_' + str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.', '-') + 'SP_2-8')

filename = os.path.join(path_rate,
                        filename.replace('.', '-') + '.pkl'
                        )
receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, change_states = False)



sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, N, N_real, K_real, syn_decay_dict, filename,
                     G.copy(), Act[state_1], Act[state_2], D_mvt, t_mvt, receiving_class_dict, t_list, dt, duration_base, duration_mvt,
                     lim_n_cycle, change_states = False)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto'
color = 'trans_n_half_cycle'
color = 'freq'

g_transient = data[(name, 'g_transient')]
g_transient = data[(name, 'g_stable')]

# x = data['tau'][(name2, name1)]
# y = data['tau'][(name1, name2)]

# z_transient = data[(name,'trans_mvt_freq')]
# z_stable = data[(name, 'stable_mvt_freq')]
# c = data[(name, 'trans_n_half_cycle')]

xlabel = r'$\tau_{decay}^{inhibition}(ms)$'
# ylabel = r'$\tau_{decay}^{excitaion}(ms)$'

# def highlight_middle_glut(ax, y, ind = 3):

#     x_spec =  data['tau'][(name2, name1)][:,0]
#     y_spec = data[(name, 'stable_mvt_freq')][:,ind]

#     ax.scatter(x_spec,np.ones_like(x_spec)* y[0,ind],
#                y_spec, c = ['k'] * len(y_spec), s = 80)

# fig, ax = scatter_3d_wireframe_plot(x,y,z_stable, z_stable,
#                                    name +' in STN-GPe circuit',
#                                    [xlabel, ylabel,'frequency(Hz)','frequency(Hz)'])
# highlight_middle_glut(ax, y, ind = 3)

ind = 0
scatter_2d_plot(data['tau'][(name2, name1)][:, 0],
                data[(name, 'stable_mvt_freq')][:, ind],
                data[(name, 'stable_mvt_freq')][:, ind],
                name + ' in STN-GP circuit',
                [xlabel, 'Frequency(Hz)', 'Frequency(Hz)'])
# save_pdf_png(fig, 'STN_GPe_timescale_inh_excit_3d.png', size = (8,6))

# %% RATE MODEL : GPe-GPe tau-sweep
N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 20500
t_list = np.arange(int(t_sim/dt))
t_mvt = 500
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]

name1 = 'Proto'

state_1 = 'awake_rest'
state_2 = 'mvt'
state = 'rest'
name_list = [name1]

g_list = np.linspace(-10, -0.5, 200, endpoint=True)

synaptic_time_constant = {
    ('Proto', 'Proto'): [10]
}

G = {
    (name1, name1): -1
}

G_ratio_dict = {
    (name1, name1): 1
}
receiving_pop_list = {('Proto', '1'): [('Proto', '1')]}


lim_n_cycle = [6, 10]
pop_list = [1]

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

n = 20
syn_decay_dict = {'tau_1':
                  {
                      'tau_ratio': {('Proto', 'Proto'): 1
                                    },
                      'tau_list': np.linspace(5, 25, n)
                  },
                  'tau_2': {
                      'tau_ratio': {('Proto', 'STN'): 1},
                      'tau_list': [6]}  # np.linspace(1,15,n)}}
                  }

filename = ('Tau_sweep_GPe-GPe_tau_ratio_' + name1[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name1, name1)]) + '_' + name1[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name1, name1)])) + '_n_' + str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.', '-'))

filename = os.path.join(path_rate,
                        filename.replace('.', '-') + '.pkl'
                        )
receiving_class_dict , nuclei_dict = set_connec_ext_inp(path,
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, change_states = False)


find_stable_oscill = True  # to find stable oscillatory regime

sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, N, N_real, K_real, syn_decay_dict, filename,
                     G.copy(), Act[state_1], Act[state_2], D_mvt, t_mvt, receiving_class_dict, t_list, dt, duration_base, duration_mvt,
                     lim_n_cycle, change_states = False, if_track_tau_2 = False)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto'
color = 'trans_n_half_cycle'
color = 'freq'

# g_transient = data[(name,'g_transient')]
# g_transient = data[(name,'g_stable')]

# x = data['tau'][(name2, name1)]
# y = data['tau'][(name1, name2)]

# z_transient = data[(name,'trans_mvt_freq')]
# z_stable = data[(name, 'stable_mvt_freq')]
# c = data[(name, 'trans_n_half_cycle')]

xlabel = r'$\tau_{decay}^{inhibition}(ms)$'
# ylabel = r'$\tau_{decay}^{excitaion}(ms)$'

# def highlight_middle_glut(ax, y, ind = 3):

#     x_spec =  data['tau'][(name2, name1)][:,0]
#     y_spec = data[(name, 'stable_mvt_freq')][:,ind]

#     ax.scatter(x_spec,np.ones_like(x_spec)* y[0,ind],
#                y_spec, c = ['k'] * len(y_spec), s = 80)

# fig, ax = scatter_3d_wireframe_plot(x,y,z_stable, z_stable,
#                                    name +' in STN-GPe circuit',
#                                    [xlabel, ylabel,'frequency(Hz)','frequency(Hz)'])
# highlight_middle_glut(ax, y, ind = 3)

ind = 0
scatter_2d_plot(data['tau'][(name1, name1)][:, 0],
                data[(name, 'stable_mvt_freq')][:, ind],
                data[(name, 'stable_mvt_freq')][:, ind],
                name + ' in STN-GP circuit',
                [xlabel, 'Frequency(Hz)', 'Frequency(Hz)'])
# save_pdf_png(fig, 'STN_GPe_timescale_inh_excit_3d.png', size = (8,6))
# %% RATE MODEL : STN-GPe with GPe-GPe tau-sweep 

N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.5
t_sim = 10000
t_list = np.arange(int(t_sim/dt))
t_mvt = 400
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]

name1 = 'Proto'
name2 = 'STN'
name_list = [name1, name2]

g_list = np.linspace(-10, -0.01, 150)


(synaptic_time_constant[(name2, name1)],
 synaptic_time_constant[(name1, name1)],
 synaptic_time_constant[(name1, name2)]) = [10], [10], [6]

G = {
    (name2, name1): -1,
    (name1, name2): 1,
    (name1, name1): -1}

receiving_pop_list = {('STN', '1'): [('Proto', '1')],
                      ('Proto', '1'): [('STN', '1'), ('Proto', '1')]}

G_ratio_dict = {
    ('Proto', 'Proto'): 1,
    ('STN', 'Proto'): 2,
    ('Proto', 'STN'): -1
}

lim_n_cycle = [6, 10]
pop_list = [1]

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

n = 30
syn_decay_dict = {'tau_1':
                  {
                      'tau_ratio': {('STN', 'Proto'): 1,
                                    ('Proto', 'Proto'): 1},
                      'tau_list': np.linspace(5, 25, n)
                  },
                  'tau_2': {
                      'tau_ratio': {('Proto', 'STN'): 1},
                      'tau_list': [6]}  # np.linspace(1,15,n)}}
                  }

filename = ('Tau_sweep_STN-GPe-GPe_tau_ratio_' + name2[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name2, name1)]) + '_' + name1[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name1, name1)]) + '_' + name1[0] + name2[0] + '_' +
            str(syn_decay_dict['tau_2']['tau_ratio'][(name1, name2)]) + '_G_ratio_' + name2[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name2, name1)])) + '_' + name1[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name1, name1)])) + '_' + name1[0] + name2[0] + '_' +
            str(abs(G_ratio_dict[(name1, name2)])) + '_n_' + str(n) + '_T_' + str(t_sim))

filename = os.path.join(path_rate,
                        filename.replace('.', '-') + '.pkl'
                        )
receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)


find_stable_oscill = True  # to find stable oscillatory regime

sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename,
                     G.copy(), A, A_mvt, D_mvt, t_mvt, receiving_class_dict, t_list, dt, duration_base, duration_mvt,
                     lim_n_cycle, find_stable_oscill)
pkl_file = open(os.path.join(path_rate, filename), 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
name = 'Proto'
color = 'trans_n_half_cycle'
color = 'freq'
# x = data['tau'][:,:,0]
# y = data['tau'][:,:,1]
# z_transient = data[(name,'trans_mvt_freq')]
# z_stable = data[(name, 'stable_mvt_freq')]
# c = data[(name, 'trans_n_half_cycle')]

# xlabel = r'$\tau_{decay}^{inhibition}(ms)$'
# ylabel = r'$\tau_{decay}^{excitaion}(ms)$'

# fig, ax = scatter_3d_wireframe_plot(x,y,z_stable, z_stable,
#                                    name +' in STN-GPe circuit',
#                                    [xlabel, ylabel,'frequency(Hz)','frequency(Hz)'],
#                                     label_fontsize=20)
param = 'stable_mvt_freq'
x_spec = data['tau'][('Proto', 'Proto')][:, 0]
y_spec = data[(name, 'stable_mvt_freq')][:, 0]
xlabel = r'$\tau_{inhibition} \; (ms)$'
# ax.azim = 60
# ax.dist = 10
# ax.elev = 30
# ax.scatter(x_spec,np.ones_like(x_spec)* y[0,4],
#            y_spec, c = ['k'] * len(y_spec), s = 80)
scatter_2d_plot(x_spec, y_spec, y_spec,
                'STN-GPe + GPe-GPe',
                [xlabel, 'Frequency(Hz)', 'Frequency(Hz)'])
# save_pdf_png(fig, 'STN_GPe_GPe-timescale_inh_excit_3d.png', size = (8,6))

# %% RATE MODEL : Any Pallidostriatal loop with GPe-GPe 

plt.close('all')
N_sim = 100
N = dict.fromkeys(N, N_sim)

if_plot = False
dt = 0.1
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = 1000
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]
plot_start = t_mvt - 200
plot_duration = 600

name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI'
name_list = [name1, name2, name3]

g = - np.power( 15 , 1/3 )
G = {
    (name2, name3): g,
    (name3, name1): g,
    (name1, name2): g,
    (name1, name1): g
}

state_1 = 'awake_rest'
state_2 = 'mvt'


receiving_pop_list = {(name3, '1'): [(name1, '1')],
                      (name1, '1'): [(name2, '1'), (name2, '1')],
                      (name2, '1'): [(name3, '1')]}

(synaptic_time_constant[(name2, name3)],
 synaptic_time_constant[(name3, name1)],
 synaptic_time_constant[(name1, name2)],
 synaptic_time_constant[(name1, name1)]) = [10], [10], [10], [10]


lim_n_cycle = [6, 10]
pop_list = [1]
nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, Act[state_1], Act[state_2], name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    Act[state_1], Act[state_2], D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)


run(receiving_class_dict, t_list, dt, nuclei_dict)
                                      
fig = plot(nuclei_dict,color_dict, dt, t_list, Act[state_1], Act[state_2], t_mvt, D_mvt, 
                 plot_end = t_sim, include_FR = False, plot_start = plot_start,
                 title_fontsize = 15, title = 'Transient Oscillation', continuous_firing_base_lines = False,
                 vspan = True)

name = 'FSI'
nucleus = nuclei_dict[name][0]
fig.tight_layout()
state = 'Transient'
# figname = 'FSI-Proto-D2 loop without Proto-Proto_' + state
# fig.savefig(os.path.join(path_rate, figname+'.png'),dpi = 300)
# fig.savefig(os.path.join(path_rate, figname+'.pdf'),dpi = 300)
n_half_cycles, last_first_peak_ratio, perc, freq, if_stable = find_freq_of_pop_act_spec_window(nucleus, *duration_mvt, dt, peak_threshold=nucleus.oscil_peak_threshold, 
                                                                                          cut_plateau_epsilon = 0.001,
                                                                                          smooth_kern_window=nucleus.smooth_kern_window, check_stability=True, plot_oscil = True)
print( "n_half_cycles = {0}, last_first_peak_ratio = {1} , \n \
      freq = {2}, if_stable = {3}".format( n_half_cycles, last_first_peak_ratio , freq, if_stable))
      
# %% RATE MODEL : Any Pallidostriatal loop with GPe-GPe fft

plt.close('all')
N_sim = 100
N = dict.fromkeys(N, N_sim)

if_plot = False
dt = 0.2
t_sim = 21000
t_list = np.arange(int(t_sim/dt))
t_start = 1000
D_mvt = 0
t_mvt = 400
duration = [int((t_start)/dt), int((t_sim)/dt)]
plot_start = t_start
plot_duration = 600
n_windows =10
name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI'
name_list = [name1, name2, name3]

g = - np.power( 15 , 1/3 )
n = 15
g_P_list = - np.linspace(0, 5, n, endpoint=True)
g_FSI_list = - np.linspace(0., 3, n, endpoint=True)
g = g_FSI_list[12]
G = {
    (name2, name3): g,
    (name3, name1): g/2,
    (name1, name2): g,
    (name1, name1): g_P_list[14]
    
}

state_1 = 'awake_rest'
# state_1 = 'rest'
state_2 = 'mvt'

receiving_pop_list = {(name3, '1'): [(name1, '1')],
                      (name1, '1'): [(name2, '1'), (name1, '1')],
                      (name2, '1'): [(name3, '1')]}

(synaptic_time_constant[(name2, name3)],
 synaptic_time_constant[(name3, name1)],
 synaptic_time_constant[(name1, name2)],
 synaptic_time_constant[(name1, name1)]) = [10], [10], [10], [10]


lim_n_cycle = [6, 10]
pop_list = [1]

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, change_states= False)

nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)

fig = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,  plot_start=t_sim - plot_duration, plt_mvt =False, plot_end = t_sim ) 

peak_threshold = 0.1
smooth_window_ms = 3
smooth_window_ms = 5
cut_plateau_epsilon = 0.1
lim_oscil_perc = 10
low_pass_filter = False
check_stability = False
freq_method = 'fft'; plot_sig = True
_,_,_ =find_freq_all_nuclei(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold,
                              smooth_kern_window, smooth_window_ms,
                              cut_plateau_epsilon, check_stability, freq_method, plot_sig,
                              low_pass_filter, 1, 100, plot_spectrum=True, 
                              ax=None, c_spec=color_dict, spec_figsize=(6, 5), find_beta_band_power=False,
                              fft_method='Welch', n_windows=n_windows, include_beta_band_in_legend=True, smooth = False, 
                              normalize_spec = True, include_peak_f_in_legend = True,
                              check_significance = True, plot_sig_thresh = False, plot_peak_sig = False,
                              min_f = 0, max_f = 200, n_std_thresh = 2,AUC_ratio_thresh = 0.8,print_AUC_ratio = True)
      
# %% RATE MODEL : Any Pallidostriatal loop without GPe-GPe

plt.close('all')
N_sim = 100
N = dict.fromkeys(N, N_sim)

if_plot = False
dt = 0.1
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = 1000
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]
plot_start = t_mvt - 200
plot_duration = 600

name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI'
name_list = [name1, name2, name3]

g = - np.power( 15 , 1/3 )
G = {
    (name2, name3): g,
    (name3, name1): g,
    (name1, name2): g
}

state_1 = 'awake_rest'
state_2 = 'mvt'


receiving_pop_list = {(name3, '1'): [(name1, '1')],
                      (name1, '1'): [(name2, '1')],
                      (name2, '1'): [(name3, '1')]}

(synaptic_time_constant[(name2, name3)],
 synaptic_time_constant[(name3, name1)],
 synaptic_time_constant[(name1, name2)]) = [10], [10], [10]


lim_n_cycle = [6, 10]
pop_list = [1]
nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, Act[state_1], Act[state_2], name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    Act[state_1], Act[state_2], D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)


run(receiving_class_dict, t_list, dt, nuclei_dict)
                                      
fig = plot(nuclei_dict,color_dict, dt, t_list, Act[state_1], Act[state_2], t_mvt, D_mvt, 
                 plot_end = t_sim, include_FR = False, plot_start = plot_start,
                 title_fontsize = 15, title = 'Transient Oscillation', continuous_firing_base_lines = False,
                 vspan = True)

name = 'FSI'
nucleus = nuclei_dict[name][0]
fig.tight_layout()
state = 'Transient'
# figname = 'FSI-Proto-D2 loop without Proto-Proto_' + state
# fig.savefig(os.path.join(path_rate, figname+'.png'),dpi = 300)
# fig.savefig(os.path.join(path_rate, figname+'.pdf'),dpi = 300)
n_half_cycles, last_first_peak_ratio, perc, freq, if_stable = find_freq_of_pop_act_spec_window(nucleus, *duration_mvt, dt, peak_threshold=nucleus.oscil_peak_threshold, 
                                                                                          cut_plateau_epsilon = 0.001,
                                                                                          smooth_kern_window=nucleus.smooth_kern_window, check_stability=True, plot_oscil = True)
print( "n_half_cycles = {0}, last_first_peak_ratio = {1} , \n \
      freq = {2}, if_stable = {3}".format( n_half_cycles, last_first_peak_ratio , freq, if_stable))

# %% RATE MODEL : Any Pallidostriatal tau-sweep

N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 20500
t_list = np.arange(int(t_sim/dt))
t_mvt = 500
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]


name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI'
name3 = 'Arky'

state_1 = 'awake_rest'
state_2 = 'mvt'

# state = 'rest'

name_list = {name1, name2, name3}
receiving_pop_list = {(name3, '1'): [(name1, '1')],
                      (name1, '1'): [(name2, '1')],
                      (name2, '1'): [(name3, '1')]}

synaptic_time_constant = {(name2, name3): [10],
                          (name3, name1): [10],
                          (name1, name2): [10]
                          }

lim_n_cycle = [6, 10]
pop_list = [1]

G = {
    (name2, name3): 1,
    (name3, name1): 1,
    (name1, name2): 1
}


G_ratio_dict = {
    (name2, name3): 1,
    (name3, name1): 1,
    (name1, name2): 1
}

# G_ratio_dict = {   #### Shouldn't matter, right?
#                 ('D2', 'Arky') : 0.2,
#                 ('Arky', 'Proto') : 1,
#                 ('Proto', 'D2'): 0.5
#                 }


nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, change_states = False)



n = 20
syn_decay_dict = {'tau_1': {
    'tau_ratio': {(name2, name3): 1,
                  (name3, name1): 1,
                  (name1, name2): 1},
    'tau_list': np.linspace(5, 25, n)
},
    'tau_2': {
    'tau_ratio': {('Proto', 'Proto'): 1},
    'tau_list': [6]}
}  # np.linspace(5,15,n)}}

filename = ('Tau_sweep_D2-P-' + name3[0] + '_tau_ratio_' + name3[0] + name2[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name2, name3)]) + '_' + name1[0] + name3[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name3, name1)]) + '_' + name2[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name1, name2)]) + '_G_ratio_' + name3[0] + name2[0] + '_' +
            str(abs(G_ratio_dict[(name2, name3)])) + '_' + name3[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name3, name1)])) + '_' + name2[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name1, name2)])) + '_n_' +
            str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.', '-'))

filename = os.path.join(path_rate,
                        filename.replace('.', '-') + '.pkl'
                        )

g_list = np.linspace(-5, -0.8, 260)
find_stable_oscill = True  # to find stable oscillatory regime

sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, N, N_real, K_real, syn_decay_dict, filename,
                     G.copy(), Act[state_1], Act[state_2], D_mvt, t_mvt, receiving_class_dict, t_list, dt, duration_base, duration_mvt,
                     lim_n_cycle, if_track_tau_2=False, change_states = False)
# %% RATE MODEL : Any Pallidostriatal without GPe-GPe G-sweep
plt.close('all')
N_sim = 100
N = dict.fromkeys(N, N_sim)

if_plot = False
dt = 0.1
t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = 700
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]
plot_start_trans = t_mvt - 200
plot_duration = 600
plot_start_stable = 0

name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI'
# name3 = 'Arky'

state_1 = 'awake_rest'
state_2 = 'mvt'

name_list = {name1, name2, name3} 

# G_list = np.linspace(-5.8, 0, n, endpoint = True)

transition_range = np.array([2.68760245, 2.82014575])# FSI Loop dt = 0.1
width = (transition_range[1] - transition_range[0])
transition_range = np.array([2.445 - width , 2.445])# FSI Loop dt = 0.1
transition_range = np.array([2.2 , 2.445])# FSI Loop dt = 0.1

# transition_range = [1.903 - (1.915 - 1.903), 1.915]  # Arky Loop dt = 0.1

# G_list = pad_high_res_spacing_with_linspace(0, transition_range[0], 20, transition_range[1], 5.8,  3, 3, base = 1.1)
# G_list = pad_high_res_spacing_with_arange(
#     1, transition_range[0], 1/25, transition_range[1], 5.8,  1/8, 22, base = 1.05)
G_list = pad_high_res_spacing_with_arange(
    0, transition_range[0], 1/20, transition_range[1], 6,  1/5, 10, base = 1.05)

G_list = - np.power(abs(G_list), 1/3)
n = len(G_list)
G_list = G_list[[40, n-1]]
receiving_pop_list = {(name3, '1'): [(name1, '1')],
                      (name1, '1'): [(name2, '1')],
                      (name2, '1'): [(name3, '1')]}

(synaptic_time_constant[(name2, name3)],
 synaptic_time_constant[(name3, name1)],
 synaptic_time_constant[(name1, name2)]) = [10], [10], [10]


G = {
    (name2, name3): -1,
    (name3, name1): -1,
    (name1, name2): -1
}


G_ratio_dict = {
    (name2, name3): 1,
    (name3, name1): 1,
    (name1, name2): 1
}

# G_ratio_dict = {   #### Shouldn't matter, right?
#                 ('D2', 'Arky') : 0.2,
#                 ('Arky', 'Proto') : 1,
#                 ('Proto', 'D2'): 0.5
#                 }


lim_n_cycle = [6, 10]
pop_list = [1]
nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, Act[state_1], Act[state_2], name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    Act[state_1], Act[state_2], D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)


filename = ('G_sweep_D2-P-' + name3[0] + '_tau_' + name3[0] + name2[0] + '_' +
            str(synaptic_time_constant[(name2, name3)][0]) + '_' + name3[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name3, name1)][0]) + '_' + name2[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name1, name2)][0]) + '_G_ratio_' + name3[0] + name2[0] + '_' +
            str(abs(G_ratio_dict[(name2, name3)])) + '_' + name3[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name3, name1)])) + '_' + name2[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name1, name2)])) + '_n_' +
            str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.', '-') + '_SP_2-8')

filename = os.path.join(path_rate,
                        filename.replace('.', '-') + '.pkl'
                        )
receiving_class_dict, nuclei_dict = set_connec_ext_inp(path,
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)
# nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, plot_start=0,title_fontsize=15,
#       title = r"$G_{SP}="+str(round(G[('Proto', 'STN')],2))+"$ "+", $G_{PS}=G_{PP}="+str(round(G[('STN', 'Proto')],2))+'$')

fig_unconnected, fig_trans, fig_stable = synaptic_weight_exploration_RM( N, N_real, K_real,
                                                        G.copy(), Act[state_1], Act[state_2], D_mvt, t_mvt, t_list, dt, filename,
                                                        lim_n_cycle, G_list, nuclei_dict, duration_mvt, duration_base,
                                                          receiving_class_dict, color_dict, G_ratio_dict=G_ratio_dict,
                                                           legend_loc='upper right', plot_start_trans=plot_start_trans,
                                                          plot_start_stable=plot_start_stable, plot_duration=plot_duration, 
                                                          if_plot = True, loop_name = 'FSI-Loop', legend_fontsize_rec= 18, 
                                                          path = os.path.join(path_rate, 'FSI-Loop_anim_g_6'), ylim  = [-4, 62])


# fig_trans , fig_stable = set_ylim_trans_stable_figs(fig_trans, fig_stable, ymax = [100, 100], ymin = [-4, -4])
# save_trans_stable_figs([fig_unconnected, fig_trans, fig_stable], 
#                        ['unconnected', 'transient', 'stable'], 
#                        path_rate, filename.split('.')[
#                        0], figsize=(5, 3), ymax=[75, 75, 75], ymin=[-4, -4, -4])

data = load_pickle(filename)

name = 'Proto'
color = 'last_first_peak_ratio_mvt'  # mvt_f'
param = 'last_first_peak_ratio_mvt'
# param = 'perc_oscil_mvt'

xlabel = r'$G_{Loop}$'


fig = scatter_2d_plot(np.squeeze(abs(Product_G(data))),
                      np.squeeze(data[(name, param)]),
                      np.squeeze(data[(name, color)]),
                      name + ' in ' + name3 + ' Loop',
                      [xlabel, param, color])


plt.axvline(data['g_loop_stable'], c='k')
plt.axvline(data['g_loop_transient'], c='grey', linestyle='--')
save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size=(8, 6))
# save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size = (8,6))
# %% RATE MODEL : Any Pallidostriatal with GPe-GPe G-sweep

n = 15
if_plot = False
dt = 0.5
t_sim = 1500
t_list = np.arange(int(t_sim/dt))
t_mvt = 700
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]

name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI'
# name3 = 'Arky'

state_1 = 'awake_rest'
state_2 = 'mvt'

name_list = {name1, name2, name3}

g_1_list = np.linspace(-2, -1, n, endpoint=True)
g_2_list = np.linspace(-2, -1, n, endpoint=True)

receiving_pop_list = {(name3, '1'): [(name1, '1')],
                      (name1, '1'): [(name2, '1'), (name1, '1')],
                      (name2, '1'): [(name3, '1')]}

(synaptic_time_constant[(name2, name3)],
 synaptic_time_constant[(name3, name1)],
 synaptic_time_constant[(name1, name1)],
 synaptic_time_constant[(name1, name2)]) = [15], [6], [10], [10]

lim_n_cycle = [6, 10]
pop_list = [1]

G = {
    (name2, name3): 1,
    (name3, name1): 1,
    (name1, name1): 1,
    (name1, name2): 1
}


G_ratio_dict = {
    (name2, name3): 1,
    (name3, name1): 1,
    (name1, name1): 1,
    (name1, name2): 1
}

# G_ratio_dict = {   #### Shouldn't matter, right?
#                 ('D2', 'Arky') : 0.2,
#                 ('Arky', 'Proto') : 1,
#                 (name1 , name1) : 1,
#                 ('Proto', 'D2'): 0.5
#                 }


G_dict = {
    ('Proto', 'Proto'): g_1_list,
    (name2, name3): g_2_list
}


nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)


filename = ('G_sweep_D2-P-' + name3[0] + '_tau_' + name3[0] + name2[0] + '_' +
            str(synaptic_time_constant[(name2, name3)][0]) + '_' + name3[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name3, name1)][0]) + '_' + name2[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name1, name2)][0]) + '_' + name1[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name1, name1)][0]) + '_G_ratio_' + name3[0] + name2[0] + '_' +
            str(G_ratio_dict[(name2, name3)]) + '_' + name3[0] + name1[0] + '_' +
            str(G_ratio_dict[(name3, name1)]) + '_' + name2[0] + name1[0] + '_' +
            str(G_ratio_dict[(name1, name2)]) + '_' + name1[0] + name1[0] + '_' +
            str(G_ratio_dict[(name1, name1)]) + '_n_' +
            str(n))

filename = os.path.join(path_rate,
                        filename.replace('.', '-') + '.pkl'
                        )

fig_trans, fig_stable = synaptic_weight_exploration_RM(G.copy(), Act[state_1], Act[state_2], D_mvt, t_mvt, t_list, dt,  filename,
                                                          lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base,
                                                          receiving_class_dict, color_dict, G_ratio_dict=G_ratio_dict,
                                                          if_plot=if_plot, plt_start=500)
pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'D2'
color = 'n_half_cycles_mvt'  # mvt_f'
param = 'mvt_freq'

xlabel = r'$G_{Loop}$'


fig = scatter_2d_plot(np.squeeze(abs(Product_G(data))),
                      np.squeeze(data[(name, param)]),
                      np.squeeze(data[(name, color)]),
                      name + ' in ' + name3 + ' Loop',
                      [xlabel, param, color])


plt.axvline(data['g_loop_stable'], c='k')
plt.axvline(data['g_loop_transient'], c='grey', linestyle='--')
save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size=(8, 6))

# %% RATE MODEL : Pallidostriatal + STN-Proto Phase space

plt.close('all')
N_sim = 100
N = dict.fromkeys(N, N_sim)

if_plot = False
dt = 0.1
t_sim = 4500
t_list = np.arange(int(t_sim/dt))
t_mvt = 500
D_mvt = t_sim - t_mvt
duration = [1500, int(t_sim/dt)]


name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI'
# name3 = 'Arky'
name4 = 'STN'

state_1 = 'awake_rest'
state_2 = 'mvt'

name_list = {name1, name2, name3, name4} 


receiving_pop_list = {(name3, '1'): [(name1, '1')],
                      (name1, '1'): [(name2, '1'), (name4, '1')],
                      (name2, '1'): [(name3, '1')],
                      (name4, '1'): [(name1, '1')]}

(synaptic_time_constant[(name2, name3)],
 synaptic_time_constant[(name3, name1)],
 synaptic_time_constant[(name1, name2)],
 synaptic_time_constant[(name4, name1)],
 synaptic_time_constant[(name1, name4)]) = [10], [10], [10], [10], [6]


G = {
    (name2, name3): -1,
    (name3, name1): -1,
    (name1, name2): -1,
    (name4, name1): -1, 
    (name1, name4): 1
}

g_STN_list = - np.linspace(0.5, 3, 15, endpoint=True)
g_FSI_list = - np.linspace(0.5, 3, 15, endpoint=True)


G_ratio_dict = {
    
    (name2, name3): g_FSI_list,
    (name3, name1): g_FSI_list,
    (name1, name2): g_FSI_list,
    (name4, name1): g_STN_list, 
    (name1, name4): -g_STN_list
}


loop_key_lists = [ 
                    [(name2, name3),
                     (name3, name1),
                     (name1, name2)], 
                  
                    [(name4, name1), 
                     (name1, name4)]]

lim_n_cycle = [6, 10]
pop_list = [1]
nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, Act[state_1], Act[state_2], name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}


receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, A, A_mvt, D_mvt, t_mvt, dt, N, 
                                                       N_real, K_real, receiving_pop_list, nuclei_dict, 
                                                       t_list, change_states = False)
filename = os.path.join(path_rate, 'RM_parameterscape_Proto-STN-0-5_3_Proto-D2_0-5_3_2d_n_15.pkl')
figs = synaptic_weight_exploration_RM_2d(N, N_real, K_real, G, Act[state_1], Act[state_2], D_mvt,
                                         t_mvt, t_list, dt, filename, 
                                         loop_key_lists, nuclei_dict, duration, receiving_class_dict, 
                                         color_dict, plot_firing = False, G_ratio_dict = G_ratio_dict, 
                                         legend_loc = 'upper left', vspan_stable = False, path = path_rate, 
                                         legend_fontsize = 12, ylim  = [-4, 76],
                                         legend_fontsize_rec= 18, n_windows = 3, check_stability = True,
                                         lower_freq_cut = 8, upper_freq_cut = 60, freq_method = 'fft',
                                         peak_threshold= 0.25, smooth_kern_window=3, smooth_window_ms = 5,
                                         cut_plateau_epsilon=0.1, plot_sig = False, low_pass_filter = False,
                                         lim_oscil_perc=10, plot_spectrum = False, normalize_spec = True,
                                         plot_sig_thresh = False, min_f = 8, max_f = 60, n_std_thresh = 2,
                                         save_pxx = True, len_f_pxx = 150, AUC_ratio_thresh = 0.1, plot_peak_sig = False,
                                         change_states = False, save_pop_act = True, print_AUC_ratio = False)


# %% RATE MODEL : Pallidostriatal + Proto-Proto Phase space

plt.close('all')
N_sim = 100
N = dict.fromkeys(N, N_sim)

if_plot = False
dt = 0.1
t_start = 1000
t_sim = 20000 + t_start
t_list = np.arange(int(t_sim/dt))
t_mvt = 0
D_mvt = t_sim - t_mvt
duration = [int(t_start/dt), int(t_sim/dt)]
# duration = [0, int(t_sim/dt)]

n_windows = 10# (t_sim - t_start) / 2000

name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI'

state_1 = 'rest'
state_1 = 'awake_rest'

state_2 = 'mvt'

name_list = {name1, name2, name3} 

receiving_pop_list = {(name3, '1'): [(name1, '1')],
                      (name1, '1'): [(name2, '1'), (name1, '1')],
                      (name2, '1'): [(name3, '1')]}

(synaptic_time_constant[(name2, name3)],
 synaptic_time_constant[(name3, name1)],
 synaptic_time_constant[(name1, name2)],
 synaptic_time_constant[(name1, name1)]) = [10], [10], [10], [10]


G = {
    (name2, name3): 1,
    # (name3, name1): 1/3, # rest state
    (name3, name1): 1/2,
    (name1, name2): 1,
    (name1, name1): 1, 
}
n = 18; n_run = 1
# g_P_list = - np.linspace(1, 4.3, n, endpoint=True) # Useless attempt to set up in rest state
# g_FSI_list = - np.linspace(0.5, 4, n, endpoint=True)

g_P_list = - np.linspace(0, 5, n, endpoint=True)
g_FSI_list = - np.linspace(0., 3, n, endpoint=True)

# g_P_list = np.array([-4.3])
# g_FSI_list = np.array([-3])
# n = len(g_P_list)

G_dict = {
    
    (name2, name3): g_FSI_list,
    (name3, name1): g_FSI_list,
    (name1, name2): g_FSI_list,
    (name1, name1): g_P_list}


loop_key_lists = [[(name1, name1)],
                  [(name2, name3),
                   (name3, name1),
                   (name1, name2)]]

lim_n_cycle = [6, 10]
pop_list = [1]
nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state_1], noise_amplitude,
                              N, Act[state_1], Act[state_2], name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list,
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}


receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, A, A_mvt, D_mvt, t_mvt, dt, N, 
                                                       N_real, K_real, receiving_pop_list, nuclei_dict, 
                                                       t_list, change_states = False)
filename = os.path.join(path_rate, 'RM_parameterscape_Proto-Proto-0_5_Proto-D2_0_3_2d_n_' + str(n) + '.pkl')

figs = synaptic_weight_exploration_RM_2d(N, N_real, K_real, G, Act[state_1], Act[state_2], D_mvt,
                                         t_mvt, t_list, dt, filename, 
                                         loop_key_lists, nuclei_dict, duration, receiving_class_dict, 
                                         color_dict, plot_firing = False, G_dict = G_dict, n_run = n_run,
                                         legend_loc = 'upper left', vspan_stable = False, path = path_rate, 
                                         legend_fontsize = 12, ylim  = [-4, 76],
                                         legend_fontsize_rec= 18, n_windows = n_windows, check_stability = True,
                                         lower_freq_cut = 8, upper_freq_cut = 60, freq_method = 'fft',
                                         peak_threshold= 0.7, ### Note that the plot is x10^2
                                         smooth_kern_window=3, smooth_window_ms = 5, fft_method = 'Welch',
                                         cut_plateau_epsilon=0.1, plot_sig = False, low_pass_filter = False,
                                         lim_oscil_perc=10, plot_spectrum = False, normalize_spec = False,
                                         plot_sig_thresh = False, min_f = 0, max_f = 200, n_std_thresh = 2,
                                         save_pxx = True, len_f_pxx = 250, AUC_ratio_thresh = 0.09, plot_peak_sig = False,
                                         change_states = False, save_pop_act = True, print_AUC_ratio = True)
# %% Parameterscape RATE MODEL

plt.close('all')

filename = 'RM_parameterscape_Proto-Proto-0_5_Proto-D2_0_3_2d_n_18.pkl'



examples_ind = {'a' : (3, 15),
                'b' : (11, 16),
                'c': (14, 5),
                'd': (13, 16),
                # 'E':(11, 0),
                # 'F':(14, 12),
                
                }
# examples_ind = {'A' : (2, 2), 'B': (4, 6) , 
#                 'C': (8, 5),  'D': (0, 9),
#                 }
# examples_ind = {'A' : (0, 0), 'B': (2, 4) , 
#                 'C': (3, 4), 'D': (2, 2), 'E':(4,3)}
# examples_ind = {'A' : (0, 0), 'B': (0, 1) , 
#                 'C': (1, 0), 'D': (1, 1)}
# examples_ind = {'A' : (0, 0)}
dt = 0.1
key_y = ('Proto', 'D2')
ylabel = r'$|G_{Proto-D2}|$'

key_x = ('Proto', 'STN')
name_list = ['D2', 'FSI', 'Proto', 'STN']
xlabel = r'$|G_{Proto-STN}|$'

key_x = ('Proto', 'Proto')
name_list = ['D2', 'FSI', 'Proto']
xlabel = r'$|G_{Proto-Proto}|$'


markerstyle_list = ['o', 'o', 's', 'o']
size_list = np.array([4, 2, 1, 0.5]) * 1000

filepath = os.path.join(path_rate, filename)



data  = load_pickle(filepath)
x_list = abs(data['g'][key_x])
y_list = abs(data['g'][key_y])    
n_x = len(x_list)
n_y = len(y_list)
run = 0

power_dict = {name: np.zeros((len(x_list), len(y_list))) for name in name_list}
freq_dict = {name: np.zeros((len(x_list), len(y_list))) for name in name_list}
f_peak_sig_dict = {name: np.zeros((len(x_list), len(y_list))) for name in name_list}


param = 'frequency (Hz)'

for name in name_list:  ####### one single run (hence the squeeze)
    power_dict[name] = np.squeeze( data[name, 'base_beta_power'] )
    freq_dict[name] = np.squeeze( data[name, 'base_freq'] )
    f_peak_sig_dict[name] =  data[name, 'peak_significance']


freq_dict, f_peak_sig_dict, last_first_peak_ratio = eval_averaged_PSD_peak_significance(data, x_list, y_list, name_list, 
                                                                                        AUC_ratio_thresh = 0,
                                                                                        examples_ind = examples_ind, n_ref_peak = 60)


colormap = freq_dict

# fig = parameterscape(x_list, y_list, name_list, markerstyle_list, freq_dict, colormap, f_peak_sig_dict, 
#                     size_list, xlabel, ylabel, label_fontsize = 30, clb_title = param, 
#                     annotate = False, ann_name='Proto',  tick_size = 18, only_significant = True)
# fig = highlight_example_pts(fig, examples_ind, x_list, y_list, size_list, highlight_color = 'grey')
# save_pdf_png(fig, filepath.split('.')[0] + '_' + param, size = (.8 * n_x + 1.5, .8 * n_y))

fig = parameterscape_imshow(x_list, y_list, name_list, markerstyle_list, freq_dict, colormap, f_peak_sig_dict, 
                    size_list, xlabel, ylabel, label_fontsize = 36, clb_title = param, 
                    x_ticks = [0, 1, 2, 3, 4, 5], y_ticks = [0, 1, 2, 3],
                    annotate = False, ann_name='Proto',  tick_size = 26, only_significant = True, figsize = (8,7))
fig = highlight_example_pts(fig, examples_ind, x_list, y_list, [2000], highlight_color = 'w', annotate_shift = 0.2)
save_pdf_png(fig, filepath.split('.')[0] + '_' + param, size = (0.4 * n_x + 2, 0.4 * n_y ))


fig_exmp = plot_pop_act_and_PSD_of_example_pts_RM(data, name_list, examples_ind, x_list, y_list, dt,
                                                  color_dict, Act, plt_duration = 400, run_no = 0, run = run, act_ylim = (-2, 95),
                                                  PSD_y_labels = [0,  1.5], act_y_labels = [0, 90], normalize_spec = False,
                                                  PSD_x_labels = [0, 20, 40, 60, 80], act_x_labels = [0, 200, 400], 
                                                  PSD_ylim = (-0.1, 1.5), state = 'awake_rest',
                                                  last_first_peak_ratio =last_first_peak_ratio, unit_variance_PSD = True,  PSD_duration = 200000)
save_pdf_png(fig_exmp, filepath.split('.')[0] + '_' + param + '_details', size = (5, 6))



# %% RATE MODEL : frequency vs. G - multiple tau-ratios
g_cte_ind = [0, 0, 0]
g_ch_ind = [1, 1, 1]
filename_list = [os.path.join(path_rate, 'data_synaptic_weight_Pallidostriatal.pkl'),
                 os.path.join(
                     path_rate, 'data_synaptic_weight_Pallidostriatal_30_10_10.pkl'),
                 os.path.join(path_rate, 'data_synaptic_weight_Pallidostriatal_30_6_6.pkl')]
nucleus_name_list = ['Proto', 'Proto', 'Proto']
legend_list = [r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=10$ $\tau_{Proto-D2}=6$',
               r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=10$ $\tau_{Proto-D2}=10$',
               r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=6$ $\tau_{Proto-D2}=6$']
color_list = ['k', 'r', 'g']
param_list = 3*['mvt_freq']
color_param_list = 3 * ['perc_t_oscil_mvt']
x_label = r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$'
synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list,
                                             color_list, g_cte_ind, g_ch_ind, param_list, color_param_list,
                                             'jet', x_axis='g_2', x_label=x_label)
# %% RATE MODEL : frequency vs. G - all circuits
filename_list = ['G_sweep_Proto-Proto_tau_10_n_50_T_15000.pkl',
                 'G_sweep_STN-GPe_tau_PS_10_SP_6_n_50_T_15000.pkl',
                 'G_sweep_D2-P-A_tau_AD_10_AP_10_DP_10_G_ratio_AD_1_AP_1_DP_1_n_40_T_15000.pkl',
                 'G_sweep_D2-P-F_tau_FD_10_FP_10_DP_10_G_ratio_FD_1_FP_1_DP_1_n_40_T_15000.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

n = len(filename_list)
nucleus_name_list = n * ['Proto']
legend_list = [r'$Proto-Proto$', r'$STN-Proto$',
               r'$FSI-D2-Proto$', r'$Arky-D2-Proto$']
color_list = [color_dict['Proto'], color_dict['STN'],
              color_dict['FSI'], color_dict['Arky']]
color_param_list = n * ['perc_t_oscil_mvt']
y_list = n * ['mvt_freq']
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list,
                                                   color_list, y_list,  marker_c_list=color_param_list,  colormap='YlOrBr',
                                                   x_axis='multiply', title="", markersize=50, alpha_transient=0.3,
                                                   x_label=r"$ \vert G_{Loop} \vert$",  leg_loc='upper left', g_key=None,
                                                   vline_txt=False, colorbar=True, ylabel='frequency(Hz)',
                                                   vline_width=2, lw=3, xlim=None)
save_pdf_png(fig, 'Freg_vs_G_Oscil_per_as_color_all_circuits', size=(10, 3))
# %% RATE MODEL : % Oscillation vs. G ( FSI loop)

nucleus_name_list = ['Proto']
n = len(nucleus_name_list)

filename_list = n * \
    ['G_sweep_D2-P-F_tau_FD_10_FP_10_DP_10_G_ratio_FD_1_FP_1_DP_1_n_40_T_25000_dt_0-1_SP_2-8.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{D2-FSI}=\tau_{FSI-Proto}=\tau_{Proto-D2}=10 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['FSI']]
param_list = n * ['perc_t_oscil_mvt']
x_label = r'$ \lvert G_{Loop} \lvert$'
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list,
                                                   color_list, param_list, colormap='hot', colorbar=False, marker_c_list=None,
                                                   x_axis='multiply', ylabel='% Oscillation', title=title, x_label=x_label,
                                                   vline_txt=False, markersize=50)
save_pdf_png(fig, filename_list[0].split('.')[0], size=(5, 3))
# %% RATE MODEL : % Oscillation vs. G ( Arky loop )

nucleus_name_list = ['Proto']
n = len(nucleus_name_list)

filename_list = n * \
    ['G_sweep_D2-P-A_tau_AD_10_AP_10_DP_10_G_ratio_AD_1_AP_1_DP_1_n_40_T_20000.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{D2-Arky}=\tau_{Arky-Proto}=\tau_{Proto-D2}=10 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['Arky']]
param_list = n * ['perc_t_oscil_mvt']
x_label = r'$ \lvert G_{Loop} \rvert$'
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list,
                                                   color_list, param_list, colormap='hot', colorbar=False, marker_c_list=None,
                                                   x_axis='multiply', ylabel='% Oscillation', title=title, x_label=x_label,
                                                   vline_txt=False, markersize=50)
save_pdf_png(fig, filename_list[0].split('.')[0], size=(5, 3))
# %% RATE MODEL : % Oscillation vs. G ( STN-GP )

nucleus_name_list = ['Proto']
n = len(nucleus_name_list)

filename_list = n * ['G_sweep_STN-GPe_tau_PS_10_SP_6_n_72_T_25000_dt_0-1_SP_2-8.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{STN-Proto}=10\: ms \; , \tau_{Proto-STN}= 6 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['STN']]
param_list = n * ['last_first_peak_ratio_mvt']
x_label = r'$ \lvert G_{Loop} \lvert$'
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list,
                                                   color_list, param_list, colormap='hot', colorbar=False, marker_c_list=None,
                                                   x_axis='multiply', ylabel='Amplitude ratio', title=title, x_label=x_label,
                                                   vline_txt=False, markersize=50, mark_pts_with= 'arrow')
fig = set_y_ticks(fig, [0, 0.5, 1])
save_pdf_png(fig, filename_list[0].split('.')[0], size=(5, 3))

# %% RATE MODEL : Amplitude ratio vs. G  save animation



nucleus_name_list = ['Proto']
n = len(nucleus_name_list)

filename_list = ['G_sweep_STN-GPe_tau_PS_10_SP_6_n_72_T_25000_dt_0-1_SP_2-8.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{STN-Proto}=10\: ms \; , \tau_{Proto-STN}= 6 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['STN']]
folder_name, anim_name =  'STN-GP_anim' , 'STN-Proto_amplitude_ratio.mp4'

filename_list = n * \
    ['G_sweep_D2-P-F_tau_FD_10_FP_10_DP_10_G_ratio_FD_1_FP_1_DP_1_n_114_T_2000_dt_0-1_SP_2-8.pkl']
folder_name, anim_name = 'FSI-Loop_anim' , 'FSI-Loop_amplitude_ratio.mp4'
lim_f_0 = 6


filename_list = n * \
    ['G_sweep_D2-P-F_tau_FD_10_FP_10_DP_10_G_ratio_FD_1_FP_1_DP_1_n_70_T_2000_dt_0-1_SP_2-8.pkl']
folder_name, anim_name = 'FSI-Loop_anim_g_6' , 'FSI-Loop_amplitude_ratio_g_6.mp4'
lim_f_0 = 3


filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{D2-FSI}=\tau_{FSI-Proto}=\tau_{Proto-D2}=10 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['FSI']]




param_list = n * ['last_first_peak_ratio_mvt']
x_label = r'$ \lvert G_{Loop} \lvert$'

data = load_pickle(filename_list[0])
g = abs(Product_G(data))
r = np.squeeze(data[('Proto','last_first_peak_ratio_mvt')])

fig = plt.figure()

  
# labeling the x-axis and y-axis
axis = plt.axes(xlim=(0.7, 6.5),  ylim=(-.1, 1.2))
axis.set_xlabel(r'$| G_{Loop}|$',fontsize = 20)
axis.set_ylabel('Amplitude ratio',fontsize=20)
axis.set_title(title,fontsize=20)
axis.tick_params(axis='x', length = 10)
axis.tick_params(axis='y', length = 8)
axis.tick_params(axis='both', labelsize=15)
fig = set_y_ticks(fig, [0, 0.5, 1])
remove_frame(axis)
# lists storing x and y values
x, y = [], []
  
ax, = axis.plot(0, 0, '-o')
  
fig.tight_layout()
def animate(frame_number):
    x.append(g[frame_number])
    y.append(r[frame_number])
    ax.set_xdata(x)
    ax.set_ydata(y)
    ax.set_color(color_list[0])
    
    m = (abs(g[-1]) - lim_f_0) / len(g)
    axis.set_xlim(0, m * frame_number + lim_f_0)

    return ax,
  
  
anim = animation.FuncAnimation(fig, animate, frames=len(r), 
                               interval=int( 1/10 * 1000), blit=True,repeat=False)
# fig.canvas.draw()
# anim.event_source.stop()  
# saving to gif using ffmpeg writer
writer = animation.FFMpegWriter(fps= 8)
# writer=animation.PillowWriter(fps=12)
anim.save( os.path.join(path_rate, folder_name, anim_name ), writer=writer)
plt.close()


# from moviepy.editor import *

# clip = VideoFileClip(os.path.join(path_rate, 'STN-GP_anim' , 'STN-Proto_amplitude_ratio.mp4'))
# clip.write_gif(os.path.join(path_rate, 'STN-GP_anim' , 'STN-Proto_amplitude_ratio.gif'))

# %% RATE MODEL : Amplitude ratio vs. G  save pdf



nucleus_name_list = ['Proto']
n = len(nucleus_name_list)

filename_list = ['G_sweep_STN-GPe_tau_PS_10_SP_6_n_72_T_25000_dt_0-1_SP_2-8.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{STN-Proto}=10\: ms \; , \tau_{Proto-STN}= 6 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['STN']]
folder_name, anim_name =  'STN-GP_anim' , 'STN-Proto_amplitude_ratio.mp4'

filename_list = n * \
    ['G_sweep_D2-P-F_tau_FD_10_FP_10_DP_10_G_ratio_FD_1_FP_1_DP_1_n_114_T_2000_dt_0-1_SP_2-8.pkl']
folder_name, anim_name = 'FSI-Loop_anim' , 'FSI-Loop_amplitude_ratio.mp4'
lim_f_0 = 6


filename_list = n * \
    ['G_sweep_D2-P-F_tau_FD_10_FP_10_DP_10_G_ratio_FD_1_FP_1_DP_1_n_70_T_2000_dt_0-1_SP_2-8.pkl']
folder_name, anim_name = 'FSI-Loop_anim_g_6' , 'FSI-Loop_amplitude_ratio_g_6.mp4'
lim_f_0 = 3


filepath_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{D2-FSI}=\tau_{FSI-Proto}=\tau_{Proto-D2}=10 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['FSI']]




param_list = n * ['last_first_peak_ratio_mvt']
x_label = r'$ \lvert G_{Loop} \lvert$'

data = load_pickle(filepath_list[0])
g = abs(Product_G(data))
r = np.squeeze(data[('Proto','last_first_peak_ratio_mvt')])

fig, ax = plt.subplots()

  
ax.plot(g, r, '-o', c = color_dict['FSI'])
# labeling the x-axis and y-axis
ax.set_xlim(0.7, 6.5)
ax.set_ylim(-.1, 1.2)
ax.set_xlabel(r'$| G_{Loop}|$', fontsize = 25)
ax.set_ylabel('Amplitude ratio', fontsize = 25)
ax.set_title(title,fontsize=20)
ax.tick_params(axis='x', length = 12)
ax.tick_params(axis='y', length = 12)
ax.tick_params(axis='both', labelsize=20)
fig = set_y_ticks(fig, [0, 0.5, 1])
remove_frame(ax)
save_pdf_png(fig, os.path.join(path_rate, filename_list[0]),
              size=(5,6))

# %% RATE MODEL : % Oscillation vs. G ( GP-GP )

nucleus_name_list = ['Proto']
n = len(nucleus_name_list)

filename_list = n * ['G_sweep_Proto-Proto_tau_10_n_50_T_20000.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{Proto-Proto}=10 \: ms$'
title = ''
legend_list = ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['Proto']]
param_list = n * ['perc_t_oscil_mvt']
x_label = r'$ \lvert G_{Loop} \lvert$'
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list,
                                                   color_list, param_list, colormap='hot', colorbar=False, marker_c_list=None,
                                                   x_axis='multiply', ylabel='% Oscillation', title=title, x_label=x_label,
                                                   vline_txt=False, markersize=50)
save_pdf_png(fig, filename_list[0].split('.')[0], size=(5, 3))
# %% RATE MODEL : frequency vs. G ( STN-GP and GP-GP ) - multiple GP-GP GP-STN G-ratios

filename_list = ['data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_2_N_100_T_800_dt_0-1.pkl',
                 # STN weight is half the inhibition
                 'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_1_N_100_T_800_dt_0-1.pkl',
                 'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_0-5_N_100_T_800_dt_0-1.pkl']
figname = 'STN_GPe_with_GP_GP_f_vs_tau_inh_different_P_proj_G_ratios'  # '_STN_one'
filename_list = [os.path.join(
    path_rate, 'STN_weight_constant', filename) for filename in filename_list]

filename_list = ['data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_2_N_100_T_800_dt_0-1_STN_1.pkl',
                 # STN weight is equal the inhibition
                 'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_1_N_100_T_800_dt_0-1_STN_1.pkl',
                 'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_0-5_N_100_T_800_dt_0-1_STN_1.pkl']
figname = 'STN_GPe_with_GP_GP_f_vs_tau_inh_different_P_proj_G_ratios_STN_one'
filename_list = [os.path.join(
    path_rate, 'STN_weight_constant', filename) for filename in filename_list]

filename_list = ['data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_2_N_100_T_800_dt_0-1_STN_changing.pkl',
                 # STN weight is equal the inhibition
                 'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_1_N_100_T_800_dt_0-1_STN_changing.pkl',
                 'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_0-5_N_100_T_800_dt_0-1_STN_changing.pkl']
figname = 'STN_GPe_with_GP_GP_f_vs_tau_inh_different_P_proj_G_ratios_STN_changing'

filename_list = ['Tau_sweep_STN-GPe-GPe_tau_ratio_SP_1_PP_1_PS_1_G_ratio_SP_2_PP_1_PS_1_n_30_T_10000.pkl',
                 'Tau_sweep_STN-GPe-GPe_tau_ratio_SP_1_PP_1_PS_1_G_ratio_SP_1_PP_1_PS_1_n_30_T_10000.pkl',
                 'Tau_sweep_STN-GPe-GPe_tau_ratio_SP_1_PP_1_PS_1_G_ratio_SP_1_PP_2_PS_1_n_30_T_10000.pkl']

pkl_file = open(os.path.join(path_rate, filename_list[0]), 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

filename_list = [os.path.join(path_rate, filename)
                 for filename in filename_list]
n = len(filename_list)
x_label = r'$\tau_{decay}^{inhibition}(ms)$'
y_label = 'frequency(Hz)'
c_label = y_label
label_list = [r'$G_{SP}=2\times G_{PP}$',
              r'$G_{SP}=G_{PP}$', r'$G_{SP}=\dfrac{G_{PP}}{2}$']
key_list = n * [('STN', 'Proto')]
title = ''

name_list = n * ['Proto']


c_list = n * ['stable_mvt_freq']
y_list = c_list
colormap = 'Y'
title = ''
c_label = 'frequency (Hz)'
# fig = multi_plot_as_f_of_timescale_shared_colorbar(y_list, color_list,c_list,  label_list,name_list,filename_list,x_label,y_label,colormap = 'YlOrBr',
# c_label = c_label,ylabelpad = 0, g_ratio_list=g_ratio_list, g_tau_2_ind = g_tau_2_ind, title = title)
# plt.ylim(28, 59)
# plt.xlim(4, 32)


markerstyle = ['s', 'o', '^']
color_list = ['k', 'maroon', 'red']
fig, ax2 = plt.subplots(1, 1, sharex=True, figsize=(6, 5))
i = 0


def plot__(ax):
    for i in range(len(filename_list)):
        # i = 0; filename_list[i] = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec = data['tau'][key_list[i]][:, 0]
        print(data[(name_list[i], y_list[i])].shape)
        y_spec = data[(name_list[i], y_list[i])][:, 0]
        c_spec = data[(name_list[i], c_list[i])][:, 0]
        # ax.plot(x_spec,y_spec, marker = markerstyle[i], c = color_list[i], lw = 1, label= label_list[i],zorder = 1, mec = color_list[i])
        ax.scatter(x_spec, y_spec,
                   marker=markerstyle[i], c=color_list[i], label=label_list[i])


plot__(ax2)


ax2.set_xlabel(r'$\tau_{decay}^{inhibition}$', fontsize=20)
ax2.set_ylabel('Frequency (Hz)',  fontsize=20)

# fig.text( -0.01, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize = 18)
y_formatter = FixedFormatter(['40', '55',   '70'])
y_locator = FixedLocator([40, 55,  70])
ax2.yaxis.set_major_formatter(y_formatter)
ax2.yaxis.set_major_locator(y_locator)

remove_frame(ax2)
ax2.set_xlim(4, 26)
ax2.set_ylim(38, 70)
ax2.legend(fontsize=12, frameon=False, framealpha=0.1,
           bbox_to_anchor=(.5, 0.85), bbox_transform=ax2.transAxes)
save_pdf_png(fig, os.path.join(
    path_rate, 'F_vs_tau_Multiple_g_ratios_STN-GPe-GPe'), size=(5, 6))

# %% RATE MODEL : frequency vs. tau_inhibition (FSI and Arky loops) - multiple tau ratios


g_tau_2_ind = 0

filename_list = ['data_Arky_D2_Proto_syn_t_scale_tau_3_1_1.pkl',
                 'data_Arky_D2_Proto_syn_t_scale_tau_2_1_1.pkl',
                 'data_Arky_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{3}$',
              r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{2}$', r'$\tau_{PA}=\tau_{DP}=\tau_{AD}$']
figname = 'Arky-D2-Proto_timescale_g_stable'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$'
y_label = 'frequency(Hz)'
c_label = y_label
title = ''
name_list = ['Proto']*3
y_list = ['stable_mvt_freq']*3
# color_list = ['k','grey','lightgrey']
fig, ax = plt.subplots(1, 1)
color_list = create_color_map(
    len(filename_list) + 1, colormap=plt.get_cmap('Oranges'))
color_list = color_list[::-1]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label,
                                       g_tau_2_ind=None, ylabelpad=-5, title='', c_label='', ax=ax)


filename_list = ['data_FSI_D2_Proto_syn_t_scale_tau_3_1_1.pkl',
                 'data_FSI_D2_Proto_syn_t_scale_tau_2_1_1.pkl',
                 'data_FSI_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{PF}=\tau_{DP}=\dfrac{\tau_{FD}}{3}$',
              r'$\tau_{PF}=\tau_{DP}=\dfrac{\tau_{FD}}{2}$', r'$\tau_{PF}=\tau_{DP}=\tau_{FD}$']
figname = 'FSI-D2-Proto_timescale_tau_g_stable'
x_label = r'$\tau_{PF/DP}^{decay}(ms)$'
y_label = 'frequency(Hz)'
c_label = y_label
title = ''

color_list = create_color_map(
    len(filename_list) + 1, colormap=plt.get_cmap('Greens'))
color_list = color_list[::-1]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label,
                                       g_tau_2_ind=None, ylabelpad=-5, title='', c_label='', ax=ax)
ax.legend(fontsize=10)

fig.savefig(figname+'.png', dpi=300)
fig.savefig(figname+'.pdf', dpi=300)

# %% RATE MODEL : frequency vs. tau_inhibition (Arky Loop)

filename_list = [
    'Tau_sweep_D2-P-A_tau_ratio_AD_1_PA_1_DP_1_G_ratio_AD_1_AP_1_DP_1_n_30']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{AP}=\tau_{PD}=\tau_{DA}$']
figname = 'test'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$'
y_label = 'frequency (Hz)'
c_label = y_label
title = ''
name_list = ['Proto']
y_list = ['stable_mvt_freq']
fig, ax = plt.subplots(1, 1)
color_list = create_color_map(
    len(filename_list) + 1, colormap=plt.get_cmap('Oranges'))
color_list = color_list[::-1]
color_list = [color_dict['Arky']]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label,
                                       tau_2_ind=0, ylabelpad=-5, title='', c_label='', ax=ax, key=('Proto', 'D2'))

# %% RATE MODEL : frequency vs. tau_inhibition (FSI Loop)

filename_list = [
    'Tau_sweep_D2-P-F_tau_ratio_FD_1_PF_1_DP_1_G_ratio_FD_1_FP_1_DP_1_n_30.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{FP}=\tau_{PD}=\tau_{DF}$']
figname = 'test'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$'
y_label = 'frequency (Hz)'
c_label = y_label
title = ''
name_list = ['Proto']
y_list = ['stable_mvt_freq']
fig, ax = plt.subplots(1, 1)
# color_list = create_color_map(len(filename_list) + 1, colormap = plt.get_cmap('Oranges'))
# color_list = color_list[::-1]
color_list = [color_dict['FSI']]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label,
                                       tau_2_ind=0, ylabelpad=-5, title='', c_label='', ax=ax, key=('Proto', 'D2'))

# %% RATE MODEL : frequency vs. tau_inhibition (GPe-GPe Loop)

filename_list = [
    'Tau_sweep_STN-GPe_tau_ratio_PS_1_SP_1_G_ratio_PS_1_SP_1_n_30.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = ['']
figname = 'test'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$'
y_label = 'frequency (Hz)'
c_label = y_label
title = ''
name_list = ['Proto']
y_list = ['stable_mvt_freq']
fig, ax = plt.subplots(1, 1)
color_list = [color_dict['Proto']]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label,
                                       tau_2_ind=0, ylabelpad=-5, title='', c_label='', ax=ax, key=('Proto', 'Proto'))

# %% RATE MODEL : frequency vs. tau_inhibition (STN-GPe Loop)

filename_list = [
    'Tau_sweep_STN-GPe_tau_ratio_PS_1_SP_1_G_ratio_PS_1_SP_1_n_30.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{PS}=6 \; ms$']
figname = 'test'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$'
y_label = 'frequency (Hz)'
c_label = y_label
title = ''
name_list = ['Proto']
y_list = ['stable_mvt_freq']
fig, ax = plt.subplots(1, 1)
color_list = [color_dict['STN']]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label,
                                       tau_2_ind=0, ylabelpad=-5, title='', c_label='', ax=ax, key=('STN', 'Proto'))
# %% RATE MODEL : frequency vs. tau_inhibition (All Loops) new
plt.close('all')
filename_list = ['Tau_sweep_GPe-GPe_tau_ratio_PP_1_PP_1_n_20_T_20500_dt_0-1.pkl',
                 'Tau_sweep_STN-GPe_tau_ratio_PS_1_SP_1_G_ratio_PS_1_SP_1_n_30_T_20500_SP_2-8_dt_0-1.pkl',
                 'Tau_sweep_D2-P-F_tau_ratio_FD_1_PF_1_DP_1_G_ratio_FD_1_FP_1_DP_1_n_20_T_20500_dt_0-1.pkl',
                 'Tau_sweep_D2-P-A_tau_ratio_AD_1_PA_1_DP_1_G_ratio_AD_1_AP_1_DP_1_n_20_T_20500_dt_0-1.pkl']


filename_list = [os.path.join(path_rate, file) for file in filename_list]
figname = 'All_circuits_timescale'
label_list = ['Proto-Proto', 'STN-Proto',  'FSI-D2-Proto', 'Arky-D2-Proto']
g_tau_2_ind = 0
color_list = create_color_map(
    len(filename_list), colormap=plt.get_cmap('viridis'))
color_list = [color_dict['Proto'], color_dict['STN'],
              color_dict['FSI'], color_dict['Arky']]
key_list = [('Proto', 'Proto'), ('STN', 'Proto'),
            ('Proto', 'D2'), ('Proto', 'D2')]
c_list = ['stable_freq'] * len(filename_list)

y_list = c_list
colormap = 'hot'
title = ''
c_label = 'frequency (Hz)'
name_list = ['Proto'] * len(filename_list)
markerstyle = ['+', 's', 'o', '^']
fig, ax2 = plt.subplots(1, 1, sharex=True, figsize=(6, 5))
i = 1


def plot__(ax):
    for i in range(len(filename_list)):
        # i = 0; filename_list[i] = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec = data['tau'][key_list[i]][:, 0]
        print(data[(name_list[i], y_list[i])].shape)
        y_spec = data[(name_list[i], y_list[i])][:, 0]
        c_spec = data[(name_list[i], c_list[i])][:, 0]
        # ax.plot(x_spec,y_spec, marker = 's', c = color_list[i], lw = 1, label= label_list[i],zorder = 1, mec = 'k')
        ax.scatter(x_spec, y_spec, marker='o',
                   c=color_list[i], lw=0.2, label=label_list[i], zorder=1, s=40)  # ,  ec = 'k')


plot__(ax2)


ax2.set_xlabel(r'$\tau_{decay}^{inhibition}$', fontsize=20)
ax2.set_ylabel('Frequency (Hz)',  fontsize=20)

# fig.text( -0.01, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize = 18)

fig = set_y_ticks(fig, [5, 20, 40, 60,  80])
fig = set_x_ticks(fig, [5, 15, 25])

ax2.tick_params(axis='both', labelsize=22, length = 8)

remove_frame(ax2)
ax2.set_xlim(4, 26)
ax2.set_ylim(5, 72)
leg = ax2.legend(fontsize=18, frameon=False, framealpha=0.1,
           bbox_to_anchor=(.3, 0.25), bbox_transform=ax2.transAxes)
ax2.axhspan(13, 30, color='lightgrey', alpha=0.5, zorder=0)


for i, text in enumerate(leg.get_texts()):
    text.set_color(color_list[i])

save_pdf_png(fig, os.path.join(path_rate, figname), size=(5, 6))
# %% RATE MODEL : G vs. tau_inhibition (All Loops)
plt.close('all')
filename_list = ['Tau_sweep_GPe-GPe_tau_ratio_PP_1_PP_1_n_30_T_10000.pkl',
                 'Tau_sweep_STN-GPe_tau_ratio_PS_1_SP_1_G_ratio_PS_1_SP_1_n_30_T_10000.pkl',
                 'Tau_sweep_D2-P-F_tau_ratio_FD_1_PF_1_DP_1_G_ratio_FD_1_FP_1_DP_1_n_30_T_10000.pkl',
                 'Tau_sweep_D2-P-A_tau_ratio_AD_1_PA_1_DP_1_G_ratio_AD_1_AP_1_DP_1_n_30_T_10000.pkl']

filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = ['Proto-Proto', 'STN-Proto',  'FSI-D2-Proto', 'Arky-D2-Proto']
g_tau_2_ind = 0
color_list = create_color_map(
    len(filename_list), colormap=plt.get_cmap('viridis'))
color_list = [color_dict['Proto'], color_dict['STN'],
              color_dict['FSI'], color_dict['Arky']]
key_list = [('Proto', 'Proto'), ('STN', 'Proto'),
            ('Proto', 'D2'), ('Proto', 'D2')]
c_list = ['stable_mvt_freq'] * len(filename_list)
y_list = c_list
colormap = 'hot'
title = ''
c_label = 'frequency (Hz)'
name_list = ['Proto'] * len(filename_list)
markerstyle = ['+', 's', 'o', '^']
fig, ax2 = plt.subplots(1, 1, sharex=True, figsize=(6, 5))
i = 1


def get_g_stable_loop(data):
    g_stables = {k: v for k, v in data.items() if k[1] == 'g_stable'}
    n = len(list(g_stables.values())[0])
    gg = np.ones((n, 1))
    for k, v in g_stables.items():
        gg = gg * v
    return abs(gg)


def plot__(ax, y=None):
    for i in range(len(filename_list)):
        # i = 0; filename_list[i] = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec = data['tau'][key_list[i]][:, 0]
        y_spec = get_g_stable_loop(data)
        c_spec = data[(name_list[i], c_list[i])][:, 0]
        ax.scatter(x_spec, y_spec, marker='s',
                   c=color_list[i], lw=0.2, label=label_list[i], zorder=1, s=20,  ec='k')


plot__(ax2, y='g_stable')
figname = 'All_circuits_G_vs_tau_inhibition'
ax2.set_ylabel(r'$|G_{Loop}|$',  fontsize=20)
ax2.set_xlabel(r'$\tau_{decay}^{inhibition}$', fontsize=20)
y_formatter = FixedFormatter(['0', '2', '4', '6',  '8', '10'])
y_locator = FixedLocator([0, 2, 4, 6, 8, 10])
ax2.yaxis.set_major_formatter(y_formatter)
ax2.yaxis.set_major_locator(y_locator)
ax2.set_xlim(4, 26)
ax2.set_ylim(0, 10)
remove_frame(ax2)
ax2.legend(fontsize=12, frameon=False, framealpha=0.1,
           bbox_to_anchor=(.5, 0.75), bbox_transform=ax2.transAxes)
save_pdf_png(fig, os.path.join(path_rate, figname), size=(5, 6))

# %% RATE MODEL : frequency vs. tau_inhibition (all loops)
# 3 The idiot that I am saved the pickles without instruction.
# Here I guess the excitatory time scale is set to 6 while changing the inhibition decay time

plt.close('all')
# filename_list = ['data_STN_GPe_syn_t_scale_g_ratio_1.pkl','data_STN_GPe_without_GP_GP_syn_t_scale_g_ratio_1.pkl',
# 'data_FSI_D2_Proto_syn_t_scale_G_ratios_1_1_0-5.pkl','data_Arky_D2_Proto_syn_t_scale_G_ratios_0-2_1_0-5.pkl']
filename_list = [os.path.join(path_rate, 'data_STN_GPe_with_GP_GP_syn_t_scale_N_100_T_800_dt_0-1_K_scaled.pkl'),
                 os.path.join(
                     path_rate, 'data_STN_GPe_without_GP_GP_syn_t_scale_g_ratio_1_N_100_T_800_dt_0-1_STN_chaning_npop_2.pkl'),
                 os.path.join(
                     path_rate, 'data_FSI_D2_Proto_syn_t_scale_tau_1_1_1.pkl'),
                 os.path.join(path_rate, 'data_Arky_D2_Proto_syn_t_scale_tau_1_1_1.pkl')]
figname = 'All_circuits_timescale'
label_list = [r'$STN-Proto + Proto-Proto$',
              r'$STN-Proto$',  r'$FSI-D2-Proto$', r'$Arky-D2-Proto$']
g_tau_2_ind = 0
color_list = create_color_map(
    len(filename_list), colormap=plt.get_cmap('viridis'))
color_list = ['plum', 'lightcoral', 'green', 'orange']
c_list = ['stable_mvt_freq'] * len(filename_list)
y_list = c_list
colormap = 'hot'
title = ''
c_label = 'frequency (Hz)'
name_list = ['Proto'] * len(filename_list)
markerstyle = ['+', 's', 'o', '^']
fig, ax2 = plt.subplots(1, 1, sharex=True, figsize=(6, 5))


def plot__(ax):
    for i in range(len(filename_list)):
        # i = 0; filename_list[i] = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec = data['tau'][:, :, 0][:, 0]
        print(data[(name_list[i], y_list[i])].shape)
        y_spec = data[(name_list[i], y_list[i])][:, g_tau_2_ind]
        c_spec = data[(name_list[i], c_list[i])][:, g_tau_2_ind]
        ax.plot(x_spec, y_spec,
                marker=markerstyle[i], c=color_list[i], lw=1, label=label_list[i], zorder=1)


plot__(ax2)
# 3 Broken axis
# f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# # plot the same data on both axes
# plot__(ax)
# plot__(ax2)

# # zoom-in / limit the view to different portions of the data
# ax.set_ylim(30, 80)  # outliers only
# ax2.set_ylim(5, 20)  # most of the data

# # hide the spines between ax and ax2
# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.xaxis.tick_top()
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()


# d = 0.015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# # What's cool about this is that now if we vary the distance between
# # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# # the diagonal lines will move accordingly, and stay right at the tips
# # of the spines they are 'breaking'
# ######################


ax2.set_xlabel(r'$\tau_{decay}^{inhibition}$', fontsize=20)
ax2.set_ylabel('Frequency (Hz)',  fontsize=20)

# fig.text( -0.01, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize = 18)
ax_label_adjust(ax2, fontsize=20, nbins=5, ybins=6)
remove_frame(ax2)
ax2.set_xlim(4, 32)
ax2.set_ylim(10, 55)
ax2.legend(fontsize=12, frameon=False, framealpha=0.1,
           bbox_to_anchor=(.4, 0.3), bbox_transform=ax2.transAxes)
ax2.axhspan(13, 30, color='lightgrey', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(path_rate, ('All_circuits_plus_STN_GP_without_GP_GP_Freq_vs_tau_STN_GPe_GPe_Gs_scaled_with_K.png')), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True)  # ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path_rate, ('All_circuits_plus_STN_GP_without_GP_GP_Freq_vs_tau_STN_GPe_GPe_Gs_scaled_with_K.pdf')), dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', transparent=True, bbox_inches="tight", pad_inches=0.1)

# %% RATE MODEL : <PARAMETER> vs. G
# All circuits
filename_list = ['data_STN_GPe_syn_t_scale_g_ratio_1.pkl',
                 'data_FSI_D2_Proto_syn_t_scale_G_ratios_1_1_0-5.pkl',
                 'data_Arky_D2_Proto_syn_t_scale_G_ratios_0-2_1_0-5.pkl']
figname = 'All_circuits_timescale'
label_list = [r'$STN-Proto \; with \; Proto-Proto$',
              r'$FSI-D2-Proto$', r'$Arky-D2-Proto$']
g_tau_2_ind = 0
g_ratio_list = [1, 1, 1]
# <<frequency>> multiple time scale ratios

# y_list  = ['stable_mvt_freq']*3
# x_label = r'$\tau_{decay}^{inhibition}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label

# filename_list = ['data_STN_GPe_syn_t_scale_g_ratio_2.pkl', 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl', 'data_STN_GPe_syn_t_scale_g_ratio_0-5.pkl']
# figname = 'STN_GPe_timescale'
# x_label = r'$\tau_{decay}^{inhibition}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label
# label_list = [r'$G_{PS}=2\times G_{PP}$',r'$G_{PS}=G_{PP}$',r'$G_{PS}=\dfrac{G_{PP}}{2}$']
# title = ''
# g_ratio_list = [1,1,1]
# g_tau_2_ind = 3

# filename_list = ['data_Arky_D2_Proto_syn_t_scale_G_ratios_0-1_1_1.pkl','data_Arky_D2_Proto_syn_t_scale_G_ratios_1_1_1.pkl','data_Arky_D2_Proto_syn_t_scale_G_ratios_0-2_2_1.pkl']
# label_list = [r'$G_{PA}=G_{DP}=\dfrac{G_{AD}}{10}$',r'$G_{PA}=G_{DP}=G_{AD}$',r'$G_{PA}=2\times G_{DP}=\dfrac{G_{AD}}{10}$']
# figname = 'Arky-D2-Proto_timescale'
# g_tau_2_ind = 0; g_ratio_list = [1,1,1]


# filename_list = ['data_FSI_D2_Proto_syn_t_scale_G_ratios_2_1_0-5.pkl','data_FSI_D2_Proto_syn_t_scale_G_ratios_1_1_0-5.pkl','data_FSI_D2_Proto_syn_t_scale_G_ratios_1_2_0-5.pkl']
# label_list = [r'$G_{PF}=\dfrac{G_{DP}}{2}=\dfrac{G_{FD}}{4}$',r'$G_{PF}=G_{DP}=2\times G_{FD}$',r'$G_{PF}=2\times G_{DP}=\dfrac{G_{FD}}{2}$']
# figname = 'FSI-D2-Proto_timescale'
# g_tau_2_ind = 0; g_ratio_list = [1,1,1]
# <<g transition>> for multiple time scale or g ratios
# y_list  = ['g_stable']*3

# filename_list = ['data_STN_GPe_syn_t_scale_g_ratio_2.pkl', 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl', 'data_STN_GPe_syn_t_scale_g_ratio_0-5.pkl']
# figname = 'STN_GPe_g_stable'
# x_label = r'$\tau_{decay}^{inhibition}(ms)$' ; y_label = r'$G_{PS}$ at transition' ; c_label = y_label
# label_list = [r'$G_{PS}=2\times G_{PP}$',r'$G_{PS}=G_{PP}$',r'$G_{PS}=\dfrac{G_{PP}}{2}$']
# title = r'$G_{SP}=0.5$'
# g_ratio_list = [2,1,1]
# g_tau_2_ind = 0

# filename_list = ['data_FSI_D2_Proto_syn_t_scale_tau_3_1_1.pkl','data_FSI_D2_Proto_syn_t_scale_tau_2_1_1.pkl','data_FSI_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
# label_list = [r'$\tau_{PF}=\tau_{DP}=\dfrac{\tau_{FD}}{3}$',r'$\tau_{PF}=\tau_{DP}=\dfrac{\tau_{FD}}{2}$',r'$\tau_{PF}=\tau_{DP}=\tau_{FD}$']
# figname = 'FSI-D2-Proto_timescale_tau_g_stable'
# x_label = r'$\tau_{PF/DP}^{decay}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label; title = ''
# g_tau_2_ind = 0

filename_list = ['data_Arky_D2_Proto_syn_t_scale_tau_3_1_1.pkl',
                 'data_Arky_D2_Proto_syn_t_scale_tau_2_1_1.pkl', 'data_Arky_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

label_list = [r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{3}$',
              r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{2}$', r'$\tau_{PA}=\tau_{DP}=\tau_{AD}$']
figname = 'Arky-D2-Proto_timescale_g_stable'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$'
y_label = 'frequency(Hz)'
c_label = y_label
title = ''

# name_list = ['Proto']*3
# g_ratio_list = [1,1,1]
color_list = ['k', 'grey', 'lightgrey']
c_list = ['stable_mvt_freq']*3
y_list = c_list
colormap = 'hot'
title = ''
c_label = 'frequency (Hz)'
# 33
fig = multi_plot_as_f_of_timescale_shared_colorbar(y_list, color_list, c_list,  label_list, name_list, filename_list, x_label, y_label, c_label=c_label,
                                                   ylabelpad=0, g_ratio_list=g_ratio_list, g_tau_2_ind=g_tau_2_ind, title=title)
fig.savefig(figname+'.png', dpi=300)
fig.savefig(figname+'.pdf', dpi=300)
# %% RATE MODEL : time scale space (GABA-a, GABA-b)

receiving_pop_list = {('STN', '1'): [('Proto', '1')], ('STN', '2'): [('Proto', '2')],
                      ('Proto', '1'): [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                      ('Proto', '2'): [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
# receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
#                    ('Proto','1') : [('Proto', '1'), ('STN', '1')],
#                    ('Proto','2') : [('Proto', '2'), ('STN', '2')]}
synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')], synaptic_time_constant[('Proto', 'STN')] = [
    decay_time_scale['GABA-A'], decay_time_scale['GABA-B']], [decay_time_scale['GABA-A'], decay_time_scale['GABA-B']], [decay_time_scale['Glut']]

pop_list = [1, 2]
Proto = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, 'Proto',
                 G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, 'STN',
               G, T, t_sim, dt, synaptic_time_constant,  receiving_pop_list, smooth_kern_window, oscil_peak_threshold)for i in pop_list]
nuclei_dict = {'Proto': Proto, 'STN': STN}

receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real_STN_Proto_diverse, receiving_pop_list, nuclei_dict, t_list)

n = 1
Glut = np.linspace(2, 16, n)
GABA_A = np.linspace(5, 32, n)
GABA_B = np.linspace(150, 310, n)
g_list = np.linspace(-20, -0.01, 100)
g_ratio = 1
lim_n_cycle = [6, 10]
find_stable_oscill = True  # to find stable oscillatory regime
filename = 'data_GABA_A_GABA_B.pkl'
sweep_time_scales_STN_GPe(g_list, g_ratio, nuclei_dict, GABA_A, GABA_B, Glut, filename, G.copy(
), A, A_mvt, D_mvt, t_mvt, receiving_class_dict, t_list, dt, duration_base, duration_mvt, lim_n_cycle, find_stable_oscill)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
name = 'Proto'
color = 'trans_n_half_cycle'
x = data['synaptic_time_constant'][:, :, :, 0].flatten()
y = data['synaptic_time_constant'][:, :, :, 1].flatten()
z = data['synaptic_time_constant'][:, :, :, 2].flatten()
c_trans = data[(name, 'trans_mvt_freq')].flatten()
c_stable = data[(name, 'stable_mvt_freq')].flatten()
c = c_trans
scatter_3d_plot(x, y, z, c, name, np.max(c), np.min(
    c), ['GABA-A', 'GABA-B', 'Glut', 'transient oscillation f'], limits=None)
# %% RATE MODEL : time scale space GABA-B

t_sim = 2000
t_list = np.arange(int(t_sim/dt))
t_mvt = int(t_sim/2)
D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
duration_base = [0, int(t_mvt/dt)]
G[('Proto', 'STN')] = 1
receiving_pop_list = {('STN', '1'): [('Proto', '1')], ('STN', '2'): [('Proto', '2')],
                      ('Proto', '1'): [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                      ('Proto', '2'): [('Proto', '2'), ('STN', '1'), ('STN', '2')]}

synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')], synaptic_time_constant[(
    'Proto', 'STN')] = [decay_time_scale['GABA-B']], [decay_time_scale['GABA-B']], [decay_time_scale['Glut']]

pop_list = [1, 2]
Proto = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, 'Proto',
                 G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, 'STN',
               G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window, oscil_peak_threshold)for i in pop_list]
nuclei_dict = {'Proto': Proto, 'STN': STN}

receiving_class_dict = set_connec_ext_inp(
    A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real_STN_Proto_diverse, receiving_pop_list, nuclei_dict, t_list)

inhibitory_trans = 'GABA_B'
n = 2
Glut = np.linspace(4, 16, n)
GABA_A = np.linspace(5, 32, n)
GABA_B = np.linspace(150, 300, n)
inhibitory_series = GABA_B
g_list = np.linspace(-80, -0.01, 80)
g_ratio = 1
lim_n_cycle = [6, 10]
find_stable_oscill = True  # to find stable oscillatory regime
filename = 'data_'+inhibitory_trans+'.pkl'
sweep_time_scales_one_GABA_STN_GPe(g_list, g_ratio, nuclei_dict, inhibitory_trans, inhibitory_series, Glut, filename, G,
                                   A, A_mvt, D_mvt, t_mvt, receiving_class_dict, t_list, dt, duration_base, duration_mvt, lim_n_cycle, find_stable_oscill)

pkl_file = open(filename, 'rb')
freq = pickle.load(pkl_file)
pkl_file.close()
name = 'Proto'
color = 'trans_n_half_cycle'
x = freq['synaptic_time_constant'][:, :, 0]
y = freq['synaptic_time_constant'][:, :, 1]
z = freq[(name, 'trans_mvt_freq')]
z_stable = freq[(name, 'stable_mvt_freq')]
c = freq[(name, 'trans_n_half_cycle')]
# scatter_3d_wireframe_plot(x,y,z, c, name,[inhibitory_trans,'Glut','freq',color])
scatter_3d_wireframe_plot_2_data_series(x, y, z, 'b', 'lightskyblue', x, y, z_stable, 'g', 'darkgreen', name, [
                                        'transient', 'stable'], [inhibitory_trans, 'Glut', 'freq'])
# %% Scribble


# AUC_of_input,_ = find_AUC_of_input(name,poisson_prop,gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt,
# D_mvt,t_mvt, N_real, K_real,t_list,color_dict, G, T, t_sim, dt, synaptic_time_constant,
# receiving_pop_list, smooth_kern_window,oscil_peak_threshold,if_plot = False)


# model = LinearRegression(fit_intercept=True) ;
# temp = firing_prop[name]['firing_mean'][:,0]
# y = temp[np.where(temp>0)].reshape(-1,1)
# x = FR_list[np.where(temp>0)].reshape(-1,1) * 1000
# reg = model.fit(x,y)
# print('slope=',reg.coef_,'intercept=',reg.intercept_)
# # plt.plot(x,model.predict(x))
# # plt.plot(x,reg.coef_[0]*x)

# def check_freq_detection(nucleus, t_list, dt)
tt = t_list[duration_mvt[0]:duration_mvt[1]]*dt/1000
#N = 1000; T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N, endpoint=False)
#sig1  = np.sin(10.0 * 2.0*np.pi*tt)*np.exp(-10*tt)+100
# indices = np.nonzero((sig1[1:] >= 0) & (sig1[:-1] < 0))[0] # zero crossing
sig2 = STN.pop_act[duration_mvt[0]:duration_mvt[1]]
sig2 = STN.pop_act[np.argmax(
    STN.pop_act[int(t_mvt/dt):int((t_mvt+D_mvt)/dt)])+int(t_mvt/dt):duration_mvt[1]]
sig2 = STN.pop_act[np.argmax(STN.pop_act[0:int(t_mvt/dt)]):int(t_mvt/dt)]
plt.figure()
plt.plot(sig2)
plt.plot(sig2-np.average(sig2))
plt.plot(sig2[cut_plateau(sig2-np.average(sig2))])
max_non_empty_array(cut_plateau(sig2-np.average(sig2)))/len(sig2)
#derivative = np.diff(sig2-np.average(sig2))
#derivative_2 = np.diff(derivative)
#der_avg = moving_average_array(derivative, 10)
# plt.plot(derivative)
# plt.plot(np.diff(derivative))

#peaks, vallies = signal.find_peaks(derivative, prominence = (0.2))

np.max(cut_plateau(sig2-np.average(sig2)))

# plt.plot(sig1[cut_plateau(sig1)])
# plt.plot(sig_filtered)
#fs = 1/(dt/1000)
# [ff,pxx] = signal.welch(sig2,axis=0,nperseg=int(fs),fs=fs)#,nfft=1024)
# plt.plot(ff,pxx)
# ff[np.argmax(pxx)]

freq_from_fft((sig2-np.average(sig2))[[cut_plateau(sig2)]], dt/1000)
# freq_from_welch((sig2[10:]-np.average(sig2[10:]))[[cut_plateau(sig2[10:])]],dt/1000)
# freq_from_fft(sig2[cut_plateau(sig2)],dt)
# freq_from_welch(sig2[cut_plateau(sig2)],dt)
#f1 = fft(sig1)
#f2 = fft(sig2)
# plt.figure(2)
#windowed1 = sig1 * signal.blackmanharris(len(sig1))
#windowed2 = sig2 * signal.blackmanharris(len(sig2))
# plt.plot(f1)
# plt.plot(windowed2)
#
# Find the peak and interpolate to get a more accurate peak
# np.argmax(abs(f1))
# np.argmax(abs(f2))


# %% Skewed normal and beta dist fitting

plt.figure()
a = - 5
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')
x = np.linspace(skewnorm.ppf(0.001, a, loc=20, scale=10),
                skewnorm.ppf(0.999, a, loc=20, scale=10), 100)
rv = skewnorm(a, loc=20, scale=10)
plt.plot(x, rv.pdf(x), lw=2, label='frozen pdf')
plt.hist(skewnorm.rvs(a, loc=-65, scale=90, size=1000))


a = nuc3[0].all_mem_pot.copy()
# y_ = nuc3[0].all_mem_pot[:,400].copy()
y_ = a.reshape(int(a.shape[0] * a.shape[1]), 1)
y = y_[np.logical_and(y_ < 25, y_ > -65)]
param = stats.beta.fit(y)  # , floc=-65, fscale=90)
plt.figure()
plt.hist(y, bins=100, label='whole data')
plt.hist(stats.beta.rvs(*param, size=len(y)),
         bins=100, label='fitted beta distribution')
plt.legend()


# %% checking performance of getting histogram for 2d array


def hist_2D(data, n_bins, range_limits):
    
    ''' histogram of 2d array '''
    
    # Setup bins and determine the bin location for each element for the bins
    R = range_limits
    N = data.shape[-1]
    bins = np.linspace(R[0],R[1],n_bins, endpoint = True)
    data2D = data.reshape(-1,N)
    idx = np.searchsorted(bins, data2D,'right')-1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx==-1) | (idx==n_bins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to 
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins*np.arange(data2D.shape[0])[:,None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins*data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    
    return counts

a = np.array( [ np.random.normal(0, 10, size = 1000) for i in range(5)] )
dist = hist_laxis(a, 100, [-100, 100])                     
bins = np.linspace( -100, 100, 100 , endpoint = True)
fig, ax = plt.subplots(1, 1)

dist_hist, _ = np.histogram(a[1,:], bins = bins)
ax.hist(a[1,:], bins = bins)
ax.bar( bins, dist[1, :])
print(dist_hist.shape, dist[1, :].shape)
print(dist_hist, dist[1, :])
