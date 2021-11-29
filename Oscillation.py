#%% Constants 
import os

root = '/home/shiva/BG_Oscillations'
# root =  r"C:/Users/azizp/BG_Oscillations"
# root = '/Users/apple/BG_Oscillations'

path = os.path.join(root, 'Outputs_SNN')
path_rate = os.path.join(root, 'Outputs_rate_model')

    
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import timeit
from numpy.fft import rfft,fft, fftfreq
from scipy import signal,stats
from tempfile import TemporaryFile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
import pickle
import os
import sys
import pandas as pd
import seaborn as sns
# import patsy
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from Oscillation_module import *

#from scipy.ndimage.filters import generic_filter

N_sim = 1000
N_sub_pop = 2
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
# MSNs make up at least 95% of all striatal cells (Kemp and Powell, 1971)
N_Str = 2.79*10**6 # Oorschot 1998
N_real = { 'STN': 13560 , 'Proto': 46000*0.70, 'Arky':46000*0.25, 'GPi': 3200, 'Str': N_Str, 'D2': int(0.475*N_Str), 'D1': int(0.475*N_Str) , 'FSI': int(0.02*N_Str),  # Oorschot 1998 , FSI-MSN: (Gerfen et al., 2010; Tepper, 2010)
          'Th': 10000} # check to find 
# A = { 'STN': 15 , 'Proto': 30, 'Arky': 18, # De la Crompe (2020) # Why??
A = { 'STN': 15 ,# De la Crompe (2020) [Fig. 6f]
     'Proto': 45, 'Arky': 8, # Mallet et al. 2016, De la Crompe (2020)
         ## Corbit et al.: GPe neurons fired at 24.5 Hz in control and 18.9 Hz in the DD model ( Fig. 3B)(Boraud et al., 2001; Kita and Kita, 2011)
     'FSI': 18.5, # FSI average firing rates:10–15 Hz. 60–80 Hz during behavioral tasks(Berke et al., 2004; Berke, 2008) or 18.5 Hz Berke et al 2010?
             # 21 Corbit et al. from HErnandez et al. 2013
     'D1': 1.1, 'D2': 1.1, #Berke et al. 2010
     'GPi':26} # Benhamou & Cohen (201)
# mean firing rate from experiments
A_DD = { 'STN': 24 ,  # De la Crompe (2020) [Fig. 6f]
        # 'Proto': 38, # De la Crompe (2020), Corbit et al. (2016):18.9+/-0.87 Hz in the DD model ( Fig. 3B)(Boraud et al., 2001; Kita and Kita, 2011)
			'Proto' : 22,  # De la Crompe (2020) [Fig. 4d] (This is for Sep 8th 2021. What was I thinking before?)
			'FSI': 24, # Corbit et al. 2016
			'D1': 6.6, 'D2': 6.6, # Kita & Kita. 2011, Corbit et al. 2016
	         'Arky': 12} # De la Crompe (2020) [Fig. 4f]
A_mvt = { 'STN': 50 , 'Proto': 22,  # Mallet et al. 2016 mean firing rate during movement from experiments
         'FSI': 32,
         'D2': 4, # Mirzaei et al. 2017
         'Arky':38} # Dodson et al. 2015
A_trans = {'STN': 65, 'Proto': A['Proto'], 'D2': 23} # with ctx stimulation
Act = {'rest': A, 'mvt': A_mvt, 'DD': A_DD, 'trans': A_trans}
threshold = { 'STN': .1 ,'Proto': .1, 'D2': .1, 'FSI': .1, 'Arky': 0.1}
neuron_type = {'STN': 'Glut', 'Proto': 'GABA', 'D2': 'GABA', 'FSI':'GABA'}
gain = { 'STN': 1 ,'Proto': 1, 'D2': 1, 'FSI':1, 'Arky': 1}

#K = { ('STN', 'Proto'): 475,
#      ('Proto', 'STN'): 161,
#      ('Proto', 'Proto'): 399}
conductance = {('STN', 'Proto'): 0,
               ('Proto', 'Proto'): 0}
syn_per_ter = {('STN', 'Proto'): int(12/10.6), #number of synapses per bouton = 12/10.6, #Baufreton et al. (2009)
               ('Proto', 'Proto'): 10, #Sadek et al. 2006 
              ('FSI', 'Proto'): 1,
              ('D1', 'FSI'): 1,
              ('D2', 'FSI'): 1,
              ('FSI','FSI'): 1,
              ('Proto', 'D2'): 1,
              ('GPi', 'D1'): 1,
              ('GPi', 'STN'): 1,
              ('GPi', 'Proto'): 1,
              ('D1', 'D2'): 1,
              ('D2', 'D2'): 1} 
Str_connec = {('D1', 'D2'): .28,('D2', 'D2'):.36,  ('D1', 'D1'):.26,  ('D2', 'D1'):.05, ('MSN','MSN'): 1350} # Taverna et al 2008
K_real = { ('STN', 'Proto'): int(243*N_real['Proto']/N_real['STN']),# 243 bouton per GP. number of synapses per bouton = 12/10.6  Baufreton et al. 2009 & Sadek et al. (2006).  
           ('Proto', 'STN'): 135, # boutons Kita & Jaeger (2016) based on Koshimizu et al. (2013)
           ('Proto', 'Proto'): int((264+581)/2),# Sadek et al. 2006 --> lateral 264, medial = 581 boutons mean 10 syn per bouton. 650 boutons Hegeman et al. (2017)
#           ('FSI', 'Proto'): int(800*15*N_real['Proto']/((0.04+0.457)*N_Str)*(0.75/(.75+.10+.54))), #800 boutons per Proto Guzman 2003, 15 contacts per bouton. 75%, 10%, 45%  connected to FSI, D1, NYP
           ('FSI', 'Proto'): 360, # averaging the FSI contacting of Proto boutons Bevan 1998
           # ('D1', 'FSI'): int(36*2/(36+53)*240),#Guzman et al  (2003): 240 from one class interneuron to each MSI # 36% (FSI-D1) Gittis et al.2010
           ('D2', 'FSI'): int(53*2/(36+53)*240),#Guzman et al  (2003): 240 from one class interneuron to each MSI # 53% (FSI-D2) Gittis et al.2010
           # ('FSI','FSI'): int(N_real['FSI']*0.58), # Gittis et al. (2010)
#           ('Proto', 'D2'): int(N_Str/2*(1-np.power(0.13,1/(.1*N_Str/2)))), # Chuhma et al. 2011 --> 10% MSN activation leads to 87% proto activation
           ('Proto', 'D2'): int(N_real['D2']*226/N_real['Proto']), # each Proto 226 from iSPN Kawaguchi et al. (1990)
           ('Proto', 'D1'): int(N_real['D1']*123/N_real['Proto']), # each Proto 123 from dSPN Kawaguchi et al. (1990)
           ('GPi', 'D1'): 1, # find !!!!
           ('GPi', 'STN'): 457, # boutons Kita & Jaeger (2016) based on Koshimizu et al. (2013)
           ('GPi', 'Proto'): 1, # find !!!
           # ('D1', 'D2'): int(Str_connec[('MSN','MSN')]*Str_connec[('D1', 'D2')]/(Str_connec[('D1', 'D2')]+Str_connec[('D2', 'D2')])), #Guzman et al (2003) based on Taverna et al (2008)
           # ('D2', 'D2'): int(Str_connec[('MSN','MSN')]*Str_connec[('D2', 'D2')]/(Str_connec[('D1', 'D2')]+Str_connec[('D2', 'D2')])),
           # ('D2', 'Th'): 1, # find
           # ('FSI', 'Th'): 1, # find
           ('D2','Arky'):100,# estimate
           ('Arky','Proto'):300} # estimate
#           ('D1', 'D1'): Str_connec[('MSN','MSN')]*Str_connec[('D1', 'D1')]/(Str_connec[('D1', 'D1')]+Str_connec[('D2', 'D1')]),  
#           ('D2', 'D1'): Str_connec[('MSN','MSN')]*Str_connec[('D2', 'D1')]/(Str_connec[('D1', 'D1')]+Str_connec[('D2', 'D1')]), #Guzman et al (2003) based on Taverna et al (2008)
#           ('D1', 'Proto'): int(N_real['Proto']*(1-np.power(64/81, 1/N_real['Proto'])))} # Klug et al 2018

K_real_DD = {
           ('D2', 'FSI'): 2*K_real[('D2', 'FSI')],
           # ('FSI','FSI'): int(N_real['FSI']*0.58), # Gittis et al. (2010)
           ('Proto', 'D2'): int(N_real['D2']*226/N_real['Proto']), # each Proto 226 from iSPN Kawaguchi et al. (1990)
           ('FSI', 'Proto'): 360, # averaging the FSI contacting of Proto boutons Bevan 1998
		   ('Proto', 'STN'): K_real[('Proto', 'STN')],
		   ('STN', 'Proto'): 2 * K_real[('STN', 'Proto')],
           ('D2','Arky'): 100,# estimate
           ('Arky','Proto'): 300, # estimate
           ('Proto', 'Proto'): K_real[('Proto', 'Proto')]}
           # ('D1', 'D2'): 0.7*K_real[('D1', 'D2')], #Damodaran et al 2015 based on Taverna et al. 2008
           # ('D2', 'D2'): 0.5*K_real[('D2', 'D2')]} #Damodaran et al 2015 based on Taverna et al. 2008


K_real_STN_Proto_diverse = K_real.copy()
K_real_STN_Proto_diverse[('Proto', 'STN')] = K_real_STN_Proto_diverse[('Proto', 'STN')] / N_sub_pop # because one subpop in STN contacts all subpop in Proto
T = { ('STN', 'Proto'): 4, # Fujimoto & Kita (1993) - [firing rate]
      ('Proto', 'STN'): 2, # kita & Kitai (1991) - [firing rate] ## Ketzef & Silberberg 2020 says 4.5
      ('Proto', 'Proto'): 5,#  Ketzef & Silberberg (2020)- [IPSP]/ or 0.96 ms Bugaysen et al. 2013 [IPSP]?
      ('Arky', 'Proto'): 5,#  Ketzef & Silberberg (2020)- [IPSP]
      ('D2','Arky') : 5, # Glajch et al. 2016 [Fig. 1] .estimate was 7 before Sep 2021. 
      ('Proto', 'D2'):  7.34, #ms proto Ketzef & Silberberg (2020) {in-vitro:striatal photostimulation recording at Proto}- [IPSP] /7ms Kita & Kitai (1991) - [IPSP] [Kita and Kitai 1991 5ms?]
      ('STN', 'Ctx'): 5.5, # kita & Kita (2011) [firing rate]/ Fujimoto & Kita 1993 say an early excitaion of 2.5
#      ('D2', 'Ctx'): 13.4 - 5, # short inhibition latency of MC--> Proto Kita & Kita (2011) - D2-Proto of Kita & Kitai (1991)
      ('D2', 'Ctx'): 10.5, # excitation of MC--> Str Kita & Kita (2011) - [firing rate]
      ('D1', 'Ctx'): 10.5,
      # ('FSI', 'Ctx'): 8/12.5 * 10.5 ,# Kita & Kita (2011) x FSI/MSN latency in SW- Mallet et al. 2005
      ('FSI', 'Ctx') : 7.5, # Based on Fig. 2A of Mallet et. al 2005 (average of MC-stim (80-400 micA))
      ('GPi', 'D1'): 7.2, #  Kita et al. 2001 - [IPSP] / 13.5 (MC-GPi) early inhibition - 10.5 = 3? Kita et al. 2011 
      ('GPi', 'STN'): 1.7, #  STN-EP Nakanishi et al. 1991 [EPSP] /1ms # STN-SNr Nakanishi et al 1987 / 6 - 5.5  (early excitaion latency of MC--> GPi Kita & Kita (2011) - Ctx-STN) - [firing rate]
      ('GPi', 'Proto'): 3, # Kita et al 2001 --> short latency of 2.8 and long latency 5.9 ms [IPSP]/ (4 - 2) ms Nakanishi et al. 1991: the IPSP following the EPSP with STN activation in EP, supposedly being due to STN-Proto-GPi circuit?
      ('Th', 'GPi'): 5, # Xu et al. (2008)
      ('FSI', 'Proto'): 5, # Glajch et al. 2016 [Fig. 2]. estimate was 6 before Sep 2021. 
      ('D1' , 'FSI'): 1, #0.84 ms mice Gittis et al 2010
      ('D2' , 'FSI'): 1, #0.93 ms mice Gittis et al 2010
      ('FSI' , 'FSI'): 1, # estimate based on proximity
      ('Ctx','Th'): 5.6, # Walker et al. (2012)
      ('D1', 'D2'): 1,
      ('D2', 'D2'): 1} 
    # transmission delay in ms
T_DD = {('D2', 'Ctx'): 5.5, # excitation of MC--> Str Kita & Kita (2011)  [firing rate]
        ('D1', 'Ctx'): 5.5,
        ('STN', 'Ctx'): 5.9} # kita & Kita (2011) [firing rate]
G = {('STN', 'Proto'): -1 ,
     ('Proto', 'STN'): .5 , 
     ('Proto', 'Proto'): -1,
     ('Arky', 'Proto'): -1,
     # ('D2', 'Arky'): 0.1,
     # ('D2', 'Ctx'): 0,
     # ('D1', 'Ctx'): 0,
     ('D2','Proto'): 0,
     ('D2', 'FSI'): -1, 
     ('FSI', 'Proto'): -1,
     ('FSI', 'FSI'): -1,
     ('Proto','D2'): -1,
     ('D2','D2'): 0,
     ('D2','D1'): 0,
     ('D1','D2'): 0,
     ('D1', 'D1'): 0,
     ('GPi', 'Proto'): 0,
     ('Th', 'GPi') : 0
     } # synaptic weight
decay_time_scale = {'GABA-A' : 6, 'GABA-B': 200, 'Glut': 5, 'AMPA': 1.8, 'NMDA':51} # Gerstner. synaptic time scale for excitation and inhibition

synaptic_time_constant = {}
# synaptic_time_constant = {('STN', 'Proto'): [10] , ## used for rate model.
#                         ('Proto', 'STN'): [decay_time_scale['Glut']], 
#                         ('Proto', 'Proto'): [10],
#                         ('D2', 'FSI'): [30], 
#                         ('FSI', 'Proto'): [6],
#                         ('FSI', 'FSI'): [6],
#                         ('Proto','D2'): [10],
#                         ('Arky','Proto'): [6],
#                         ('D2', 'Arky'): [30]}


# mean_APth_Proto = ( (-56.6 * 14) + (-49.9*5) ) /(14 + 5)
# SD_APth_Proto = np.sqrt( 
#                         ( (14-1) * 1.8**2 + 14 * (-56.6 - mean_APth_Proto)**2 + 
#                          (5-1) * 2.8**2 + 5 * (-49.9 - mean_APth_Proto) **2 )  / 
#                         (5 + 14 -1 ) 
#                         ) # Abdli et al 2015 Table. 1 : average of PV+ and PV- proto

neuronal_consts = { 
				'Proto': {
					'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': {'mean': -66.3, 'var': 0.8, 'truncmin': -85, 'truncmax': -40}, #Stanford & cooper 2000 [type A] trun bounds of RMP is estimated
                'u_initial':{'min':-66.3, 'max':-37.57}, 
					# 'membrane_time_constant':{'mean':20,'var':1.5},'spike_thresh': {'mean':-37,'var':5}}, # tau_m :Cooper & Stanford 2000 (25) spike_thresh: Karube et al 2019
					# 'membrane_time_constant':{'mean':12.94,'var':2},'spike_thresh': {'mean':-37,'var':5}}, # tau_m :#Projecting to STN from Karube et al 2019
					# 'membrane_time_constant':{'mean':25,'var':1.5},'spike_thresh': {'mean':-37,'var':5}}, # tau_m :Cooper & Stanford 2000 (25) spike_thresh: Karube et al 2019
					'membrane_time_constant':{'mean':43,'var':10, 'truncmin': 2, 'truncmax': 80},'spike_thresh': {'mean':-37.57,'var':4.79}}, # tau_m : Jerome's measurements AP_thresh & tau_m trun bounds: Karube et al. 2019
				'Arky': {
					'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': {'mean': -58.1, 'var': 1.1, 'truncmin': -85, 'truncmax': -50}, # Stanford & cooper 2000 [type B]
                    'u_initial':{'min':-58.1, 'max':-42.9},
					# 'membrane_time_constant':{'mean':19.9,'var':1.6},'spike_thresh': {'mean':-43,'var':0.8}}, # Cooper & Stanford 2000
					'membrane_time_constant':{'mean':36.5,'var':10, 'truncmin': 2, 'truncmax': 80},'spike_thresh': {'mean':-42.9,'var':0.8}}, # tau_m: Jerome . AP_threh : Stanford & Cooper 2000 [type B] trun bounds of RMP is estimated
				'D2': {
 					# 'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': {'mean': -64.47, 'var': 14.25, 'truncmin': -100, 'truncmax': -45}, 'u_initial':{'min':-64.47, 'max': -41.94},  #  Planert et al. 2013 RMP trunc bound estimated
 					'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': {'mean': -64.47, 'var': 1, 'truncmin': -100, 'truncmax': -50},  #  Planert et al. 2013 RMP trunc bound estimated
                    'u_initial':{'min':-64.47, 'max': -41.94}, 

					# 'spike_thresh': {'mean':-55,'var':2}, # Willet et al. 2019
					# 'membrane_time_constant':{'mean':13,'var':1.5}}, # tau_m : Planert et al. 2013
					'membrane_time_constant':{'mean':13.85,'var':6.25, 'truncmin': 2, 'truncmax': 80},'spike_thresh': {'mean':-41.94,'var':3.19}}, #  Planert et al. 2013
				'FSI': {
					'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': {'mean': -75, 'var': 1, 'truncmin': -100, 'truncmax': -65}, #  Russo et al. 2013. RMP trunc bound estimated
                    'u_initial':{'min':-75, 'max':-46}, 
					'membrane_time_constant':{'mean':9.2,'var':0.2, 'truncmin': 2, 'truncmax': 80},'spike_thresh': {'mean':-46,'var':1}}, #  Russo et al 2013
				'STN': {
					'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': {'mean': -59, 'var': 0.4, 'truncmin': -100, 'truncmax': -65},  # Paz et al. 2005
                    'u_initial':{'min':-59, 'max':-50.8},
					'membrane_time_constant':{'mean': 5.13,'var':0.97, 'truncmin': 2, 'truncmax': 80},'spike_thresh': {'mean':-50.8,'var':0.5}}} # Paz et al 2005 
	
tau = {
       ('D2','FSI'):{'rise':[1],'decay':[14]} , # Straub et al. 2016
		('D1','D2'):{'rise':[3],'decay':[35]},# Straub et al. 2016
        ('STN','Proto'): {'rise':[1.1],'decay':[7.8]}, # Baufreton et al. 2009, decay=6.48 Fan et. al 2012
       # ('STN','Proto'): {'rise':[1.1, 40],'decay':[7.8, 200]}, # Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Gertsner
       ('Proto','STN'): {'rise':[0.2],'decay':[6]}, # Glut estimate
       ('Proto','Proto'): {'rise':[0.5, 40],'decay':[4.9, 200]}, # Sims et al. 2008
       ('Proto','D2'): {'rise':[0.8],'decay':[6.13]}, # Sims et al. 2008 ( in thesis it was 2 and 10)
       ('FSI','Proto'): {'rise':[1],'decay':[14.5]} ,# Saunders et al. 2016 (estimate was 6 before Sep 2021) 
       ('Arky', 'Proto') : {'rise':[0.5, 40],'decay':[4.9, 200]}, # Sims et al. 2008
       ('D2','Arky'): {'rise':[4],'decay':[28]} # Jerome now. Before: 65 was measured from Glajch et al. 2016 [Fig. 2]. They report >200ms
      }

syn_coef_GABA_b = 1
syn_component_weight = {
                        ('D2','FSI'):[1],
                        ('STN','Proto'): [1, syn_coef_GABA_b], # the relative weight of the GABA-a and GABA-b components
                        ('Proto','STN'): [1],
                        ('Proto','Proto'): [1, syn_coef_GABA_b],
                        ('Proto','D2'): [1],
                        ('FSI','Proto'): [1],
                        ('Arky', 'Proto') : [1, syn_coef_GABA_b],
                        ('D2','Arky'): [1]
                        }

syn_component_weight = {key: [1] for key in list(tau.keys())}
tau_DD = {('STN','Proto'): {'rise':[0.1],'decay':[7.78]}} # Fan et. al 2012}
G_DD = {
      ('D2','Proto'): G[('D2','Proto')]*108/28} # IPSP amplitude in Ctr: 28pA, in DD: 108pA Corbit et al. (2016) [Is it due to increased connections or increased synaptic gain?]

color_dict = {'Proto' : 'r', 'STN': 'k', 'D2': 'b', 'FSI': 'g','Arky':'darkorange'}
I_ext_range = {'Proto' : [0.175*100, 0.185*100], 
               'STN': [0.012009 * 100, 0.02 * 100], 
               'D2': [0.0595*100 / 3 , 0.0605*100 / 3] , 
               'FSI': [0.05948*100, 0.0605*100], 
               'Arky': []}

FR_ext_range = {'Proto': {'rest': [4/300, 9/300], 'DD':[1.8/300, 4.5/300], 'mvt':[1.8/300, 5/300]},
                'STN': {'rest': [5.5/300, 7.3/300],  'DD':[6/300, 7.5/300], 'mvt':[7.2/300, 8.8/300]},
                'D2': {'rest': [0.2/300 , 1.5/300] ,  'DD':[0.2/300 , 3/300], 'mvt':[0.8/300 , 2.5/300]}, #u_rest variance ~15
                'FSI': {'rest': [ 8/300, 10.2/300 ],  'DD': [ 8.7/300, 10.2/300 ], 'mvt':[9.3/300, 10.7/300]},
                'Arky': {'rest': [0.8/300, 1.8/300], 'DD':[0.9/300, 2/300], 'mvt':[2/300, 3.3/300]}}
FR_ext_range ['D2'] = {'rest': [1.8/300 , 3.8/300] ,  'DD':[2.9/300 , 4.5/300], 'mvt':[2.5/300 , 4.3/300]}#u_rest variance ~0.1 to 1
FR_ext_range ['STN'] = {'rest': [6.5/300, 8/300],  'DD':[6/300, 7.5/300], 'mvt':[7.2/300, 8.8/300]} ### without noise

noise_variance = {'Proto' : 250, 
                  'STN': 30, 
                  'D2': 2 , 
                  'FSI': 10, 
                  'Arky': 6}

noise_amplitude = {'Proto' : 1, 'STN': 1, 'D2': 1, 'FSI': 1, 'Arky': 1}


########33 OU
noise_variance = {'Proto' : 50000, 
                  'STN': 1000,
                  'FSI': 15000, 
                  'D2': 5000}


FR_ext_range = {'Proto': {'rest': np.array([12/1000, 20/1000]), 'DD':[1.8/300, 4.5/300], 'mvt':[1.8/300, 5/300]},
                'STN': {'rest': np.array([18/1000, 25/1000]),  'DD':[6/300, 7.5/300], 'mvt':[7.2/300, 8.8/300]},
                'FSI': {'rest': np.array([2/1000, 8/1000]),  'DD':[6/300, 7.5/300], 'mvt':[7.2/300, 8.8/300]},
                'D2': {'rest': np.array([0.01/1000, 12/1000]),  'DD':[6/300, 7.5/300], 'mvt':[7.2/300, 8.8/300]}}

############3

end_of_nonlinearity = {
                  'FSI': { 'rest' : 10 , 'mvt': 20, 'DD': 35 } ,
                  'D2':  { 'rest' : 20 , 'mvt': 20 , 'DD': 20, 'trans': 40 } ,
                  'Proto':  { 'rest' :  20, 'mvt': 20 , 'DD': 20  },
                  'STN': { 'rest' : 35 , 'mvt': 25, 'DD': 20, 'trans': 35 }, 
                  'Arky':  { 'rest' :  35, 'mvt': 25 , 'DD': 10  }}

oscil_peak_threshold = {'Proto' : 0.1, 'STN': 0.1, 'D2': 0.1, 'FSI': 0.1, 'Arky': 0.1}

smooth_kern_window = {key: value * 30 for key, value in noise_variance.items()}
#oscil_peak_threshold = {key: (gain[key]*noise_amplitude[key]*noise_variance[key]-threshold[key])/5 for key in noise_variance.keys()}
pert_val = 10
mvt_selective_ext_input_dict = {('Proto','1') : pert_val, ('Proto','2') : -pert_val,
                                ('STN','1') : pert_val, ('STN','2') : -pert_val} # external input coming from Ctx and Str
dopamine_percentage = 100
t_sim = 400 # simulation time in ms
t_mvt = int(t_sim/2)
D_mvt = t_sim - t_mvt
D_perturb = 250 # transient selective perturbation
d_Str = 200 # duration of external input to Str
#    duration_mvt = [int((t_mvt+ max(T[('Proto', 'D2')],T[('STN', 'Ctx')]))/dt), int((t_mvt+D_mvt)/dt)]
#    duration_base = [int((max(T[('Proto', 'STN')],T[('STN', 'Proto')]))/dt), int(t_mvt/dt)]

#ext_inp_delay = {'Proto': T[('Proto', 'D2')], 'STN': T[('STN', 'Ctx')]}
ext_inp_delay = 0
#%% D2-FSI- Proto with derived external inputs

N_sim = 1000
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt


g = -0.01
G = {}
G[('D2', 'FSI')], G[('FSI', 'Proto')], G[('Proto', 'D2')] = g, g, g*0.5
noise_variance = { 'FSI': 0.025003981672404058, 'D2' : 0.0018347080766611102, 'Proto': 10**-10}
g_ext = -g
name1 = 'D2'
name2 = 'Proto'
name3 = 'FSI'


poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name3:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') : [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]  
init_method = 'heterogeneous'
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')
nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict,neuronal_model='spiking')
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,ax = None,title_fontsize=15,plot_start = 100,
        title = '')#r"$G_{SP}="+str(round(G[('Proto', 'STN')],2))+"$ "+", $G_{PS}=G_{PP}="+str(round(G[('STN', 'Proto')],2))+'$')

fig, axs = plt.subplots(len(nuclei_dict), 1, sharex=True, sharey=True)
count = 0
for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        count +=1
        nucleus.pop_act = moving_average_array(nucleus.pop_act,50)
        print(nucleus.name,np.average(nucleus.pop_act[int(len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]),2))
        spikes_sparse = [np.where(nucleus.spikes[i,:]==1)[0]*dt for i in range(nucleus.n)]

        axs[count-1].eventplot(spikes_sparse, colors='k',linelengths=2,lw = 2,orientation='horizontal')
        axs[count-1].tick_params(axis='both', labelsize=10)
        axs[count-1].set_title(nucleus.name, c = color_dict[nucleus.name],fontsize = 15)
        find_freq_of_pop_act_spec_window_spiking(nucleus, 0,t_list[-1], dt, cut_plateau_epsilon =0.1, peak_threshold = 0.1, smooth_kern_window= 3 , check_stability = False)

fig.text(0.5, 0.02, 'time (ms)', ha='center', va='center',fontsize= 15)
fig.text(0.02, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 15)

fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
fig2, ax2 = plt.subplots(1, 1, sharex=True, sharey=True)
fig3, ax3 = plt.subplots(1, 1, sharex=True, sharey=True)

for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        ax1.plot(t_list*dt,nucleus.voltage_trace,c = color_dict[nucleus.name],label = nucleus.name)
        ax2.plot(t_list*dt,nucleus.representative_inp['ext_pop','1'],c = color_dict[nucleus.name],label = nucleus.name)
        ax3.plot(t_list*dt,np.sum([nucleus.representative_inp[key].reshape(-1,) for key in nucleus.representative_inp.keys()],axis =0)-nucleus.representative_inp['ext_pop','1'],
                 c = color_dict[nucleus.name],label = nucleus.name)
ax1.set_title('membrane potential',fontsize = 15)
ax2.set_title('external input',fontsize = 15)
ax3.set_title('synaptic input',fontsize = 15)
ax1.legend(); ax2.legend();ax3.legend();

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

#%% FR simulation vs FR_expected (I = cte + Gaussian noise)

def run_FR_sim_vs_FR_expected_with_I_cte_and_noise(name, FR_list, variance, amplitude):
    N_sim = 1000
    N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
    dt = 0.25
    t_sim = 1000; t_list = np.arange(int(t_sim/dt))
    t_mvt = t_sim ; D_mvt = t_sim - t_mvt
    
    G = {}
    g = -0.01; g_ext = -g
    poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}
    noise_variance = {name : variance}
    noise_amplitude = {name : amplitude}
    receiving_pop_list = {(name,'1') : []}
    
    pop_list = [1]  
    nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop, init_method= init_method) for i in pop_list]
    nuclei_dict = {name: nuc}
    receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')
    
    firing_prop_hetero = find_FR_sim_vs_FR_expected(FR_list,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt)

    return firing_prop_hetero
name = 'Proto'
init_method = 'homogeneous'
# init_method = 'heterogeneous'
n = 10
start = 1 ; end = 10;
start = .04 ; end = .08; FR_list=np.linspace(start,end,n)

n_samples = 4
colormap = plt.cm.viridis# LinearSegmentedColormap
Ncolors = min(colormap.N,n_samples)
mapcolors = [colormap(int(x*colormap.N/Ncolors)) for x in range(Ncolors)]
amplitude_list = np.full(n_samples, 1)
variance_list = np.logspace(start = -10, stop = 1.5, num = n_samples , base = 10)
plt.figure()
for i in range (n_samples):
    firing_prop_hetero = run_FR_sim_vs_FR_expected_with_I_cte_and_noise(name, FR_list, variance_list[i], amplitude_list[i])
    plt.plot(FR_list,firing_prop_hetero[name]['firing_mean'][:,0],'-o',label = r'$\sigma=$'+"{:e}".format(variance_list[i]), c = mapcolors[i], markersize= 4)
    plt.fill_between(FR_list,firing_prop_hetero[name]['firing_mean'][:,0]-firing_prop_hetero[name]['firing_var'][:,0],
                  firing_prop_hetero[name]['firing_mean'][:,0]+firing_prop_hetero[name]['firing_var'][:,0],alpha = 0.1, color = mapcolors[i])
plt.plot(FR_list,FR_list,'--', label = 'y=x', c = 'k')
plt.xlabel(r'$FR_{expected}$',fontsize = 15)
plt.ylabel(r'$FR_{simulation}$',fontsize = 15)
plt.title(name + ' ' + init_method, fontsize = 20)
plt.legend()
#%% FR simulation vs FR_ext (I = cte + Gaussian noise) 
plt.close('all')
def run_FR_sim_vs_FR_ext_with_I_cte_and_noise(name, g_ext, poisson_prop, FR_list, variance, amplitude):

    N_sim = 1000
    N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
    dt = 0.25
    t_sim = 1000; t_list = np.arange(int(t_sim/dt))
    t_mvt = t_sim ; D_mvt = t_sim - t_mvt
    G = {}
    receiving_pop_list = {(name,'1') : []}
    
    pop_list = [1]  

    noise_variance = {name : variance}
    noise_amplitude = {name : amplitude}
    nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop, init_method = init_method) for i in pop_list]
    nuclei_dict = {name: nuc}
    receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
    firing_prop = find_FR_sim_vs_FR_ext(FR_list,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt)

    return firing_prop

init_method = 'heterogeneous'
# init_method = 'homogeneous'
name = 'FSI'
n = 100

g = -0.01; g_ext = -g
poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}
# FR_list = spacing_with_high_resolution_in_the_middle(n, *I_ext_range[name]).reshape(-1,) / poisson_prop[name]['g'] / poisson_prop [name ]['n'] / 3
FR_list = np.linspace(*I_ext_range[name], n).reshape(-1,) / poisson_prop[name]['g'] / poisson_prop [name ]['n'] / 3

n_samples = 1
mapcolors = create_color_map(n_samples, colormap = plt.get_cmap('viridis'))
mapcolors = create_color_map(n_samples, colormap = plt.get_cmap('Set1'))
mapcolors = ['#354b61','#2c8bc8', '#e95849','#3c7c24', '#854a45']
amplitude_list = np.full(n_samples, 1)
variance_list = np.logspace(start = -11.8, stop = -1, num = n_samples , base = 10)
variance_list = [0, .1,1, 5,15]
variance_list = [5]
fig, ax = plt.subplots(1,1, figsize = (6,5))


for i in range (n_samples):
    print(i+1, 'from', n_samples)
    # label_str = fmt(variance_list[i])
    label_str = (variance_list[i])
    # x_series  = FR_list * 1000
    x_series =FR_list * g_ext * poisson_prop[name]['tau']['decay']['mean']* poisson_prop[name]['n']
    firing_prop_hetero = run_FR_sim_vs_FR_ext_with_I_cte_and_noise(name, g_ext, poisson_prop, FR_list, variance_list[i], amplitude_list[i])
    plt.plot(x_series, firing_prop_hetero[name]['firing_mean'][:,0] ,'-o',label = r'$\sigma=$'+"{}".format(label_str), c = mapcolors[i], markersize = 4)
    plt.fill_between(x_series,
                      (firing_prop_hetero[name]['firing_mean'][:,0] - firing_prop_hetero[name]['firing_var'][:,0]) ,
                      (firing_prop_hetero[name]['firing_mean'][:,0] + firing_prop_hetero[name]['firing_var'][:,0]) ,
                      alpha = 0.1, color = mapcolors[i])
    # plt.plot(x_series, x_series*i ,'-o',label = r'$\sigma=$'+"{}".format(label_str), c = mapcolors[i], markersize = 4)

# plt.xlabel(r'$FR_{external}$',fontsize = 15)

plt.title(name + ' ' + init_method, fontsize = 20)
# plt.ylim(min(FR_list * 1000), max(FR_list * 1000))
plt.legend()    
# plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, I_ext_range[name], neuronal_consts, x_val = 'I_ext', ax = ax)
plt.xlabel(r'$I_{external} \; (mV)$',fontsize = 15)
plt.ylabel(r'$FR_{simulation} \; (Hz)$',fontsize = 15)
remove_frame(ax)
filename = name + '_N_1000_response_curve_with_different_noise_[0_0-1_1_5_15].png'
fig.savefig(os.path.join(path, filename), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)### extrapolate with the average firing rate ofthe  population
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


#%% Deriving F_ext from response curve of collective behavior in heterogeneous mode Demo

plt.close('all')
def run_FR_sim_vs_FR_ext_with_I_cte_and_noise(name, g_ext, poisson_prop, FR_list, variance, amplitude):

    N_sim = 1000
    N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
    dt = 0.25
    t_sim = 1000; t_list = np.arange(int(t_sim/dt))
    t_mvt = t_sim ; D_mvt = t_sim - t_mvt
    G = {}
    receiving_pop_list = {(name,'1') : []}
    
    pop_list = [1]  

    noise_variance = {name : variance}
    noise_amplitude = {name : amplitude}
    nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop, init_method = init_method) for i in pop_list]
    print("noise variance = ", nuc[0].noise_variance)
    nuclei_dict = {name: nuc}
    receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
    firing_prop = find_FR_sim_vs_FR_ext(FR_list,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt)

    return firing_prop

init_method = 'heterogeneous'
# init_method = 'homogeneous'
name = 'D2'
n = 10

g = -0.01; g_ext = -g
poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}
# FR_list = spacing_with_high_resolution_in_the_middle(n, *I_ext_range[name]).reshape(-1,) / poisson_prop[name]['g'] / poisson_prop [name ]['n'] / 3
FR_list = np.linspace(*I_ext_range[name], n).reshape(-1,) 


fig, ax = plt.subplots(1,1, figsize = (6,5))
x_series =FR_list * g_ext * poisson_prop[name]['tau']['decay']['mean']* poisson_prop[name]['n']
firing_prop_hetero = run_FR_sim_vs_FR_ext_with_I_cte_and_noise(name, g_ext, poisson_prop, FR_list, noise_variance[name],1)

plt.plot(x_series, firing_prop_hetero[name]['firing_mean'][:,0] ,'-o',label = r'$\sigma=$'+"{}".format(noise_variance[name]), 
         c = color_dict[name], markersize = 4)

plt.fill_between(x_series,
                  (firing_prop_hetero[name]['firing_mean'][:,0] - firing_prop_hetero[name]['firing_var'][:,0]) ,
                  (firing_prop_hetero[name]['firing_mean'][:,0] + firing_prop_hetero[name]['firing_var'][:,0]) ,
                  alpha = 0.1, color = color_dict[name])


plt.title(name + ' ' + init_method, fontsize = 20)
plt.legend()    
# plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, I_ext_range[name], neuronal_consts, x_val = 'I_ext', ax = ax)
plt.xlabel(r'$I_{external} \; (mV)$',fontsize = 15)
plt.ylabel(r'$FR_{simulation} \; (Hz)$',fontsize = 15)
remove_frame(ax)
filename = name + '_N_1000_response_curve_with_different_noise_[0_0-1_1_5_15].png'
fig.savefig(os.path.join(path, filename), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)### extrapolate with the average firing rate ofthe  population



#%% Deriving F_ext from response curve of collective behavior in heterogeneous mode 

plt.close('all')
name = 'D2'
state = 'rest'
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.1
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
duration = [int(t_sim/dt/2), int(t_sim/dt)]
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

G = {}
receiving_pop_list = {(name,'1') : []}

pop_list = [1]  

g = -0.01; g_ext = -g
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
# mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = True
set_FR_range_from_theory = False
set_input_from_response_curve = True
save_init = False
der_ext_I_from_curve= True
if_plot = True
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = False

poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.5},'decay':{'mean':5,'var':3}}, 'g':g_ext}}

class Nuc_keep_V_m(Nucleus):
    
    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = self.sum_synaptic_input(receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, t, receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        self.reset_potential_with_interpolation(spiking_ind,dt)
        self.all_mem_pot[:, t] = self.mem_potential
        
nuc = [Nuc_keep_V_m(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,
               keep_mem_pot_all_t= keep_mem_pot_all_t, ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, 
               path = path, save_init = save_init , noise_method = noise_method) for i in pop_list]

nuclei_dict = {name: nuc}
nucleus = nuc[0]
                    
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)

print("rest ext inp mean = ", np.average(nuc[0].rest_ext_input))

nuclei_dict = run(receiving_class_dict, t_list, dt,  {name: nuc})


save_all_mem_potential(nuclei_dict, path)
fig, ax = plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
nucleus.smooth_pop_activity(dt, window_ms = 5)
fig = plot(nuclei_dict,color_dict, dt,  t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
            title_fontsize=15, plot_start = 0, title = str(dt),
            include_FR = False, include_std=False, plt_mvt=False,
            legend_loc='upper right', ylim =None)

peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; 
cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig_spec, ax = plt.subplots(1,1)
_, f,pxx = find_freq_SNN_not_saving(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold , smooth_kern_window , 
                         smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                         low_pass_filter, 0, 2000, plot_spectrum = True, ax = ax, c_spec = color_dict, 
                         spec_figsize = (6,5), find_beta_band_power = False, fft_method = 'Welch', n_windows = 3, 
                         include_beta_band_in_legend = False)
ax.set_xlim(0,70)

#%% Deriving F_ext from the response curve
# np.random.seed(1006)
plt.close('all')
name = 'Proto'
N_sim = 1
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
G = {}
receiving_pop_list = {(name,'1') : []}

pop_list = [1]  
noise_variance = {name : 0}
noise_amplitude = {name : 1}
g = -0.01; g_ext = -g
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
set_input_from_response_curve = True
save_init =False
der_ext_I_from_curve= True
if_plot = True
# bound_to_mean_ratio = [0.5, 20]
# spike_thresh_bound_ratio = [1/5, 1/5]
poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.5},'decay':{'mean':5,'var':.2}}, 'g':g_ext}}

Act = A
# Act = A_DD
nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, Act, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuclei_dict = {name: nuc}
nucleus = nuc[0]
n = 70
pad = [0.001, 0.001]
all_FR_list ={'FSI': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1),
              'D2': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1),
              'Proto':   (0.04, 0.07)}#[0.02, 0.05]}
all_FR_list = {name: all_FR_list[name]}
name1 = 'FSI' ; name2 = 'D2' ; name3 = 'Proto'
filepaths = {name1: name1+ '_N_'+str(N_sim) +'_T_2000.pkl',
              name2:name2 + '_N_'+str(N_sim) +'_T_2000.pkl',
            name3: name3 + '_N_'+str(N_sim) +'_T_2000_noise_var_15.pkl'}
# nuc[0].set_init_from_pickle( os.path.join( path, filepaths[name]))
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
mapcolors = ['#354b61','#2c8bc8', '#e95849','#3c7c24', '#854a45']
color = mapcolors[0]
nuc[0].noise_variance = 0
# pad = [0.003, 0.0033] # FSI mvt
receiving_class_dict = set_connec_ext_inp(Act, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n, if_plot = if_plot, end_of_nonlinearity = 30, c=color, ax=ax,
                                          left_pad =pad[0], right_pad=pad[1])
plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, I_ext_range[name], neuronal_consts, x_val = 'I_ext', ax = ax)

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



#### Check behavior
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

### see the individual spikes and firing rates with instantenous and exponential input methods.
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
#%% Deriving F_ext from the response curve for each neuron individually
# np.random.seed(1006)
plt.close('all')
name = 'FSI'
state = 'rest'
N_sim = 1
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
G = {}
receiving_pop_list = {(name,'1') : []}

pop_list = [1]  

# noise_variance_tau_real = {'rest': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*7, 'STN':4} , # Proto tau_m = 20
#                            'mvt': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*7, 'STN':4} ,
#                            'DD': {'FSI' : 10 , 'D2': 3 , 'Proto': 15*7, 'STN':4} ,
#                           }
# noise_variance_tau_real = {'rest': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*8, 'STN':4} , # Proto tau_m = 25
#                            'mvt': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*8, 'STN':4} ,
#                            'DD': {'FSI' : 10 , 'D2': 3 , 'Proto': 15*8, 'STN':4} ,
#                          }
noise_variance_tau_real = {'rest': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*2, 'STN':4} , # Proto tau_m = 13
                            'mvt': {'FSI' : 8 , 'D2': 3 , 'Proto': 15*2, 'STN':4} ,
                            'DD': {'FSI' : 15 , 'D2': 3 , 'Proto': 15*2, 'STN':6} ,
                            'trans': {'STN' : 5, 'D2' : 12}
                          }
noise_variance = noise_variance_tau_real[state]

noise_amplitude = {name : 1}
g = -0.01; g_ext = -g
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
der_ext_I_from_curve= True
if_plot = False
keep_mem_pot_all_t= True
# bound_to_mean_ratio = [0.5, 20]
# spike_thresh_bound_ratio = [1/5, 1/5]
poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.5},'decay':{'mean':5,'var':3}}, 'g':g_ext}}

nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, Act[state], A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,keep_mem_pot_all_t= keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuclei_dict = {name: nuc}
nucleus = nuc[0]
n = 50

pad_tau_real = {'FSI': {'rest': [0.005, 0.0057], 'mvt': [0.005, 0.006] , 'DD': [0.005, 0.01] },
               'D2': {'rest': [0.002, 0.002], 'mvt': [0.001, 0.001] , 'DD': [0.002, 0.002] , 'trans' : [0.003, 0.005]},
               'Proto': {'rest': [0.001, 0.001], 'mvt': [0.01, 0.01] , 'DD': [] },
               'STN': {'rest': [0.004, 0.006], 'mvt': [0.003, 0.0033] , 'DD': [0.003, 0.0055], 'trans': [0.003, 0.003]}}

pad  = pad_tau_real
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
                      'FSI': { 'rest' : np.linspace ( 0.020, 0.05 , 250).reshape(-1,1) , 
                               'mvt': np.linspace ( 0.02, 0.05 , 250).reshape(-1,1), 
                               'DD': np.linspace ( 0.02,  0.05 , 250).reshape(-1,1) } ,
                      'D2': { 'rest' : np.linspace ( 0.001, 0.005 , 250).reshape(-1,1), 
                              'mvt': np.linspace ( 0.015, 0.05 , 250).reshape(-1,1) , 
                              'DD': np.linspace ( 0.015, 0.05 , 250).reshape(-1,1) ,
                              'trans' : np.linspace ( 0.01, 0.04 , 250).reshape(-1,1)} ,
                       # 'Proto': { 'rest' : np.array([0.02,0.05]), # tau_m = 20
                       #            'mvt' : np.linspace ( 0.13, 0.3 , 250).reshape(-1,1) , 
                       #            'DD' : []},
                       'Proto': { 'rest' : np.array([0.01,0.02]), # tau_m = 43
                                  'mvt' : np.linspace ( 0.13, 0.3 , 250).reshape(-1,1) , 
                                  'DD' : []},                       
                      # 'Proto': { 'rest' : np.array([0.01,0.03]), # tau_m = 13
                      #            'mvt' : np.linspace (  0.015, 0.022 , 250).reshape(-1,1), 
                      #            'DD' : []},
                    'STN': { 'rest' : np.linspace (  0.008, 0.03 , 250).reshape(-1,1) , 
                             'mvt': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1), 
                             'DD': np.linspace ( 0.01, 0.02 , 250).reshape(-1,1),
                             'trans' : np.linspace ( 0.0135, 0.018 , 250).reshape(-1,1)}
                      }
end_of_nonlinearity = {
                      'FSI': { 'rest' : 35 , 'mvt': 40, 'DD':40 } ,
                      'D2':  { 'rest' : 10 , 'mvt': 10 , 'DD': 20, 'trans': 40 } ,
                      'Proto':  { 'rest' :  25, 'mvt': 40 , 'DD': 35  },
                      'STN': { 'rest' : 35 , 'mvt': 25, 'DD':40, 'trans': 35 } 
                      }
all_FR_list = {name: all_FR_list_tau_real[name][state]}

filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
              'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl',
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
             'STN':'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl' }

# nuc[0].set_init_from_pickle( os.path.join( path,filepaths[name]), set_noise = False)
# receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n, if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity[name][state], 
#                                           left_pad =pad[name][state][0], right_pad=pad[name][state][1], set_FR_range_from_theory=set_FR_range_from_theory)
# ########## Collective 

receiving_class_dict = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =20, if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity[name][state], 
                                          left_pad =pad[name][state][0], right_pad=pad[name][state][1], set_FR_range_from_theory=set_FR_range_from_theory,
                                          method = 'single_neuron')
print("rest ext inp mean = ", np.average(nuc[0].rest_ext_input))
print("FR_ext mean = ", np.average(nuc[0].FR_ext))
print("number of nans = ", np.sum(np.isnan(nuc[0].FR_ext)))

# save_all_mem_potential(nuclei_dict, path)

nuclei_dict = run(receiving_class_dict, t_list, dt,  {name: nuc})
plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
# nucleus.smooth_pop_activity(dt, window_ms = 5)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, title = init_method)
#%% plot fitted response curve different noise 
plt.close('all')
name = 'Proto'
fit = 'linear'
N_sim = 1
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
G = {}
receiving_pop_list = {(name,'1') : []}

pop_list = [1]  
# init_method = 'heterogeneous'
init_method = 'homogeneous'
noise_variance = {name : 0.1}
noise_amplitude = {name : 1}
g = -0.01; g_ext = -g
ext_inp_method = 'const+noise'
ext_input_integ_method = 'dirac_delta_input'
syn_input_integ_method = 'exp_rise_and_decay'
bound_to_mean_ratio = [0.5, 20]
spike_thresh_bound_ratio = [1/5, 1/5]
poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.5},'decay':{'mean':5,'var':3}}, 'g':g_ext}}
Act = A_DD
nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, Act, A_mvt, name, G, T, t_sim, dt,
           synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',
           poisson_prop =poisson_prop, init_method = init_method, der_ext_I_from_curve= True, ext_inp_method = ext_inp_method,
           syn_input_integ_method = syn_input_integ_method, ext_input_integ_method= ext_input_integ_method, path = path) for i in pop_list]
nuclei_dict = {name: nuc}
nucleus = nuc[0]
n = 50
pad = [0.001, 0.001]
all_FR_list ={'FSI': np.linspace ( 0.05, 0.07 , 250).reshape(-1,1),
              'D2': np.linspace ( 0.05, 0.07 , 250).reshape(-1,1),
              'Proto': [0.04, 0.07]}
all_FR_list = {name: all_FR_list[name]}
n_samples = 3
# mapcolors = create_color_map(n_samples, colormap = plt.get_cmap('viridis'))
mapcolors = ['#354b61','#3c7c24', '#854a45']
if_plot = True
fig,ax = plt.subplots(1,1)
variance_list = [0, 5, 15]
count = 0
for count, noise in enumerate(variance_list):
    c = mapcolors[count]
    nucleus.noise_variance = noise
    receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n, if_plot = if_plot, end_of_nonlinearity = 30, 
                                          left_pad =pad[0], right_pad=pad[1], ax = ax, c = c)
    count +=1
##### Proto
FR_str = FR_ext_of_given_FR_theory(nucleus.spike_thresh, nucleus.u_rest, nucleus.membrane_time_constant, nucleus.syn_weight_ext_pop, all_FR_list[name][0], nucleus.n_ext_population)
FR_end = FR_ext_of_given_FR_theory(nucleus.spike_thresh, nucleus.u_rest, nucleus.membrane_time_constant, nucleus.syn_weight_ext_pop, all_FR_list[name][1], nucleus.n_ext_population)
FR_list = np.linspace(FR_str, FR_end, 100)
ax.plot( FR_list * nucleus.membrane_time_constant * nucleus.syn_weight_ext_pop * nucleus.n_ext_population,
        FR_ext_theory(nucleus.spike_thresh, nucleus.u_rest, nucleus.membrane_time_constant, nucleus.syn_weight_ext_pop, 
                      FR_list, nucleus.n_ext_population)* 1000 ,c= 'lightcoral', label = 'theory', lw = 2.5)
##### FSI
plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, [I_ext_range[name][0], 6.01], neuronal_consts, x_val = 'I_ext',ax = ax)

#### mutual
ax.legend()
ax_label_adjust(ax, fontsize = 20)
filename = name + '_one_neuron_'+ fit +'_fit_to_response_curve.png'
fig.savefig(os.path.join(path, filename), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, filename.replace('png', 'pdf')), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
#%% two connected nuclei with derived I_ext from response curve
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 200; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

name1 = 'Proto' # projecting
name2 = 'FSI' # recieving
g = -10**-5; g_ext = 0.01
G = {}
G[(name2, name1)] = g

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') : [],
                      (name2, '1'): [(name1,'1')]}

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
noise_variance = {name1 : 3, name2 : .1}
noise_amplitude = {name1 : 1, name2: 1}
# Act = A_DD
Act = A
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, Act, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve =False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init, set_input_from_response_curve=True ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, Act, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init, set_input_from_response_curve=True ) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
n = 50
pad = [0.001, 0.001]
all_FR_list ={'FSI': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1),
              'D2': np.linspace ( 0.045, 0.08 , 250).reshape(-1,1),
              'Proto': [0.04, 0.07]}
if_plot = False
filepaths = {'FSI': 'FSI_A_12-5_N_1000_T_2000_noise_var_0-1.pkl' ,
             'D2': 'D2_A_1-1_N_1000_T_2000_noise_var_0-1.pkl' ,
            'Proto': 'Proto_A_45_N_1000_T_2000_noise_var_3.pkl'}
DD_init_filepaths ={'Proto': 'Proto_A_38_N_1000_T_2000_noise_var_10.pkl',
               'FSI': 'FSI_A_24_N_1000_T_2000_noise_var_1.pkl',
               'D2' :'D2_A_6-6_N_1000_T_2000_noise_var_0-1.pkl' }
set_init_all_nuclei(nuclei_dict, filepaths = filepaths)
receiving_class_dict = set_connec_ext_inp(Act, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                          all_FR_list = all_FR_list , n_FR =n, if_plot = if_plot, end_of_nonlinearity = 25, left_pad =pad[0], right_pad=pad[1])

#### Check behavior
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, Act, A_mvt, D_mvt, t_mvt, t_list, dt, 
                                      mem_pot_init_method=mem_pot_init_method, set_noise = False)
nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)

for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        nucleus.smooth_pop_activity(dt, window_ms = 5)


fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = 0, plot_end = t_sim, ax_label=True,
                            tick_label_fontsize= 18, labelsize = 20, title_fontsize = 30, lw  = 2, linelengths = 2, n_neuron = 100, include_title = True, set_xlim=True)
fig_.set_size_inches((10, 3.5), forward=False)
filename = ( 'Raster_plot_' + mem_pot_init_method + '_'+  name1 + '_' + name2 + '_' + init_method + '_' + ext_inp_method + '_noise=' + 
            str(noise_variance[name1]) + '_' + str(noise_variance[name2])
            + '_N=' + str(N_sim) + '_N_ext=' +  str(poisson_prop[name1]['n']) + '.png' ) 

plt.savefig(os.path.join(path, filename), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', 
        transparent=True ,bbox_inches = "tight", pad_inches=0.1)
plt.savefig(os.path.join(path, filename.replace('png', 'pdf')), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', 
        transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig = plot(nuclei_dict,color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, 
           plot_start = 0, title = "", tick_label_fontsize=18, plt_mvt=False, ylim=(-10, 110))
fig.set_size_inches((6, 5), forward=False)
# fig.tight_layout()
filename = ( 'Smoothed_average_FR_'+ mem_pot_init_method + '_' +  name1 + '_' + name2 + '_' + init_method + '_' + ext_inp_method + '_noise=' + 
            str(noise_variance[name1]) + '_' + str(noise_variance[name2])
            + '_N=' + str(N_sim) + '_N_ext=' +  str(poisson_prop[name1]['n']) + '.png' ) 

plt.savefig(os.path.join(path, filename), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', 
        transparent=True ,bbox_inches = "tight", pad_inches=0.1)

plt.savefig(os.path.join(path, filename.replace('png','pdf')), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', 
        transparent=True ,bbox_inches = "tight", pad_inches=0.1)
#%% Run three connected nuclei save membrane potential distribution
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [0, int(t_mvt/dt)]
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
g = 0; g_ext =  0.01
G = {}
G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] = g ,0.5 * g, g

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name3:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                      (name3, '1'): [(name2,'1')]}

pop_list = [1]  
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
save_init = False
if_plot = False
noise_variance = {name1 : 0.1, name2: 0.1, name3 : 15}
noise_amplitude = {name1 : 1, name2: 1, name3: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,keep_mem_pot_all_t = True,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,keep_mem_pot_all_t = True,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t = True,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

filepaths = {'FSI': 'tau_m_9.5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
            'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_15.pkl'}
            # 'Proto':'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl'
set_init_all_nuclei(nuclei_dict, filepaths = filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)
nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = 0, title = init_method)

# json_name = os.path.join(path, 'FSI_D2_Proto_N_' + str( N_sim) +'_T_'+ str(t_sim) +  '_ms.json')
# write_obj_to_json(nuclei_dict, json_name)
save_all_mem_potential(nuclei_dict, path)
plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
#%% Effect of proto tau_m on STN-GPe-GPe frequency

# plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [400, int(t_sim/dt)]
name1 = 'STN'
name2 = 'Proto'
state = 'rest'
name_list = [name1, name2]

g_ext =  0.01
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
G[(name1, name2)] , G[(name2, name1)] , G[(name2, name2)]= g , -g, g
G = { k: v * K[k] for k, v in G.items()}
relative_G =  G[(name2, name2)]**2 / ( G[(name2, name1)] * G[(name1, name2)] )
print('relative G tau 13 = ', relative_G)
neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
                    'membrane_time_constant':{'mean':12.94,'var':10, 'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
noise_variance['Proto'] = 50


poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name2,'1')],
                       (name2, '1'): [(name1,'1'), (name2,'1')]}

               
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) } 

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N= True)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

#### Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path,  'FR_ext_STN-Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N= True)

        
n_run  = 1 ; n_iter = 1

peak_threshold = 0.1;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False; 
check_stability = False; plot_sig = False;  lower_freq_cut=1 ; upper_freq_cut=2000; freq_method='fft'; fft_method = 'Welch' 
fig, ax = plt.subplots()
data = {}
for nucleus_list in nuclei_dict.values():
    nucleus = nucleus_list[0]  # get only on class from each population
    data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_run), dtype = bool)


for i in range(n_run):
    if i > 0:

        nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A,
                                              A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                              reset_init_dist= True, poisson_prop = poisson_prop, normalize_G_by_N= True)  # , mem_pot_init_method = 'uniform')
        n_FR = 20
        all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) } 
        # receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
        #                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
        #                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
        #                                           use_saved_FR_ext= False, normalize_G_by_N= False)
        # pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
        FR_ext_all_nuclei  = load_pickle( os.path.join(path,  'FR_ext_STN-Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
        receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=False)
    
    nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
    smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)


    data = find_freq_SNN(data, 0, i, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 3, smooth_window_ms, cut_plateau_epsilon,
                         check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=True, ax=ax,
                        c_spec=color_dict, n_windows=6, fft_method=fft_method, find_beta_band_power=True,
                        include_beta_band_in_legend=True, half_peak_range = 5, 
                        n_std = 2, cut_off_freq = 100, check_peak_significance = True)

    fig.set_size_inches((6, 5), forward=False)
    ax.set_xlim(0,100)

# pickle_obj(data, os.path.join(path, 'STN-GP-GP-frequency-tau' +  str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

status = 'STN-GPe-GPe_tau_m_' + str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') 

# fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = '',
            include_FR = True, include_std=True, plt_mvt=False, legend_loc='upper right', ylim = None)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.set_size_inches((10, 6), forward=False)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

#%% Effect of proto tau_m on STN-GPe frequency
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [400, int(t_sim/dt)]
name1 = 'STN'
name2 = 'Proto'
state = 'rest'
name_list = [name1, name2]

g_ext =  0.01
G = {}
plot_start = 1000
plot_start_raster = 1000

g = - 0.01
G[(name1, name2)] , G[(name2, name1)] = g , -g

# neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
#                     'membrane_time_constant':{'mean':43,'var':10,  'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
# noise_variance['Proto'] = 100

neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
                    'membrane_time_constant':{'mean':12.94,'var':10, 'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
noise_variance['Proto'] = 50


poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}
G = { k: v * K[k] for k, v in G.items()}

receiving_pop_list = {(name1,'1') :  [(name2,'1')],
                       (name2, '1'): [(name1,'1')]}

               
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

n_run  = 10 ; n_iter = 1

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}

peak_threshold = 0.1;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False; 
check_stability = False; plot_sig = False;  lower_freq_cut=1 ; upper_freq_cut=2000; freq_method='fft'; fft_method = 'Welch' 
fig, ax = plt.subplots()
data = {}

for nucleus_list in nuclei_dict.values():
    nucleus = nucleus_list[0]  # get only on class from each population
    data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_run), dtype = bool)


for i in range(n_run):

    nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
                   poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
                   ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}
    
    n_FR = 20
    all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) } 
    
    # receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
    #                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
    #                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
    #                                           use_saved_FR_ext= False, normalize_G_by_N= True)
    # pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
    
    #### Run on previously saved data
    FR_ext_all_nuclei  = load_pickle( os.path.join(path,  'FR_ext_STN-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
    receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                              all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                              set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                              use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N= True)
    
            
    nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
    # smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)


    data = find_freq_SNN(data, 0, i, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 3, smooth_window_ms, cut_plateau_epsilon,
                         check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=True, ax=ax,
                        c_spec=color_dict, n_windows=6, fft_method=fft_method, find_beta_band_power=True,
                        include_beta_band_in_legend=True, half_peak_range = 5, 
                        n_std = 2, cut_off_freq = 100, check_peak_significance = True)

    fig.set_size_inches((6, 5), forward=False)
    ax.set_xlim(0,100)

pickle_obj(data, os.path.join(path, 'STN-GP-frequency-tau' +  str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

status = 'STN-GPe_tau_m_' + str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') 

fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = '',
            include_FR = True, include_std=True, plt_mvt=False, legend_loc='upper right', ylim = None)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.set_size_inches((10, 6), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
#%% Effect of proto tau_m on STN-GPe-FSI-D2 frequency

# plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'STN'
state = 'rest'
name_list = [name1, name2, name3, name4]

g_ext =  0.01
G = {}
plot_start = int(t_sim / 2)
plot_start_raster = int(t_sim / 2)
g = -0.0035
G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name3, name4)], G[(name4, name3)] = g, g, g, -g * 3, g * 3
G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name3, name4)], G[(name4, name3)] = 0, 0, 0, -g * 3, g * 3

neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
                    'membrane_time_constant':{'mean':43,'var':10,  'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
noise_variance['Proto'] = 100

# neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
#                     'membrane_time_constant':{'mean':12.94,'var':10, 'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
# noise_variance['Proto'] = 50

G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1'), (name4, '1')],
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}
    
n_run  = 1 ; n_iter = 1

peak_threshold = 0.1;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False; 
check_stability = False; plot_sig = False;  lower_freq_cut=1 ; upper_freq_cut=2000; freq_method='fft'; fft_method = 'Welch' 
fig, ax = plt.subplots()
data = {}
for nucleus_list in nuclei_dict.values():
    nucleus = nucleus_list[0]  # get only on class from each population
    data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_run), dtype = bool)


for i in range(n_run):

    nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
                   synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
                   poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
                   ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}
    n_FR = 20
    all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) } 
    
    # receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
    #                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
    #                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
    #                                           use_saved_FR_ext= False, normalize_G_by_N= True)
    # pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-GPe-FSI-D2_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
    
    # ### Run on previously saved data
    FR_ext_all_nuclei  = load_pickle( os.path.join(path,  'FR_ext_STN-GPe-FSI-D2_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
    receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                              all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                              set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                              use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N= True)
    
            

    nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
    smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)


    data = find_freq_SNN(data, 0, i, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 3, smooth_window_ms, cut_plateau_epsilon,
                         check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=True, ax=ax,
                        c_spec=color_dict, n_windows=6, fft_method=fft_method, find_beta_band_power=True,
                        include_beta_band_in_legend=True, half_peak_range = 5, 
                        n_std = 2, cut_off_freq = 100, check_peak_significance = True)

    fig.set_size_inches((6, 5), forward=False)
    ax.set_xlim(0,100)


status = 'STN-GPe-FSI-D2_tau_m_' + str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') 
pickle_obj(data, os.path.join(path, status + '.pkl'))

fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)


fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = '',
           include_FR = True, include_std=True, plt_mvt=False, legend_loc='upper right', ylim = None)

fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.set_size_inches((10, 6), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

#%% Effect of Proto tau_m on GPe-GPe frequency

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]
name1 = 'Proto'
name_list = [name1]
state = 'rest'
g_ext =  0.01
G = {}
plot_start = int(t_sim/2)
plot_start_raster =  int(t_sim/2)

# G[(name1, name1)] = -0.005#-0.025
# neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
#                     'membrane_time_constant':{'mean':43,'var':10,  'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
# noise_variance['Proto'] = 100


G[(name1, name1)] = -0.005
neuronal_consts['Proto'] = {'nonlin_thresh':-20 , 'nonlin_sharpness': 1, 'u_rest': -65, 'u_initial':{'min':-65, 'max':-37}, # Bogacz et al. 2016
                    'membrane_time_constant':{'mean':12.94,'var':10,  'truncmin': 2, 'truncmax': 100},'spike_thresh': {'mean':-37,'var':5}} # tau_m :#Projecting to STN from Karube et al 2019
noise_variance['Proto'] = 50

G = { k: v * K[k] for k, v in G.items()}
poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}
receiving_pop_list = {(name1,'1') :  [(name1,'1')] }


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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}

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

        
n_run  = 10 ; n_iter = 1

peak_threshold = 0.1;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False; 
check_stability = False; plot_sig = False;  lower_freq_cut=1 ; upper_freq_cut=2000; freq_method='fft'; fft_method = 'Welch' 
fig, ax = plt.subplots()
data = {} 
for nucleus_list in nuclei_dict.values():
    nucleus = nucleus_list[0]  # get only on class from each population
    data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
    data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_run), dtype = bool)


for i in range(n_run):


    # nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A,
    #                                       A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
    #                                       reset_init_dist= True, poisson_prop = poisson_prop, normalize_G_by_N= True)  # , mem_pot_init_method = 'uniform')
    nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
           synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
           poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
           ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}

    n_FR = 20
    all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) } 
    receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                              all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                              set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
                                              use_saved_FR_ext= False, normalize_G_by_N= True)
    pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
    # FR_ext_all_nuclei  = load_pickle( os.path.join(path,  'FR_ext_Proto-Proto_tau_'+ str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))
    # receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
    #                                   all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
    #                                   set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
    #                                   use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N= True)

   
    nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
    # smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)


    data = find_freq_SNN(data, 0, i, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 3, smooth_window_ms, cut_plateau_epsilon,
                         check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=True, ax=ax,
                        c_spec=color_dict, n_windows=6, fft_method=fft_method, find_beta_band_power=True,
                        include_beta_band_in_legend=True, half_peak_range = 5, 
                        n_std = 2, cut_off_freq = 100, check_peak_significance = True)

    fig.set_size_inches((6, 5), forward=False)
    ax.set_xlim(0,100)

pickle_obj(data, os.path.join(path, 'GP-GP-frequency-tau' +  str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') + '.pkl'))

status = 'GPe-GPe_tau_m_' + str(neuronal_consts['Proto']['membrane_time_constant']).replace('.','-') 

# fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_spectrum_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = '',
            include_FR = True, include_std=True, plt_mvt=False, legend_loc='upper right', ylim = None)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.set_size_inches((10, 6), forward=False)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

#%% plot freq vs. tau_m of Proto

######### Proto-Proto
status = "GPe-GPe"
data_13 = load_pickle(os.path.join(path, "GP-GP-frequency-tau{'mean': 12-94, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))
data_43 = load_pickle(os.path.join(path,"GP-GP-frequency-tau{'mean': 43, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))

############ STN-GPe-GPe
# status = "STN-GPe-GPe"
# data_13 = load_pickle(os.path.join(path, "STN-GP-GP-frequency-tau{'mean': 12-94, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))
# data_43 = load_pickle(os.path.join(path,"STN-GP-GP-frequency-tau{'mean': 43, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))

############ STN-GPe
status = "STN-GPe"
data_13 = load_pickle(os.path.join(path, "STN-GP-frequency-tau{'mean': 12-94, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))
data_43 = load_pickle(os.path.join(path,"STN-GP-frequency-tau{'mean': 43, 'var': 10, 'truncmin': 2, 'truncmax': 100}.pkl"))


name = 'Proto'
n_run = data_13[name,'base_freq'].shape[1]
df_13_Proto = pd.DataFrame({'f' : data_13[name,'base_freq'].reshape(n_run,), 'tau': [13] * n_run})
df_43_Proto = pd.DataFrame({'f' : data_43[name,'base_freq'].reshape(n_run,), 'tau': [43] * n_run})
df = pd.concat([df_13_Proto, df_43_Proto], ignore_index = True)


my_pal = {13: "white", 43: "white"}
my_pal_pts = {13: "g", 43: "b"}
fig, ax = plt.subplots()
ax = sns.boxplot(x="tau", y="f", data=df,  palette = my_pal, width = 0.3)
ax = sns.swarmplot(x="tau", y="f", data=df ,palette=my_pal_pts)
ax.set_xlabel(r'$\tau_{m}$', fontsize  = 15)
ax.set_ylabel('Frequency (Hz)', fontsize  = 15)
ax.set_title(status, fontsize = 18)

fig.savefig(os.path.join(path, 'Frequency_vs_Proto_tau_m_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)



#%% Polar phase histogram
low_f = 8 ; high_f = 20
plot_phase_histogram_all_nuclei(nuclei_dict, dt, color_dict, low_f, high_f, filter_order = 6, height = 0, density = False, n_bins = 20,
                            start = int(t_sim/dt/2), projection = 'polar', total_phase = 360, ref_nuc_name = 'self')

#%% Linear phase histogram
low_f = 8 ; high_f = 20
plot_phase_histogram_all_nuclei(nuclei_dict, dt, color_dict, low_f, high_f, filter_order = 6, height = 0, density = False, n_bins = 20,
                            start = int(t_sim/dt/2), projection = None, total_phase = 720, ref_nuc_name = 'Proto')
#%% STN-GPe
plt.close('all')

## Gaussian noise
# noise_variance = {'Proto' : 300, 
#                   'STN': 10}
# FR_ext_range = {'Proto': {'rest': np.array([15/1000, 25/1000]), 'DD':[1.8/300, 4.5/300], 'mvt':[1.8/300, 5/300]},
#                 'STN': {'rest': np.array([18/1000, 33/1000]),  'DD':[6/300, 7.5/300], 'mvt':[7.2/300, 8.8/300]}}


# ## OU noise
# noise_variance = {'Proto' : 50000, 
#                   'STN': 1000}


# FR_ext_range = {'Proto': {'rest': np.array([12/1000, 20/1000]), 'DD':[1.8/300, 4.5/300], 'mvt':[1.8/300, 5/300]},
#                 'STN': {'rest': np.array([18/1000, 25/1000]),  'DD':[6/300, 7.5/300], 'mvt':[7.2/300, 8.8/300]}}

N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.05
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration = [int(t_sim/dt/2), int(t_sim/dt)]

name1 = 'STN'
name2 = 'Proto'
state = 'rest'
name_list = [name1, name2]

g = -0.015
G = {}

G[(name1, name2)] , G[(name2, name1)] = g , -g
G = { k: v * K[k] for k, v in G.items()}

plot_start = 1400
plot_start_raster = 1400
plot_start = 100
plot_start_raster = 100



poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':0.01} for name in name_list}


receiving_pop_list = {(name1,'1') :  [(name2,'1')],
                      (name2, '1'): [(name1,'1')]}
                      # (name2, '1'): [(name1,'1'), (name2,'1')]}
                      
# receiving_pop_list = {(name1,'1') :  []
#                       }

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
use_saved_FR_ext = False
noise_tau = 20 

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight, noise_method = noise_method) for i in pop_list] for name in name_list}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 


receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective',  save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)


nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)

# # fig, ax = plt.subplots()
# ax.plot(t_list * dt, nuclei_dict['STN'][0].voltage_trace[:-1], label = "dt=" + str(dt))
# # ax.plot(t_list * dt, nuclei_dict['STN'][0].representative_inp['ext_pop', '1'])
# ax.legend(fontsize = 15, loc= 'upper left')
# ax.set_title('Voltage', fontsize = 20)

# status = 'STN-GPe' + '_G_SP_' + str(round(abs(G[('STN', 'Proto')]),1)) + '_G_PS_' + str(round(abs(G[('Proto', 'STN')]),1))

# n_neuron = 50
# fig_sizes = {'firing': (10,6),
#              'raster': (11,7),
#              'spectrum': (6, 5)}

# firing_fig_ylims = [0, 65]
# three_nuc_raster_y = (60 + 5) * 0.05
# fig_sizes = {'firing': (5, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05),
#              'raster': (5, three_nuc_raster_y/3 * 2),
#              'spectrum': (3, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05)}

print('rest ext input for Proto', np.average( nuclei_dict['Proto'][0].rest_ext_input))
print('rest ext input for Proto', np.average( nuclei_dict['STN'][0].rest_ext_input))

fig = plot(nuclei_dict,color_dict, dt,  t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
            title_fontsize=15, plot_start = plot_start, title = str(dt),
            include_FR = False, include_std=False, plt_mvt=False,
            legend_loc='upper right', ylim =None)

# # fig = remove_all_x_labels(fig)
# # fig.axes[0].set_ylim(firing_fig_ylims)
# # fig = set_y_ticks(fig, firing_fig_ylims)
# # fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(), 
# #            title_fontsize=15, plot_start = plot_start, title = '',
# #             include_FR = False, include_std=False, plt_mvt=False, 
# #             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

# # save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
# #              size = fig_sizes['firing'])

# include_nuc_name = False
# raster_order = ['Proto', 'STN']
# fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', 
#                                     plot_start = plot_start_raster, plot_end = t_sim, tick_label_fontsize = 12, 
#                                     title_fontsize = 25, lw  = 1, linelengths = 1, n_neuron = n_neuron, 
#                                     include_nuc_name = include_nuc_name, set_xlim=True, name_list = raster_order,
#                                     remove_ax_frame= False, y_tick_length= 2, x_tick_length = 3)

# # # fig_raster = remove_all_x_labels(fig_raster)
# # # fig_raster = set_y_ticks(fig_raster, [0, n_neuron])
# # # save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status ),
# # #              size = fig_sizes['raster'])

peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; 
cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig_spec, ax = plt.subplots(1,1)
_, f,pxx = find_freq_SNN_not_saving(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold , smooth_kern_window , 
                          smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                          low_pass_filter, 0, 2000, plot_spectrum = True, ax = ax, c_spec = color_dict, 
                          spec_figsize = (6,5), find_beta_band_power = False, fft_method = 'Welch', n_windows = 3, 
                          include_beta_band_in_legend = False)

# # fig_spec = remove_all_x_labels(fig_spec)

# # x_l = 0.75
# # ax.axhline(x_l, ls = '--', c = 'grey')
# # ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')
ax.set_xlim(0,100)
# # save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status ),
# #              size = fig_sizes['spectrum'])



# ax.set_xlim(0, 40)
# fig, ax = plt.subplots()
# check_significance_of_PSD_peak(f, pxx['STN'],  n_std_thresh = 2, min_f = 0, max_f = 250, n_pts_above_thresh = 3, ax = ax, legend = 'STN', c = color_dict['STN'])
# check_significance_of_PSD_peak(f, pxx['Proto'],  n_std_thresh = 2, min_f = 0, max_f = 250, n_pts_above_thresh = 3, ax = ax, legend = 'Proto', c = color_dict['Proto'])

#### To see how removing the so-called non-entrained neurons will change the population mean firing rate
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
ref_nuc_name = 'Proto'; low_f, high_f = 8, 70
find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
                                  height = 0, ref_nuc_name = ref_nuc_name, start = 0, total_phase = 720,
                                  only_entrained_neurons =False)
fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                    density = False, ref_nuc_name = ref_nuc_name, total_phase = 720, projection = None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18,
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

#%% Coherence

plt.close('all')
N_sim = 500
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.05
t_sim = 10000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'STN'
name2 = 'Proto'
state = 'rest'
name_list = [name1, name2]
g = -0.01

G = {}

G[(name1, name2)] , G[(name2, name1)] = g , -g
G = { k: v * K[k] for k, v in G.items()}


poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':0.01} for name in name_list}


receiving_pop_list = {(name1,'1') :  [(name2,'1')],
                      (name2, '1'): [(name1,'1')]}
                      # (name2, '1'): [(name1,'1'), (name2,'1')]}
                     

pop_list = [1]  
init_method = 'heterogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'constant'
mem_pot_init_method = 'draw_from_data'
keep_mem_pot_all_t = True
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False

class Nuc_keep_V_m(Nucleus):
    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = self.sum_synaptic_input(receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, t, receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        # self.reset_potential(spiking_ind)
        self.reset_potential_with_interpolation(spiking_ind,dt)
        
        # self.cal_population_activity(dt, t)
        # self.update_representative_measures(t)
        # if self.keep_mem_pot_all_t:
        self.all_mem_pot[:, t] = self.mem_potential
        
nuclei_dict = {name:  [Nuc_keep_V_m(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = True, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Proto-STN.pkl'))


# Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_Proto-STN.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

# x = np.array([0.0, 0.01, 0.015])
x = np.linspace(0, 0.015 ** 2, num = 20)
x = np.sqrt(x)
n = len(x)

G_dict = {(name1, name2) : -x ,
          (name2, name1) : x
         }
G_dict = { k: v * K[k] for k, v in G_dict.items()}

coherence = coherence_exploration(nuclei_dict, G_dict.copy(), noise_amplitude, noise_variance, A, N, N_real, K_real, receiving_pop_list,
                          A_mvt, D_mvt, t_mvt, t_list, dt,  all_FR_list , n_FR, receiving_class_dict,
                          end_of_nonlinearity, FR_ext_all_nuclei, color_dict,
                          poisson_prop, reset_init_dist = True, if_plot =False)

pickle_obj(coherence, os.path.join(path, 'coherence_' + str(dt).replace('.','-') + 
                                   '_T_' + str(t_sim) + 
                                   '_N_' + str(N_sim) +
                                   '.pkl' ))

#%% Plot Coherence

filename_list = ['coherence_0-2_T_6000_N_500.pkl',
                 'coherence_0-01_T_6000_N_500.pkl', 
                 'coherence_0-05_T_10000_N_500.pkl']
filename_list = [os.path.join(path, file) for file in filename_list]

name_list = ['STN', 'Proto']
for name in name_list:
    fig, ax = plt.subplots()
    for i, filename in enumerate(filename_list):
        coh = load_pickle( filename )
        ax.plot(  multiply_values_of_dict(coh['G']) , coh[name], '-o', label = name + ' (dt = ' + os.path.basename(filename).split('_')[1] + ')',
               )# alpha = 1 - 0.2 * i)
    ax.set_ylabel('Coherence', fontsize = 15)
    ax.set_xlabel(r'$G_{Loop}$')
    ax.legend(fontsize = 10)   


#%% GPe-GPe
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto'
state = 'rest'
name_list = [name1]
g = -0.005

G = {}
plot_start = 1400
plot_start_raster = 1400

G[(name1, name1)] = g
G = { k: v * K[k] for k, v in G.items()}


poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':0.01} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name1,'1')]}
                   
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


nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight) for i in pop_list] for name in name_list}

# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Proto-Proto.pkl'))


# Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_Proto-Proto.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
low_f = 8 ; high_f = 80
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'GPe-GPe' + '_G_PP_' + str(round( abs(G[('Proto', 'Proto')]),1))
fig_sizes = {'firing': (10,6),
             'raster': (11,7),
             'spectrum': (6, 5)}

n_neuron = 50
firing_fig_ylims = [25, 65]
three_nuc_raster_y = (60 + 5) * 0.05
fig_sizes = {'firing': (5, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05),
             'raster': (5, three_nuc_raster_y/3),
             'spectrum': (3, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05)}

fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
           title_fontsize=15, plot_start = plot_start, title = '',
           include_FR = False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim =None)

fig = remove_all_x_labels(fig)
fig.axes[0].set_ylim(firing_fig_ylims)
fig = set_y_ticks(fig, firing_fig_ylims)
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(), 
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False, 
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
             size = fig_sizes['firing'])

include_nuc_name = False
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', 
                                    plot_start = plot_start_raster, plot_end = t_sim, tick_label_fontsize = 12, 
                                    title_fontsize = 25, lw  = 1, linelengths = 1, n_neuron = n_neuron, 
                                    include_nuc_name = include_nuc_name, set_xlim=True,
                                    remove_ax_frame= False, y_tick_length= 2, x_tick_length = 3)
fig_raster = remove_all_x_labels(fig_raster)
fig_raster = set_y_ticks(fig_raster, [0, n_neuron])
save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status ),
             size = fig_sizes['raster'])

peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; 
cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig_spec, ax = plt.subplots(1,1)
freq, f, pxx = find_freq_SNN_not_saving(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold , smooth_kern_window , 
                         smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                         low_pass_filter, 0, 2000, plot_spectrum = True, ax = ax, c_spec = color_dict, 
                         spec_figsize = (6,5), find_beta_band_power = False, fft_method = 'Welch', n_windows = 3, 
                         include_beta_band_in_legend = False)

fig_spec = remove_all_x_labels(fig_spec)
# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0,70)
# ax.yaxis.set_major_locator(MaxNLocator(2)) 
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status ),
             size = fig_sizes['spectrum'])

fig, ax = plt.subplots()
check_significance_of_PSD_peak(f, pxx['Proto'],  n_std_thresh = 2, min_f = 0, max_f = 250, n_pts_above_thresh = 3, ax = ax, legend = 'Proto', c = color_dict['Proto'])


# ref_nuc_name = 'Proto'
# find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
#                                   height = 0, ref_nuc_name = ref_nuc_name, start = 0, total_phase = 720,
#                                   only_entrained_neurons =False)
# fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
#                                     density = False, ref_nuc_name = ref_nuc_name, total_phase = 720, projection = None,
#                                     outer=None, fig=None,  title='', tick_label_fontsize=18,
#                                     labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

#%% STN-GPe + GP-GP
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto' # projecting
name2 = 'STN' # recieving
g = -0.003; g_ext =  0.01
G = {}
plot_start = 0
plot_start_raster = 500
G[(name2, name1)] , G[(name1, name2)] , G[(name1, name1)]  = g, -g, g

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [(name2,'1'), (name1,'1') ],
                      (name2, '1'): [(name1,'1')]}

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
noise_variance = {name1 : 0.1, name2: 0.1, name3 : 15}
noise_amplitude = {name1 : 1, name2: 1, name3: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt, 
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# filepaths = {name1: name1+ '_N_'+str(N_sim) +'_T_2000.pkl',
#              name2:name2 + '_N_'+str(N_sim) +'_T_2000.pkl',
#             name3: name3 + '_N_'+str(N_sim) +'_T_2000_noise_var_15.pkl'}
# filepaths = {'FSI': 'FSI_A_12-5_N_1000_T_2000_noise_var_0-1.pkl' ,
#              'D2': 'D2_A_1-1_N_1000_T_2000_noise_var_0-1.pkl' ,
#             'Proto': 'Proto_A_45_N_1000_T_2000_noise_var_15.pkl'}
filepaths = {'STN':'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl',
            # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
            'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

set_init_all_nuclei(nuclei_dict, filepaths = filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)

nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
# save_all_mem_potential(nuclei_dict, path)
# fig, ax = plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
# fig.savefig(os.path.join(path, 'V_m_Distribution_STN_GPe_'+mem_pot_init_method+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'V_m_Distribution_STN_GPe_'+mem_pot_init_method+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
state = 'STN_GPe_GPe-GPe_Real_tau_Proto_13_'

fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
           title_fontsize=15, plot_start = plot_start, title = '',
           include_FR = False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim =None)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(), 
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False, 
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
             size = (10, 6))

fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', 
                                    plot_start = plot_start_raster, plot_end = t_sim, labelsize = 20, 
                                    title_fontsize = 25, lw  = 2, linelengths = 2, n_neuron = 40, 
                                    include_title = True, set_xlim=True)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status ),
             size = (11, 6))

peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; 
cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig, ax = plt.subplots(1,1)
find_freq_SNN_not_saving(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold , smooth_kern_window , 
                         smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                         low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, 
                         spec_figsize = (6,5), find_beta_band_power = False, fft_method = 'Welch', n_windows = 3, 
                         include_beta_band_in_legend = False)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0,70)
save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_' + status ),
             size = (6, 5))
#%% FSI-D2-Proto 
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
state = 'rest'
name_list = [name1, name2, name3]
g = -0.008; g_ext =  0.01
G = {}
plot_start = 1400
plot_start_raster = 1400

plot_start = 0
plot_start_raster = 0
G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] = 1.4 * g, 1.2 * g, g
# G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name3, name3)]  = g,g,g,g
G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1')]}
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
noise_tau = 20 

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight, noise_method = noise_method) for i in pop_list] for name in name_list}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', filename_dict = filename_dict, save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)



nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'FSI-D2-Proto' +  ('_G_DF_' + str(round( abs(G[('D2', 'FSI')]),1)) +
                            '_G_PD_' + str(round( abs(G[('Proto', 'D2')]),1)) +
                            '_G_FP_' + str(round( abs(G[('FSI', 'Proto')]),1)) )
n_neuron = 50
fig_sizes = {'firing': (10,6),
             'raster': (11,7),
             'spectrum': (6, 5)}

firing_fig_ylims = [-5, 60]
fig_sizes = {'firing': (5, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05),
             'raster': (5, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05),
             'spectrum':(3, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05)}

fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
           title_fontsize=15, plot_start = plot_start, title = '',
           include_FR = False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim =None)

fig = remove_all_x_labels(fig)
fig.axes[0].set_ylim(firing_fig_ylims)
fig = set_y_ticks(fig, [0, 30, 60])
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(), 
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False, 
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
             size = fig_sizes['firing'])

include_nuc_name = False
raster_order = ['Proto', 'FSI', 'D2']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', 
                                    plot_start = plot_start_raster, plot_end = t_sim, tick_label_fontsize = 12, 
                                    title_fontsize = 25, lw  = 1, linelengths = 1, n_neuron = n_neuron, 
                                    include_nuc_name = include_nuc_name, set_xlim=True, name_list = raster_order,
                                    remove_ax_frame= False, y_tick_length= 2, x_tick_length = 3)

fig_raster = remove_all_x_labels(fig_raster)
fig_raster = set_y_ticks(fig_raster, [0, n_neuron])
save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status ),
             size = fig_sizes['raster'])

peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; 
cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig_spec, ax = plt.subplots(1,1)
find_freq_SNN_not_saving(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold , smooth_kern_window , 
                         smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                         low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, 
                         spec_figsize = (6,5), find_beta_band_power = False, fft_method = 'Welch', n_windows = 3, 
                         include_beta_band_in_legend = False)

fig_spec = remove_all_x_labels(fig_spec)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0,70)
# ax.yaxis.set_major_locator(MaxNLocator(2)) 
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status ),
             size = fig_sizes['spectrum'])
#%% Arky-D2-Proto

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Arky' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
state = 'rest'
name_list = [name1, name2, name3]
g = -0.006; g_ext =  0.01
G = {}
plot_start = 1400
plot_start_raster = 1400

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] = 4 * g, g, 0.6 * g
# G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name3, name3)]  = g,g,g,g
G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1')]}
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight) for i in pop_list] for name in name_list}


# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Proto-Arky-D2.pkl'))


# Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_Proto-Arky-D2.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N= True)


nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
# save_all_mem_potential(nuclei_dict, path)

# fig, ax = plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
# fig.savefig(os.path.join(path, 'V_m_Distribution_all_nuclei_'+mem_pot_init_method+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'V_m_Distribution_all_nuclei_'+mem_pot_init_method+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'Arky-D2-Proto' +  ('_G_DA_' + str(round( abs(G[('D2', 'Arky')]),1)) +
                            '_G_PD_' + str(round( abs(G[('Proto', 'D2')]),1)) +
                            '_G_AP_' + str(round( abs(G[('Arky', 'Proto')]),1)) )
n_neuron = 50
fig_sizes = {'firing': (10,6),
             'raster': (11,7),  
             'spectrum': (6, 5)}

firing_fig_ylims = [-5, 60]
fig_sizes = {'firing': (5, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05),
             'raster': (5, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05),
             'spectrum':(3, ( firing_fig_ylims[1] - firing_fig_ylims[0] ) * 0.05)}

fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
           title_fontsize=15, plot_start = plot_start, title = '',
           include_FR = False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim =None)

fig = remove_all_x_labels(fig)
fig.axes[0].set_ylim(firing_fig_ylims)
fig = set_y_ticks(fig, [0, 30, 60])
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(), 
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False, 
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
             size = fig_sizes['firing'])

include_nuc_name = False
raster_order = ['Proto', 'Arky', 'D2']

fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', 
                                    plot_start = plot_start_raster, plot_end = t_sim, tick_label_fontsize = 12, 
                                    title_fontsize = 25, lw  = 1, linelengths = 1, n_neuron = n_neuron, 
                                    include_nuc_name = include_nuc_name, set_xlim=True, name_list = raster_order,
                                    remove_ax_frame= False, y_tick_length= 2, x_tick_length = 3)

fig_raster = remove_all_x_labels(fig_raster)
fig_raster = set_y_ticks(fig_raster, [0, n_neuron])
save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status ),
             size = fig_sizes['raster'])

peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; 
cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig_spec, ax = plt.subplots(1,1)
find_freq_SNN_not_saving(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold , smooth_kern_window , 
                         smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                         low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, 
                         spec_figsize = (6,5), find_beta_band_power = False, fft_method = 'Welch', n_windows = 3, 
                         include_beta_band_in_legend = False)

fig_spec = remove_all_x_labels(fig_spec)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0,70)
# ax.yaxis.set_major_locator(MaxNLocator(2)) 
save_pdf_png(fig_spec, os.path.join(path, 'SNN_spectrum_' + status ),
             size = fig_sizes['spectrum'])
#%% FSI-D2-Proto + STN-GPe

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'STN'
state = 'rest'
name_list = [name1, name2, name3, name4]

g = -0.003; g_ext =  0.01
G = {}
plot_start = 1000
plot_start_raster = 1000

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name3, name4)], G[(name4, name3)] = g, g, g, -g * 2 * 10/4, 2* g
G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}


receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1'), (name4, '1')],
					   (name4, '1'): [(name3, '1')]}
                       # (name3, '1'): [(name2,'1'), (name3, '1')]} # with GP-GP


pop_list = [1]  
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
# mem_pot_init_method = 'draw_from_data'
mem_pot_init_method = 'uniform'
keep_mem_pot_all_t = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
save_init = False

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}


n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
                                          use_saved_FR_ext= False, normalize_G_by_N = True)
pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-FSI-D2.pkl'))


# Run on previously saved data
# FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-FSI-D2.pkl'))
# receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
#                                           use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

# nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
#                                       t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)

nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
# save_all_mem_potential(nuclei_dict, path)

# fig, ax = plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)
# fig.savefig(os.path.join(path, 'V_m_Distribution_all_nuclei_'+mem_pot_init_method+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'V_m_Distribution_all_nuclei_'+mem_pot_init_method+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'Pallidostratal_plus_STN_GPe'

fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
           title_fontsize=15, plot_start = plot_start, title = '',
           include_FR = False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim =None)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(), 
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False, 
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
             size = (10, 6))

raster_order = ['Proto', 'FSI', 'D2']
fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', 
                                    plot_start = plot_start_raster, plot_end = t_sim, tick_label_fontsize = 12, 
                                    title_fontsize = 25, lw  = 1, linelengths = 1, n_neuron = n_neuron, 
                                    include_nuc_name = include_nuc_name, set_xlim=True, name_list = raster_order,
                                    remove_ax_frame= False, y_tick_length= 2, x_tick_length = 3)
save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status ),
             size = (11, 6))

peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; 
cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig, ax = plt.subplots(1,1)
find_freq_SNN_not_saving(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold , smooth_kern_window , 
                         smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                         low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, 
                         spec_figsize = (6,5), find_beta_band_power = False, fft_method = 'Welch', n_windows = 3, 
                         include_beta_band_in_legend = False)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0,70)
save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_' + status ),
             size = (6, 5))
#%% FSI-D2-Proto + Arky-D2-Proto + STN-GPe

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 5000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]
plot_start = t_sim - 400
plot_start_raster = plot_start
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'
state = 'rest'
name_list = [name1, name2, name3, name4, name5]

g_ext =  0.01 ; G = {}

g = -0.004
(G[(name2, name1)], G[(name3, name2)], 
 G[(name1, name3)], G[(name2, name4)], 
 G[(name4, name3)], G[(name3, name5)], 
 G[(name5, name3)], G[(name3, name3)]) = 2*g,0.5* g, 0.5*g, g, g, -g * 3, g * 3 ,g * 0.1
# g*1.1 , g*1.8, g, g, g, -g * 3.2, g*3.5 ,g * 0.1

##### D2 RMP var = 14
# g, g , g, g, g , -g * 3, g * 3 ,g * 0.1 --> 30 Hz
# g, 2*g, g, g, g, -g * 3, g * 3 ,g * 0.1 --> 22 Hz
# g, 2*g, g, g, g, -g * 3, g*2.5 ,g * 0.1 --> 20 Hz
# g, 2*g, g, g, g, -g * 3, g*2 ,g * 0.1 --> 18 Hz
# g, 2*g, g, g, g, -g * 3, g ,g * 0.1 --> 15 Hz

##### D2 RMP var = 1 t = 2s
# g, g , g, g, g , -g * 3, g * 3 ,g * 0.1 --> 35 Hz
# g, 2*g, g, g, g, -g * 3, g * 3 ,g * 0.1 --> 18.8 Hz double peak with 35
# g, 1.5*g, g, g, g, -g * 3, g*2.5 ,g * 0.1 --> 18.8 Hz
# g, 2*g, g, g, g, -g * 3, g*2 ,g * 0.1 --> 16.9 Hz
# g, 1.8*g, g, g, g, -g * 3, g*1.5 ,g * 0.1 --> 16.9 Hz
# g, 2*g, g, g, g, -g * 3, g ,g * 0.1 --> 15 Hz  works


##### D2 RMP var = 1 t = 5s
# 2*g, 0.5*g, 0.5*g, g, g, -g * 3, g * 3, g * 0.1 --> 30 Hz
# g, 2*g, g, g, g, -g * 3, g*2.5 , g * 0.1 --> 18.8 Hz double peak with 35
# g, 1.5*g, g, g, g, -g * 3, g*2.5, g * 0.1 --> 18.8 Hz almost single peak
# g, 2*g, g, g, g, -g * 3, g*2, g * 0.1 --> 17.6 Hz single peak
# g, 1.8*g, g, g, g, -g * 3, g*1.5, g * 0.1 --> 16.9 Hz
# g, 2*g, g, g, g, -g * 3, g, g * 0.1 --> 15 Hz  works
G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') : [(name3,'1')],
                      (name2, '1'): [(name1,'1'), (name4,'1')],
                      # (name3, '1'): [(name2,'1'), (name5,'1')],
                     (name3, '1'): [(name2,'1'), (name3, '1'), (name5,'1')], # with GP-GP
					   (name4, '1'): [(name3, '1')],
                       (name5, '1'): [(name3, '1')]}


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
low_f, high_f = 8, 30

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight= syn_component_weight) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 


# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)

# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI_small_var.pkl'))


## Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI_small_var.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'Pallidostratal_plus_STN_GPe'

fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
           title_fontsize=15, plot_start = plot_start, title = '',
           include_FR = False, include_std=False, plt_mvt=False,
           legend_loc='upper right', ylim =None)

# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = plt.gca(), 
#            title_fontsize=15, plot_start = plot_start, title = '',
#             include_FR = False, include_std=False, plt_mvt=False, 
#             legend_loc='upper right', ylim = None, plot_filtered=True, low_f = 8, high_f = 70)

save_pdf_png(fig, os.path.join(path, 'SNN_firing_' + status ),
             size = (10, 6))

fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', 
                                    plot_start = plot_start_raster, plot_end = t_sim, labelsize = 20, 
                                    title_fontsize = 25, lw  = 2, linelengths = 2, n_neuron = 40, 
                                    include_title = True, set_xlim=True)

save_pdf_png(fig_raster, os.path.join(path, 'SNN_raster_' + status ),
             size = (11, 6))

peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; 
cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False

fig, ax = plt.subplots(1,1)
find_freq_SNN_not_saving(dt, nuclei_dict, duration, lim_oscil_perc, peak_threshold , smooth_kern_window , 
                         smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                         low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, 
                         spec_figsize = (6,5), find_beta_band_power = False, fft_method = 'Welch', n_windows = 3, 
                         include_beta_band_in_legend = False)

# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
# ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')

ax.set_xlim(0,70)
save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_' + status ),
             size = (6, 5))
ref_nuc_name = 'D2'; 
find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
                                  height = 0, ref_nuc_name = ref_nuc_name, start = 0, total_phase = 720,
                                  only_entrained_neurons =False, troughs = True)
fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, nuc_order = ['D2', 'STN', 'Arky', 'Proto', 'FSI'],
                                    density = False, ref_nuc_name = ref_nuc_name, total_phase = 720, projection = None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18,
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

##### to see the heterogenity of D2 spikes
# nucleus = nuclei_dict['D2'][0]
# neurons = np.random.choice(nucleus.n, 1000, replace=False)
# spikes_sparse = create_sparse_matrix(nucleus.spikes[neurons, :], end=(
#     t_sim / dt), start=(plot_start / dt)) * dt
# ax = raster_plot(spikes_sparse, nucleus.name, color_dict,  ax=None, labelsize=10, title_fontsize = 15, lw =3 , linelengths = 2, orientation = 'vertical')
#%% Autocorrelation of individual neurons Demo
plt.close('all')
np.random.seed(10)
name = 'D2'
n_neuron = 4 ; window_ms = 10 ; t_lag = 200
nucleus = nuclei_dict[name][0]
n = nucleus.n
entrained_ind = significance_of_oscil_all_neurons( nuclei_dict[name][0], dt, window_mov_avg = 10, max_f = 250, 
                                      n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2, min_f_sig_thres = 0)

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
f, pxx = cut_PSD(f, pxx, max_f = 250)


t_series = np.arange(int(t_lag / dt)) * dt
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
c_list = ['g','g', 'r', 'r']
line_style_list = ['-', '--' ,'-', '--']
n_sd_thresh = 2
signif_thresh = cal_sig_thresh(f, pxx, n_sd_thresh = 2, min_f = 7, max_f = 250)
for n,i in enumerate(neuron_list):
    ax.plot(f, pxx[i,:], color = c_list[n], ls = line_style_list[n], marker = 'o')
    # ax.axhline(np.average(pxx[i,:]), f[0], f[-1], c = c_list[n])
    ax.axhline(signif_thresh[i], f[0], f[1], ls = '--', c = c_list[n])
    ax1.plot(t_series, autc[i,:int(t_lag / dt)], color = c_list[n])
    
ax.set_xlim(0, 100)  
ax.set_ylabel('PSD', fontsize = 15)
ax.set_xlabel('Frequency (Hz)', fontsize = 15)  
ax1.set_ylabel('Autocorrelation', fontsize = 15)
ax1.set_xlabel('lag (ms)', fontsize = 15)
#%% Check effect of G=1 from pre to post synaptic FR

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'STN'
name2 = 'Proto'
state = 'rest'
g = -0.01; g_ext =  0.01
G = {}
plot_start = 0
plot_start_raster = 0

G[(name2, name1)] = 1

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [],
                      (name2, '1'): [(name1,'1')]}


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
noise_amplitude = {name1 : 1, name2: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt, 
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}

n_FR = 20
all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) } 

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Proto-STN.pkl'))


# Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_Proto-STN.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

# nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
#                                       t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False, normalize_G_by_N=True)

t_transition = int( 400 / dt)
name  = 'STN' ; A_new  = 25 ; FR_list =  [2.9/300, 3.8/300]
nuclei_dict[name][0].change_basal_firing(A_new)
FR_ext_new = nuclei_dict[name][0].set_ext_inp_const_plus_noise_collective(FR_list, t_list, dt, receiving_class_dict, end_of_nonlinearity = 20, n_FR = n_FR)
nuclei_dict[name][0].change_basal_firing(A[name])
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_Proto-STN.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=False)

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
nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict,  name, FR_ext_new, A, t_transition, A_mvt, D_mvt, t_mvt)

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)

status = 'STN-GPe'
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = '',
            include_FR = False, include_std=False, plt_mvt=False, legend_loc='upper right', ylim = None)
round_dec = 1
for k,nuclei_list in enumerate(nuclei_dict.values()):
    for j, nucleus in enumerate(nuclei_list):
        FR_mean, FR_std = nucleus. average_pop_activity( 0, t_transition)
        txt =  r"$\overline{{FR_{{{0}}}}}$ ={1} ".format(nucleus.name,  round(FR_mean,round_dec) )
        
        fig.gca().text(0.05, 0.48 * (k+1) , txt, ha='left', va='center', rotation='horizontal',fontsize = 15, color = color_dict[nucleus.name], 
                       transform=fig.gca().transAxes)

        FR_mean, FR_std = nucleus. average_pop_activity( t_transition, len(t_list))
        txt =  r"$\overline{{FR_{{{0}}}}}$ ={1} ".format(nucleus.name,  round(FR_mean,round_dec) )
        
        fig.gca().text(0.5, 0.48 * (k+1) , txt, ha='left', va='center', rotation='horizontal',fontsize = 15, color = color_dict[nucleus.name], 
                       transform=fig.gca().transAxes)
plt.axvline(t_transition * dt, linestyle = '--')
fig.savefig(os.path.join(path, 'G_STN_Proto_equal_to_1_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
#%% effect of transient increase in STN activity onto GPe (with/without GABA-B)
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 800; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto' # projecting
name2 = 'STN' # recieving
g = -0.008; g_ext =  0.01
G = {}
plot_start = 600
plot_start_raster = 600
G[(name2, name1)] , G[(name1, name2)]  = g, -g
# G[(name2, name1)] , G[(name1, name2)]  = 0,0
tau[('STN','Proto')] =  {'rise':[  40],'decay':[  200]} # Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner
# tau[('STN','Proto')] =  {'rise':[1.1],'decay':[7.8]} # Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [(name2,'1')],
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
noise_variance = {name1 : 0.1,  name2 : 15}
noise_amplitude = {name1 : 1,  name2: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt, 
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

rest_init_filepaths = {'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl',
                    # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
                    'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}
                    # 'Proto': 'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl'}

trans_init_filepaths = {'STN':'tau_m_5-13_STN_A_100_N_1000_T_2000_noise_var_10.pkl',
                        'Proto': rest_init_filepaths['Proto']}
t_transient = 600 # ms
duration = 5
n_run = 10
syn_trans_delay_dict = {'STN' :0}
set_init_all_nuclei(nuclei_dict, filepaths = rest_init_filepaths)

nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)

# run_with_transient_external_input(receiving_class_dict,t_list, dt, nuclei_dict, rest_init_filepaths, trans_init_filepaths, Act['rest'], 
# 										Act['trans'],list_of_nuc_with_trans_inp, t_transient = int( t_transient / dt), duration = int( duration / dt))

# nuc1[0].low_pass_filter( dt, 1,200, order = 6)
# nuc2[0].low_pass_filter( dt, 1,200, order = 6)
# # smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)

avg_act = average_multi_run(receiving_class_dict,t_list, dt, nuclei_dict, rest_init_filepaths, Act['rest'], Act['trans'], 
                            syn_trans_delay_dict, t_transient = int( t_transient / dt), transient_init_filepaths= trans_init_filepaths,
                                        duration = int( duration / dt) ,n_run = n_run)
for nuclei_list in nuclei_dict.values():
    for k,nucleus in enumerate( nuclei_list) :
        nucleus.pop_act = avg_act[nucleus.name][:,k]
state = 'Only_STN_GPe_Real_tau_Proto_13_ms_only_STN-Proto_trans_Ctx_' + str(n_run) + '_run'
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=20, plot_start = plot_start,
            title = r'$\tau_{{m}}^{{Proto}} = 13\;ms\; , \; G={0}$'.format(g), plt_mvt = False, include_FR=False, ylim = [0,150])
# fig.set_size_inches((15, 7), forward=False)
plt.axvspan(t_transient , (t_transient + duration) , alpha=0.2, color='yellow')
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 500, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
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
# find_freq_SNN_not_saving(dt, nuclei_dict, duration_2, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
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

#%% effect of transient increase in STN activity onto GPe (with/without GABA-B) collectuve
#%% effect of transient increase in STN activity onto GPe (with/without GABA-B) collective
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 900; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
name1 = 'Proto' # projecting
name2 = 'STN' # recieving
state = 'rest'
name_list = [name1, name2]
g = -0.005
G = {}; g_ext =  0.01

# A['Proto'] = 28
# Act = {'rest': A, 'mvt': A_mvt, 'DD': A_DD, 'trans': A_trans}
# FR_ext_range['Proto']= {'rest': [1.5/300, 4.5/300], 'DD':[1.8/300, 4.5/300], 'mvt':[1.8/300, 5/300]}
# tau[('STN','Proto')] =  {'rise':[1.1, 40],'decay':[15, 200]}
# tau [('Proto','STN')]  = {'rise':[0.2],'decay':[20]} # Glut estimate]

syn_coef_GABA_b = 1
syn_component_weight = {
                        ('D2','FSI'):[1],
                        ('STN','Proto'): [1, syn_coef_GABA_b], # the relative weight of the GABA-a and GABA-b components
                        ('Proto','STN'): [1],
                        ('Proto','Proto'): [1, syn_coef_GABA_b],
                        ('Proto','D2'): [1],
                        ('FSI','Proto'): [1],
                        ('Arky', 'Proto') : [1, syn_coef_GABA_b],
                        ('D2','Arky'): [1]
                        }

plot_start = 750
plot_start_raster = 500
G[(name2, name1)] , G[(name1, name2)] ,   G[(name1, name1)]  = -0.00, 0.005, -0.000 # same strength GABA_a and b
# G[(name2, name1)] , G[(name1, name2)] ,  G[(name1, name3)],  G[(name1, name1)]  = -0.002, 0.005, -0.0005, -0.0001

G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name2,'1')],#, (name1, '1')],
                      (name2, '1'): [(name1,'1')]

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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking', 
               set_input_from_response_curve = set_input_from_response_curve, poisson_prop =poisson_prop,init_method = init_method, 
               der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t, 
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, 
               save_init = save_init, syn_component_weight =syn_component_weight ) for i in pop_list] for name in name_list}


n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity =end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto.pkl'))


# Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)


t_transient = 800 # ms
duration = 5
n_run = 10
duration_fft = [int((t_transient+100)/dt), int(t_sim/dt)]

list_of_nuc_with_trans_inp = ['STN']
coef_trans_inp = {'STN': 0.5} # for equal GABA strengths
# coef_trans_inp = {'STN': 0.5, 'D2': 2.5}

ext_inp_dict = {}
for name in list_of_nuc_with_trans_inp:
    nucleus = nuclei_dict[name][0]
    val = coef_trans_inp[name] * nucleus.FR_ext[0] * nucleus.syn_weight_ext_pop * \
                                    nucleus.n_ext_population * nucleus.neuronal_consts['membrane_time_constant']['mean']
    ext_inp_dict [ nucleus.name] = { 'mean': val , 
                                    'sigma' : 0.1 * val }
    

syn_trans_delay_dict = {k[0]: v for k,v in T.items() if k[0] == 'STN' and k[1] == 'Ctx'}



# nuc1[0].low_pass_filter( dt, 1,200, order = 6)
# nuc2[0].low_pass_filter( dt, 1,200, order = 6)

avg_act = average_multi_run_collective(receiving_pop_list, receiving_class_dict,t_list, dt, nuclei_dict,  A, G, N,N_real, K_real, syn_trans_delay_dict, poisson_prop, 
                       list_of_nuc_with_trans_inp,FR_ext_all_nuclei, n_FR, all_FR_list, end_of_nonlinearity,
                       t_transient = int (t_transient/dt ), duration = int( duration/dt ) ,n_run = n_run, A_mvt = None, D_mvt = 0, t_mvt = t_mvt,
                       ext_inp_dict = ext_inp_dict, noise_amplitude = None, noise_variance = None, reset_init_dist = True, color_dict=color_dict)

for nuclei_list in nuclei_dict.values():
    for k,nucleus in enumerate( nuclei_list) :
        nucleus.pop_act = avg_act[nucleus.name][:,k]
    
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'STN_GPe_D2_Proto_trans_Ctx_' + str(n_run) + '_run_'
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=20, plot_start = plot_start,
            title = "", plt_mvt = False, include_FR=False)#, ylim = [0,150])
# fig.set_size_inches((15, 7), forward=False)
plt.axvspan(t_transient , (t_transient + duration) , alpha=0.2, color='yellow')
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 500, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_fft, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
fig.set_size_inches((6, 5), forward=False)
# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(0,70)


#%% effect of MC-induced transient input on a STR-GPe-STN network taking into accound relative transmission delays of MC-STR and MC-STN single neuron

# plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 400; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto' # projecting
name2 = 'STN' # recieving
name3 = 'D2'
g = -0.004; g_ext =  0.01
G = {}

tau[('STN','Proto')] =  {'rise':[1.1, 40],'decay':[7.8, 200]} # Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner

plot_start = 150
plot_start_raster = 500
G[(name2, name1)] , G[(name1, name2)] ,  G[(name1, name3)]  = -.001, 0.002 , -0.001
# G[(name2, name1)] , G[(name1, name2)] ,  G[(name1, name3)]  = 0,0, 0

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name3:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [(name2,'1'), (name3,'1')],
                      (name2, '1'): [(name1,'1')],
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
noise_variance = {name1 : 0.1,  name2 : 15, name3 : .1}
noise_amplitude = {name1 : 1,  name2: 1, name3 : 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt, 
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt, 
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

rest_init_filepaths = {
                    'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl',
                    'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',              
                    # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
                    'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}
                    # 'Proto': 'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl'}

trans_init_filepaths = {
                        # 'STN':'tau_m_5-13_STN_A_46_N_1000_T_2000_noise_var_5.pkl',
                        'STN':'tau_m_5-13_STN_A_65_N_1000_T_2000_noise_var_5.pkl',
                        'Proto': rest_init_filepaths['Proto'],
                        # 'D2':'tau_m_13_D2_A_30_N_1000_T_2000_noise_var_20.pkl',
                        'D2' : 'tau_m_13_D2_A_23_N_1000_T_2000_noise_var_12.pkl',
                        # 'D2' : 'tau_m_13_D2_A_20_N_1000_T_2000_noise_var_10.pkl', 
                        }
t_transient = 200 # ms
duration = 5
n_run = 1
list_of_nuc_with_trans_inp = ['STN', 'D2']


			
syn_trans_delay_dict_STN = {k[0]: v for k,v in T.items() if k[0] == 'STN' and k[1] == 'Ctx'}
syn_trans_delay_dict_STR = {k[0]: v for k,v in T.items() if k[0] == 'D2' and k[1] == 'Ctx'}
syn_trans_delay_dict = {**syn_trans_delay_dict_STN, **syn_trans_delay_dict_STR}
syn_trans_delay_dict = {k: v / dt for k,v in syn_trans_delay_dict.items()}


set_init_all_nuclei(nuclei_dict, filepaths = rest_init_filepaths)

nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)


# nuc1[0].low_pass_filter( dt, 1,200, order = 6)
# nuc2[0].low_pass_filter( dt, 1,200, order = 6)

avg_act = average_multi_run(receiving_class_dict,t_list, dt, nuclei_dict, rest_init_filepaths, Act['rest'], Act['trans'], 
                            syn_trans_delay_dict, t_transient = int( t_transient / dt),  transient_init_filepaths = trans_init_filepaths,
                            duration = int( duration / dt), n_run = n_run)

for nuclei_list in nuclei_dict.values():
    for k,nucleus in enumerate( nuclei_list) :
        nucleus.pop_act = avg_act[nucleus.name][:,k]
state = 'STN_GPe_D2_Real_tau_Proto_13_ms_trans_Ctx_'+str(n_run) + '_trans_delay_not_included_run_tau_SP_6'
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=20, plot_start = plot_start,
            title = r'$\tau_{{m}}^{{Proto}} = 13\;ms\; , \; G={0}, \; \tau_{{SP}}=12$'.format(g), plt_mvt = False, include_FR=False)#, ylim = [0,150])
fig.set_size_inches((15, 7), forward=False)
plt.axvspan(t_transient , (t_transient + duration) , alpha=0.2, color='yellow')
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 500, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
#%% effect of MC-induced transient input on a STR-GPe-STN network taking into accound relative transmission delays of MC-STR and MC-STN collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 1300; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
name1 = 'Proto' # projecting
name2 = 'STN' # recieving
name3 = 'D2'
state = 'rest'
name_list = [name1, name2, name3]
g = -0.005
G = {}; g_ext =  0.01

# A['Proto'] = 28
# Act = {'rest': A, 'mvt': A_mvt, 'DD': A_DD, 'trans': A_trans}
# FR_ext_range['Proto']= {'rest': [1.5/300, 4.5/300], 'DD':[1.8/300, 4.5/300], 'mvt':[1.8/300, 5/300]}
# tau[('STN','Proto')] =  {'rise':[1.1, 40],'decay':[15, 200]}
# tau [('Proto','STN')]  = {'rise':[0.2],'decay':[20]} # Glut estimate]

syn_coef_GABA_b = 5
syn_component_weight = {
                        ('D2','FSI'):[1],
                        ('STN','Proto'): [1, syn_coef_GABA_b], # the relative weight of the GABA-a and GABA-b components
                        ('Proto','STN'): [1],
                        ('Proto','Proto'): [1, syn_coef_GABA_b],
                        ('Proto','D2'): [1],
                        ('FSI','Proto'): [1],
                        ('Arky', 'Proto') : [1, syn_coef_GABA_b],
                        ('D2','Arky'): [1]
                        }

plot_start = 750
plot_start_raster = 500
G[(name2, name1)] , G[(name1, name2)] ,  G[(name1, name3)], G[(name1, name1)]  = -0.005, 0.0035, -0.001, -0.0001 # same strength GABA_a and b
# G[(name2, name1)] , G[(name1, name2)] ,  G[(name1, name3)],  G[(name1, name1)]  = -0.002, 0.005, -0.0005, -0.0001

G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name2,'1'), (name3,'1'), (name1, '1')],
                      (name2, '1'): [(name1,'1')],
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking', 
               set_input_from_response_curve = set_input_from_response_curve, poisson_prop =poisson_prop,init_method = init_method, 
               der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t, 
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, 
               save_init = save_init, syn_component_weight =syn_component_weight ) for i in pop_list] for name in name_list}


n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity =end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-D2.pkl'))


# Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-D2.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)


t_transient = 800 # ms
duration = 5
n_run = 10
duration_fft = [int((t_transient+100)/dt), int(t_sim/dt)]

list_of_nuc_with_trans_inp = ['STN', 'D2']
coef_trans_inp = {'STN': 0.5, 'D2': 3.5} # for equal GABA strengths
# coef_trans_inp = {'STN': 0.5, 'D2': 2.5}

ext_inp_dict = {}
for name in list_of_nuc_with_trans_inp:
    nucleus = nuclei_dict[name][0]
    val = coef_trans_inp[name] * nucleus.FR_ext[0] * nucleus.syn_weight_ext_pop * \
                                    nucleus.n_ext_population * nucleus.neuronal_consts['membrane_time_constant']['mean']
    ext_inp_dict [ nucleus.name] = { 'mean': val , 
                                    'sigma' : 0.1 * val }
    
def get_syn_trans_delay(T):
    
    syn_trans_delay_dict_STN = {k[0]: v for k,v in T.items() if k[0] == 'STN' and k[1] == 'Ctx'}
    syn_trans_delay_dict_STR = {k[0]: v for k,v in T.items() if k[0] == 'D2' and k[1] == 'Ctx'}
    syn_trans_delay_dict = {**syn_trans_delay_dict_STN, **syn_trans_delay_dict_STR}
    syn_trans_delay_dict = {k: v / dt for k,v in syn_trans_delay_dict.items()}
    return syn_trans_delay_dict

syn_trans_delay_dict = get_syn_trans_delay(T)



# nuc1[0].low_pass_filter( dt, 1,200, order = 6)
# nuc2[0].low_pass_filter( dt, 1,200, order = 6)

avg_act = average_multi_run_collective(receiving_pop_list, receiving_class_dict,t_list, dt, nuclei_dict,  A, G, N,N_real, K_real, syn_trans_delay_dict, poisson_prop, 
                       list_of_nuc_with_trans_inp,FR_ext_all_nuclei, n_FR, all_FR_list, end_of_nonlinearity,
                       t_transient = int (t_transient/dt ), duration = int( duration/dt ) ,n_run = n_run, A_mvt = None, D_mvt = 0, t_mvt = t_mvt,
                       ext_inp_dict = ext_inp_dict, noise_amplitude = None, noise_variance = None, reset_init_dist = True, color_dict=color_dict)

for nuclei_list in nuclei_dict.values():
    for k,nucleus in enumerate( nuclei_list) :
        nucleus.pop_act = avg_act[nucleus.name][:,k]
    
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'STN_GPe_D2_Proto_trans_Ctx_' + str(n_run) + '_run_'
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=20, plot_start = plot_start,
            title = "", plt_mvt = False, include_FR=False)#, ylim = [0,150])
# fig.set_size_inches((15, 7), forward=False)
plt.axvspan(t_transient , (t_transient + duration) , alpha=0.2, color='yellow')
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 500, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_fft, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
fig.set_size_inches((6, 5), forward=False)
# x_l = 0.75
# ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(0,70)
#%% effect of MC-induced transient input on FSI and D2
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 300; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
state = 'rest'
g = -0.005; g_ext =  0.01
G = {}


plot_start = 150
plot_start_raster = 500
G[(name2, name1)]   = g
G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [],
                      (name2, '1'): [(name1,'1')]
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
noise_amplitude = {name1 : 1,  name2: 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt, 
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# rest_init_filepaths = {
#                     'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl',
#                     'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl'}
# set_init_all_nuclei(nuclei_dict, filepaths = rest_init_filepaths)

n_FR = 20
all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) } 


receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
                                          use_saved_FR_ext= False)

pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_FSI-D2.pkl'))


## Run on previously saved data
# FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_FSI-D2.pkl'))
# receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
#                                           use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

t_transient = 200 # ms
duration = 5
n_run = 5
list_of_nuc_with_trans_inp = ['STN', 'D2']
inp = 2 # external input in mV to FSI and D2 from Cortex
g_rel_MC = 1.5 # relative gain of MC-FSI to MC-D2
g_rel_MC_series = np.linspace(1, 4, 4)
inp_series = np.linspace(0.5, 5, 4)
n_subplots =  int( len(g_rel_MC_series) * len(inp_series))


			
syn_trans_delay_dict_STN = {k[0]: v for k,v in T.items() if k[0] == 'FSI' and k[1] == 'Ctx'}
syn_trans_delay_dict_STR = {k[0]: v for k,v in T.items() if k[0] == 'D2' and k[1] == 'Ctx'}
syn_trans_delay_dict = {**syn_trans_delay_dict_STN, **syn_trans_delay_dict_STR}
syn_trans_delay_dict = {k: v / dt for k,v in syn_trans_delay_dict.items()}



count = 0
fig = plt.figure()
# fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(6, 6))

for inp in inp_series:
    
    for g_rel_MC in g_rel_MC_series:
        print(count, "from ", n_subplots)
        ext_inp_dict = {'FSI': {'mean' : g_rel_MC * inp ,'sigma' : 5} ,
                'D2': {'mean' : inp, 'sigma' : 5 }
                }
        nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                              t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)
        
        
        avg_act = average_multi_run(receiving_class_dict,t_list, dt, nuclei_dict, rest_init_filepaths, Act['rest'], 
        										Act['trans'], syn_trans_delay_dict, t_transient = int( t_transient / dt), 
                                                duration = int( duration / dt) ,n_run = n_run, inp_method = 'add', ext_inp_dict = ext_inp_dict)
        
        for nuclei_list in nuclei_dict.values():
            for k,nucleus in enumerate( nuclei_list) :
                nucleus.pop_act = avg_act[nucleus.name][:,k]
        state = 'FSI_D2_trans_Ctx_'+str(n_run) + '_runs_delay_included'
        ax = fig.add_subplot( len(inp_series) , len( g_rel_MC_series) , count+1)
        plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title_fontsize=10, plot_start = plot_start, n_subplots = n_subplots,
                    title = r'$\frac{{G_{{MC-FSI}}}}{{G_{{MC-D2}}}} = {0} \; I_{{MC}}={1}$'.format(g_rel_MC, inp), 
                    plt_mvt = False, include_FR=False, tick_label_fontsize = 10)#, ylim = [0,150])
        plt.axvspan(t_transient , (t_transient + duration) , alpha=0.2, color='yellow')
        plt.xlabel("")
        plt.ylabel("")
        count += 1
        if count <  ( len(inp_series) - 1) * len(g_rel_MC_series) - 1:
            ax.axes.xaxis.set_ticklabels([])
            

        
        
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("time (ms)")
plt.ylabel("Firing rate (Hz)")
fig.set_size_inches((15, 7), forward=False)
# fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 500, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig.savefig(os.path.join(path, 'SNN_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)


#%% Transition to activated state FSI-D2-GPe + STN-GPe 

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
t_transition = 1000
duration_2 = [int(t_sim/dt/2), int(t_sim/dt)]
name1 = 'Proto' # projecting
name2 = 'STN' # recieving
name3 = 'D2'
name4 = 'FSI'
g = -0.0012; g_ext =  0.01
G = {}
plot_start = 500
plot_start_raster = 600

G[(name2, name1)] , G[(name1, name2)],  = g, -g
G[(name3, name4)], G[(name4, name1)] , G[(name1, name3)] =   g, g, g
tau[('STN','Proto')] =  {'rise':[ 1.1 , 40],'decay':[ 7.8 ,  200]} # Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner
# tau[('STN','Proto')] =  {'rise':[1.1],'decay':[7.8]} # Baufreton et al. 2009, decay=6.48 Fan et. al 2012, GABA-b from Geetsner

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
				name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
				name3:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name4:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [(name2,'1'), (name3, '1')],
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
noise_variance = {name1 : 0.1,  name2 : 15,  name3 : 1,  name4 : 1}
noise_amplitude = {name1 : 1,  name2: 1,  name3 : 1,  name4 : 1}

nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuc4 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name4, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]



nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3, name4: nuc4}



filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
              'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl',
               # 'Proto': 'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl',
			 'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl'
			}

mvt_init_filepaths ={'Proto': 'tau_m_12-94_Proto_A_22_N_1000_T_2000_noise_var_30.pkl',
					 # 'FSI': 'FSI_A_70_N_1000_T_2000_noise_var_10.pkl',
					 'FSI': 'tau_m_9-2_FSI_A_32_N_1000_T_2000_noise_var_8.pkl',
					 'D2' : 'tau_m_13_D2_A_4_N_1000_T_2000_noise_var_3.pkl',
					 'STN': "tau_m_5-13_STN_A_50_N_1000_T_2000_noise_var_3.pkl"
					 }


receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
set_init_all_nuclei(nuclei_dict, filepaths = filepaths)#filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method='draw_from_data', set_noise= False)


nuclei_dict = run_transition_to_movement(receiving_class_dict,t_list, dt, nuclei_dict, mvt_init_filepaths, N, N_real, 
										 A_mvt, D_mvt,t_mvt,t_transition = int(t_transition/dt))

# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
state = 'transition_to_mvt_transient'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_transition, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = init_method,
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt= 0.8, axvspan_color='lightskyblue', ylim=(-10,80))
fig.set_size_inches((15, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start, plot_end = t_sim,
                            labelsize = 20, title_fontsize = 25, lw  = 1.5, linelengths = 2, n_neuron = 40, include_title = True, set_xlim=True,
                            axvspan = True, span_start = t_transition, span_end = t_sim, axvspan_color = 'lightskyblue')
fig.set_size_inches((15, 5), forward=False)
fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center',fontsize= 18)
fig_.text(0.03, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 18)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_2, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
x_l = 0.75
ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(0,55)
ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')
fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(0,55)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

#%% Transition to DD FSI-D2-GPe + STN-GPe 

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

plot_start = int(t_sim / 5)
t_transition = plot_start + int( t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)] 
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim /dt) - length, int(t_sim /dt)]

name1 = 'Proto' # projecting
name2 = 'STN' # recieving
name3 = 'D2'
name4 = 'FSI'
g = -0.0009; g_ext =  0.01
G = {}
G[(name2, name1)] , G[(name1, name2)],  = g, -g
G[(name3, name4)], G[(name4, name1)] , G[(name1, name3)] =   g, g, g

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
				name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
				name3:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name4:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [(name2,'1'), (name3, '1')],
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


nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuc4 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name4, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]



nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3, name4: nuc4}



filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
              'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl',
               # 'Proto': 'tau_m_25_Proto_A_45_N_1000_T_2000_noise_var_120.pkl',
			 'STN': 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl'
			}

DD_init_filepaths ={'Proto': 'tau_m_12-94_Proto_A_22_N_1000_T_2000_noise_var_30.pkl',
					 # 'FSI': 'FSI_A_70_N_1000_T_2000_noise_var_10.pkl',
					 'FSI': 'tau_m_9-2_FSI_A_24_N_1000_T_2000_noise_var_15.pkl',
					 'D2' : 'tau_m_13_D2_A_6-6_N_1000_T_2000_noise_var_3.pkl',
					 'STN': "tau_m_5-13_STN_A_24_N_1000_T_2000_noise_var_6.pkl"
					 }


receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
set_init_all_nuclei(nuclei_dict, filepaths = filepaths)#filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method='draw_from_data', set_noise= False)


nuclei_dict = run_transition_to_DA_depletion(receiving_class_dict,t_list, dt, nuclei_dict, DD_init_filepaths,
                                                K_real_DD, N, N_real, A_DD, A_mvt, D_mvt,t_mvt,t_transition = int(t_transition/dt))

# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
state = 'transition_to_DD_with_GABA-b'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_DD, t_transition, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = init_method,
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt= 0.8, axvspan_color='darkseagreen')
fig.set_size_inches((15, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start, plot_end = t_sim,
                            labelsize = 20, title_fontsize = 25, lw  = 1.5, linelengths = 2, n_neuron = 40, include_title = True, set_xlim=True,
                            axvspan = True, span_start = t_transition, span_end = t_sim, axvspan_color = 'darkseagreen')
fig.set_size_inches((15, 5), forward=False)
fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center',fontsize= 18)
fig_.text(0.03, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 18)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
x_l = 5
ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(5,55)
ax.axvspan(5,55, alpha = 0.2, color = 'darkseagreen')
fig.savefig(os.path.join(path, 'SNN_spectrum_DD_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_DD_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
x_l = 5
ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(5,55)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

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

#%% Transition to DD FSI-D2-GPe+ Arky-D2-GPe + STN-GPe collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 5)
# plot_end = int((t_transition + 1000)/dt)
t_transition = plot_start + int( t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)] 
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim /dt) - length + int(t_transition / 4/ dt), int(t_sim /dt)]
name1 = 'FSI' 
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'
state = 'rest'
name_list = [name1, name2, name3, name4, name5]

g_ext =  0.01 ; G = {}

g = -0.0025
(G[(name2, name1)], G[(name3, name2)], 
 G[(name1, name3)], G[(name2, name4)], 
 G[(name4, name3)], G[(name3, name5)], 
 G[(name5, name3)], G[(name3, name3)]) = g, g, g, g, g, -g*2.5, g*2.5 ,g * 0.1


# g, 1.5*g, g, g, g, -g * 3, g*2.5 ,g * 0.1 --> 18.8 Hz almost single peak

G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') : [(name3,'1')],
                      (name2, '1'): [(name1,'1'), (name4,'1')],
                      # (name3, '1'): [(name2,'1'), (name5,'1')],
                     (name3, '1'): [(name2,'1'), (name3, '1'), (name5,'1')], # with GP-GP
					   (name4, '1'): [(name3, '1')],
                       (name5, '1'): [(name3, '1')]}


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
low_f, high_f = 8, 70

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight ) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 


# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)

# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI.pkl'))


# Run on previously saved data
FR_ext_all_nuclei_rest  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei_rest, normalize_G_by_N=True)

all_FR_list = {name: FR_ext_range[name]['DD'] for name in list(nuclei_dict.keys()) } 
# change_basal_firing_all_nuclei(Act['DD'], nuclei_dict)
# receiving_class_dict , FR_ext_all_nuclei_DD = set_connec_ext_inp(Act['DD'], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real_DD, receiving_pop_list, nuclei_dict,t_list, 
#                                                                   all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                                                   set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                                                   use_saved_FR_ext= False, normalize_G_by_N = False, state = 'DD')

# pickle_obj(FR_ext_all_nuclei_DD, os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI_DD.pkl'))


## Run on previously saved data
FR_ext_all_nuclei_DD  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI_DD.pkl'))
# change_basal_firing_all_nuclei(Act['rest'], nuclei_dict)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, Act['rest'],
                                      A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                      reset_init_dist= True, poisson_prop = poisson_prop, 
                                      normalize_G_by_N= True)  

receiving_class_dict = set_connec_ext_inp(Act['rest'], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', use_saved_FR_ext= True,
                                          FR_ext_all_nuclei_saved = FR_ext_all_nuclei_rest, 
                                          return_saved_FR_ext= False, normalize_G_by_N= False)
# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
nuclei_dict = run_transition_to_DA_depletion_collective_setting(receiving_class_dict, receiving_pop_list, t_list, dt, nuclei_dict,
                                                      FR_ext_all_nuclei_DD, K_real_DD, N, N_real, Act['DD'],
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity,t_transition = int(t_transition/dt))

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'transition_to_DD'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_DD, t_transition, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title ='',
           legend_loc = 'upper left',plot_end = t_transition-10, ylim = (-4, 80),
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt= 0.8, axvspan_color='darkseagreen')
fig.set_size_inches((8, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'_1.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'1_.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_DD, t_transition, D_mvt, ax = None, title_fontsize=15, plot_start = t_sim - (t_transition-plot_start), title ='',
           legend_loc = 'upper left',plot_end = t_sim, vspan = True, ylim = (-4, 80),
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt= 0.8, axvspan_color='darkseagreen')
fig.set_size_inches((8, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'_2.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'_2.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

# fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start, plot_end = t_sim,
#                             labelsize = 20, title_fontsize = 25, lw  = 1.5, linelengths = 2, n_neuron = 40, include_title = True, set_xlim=True,
#                             axvspan = True, span_start = t_transition, span_end = t_sim, axvspan_color = 'darkseagreen')

# fig.set_size_inches((15, 5), forward=False)
# fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center',fontsize= 18)
# fig_.text(0.03, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 18)
# fig_.savefig(os.path.join(path, 'SNN_raster_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
# fig_.savefig(os.path.join(path, 'SNN_raster_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
#                 orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)



fig_rest, fig_DD = raster_plot_all_nuclei_DD(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '',plot_start=plot_start,
                            labelsize = 10, title_fontsize = 15, lw  = 1.5, linelengths = 2, n_neuron = 40, include_title = True, set_xlim=True,
                            axvspan_color = 'darkseagreen', n = 1000,  ylabel_x = 0.01,
                            t_transition = t_transition, t_sim = t_sim, tick_label_fontsize = 12)

save_pdf_png(fig_rest, os.path.join(path, 'SNN_raster_'+status+'_1.png'), size = (3,5))
save_pdf_png(fig_DD, os.path.join(path, 'SNN_raster_'+status+'_2.png'), size = (3,5))
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
# x_l = 5
# ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(5,55)
ax.axvspan(5,55, alpha = 0.2, color = 'darkseagreen')
ax.set_ylim(-0.1,9)
ax.legend(fontsize = 10, frameon = False)
ax.tick_params(axis='both', labelsize=15)
save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_DD' ),
             size = (5,3))
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
# x_l = 9
# ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(5,55)
ax.set_ylim(-0.1,9)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize = 10, frameon = False)
save_pdf_png(fig, os.path.join(path, 'SNN_spectrum_basal' ),
             size = (5,3))
# ref_nuc_name = 'D2'; 
# find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
#                                   height = 0, ref_nuc_name = ref_nuc_name, start = 0, total_phase = 720,
#                                   only_entrained_neurons =False, troughs = True)
# fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, nuc_order = ['D2', 'STN', 'Arky', 'Proto', 'FSI'],
#                                     density = False, ref_nuc_name = ref_nuc_name, total_phase = 720, projection = None,
#                                     outer=None, fig=None,  title='', tick_label_fontsize=18,
#                                     labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)



D#%% Transition to activated state FSI-D2-GPe + Arky-D2-GPe + STN-GPe collective

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 5)
# plot_end = int((t_transition + 1000)/dt)
t_transition = plot_start + int( t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)] 
length = duration_base[1] - duration_base[0]
duration_mvt = [int(t_sim /dt) - length , int(t_sim /dt)]
duration_last = [int(t_sim /dt) - length , int(t_sim /dt)]
name1 = 'FSI' 
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'
state = 'rest'
name_list = [name1, name2, name3, name4, name5]

g_ext =  0.01 ; G = {}

g = -0.0022
(G[(name2, name1)], G[(name3, name2)], 
 G[(name1, name3)], G[(name2, name4)], 
 G[(name4, name3)], G[(name3, name5)], 
 G[(name5, name3)], G[(name3, name3)]) = g, g*1.1, g, g, g * 0.8, -g*2.5, g*2.5 ,g * 0.1
G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') : [(name3,'1')],
                      (name2, '1'): [(name1,'1'), (name4,'1')],
                      # (name3, '1'): [(name2,'1'), (name5,'1')],
                     (name3, '1'): [(name2,'1'), (name3, '1'), (name5,'1')], # with GP-GP
					   (name4, '1'): [(name3, '1')],
                       (name5, '1'): [(name3, '1')]}


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
low_f, high_f = 8, 70

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 


# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)

# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI.pkl'))


## Run on previously saved data
FR_ext_all_nuclei_rest  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei_rest, normalize_G_by_N=True)

all_FR_list = {name: FR_ext_range[name]['mvt'] for name in list(nuclei_dict.keys()) } 
# change_basal_firing_all_nuclei(Act['mvt'], nuclei_dict)
# receiving_class_dict , FR_ext_all_nuclei_mvt = set_connec_ext_inp(Act['mvt'], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real_DD, receiving_pop_list, nuclei_dict,t_list, 
#                                                                   all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                                                   set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                                                   use_saved_FR_ext= False, normalize_G_by_N = False, state = 'mvt')

# pickle_obj(FR_ext_all_nuclei_mvt, os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI_mvt.pkl'))


## Run on previously saved data
FR_ext_all_nuclei_mvt  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI_mvt.pkl'))
# change_basal_firing_all_nuclei(Act['rest'], nuclei_dict)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, Act['rest'],
                                      A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                      reset_init_dist= True, poisson_prop = poisson_prop, 
                                      normalize_G_by_N= True)  

receiving_class_dict = set_connec_ext_inp(Act['rest'], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', use_saved_FR_ext= True,
                                          FR_ext_all_nuclei_saved = FR_ext_all_nuclei_rest, 
                                          return_saved_FR_ext= False, normalize_G_by_N= False)
nuclei_dict = run_transition_to_mvt_collective_setting(receiving_class_dict, receiving_pop_list, t_list, dt, nuclei_dict,
                                                      FR_ext_all_nuclei_mvt, K_real, N, N_real, Act['mvt'],
                                                      A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, end_of_nonlinearity,t_transition = int(t_transition/dt))

smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
status = 'transition_to_mvt'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_transition, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title ='',
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt= 0.8, axvspan_color='lightskyblue', ylim=(-10,80), 
           legend_loc = 'upper left', vspan =True)
fig.set_size_inches((15, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start, plot_end = t_sim,
                            labelsize = 20, title_fontsize = 25, lw  = 1.5, linelengths = 2, n_neuron = 40, include_title = True, set_xlim=True,
                            axvspan = True, span_start = t_transition, span_end = t_sim, axvspan_color = 'lightskyblue')
fig.set_size_inches((15, 5), forward=False)
fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center',fontsize= 18)
fig_.text(0.03, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 18)
fig_.savefig(os.path.join(path, 'SNN_raster_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig_.savefig(os.path.join(path, 'SNN_raster_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_mvt, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
x_l = 0.75
ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(0,55)
ax.axvspan(0,55, alpha = 0.2, color = 'lightskyblue')
fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_mvt_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(0,55)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+status+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+status+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
ref_nuc_name = 'D2'; 
find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
                                  height = 0, ref_nuc_name = ref_nuc_name, start = 0, total_phase = 720,
                                  only_entrained_neurons =False, troughs = True)
fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, nuc_order = ['D2', 'STN', 'Arky', 'Proto', 'FSI'],
                                    density = False, ref_nuc_name = ref_nuc_name, total_phase = 720, projection = None,
                                    outer=None, fig=None,  title='', tick_label_fontsize=18,
                                    labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)
#%% Transition to activated state Pallidostrital 
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
duration_base = [0, int(t_mvt/dt)]
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
g = -0.5; g_ext =  0.01
G = {}
plot_start = int(t_sim / 5)
t_transition = plot_start + int( t_sim / 4)
# plot_start = 300
# t_transition = 800

duration_base = [int(100/dt), int(t_transition/dt)] 
length = duration_base[1] - duration_base[0]
duration_2 = [int(t_sim /dt) - length, int(t_sim /dt)]

# G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] =  0,0,0

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] =  -1.8 * 10**-4, -3.5* 10**-4, -12 *10**-4 ##FSI 70
G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] =  -1.1 * 10**-4, -3.2* 10**-4, -3.2 *10**-4 ## close to oscillatory regime

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name3:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                      (name3, '1'): [(name2,'1')]}

pop_list = [1]  
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
set_input_from_response_curve = True
save_init = False
noise_variance = {name1 : 0.1, name2: 0.1, name3 : 15}
noise_amplitude = {name1 : 1, name2: 1, name3: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}

filepaths = {name1: 'FSI_A_18-5_N_1000_T_2000_noise_var_1.pkl' ,
             name2: 'D2_A_1-1_N_1000_T_2000_noise_var_0-1.pkl' ,
            name3: name3 + '_N_'+str(N_sim) +'_T_2000_noise_var_15.pkl'}

mvt_init_filepaths ={'Proto': 'Proto_A_22_N_1000_T_2000_noise_var_15.pkl',
               # 'FSI': 'FSI_A_70_N_1000_T_2000_noise_var_10.pkl',
               'FSI': 'FSI_A_32_N_1000_T_2000_noise_var_1.pkl',
               'D2' : 'D2_A_4_N_1000_T_2000_noise_var_0-1.pkl'}


receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
set_init_all_nuclei(nuclei_dict, filepaths = filepaths)#filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method='draw_from_data', set_noise= False)


nuclei_dict = run_transition_to_movement(receiving_class_dict,t_list, dt, nuclei_dict, mvt_init_filepaths, N, N_real, A_mvt, D_mvt,t_mvt,t_transition = int(t_transition/dt))

# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
state = 'transition_to_mvt_transient'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_transition, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = init_method,
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt= 0.8, axvspan_color='lightskyblue', ylim=(-10,80))
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
# find_freq_SNN_not_saving(dt, nuclei_dict, duration_2, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
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
# find_freq_SNN_not_saving(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
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

#%% Transition to DD Pallidostrital 
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
# A['D2'] = 3
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [0, int(t_mvt/dt)]
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
g = -0.5; g_ext =  0.01
G = {}
plot_start = int(t_sim / 5)
t_transition = plot_start + int( t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)] 
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim /dt) - length, int(t_sim /dt)]

# G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] =  0,0,0

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] =  -1.2 * 10**-4, -3.8* 10**-4, -1. *10**-4 ## close to oscillatory regime

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name3:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                      (name3, '1'): [(name2,'1')]}

pop_list = [1]  
init_method = 'heterogeneous'
# init_method = 'homogeneous'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
mem_pot_init_method = 'draw_from_data'
set_input_from_response_curve = True
save_init = False
noise_variance = {name1 : 0.1, name2: 0.1, name3 : 15}
noise_amplitude = {name1 : 1, name2: 1, name3: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = False, mem_pot_init_method=mem_pot_init_method,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method , path = path, save_init = save_init) for i in pop_list]
nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}

filepaths = {name1: name1+ '_N_'+str(N_sim) +'_T_2000.pkl',
             # name2:name2 + '_N_'+str(N_sim) +'_T_2000.pkl', ## A = 3
             name2: 'D2_A_1-1_N_1000_T_2000_noise_var_0-1.pkl' ,
            name3: name3 + '_N_'+str(N_sim) +'_T_2000_noise_var_15.pkl'}
DD_init_filepaths ={'Proto': 'Proto_A_38_N_1000_T_2000_noise_var_10.pkl',
               'FSI': 'FSI_A_24_N_1000_T_2000_noise_var_1.pkl',
               'D2' :'D2_A_6-6_N_1000_T_2000_noise_var_0-1.pkl' }


receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
set_init_all_nuclei(nuclei_dict, filepaths = filepaths)#filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method='draw_from_data', set_noise= False)


nuclei_dict = run_transition_to_DA_depletion(receiving_class_dict,t_list, dt, nuclei_dict, DD_init_filepaths,
                                                K_real_DD, N, N_real, A_DD, A_mvt, D_mvt,t_mvt,t_transition = int(t_transition/dt))

# nuclei_dict = run(receiving_class_dict,t_list, dt,  nuclei_dict)
smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms = 5)
state = 'transition_to_DD'
D_mvt = t_sim - t_transition
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_DD, t_transition, D_mvt, ax = None, title_fontsize=15, plot_start = plot_start, title = init_method,
           include_FR=False, continuous_firing_base_lines=False, plt_mvt=True, alpha_mvt= 0.8, axvspan_color='darkseagreen')
fig.set_size_inches((15, 5), forward=False)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_firing_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig_ = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start, plot_end = t_sim,
                            labelsize = 20, title_fontsize = 25, lw  = 1.5, linelengths = 2, n_neuron = 40, include_title = True, set_xlim=True,
                            axvspan = True, span_start = t_transition, span_end = t_sim, axvspan_color = 'darkseagreen')
fig.set_size_inches((15, 5), forward=False)
fig_.text(0.5, 0.05, 'time (ms)', ha='center', va='center',fontsize= 18)
fig_.text(0.03, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 18)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig_.savefig(os.path.join(path, 'SNN_raster_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_DD, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
x_l = 8
ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(5,55)
ax.axvspan(5,55, alpha = 0.2, color = 'darkseagreen')
fig.savefig(os.path.join(path, 'SNN_spectrum_DD_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_DD_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig, ax = plt.subplots(1,1)
peak_threshold = 0.1; smooth_window_ms = 3 ;smooth_window_ms = 5 ; cut_plateau_epsilon = 0.1; lim_oscil_perc = 10; low_pass_filter = False
find_freq_SNN_not_saving(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold , smooth_kern_window , smooth_window_ms, cut_plateau_epsilon , False , 'fft' , False , 
                low_pass_filter, 0,2000, plot_spectrum = True, ax = ax, c_spec = color_dict, spec_figsize = (6,5), find_beta_band_power = False, 
                fft_method = 'Welch', n_windows = 3, include_beta_band_in_legend = False)
x_l = 8
ax.axhline(x_l, ls = '--', c = 'grey')
ax.set_xlim(5,55)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_spectrum_basal_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

fig,ax = plt.subplots(1,1)
f, t, Sxx = spectrogram(nuc2[0].pop_act[int(plot_start/dt):], 1/(dt/1000))
img = ax.pcolormesh(t, f, 10*np.log(Sxx), cmap = plt.get_cmap('YlOrBr'),shading='gouraud', vmin=-30, vmax=0)
ax.axvline( (t_transition - plot_start)/1000, ls = '--', c = 'grey')
ax.set_ylabel('Frequency (Hz)',  fontsize = 15)
ax.set_xlabel('Time (sec)', fontsize = 15)
ax.set_ylim(0,70)
fig.set_size_inches((15, 5), forward=False)
clb = fig.colorbar(img)
clb.set_label(r'$10Log(PSD)$', labelpad=15, y=0.5, rotation=-90, fontsize = 15)
fig.savefig(os.path.join(path, 'SNN_temporal_spectrum_'+state+'.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path, 'SNN_temporal_spectrum_'+state+'.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

#%% Transition to DD FSI-D2-GPe+ Arky-D2-GPe + STN-GPe collective multi run

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K_DD = calculate_number_of_connections(N, N_real, K_real_DD)
dt = 0.25
t_sim = 8000; t_list = np.arange(int(t_sim/dt))
plot_start = int(t_sim / 5)
# plot_end = int((t_transition + 1000)/dt)
t_transition = plot_start + int( t_sim / 4)
duration_base = [int(100/dt), int(t_transition/dt)] 
length = duration_base[1] - duration_base[0]
duration_DD = [int(t_sim /dt) - length + int(t_transition / 4/ dt), int(t_sim /dt)]
name1 = 'FSI' 
name2 = 'D2'
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'
state = 'rest'
name_list = [name1, name2, name3, name4, name5]

g_ext =  0.01 ; G = {}

g = -0.0025
(G[(name2, name1)], G[(name3, name2)], 
 G[(name1, name3)], G[(name2, name4)], 
 G[(name4, name3)], G[(name3, name5)], 
 G[(name5, name3)], G[(name3, name3)]) = g, g, g, g, g, -g*2.5, g*2.5 ,g * 0.1 * 2



G = { k: v * K_DD[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') : [(name3,'1')],
                      (name2, '1'): [(name1,'1'), (name4,'1')],
                      # (name3, '1'): [(name2,'1'), (name5,'1')],
                     (name3, '1'): [(name2,'1'), (name3, '1'), (name5,'1')], # with GP-GP
					   (name4, '1'): [(name3, '1')],
                       (name5, '1'): [(name3, '1')]}


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
low_f, high_f = 8, 30

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight ) for i in pop_list] for name in name_list}

n_FR = 20

all_FR_list = {name: FR_ext_range[name]['DD'] for name in list(nuclei_dict.keys()) } 
change_basal_firing_all_nuclei(Act['DD'], nuclei_dict)
# receiving_class_dict , FR_ext_all_nuclei_DD = set_connec_ext_inp(Act['DD'], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real_DD, receiving_pop_list, nuclei_dict,t_list, 
#                                                                   all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                                                   set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                                                   use_saved_FR_ext= False, normalize_G_by_N = False, state = 'DD')

# pickle_obj(FR_ext_all_nuclei_DD, os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI_DD.pkl'))


## Run on previously saved data
FR_ext_all_nuclei_DD  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-Arky-D2-FSI_DD.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act['DD'], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real_DD, receiving_pop_list, nuclei_dict,t_list, 
                                                                  all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                                  set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                                                  use_saved_FR_ext= True, normalize_G_by_N = False, state = 'DD', FR_ext_all_nuclei_saved = FR_ext_all_nuclei_DD)


# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 5; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True

# save_figures = True ; save_pkl = True
round_dec = 1 ; include_std = False
plot_start =  int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
check_peak_significance = False

filename = 'All_nuc_DD_N_1000_T_2000_n' + str(n_run) + '_runs' + '.pkl'

G_dict = { k: [v] for k, v in G.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = [ 'Proto', 'STN']
ref_nuc_name = 'D2'
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act['DD'], A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei_DD, check_peak_significance = check_peak_significance, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = low_f, high_f = high_f, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = ref_nuc_name, save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, 
                                                    nuc_order = nuc_order)

status = 'transition_to_DD'
#%% Synapric weight exploraion (STN-GPe)  resetting inti dists and setting ext input collectively
plt.close('all')


N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.1
t_sim = 3000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'Proto'
name2 = 'STN'
name_list = [name1, name2]
state = 'rest'
g_ext =  0.01
g = 0
G = {}

G[(name1, name2)] , G[(name2, name1)] = -g, g

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name2,'1')],
                       (name2, '1'): [(name1,'1')]
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, 
               save_init = save_init, syn_component_weight=syn_component_weight, noise_method = noise_method ) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)



n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = True
# n_run = 5; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True

# save_figures = True ; save_pkl = True
round_dec = 1 ; include_std = False
plot_start =  int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
check_peak_significance = True
x = np.array([0, 0.5, 1])
g = -0.0118  # start
n = len(x)

G_dict = {(name1, name2) : -g * x ,
          (name2, name1) : g * x
        }

filename = 'STN_Proto_N_1000_T_2000_' + str(n) + '_pts_' + str(n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') + '_' + noise_method + '.pkl'

G_dict = { k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = [ 'Proto', 'STN']
ref_nuc_name = 'Proto'
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, check_peak_significance = check_peak_significance, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = 8, high_f = 40, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = ref_nuc_name, save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, 
                                                    nuc_order = nuc_order, len_f_pxx = 150)

# pickle_obj(data, filepath)
def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [
          str(round(G[('STN', 'Proto')][0],1)) + '_' + str(round(G[('STN', 'Proto')][-1]*scale,1)),
          str(round(G[('Proto', 'STN')][0],1)) + '_' + str(round(G[('Proto', 'STN')][-1]*scale,1))]


    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    filename = ( names[0] + '_' + names[1]  + '_G(PS)=' + gs[0]+ '_G(SP)=' + gs[1] +
             '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + 
              '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
        
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_', 'Phase_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(8, 6), (5, 5) , (10, 6), (4, 12)]

if save_figures:
 	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*4, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
#%% Synapric weight exploraion (GPe-GPe)  resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 5000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'Proto'
state = 'rest'

name_list = [name1]
g_ext =  0.01
G = {}

G[(name1, name1)] = - 0.01 

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name1,'1')]
                       }
					 


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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, 
               save_init = save_init, syn_component_weight=syn_component_weight, noise_method = noise_method ) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)



# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 10; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True

# save_figures = True ; save_pkl = True
round_dec = 1 ; include_std = False
plot_start =  int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'
low_f = 8; high_f = 70

x = np.array([ 1])

n = len(x)
g = -0.005  # start


G_dict = {(name1, name1) : g * x
         
        }

filename = 'Proto_Proto_N_1000_T_5000_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = { k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = [ 'Proto']
ref_nuc_name = 'Proto'
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = [30,70], n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, check_peak_significance = False, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = low_f, high_f = high_f, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = ref_nuc_name, save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, 
                                                    nuc_order = nuc_order)

# pickle_obj(data, filepath)
def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [
          str(round(G[('Proto', 'Proto')][0],1)) + '_' + str(round(G[('Proto', 'Proto')][-1]*scale,1))]

    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    filename = ( names[0] +  '_G(PP)= '  + gs[0] + 
             '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + 
              '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
        
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_', 'Phase_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(8, 6), (5, 5) , (10, 6), (4, 12)]

if save_figures:
 	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*4, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
#%% synaptic weight exploration SNN (FSI-D2-Proto) resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 3000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
state = 'rest'
name_list = [name1, name2, name3]

g = 0
G = {}

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] = g,g,g
G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':0.01} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1')]}
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
noise_tau = 20 

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight, noise_method = noise_method) for i in pop_list] for name in name_list}
######### Set ext inp for individual neurons
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)
# nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
#                                       t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)

######## Set ext inp collectively
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 
receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)



n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True; plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
# n_run = 5; plot_firing = False; plot_spectrum= False; plot_raster = False; plot_phase = False; low_pass_filter= False ;save_pkl = True ; save_figures = False; save_pxx = True
round_dec = 1 ; include_std = False
plot_start = 1500# int(t_sim/2)
plot_raster_start =  1500# int(t_sim/2)
n_neuron = 30
legend_loc = 'center right'
# x = np.flip(np.geomspace(-40, -0.1, n))

x = np.array([0, 0.5, 1 ])

# x = np.linspace(.1, 1.5, 20)

n = len(x)

g = -0.008  # start

#### 0.0045, 2.5, 0.5, 0.5 ,x = 1.5
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

G_dict = {(name2, name1): 1.4 * g * x ,
          (name3, name2): 1.2 * g * x , 
          (name1, name3): g * x  }
filename = 'D2_Proto_FSI_N_1000_T_5000_G_all_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '_dt_' + str(dt).replace('.', '-') + '_' + noise_method + '.pkl'

G_dict = { k: v * K[k] for k, v in G_dict.items()}

G_FSI_loop = G_dict[(name1, name3)] * G_dict[(name2, name1)] * G_dict[(name3, name2)]


fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
ref_nuc_name = 'Proto'
nuc_order = ['Proto', 'D2', 'FSI']
figs, title, data = synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, check_peak_significance = False, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = 8, high_f = 30, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = ref_nuc_name, save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, 
                                                    nuc_order = nuc_order, len_f_pxx= 150)

def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0],3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale,3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '--' + str(round(G[('Proto', 'D2')][-1]*scale,3)), 
          str(round(G[('FSI', 'Proto')][0],3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale,3))]
    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + '_G(FD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PF)= '  + gs[2] + 
              '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_', 'Phase_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(8, 13), (5, 10) , (6, 12), (6, 12)]
s = [(8, 4), (5, 5) , (6, 3), (4, len(name_list))]

if save_figures:
	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*4, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

#%% synaptic weight exploration SNN (Arky-D2-Proto) resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 5000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'Arky' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
state = 'rest'
name_list = [name1, name2, name3]

g_ext =  0.01
g = 0
G = {}

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)] = g,g,g
G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1')]}
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


nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init,
               syn_component_weight=syn_component_weight) for i in pop_list] for name in name_list}

######### Set ext inp for individual neurons
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
#              'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
#             # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
#             'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl'}

# set_init_all_nuclei(nuclei_dict, filepaths = filepaths)
# nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
#                                       t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)

######## Set ext inp collectively
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
                                          use_saved_FR_ext= False, normalize_G_by_N= True)

pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_D2-Proto-Arky.pkl'))


## Run on previously saved data
# FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_D2-Proto-Arky.pkl'))


# receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
#                                           use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N = True)

# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True; plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = True
n_run = 10; plot_firing = False; plot_spectrum= False; plot_raster = False; plot_phase = False; low_pass_filter= False ;save_pkl = True ; save_figures = False; save_pxx = True
round_dec = 1 ; include_std = False
plot_start = 1500# int(t_sim/2)
plot_raster_start =  1500# int(t_sim/2)
n_neuron = 30
legend_loc = 'center right'
# x = np.flip(np.geomspace(-40, -0.1, n))

x = np.array([ 1])

# x = np.linspace(.1, 1.5, 20)

n = len(x)

g = -0.006  # start

#### 0.0045, 2.5, 0.5, 0.5 ,x = 1.5
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

G_dict = {(name2, name1): 4 * g * x ,
          (name3, name2): g * x , 
          (name1, name3): 0.6 * g * x  }
filename = 'D2_Proto_Arky_N_1000_T_5000_G_all_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = { k: v * K[k] for k, v in G_dict.items()}

G_FSI_loop = G_dict[(name1, name3)] * G_dict[(name2, name1)] * G_dict[(name3, name2)]


fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['D2', 'Proto', 'Arky']
ref_nuc_name = 'Proto'
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, check_peak_significance = False, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = 8, high_f = 30, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = ref_nuc_name, save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, 
                                                    nuc_order = nuc_order)

def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'Arky')][0],3)) + '--' + str(round(G[('D2', 'Arky')][-1]*scale,3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '--' + str(round(G[('Proto', 'D2')][-1]*scale,3)), 
          str(round(G[('Arky', 'Proto')][0],3)) + '--' + str(round(G[('Arky', 'Proto')][-1]*scale,3))]
    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + '_G(AD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PA)= '  + gs[2] + 
              '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_', 'Phase_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        
s = [(8, 13), (5, 10) , (6, 12), (6, 12)]
s = [(8, 4), (5, 5) , (6, 3), (4, len(name_list))]

if save_figures:
	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*4, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()


#%% Synapric weight exploraion STN-GPe + FSI-D2-GPe individual neurons 

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]

g_ext =  0.01
g = 0
G = {}

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name3, name4)], G[(name4, name3)] = g, g, g, -g, g

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1'), (name4, '1')],
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3, name4: nuc4}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)


filepaths = {'FSI': 'tau_m_9-5_FSI_A_18-5_N_1000_T_2000_noise_var_8.pkl' ,
             'D2': 'tau_m_13_D2_A_1-1_N_1000_T_2000_noise_var_3.pkl' ,
            # 'Proto': 'tau_m_20_Proto_A_45_N_1000_T_2000_noise_var_105.pkl'}
            'Proto': 'tau_m_12-94_Proto_A_45_N_1000_T_2000_noise_var_30.pkl',
			'STN' : 'tau_m_5-13_STN_A_15_N_1000_T_2000_noise_var_4.pkl'}

set_init_all_nuclei(nuclei_dict, filepaths = filepaths)
nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, 
                                      t_mvt, t_list, dt, mem_pot_init_method=mem_pot_init_method, set_noise= False)

n = 3 ; n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True; low_pass_filter= False ; save_pkl = False 
# plot_firing = False; plot_spectrum= False; plot_raster = False; save_pkl = True ; save_figures = False
save_figures = True
save_pkl = True
round_dec = 1 ; include_std = False
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

G_dict = {(name2, name1) : np.array( [g*1.5]* (n)) ,
          (name3, name2) : np.array( [g*1.5]* (n)) , 
          (name1, name3) : np.array( [g*1.5]* (n)),
		  (name3, name4) :  -g * x   ,
		  (name4, name3) : np.array( [g*1.5]* (n))}
filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_and_Proto_STN_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)

# figs, title, data = synaptic_weight_exploration_SNN_all_changing(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
#                                                     noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter, legend_loc = legend_loc,
#                                                     lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = [-10,70], n_run = n_run,  plot_start_raster= plot_raster_start, 
#                                                     plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
#                                                     find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=False, save_pkl = save_pkl)

##### Note: "synaptic_weight_exploration_SNN" when signal is all plateau it has problem saving empty f array to a designated 200 array.
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = [-10,70], n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0],3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale,3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '--' + str(round(G[('Proto', 'D2')][-1]*scale,3)), 
          str(round(G[('FSI', 'Proto')][0],3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale,3))]
    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + '_G(FD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PF)= '  + gs[2] + 
              '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(15, 15), (5, 15) , (6, 11)]
s = [(8, 13), (5, 10) , (6, 12)]

if save_figures:
	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*3, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
#%% Synapric weight exploraion STN-GPe + FSI-D2-GPe resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 5000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]
state = 'rest'
g_ext =  0.01
g = 0
G = {}

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name3, name4)], G[(name4, name3)] = g, g, g, -g, g

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1'), (name4, '1')],
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
noise_method = 'Gaussian'
noise_method = 'Ornstein-Uhlenbeck'
use_saved_FR_ext = True

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, 
               save_init = save_init, syn_component_weight=syn_component_weight, noise_method = noise_method ) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)



n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
# n_run = 5; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True
# save_figures = True ; save_pkl = True
round_dec = 1 ; include_std = False
plot_start =  int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'


# coef = 2
# x = np.array([ 1/4 , 4.5/4 , 6/4, 7./4])
# g = -0.005  # start
# n = len(x)

# x0 = np.array([1.5, 1.5, 1.5, 1.5])
# x1 = np.array([1.5, 1.5, 1.5, 1.5])
# x2 = np.array([1.5, 1.5, 1.3, 1])
# x3 = np.array([0.25, 3, 3.5, 4])
# x4 = np.array([0.5, 1, 3, 3])

x0 = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
x1 = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
x2 = np.array([1.5, 1.5, 1.4, 1.45, 1.3, 1.2, 1.1, 1])
x3 = np.array([0.25, 3, 3.1, 3.2, 3.5, 3.6, 3.7, 4])
x4 = np.array([0.5, 1, 1.8, 2.6, 3, 3, 3, 3])
g = -0.0035  # start
n = len(x0)

g = -.0045; coef = 2
# for k,v in data['g'].items():
#     print(k)
#     print(v/K[k])

x = np.array([0, 1.7, 3.8])

# x = np.linspace(0, 3.8, 15)

n = len(x)
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

G_dict = {(name2, name1) : np.array( [g]* (n)) ,
          (name3, name2) : np.array( [g]* (n)) * 1.5, 
          (name1, name3) : np.array( [g]* (n)) ,
		  (name3, name4) :  -g *  x  ,
		  (name4, name3) :   np.array( [g * coef]* (n))  }
filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

# G_dict = {(name2, name1) : g *  x0,
#           (name3, name2) : g * x1, 
#           (name1, name3) : g * x2,
# 		  (name3, name4) : -g  * x3,
# 		  (name4, name3) : g * x4}

# filename = 'D2_Proto_FSI_STN_N_1000_T_2000_G_all_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = { k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['D2', 'STN', 'Proto', 'FSI']
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, check_peak_significance = False, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = 8, high_f = 35, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = 'D2',save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, nuc_order = nuc_order)

# pickle_obj(data, filepath)
def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0],3)) + '_' + str(round(G[('D2', 'FSI')][-1]*scale,3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '_' + str(round(G[('Proto', 'D2')][-1]*scale,3)), 
          str(round(G[('FSI', 'Proto')][0],3)) + '_' + str(round(G[('FSI', 'Proto')][-1]*scale,3)),
          str(round(G[('Proto', 'STN')][0],3)) + '_' + str(round(G[('Proto', 'STN')][-1]*scale,3))]

    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + names[3] + '_G(FD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PF)= '  + gs[2] + '_G(SP)= '  + gs[3] + 
             '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + 
              '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_', 'Phase_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(8, 13), (5, 10) , (6, 12), (6, 12)]

if save_figures:
 	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*4, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

#%% Synapric weight exploraion STN-GPe + Arky resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 5000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'Arky' # projecting
name2 = 'Proto'
name3 = 'STN'
name_list = [name1, name2, name3]
state = 'rest'
g_ext =  0.01
g = 0
G = {}

G[(name1, name2)] , G[(name3, name2)] , G[(name2, name3)], G[(name2, name2)]= g, g, -g, g*.2
G[(name1, name2)] , G[(name3, name2)] , G[(name2, name3)]= g, g, -g

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name2,'1')],
                       (name2, '1'): [(name3,'1')],#, (name3, '1')],
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False, normalize_G_by_N = True)
# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_STN-Proto-Arky.pkl'))


# Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_STN-Proto-Arky.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True
n_run = 20; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True
# save_figures = True ; save_pkl = True
round_dec = 1 ; include_std = False
plot_start =  int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'

# x = np.linspace(0, 14/4, 20)
coef = 2
x = np.array([ 0 , 4.5/4 , 6/4, 7.5/4])
x = np.array([   1 ])

n = len(x)
g = -0.01  # start


G_dict = {(name1, name2) : np.array( [g]* (n)) ,
          (name3, name2) : np.array( [g]* (n)) , 
          (name2, name3) : np.array( [-g]* (n))}
           # (name2, name2) : np.array( [g]* (n)) * 0.2}

filename = 'STN_Proto_Arky_N_1000_T_2000_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = { k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['STN', 'Proto', 'Arky']
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, check_peak_significance = True, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = 8, high_f = 70, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = 'STN',save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, nuc_order = nuc_order)

# pickle_obj(data, filepath)
def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('Arky', 'Proto')][0],3)) + '_' + str(round(G[('Arky', 'Proto')][-1]*scale,3)), 
          str(round(G[('STN', 'Proto')][0],3)) + '_' + str(round(G[('STN', 'Proto')][-1]*scale,3)),
          str(round(G[('Proto', 'STN')][0],3)) + '_' + str(round(G[('Proto', 'STN')][-1]*scale,3))]

    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] )
    
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_', 'Phase_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(8, 6), (5, 5) , (10, 6), (4, 12)]

if save_figures:
 	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*4, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
#%% Synapric weight exploraion STN-GPe-GPe + Arky resetting inti dists and setting ext input collectively
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 5000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'Arky' # projecting
name2 = 'Proto'
name3 = 'STN'
name_list = [name1, name2, name3]
state = 'rest'
g_ext =  0.01
g = 0
G = {}

G[(name1, name2)] , G[(name3, name2)] , G[(name2, name3)], G[(name2, name2)]= g, g, -g, g*.2

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name2,'1')],
                       (name2, '1'): [(name3,'1'), (name3, '1')],
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, 
               save_init = save_init, syn_component_weight=syn_component_weight, noise_method = noise_method ) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)



# n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True;plot_phase = True; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
n_run = 20; plot_firing = False; plot_spectrum= False; plot_raster = False;plot_phase = False; low_pass_filter= False; save_pkl = True ; save_figures = False; save_pxx = True

# save_figures = True ; save_pkl = True
round_dec = 1 ; include_std = False
plot_start =  int(t_sim * 3/4)
plot_raster_start = int(t_sim * 3/4)
n_neuron = 50
legend_loc = 'center right'

x = np.array([   1 ])

n = len(x)
g = -0.005  # start


G_dict = {(name1, name2) : np.array( [g]* (n)) ,
          (name3, name2) : np.array( [g]* (n)) , 
          (name2, name3) : np.array( [-g]* (n)),
            (name2, name2) : np.array( [g]* (n)) * 0.2}

filename = 'STN_Proto_Proto_Arky_N_1000_T_2000_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = { k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['STN', 'Proto', 'Arky']
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, check_peak_significance = False, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = 8, high_f = 70, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = 'STN',save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, nuc_order = nuc_order)

# pickle_obj(data, filepath)
def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('Arky', 'Proto')][0],3)) + '_' + str(round(G[('Arky', 'Proto')][-1]*scale,3)), 
          str(round(G[('STN', 'Proto')][0],3)) + '_' + str(round(G[('STN', 'Proto')][-1]*scale,3)),
          str(round(G[('Proto', 'STN')][0],3)) + '_' + str(round(G[('Proto', 'STN')][-1]*scale,3)),
          str(round(G[('Proto', 'Proto')][0],3)) + '_' + str(round(G[('Proto', 'Proto')][-1]*scale,3))]

    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    filename = ( names[0] + '_' + names[1] + '_'+  names[2]  + '_G(PA)=' + gs[0]+ '_G(PS)=' +gs[1] + '_G(SP)= '  + gs[2] + '_G(PP)= '  + gs[3] + 
             '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + 
              '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
        
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_', 'Phase_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Syn_g_explore_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(8, 6), (5, 5) , (10, 6), (4, 12)]

if save_figures:
 	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*4, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()


#%% Synapric weight exploraion STN-GPe-GPe + FSI-D2-GPe-GPe resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)

dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'STN'
name_list = [name1, name2, name3, name4]

state = 'rest'
g_ext =  0.01
g = 0
G = {}

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name3, name4)], G[(name4, name3)], G[(name3, name3)] , = g, g, g, -g, g, g
poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1')],
                       (name3, '1'): [(name2,'1'), (name4, '1'), (name3, '1')],
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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, 
               save_init = save_init, syn_component_weight=syn_component_weight, noise_method = noise_method ) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)



n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True; low_pass_filter= False ; save_pkl = False ; save_figures = True
# plot_firing = False; plot_spectrum= False; plot_raster = False; save_pkl = True ; save_figures = False

round_dec = 1 ; include_std = False
plot_start =  int(t_sim/2)
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

G_dict = {(name2, name1) : g * x,
          (name3, name2) : g * x, 
          (name1, name3) : g * x,
          (name3, name3) : g * x,
		  (name3, name4) : -g * x,
		  (name4, name3) : g * x }
filename = 'D2_Proto_Proto_FSI_STN_N_1000_T_1000_G_all_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'

G_dict = { k: v * K[k] for k, v in G_dict.items()}

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)

# figs, title, data = synaptic_weight_exploration_SNN_all_changing(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
#                                                     noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter, legend_loc = legend_loc,
#                                                     lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = [-10,70], n_run = n_run,  plot_start_raster= plot_raster_start, 
#                                                     plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
#                                                     find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=False, save_pkl = save_pkl)

##### Note: "synaptic_weight_exploration_SNN" when signal is all plateau it has problem saving empty f array to a designated 200 array.
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, 
                                                    include_std = include_std, find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, 
                                                    save_pkl = save_pkl, reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, decimal = 1, divide_beta_band_in_power=True,
                                                    spec_lim = [0, 65], include_FR = False)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0],3)) + '_' + str(round(G[('D2', 'FSI')][-1]*scale,3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '_' + str(round(G[('Proto', 'D2')][-1]*scale,3)), 
          str(round(G[('FSI', 'Proto')][0],3)) + '_' + str(round(G[('FSI', 'Proto')][-1]*scale,3)),
          str(round(G[('Proto', 'STN')][0],3)) + '_' + str(round(G[('Proto', 'STN')][-1]*scale,3))]

    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + names[3] + '_G(FD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PF)= '  + gs[2] + '_G(SP)= '  + gs[3] + 
             '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + 
              '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(15, 15), (5, 15) , (6, 11)]
s = [(8, 13), (5, 10) , (6, 12)]

if save_figures:
 	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*3, s = s)

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

#%% Synapric weight exploraion Arky-D2-Proto + FSI-D2-GPe resetting inti dists and setting ext input collectively

import sys
# sys.modules[__name__].__dict__.clear()
runcell('Constants', '/home/shiva/BG_Oscillations/Oscillation.py')
plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'Arky'
state = 'rest'
name_list = [name1, name2, name3, name4]

g_ext =  0.01
g = 0
G = {}

G[(name2, name1)] , G[(name3, name2)] , G[(name1, name3)], G[(name2, name4)], G[(name4, name3)] = g, g, g, g, g

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') :  [(name3,'1')],
                      (name2, '1'): [(name1,'1'), (name4,'1')],
                       (name3, '1'): [(name2,'1')],
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
noise_amplitude = {name1 : 1, name2: 1, name3: 1, name4 : 1}

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, save_init = save_init ) for i in pop_list] for name in name_list}

n_FR = 20
all_FR_list = {name: FR_ext_range[name] for name in list(nuclei_dict.keys()) } 


# receiving_class_dict , FR_ext_all_nuclei = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
#                                           all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
#                                           set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= True, 
#                                           use_saved_FR_ext= False)

# pickle_obj(FR_ext_all_nuclei, os.path.join(path, 'FR_ext_Arky-D2-Proto-FSI.pkl'))


## Run on previously saved data
FR_ext_all_nuclei  = load_pickle( os.path.join(path, 'FR_ext_Arky-D2-Proto-FSI.pkl'))
receiving_class_dict  = set_connec_ext_inp(Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = 35, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= True, FR_ext_all_nuclei_saved=FR_ext_all_nuclei, normalize_G_by_N=True)

n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =True; low_pass_filter= False ; save_pkl = False ; save_figures = True
# n_run = 5; plot_firing = False; plot_spectrum= False; plot_raster = False;  low_pass_filter= False ;save_pkl = True ; save_figures = False
# save_figures = True
# save_pkl = False
round_dec = 1 ; include_std = False
plot_start = 1500# int(t_sim/2)
plot_raster_start =  1500# int(t_sim/2)
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

G_dict = {(name2, name1) : g * x ,
          (name3, name2) : g * x , 
          (name1, name3) : g * x , 
		  (name2, name4) : g * x   ,
		  (name4, name3) : g * x}
filename = 'D2_Proto_FSI_Arky_N_1000_T_1000_G_all_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '.pkl'
name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'Arky'
G_dict = { k: v * K[k] for k, v in G_dict.items()}
G_FSI_loop = G_dict[(name1, name3)] * G_dict[(name2, name1)] * G_dict[(name3, name2)]
G_Arky_loop = G_dict[(name2, name4)] * G_dict[(name4, name3)] * G_dict[(name3, name2)]
G_loop = G_FSI_loop + G_Arky_loop

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)

figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, state = state, K_real = K_real, N_real = N_real, N = N,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0],3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale,3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '--' + str(round(G[('Proto', 'D2')][-1]*scale,3)), 
          str(round(G[('FSI', 'Proto')][0],3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale,3))]
    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + '_G(FD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PF)= '  + gs[2] + 
              '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(8, 13), (5, 10) , (6, 12)]
s = [(13, 8), (5, 10) , (6, 12)]

if save_figures:
 	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*3, s = s)

# manager = plt.get_current_fig_manager()

#%% Synapric weight exploraion Arky-D2-Proto + FSI-D2-GPe + STN-GPe resetting inti dists and setting ext input collectively

plt.close('all')
N_sim = 1000
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.25
t_sim = 5000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
duration_base = [int(400/dt), int(t_sim/dt)]

name1 = 'FSI' # projecting
name2 = 'D2' # recieving
name3 = 'Proto'
name4 = 'Arky'
name5 = 'STN'
state = 'rest'
name_list = [name1, name2, name3, name4, name5]

g_ext =  0.01
g = 0
G = {}


(G[(name2, name1)] , G[(name3, name2)], 
 G[(name1, name3)], G[(name2, name4)], 
 G[(name4, name3)], G[(name3, name5)], 
 G[(name5, name3)], G[(name3, name3)]) = g, g , g, g, g , -g * 3, g * 3 ,g * 0.1

G = { k: v * K[k] for k, v in G.items()}

poisson_prop = {name : {'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext} for name in name_list}

receiving_pop_list = {(name1,'1') : [(name3,'1')],
                      (name2, '1'): [(name1,'1'), (name4,'1')],
                      # (name3, '1'): [(name2,'1'), (name5,'1')],
                     (name3, '1'): [(name2,'1'), (name3, '1'), (name5,'1')], # with GP-GP
					   (name4, '1'): [(name3, '1')],
                       (name5, '1'): [(name3, '1')]}

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

nuclei_dict = {name:  [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',set_input_from_response_curve = set_input_from_response_curve,
               poisson_prop =poisson_prop,init_method = init_method, der_ext_I_from_curve = der_ext_I_from_curve, mem_pot_init_method=mem_pot_init_method,  keep_mem_pot_all_t = keep_mem_pot_all_t,
               ext_input_integ_method=ext_input_integ_method,syn_input_integ_method = syn_input_integ_method, path = path, 
               save_init = save_init, syn_component_weight=syn_component_weight, noise_method = noise_method ) for i in pop_list] for name in name_list}
n_FR = 20
all_FR_list = {name: FR_ext_range[name][state] for name in list(nuclei_dict.keys()) } 

receiving_class_dict = set_connec_ext_inp(path, Act[state], A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory = False, method = 'collective', save_FR_ext=True,
                                          use_saved_FR_ext= use_saved_FR_ext, normalize_G_by_N=True, state = state)



n_run = 1; plot_firing = True; plot_spectrum= True; plot_raster =False; plot_phase = False; low_pass_filter= False ; save_pkl = False ; save_figures = True; save_pxx = False
# n_run = 8; plot_firing = False; plot_spectrum= False; plot_raster = False; plot_phase = False; low_pass_filter= False ;save_pkl = True ; save_figures = False; save_pxx = True
# save_figures = True
# save_pkl = False
round_dec = 1 ; include_std = False
# plot_start = 1600# int(t_sim/2)
# plot_raster_start =  1600# int(t_sim/2)
plot_start = 1500
plot_raster_start =  1500
n_neuron = 30
legend_loc = 'center right'
g = -0.004


##### sd = 14 D2 u-rest
# g, g , g, g, g , -g * 3, g * 3 ,g * 0.1 --> 30 Hz
# g, 2*g, g, g, g, -g * 3, g * 3 ,g * 0.1 --> 22 Hz
# g, 2*g, g, g, g, -g * 3, g*2.5 ,g * 0.1 --> 20 Hz
# g, 2*g, g, g, g, -g * 3, g*2 ,g * 0.1 --> 18 Hz
# g, 2*g, g, g, g, -g * 3, g ,g * 0.1 --> 15 Hz

# x2 = np.array([1, 2, 2, 2, 2, 1]) 
# x6 = np.array([2, 3, 3, 3, 3, 3]) 
# x7 = np.array([1, 1, 2, 2.5, 3, 3 ])
# x8 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) 
# n = len(x6)
##### D2 RMP var = 1 t = 5s
# 2*g,0.5* g, 0.5*g, g, g, -g * 3, g * 3 ,g * 0.1 --> 33 Hz
# g, 2*g, g, g, g, -g * 3, g*2.5 ,g * 0.1 --> 18.8 Hz double peak with 35
# g, 1.5*g, g, g, g, -g * 3, g*2.5 ,g * 0.1 --> 18.8 Hz almost single peak
# g, 2*g, g, g, g, -g * 3, g*2 ,g * 0.1 --> 17.6 Hz single peak
# g, 1.8*g, g, g, g, -g * 3, g*1.5 ,g * 0.1 --> 16.9 Hz
# g, 2*g, g, g, g, -g * 3, g ,g * 0.1 --> 15 Hz  works

############## 15, 18.8, 33 Hz
x1 = np.array([1, 1, 2]) 
x2 = np.array([2, 1.5, 0.5]) 
x3 = np.array([1, 1, 0.5]) 
x6 = np.array([3, 3, 3]) 
x7 = np.array([1, 2.5, 3])
x8 = np.array([0.1, 0.1, 0.1]) 
n = len(x6)
# x1 = x3 = x4 = x5 = np.ones(n)
x4 = x5 = np.ones(n)

m = 0
x1 = np.array([x1[m]]) ; x2 = np.array([x2[m]]); x3 = np.array([x3[m]]); x4 = np.array([x4[m]]); x5 =np.array( [x5[m]]); 
x6 = np.array([x6[m]]); x7 =np.array([ x7[m]]); x8 = np.array([ x8[m]])

n = len(x6)
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

G_dict = {(name2, name1) : np.array([g]* (n)) * x1,
          (name3, name2) : np.array([g]* (n)) * x2, 
          (name1, name3) : np.array([g]* (n)) * x3, 
		  (name2, name4) : np.array([g]* (n)) * x4,
		  (name4, name3) : np.array([g]* (n)) * x5,
          (name3, name5) : -g * x6,
          (name5, name3) : np.array([g]* (n)) * x7,
          (name3, name3): np.array([g]* (n)) * x8}

# G_dict = {(name2, name1) : g * y,
#           (name3, name2) : g * y , 
#           (name1, name3) : g * y , 
# 		  (name2, name4) : g * y ,
# 		  (name4, name3) : g * y ,
#           (name3, name5) : -g * x,
#           (name5, name3) : g * x}
filename = 'STN_D2_Proto_FSI_Arky_N_1000_T_5000_G_all_changing_' + str(n) + '_pts_' + str(n_run) + '_runs' + '_14-3Hz-2.pkl'

# pickle_obj(data, os.path.join(path, 'Beta_power', filename))
G_dict = { k: v * K[k] for k, v in G_dict.items()}
G_FSI_loop = G_dict[(name1, name3)] * G_dict[(name2, name1)] * G_dict[(name3, name2)]
G_Arky_loop = G_dict[(name2, name4)] * G_dict[(name4, name3)] * G_dict[(name3, name2)]
G_loop = G_FSI_loop + G_Arky_loop

fft_method = 'Welch'
filepath = os.path.join(path, 'Beta_power', filename)
nuc_order = ['D2', 'STN', 'Arky', 'Proto', 'FSI']
figs, title, data = synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, 
                                                    noise_amplitude, noise_variance, lim_oscil_perc = 10, plot_firing = plot_firing, low_pass_filter= low_pass_filter,legend_loc = legend_loc,
                                                    lower_freq_cut= 8, upper_freq_cut = 40, set_seed = False, firing_ylim = None, n_run = n_run,  plot_start_raster= plot_raster_start, 
                                                    plot_spectrum= plot_spectrum, plot_raster = plot_raster, plot_start = plot_start, plot_end = t_sim, n_neuron= n_neuron, round_dec = round_dec, include_std = include_std,
                                                    find_beta_band_power = True, fft_method= fft_method, n_windows = 3, include_beta_band_in_legend=True, save_pkl = save_pkl,
                                                    reset_init_dist = True, all_FR_list = all_FR_list , n_FR = n_FR, if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                                    state = state, K_real = K_real, N_real = N_real, N = N, divide_beta_band_in_power=True,
                                                    receiving_pop_list = receiving_pop_list, poisson_prop = poisson_prop, return_saved_FR_ext= False, 
                                                    use_saved_FR_ext=True, FR_ext_all_nuclei_saved = FR_ext_all_nuclei, check_peak_significance = False, 
                                                    find_phase = True, phase_thresh_h = 0, filter_order = 6, low_f = 8, high_f = 38, 
                                                    n_phase_bins = 70, start_phase = int(t_sim/4), ref_nuc_name = 'D2', save_pxx = save_pxx,
                                                    plot_phase = plot_phase, total_phase = 720, phase_projection = None, troughs = True, 
                                                    nuc_order = nuc_order)


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale= 1):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0],3)) + '--' + str(round(G[('D2', 'FSI')][-1]*scale,3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '--' + str(round(G[('Proto', 'D2')][-1]*scale,3)), 
          str(round(G[('FSI', 'Proto')][0],3)) + '--' + str(round(G[('FSI', 'Proto')][-1]*scale,3))]
    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + '_G(FD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PF)= '  + gs[2] + 
              '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(figs,nuclei_dict,  G, noise_variance, path, fft_method, pre_prefix = ['']*4, s= [(15,15)]*4, scale = 1):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_', 'Phase_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method, scale = scale)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
s = [(8, 13), (5, 10) , (6, 12), (6, 12), (6, 12)]

if save_figures:
 	save_figs(figs, nuclei_dict, G_dict, noise_variance, path, fft_method, pre_prefix = ['Dem_norm_']*4, s = s)

#%% Beta power vs. G low and high beta separate
plt.close('all')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def _str_G_with_key(key):
    return r'$G_{' + list(key)[1] + '-' + list(key)[0] + r'}$'

# title = (r"$G_{"+list(G_dict.keys())[0][0]+"-"+list(G_dict.keys())[0][1]+"}$ = "+ str(round(list(G_dict.values())[0][0],2)) +
#         r"  $G_{"+list(G_dict.keys())[2][0]+"-"+list(G_dict.keys())[2][1]+"}$ ="+str(round(list(G_dict.values())[2][0],2)))
     
title = ""
n_nuclei = 4
g_cte_ind = [0,0,0]; g_ch_ind = [1,1,1]
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_N_1000_T_2000_G_D2_Proto_changing_15_pts_2_runs.pkl')]; key = ('Proto','D2')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_N_1000_T_2000_G_FSI_D2_changing_15_pts_2_runs.pkl')]; key = ('D2','FSI')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2 _Proto_FSI_N_1000_T_2000_G_Proto_FSI_changing_15_pts_2_runs.pkl')]; key = ('FSI', 'Proto')

filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_FSI_D2_changing_20_pts_5_runs.pkl')]; key = ('D2', 'FSI')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_Proto_FSI_changing_20_pts_5_runs.pkl')]; key = ('FSI', 'Proto')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_Proto_STN_changing_20_pts_5_runs.pkl')]; key = ('STN', 'Proto')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_20_pts_5_runs.pkl')]; key = ('Proto', 'STN')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_D2_Proto_changing_20_pts_5_runs.pkl')]; key = ('Proto', 'D2')
# filename_list = n_nuclei * [os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_20_pts_10_runs.pkl')]; key = ('Proto', 'STN')
nucleus_name_list = ['FSI', 'Proto','D2', 'STN']

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
y_line_fix = 2 ; legend_loc = 'center' ; clb_loc = 'upper right' ; clb_borderpad = 5
y_line_fix = 2 ; legend_loc = 'upper left' ; clb_loc = 'lower left' ; clb_borderpad = 2

fig = synaptic_weight_transition_multiple_circuit_SNN(filename_list, nucleus_name_list, legend_list, color_list,g_cte_ind,g_ch_ind,param_list,
                                                      color_param_list,'YlOrBr',x_axis = x_axis,param = param,  key = key, y_line_fix = y_line_fix,
                                                      clb_higher_lim = 30, clb_lower_lim= 5, legend_loc = legend_loc, clb_loc = clb_loc, clb_borderpad = clb_borderpad)   
fig.savefig(os.path.join(path,'Beta_power', 'abs_norm_G_' + param + '_beta_' + os.path.basename(filename_list[0]).replace('.pkl','.png')), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

#%% Beta power vs. G low and high beta separate (frequency as inset)
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
nucleus_name_list = ['FSI', 'Proto','D2', 'STN']
filename_list = os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_16_pts_6_runs.pkl')
filename_list = os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_15_pts_5_runs.pkl')

# filename_list = os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_20_pts_5_runs.pkl')
# filename_list = os.path.join(path, 'Beta_power', 'D2_Proto_FSI_STN_N_1000_T_2000_G_all_changing_8_pts_1_runs.pkl')

data = load_pickle(filename_list)
key = [('D2', 'FSI'),
('FSI', 'Proto'),
('Proto', 'D2'),
('STN', 'Proto'),
('Proto', 'STN')]
nuc_loop_lists = [['Proto', 'FSI', 'D2'],['Proto','STN']]
new_tick_locations = np.array([0.2, 0.5, 0.8]) ; second_axis_label = r"$G_{FSI-D2-P}$";  inset_props = [0.6, 0.6, 0.35, 0.35]
key_sec_ax = [('STN','Proto'), ('Proto', 'STN')]
second_axis_label = r"$G_{Proto-STN}$"
y_line_fix = None ; legend_loc = 'upper left' 
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
double_xaxis = False #True
# ylim = [-0.5, 1.5]
ylim = None
plot_inset = True
fig = synaptic_weight_transition_multiple_circuit_SNN_Fr_inset(filename_list, nucleus_name_list, legend_list, color_list,
                                                               param_list, freq_list,'YlOrBr', include_Gs = include_Gs,
                                                               x_axis = x_axis, param = param, key = key, y_line_fix = y_line_fix, ylim = ylim,
                                                               legend_loc = legend_loc, double_xaxis = double_xaxis, 
                                                               new_tick_locations = new_tick_locations, second_axis_label = second_axis_label,
                                                               key_sec_ax = key_sec_ax, loops = 'multi', nuc_loop_lists = nuc_loop_lists,
                                                               plot_inset = plot_inset, markersize= 5,
                                                               inset_props = [0.3, 0.6, 0.35, 0.35])  
 

save_pdf_png(fig, 
             os.path.join(path,'Beta_power', 'abs_norm_G_single_loop_derivative_' + param + '_beta_' + 
                          os.path.basename(filename_list).split('.')[0]),
             size = (7, 5))

#%% Phase summary


# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_20_pts_10_runs.pkl' )
# n_g_list = np.linspace(0, 19, endpoint = True, num = 4).astype(int)

# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_STN_N_1000_T_2000_G_STN_Proto_changing_4_pts_12_runs.pkl' )
# n_g_list = np.arange(4)
# name_list = ['D2', 'STN', 'Proto', 'FSI']

# filename = os.path.join(path, 'Beta_power','STN_Proto_Proto_Arky_N_1000_T_2000_1_pts_20_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_Proto_Arky_N_1000_T_2000_1_pts_20_runs.pkl' )
# name_list = [ 'STN', 'Arky', 'Proto']
# n_g_list = np.array([0])

# filename = os.path.join(path, 'Beta_power','D2_Proto_Arky_N_1000_T_5000_G_all_changing_4_pts_5_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','D2_Proto_Arky_N_1000_T_5000_G_all_changing_1_pts_10_runs.pkl' )
# name_list = ['Proto',  'Arky', 'D2']
# n_g_list = np.array([0])
# y_max_series = {'D2':0.4,  'Arky': 3, 'Proto': 7.5}
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (1.8, three_nuc_raster_y * 1/3)

# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_all_changing_4_pts_5_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_all_changing_1_pts_10_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_all_changing_3_pts_5_runs_dt_0-1_Ornstein-Uhlenbeck.pkl' )

# name_list = ['Proto', 'FSI', 'D2']
# n_g_list = np.array([2])
# y_max_series = {'D2':0.2,'Proto': 4, 'FSI': 2.5}
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (1.8, three_nuc_raster_y )


# filename = os.path.join(path, 'Beta_power','STN_Proto_N_1000_T_2000_1_pts_10_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_Proto_N_1000_T_2000_4_pts_5_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_Proto_N_1000_T_5000_1_pts_10_runs.pkl' )
filename = os.path.join(path, 'Beta_power','STN_Proto_N_1000_T_2000_3_pts_5_runs_dt_0-1_Ornstein-Uhlenbeck.pkl' )

n_g_list = np.array([0])
name_list = ['Proto', 'STN']
y_max_series = {'Proto': 4, 'STN' : 2.5}
three_nuc_raster_y = (60 + 5) * 0.05
figsize = (1.8, three_nuc_raster_y * 2/3)

# filename = os.path.join(path, 'Beta_power','Proto_Proto_N_1000_T_5000_4_pts_5_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','Proto_Proto_N_1000_T_5000_1_pts_10_runs.pkl' )
# n_g_list = np.array([0])
# name_list = ['Proto']
# y_max_series = {'Proto': 8}
# three_nuc_raster_y = (60 + 5) * 0.05
# figsize = (1.8, three_nuc_raster_y * 1/3)

# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_FSI_D2_changing_16_pts_8_runs.pkl' )
# n_g_list = np.linspace(0, 15, endpoint = True, num = 4).astype(int)
# name_list = ['D2', 'Proto', 'FSI']

# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_2000_G_all_changing_1_pts_8_runs_20Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_2000_G_all_changing_1_pts_8_runs_30Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_2000_G_all_changing_1_pts_8_runs_15Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_2000_G_all_changing_1_pts_10_runs_19Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_5000_G_all_changing_1_pts_8_runs_18-8Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_5000_G_all_changing_1_pts_8_runs_33Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_5000_G_all_changing_1_pts_8_runs_33Hz.pkl' )

# y_max_series = {'D2': 0.5, 'STN': 4, 'Arky': 3, 'Proto': 8, 'FSI': 4}
# filename = os.path.join(path, 'Beta_power','All_nuc_DD_N_1000_T_2000_4_pts_5_runs.pkl' )
# y_max_series = {'D2':3, 'STN': 8, 'Arky': 5, 'Proto': 8, 'FSI': 8}
# name_list = ['D2', 'STN', 'Arky', 'Proto', 'FSI']
# n_g_list = np.array([0])



ref_nuc_name = 'Proto'
# shift_phase = 'backward'
# shift_phase = 'forward'
shift_phase = None
fig = phase_summary(filename, name_list, color_dict, n_g_list, ref_nuc_name= ref_nuc_name, 
                    shift_phase = shift_phase, set_ylim = True, y_max_series=y_max_series)

fig = remove_all_x_labels(fig)
# fig = set_y_ticks(fig, [0, n_neuron])

figname = filename.split('.')[0] + '_Phase'
save_pdf_png(fig, figname, size = (1.8, len(name_list) * 1))
# save_pdf_png(fig, figname, size = figsize)


#%% PSD summary

# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_2000_G_all_changing_1_pts_8_runs_30Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_2000_G_all_changing_1_pts_8_runs_20Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_2000_G_all_changing_1_pts_8_runs_15Hz.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_5000_G_all_changing_1_pts_8_runs_18-8Hz-2.pkl' )
# inset_props = [0.6, 0.3, 0.35, 0.35]
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_5000_G_all_changing_1_pts_8_runs_33Hz.pkl' )
# inset_props = [0.1, 0.22, 0.35, 0.35]
# filename = os.path.join(path, 'Beta_power','STN_D2_Proto_FSI_Arky_N_1000_T_5000_G_all_changing_1_pts_8_runs_14-3Hz-2.pkl' )
# inset_props = [0.6, 0.3, 0.35, 0.35]
# name_list = ['D2', 'STN', 'Arky', 'Proto', 'FSI']

# filename = os.path.join(path, 'Beta_power','D2_Proto_Arky_N_1000_T_5000_G_all_changing_4_pts_5_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','D2_Proto_Arky_N_1000_T_5000_G_all_changing_1_pts_10_runs.pkl' )
# inset_props = [0.6, 0.3, 0.35, 0.35]
# name_list = ['D2', 'Arky', 'Proto']
# n_g_list = np.array([0])

# filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_all_changing_4_pts_5_runs.pkl' )
# # filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_all_changing_1_pts_10_runs.pkl' )
filename = os.path.join(path, 'Beta_power','D2_Proto_FSI_N_1000_T_5000_G_all_changing_3_pts_5_runs_dt_0-1_Ornstein-Uhlenbeck.pkl' )
inset_props = [0.6, 0.3, 0.35, 0.35]
name_list = ['D2', 'FSI', 'Proto']
# n_g_list = np.array([0])

# filename = os.path.join(path, 'Beta_power','STN_Proto_Proto_Arky_N_1000_T_2000_1_pts_20_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_Proto_Arky_N_1000_T_2000_1_pts_20_runs.pkl' )
# name_list = [ 'STN', 'Arky', 'Proto']


# filename = os.path.join(path, 'Beta_power','STN_Proto_N_1000_T_2000_1_pts_10_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_Proto_N_1000_T_5000_4_pts_5_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_Proto_N_1000_T_5000_1_pts_10_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','STN_Proto_N_1000_T_2000_3_pts_5_runs_dt_0-1_Ornstein-Uhlenbeck.pkl' )

# n_g_list = np.array([0])
# name_list = [ 'STN', 'Proto']
# inset_props = [0.6, 0.3, 0.35, 0.35]

n_g_list = np.arange(3)


# filename = os.path.join(path, 'Beta_power','Proto_Proto_N_1000_T_5000_4_pts_5_runs.pkl' )
# filename = os.path.join(path, 'Beta_power','Proto_Proto_N_1000_T_5000_1_pts_10_runs.pkl' )
# n_g_list = np.array([0])
# name_list = ['Proto']
# inset_props = [0.6, 0.3, 0.35, 0.35]

# n_g_list = np.linspace(0, 19, endpoint = True, num = 4).astype(int)
# n_g_list = np.array([0])
# data = load_pickle(filename)

fig = PSD_summary(filename, name_list, color_dict, n_g_list, xlim = (0,70), inset_props = inset_props, 
                   # err_plot = 'errorbar', inset_name=None)#, inset_yaxis_loc = 'left')
                   err_plot = 'fill_between', inset_name=None, plot_lines = True, legend_font_size = 10, legend_loc = 'upper right')

save_pdf_png(fig, filename.split('.')[0] + '_PSD',
             size = (7,5))



#%% Boxplot frequency vs loop

n = 10
filename_list = ['Proto_Proto_N_1000_T_5000_1_pts_10_runs.pkl',
                 'STN_Proto_N_1000_T_5000_1_pts_10_runs.pkl',
                 'D2_Proto_FSI_N_1000_T_5000_G_all_changing_1_pts_10_runs.pkl',
                 'D2_Proto_Arky_N_1000_T_5000_G_all_changing_1_pts_10_runs.pkl']

filename_list = [os.path.join(path, 'Beta_power', file) for file in filename_list]

xs = [] ; vals = []
freq = np.zeros((n, len(filename_list)))
for i, filename in enumerate(filename_list):
    data = load_pickle(filename)
    freq[:, i] = data[('Proto', 'base_freq')]
    xs.append(np.random.normal(i+1, 0.04, n))
    vals.append(data[('Proto', 'base_freq')])
color_list = [color_dict['Proto'], color_dict['STN'], color_dict['FSI'], color_dict['Arky']]

x_labels = ['Proto-Proto',
            'STN-Proto',
            'FSI Loop',
            'Arky Loop']

fig, ax = plt.subplots()
bp = ax.boxplot(freq, labels = x_labels, patch_artist=True, whis = (0,100), zorder = 0 )
for patch, color in zip(bp['boxes'], color_list):
    patch.set_facecolor(color)
    
for median in (bp['medians']): 
    median.set(color = 'k', 
               linewidth = 0.5) 
for x, val, c in zip(xs, vals, color_list):
    plt.scatter(x, val, c=c, alpha=0.4, s = 10, ec = 'k', zorder = 1)
ax.tick_params(axis='x', labelsize= 10)
ax.tick_params(axis='y', labelsize= 12)

# set_y_ticks(fig, [10,30,60])
plt.xticks(rotation=40)
remove_frame(ax)
ax.set_ylim(0,60)
# set_boxplot_prop(bp, color_list)
save_pdf_png(fig, os.path.join(path, 'mean_F_all_loops'),
             size = (2,4))
#%% Phase summary only entrained

ref_nuc_name = 'D2'
entr_nuc_name = 'D2'
low_f, high_f = 8, 70
filter_based_on_AUC_of_PSD = False
only_entrained_neurons = False
c_dict = color_dict.copy()
find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
                                              height = 0, ref_nuc_name = ref_nuc_name, start = 0, total_phase = 720,
                                              only_entrained_neurons =only_entrained_neurons)
c_dict[entr_nuc_name] = color_dict[entr_nuc_name]
fig = phase_plot_all_nuclei_in_grid(nuclei_dict, c_dict, dt, 
                          density = False, ref_nuc_name = ref_nuc_name, total_phase = 720, projection = None,
                          outer=None, fig=None,  title='', tick_label_fontsize=18,plot_mode = 'hist',
                           labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)
c_dict[entr_nuc_name] = 'g'
only_entrained_neurons = True
find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
                                  height = 0, ref_nuc_name = ref_nuc_name, start = 0, total_phase = 720,
                                  only_entrained_neurons =only_entrained_neurons, min_f_sig_thres = 0,window_mov_avg = 10, max_f = 250,
                                  n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2,
                                  min_f_AUC_thres = 7,  PSD_AUC_thresh = 10**-6, filter_based_on_AUC_of_PSD = filter_based_on_AUC_of_PSD)

fig = phase_plot_all_nuclei_in_grid(nuclei_dict, c_dict, dt, 
                          density = False, ref_nuc_name = ref_nuc_name, total_phase = 720, projection = None,
                          outer=None, fig=fig,  title='', tick_label_fontsize=18, plot_mode = 'hist',
                           labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)
# c_dict[entr_nuc_name] = 'g'
# only_entrained_neurons = True
# find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 100,
#                                   height = 0, ref_nuc_name = ref_nuc_name, start = 0, total_phase = 720,
#                                   only_entrained_neurons =only_entrained_neurons, min_f_sig_thres = 0,window_mov_avg = 10, max_f = 250,
#                                   n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2,
#                                   min_f_AUC_thres = 7,  PSD_AUC_thresh = 10**-4.5, filter_based_on_AUC_of_PSD = filter_based_on_AUC_of_PSD)

# fig = phase_plot_all_nuclei_in_grid(nuclei_dict, c_dict, dt, 
#                           density = False, ref_nuc_name = ref_nuc_name, total_phase = 720, projection = None,
#                           outer=None, fig=fig,  title='', tick_label_fontsize=18,
#                            labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False)

#%% FR simulation vs FR_expected ( heterogeneous vs. homogeneous initialization)

N_sim = 20
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

G = {}
name = 'D2'
g = -0.01; g_ext = -g
poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name,'1') : []}

pop_list = [1]  

tuning_param = 'firing'; n =10
# p = np.arange(1,5,dtype=float)
# FR_list = np.ones(len(p))*np.power(10,-p)
start = 1 ; end = 200; FR_list=np.linspace(start,end,n)


init_method = 'homogeneous'
nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop, init_method= init_method) for i in pop_list]
nuclei_dict = {name: nuc}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')

firing_prop = find_FR_sim_vs_FR_expected(FR_list,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt)

init_method = 'heterogeneous'
nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop, init_method= init_method) for i in pop_list]
nuclei_dict = {name: nuc}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')

firing_prop_hetero = find_FR_sim_vs_FR_expected(FR_list,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt)

plt.figure()

plt.plot(FR_list,firing_prop[name]['firing_mean'][:,0],'-o',label = 'simulation_homogeneous', c = 'darkred')
plt.fill_between(FR_list,firing_prop[name]['firing_mean'][:,0]-firing_prop[name]['firing_var'][:,0],
                  firing_prop[name]['firing_mean'][:,0]+firing_prop[name]['firing_var'][:,0],alpha = 0.2, color = 'darkred')

plt.plot(FR_list,firing_prop_hetero[name]['firing_mean'][:,0],'-o',label = 'simulation_heterogeneous', c = 'teal')
plt.fill_between(FR_list,firing_prop_hetero[name]['firing_mean'][:,0]-firing_prop_hetero[name]['firing_var'][:,0],
                  firing_prop_hetero[name]['firing_mean'][:,0]+firing_prop_hetero[name]['firing_var'][:,0],alpha = 0.2, color = 'teal')

plt.plot(FR_list,FR_list,'--', label = 'y=x', c = 'k')
plt.xlabel(r'$FR_{expected}$',fontsize = 10)
plt.ylabel(r'$FR_{simulation}$',fontsize = 10)
plt.legend()


#%% FR vs FR_ext theory vs. simulation

N_sim = 100
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

G = {}
name = 'Proto'

g = -0.01; g_ext = -g
poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}
receiving_pop_list = {(name,'1') : []}

pop_list = [1]  
init_method = 'homogeneous'
# ext_inp_method = 'Poisson'
ext_inp_method = 'const+noise'

noise_variance = {name : 3}
noise_amplitude = {name : 1}

label =ext_inp_method+ r' $\sigma = ' + str(noise_variance[name]) + 'mV$'
# label = ext_inp_method


nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',
               poisson_prop =poisson_prop, init_method=init_method, der_ext_I_from_curve= False, ext_inp_method = ext_inp_method) for i in pop_list]

nuclei_dict = {name: nuc}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')

start=0.175; end=0.185; n = 20
FR_list = spacing_with_high_resolution_in_the_middle(n, start, end).reshape(-1,)


firing_prop = find_FR_sim_vs_FR_ext(FR_list,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt)
plt.figure()
plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, I_ext_range, neuronal_consts)
plt.plot(FR_list * 1000,firing_prop[name]['firing_mean'][:,0], '-o' , c = 'teal', label = label)# + ' N=' + str(poisson_prop[name]['n']) , )

plt.fill_between(FR_list * 1000,firing_prop[name]['firing_mean'][:,0]-firing_prop[name]['firing_var'][:,0] ,
                  firing_prop[name]['firing_mean'][:,0] + firing_prop[name]['firing_var'][:,0] , alpha = 0.2 , color = 'teal')
plt.title(name + ' ' + init_method, fontsize  = 18)
plt.legend()
if ext_inp_method == 'Poisson':
    filename = ('FR_sim_vs_FR_ext_' +  name + '_' + init_method + '_' + ext_inp_method + '_N=' + str(N_sim) + '_N_ext=' +  str(poisson_prop[name]['n']) + '.png')
else:
    filename = ('FR_sim_vs_FR_ext_' +  name + '_' + init_method + '_' + ext_inp_method + '_noise=' + 
            str(noise_variance[name]) + '_N=' + str(N_sim) + '.png')
plt.savefig(os.path.join(path, filename), dpi = 300, facecolor='w', edgecolor='w',
        orientation='portrait', 
        transparent=True,bbox_inches = "tight", pad_inches=0.1)



#%% two nuclei with given I_ext + noise

N_sim = 1000
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

name1 = 'D2'
name2 = 'Proto'
G = {}
g = -0.01; g_ext = -g
G[('Proto', 'D2')] = g

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') : [],
                      (name2, '1'): [(name1,'1')]}

pop_list = [1]  
init_method = 'heterogeneous'
noise_variance = {name1 : D2_noise, name2: D2_noise}
noise_amplitude = {name1 : 1, name2: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')

noise_std_min = 10**-17
noise_std_max = 10**-1 # interval of search for noise variance

nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict,neuronal_model='spiking')
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,ax = None,title_fontsize=15,plot_start = 100,
        title = '')#r"$G_{SP}="+str(round(G[('Proto', 'STN')],2))+"$ "+", $G_{PS}=G_{PP}="+str(round(G[('STN', 'Proto')],2))+'$')

fig, axs = plt.subplots(len(nuclei_dict), 1, sharex=True, sharey=True)
count = 0
for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        count +=1
        nucleus.smooth_pop_activity(dt, window_ms = 5)
        FR_mean, FR_std = nucleus. average_pop_activity( t_list, last_fraction = 1/2)
        print(nucleus.name, 'average ={}, std = {}'.format(FR_mean, FR_std  ) )
        spikes_sparse = create_sparse_matrix (nucleus.spikes) * dt
        raster_plot(axs[count - 1], spikes_sparse, nucleus.name, color_dict, labelsize=10, title_fontsize = 15)
        find_freq_of_pop_act_spec_window_spiking(nucleus, 0,t_list[-1], dt, cut_plateau_epsilon =0.1, peak_threshold = 0.1, smooth_kern_window= 3 , check_stability = False)

fig.text(0.5, 0.02, 'time (ms)', ha='center', va='center',fontsize= 15)
fig.text(0.02, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 15)
#%% Proto and FSI binary search
def find_D2_I_ext_in_D2_FSI(x):
    print(x)
    noise_variance = {name1 : x, name2: 10**-10}
    noise_amplitude = {name1 : 1, name2: 1}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(neuronal_model = 'spiking')
            nucleus.set_noise_param(noise_variance, noise_amplitude)
            # nucleus.reset_ext_pop_properties(poisson_prop,dt)
    print('I_ext', np.average(nuc1[0].rest_ext_input))
    nuclei_dict_p = run(receiving_class_dict,t_list, dt, nuclei_dict,neuronal_model='spiking')
    for nuclei_list in nuclei_dict_p.values():
        for nucleus in nuclei_list:
            nucleus.smooth_pop_activity(dt, window_ms = 5)            
            print(nucleus.name,np.average(nucleus.pop_act[int(len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]),2))
    return np.average(nuc1[0].pop_act[int(len(t_list)/2):])- A[name1]
N_sim = 1000
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

name1 = 'FSI'
name2 = 'Proto'
G = {}
g = -0.01; g_ext = -g
G[('FSI', 'Proto')] = g


poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') : [(name2, '1')],
                      (name2, '1'): []}

pop_list = [1]  
init_method = 'heterogeneous'
noise_variance = {name1 : 0, name2: 0}
noise_amplitude = {name1 : 1, name2: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')
ratio_list = np.linspace(0.9, 1.5, 1)
noise_std_min = 10**-17
noise_std_max = 10**-1 # interval of search for noise variance
for ratio in ratio_list:
    try: 
        noise = optimize.bisect(find_D2_I_ext_in_D2_FSI, noise_std_min, noise_std_max, xtol = 10**-10)
        print('noise = ',noise)
        break
    except ValueError as err:
        print(err)
# find_D2_I_ext( 0.025003981672404058)

noise_std_min = 10**-17
noise_std_max = 10**-10 # interval of search for noise variance
from scipy import optimize
# D2_noise = optimize.bisect(find_D2_I_ext, noise_std_min, noise_std_max, xtol = 10**-20)
# print('D2 noise = ',D2_noise)
#%% D2 and FSI binary search
def find_D2_I_ext_in_D2_FSI(x):
    print(x)
    noise_variance = {name1 : x, name2: 10**-10}
    noise_amplitude = {name1 : 1, name2: 1}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(neuronal_model = 'spiking')
            nucleus.set_noise_param(noise_variance, noise_amplitude)
            # nucleus.reset_ext_pop_properties(poisson_prop,dt)
    print('I_ext', np.average(nuc1[0].rest_ext_input))
    nuclei_dict_p = run(receiving_class_dict,t_list, dt, nuclei_dict,neuronal_model='spiking')
    for nuclei_list in nuclei_dict_p.values():
        for nucleus in nuclei_list:
            nucleus.smooth_pop_activity(dt, window_ms = 5)
            print(nucleus.name,np.average(nucleus.pop_act[int(len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]),2))
    return np.average(nuc1[0].pop_act[int(len(t_list)/2):])- A[name1]
N_sim = 1000
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

name1 = 'D2'
name2 = 'FSI'
G = {}
g = -0.01; g_ext = -g
G[('D2', 'FSI')] = g

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') : [],#[(name2, '1')],
                      (name2, '1'): []}

pop_list = [1]  
init_method = 'heterogeneous'
noise_variance = {name1 : 0, name2: 0}
noise_amplitude = {name1 : 1, name2: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')
ratio_list = np.linspace(1, 1.5, 1)
noise_std_min = 10**-17
noise_std_max = 10**-1 # interval of search for noise variance
for ratio in ratio_list:
    try: 
        D2_noise = optimize.bisect(find_D2_I_ext_in_D2_FSI, noise_std_min, noise_std_max, xtol = 10**-10)
        print('D2 noise = ',D2_noise)
        break
    except ValueError as err:
        print(err)
# find_D2_I_ext( 3.0766320304347586e-13)

# D2_noise = optimize.bisect(find_D2_I_ext, noise_std_min, noise_std_max, xtol = 10**-20)
# print('D2 noise = ',D2_noise)
#%% D2- FSI - Proto binary search
def find_D2_I_ext_in_D2_FSI_Proto(x):
    print('x = ', x)
    noise_variance = {name1 : x, name2: 10**-10, name3 : 0.025003981672404058}
    noise_amplitude = {name1 : 1, name2: 1, name3: 1}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(neuronal_model = 'spiking')
            nucleus.set_noise_param(noise_variance, noise_amplitude)
            # nucleus.reset_ext_pop_properties(poisson_prop,dt)
    nuclei_dict_p = run(receiving_class_dict,t_list, dt, nuclei_dict,neuronal_model='spiking')
    for nuclei_list in nuclei_dict_p.values():
        for nucleus in nuclei_list:
            nucleus.smooth_pop_activity(dt, window_ms = 5)
            print(nucleus.name,np.average(nucleus.pop_act[int(len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]),2))
    return np.average(nuc1[0].pop_act[int(len(t_list)/2):])- A[name1]
N_sim = 100
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

name1 = 'D2'
name2 = 'Proto'
name3 = 'FSI'
G = {}
g = -0.1; g_ext = -g
G[('D2', 'FSI')], G[('FSI', 'Proto')], G[('Proto', 'D2')] = g, g, g*0.5

poisson_prop = {name1:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name2:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext},
                name3:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name1,'1') : [(name3, '1')],
                      (name2, '1'): [(name1, '1')],
                      (name3, '1'): [(name2, '1')]}

pop_list = [1]  
init_method = 'heterogeneous'
noise_variance = {name1 : 0, name2: 0, name3: 0}
noise_amplitude = {name1 : 1, name2: 1, name3: 1}
nuc1 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name1, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]
nuc2 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name2, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]
nuc3 = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name3, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop,init_method = init_method) for i in pop_list]

nuclei_dict = {name1: nuc1, name2: nuc2, name3: nuc3}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')
I_ext_list = np.linspace(0.055, 0.06, 10)
noise_std_min = 10**-9
noise_std_max = 10**-2 # interval of search for noise variance
for I_ext in I_ext_list:
    try: 
        # nuc3[0].rest_ext_input = nuc3[0].rest_ext_input * 0.9
        D2_noise = optimize.bisect(find_D2_I_ext_in_D2_FSI_Proto, noise_std_min, noise_std_max, xtol = 10**-10)
        print('D2 noise = ',D2_noise)
        break
    except ValueError as err:
        print(err)


#%% Check single population firing with external poisson spikes

N_sim = 1000
N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
dt = 0.25
t_sim = 1000; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt
G = {}
receiving_pop_list = {(name,'1') : []}
g = -0.01; g_ext = -g
poisson_prop = {name:{'n':10000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

pop_list = [1]  
# init_method = 'heterogeneous'
init_method = 'homogeneous'
noise_variance = {name : 0.1}
noise_amplitude = {name : 1}
nuc = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
            synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop, init_method = init_method) for i in pop_list]
nuclei_dict = {name: nuc}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')
nuc[0].rest_ext_input = I_ext/ 1000
nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict,neuronal_model='spiking')
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title_fontsize=15, plot_start = 100, title = '')

fig, axs = plt.subplots(len(nuclei_dict), 1, sharex=True, sharey=True)
count = 0
for nuclei_list in nuclei_dict.values():
    for nucleus in nuclei_list:
        count +=1
        nucleus.smooth_pop_activity(dt, window_ms = 5)
        print(nucleus.name,np.average(nucleus.pop_act[int(len(t_list)/2):]), round(np.std(nucleus.pop_act[int(len(t_list)/2):]),2))
        spikes_sparse = [np.where(nucleus.spikes[i,:]==1)[0]*dt for i in range(nucleus.n)]

        axs.eventplot(spikes_sparse, colors='k',linelengths=2,lw = 2,orientation='horizontal')
        axs.tick_params(axis='both', labelsize=10)
        axs.set_title(nucleus.name, c = color_dict[nucleus.name],fontsize = 15)
        find_freq_of_pop_act_spec_window_spiking(nucleus, 0,t_list[-1], dt, cut_plateau_epsilon =0.1, peak_threshold = 0.1, smooth_kern_window= 3 , check_stability = False)

fig.text(0.5, 0.02, 'time (ms)', ha='center', va='center',fontsize= 15)
fig.text(0.02, 0.5, 'neuron', ha='center', va='center', rotation='vertical',fontsize = 15)
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

#%% Measuring AUC of the input of one spike (what constant external input reproduces the same firing with poisson spike inputs)

N_sim = 10
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 100; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

g = -1
G = {}
g_ext = 1
name = 'Proto'
poisson_prop = {name:{'n':1000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name,'1') : []}
pop_list = [1]  
find_AUC_of_input(name,path,poisson_prop,gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude,
                      N, A, A_mvt,D_mvt,t_mvt, N_real, K_real,t_list,color_dict, G, T, t_sim, dt, synaptic_time_constant, 
                      receiving_pop_list, smooth_kern_window,oscil_peak_threshold,syn_component_weight,end_of_nonlinearity, if_plot = True)
#%% Compare FR of Possion ext inp with Const+noise
np.random.seed(19090)
# N_sim = 1000
N_sim = 10
N = dict.fromkeys(N, N_sim)
K = calculate_number_of_connections(N, N_real, K_real)
dt = 0.1
t_sim = 100; t_list = np.arange(int(t_sim/dt))
t_mvt = t_sim ; D_mvt = t_sim - t_mvt

g = -1
G = {}
g_ext = 1
name = 'Proto'
poisson_prop = {name:{'n':1000, 'firing':0.0475,'tau':{'rise':{'mean':1,'var':.1},'decay':{'mean':5,'var':0.5}}, 'g':g_ext}}

receiving_pop_list = {(name,'1') : []}
pop_list = [1]  
class Nuc_poisson(Nucleus):
        def cal_ext_inp(self,dt,t):
            poisson_spikes = possion_spike_generator(self.n,self.n_ext_population,self.firing_of_ext_pop,dt)
            self.syn_inputs['ext_pop','1'] =  (np.sum(poisson_spikes,axis = 1)*self.membrane_time_constant*self.syn_weight_ext_pop).reshape(-1,)
            # self.I_syn['ext_pop','1'] += np.true_divide((-self.I_syn['ext_pop','1'] + self.syn_inputs['ext_pop','1']),self.tau_ext_pop['decay']) # without rise
            self.I_rise['ext_pop','1'] += ((-self.I_rise['ext_pop','1'] + self.syn_inputs['ext_pop','1'])/self.tau_ext_pop['rise'])
            self.I_syn['ext_pop','1'] += np.true_divide((-self.I_syn['ext_pop','1'] + self.I_rise['ext_pop','1']),self.tau_ext_pop['decay'])
        def solve_IF(self,t,dt,receiving_from_class_list,mvt_ext_inp):
        
            self.cal_ext_inp(dt,t)
            inputs =  self.I_syn['ext_pop','1']*10
            self.mem_potential = self.mem_potential + np.true_divide((inputs - self.mem_potential+self.u_rest)*dt,self.membrane_time_constant)        
            spiking_ind = np.where(self.mem_potential > self.spike_thresh) # gaussian distributed spike thresholds
            self.spikes[spiking_ind, t] = 1
            self.mem_potential[spiking_ind] = self.neuronal_consts['u_rest']
            self.pop_act[t] = np.average(self.spikes[:, t],axis = 0)/(dt/1000)

nuc = [Nuc_poisson(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop) for i in pop_list]
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')

print(Proto[0].mem_potential[5])
nuclei_dict = {name : nuc}
nuclei_names = list(nuclei_dict.keys()) 

tuning_param = 'firing'; start=0.05; end=0.3; n =12

list_1=np.linspace(start,end,n)

nucleus_name = list(nuclei_dict.keys()); m = len(list_1)
firing_prop = {k: {'firing_mean': np.zeros((m,len(nuclei_dict[nucleus_name[0]]))),'firing_var':np.zeros((m,len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
ext_firing = np.zeros((m,len(nucleus_name)))
i =0
for g in list_1:
    for j in range(len(nucleus_name)):
        poisson_prop[nucleus_name[j]][tuning_param] = g 
        ext_firing[i,j] = g
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history()
            nucleus.reset_ext_pop_properties(poisson_prop,dt)
    nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict,neuronal_model = 'spiking')
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            # firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = np.average(nucleus.pop_act[int(len(t_list)/2):]) # population activity
            firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = np.average(nucleus.spikes[5,int(len(t_list)/2):] )/(dt/1000) # single neuron activity
            firing_prop[nucleus.name]['firing_var'][i,nucleus.population_num-1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
            print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop,3),
                'FR=',firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] ,'std=',round(firing_prop[nucleus.name]['firing_var'][i,nucleus.population_num-1],2))
    i+=1

''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''

class Nuc_I_ext_cte(Nucleus):
        def solve_EIF(self,t,dt,receiving_from_class_list,mvt_ext_inp):
            inputs =  self.rest_ext_input*self.membrane_time_constant*self.n_ext_population
            self.mem_potential = self.mem_potential + np.true_divide((inputs - self.mem_potential+self.u_rest)*dt,self.membrane_time_constant)        
            spiking_ind = np.where(self.mem_potential > self.spike_thresh) # gaussian distributed spike thresholds
            self.spikes[spiking_ind, t] = 1
            self.mem_potential[spiking_ind] = self.neuronal_consts['u_rest']
            self.pop_act[t] = np.average(self.spikes[:, t],axis = 0)/(dt/1000)
np.random.seed(19090)
Proto = [Nuc_I_ext_cte(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop) for i in pop_list]
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list,neuronal_model='spiking')
print(Proto[0].mem_potential[5])
nuclei_dict = {'Proto': Proto}
nuclei_names = list(nuclei_dict.keys()) 

tuning_param = 'firing'; start=.05; end=0.30; n =20
list_cte=np.linspace(start,end,n)

nucleus_name = list(nuclei_dict.keys()); m = len(list_cte)
firing_prop_cte = {k: {'firing_mean': np.zeros((m,len(nuclei_dict[nucleus_name[0]]))),'firing_var':np.zeros((m,len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
ext_firing = np.zeros((m,len(nucleus_name)))
i =0
for g in list_cte:
    for j in range(len(nucleus_name)):
        poisson_prop[nucleus_name[j]][tuning_param] = g 
        ext_firing[i,j] = g
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(neuronal_model = 'spiking')
            nucleus.reset_ext_pop_properties(poisson_prop,dt)
            nucleus.rest_ext_input = g*np.ones(nucleus.n)
    nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict,neuronal_model = 'spiking')
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            # firing_prop_cte[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
            firing_prop_cte[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = np.average(nucleus.spikes[5,int(len(t_list)/2):])/(dt/1000)
            firing_prop_cte[nucleus.name]['firing_var'][i,nucleus.population_num-1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
            print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop,3),
                'FR=',firing_prop_cte[nucleus.name]['firing_mean'][i,nucleus.population_num-1] ,'std=',round(firing_prop_cte[nucleus.name]['firing_var'][i,nucleus.population_num-1],2))
    i+=1
    
plt.figure()
plt.plot(list_1, firing_prop['Proto']['firing_mean'],'-o',label = r'$G\times FR (decay=5)$',)    
plt.plot(list_cte, firing_prop_cte['Proto']['firing_mean'],'-o',label = r'$I_{ext}*10$')
plt.legend()
plt.ylabel('FR(Hz)')
plt.xlabel(r'$G\times FR (Spk/ms)$')

#%% RATE MODEL : STN-Proto network
plt.close('all')
N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = 700 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
plot_start_trans = t_mvt - 200 ; plot_duration = 600
plot_start_stable = 0

name1 = 'Proto'
name2 = 'STN'
name_list = [name1, name2]

(synaptic_time_constant[(name2, name1)],
 synaptic_time_constant[(name1, name2)] )  =  [10], [6]
g = - 2
G = { ('STN', 'Proto'): g,
     ('Proto', 'STN'): -g} # synaptic weight
# G[('Proto', 'Proto')] = G[('STN', 'Proto')]
receiving_pop_list = {('STN','1') : [('Proto', '1')],
                    ('Proto','1') : [('STN', '1')]}
                    

lim_n_cycle = [6,10]
pop_list = [1]  

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, plot_start=plot_start,title_fontsize=15,ylim=ylim,include_FR=False,
     title = r"$G_{SP}="+str(round(G[('Proto', 'STN')],2))+"$ "+", $G_{PS}=G_{PP}="+str(round(G[('STN', 'Proto')],2))+'$')
#g_list = np.linspace(-.6,-0.1, 20)
# n_half_cycle, G, nuclei_dict = find_oscillation_boundary_STN_GPe(g_list,nuclei_dict, A, A_mvt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base)
figname = 'STN-GPe loop with Proto-Proto_no_ocsill'
# fig.savefig(os.path.join(path_rate, figname+'.png'),dpi = 300)
# fig.savefig(os.path.joijn(path_rate, figname+'.pdf'),dpi = 300)
#print(find_freq_of_pop_act_spec_window(STN[0],*duration_mvt))
temp_oscil_check(nuclei_dict['STN'][0].pop_act,oscil_peak_threshold['STN'], 3,dt,*duration_mvt)
# temp_oscil_check(nuclei_dict['STN'][0].pop_act,oscil_peak_threshold['STN'], 3,dt,*duration_base)
# temp_oscil_check(nuclei_dict['Proto'][0].pop_act,oscil_peak_threshold['Proto'], 3,dt,*duration_mvt)
#plt.title(r"$\tau_{GABA_A}$ = "+ str(round(x[n_plot],2))+r' $\tau_{GABA_B}$ ='+str(round(y[n_plot],2))+ r' $\tau_{Glut}$ ='+str(round(z[n_plot],2))+' f ='+str(round(c[n_plot],2)) , fontsize = 10)
#%% RATE MODEL : GPe-GPe G-sweep

N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 20000; t_list = np.arange(int(t_sim/dt))
t_mvt = 700 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
plot_start_trans = t_mvt - 200 ; plot_duration = 600
plot_start_stable = 0

name1 = 'Proto'
name_list = [name1]

n = 50
G_list = np.linspace(-7, 0, n, endpoint = True)

transition_range = [4.25 - (4.27 - 4.245), 4.27] # dt = 0.5
transition_range = [3.8778 - (3.913 - 3.8778), 3.913] # dt = 0.1
G_list = -pad_high_res_spacing_with_linspace(0, transition_range[0], 20, transition_range[1], 4.5,  10, 10)
G_list = -pad_high_res_spacing_with_arange(1, transition_range[0], 1/10, transition_range[1], 4.5,  1/10, 20)


synaptic_time_constant[('Proto', 'Proto')] = [10]

G = {
     (name1 , name1) : -1
     }

G_ratio_dict = {
     (name1 , name1) : 1
     }
receiving_pop_list = {('Proto','1') : [('Proto', '1')]}


lim_n_cycle = [6,10]
pop_list = [1]  

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# run(receiving_class_dict,t_list, dt, nuclei_dict)
# plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, 
#                                       include_FR = False, plot_start = 0, legend_loc = 'upper left',
#                                       title_fontsize = 15, ax = None)

filename = ( 'G_sweep_Proto-Proto' + '_tau_' + 
            str(synaptic_time_constant[(name1 , name1)][0]) + 
            '_n_' + str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.', '-'))
filename = os.path.join(path_rate, 
                        filename.replace('.', '-') +  '.pkl' 
                        )
fig_trans , fig_stable = synaptic_weight_space_exploration(G.copy(), A, A_mvt, D_mvt, t_mvt, t_list, dt,  filename, 
                                                           lim_n_cycle, G_list, nuclei_dict, duration_mvt, duration_base, 
                                                           receiving_class_dict,color_dict, G_ratio_dict = G_ratio_dict,
                                                           if_plot = if_plot, legend_loc= 'upper right', plot_start_trans = plot_start_trans,
                                                           plot_start_stable = plot_start_stable, plot_duration = plot_duration)

# fig_trans , fig_stable = set_ylim_trans_stable_figs(fig_trans, fig_stable, ymax = [100, 100], ymin = [-4. -4])
save_trans_stable_figs(fig_trans, fig_stable, path_rate, filename.split('.')[0], figsize = (5,3), 
                       ymax =[ [75, 100], [75, 100], [75, 75]], ymin = [-4, -4])
pkl_file = open(filename , 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto'
color = 'perc_t_oscil_mvt' #mvt_f'
param = 'perc_t_oscil_mvt'

xlabel = r'$G_{Loop}$'


g = Product_G(data)
fig = scatter_2d_plot(abs(g),
                np.squeeze(data[(name,param)]),
                np.squeeze(data[(name,color)]), 
                name + ' in GPe-GPe circuit' , 
                [xlabel, param, color] )

plt.axvline(data['g_loop_stable'], c = 'k')
plt.axvline(data['g_loop_transient'], c = 'grey', linestyle = '--')
save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size = (8,6))
#%% RATE MODEL : STN-GPe without GPe-GPe G-sweep
N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 20000; t_list = np.arange(int(t_sim/dt))
t_mvt = 700 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
plot_start_trans = t_mvt - 200 ; plot_duration = 600
plot_start_stable = 0

name1 = 'Proto'
name2 = 'STN'
name_list = [name1, name2]

n = 50
# transition_range = [4.55-(4.6-4.55), 4.6] #dt = 0.5
transition_range = [3.55-(3.57 - 3.55), 3.57] # dt = 0.1
G_list = pad_high_res_spacing_with_arange(1, transition_range[0], 1/10, transition_range[1], 5,  1/10, 20)

G_list = -np.power(G_list, 1/2)
(synaptic_time_constant[(name2, name1)],
 synaptic_time_constant[(name1, name2)] )  =  [10], [6]

G = {
     (name2 , name1) : -1,
     (name1 , name2) : 1
     }

receiving_pop_list = {('STN', '1') : [('Proto', '1')],
                      ('Proto', '1') : [('STN', '1')] }

G_ratio_dict = {
                (name2 , name1) : 1,
                (name1 , name2) : -1
                }


lim_n_cycle = [6,10]
pop_list = [1]  

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

# run(receiving_class_dict,t_list, dt, nuclei_dict)
# plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, 
#                                       include_FR = False, plot_start = 0, legend_loc = 'upper left',
#                                       title_fontsize = 15, ax = None)

filename = ( 'G_sweep_STN-GPe' + '_tau_' + name1[0] + name2[0] + '_' +
           str( synaptic_time_constant[(name2, name1)][0]) + '_' + name2[0] + name1[0] + '_' +
            str( synaptic_time_constant[(name1, name2)][0])  +
            '_n_' + str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.','-'))

filename = os.path.join(path_rate, 
                        filename.replace('.', '-') +  '.pkl' 
                        )
fig_trans , fig_stable = synaptic_weight_space_exploration(G.copy(), A, A_mvt, D_mvt, t_mvt, t_list, dt, filename, 
                                                            lim_n_cycle, G_list, nuclei_dict, duration_mvt, duration_base, 
                                                            receiving_class_dict,color_dict, G_ratio_dict = G_ratio_dict,
                                                            if_plot = if_plot, legend_loc = 'upper right', plot_start_trans = plot_start_trans,
                                                           plot_start_stable = plot_start_stable, plot_duration = plot_duration)


# fig_trans , fig_stable = set_ylim_trans_stable_figs(fig_trans, fig_stable, ymax = [100, 100], ymin = [-4, -4])
save_trans_stable_figs(fig_trans, fig_stable, path_rate, filename.split('.')[0], figsize = (5,3), ymax = [75, 75], ymin = [-4, -4])

pkl_file = open(filename , 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto'
color = 'perc_t_oscil_mvt' #mvt_f'
param = 'perc_t_oscil_mvt'

xlabel = r'$G_{Loop}$'


g = Product_G(data)
fig = scatter_2d_plot(abs(g),
                np.squeeze(data[(name,param)]),
                np.squeeze(data[(name,color)]), 
                name + ' in STN-GPe circuit' , 
                [xlabel, param, color] )

plt.axvline(data['g_loop_stable'], c = 'k')
plt.axvline(data['g_loop_transient'], c = 'grey', linestyle = '--')
save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size = (8,6))

#%% RATE MODEL : STN-GPe without GPe-GPe tau-sweep ( GABA and glut)
N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 10000; t_list = np.arange(int(t_sim/dt))
t_mvt = 500 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]

name1 = 'Proto'
name2 = 'STN'
name_list = [name1, name2]

g_list = np.linspace(-5,-0.5, 150)


(synaptic_time_constant[(name2, name1)],
 synaptic_time_constant[(name1, name2)] )  =  [10], [6]

G = {
     (name2 , name1) : -1,
     (name1 , name2) : 1
     }

receiving_pop_list = {('STN', '1') : [('Proto', '1')],
                      ('Proto', '1') : [('STN', '1')] }

G_ratio_dict = { 
                ('STN', 'Proto'): 1, 
                ('Proto', 'STN'): -1
                }

lim_n_cycle = [6,10]
pop_list = [1]  

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

n = 30;
syn_decay_dict = {'tau_1': 
                          {
                           'tau_ratio':{ ('STN', 'Proto'): 1
                                        },
                           'tau_list': np.linspace(5, 25, n)
                           },
                'tau_2':{
                        'tau_ratio': { ('Proto', 'STN'): 1},
                        'tau_list': [6]}#np.linspace(1,15,n)}}
                        } 
    
filename = ( 'Tau_sweep_STN-GPe_tau_ratio_' + name1[0] + name2[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name2 , name1)]) + '_' + name2[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_2']['tau_ratio'][(name1 , name2)]) +  '_G_ratio_' + name1[0] + name2[0] + '_' +
            str( abs(G_ratio_dict[(name2 , name1)]) ) + '_' + name2[0] + name1[0] + '_' +
            str( abs(G_ratio_dict[(name1 , name2)]) ) + '_n_' + str(n) + '_T_' + str(t_sim)  + 
            '_dt_' + str(dt).replace('.','-'))

filename = os.path.join(path_rate, 
                        filename.replace('.', '-') +  '.pkl' 
                        )
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)


find_stable_oscill = True # to find stable oscillatory regime

sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, 
                  G.copy(),A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, 
                  lim_n_cycle,find_stable_oscill)

pkl_file = open( filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto' ; 
color = 'trans_n_half_cycle'
color = 'freq'

g_transient = data[(name,'g_transient')]
g_transient = data[(name,'g_stable')]

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
scatter_2d_plot(data['tau'][(name2, name1)][:,0],
                data[(name, 'stable_mvt_freq')][:,ind],
                data[(name, 'stable_mvt_freq')][:,ind], 
                    name +' in STN-GP circuit', 
                    [xlabel, 'Frequency(Hz)', 'Frequency(Hz)'] )
# save_pdf_png(fig, 'STN_GPe_timescale_inh_excit_3d.png', size = (8,6))

#%% RATE MODEL : GPe-GPe tau-sweep
N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 10000; t_list = np.arange(int(t_sim/dt))
t_mvt = 500 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]

name1 = 'Proto'
name_list = [name1]

g_list = np.linspace(-10, -0.5, 200, endpoint = True)

synaptic_time_constant = {
                        ('Proto', 'Proto') : [10]
                        }

G = {
     (name1 , name1) : -1
     }

G_ratio_dict = {
     (name1 , name1) : 1
     }
receiving_pop_list = {('Proto','1') : [('Proto', '1')]}


lim_n_cycle = [6,10]
pop_list = [1]  

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

n = 30;
syn_decay_dict = {'tau_1': 
                          {
                           'tau_ratio':{ ('Proto', 'Proto'): 1
                                        },
                           'tau_list': np.linspace(5, 25, n)
                           },
                'tau_2':{
                        'tau_ratio': { ('Proto', 'STN'): 1},
                        'tau_list': [6]}#np.linspace(1,15,n)}}
                        } 
    
filename = ( 'Tau_sweep_GPe-GPe_tau_ratio_' + name1[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name1 , name1)]) + '_' + name1[0] + name1[0] + '_' +
            str( abs(G_ratio_dict[(name1 , name1)]) ) + '_n_' + str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.','-'))

filename = os.path.join(path_rate, 
                        filename.replace('.', '-') +  '.pkl' 
                        )
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)


find_stable_oscill = True # to find stable oscillatory regime

sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, 
                  G.copy(),A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, 
                  lim_n_cycle,find_stable_oscill, if_track_tau_2= False)

pkl_file = open( filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto' ; 
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
scatter_2d_plot(data['tau'][(name1, name1)][:,0],
                data[(name, 'stable_mvt_freq')][:,ind],
                data[(name, 'stable_mvt_freq')][:,ind], 
                    name +' in STN-GP circuit', 
                    [xlabel, 'Frequency(Hz)', 'Frequency(Hz)'] )
# save_pdf_png(fig, 'STN_GPe_timescale_inh_excit_3d.png', size = (8,6))
#%% RATE MODEL : STN-GPe with GPe-GPe tau-sweep ( GABA and glut)

N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.5
t_sim = 10000; t_list = np.arange(int(t_sim/dt))
t_mvt = 400 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]

name1 = 'Proto'
name2 = 'STN'
name_list = [name1, name2]

g_list = np.linspace(-10,-0.01, 150)


(synaptic_time_constant[(name2, name1)],
 synaptic_time_constant[(name1, name1)],
 synaptic_time_constant[(name1, name2)] )  =  [10], [10], [6]

G = {
     (name2 , name1) : -1,
     (name1 , name2) : 1, 
     (name1 , name1) : -1}

receiving_pop_list = {('STN', '1') : [('Proto', '1')],
                      ('Proto', '1') : [('STN', '1'), ('Proto', '1')] }

G_ratio_dict = { 
                ('Proto', 'Proto'): 1, 
                ('STN', 'Proto'): 2, 
                ('Proto', 'STN'): -1
                }

lim_n_cycle = [6,10]
pop_list = [1]  

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

n = 30;
syn_decay_dict = {'tau_1': 
                          {
                           'tau_ratio':{ ('STN', 'Proto'): 1,
                                        ('Proto', 'Proto'): 1 },
                           'tau_list': np.linspace(5, 25, n)
                           },
                'tau_2':{
                        'tau_ratio': { ('Proto', 'STN'): 1},
                        'tau_list': [6]}#np.linspace(1,15,n)}}
                        } 
    
filename = ( 'Tau_sweep_STN-GPe-GPe_tau_ratio_' + name2[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name2 , name1)]) + '_' + name1[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name1 , name1)]) + '_' + name1[0] + name2[0] + '_' +
            str(syn_decay_dict['tau_2']['tau_ratio'][(name1 , name2)]) +'_G_ratio_' + name2[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name2 , name1)])) + '_' + name1[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name1 , name1)])) + '_' +  name1[0] + name2[0] + '_' +
            str(abs(G_ratio_dict[(name1 , name2)])) + '_n_' + str(n) + '_T_' + str(t_sim))

filename = os.path.join(path_rate, 
                        filename.replace('.', '-') +  '.pkl' 
                        )
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)


find_stable_oscill = True # to find stable oscillatory regime

sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, 
                  G.copy(),A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, 
                  lim_n_cycle,find_stable_oscill)
pkl_file = open( os.path.join(path_rate, filename) , 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
name = 'Proto' ; color = 'trans_n_half_cycle'
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
x_spec =  data['tau'][('Proto', 'Proto')][:,0]
y_spec = data[(name, 'stable_mvt_freq')][:,0]
xlabel  = r'$\tau_{inhibition} \; (ms)$'
# ax.azim = 60
# ax.dist = 10
# ax.elev = 30
# ax.scatter(x_spec,np.ones_like(x_spec)* y[0,4], 
#            y_spec, c = ['k'] * len(y_spec), s = 80)
scatter_2d_plot(x_spec,y_spec,y_spec, 
                'STN-GPe + GPe-GPe', 
                [xlabel, 'Frequency(Hz)', 'Frequency(Hz)'] )
# save_pdf_png(fig, 'STN_GPe_GPe-timescale_inh_excit_3d.png', size = (8,6))

#%% RATE MODEL : Arky-Proto-D2 loop without GPe-GPe
g = -2.7
dt = 0.5
t_sim = 1700; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]

name_list = {'Proto', 'D2', 'Arky'}
G = {}
G[('D2', 'Arky')], G[('Arky', 'Proto')], G[('Proto', 'D2')] = g*0.2, g, g*0.5

receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                      ('Proto','1') : [('D2', '1')],
                      ('D2','1') : [('Arky','1')]}

( synaptic_time_constant[('D2', 'Arky')], 
 synaptic_time_constant[('Arky', 'Proto')], 
 synaptic_time_constant[('Proto', 'D2')] ) =  [15],[6],[10]

pop_list = [1]  ; lim_n_cycle = [6,10]

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list)

run(receiving_class_dict,t_list, dt, nuclei_dict)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,ax = None, plot_start=500,title_fontsize=15,include_FR = False,
     title = r"$G_{AD}="+str(round(G[('D2', 'Arky')],2))+"$ "+", $G_{PA}="+str(round(G[('Arky', 'Proto')],2))+"$"+", $G_{DP}="+str(round(G[('Proto', 'D2')],2))+"$")
name = 'Arky'
nucleus = nuclei_dict[name][0]
fig.tight_layout()
find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
# temp_oscil_check(nuclei_dict[name][0].pop_act,oscil_peak_threshold[name], 3,dt,*duration_base)
figname = 'Arky-Proto-D2 loop without Proto-Proto'
fig.savefig(os.path.join(path_rate,figname+'.png'),dpi = 300)
fig.savefig(os.path.join(path_rate,figname+'.pdf'),dpi = 300)
#%% RATE MODEL : Arky-Proto-D2 loop with GPe-GPe
g = -2
t_sim = 800; t_list = np.arange(int(t_sim/dt))
t_mvt = 200 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]

name_list = {'Proto', 'D2', 'Arky'}

G[('D2', 'Arky')], G[('Arky', 'Proto')], G[('Proto', 'D2')] = g*0.2, g, g*0.5

G[('Proto', 'Proto')] = G[('Arky', 'Proto')]

receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                      ('Proto','1') : [('D2', '1'),('Proto','1')],
                      ('D2','1') : [('Arky','1')]}

(synaptic_time_constant[('D2', 'Arky')], 
synaptic_time_constant[('Arky', 'Proto')],
synaptic_time_constant[('Proto', 'D2')],
synaptic_time_constant[('Proto', 'Proto')] ) =  [30],[6],[10],[10]

pop_list = [1]  

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

run(receiving_class_dict,t_list, dt, nuclei_dict)
plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,ax = None, 
     title = r"$G_{Proto-Proto}="+str(round(G[('Proto', 'Proto')],2))+"$"+r" $G_{Arky-P}="+str(round(G[('Arky', 'Proto')],2))+"$\n"+r" $=2\times G_{P-D2}=5\times G_{D2-Arky}$")
name = 'Proto'
nucleus = nuclei_dict[name][0]
# find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
temp_oscil_check(nuclei_dict[name][0].pop_act,oscil_peak_threshold[name], 3,dt,*duration_mvt)
#%% RATE MODEL : Arky-Proto-D2 loop with GPe-GPe G-sweep
n = 50 ; if_plot = False
t_sim = 1500; t_list = np.arange(int(t_sim/dt))
t_mvt = 700 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
g_1_list = [-1]#np.linspace(-6, -1, n, endpoint = True)
g_2_list = np.linspace(-6, -1, n, endpoint = True)
receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1'),('Proto','1')],
                    ('D2','1') : [('Arky','1')]}
synaptic_time_constant[('D2', 'Proto')], synaptic_time_constant[('Arky', 'Proto')],synaptic_time_constant[('Proto', 'D2')],synaptic_time_constant[('Proto', 'Proto')]  =  [30],[6],[10],[10]

pop_list = [1]  
G_ratio_dict = {('D2', 'Arky') : .2, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 0.5, ('Proto', 'Proto') : 1}
G_dict = {('Proto','Proto'): g_1_list, ('D2', 'Arky') : g_2_list}
Proto = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
Arky = [Nucleus(i, gain, threshold,neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
lim_n_cycle = [6,10]
nuclei_dict = {'Proto': Proto, 'D2' : D2, 'Arky':Arky}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
filename = 'data_synaptic_weight_D2-P-A_with_P_P_tau_'+str(g_1_list[0])+'_'+str(synaptic_time_constant[('D2', 'Arky')][0])+'_'+str(synaptic_time_constant[('Arky', 'Proto')][0])+'_'+str(synaptic_time_constant[('Proto', 'D2')][0])+'_'+str(synaptic_time_constant[('Proto', 'Proto')][0])+'.pkl'
synaptic_weight_space_exploration(G.copy(),A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, receiving_class_dict,color_dict, G_ratio_dict=G_ratio_dict,if_plot=if_plot)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'D2'
color = 'n_half_cycles_mvt' #mvt_f'
param = 'mvt_freq'
g_transient = data[(name,'g_transient_boundary')][0] # 
# scatter_3d_wireframe_plot(data['g'][:,:,0],data['g'][:,:,1],data[(name,param)],data[(name,color)], name, [r'$G_{Proto-Proto}$', r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$', param,  color],x_axis = 'g_2') 
scatter_2d_plot(np.squeeze(data['g'][:,:,1]),np.squeeze(data[(name,param)]),np.squeeze(data[(name,color)]), name +' in Pallidostriatal',  [r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$' , param, color] )
plt.axvline(g_transient[1], c = 'k')
#%% RATE MODEL : FSI-D2-Proto loop without GPe-GPe

g = -1.7
dt = 0.1
t_sim = 1700; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
G = {}

name_list = {'Proto', 'D2', 'FSI'}

G[('D2', 'FSI')], G[('FSI', 'Proto')], G[('Proto', 'D2')] = g, g, g

receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
                      ('Proto','1') : [('D2', '1')],
                      ('D2','1') : [('FSI','1')]}

(synaptic_time_constant[('D2', 'FSI')], 
 synaptic_time_constant[('FSI', 'Proto')], 
 synaptic_time_constant[('Proto', 'D2')])  =  [10],[10],[10]


pop_list = [1]  

nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

run(receiving_class_dict,t_list, dt, nuclei_dict)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,ax = None, plot_start=500,title_fontsize=15, include_FR=False, axvspan_color = 'lightskyblue',
     title = r"$G_{FD}="+str(round(G[('D2', 'FSI')],2))+"$ "+", $G_{PF}="+str(round(G[('FSI', 'Proto')],2))+"$"+", $G_{DP}="+str(round(G[('Proto', 'D2')],2))+"$")

name = 'FSI'
nucleus = nuclei_dict[name][0]
fig.tight_layout()
state = 'Transient'
# figname = 'FSI-Proto-D2 loop without Proto-Proto_' + state
# fig.savefig(os.path.join(path_rate, figname+'.png'),dpi = 300)
# fig.savefig(os.path.join(path_rate, figname+'.pdf'),dpi = 300)
find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
temp_oscil_check(nuclei_dict[name][0].pop_act,oscil_peak_threshold[name], 3,dt,*duration_base)
#%% RATE MODEL : FSI-D2-Proto tau-sweep

n = 10 ; if_plot = False
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('FSI','1')]}
pop_list = [1]  
G_ratio_dict = {('D2', 'FSI') : 1, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 0.5}
Proto = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
FSI = [Nucleus(i, gain, threshold,neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
lim_n_cycle = [6,10]
nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
syn_decay_dict = {'tau_1': {'tau_ratio':{('D2', 'FSI') : 1, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 1},'tau_list':np.linspace(5,15,n)},
                'tau_2':{'tau_ratio':{('Proto', 'Proto'): 1},'tau_list': [5]}}#np.linspace(5,15,n)}}
g_list = np.linspace(-20,-0.01, 150); 
lim_n_cycle = [6,10] ; find_stable_oscill = True # to find stable oscillatory regime
# filename = 'data_FSI_D2_Proto_syn_t_scale_G_ratios_'+str(G_ratio_dict[('D2', 'FSI')])+'_'+str(G_ratio_dict[('FSI', 'Proto')])+'_'+str(G_ratio_dict[('Proto', 'D2')])

filename = 'data_FSI_D2_Proto_syn_t_scale_tau_1_1_1'
filename= filename.replace('.','-')+'.pkl'
sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)
#%% RATE MODEL : FSI-D2-Proto loop with GPe-GPe
g = -2
t_sim = 800; t_list = np.arange(int(t_sim/dt))
t_mvt = 200 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
G[('D2', 'FSI')], G[('FSI', 'Proto')], G[('Proto', 'D2')] = g, g, g*0.5
G[('Proto', 'Proto')] = -4.57
receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1'),('Proto','1')],
                    ('D2','1') : [('FSI','1')]}
synaptic_time_constant[('D2', 'FSI')], synaptic_time_constant[('FSI', 'Proto')],synaptic_time_constant[('Proto', 'D2')],synaptic_time_constant[('Proto', 'Proto')]  =  [30],[6],[10],[10]

pop_list = [1]  
Proto = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
FSI = [Nucleus(i, gain, threshold,neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

run(receiving_class_dict,t_list, dt, nuclei_dict)
plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,ax = None, title = r"$G_{Proto-Proto}="+str(G[('Proto', 'Proto')])+"$"+ r" $G_{D2-FSI}="+str(G[('D2', 'FSI')])+"$")
name = 'D2'
nucleus = nuclei_dict[name][0]
# find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
temp_oscil_check(nuclei_dict[name][0].pop_act,oscil_peak_threshold[name], 3,dt,*duration_mvt)

#%% RATE MODEL : Any Pallidostriatal tau-sweep

N_sim = 100
N = dict.fromkeys(N, N_sim)
if_plot = False
dt = 0.1
t_sim = 10000; t_list = np.arange(int(t_sim/dt))
t_mvt = 400 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]


name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI' 
name3 = 'Arky'
name_list = {name1, name2, name3}
receiving_pop_list = {(name3,'1') : [(name1, '1')], 
                      (name1,'1') : [(name2, '1')],
                      (name2,'1') : [(name3,'1')]}

synaptic_time_constant = {(name2 , name3) : [10], 
                          (name3 , name1) : [10],
                          (name1 , name2) : [10]
                          }

lim_n_cycle = [6,10]
pop_list = [1]  

G = {
     (name2 , name3) : 1,
     (name3 , name1) : 1, 
     (name1 , name2): 1
     }


G_ratio_dict = {
             (name2 , name3) : 1,
             (name3 , name1) : 1, 
             (name1 , name2): 1
             }

# G_ratio_dict = {   #### Shouldn't matter, right?
#                 ('D2', 'Arky') : 0.2, 
#                 ('Arky', 'Proto') : 1, 
#                 ('Proto', 'D2'): 0.5
#                 }


nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

n = 30 
syn_decay_dict = {'tau_1': {
                            'tau_ratio':{ (name2 , name3) : 1,
                                         (name3 , name1) : 1, 
                                         (name1 , name2): 1},
                            'tau_list': np.linspace( 5, 25, n)
                            },
                'tau_2':{
                        'tau_ratio':{('Proto', 'Proto'): 1},
                        'tau_list': [6]}
                        }#np.linspace(5,15,n)}}

filename = ( 'Tau_sweep_D2-P-' + name3[0] + '_tau_ratio_' + name3[0] + name2[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name2 , name3)]) + '_' + name1[0] + name3[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name3 , name1)]) + '_' + name2[0] + name1[0] + '_' +
            str(syn_decay_dict['tau_1']['tau_ratio'][(name1 , name2)]) + '_G_ratio_' + name3[0] + name2[0] + '_' +
            str(abs(G_ratio_dict[(name2 , name3)])) + '_' + name3[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name3 , name1)])) + '_' + name2[0] + name1[0] + '_' +
            str(abs(G_ratio_dict[(name1 , name2)])) + '_n_' +
            str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.','-'))

filename = os.path.join(path_rate, 
                        filename.replace('.', '-') +  '.pkl' 
                        )
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

g_list = np.linspace(-5,-0.8, 260); 
find_stable_oscill = True # to find stable oscillatory regime

sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, 
                  G.copy(),A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, 
                  lim_n_cycle,find_stable_oscill, if_track_tau_2=False)
#%% RATE MODEL : Any Pallidostriatal without GPe-GPe G-sweep
plt.close('all')
N_sim = 100
N = dict.fromkeys(N, N_sim)
n = 40 ; if_plot = False
dt = 0.1
t_sim = 20000; t_list = np.arange(int(t_sim/dt))
t_mvt = 700 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
plot_start_trans = t_mvt - 200 ; plot_duration = 600
plot_start_stable = 0

name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI' 
name3 = 'Arky'

name_list = {name1, name2, name3}

# G_list = np.linspace(-5, 0, n, endpoint = True)

transition_range = [2.46 - (2.695 - 2.46), 2.695] # FSI Loop dt = 0.5
transition_range = [2.197 - (2.204 - 2.197), 2.204] # FSI Loop dt = 0.1

transition_range = [1.903 - (1.915 - 1.903), 1.915] # Arky Loop dt = 0.1

# G_list = pad_high_res_spacing_with_linspace(0, transition_range[0], 20, transition_range[1], 4.5,  10, 10)
G_list = pad_high_res_spacing_with_arange(1, transition_range[0], 1/25, transition_range[1], 4.5,  1/10, 20)

G_list = - np.power(abs(G_list), 1/3)

receiving_pop_list = {(name3,'1') : [(name1, '1')], 
                      (name1,'1') : [(name2, '1')],
                      (name2,'1') : [(name3,'1')]}

(synaptic_time_constant[(name2 , name3)], 
 synaptic_time_constant[(name3 , name1)],
 synaptic_time_constant[(name1 , name2)])  =  [10],[10],[10]



G = {
     (name2 , name3) : -1,
     (name3 , name1) : -1, 
     (name1 , name2): -1
     }


G_ratio_dict = {
             (name2 , name3) : 1,
             (name3 , name1) : 1, 
             (name1 , name2): 1
             }

# G_ratio_dict = {   #### Shouldn't matter, right?
#                 ('D2', 'Arky') : 0.2, 
#                 ('Arky', 'Proto') : 1, 
#                 ('Proto', 'D2'): 0.5
#                 }


lim_n_cycle = [6,10]
pop_list = [1]  
nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)


filename = ( 'G_sweep_D2-P-' + name3[0] + '_tau_' + name3[0] + name2[0] + '_' +
            str(synaptic_time_constant[(name2 , name3)][0]) + '_' +  name3[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name3 , name1)][0]) + '_' + name2[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name1 , name2)][0]) + '_G_ratio_' + name3[0] + name2[0] + '_' +
            str( abs(G_ratio_dict[(name2 , name3)]) ) + '_' + name3[0] + name1[0] + '_' +
            str( abs(G_ratio_dict[(name3 , name1)])) + '_' + name2[0] + name1[0] + '_' +
            str( abs(G_ratio_dict[(name1 , name2)])) + '_n_' +
            str(n) + '_T_' + str(t_sim) +
            '_dt_' + str(dt).replace('.','-'))

filename = os.path.join(path_rate, 
                        filename.replace('.', '-') +  '.pkl' 
                        )
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
# nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
# fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, plot_start=0,title_fontsize=15,
#       title = r"$G_{SP}="+str(round(G[('Proto', 'STN')],2))+"$ "+", $G_{PS}=G_{PP}="+str(round(G[('STN', 'Proto')],2))+'$')

fig_trans , fig_stable = synaptic_weight_space_exploration(G.copy(), A, A_mvt, D_mvt, t_mvt, t_list, dt,  filename, 
                                                           lim_n_cycle, G_list, nuclei_dict, duration_mvt, duration_base, 
                                                           receiving_class_dict,color_dict, G_ratio_dict=G_ratio_dict,
                                                           if_plot = if_plot, plot_start_trans = plot_start_trans,
                                                           plot_start_stable = plot_start_stable, plot_duration = plot_duration)


# fig_trans , fig_stable = set_ylim_trans_stable_figs(fig_trans, fig_stable, ymax = [75, 75], ymin = [-4, -4])
save_trans_stable_figs(fig_trans, fig_stable, path_rate, filename.split('.')[0], figsize = (5,3), ymax = [75, 75], ymin = [-4, -4])
pkl_file = open(filename , 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto'
color = 'perc_t_oscil_mvt' #mvt_f'
param = 'perc_t_oscil_mvt'

xlabel = r'$G_{Loop}$'


fig = scatter_2d_plot(np.squeeze(abs(Product_G(data))),
                      np.squeeze(data[(name,param)]),
                      np.squeeze(data[(name,color)]), 
                      name + ' in ' + name3 + ' Loop',  
                      [xlabel, param, color] )


plt.axvline(data['g_loop_stable'], c = 'k')
plt.axvline(data['g_loop_transient'], c = 'grey', linestyle = '--')
save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size = (8,6))
# save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size = (8,6))
#%% RATE MODEL : Any Pallidostriatal with GPe-GPe G-sweep

n = 15 ; if_plot = False
dt = 0.5
t_sim = 1500; t_list = np.arange(int(t_sim/dt))
t_mvt = 700 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]

name1 = 'Proto'
name2 = 'D2'
name3 = 'FSI' 
# name3 = 'Arky'
name_list = {name1, name2, name3}

g_1_list = np.linspace(-2, -1, n, endpoint = True)
g_2_list = np.linspace(-2, -1, n, endpoint = True)

receiving_pop_list = {(name3,'1') : [(name1, '1')], 
                      (name1,'1') : [(name2, '1'), (name1, '1')],
                      (name2,'1') : [(name3,'1')]}

(synaptic_time_constant[(name2 , name3)], 
 synaptic_time_constant[(name3 , name1)],
 synaptic_time_constant[(name1 , name1)],
 synaptic_time_constant[(name1 , name2)])  =  [15],[6],[10], [10]

lim_n_cycle = [6,10]
pop_list = [1]  

G = {
     (name2 , name3) : 1,
     (name3 , name1) : 1, 
     (name1 , name1) : 1, 
     (name1 , name2): 1
     }


G_ratio_dict = {
             (name2 , name3) : 1,
             (name3 , name1) : 1, 
             (name1 , name1) : 1, 
             (name1 , name2): 1
             }

# G_ratio_dict = {   #### Shouldn't matter, right?
#                 ('D2', 'Arky') : 0.2, 
#                 ('Arky', 'Proto') : 1, 
#                 (name1 , name1) : 1, 
#                 ('Proto', 'D2'): 0.5
#                 }


G_dict = {
         ('Proto','Proto'): g_1_list, 
          (name2, name3) : g_2_list
          }


nuclei_dict = {name: [Nucleus(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude,
                              N, A, A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                              smooth_kern_window, oscil_peak_threshold) for i in pop_list] for name in name_list}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)



filename = ( 'G_sweep_D2-P-' + name3[0] + '_tau_' + name3[0] + name2[0] + '_' +
            str(synaptic_time_constant[(name2 , name3)][0]) + '_' +  name3[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name3 , name1)][0]) + '_' + name2[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name1 , name2)][0]) + '_' + name1[0] + name1[0] + '_' +
            str(synaptic_time_constant[(name1 , name1)][0]) + '_G_ratio_' + name3[0] + name2[0] + '_' +
            str(G_ratio_dict[(name2 , name3)]) + '_' + name3[0] + name1[0] + '_' +
            str(G_ratio_dict[(name3 , name1)]) + '_' + name2[0] + name1[0] + '_' +
            str(G_ratio_dict[(name1 , name2)]) + '_' + name1[0] + name1[0] + '_' +
            str(G_ratio_dict[(name1 , name1)]) + '_n_' +
            str(n) )

filename = os.path.join(path_rate, 
                        filename.replace('.', '-') +  '.pkl' 
                        )

fig_trans , fig_stable = synaptic_weight_space_exploration(G.copy(), A, A_mvt, D_mvt, t_mvt, t_list, dt,  filename, 
                                                           lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, 
                                                           receiving_class_dict,color_dict, G_ratio_dict=G_ratio_dict,
                                                           if_plot = if_plot, plt_start = 500)
pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'D2'
color = 'n_half_cycles_mvt' #mvt_f'
param = 'mvt_freq'

xlabel = r'$G_{Loop}$' 


fig = scatter_2d_plot(np.squeeze(abs(Product_G(data))),
                      np.squeeze(data[(name,param)]),
                      np.squeeze(data[(name,color)]), 
                      name + ' in ' + name3 + ' Loop',  
                      [xlabel, param, color] )


plt.axvline(data['g_loop_stable'], c = 'k')
plt.axvline(data['g_loop_transient'], c = 'grey', linestyle = '--')
save_pdf_png(fig, os.path.join(path_rate, filename).split('.')[0], size = (8,6))

#%% RATE MODEL : frequency vs. G - multiple tau-ratios
g_cte_ind = [0,0,0]; g_ch_ind = [1,1,1]
filename_list = [os.path.join(path_rate, 'data_synaptic_weight_Pallidostriatal.pkl'),
                 os.path.join(path_rate, 'data_synaptic_weight_Pallidostriatal_30_10_10.pkl'),
                 os.path.join(path_rate, 'data_synaptic_weight_Pallidostriatal_30_6_6.pkl')]
nucleus_name_list = ['Proto', 'Proto','Proto']
legend_list = [r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=10$ $\tau_{Proto-D2}=6$', 
               r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=10$ $\tau_{Proto-D2}=10$', 
               r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=6$ $\tau_{Proto-D2}=6$']
color_list = ['k','r','g']
param_list = 3*['mvt_freq']
color_param_list = 3* ['perc_t_oscil_mvt']
x_label = r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$'
synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list, 
                                             color_list,g_cte_ind,g_ch_ind,param_list,color_param_list,
                                             'jet',x_axis = 'g_2',x_label = x_label)
#%% RATE MODEL : frequency vs. G - all circuits
filename_list = ['G_sweep_Proto-Proto_tau_10_n_50_T_15000.pkl',
                 'G_sweep_STN-GPe_tau_PS_10_SP_6_n_50_T_15000.pkl',
                'G_sweep_D2-P-A_tau_AD_10_AP_10_DP_10_G_ratio_AD_1_AP_1_DP_1_n_40_T_15000.pkl',
                'G_sweep_D2-P-F_tau_FD_10_FP_10_DP_10_G_ratio_FD_1_FP_1_DP_1_n_40_T_15000.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

n = len(filename_list)
nucleus_name_list = n *  ['Proto']
legend_list = [r'$Proto-Proto$', r'$STN-Proto$',  r'$FSI-D2-Proto$',r'$Arky-D2-Proto$']
color_list = [color_dict['Proto'], color_dict['STN'], color_dict['FSI'], color_dict['Arky']]
color_param_list = n * ['perc_t_oscil_mvt']
y_list = n * ['mvt_freq']
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list, 
                                                   color_list, y_list,  marker_c_list = color_param_list,  colormap ='YlOrBr', 
                                                   x_axis = 'multiply', title = "", markersize = 50, alpha_transient = 0.3,
                                                   x_label = r"$ \vert G_{Loop} \vert$",  leg_loc = 'upper left', g_key = None,
                                                   vline_txt = False, colorbar = True, ylabel = 'frequency(Hz)',
                                                   vline_width = 2, lw = 3, xlim = None)
save_pdf_png(fig, 'Freg_vs_G_Oscil_per_as_color_all_circuits', size = (10,3))
#%% RATE MODEL : % Oscillation vs. G ( FSI loop) 

nucleus_name_list = ['Proto']
n = len(nucleus_name_list)

filename_list = n * ['G_sweep_D2-P-F_tau_FD_10_FP_10_DP_10_G_ratio_FD_1_FP_1_DP_1_n_40_T_20000.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{D2-FSI}=\tau_{FSI-Proto}=\tau_{Proto-D2}=10 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['FSI']]
param_list = n * ['perc_t_oscil_mvt']
x_label = r'$ \lvert G_{Loop} \lvert$'
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list, 
                                             color_list, param_list, colormap = 'hot', colorbar = False, marker_c_list = None,
                                             x_axis = 'multiply', ylabel= '% Oscillation', title = title,x_label = x_label, 
                                             vline_txt = False, markersize = 50)
save_pdf_png(fig, filename_list[0].split('.')[0], size = (5,3))
#%% RATE MODEL : % Oscillation vs. G ( Arky loop )

nucleus_name_list = [ 'Proto']
n = len(nucleus_name_list)

filename_list = n * ['G_sweep_D2-P-A_tau_AD_10_AP_10_DP_10_G_ratio_AD_1_AP_1_DP_1_n_40_T_20000.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{D2-Arky}=\tau_{Arky-Proto}=\tau_{Proto-D2}=10 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['Arky']]
param_list = n * ['perc_t_oscil_mvt']
x_label = r'$ \lvert G_{Loop} \rvert$'
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list,
                                             color_list, param_list, colormap = 'hot', colorbar = False, marker_c_list = None,
                                             x_axis = 'multiply', ylabel= '% Oscillation', title = title,x_label = x_label, 
                                             vline_txt = False , markersize = 50)
save_pdf_png(fig, filename_list[0].split('.')[0], size = (5,3))
#%% RATE MODEL : % Oscillation vs. G ( STN-GP )

nucleus_name_list = [ 'Proto']
n = len(nucleus_name_list)

filename_list = n * ['G_sweep_STN-GPe_tau_PS_10_SP_6_n_50_T_20000.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

title = r'$\tau_{STN-Proto}=10\: ms \; , \tau_{Proto-STN}= 6 \: ms$'
title = ''
legend_list = n * ['']
# color_list = [color_dict[name] for name in nucleus_name_list] ## if all nuclei are plotted
color_list = [color_dict['STN']]
param_list = n * ['perc_t_oscil_mvt']
x_label = r'$ \lvert G_{Loop} \lvert$'
fig = synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list,
                                             color_list, param_list, colormap = 'hot', colorbar = False, marker_c_list = None,
                                             x_axis = 'multiply', ylabel= '% Oscillation', title = title,x_label = x_label,
                                             vline_txt = False, markersize = 50)
save_pdf_png(fig, filename_list[0].split('.')[0], size = (5,3))

#%% RATE MODEL : % Oscillation vs. G ( GP-GP )

nucleus_name_list = [ 'Proto']
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
                                             color_list, param_list, colormap = 'hot', colorbar = False, marker_c_list = None,
                                             x_axis = 'multiply', ylabel= '% Oscillation', title = title,x_label = x_label,
                                             vline_txt = False, markersize = 50)
save_pdf_png(fig, filename_list[0].split('.')[0], size = (5,3))
#%% RATE MODEL : frequency vs. G ( STN-GP and GP-GP ) - multiple GP-GP GP-STN G-ratios

filename_list = ['data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_2_N_100_T_800_dt_0-1.pkl', 
                  'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_1_N_100_T_800_dt_0-1.pkl',  ## STN weight is half the inhibition
                  'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_0-5_N_100_T_800_dt_0-1.pkl']
figname = 'STN_GPe_with_GP_GP_f_vs_tau_inh_different_P_proj_G_ratios'#'_STN_one'
filename_list = [os.path.join(path_rate,'STN_weight_constant', filename) for filename in filename_list]

filename_list = ['data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_2_N_100_T_800_dt_0-1_STN_1.pkl', 
                  'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_1_N_100_T_800_dt_0-1_STN_1.pkl',  ## STN weight is equal the inhibition
                  'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_0-5_N_100_T_800_dt_0-1_STN_1.pkl']
figname = 'STN_GPe_with_GP_GP_f_vs_tau_inh_different_P_proj_G_ratios_STN_one'
filename_list = [os.path.join(path_rate, 'STN_weight_constant',filename) for filename in filename_list]

filename_list = ['data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_2_N_100_T_800_dt_0-1_STN_changing.pkl', 
                  'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_1_N_100_T_800_dt_0-1_STN_changing.pkl',  ## STN weight is equal the inhibition
                  'data_STN_GPe_with_GP_GP_syn_t_scale_g_ratio_0-5_N_100_T_800_dt_0-1_STN_changing.pkl']
figname = 'STN_GPe_with_GP_GP_f_vs_tau_inh_different_P_proj_G_ratios_STN_changing'

filename_list = ['Tau_sweep_STN-GPe-GPe_tau_ratio_SP_1_PP_1_PS_1_G_ratio_SP_2_PP_1_PS_1_n_30_T_10000.pkl',
                 'Tau_sweep_STN-GPe-GPe_tau_ratio_SP_1_PP_1_PS_1_G_ratio_SP_1_PP_1_PS_1_n_30_T_10000.pkl',
                 'Tau_sweep_STN-GPe-GPe_tau_ratio_SP_1_PP_1_PS_1_G_ratio_SP_1_PP_2_PS_1_n_30_T_10000.pkl']

pkl_file = open( os.path.join(path_rate, filename_list[0]) , 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

filename_list = [os.path.join(path_rate, filename) for filename in filename_list]
n = len(filename_list)
x_label = r'$\tau_{decay}^{inhibition}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label
label_list = [r'$G_{SP}=2\times G_{PP}$',r'$G_{SP}=G_{PP}$',r'$G_{SP}=\dfrac{G_{PP}}{2}$']
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
color_list = ['k','maroon','red']
fig, ax2 = plt.subplots(1, 1, sharex=True, figsize =(6,5))
i = 0
def plot__(ax):
    for i in range(len(filename_list)):
        # i = 0; filename_list[i] = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec =  data['tau'][key_list[i]][:,0]
        print(data[(name_list[i], y_list[i])].shape)
        y_spec = data[(name_list[i], y_list[i])][:,0]
        c_spec = data[(name_list[i], c_list[i])][:,0]
        # ax.plot(x_spec,y_spec, marker = markerstyle[i], c = color_list[i], lw = 1, label= label_list[i],zorder = 1, mec = color_list[i])
        ax.scatter(x_spec,y_spec, marker = markerstyle[i], c = color_list[i], label= label_list[i])

plot__(ax2)


ax2.set_xlabel(r'$\tau_{decay}^{inhibition}$',fontsize = 20)
ax2.set_ylabel( 'Frequency (Hz)',  fontsize = 20)

# fig.text( -0.01, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize = 18)
y_formatter = FixedFormatter(['40', '55',   '70'])
y_locator = FixedLocator([ 40, 55,  70])
ax2.yaxis.set_major_formatter(y_formatter)
ax2.yaxis.set_major_locator(y_locator)

remove_frame(ax2)
ax2.set_xlim(4,26)
ax2.set_ylim(38,70)
ax2.legend(fontsize = 12, frameon = False, framealpha = 0.1, bbox_to_anchor=(.5, 0.85), bbox_transform=ax2.transAxes)
save_pdf_png(fig, os.path.join(path_rate, 'F_vs_tau_Multiple_g_ratios_STN-GPe-GPe'), size = (5,6))

#%% RATE MODEL : frequency vs. tau_inhibition (FSI and Arky loops) - multiple tau ratios


g_tau_2_ind = 0 

filename_list = ['data_Arky_D2_Proto_syn_t_scale_tau_3_1_1.pkl',
                 'data_Arky_D2_Proto_syn_t_scale_tau_2_1_1.pkl',
                 'data_Arky_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{3}$',r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{2}$',r'$\tau_{PA}=\tau_{DP}=\tau_{AD}$']
figname = 'Arky-D2-Proto_timescale_g_stable'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label; title = ''
name_list = ['Proto']*3
y_list  = ['stable_mvt_freq']*3
# color_list = ['k','grey','lightgrey']
fig,ax = plt.subplots(1,1)
color_list = create_color_map(len(filename_list) + 1, colormap = plt.get_cmap('Oranges'))
color_list = color_list[::-1]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label, 
                                    g_tau_2_ind = None, ylabelpad = -5, title = '', c_label = '', ax = ax)


filename_list = ['data_FSI_D2_Proto_syn_t_scale_tau_3_1_1.pkl', 
                 'data_FSI_D2_Proto_syn_t_scale_tau_2_1_1.pkl', 
                 'data_FSI_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{PF}=\tau_{DP}=\dfrac{\tau_{FD}}{3}$',r'$\tau_{PF}=\tau_{DP}=\dfrac{\tau_{FD}}{2}$',r'$\tau_{PF}=\tau_{DP}=\tau_{FD}$']
figname = 'FSI-D2-Proto_timescale_tau_g_stable'
x_label = r'$\tau_{PF/DP}^{decay}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label; title = ''

color_list = create_color_map(len(filename_list) + 1, colormap = plt.get_cmap('Greens'))
color_list = color_list[::-1]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label, 
                                    g_tau_2_ind = None, ylabelpad = -5, title = '', c_label = '', ax = ax)
ax.legend(fontsize = 10)

fig.savefig(figname+'.png',dpi = 300)
fig.savefig(figname+'.pdf',dpi = 300)

#%% RATE MODEL : frequency vs. tau_inhibition (Arky Loop)

filename_list = ['Tau_sweep_D2-P-A_tau_ratio_AD_1_PA_1_DP_1_G_ratio_AD_1_AP_1_DP_1_n_30']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{AP}=\tau_{PD}=\tau_{DA}$']
figname = 'test'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$' ; y_label = 'frequency (Hz)' ; c_label = y_label; title = ''
name_list = ['Proto']
y_list  = ['stable_mvt_freq']
fig,ax = plt.subplots(1,1)
color_list = create_color_map(len(filename_list) + 1, colormap = plt.get_cmap('Oranges'))
color_list = color_list[::-1]
color_list = [color_dict['Arky']]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label, 
                                    tau_2_ind = 0, ylabelpad = -5, title = '', c_label = '', ax = ax, key = ('Proto', 'D2'))

#%% RATE MODEL : frequency vs. tau_inhibition (FSI Loop)

filename_list = ['Tau_sweep_D2-P-F_tau_ratio_FD_1_PF_1_DP_1_G_ratio_FD_1_FP_1_DP_1_n_30.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{FP}=\tau_{PD}=\tau_{DF}$']
figname = 'test'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$' ; y_label = 'frequency (Hz)' ; c_label = y_label; title = ''
name_list = ['Proto']
y_list  = ['stable_mvt_freq']
fig, ax = plt.subplots(1,1)
# color_list = create_color_map(len(filename_list) + 1, colormap = plt.get_cmap('Oranges'))
# color_list = color_list[::-1]
color_list = [color_dict['FSI']]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label, 
                                    tau_2_ind = 0, ylabelpad = -5, title = '', c_label = '', ax = ax, key = ('Proto', 'D2'))

#%% RATE MODEL : frequency vs. tau_inhibition (GPe-GPe Loop)

filename_list = ['Tau_sweep_STN-GPe_tau_ratio_PS_1_SP_1_G_ratio_PS_1_SP_1_n_30.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = ['']
figname = 'test'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$' ; y_label = 'frequency (Hz)' ; c_label = y_label; title = ''
name_list = ['Proto']
y_list  = ['stable_mvt_freq']
fig,ax = plt.subplots(1,1)
color_list = [color_dict['Proto']]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label, 
                                    tau_2_ind = 0, ylabelpad = -5, title = '', c_label = '', ax = ax, key = ('Proto', 'Proto'))

#%% RATE MODEL : frequency vs. tau_inhibition (STN-GPe Loop)

filename_list = ['Tau_sweep_STN-GPe_tau_ratio_PS_1_SP_1_G_ratio_PS_1_SP_1_n_30.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = [r'$\tau_{PS}=6 \; ms$']
figname = 'test'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$' ; y_label = 'frequency (Hz)' ; c_label = y_label; title = ''
name_list = ['Proto']
y_list  = ['stable_mvt_freq']
fig,ax = plt.subplots(1,1)
color_list = [color_dict['STN']]
fig, ax = multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label, 
                                    tau_2_ind = 0, ylabelpad = -5, title = '', c_label = '', ax = ax, key = ('STN', 'Proto'))
#%% RATE MODEL : frequency vs. tau_inhibition (All Loops) new
plt.close('all')
filename_list = ['Tau_sweep_GPe-GPe_tau_ratio_PP_1_PP_1_n_30_T_10000_dt_0-1.pkl',
                 'Tau_sweep_STN-GPe_tau_ratio_PS_1_SP_1_G_ratio_PS_1_SP_1_n_30_T_10000_dt_0-1.pkl',
                'Tau_sweep_D2-P-F_tau_ratio_FD_1_PF_1_DP_1_G_ratio_FD_1_FP_1_DP_1_n_30_T_10000_dt_0-1.pkl',
                'Tau_sweep_D2-P-A_tau_ratio_AD_1_PA_1_DP_1_G_ratio_AD_1_AP_1_DP_1_n_30_T_10000_dt_0-1.pkl']

filename_list = [os.path.join(path_rate, file) for file in filename_list]
figname = 'All_circuits_timescale'
label_list = ['Proto-Proto', 'STN-Proto',  'FSI-D2-Proto','Arky-D2-Proto']
g_tau_2_ind = 0
color_list =  create_color_map(len(filename_list), colormap = plt.get_cmap('viridis'))
color_list = [color_dict['Proto'], color_dict['STN'], color_dict['FSI'], color_dict['Arky']]
key_list = [('Proto', 'Proto'), ('STN', 'Proto'), ('Proto', 'D2'), ('Proto', 'D2')]
c_list = ['stable_mvt_freq'] * len(filename_list)
y_list = c_list
colormap = 'hot'
title = ''
c_label = 'frequency (Hz)'
name_list = ['Proto'] * len(filename_list)
markerstyle = ['+', 's', 'o', '^']
fig, ax2 = plt.subplots(1, 1, sharex=True, figsize =(6,5))
i= 1
def plot__(ax):
    for i in range(len(filename_list)):
        # i = 0; filename_list[i] = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec =  data['tau'][key_list[i]][:,0]
        print(data[(name_list[i], y_list[i])].shape)
        y_spec = data[(name_list[i], y_list[i])][:,0]
        c_spec = data[(name_list[i], c_list[i])][:,0]
        # ax.plot(x_spec,y_spec, marker = 's', c = color_list[i], lw = 1, label= label_list[i],zorder = 1, mec = 'k')
        ax.scatter(x_spec,y_spec, marker = 's', c = color_list[i], lw = 0.2, label= label_list[i],zorder = 1, s = 40)#,  ec = 'k')
        
plot__(ax2)


ax2.set_xlabel(r'$\tau_{decay}^{inhibition}$',fontsize = 20)
ax2.set_ylabel( 'Frequency (Hz)',  fontsize = 20)

# fig.text( -0.01, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize = 18)

fig = set_y_ticks(fig, [5, 20, 40, 60,  80])
fig = set_x_ticks(fig, [5, 15, 25])

ax2.tick_params(axis='both', labelsize=22)

remove_frame(ax2)
ax2.set_xlim(4,26)
ax2.set_ylim(5,70)
ax2.legend(fontsize = 18, frameon = False, framealpha = 0.1, bbox_to_anchor=(.4, 0.5), bbox_transform=ax2.transAxes)
ax2.axhspan(13,30, color = 'lightgrey', alpha = 0.5, zorder = 0)

save_pdf_png(fig, os.path.join(path_rate, figname), size = (5,6))
#%% RATE MODEL : G vs. tau_inhibition (All Loops) 
plt.close('all')
filename_list = ['Tau_sweep_GPe-GPe_tau_ratio_PP_1_PP_1_n_30_T_10000.pkl',
                 'Tau_sweep_STN-GPe_tau_ratio_PS_1_SP_1_G_ratio_PS_1_SP_1_n_30_T_10000.pkl',
                'Tau_sweep_D2-P-F_tau_ratio_FD_1_PF_1_DP_1_G_ratio_FD_1_FP_1_DP_1_n_30_T_10000.pkl',
                'Tau_sweep_D2-P-A_tau_ratio_AD_1_PA_1_DP_1_G_ratio_AD_1_AP_1_DP_1_n_30_T_10000.pkl']

filename_list = [os.path.join(path_rate, file) for file in filename_list]
label_list = ['Proto-Proto', 'STN-Proto',  'FSI-D2-Proto','Arky-D2-Proto']
g_tau_2_ind = 0
color_list =  create_color_map(len(filename_list), colormap = plt.get_cmap('viridis'))
color_list = [color_dict['Proto'], color_dict['STN'], color_dict['FSI'], color_dict['Arky']]
key_list = [('Proto', 'Proto'), ('STN', 'Proto'), ('Proto', 'D2'), ('Proto', 'D2')]
c_list = ['stable_mvt_freq'] * len(filename_list)
y_list = c_list
colormap = 'hot'
title = ''
c_label = 'frequency (Hz)'
name_list = ['Proto'] * len(filename_list)
markerstyle = ['+', 's', 'o', '^']
fig, ax2 = plt.subplots(1, 1, sharex=True, figsize =(6,5))
i= 1

def get_g_stable_loop(data):
    g_stables = {k: v for k, v in data.items() if k[1] == 'g_stable'}
    n = len( list(g_stables.values()) [0])
    gg = np.ones((n,1))
    for k, v in g_stables.items():
        gg = gg * v
    return abs(gg)   
 
def plot__(ax, y = None):
    for i in range(len(filename_list)):
        # i = 0; filename_list[i] = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec =  data['tau'][key_list[i]][:,0]
        y_spec = get_g_stable_loop(data)
        c_spec = data[(name_list[i], c_list[i])][:,0]
        ax.scatter(x_spec,y_spec, marker = 's', c = color_list[i], lw = 0.2, label= label_list[i],zorder = 1, s = 20,  ec = 'k')
    


plot__(ax2, y = 'g_stable')
figname = 'All_circuits_G_vs_tau_inhibition'
ax2.set_ylabel( r'$|G_{Loop}|$',  fontsize = 20)
ax2.set_xlabel(r'$\tau_{decay}^{inhibition}$',fontsize = 20)
y_formatter = FixedFormatter(['0', '2', '4', '6',  '8', '10'])
y_locator = FixedLocator([0, 2, 4, 6, 8, 10])
ax2.yaxis.set_major_formatter(y_formatter)
ax2.yaxis.set_major_locator(y_locator)
ax2.set_xlim(4,26)
ax2.set_ylim(0,10)
remove_frame(ax2)
ax2.legend(fontsize = 12, frameon = False, framealpha = 0.1, bbox_to_anchor=(.5, 0.75), bbox_transform=ax2.transAxes)
save_pdf_png(fig, os.path.join(path_rate, figname), size = (5,6))

#%% RATE MODEL : frequency vs. tau_inhibition (all loops)
###################3 The idiot that I am saved the pickles without instruction. Here I guess the excitatory time scale is set to 6 while changing the inhibition decay time

plt.close('all')
# filename_list = ['data_STN_GPe_syn_t_scale_g_ratio_1.pkl','data_STN_GPe_without_GP_GP_syn_t_scale_g_ratio_1.pkl',
                 # 'data_FSI_D2_Proto_syn_t_scale_G_ratios_1_1_0-5.pkl','data_Arky_D2_Proto_syn_t_scale_G_ratios_0-2_1_0-5.pkl']
filename_list = [os.path.join(path_rate,'data_STN_GPe_with_GP_GP_syn_t_scale_N_100_T_800_dt_0-1_K_scaled.pkl'),
                 os.path.join(path_rate, 'data_STN_GPe_without_GP_GP_syn_t_scale_g_ratio_1_N_100_T_800_dt_0-1_STN_chaning_npop_2.pkl'),
                 os.path.join(path_rate,'data_FSI_D2_Proto_syn_t_scale_tau_1_1_1.pkl'),
                 os.path.join(path_rate,'data_Arky_D2_Proto_syn_t_scale_tau_1_1_1.pkl')]
figname = 'All_circuits_timescale'
label_list = [r'$STN-Proto + Proto-Proto$', r'$STN-Proto$',  r'$FSI-D2-Proto$',r'$Arky-D2-Proto$']
g_tau_2_ind = 0
color_list =  create_color_map(len(filename_list), colormap = plt.get_cmap('viridis'))
color_list = ['plum', 'lightcoral', 'green', 'orange']
c_list = ['stable_mvt_freq'] * len(filename_list)
y_list = c_list
colormap = 'hot'
title = ''
c_label = 'frequency (Hz)'
name_list = ['Proto'] * len(filename_list)
markerstyle = ['+', 's', 'o', '^']
fig, ax2 = plt.subplots(1, 1, sharex=True, figsize =(6,5))
def plot__(ax):
    for i in range(len(filename_list)):
        # i = 0; filename_list[i] = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec =  data['tau'][:,:,0][:,0]
        print(data[(name_list[i], y_list[i])].shape)
        y_spec = data[(name_list[i], y_list[i])][:,g_tau_2_ind]
        c_spec = data[(name_list[i], c_list[i])][:,g_tau_2_ind]
        ax.plot(x_spec,y_spec, marker = markerstyle[i], c = color_list[i], lw = 1, label= label_list[i],zorder = 1)
plot__(ax2)
#############################3 Broken axis 
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


ax2.set_xlabel(r'$\tau_{decay}^{inhibition}$',fontsize = 20)
ax2.set_ylabel( 'Frequency (Hz)',  fontsize = 20)

# fig.text( -0.01, 0.5, 'Frequency (Hz)', va='center', rotation='vertical', fontsize = 18)
ax_label_adjust(ax2, fontsize = 20, nbins= 5, ybins = 6)
remove_frame(ax2)
ax2.set_xlim(4,32)
ax2.set_ylim(10,55)
ax2.legend(fontsize = 12, frameon = False, framealpha = 0.1, bbox_to_anchor=(.4, 0.3), bbox_transform=ax2.transAxes)
ax2.axhspan(13,30, color = 'lightgrey', alpha = 0.5)
fig.tight_layout()
fig.savefig(os.path.join(path_rate, ('All_circuits_plus_STN_GP_without_GP_GP_Freq_vs_tau_STN_GPe_GPe_Gs_scaled_with_K.png')),dpi = 300, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True )#,bbox_inches = "tight", pad_inches=0.1)
fig.savefig(os.path.join(path_rate, ('All_circuits_plus_STN_GP_without_GP_GP_Freq_vs_tau_STN_GPe_GPe_Gs_scaled_with_K.pdf')),dpi = 300, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

#%% RATE MODEL : <PARAMETER> vs. G
#################### All circuits
filename_list = ['data_STN_GPe_syn_t_scale_g_ratio_1.pkl',
                 'data_FSI_D2_Proto_syn_t_scale_G_ratios_1_1_0-5.pkl',
                 'data_Arky_D2_Proto_syn_t_scale_G_ratios_0-2_1_0-5.pkl']
figname = 'All_circuits_timescale'
label_list = [r'$STN-Proto \; with \; Proto-Proto$',r'$FSI-D2-Proto$',r'$Arky-D2-Proto$']
g_tau_2_ind = 0; g_ratio_list = [1,1,1]
################### <<frequency>> multiple time scale ratios

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
################## <<g transition>> for multiple time scale or g ratios 
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

filename_list = ['data_Arky_D2_Proto_syn_t_scale_tau_3_1_1.pkl','data_Arky_D2_Proto_syn_t_scale_tau_2_1_1.pkl','data_Arky_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
filename_list = [os.path.join(path_rate, file) for file in filename_list]

label_list = [r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{3}$',r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{2}$',r'$\tau_{PA}=\tau_{DP}=\tau_{AD}$']
figname = 'Arky-D2-Proto_timescale_g_stable'
x_label = r'$\tau_{PA/DP}^{decay}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label; title = ''

# name_list = ['Proto']*3
# g_ratio_list = [1,1,1]
color_list = ['k','grey','lightgrey']
c_list = ['stable_mvt_freq']*3
y_list = c_list
colormap = 'hot'
title = ''
c_label = 'frequency (Hz)'
##################33
fig = multi_plot_as_f_of_timescale_shared_colorbar(y_list, color_list,c_list,  label_list,name_list,filename_list,x_label,y_label,c_label = c_label,
                                                   ylabelpad = 0, g_ratio_list=g_ratio_list, g_tau_2_ind = g_tau_2_ind, title = title)
fig.savefig(figname+'.png',dpi = 300)
fig.savefig(figname+'.pdf',dpi = 300)
#%% RATE MODEL : time scale space (GABA-a, GABA-b)

receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
#receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
#                    ('Proto','1') : [('Proto', '1'), ('STN', '1')],
#                    ('Proto','2') : [('Proto', '2'), ('STN', '2')]}
synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')],synaptic_time_constant[('Proto', 'STN')]  =  [decay_time_scale['GABA-A'],decay_time_scale['GABA-B']],[decay_time_scale['GABA-A'],decay_time_scale['GABA-B']],[decay_time_scale['Glut']]

pop_list = [1,2]  
Proto = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold,neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'STN', G, T, t_sim, dt, synaptic_time_constant,  receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
nuclei_dict = {'Proto': Proto, 'STN' : STN}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real_STN_Proto_diverse, receiving_pop_list, nuclei_dict,t_list)

n = 1
Glut = np.linspace(2,16,n)
GABA_A = np.linspace(5,32,n); GABA_B = np.linspace(150,310,n)
g_list = np.linspace(-20,-0.01, 100); g_ratio = 1
lim_n_cycle = [6,10] ; find_stable_oscill = True # to find stable oscillatory regime
filename = 'data_GABA_A_GABA_B.pkl'
sweep_time_scales_STN_GPe(g_list,g_ratio,nuclei_dict, GABA_A,GABA_B, Glut, filename,G.copy(), A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
name = 'Proto' ; color = 'trans_n_half_cycle'
x = data['synaptic_time_constant'][:,:,:,0].flatten()
y = data['synaptic_time_constant'][:,:,:,1].flatten()
z = data['synaptic_time_constant'][:,:,:,2].flatten()
c_trans = data[(name,'trans_mvt_freq')].flatten()
c_stable = data[(name, 'stable_mvt_freq')].flatten()
c = c_trans
scatter_3d_plot(x,y,z,c,name, np.max(c), np.min(c),['GABA-A','GABA-B','Glut','transient oscillation f'], limits = None)
#%% RATE MODEL : time scale space GABA-B

t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = int(t_sim/2); D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
G[('Proto','STN')] = 1
receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}

synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')],synaptic_time_constant[('Proto', 'STN')]  =  [decay_time_scale['GABA-B']],[decay_time_scale['GABA-B']],[decay_time_scale['Glut']]

pop_list = [1,2]
Proto = [Nucleus(i, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold,neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'STN', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
nuclei_dict = {'Proto': Proto, 'STN' : STN}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real_STN_Proto_diverse, receiving_pop_list, nuclei_dict,t_list)

inhibitory_trans = 'GABA_B'; n = 2
Glut = np.linspace(4,16,n)
GABA_A = np.linspace(5,32,n); GABA_B = np.linspace(150,300,n)
inhibitory_series = GABA_B
g_list = np.linspace(-80,-0.01, 80) ; g_ratio = 1
lim_n_cycle = [6,10] ; find_stable_oscill = True # to find stable oscillatory regime
filename = 'data_'+inhibitory_trans+'.pkl'
sweep_time_scales_one_GABA_STN_GPe(g_list,g_ratio, nuclei_dict, inhibitory_trans,inhibitory_series, Glut, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base,duration_mvt, lim_n_cycle,find_stable_oscill)

pkl_file = open(filename, 'rb')
freq = pickle.load(pkl_file)
pkl_file.close()
name = 'Proto' ; color = 'trans_n_half_cycle'
x = freq['synaptic_time_constant'][:,:,0]
y = freq['synaptic_time_constant'][:,:,1]
z = freq[(name,'trans_mvt_freq')]
z_stable = freq[(name, 'stable_mvt_freq')]
c = freq[(name, 'trans_n_half_cycle')]
# scatter_3d_wireframe_plot(x,y,z, c, name,[inhibitory_trans,'Glut','freq',color])
scatter_3d_wireframe_plot_2_data_series(x,y,z,'b','lightskyblue', x,y,z_stable,'g', 'darkgreen',name, ['transient', 'stable'],[inhibitory_trans,'Glut','freq'] )
#%% Scribble


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
                
#def check_freq_detection(nucleus, t_list, dt)
tt = t_list[duration_mvt[0]:duration_mvt[1]]*dt/1000
#N = 1000; T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N, endpoint=False)
#sig1  = np.sin(10.0 * 2.0*np.pi*tt)*np.exp(-10*tt)+100
#indices = np.nonzero((sig1[1:] >= 0) & (sig1[:-1] < 0))[0] # zero crossing
sig2 = STN.pop_act[duration_mvt[0]:duration_mvt[1]]
sig2 = STN.pop_act[np.argmax(STN.pop_act[int(t_mvt/dt):int((t_mvt+D_mvt)/dt)])+int(t_mvt/dt):duration_mvt[1]]
sig2 = STN.pop_act[np.argmax(STN.pop_act[0:int(t_mvt/dt)]):int(t_mvt/dt)]
plt.figure()
plt.plot(sig2)
plt.plot(sig2-np.average(sig2))
plt.plot(sig2[cut_plateau(sig2-np.average(sig2))])
max_non_empty_array(cut_plateau(sig2-np.average(sig2)))/len(sig2)
#derivative = np.diff(sig2-np.average(sig2))
#derivative_2 = np.diff(derivative)
#der_avg = moving_average_array(derivative, 10)
#plt.plot(derivative)
#plt.plot(np.diff(derivative))

#peaks, vallies = signal.find_peaks(derivative, prominence = (0.2))

np.max(cut_plateau(sig2-np.average(sig2)))

#plt.plot(sig1[cut_plateau(sig1)])
#plt.plot(sig_filtered)
#fs = 1/(dt/1000)
#[ff,pxx] = signal.welch(sig2,axis=0,nperseg=int(fs),fs=fs)#,nfft=1024)
#plt.plot(ff,pxx)
#ff[np.argmax(pxx)]

freq_from_fft((sig2-np.average(sig2))[[cut_plateau(sig2)]],dt/1000)
#freq_from_welch((sig2[10:]-np.average(sig2[10:]))[[cut_plateau(sig2[10:])]],dt/1000)
#freq_from_fft(sig2[cut_plateau(sig2)],dt)
#freq_from_welch(sig2[cut_plateau(sig2)],dt)
#f1 = fft(sig1)
#f2 = fft(sig2)
#plt.figure(2)
#windowed1 = sig1 * signal.blackmanharris(len(sig1))
#windowed2 = sig2 * signal.blackmanharris(len(sig2))
#plt.plot(f1)
#plt.plot(windowed2)
#
## Find the peak and interpolate to get a more accurate peak
#np.argmax(abs(f1))
#np.argmax(abs(f2))
    


#%% Skewed normal and beta dist fitting 
plt.figure()
a =- 5
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')
x = np.linspace(skewnorm.ppf(0.001, a, loc= 20, scale = 10),
                skewnorm.ppf(0.999, a, loc= 20, scale = 10), 100)
rv = skewnorm(a, loc = 20, scale = 10)
plt.plot(x, rv.pdf(x), lw=2, label='frozen pdf')
plt.hist(skewnorm.rvs(a, loc = -65, scale = 90, size=1000))


a = nuc3[0].all_mem_pot.copy()
# y_ = nuc3[0].all_mem_pot[:,400].copy()
y_ = a.reshape(int(a.shape[0]* a.shape[1]), 1)
y = y_[np.logical_and(y_ < 25 , y_ > -65)]
param = stats.beta.fit(y)#, floc=-65, fscale=90)
plt.figure()
plt.hist(y, bins = 100, label = 'whole data')
plt.hist(stats.beta.rvs(*param, size = len(y)), bins = 100, label = 'fitted beta distribution')
plt.legend()