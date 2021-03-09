
#%% Constants 

if 1:
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
    file_dir = os.path.dirname(__file__)
    sys.path.append(file_dir)
    from Oscillation_module import *
    
    #from scipy.ndimage.filters import generic_filter
    
    N_sim = 100
    population_list = ['STN', 'Proto']
    N_sub_pop = 2
    N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
    # MSNs make up at least 95% of all striatal cells (Kemp and Powell, 1971)
    N_Str = 2.79*10**6 # Oorschot 1998
    N_real = { 'STN': 13560 , 'Proto': 46000*0.70, 'Arky':46000*0.25, 'GPi': 3200, 'Str': N_Str, 'D2': int(0.475*N_Str), 'D1': int(0.475*N_Str) , 'FSI': int(0.02*N_Str),  # Oorschot 1998 , FSI-MSN: (Gerfen et al., 2010; Tepper, 2010)
              'Th': 10000} # check to find 
    # A = { 'STN': 15 , 'Proto': 30, 'Arky': 18, # De la Crompe (2020) # Why??
    A = { 'STN': 15 , 'Proto': 45, 'Arky': 8, # De la Crompe (2020)
         'FSI': 12.5, # FSI average firing rates:10–15 Hz. 60–80 Hz during behavioral tasks(Berke et al., 2004; Berke, 2008) or 18.5 Hz Berke et al 2010?
         'D1': 1.1, 'D2': 1.1, #Berke et al. 2010
         'GPi':26} # Benhamou & Cohen (201)
    # mean firing rate from experiments
    A_DD = { 'STN': 0 , 'Proto': 0,
         'FSI': 0, # FSI average firing rates:10–15 Hz. 60–80 Hz during behavioral tasks(Berke et al., 2004; Berke, 2008) or 18.5 Hz Berke et al 2010?
         'D1': 6.6, 'D2': 6.6, # Kita & Kita. 2011
         'GPi':0} 
    A_mvt = { 'STN': 50 , 'Proto': 22, 'FSI': 70, 'D2':10, 
             'Arky':15} # Mallet et al. 2016 mean firing rate during movement from experiments
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
               ('D1', 'FSI'): int(36*2/(36+53)*240),#Guzman et al  (2003): 240 from one class interneuron to each MSI # 36% (FSI-D1) Gittis et al.2010
               ('D2', 'FSI'): int(53*2/(36+53)*240),#Guzman et al  (2003): 240 from one class interneuron to each MSI # 53% (FSI-D2) Gittis et al.2010
               ('FSI','FSI'): int(N_real['FSI']*0.58), # Gittis et al. (2010)
    #           ('Proto', 'D2'): int(N_Str/2*(1-np.power(0.13,1/(.1*N_Str/2)))), # Chuhma et al. 2011 --> 10% MSN activation leads to 87% proto activation
               ('Proto', 'D2'): int(N_real['D2']*226/N_real['Proto']), # each Proto 226 from iSPN Kawaguchi et al. (1990)
               ('Proto', 'D1'): int(N_real['D1']*123/N_real['Proto']), # each Proto 123 from dSPN Kawaguchi et al. (1990)
               ('GPi', 'D1'): 1, # find !!!!
               ('GPi', 'STN'): 457, # boutons Kita & Jaeger (2016) based on Koshimizu et al. (2013)
               ('GPi', 'Proto'): 1, # find !!!
               ('D1', 'D2'): int(Str_connec[('MSN','MSN')]*Str_connec[('D1', 'D2')]/(Str_connec[('D1', 'D2')]+Str_connec[('D2', 'D2')])), #Guzman et al (2003) based on Taverna et al (2008)
               ('D2', 'D2'): int(Str_connec[('MSN','MSN')]*Str_connec[('D2', 'D2')]/(Str_connec[('D1', 'D2')]+Str_connec[('D2', 'D2')])),
               ('D2', 'Th'): 1, # find
               ('FSI', 'Th'): 1, # find
               ('D2','Arky'):100,# estimate
               ('Arky','Proto'):300} # estimate
    #           ('D1', 'D1'): Str_connec[('MSN','MSN')]*Str_connec[('D1', 'D1')]/(Str_connec[('D1', 'D1')]+Str_connec[('D2', 'D1')]),  
    #           ('D2', 'D1'): Str_connec[('MSN','MSN')]*Str_connec[('D2', 'D1')]/(Str_connec[('D1', 'D1')]+Str_connec[('D2', 'D1')]), #Guzman et al (2003) based on Taverna et al (2008)
    #           ('D1', 'Proto'): int(N_real['Proto']*(1-np.power(64/81, 1/N_real['Proto'])))} # Klug et al 2018
    
    K_real_DD = {('STN', 'Proto'):0, #883, # Baufreton et al. (2009)
               ('Proto', 'STN'): 0,#190, # Kita, H., and Jaeger, D. (2016)
               ('Proto', 'Proto'): 0,#650, # Hegeman et al. (2017)
    #           ('FSI', 'Proto'): int(800*15*N_real['Proto']/((0.04+0.457)*N_Str)*(0.75/(.75+.10+.54))), #800 boutons per Proto Guzman 2003, 15 contacts per bouton. 75%, 10%, 45%  connected to FSI, D1, NYP
               ('FSI', 'Proto'): 0,#360, # averaging the FSI contacting of Proto boutons Bevan 1998
               ('D1', 'FSI'): int(36*2/(36+53)*240),#Guzman et al  (2003): 240 from one class interneuron to each MSI # 36% (FSI-D1) Gittis et al.2010
               ('D2', 'FSI'): 2*K_real[('D2', 'FSI')],
               ('FSI','FSI'): int(N_real['FSI']*0.58), # Gittis et al. (2010)
               ('Proto', 'D2'): int(N_Str/2*(1-np.power(0.13,1/(.1*N_Str/2)))), # Chuhma et al. 2011 --> 10% MSN activation leads to 87% proto activation
               ('GPi', 'D1'): 0,
               ('GPi', 'STN'): 0,
               ('GPi', 'Proto'): 0,
               ('D1', 'D2'): 0.7*K_real[('D1', 'D2')], #Damodaran et al 2015 based on Taverna et al. 2008
               ('D2', 'D2'): 0.5*K_real[('D2', 'D2')]} #Damodaran et al 2015 based on Taverna et al. 2008
    
    
    K_real_STN_Proto_diverse = K_real.copy()
    K_real_STN_Proto_diverse[('Proto', 'STN')] = K_real_STN_Proto_diverse[('Proto', 'STN')] / N_sub_pop # because one subpop in STN contacts all subpop in Proto
    T = { ('STN', 'Proto'): 4, # Fujimoto & Kita (1993) - [firing rate]
          ('Proto', 'STN'): 2, # kita & Kitai (1991) - [firing rate]
          ('Proto', 'Proto'): 5,#  Ketzef & Silberberg (2020)- [IPSP]/ or 0.96 ms Bugaysen et al. 2013 [IPSP]?
          ('Arky', 'Proto'): 5,#  Ketzef & Silberberg (2020)- [IPSP]
          ('D2','Arky') : 7, # estimate
          ('D2','Arky') : 7, #estimate
          ('Proto', 'D2'):  7.34, #ms proto Ketzef & Silberberg (2020) {in-vitro:striatal photostimulation recording at Proto}- [IPSP] /7ms Kita & Kitai (1991) - [IPSP] [Kita and Kitai 1991 5ms?]
          ('STN', 'Ctx'): 5.5, # kita & Kita (2011) [firing rate]/ Fujimoto & Kita 1993 say an early excitaion of 2.5
    #      ('D2', 'Ctx'): 13.4 - 5, # short inhibition latency of MC--> Proto Kita & Kita (2011) - D2-Proto of Kita & Kitai (1991)
          ('D2', 'Ctx'): 10.5, # excitation of MC--> Str Kita & Kita (2011) - [firing rate]
          ('D1', 'Ctx'): 10.5,
          ('FSI', 'Ctx'): 8/12.5 * 10.5 ,# Kita & Kita (2011) x FSI/MSN latency in SW- Mallet et al. 2005
          ('GPi', 'D1'): 7.2, #  Kita et al. 2001 - [IPSP] / 13.5 (MC-GPi) early inhibition - 10.5 = 3? Kita et al. 2011 
          ('GPi', 'STN'): 1.7, #  STN-EP Nakanishi et al. 1991 [EPSP] /1ms # STN-SNr Nakanishi et al 1987 / 6 - 5.5  (early excitaion latency of MC--> GPi Kita & Kita (2011) - Ctx-STN) - [firing rate]
          ('GPi', 'Proto'): 2.8, # Kita et al 2001 --> short latency of 2.8 and long latency 5.9 ms [IPSP]/ (4 - 2) ms Nakanishi et al. 1991: the IPSP following the EPSP with STN activation in EP, supposedly being due to STN-Proto-GPi circuit?
          ('Th', 'GPi'): 5, # estimate 
          ('FSI', 'Proto'): 6, #estimate
          ('D1' , 'FSI'): 1, #0.84 ms mice Gittis et al 2010
          ('D2' , 'FSI'): 1, #0.93 ms mice Gittis et al 2010
          ('FSI' , 'FSI'): 1, # estimate based on proximity
    #      ('D2', 'D1'): 1,
    #      ('D1', 'D1'): 1,
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
         ('D2', 'Arky'): 0.1,
         ('D2', 'Ctx'): 0,
         ('D1', 'Ctx'): 0,
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
    synaptic_time_constant = {('STN', 'Proto'): [10] ,
                            ('Proto', 'STN'): [decay_time_scale['Glut']], 
                            ('Proto', 'Proto'): [10],
                            ('D2', 'FSI'): [30], 
                            ('FSI', 'Proto'): [6],
                            ('FSI', 'FSI'): [6],
                            ('Proto','D2'): [10],
                            ('Arky','Proto'): [6],
                            ('D2', 'Arky'): [30]}
    G[('D1', 'D1')] = 0.5* G[('D2', 'D2')]
    G_DD = {('STN', 'Proto'): -3 ,
          ('Proto', 'STN'): 0.8 , 
          ('Proto', 'Proto'): 0, # become stronger (Bugaysen et al., 2013) 
          ('Str', 'Ctx'): 0,
          ('D2','Proto'): G[('D2','Proto')]*108/28} # IPSP amplitude in Ctr: 28pA, in DD: 108pA Corbit et al. (2016) [Is it due to increased connections or increased synaptic gain?]
    G_DD[('Proto', 'Proto')] = 0.5* G_DD[('STN', 'Proto')]
    color_dict = {'Proto' : 'r', 'STN': 'k', 'D2': 'b', 'FSI': 'g','Arky':'darkorange'}
    noise_variance = {'Proto' : 0.1, 'STN': 0.1, 'D2': 0.1, 'FSI': 0.1, 'Arky':0.1}
    noise_amplitude = {'Proto' : 10, 'STN': 10, 'D2': 10, 'FSI': 10, 'Arky': 10}
    oscil_peak_threshold = {'Proto' : 0.1, 'STN': 0.1, 'D2': 0.1, 'FSI': 0.1, 'Arky': 0.1}
    smooth_kern_window = {key: value * 30 for key, value in noise_variance.items()}
    #oscil_peak_threshold = {key: (gain[key]*noise_amplitude[key]*noise_variance[key]-threshold[key])/5 for key in noise_variance.keys()}
    pert_val = 10
    mvt_selective_ext_input_dict = {('Proto','1') : pert_val, ('Proto','2') : -pert_val,
                                    ('STN','1') : pert_val, ('STN','2') : -pert_val} # external input coming from Ctx and Str
    dopamine_percentage = 100
    t_sim = 400 # simulation time in ms
    dt = 0.5 # euler time step in ms
    t_mvt = int(t_sim/2)
    D_mvt = t_sim - t_mvt
    D_perturb = 250 # transient selective perturbation
    d_Str = 200 # duration of external input to Str
    t_list = np.arange(int(t_sim/dt))
    #    duration_mvt = [int((t_mvt+ max(T[('Proto', 'D2')],T[('STN', 'Ctx')]))/dt), int((t_mvt+D_mvt)/dt)]
    #    duration_base = [int((max(T[('Proto', 'STN')],T[('STN', 'Proto')]))/dt), int(t_mvt/dt)]
    duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)]
    duration_base = [0, int(t_mvt/dt)]
    #ext_inp_delay = {'Proto': T[('Proto', 'D2')], 'STN': T[('STN', 'Ctx')]}
    ext_inp_delay = 0
#%%
#%% Arky-Proto-D2 loop without Proto-Proto
g = -2.7
t_sim = 1700; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
G[('D2', 'Arky')], G[('Arky', 'Proto')], G[('Proto', 'D2')] = g*0.2, g, g*0.5
receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('Arky','1')]}
synaptic_time_constant[('D2', 'Arky')], synaptic_time_constant[('Arky', 'Proto')],synaptic_time_constant[('Proto', 'D2')]  =  [30],[6],[10]
color_dict = {'Proto' : 'r', 'STN': 'k', 'D2': 'b', 'FSI': 'g','Arky':'k'}
pop_list = [1]  ; lim_n_cycle = [6,10]
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
Arky = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

nuclei_dict = {'Proto': Proto, 'D2' : D2, 'Arky':Arky}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

run(receiving_class_dict,t_list, dt, nuclei_dict)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, plot_start=1000,title_fontsize=15,
     title = r"$G_{AD}="+str(round(G[('D2', 'Arky')],2))+"$ "+", $G_{PA}="+str(round(G[('Arky', 'Proto')],2))+"$"+", $G_{DP}="+str(round(G[('Proto', 'D2')],2))+"$")
name = 'Arky'
nucleus = nuclei_dict[name][0]
find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
# temp_oscil_check(nuclei_dict[name][0].pop_act,oscil_peak_threshold[name], 3,dt,*duration_base)
figname = 'Arky-Proto-D2 loop without Proto-Proto'
fig.savefig(figname+'.png',dpi = 300)
fig.savefig(figname+'.pdf',dpi = 300)
#%%
#%% Arky-Proto-D2 loop without Proto-Proto sweep
n = 50 ; if_plot = False
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
g_1_list = [1] #np.linspace(-2, 0, n, endpoint = True)
g_2_list = np.linspace(-6, -1, n, endpoint = True)
receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('Arky','1')]}
synaptic_time_constant[('D2', 'Arky')], synaptic_time_constant[('Arky', 'Proto')],synaptic_time_constant[('Proto', 'D2')]  =  [30],[6],[10]

pop_list = [1]; lim_n_cycle = [6,10]
G_ratio_dict = {('D2', 'Arky') : 0.2, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 0.5}
G_dict = {('Proto','Proto'): g_1_list, ('D2', 'Arky') : g_2_list}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
Arky = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

nuclei_dict = {'Proto': Proto, 'D2' : D2, 'Arky':Arky}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
filename = 'data_synaptic_weight_D2-P-A_tau_'+str(synaptic_time_constant[('D2', 'FSI')][0])+'_'+str(synaptic_time_constant[('FSI', 'Proto')][0])+'_'+str(synaptic_time_constant[('Proto', 'D2')][0])+'.pkl'
synaptic_weight_space_exploration(G.copy(),A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, receiving_class_dict,color_dict, G_ratio_dict=G_ratio_dict,if_plot=if_plot)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto'
color = 'n_half_cycles_mvt' #mvt_f'
param = 'mvt_freq'
g_transient = data[(name,'g_transient_boundary')][0] # 
# scatter_3d_wireframe_plot(data['g'][:,:,0],data['g'][:,:,1],data[(name,param)],data[(name,color)], name, ['STN-Proto', 'Proto-Proto', param,  color]) 
scatter_2d_plot(np.squeeze(data['g'][:,:,1]),np.squeeze(data[(name,param)]),np.squeeze(data[(name,color)]), name +' in Arky-D2-P',  [r'$G_{Arky-P}=5\times G_{D2-Arky}=2\times G_{P-D2}$' , param, color] )
plt.axvline(g_transient[1], c = 'k')
# find_oscillation_boundary_Pallidostriatal(g_list,g_loop, g_ratio, nuclei_dict, A, A_mvt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle = [6,10], find_stable_oscill = False)
#%%  
#%% Arky-D2-Proto time scale space

n = 10 ; if_plot = False
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('Arky','1')]}

pop_list = [1]; lim_n_cycle = [6,10]
G_ratio_dict = {('D2', 'Arky') : 0.2, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 0.5}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
Arky = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
syn_decay_dict = {'tau_1': {'tau_ratio':{('D2', 'Arky') : 1, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 1},'tau_list':np.linspace(5,15,n)},
                'tau_2':{'tau_ratio':{('Proto', 'Proto'): 1},'tau_list': [5]}}#np.linspace(5,15,n)}}
# filename = 'data_Arky_D2_Proto_syn_t_scale_G_ratios_'+str(G_ratio_dict[('D2', 'Arky')])+'_'+str(G_ratio_dict[('Arky', 'Proto')])+'_'+str(G_ratio_dict[('Proto', 'D2')])
filename = 'data_Arky_D2_Proto_syn_t_scale_tau_1_1_1'
filename= filename.replace('.','-')+'.pkl'
nuclei_dict = {'Proto': Proto, 'D2' : D2, 'Arky':Arky}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

g_list = np.linspace(-20,-1, 250); 
find_stable_oscill = True # to find stable oscillatory regime

sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G.copy(),A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)

#%%
#%% Arky-Proto-D2 loop with Proto-Proto
g = -2
t_sim = 800; t_list = np.arange(int(t_sim/dt))
t_mvt = 200 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
G[('D2', 'Arky')], G[('Arky', 'Proto')], G[('Proto', 'D2')] = g*0.2, g, g*0.5
G[('Proto', 'Proto')] = G[('Arky', 'Proto')]
receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1'),('Proto','1')],
                    ('D2','1') : [('Arky','1')]}
synaptic_time_constant[('D2', 'Arky')], synaptic_time_constant[('Arky', 'Proto')],synaptic_time_constant[('Proto', 'D2')],synaptic_time_constant[('Proto', 'Proto')]  =  [30],[6],[10],[10]

pop_list = [1]  
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
Arky = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

nuclei_dict = {'Proto': Proto, 'D2' : D2, 'Arky':Arky}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

run(receiving_class_dict,t_list, dt, nuclei_dict)
plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$G_{Proto-Proto}="+str(round(G[('Proto', 'Proto')],2))+"$"+r" $G_{Arky-P}="+str(round(G[('Arky', 'Proto')],2))+"$\n"+r" $=2\times G_{P-D2}=5\times G_{D2-Arky}$")
name = 'Proto'
nucleus = nuclei_dict[name][0]
# find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
temp_oscil_check(nuclei_dict[name][0].pop_act,oscil_peak_threshold[name], 3,dt,*duration_mvt)

#%%
#%% Arky-Proto-D2 loop with Proto-Proto sweep
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
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
Arky = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
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
#%%
#%% Critical g Combine different circuits 3 nuclei
g_cte_ind = [0,0,0]; g_ch_ind = [1,1,1]
nucleus_name_list = ['Arky', 'Proto','D2']
####### with GP-GP
filename_list = 3*['data_synaptic_weight_D2-P-A_with_P_P_tau_-1_30_6_10_10.pkl']
# filename_list = 3*['data_synaptic_weight_D2-P-A_tau_30_6_10.pkl']
title = r'$\tau_{D2-Arky}=30$ $\tau_{Arky-P}=6$ $\tau_{P-D2}=10$ $\tau_{P-P}=10$'
####### without GP-GP
# filename_list = 3*['data_synaptic_weight_Pallidostriatal_tau_30_6_10.pkl']
# title = r'$\tau_{D2-FSI}=30$ $\tau_{FSI-P}=6$ $\tau_{P-D2}=10$'
legend_list = nucleus_name_list
color_list = ['darkorange','r','b']
param_list = 3*['mvt_freq']
color_param_list = 3* ['perc_t_oscil_mvt']
x_label = r'$G_{Arky-P}=5\times G_{D2-Arky}=2\times G_{P-D2}$'
synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list, 
                                             color_list,g_cte_ind,g_ch_ind,param_list,color_param_list,'hot',x_axis = 'g_2',title = title,x_label = x_label)

#%%
#%% Pallidostriatal loop without GP-GP
g = -1.7
t_sim = 1700; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
G[('D2', 'FSI')], G[('FSI', 'Proto')], G[('Proto', 'D2')] = g, g, g*0.5
receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('FSI','1')]}
synaptic_time_constant[('D2', 'FSI')], synaptic_time_constant[('FSI', 'Proto')],synaptic_time_constant[('Proto', 'D2')]  =  [30],[10],[10]
color_dict = {'Proto' : 'r', 'STN': 'k', 'D2': 'b', 'FSI': 'grey','Arky':'k'}

pop_list = [1]  
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
FSI = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

run(receiving_class_dict,t_list, dt, nuclei_dict)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, plot_start=1000,title_fontsize=15,
     title = r"$G_{FD}="+str(round(G[('D2', 'FSI')],2))+"$ "+", $G_{PF}="+str(round(G[('FSI', 'Proto')],2))+"$"+", $G_{DP}="+str(round(G[('Proto', 'D2')],2))+"$")
name = 'FSI'
nucleus = nuclei_dict[name][0]
figname = 'FSI-Proto-D2 loop without Proto-Proto'
fig.savefig(figname+'.png',dpi = 300)
fig.savefig(figname+'.pdf',dpi = 300)
# find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
temp_oscil_check(nuclei_dict[name][0].pop_act,oscil_peak_threshold[name], 3,dt,*duration_base)

#%%
#%% Pallidostriatal without GP-GP sweep
n = 50 ; if_plot = False
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
g_1_list = [1] #np.linspace(-2, 0, n, endpoint = True)
g_2_list = np.linspace(-3.5, -1, n, endpoint = True)
receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('FSI','1')]}
synaptic_time_constant[('D2', 'FSI')], synaptic_time_constant[('FSI', 'Proto')],synaptic_time_constant[('Proto', 'D2')]  =  [30],[6],[10]

pop_list = [1]  
G_ratio_dict = {('D2', 'FSI') : 1, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 0.5}
G_dict = {('Proto','Proto'): g_1_list, ('D2', 'FSI') : g_2_list}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
FSI = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
lim_n_cycle = [6,10]
nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
filename = 'data_synaptic_weight_Pallidostriatal_tau_'+str(synaptic_time_constant[('D2', 'FSI')][0])+'_'+str(synaptic_time_constant[('FSI', 'Proto')][0])+'_'+str(synaptic_time_constant[('Proto', 'D2')][0])+'.pkl'
synaptic_weight_space_exploration(G.copy(),A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, receiving_class_dict,color_dict, G_ratio_dict=G_ratio_dict,if_plot=if_plot)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'Proto'
color = 'n_half_cycles_mvt' #mvt_f'
param = 'mvt_freq'
g_transient = data[(name,'g_transient_boundary')][0] # 
# scatter_3d_wireframe_plot(data['g'][:,:,0],data['g'][:,:,1],data[(name,param)],data[(name,color)], name, ['STN-Proto', 'Proto-Proto', param,  color]) 
scatter_2d_plot(np.squeeze(data['g'][:,:,1]),np.squeeze(data[(name,param)]),np.squeeze(data[(name,color)]), name +' in Pallidostriatal',  [r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$' , param, color] )
plt.axvline(g_transient[1], c = 'k')
# find_oscillation_boundary_Pallidostriatal(g_list,g_loop, g_ratio, nuclei_dict, A, A_mvt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle = [6,10], find_stable_oscill = False)

#%%  
#%% FSI-D2-Proto time scale space

n = 10 ; if_plot = False
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('FSI','1')]}
pop_list = [1]  
G_ratio_dict = {('D2', 'FSI') : 1, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 0.5}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
FSI = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
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

#%% Pallidostriatal loop with GP-GP
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
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
FSI = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)

run(receiving_class_dict,t_list, dt, nuclei_dict)
plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$G_{Proto-Proto}="+str(G[('Proto', 'Proto')])+"$"+ r" $G_{D2-FSI}="+str(G[('D2', 'FSI')])+"$")
name = 'D2'
nucleus = nuclei_dict[name][0]
# find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
temp_oscil_check(nuclei_dict[name][0].pop_act,oscil_peak_threshold[name], 3,dt,*duration_mvt)

#%%
#%% Pallidostriatal with GP-GP sweep
n = 8 ; if_plot = False
t_sim = 1500; t_list = np.arange(int(t_sim/dt))
t_mvt = 700 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
g_1_list = np.linspace(-6, -1, n, endpoint = True)
g_2_list = np.linspace(-6, -1, n, endpoint = True)
receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1'),('Proto','1')],
                    ('D2','1') : [('FSI','1')]}
synaptic_time_constant[('D2', 'FSI')], synaptic_time_constant[('FSI', 'Proto')],synaptic_time_constant[('Proto', 'D2')],synaptic_time_constant[('Proto', 'Proto')]  =  [30],[6],[10],[10]

pop_list = [1]  
G_ratio_dict = {('D2', 'FSI') : 1, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 0.5}
G_dict = {('Proto','Proto'): g_1_list, ('D2', 'FSI') : g_2_list}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
FSI = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
lim_n_cycle = [6,10]
nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
filename = 'data_synaptic_weight_Pallidostriatal_with_P_P_tau_'+str(g_1_list[0])+'_tau'+str(synaptic_time_constant[('D2', 'FSI')][0])+'_'+str(synaptic_time_constant[('FSI', 'Proto')][0])+'_'+str(synaptic_time_constant[('Proto', 'D2')][0])+'_'+str(synaptic_time_constant[('Proto', 'Proto')][0])+'.pkl'
synaptic_weight_space_exploration(G.copy(),A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, receiving_class_dict,color_dict, G_ratio_dict=G_ratio_dict,if_plot=if_plot)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'D2'
color = 'n_half_cycles_mvt' #mvt_f'
param = 'mvt_freq'
# g_transient = data[(name,'g_transient_boundary')][0] # 
scatter_3d_wireframe_plot(data['g'][:,:,0],data['g'][:,:,1],data[(name,param)],data[(name,color)], name, [r'$G_{Proto-Proto}$', r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$', param,  color]) 
# scatter_2d_plot(np.squeeze(data['g'][:,:,1]),np.squeeze(data[(name,param)]),np.squeeze(data[(name,color)]), name +' in Pallidostriatal',  [r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$' , param, color] )
# plt.axvline(g_transient[1], c = 'k')

#%%
#%% Critical g Combine different circuits 3 sets of syn time constants
g_cte_ind = [0,0,0]; g_ch_ind = [1,1,1]
filename_list = ['data_synaptic_weight_Pallidostriatal.pkl','data_synaptic_weight_Pallidostriatal_30_10_10.pkl', 'data_synaptic_weight_Pallidostriatal_30_6_6.pkl']
nucleus_name_list = ['Proto', 'Proto','Proto']
legend_list = [r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=10$ $\tau_{Proto-D2}=6$', r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=10$ $\tau_{Proto-D2}=10$', r'$\tau_{D2-FSI}=30$ $\tau_{FSI-Proto}=6$ $\tau_{Proto-D2}=6$']
color_list = ['k','r','g']
param_list = 3*['mvt_freq']
color_param_list = 3* ['perc_t_oscil_mvt']
x_label = r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$'
synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list, 
                                             color_list,g_cte_ind,g_ch_ind,param_list,color_param_list,'jet',x_axis = 'g_2',x_label = x_label)

#%%
#%% Critical g Combine different circuits 3 nuclei
g_cte_ind = [0,0,0]; g_ch_ind = [1,1,1]
nucleus_name_list = ['FSI', 'Proto','D2']
####### with GP-GP
filename_list = 3*['data_synaptic_weight_Pallidostriatal_with_P_P_tau_30_6_10_10.pkl']
title = r'$\tau_{D2-FSI}=30$ $\tau_{FSI-P}=6$ $\tau_{P-D2}=10$ $\tau_{P-P}=10$'
####### without GP-GP
# filename_list = 3*['data_synaptic_weight_Pallidostriatal_tau_30_6_10.pkl']
# title = r'$\tau_{D2-FSI}=30$ $\tau_{FSI-P}=6$ $\tau_{P-D2}=10$'
legend_list = nucleus_name_list
color_list = ['g','r','b']
param_list = 3*['mvt_freq']
color_param_list = 3* ['perc_t_oscil_mvt']
x_label = r'$G_{D2-FSI}=G_{FSI-P}=\frac{G_{P-D2}}{2}$'
synaptic_weight_transition_multiple_circuits(filename_list, nucleus_name_list, legend_list, 
                                             color_list,g_cte_ind,g_ch_ind,param_list,color_param_list,'hot',x_axis = 'g_2',title = title,x_label = x_label)

#%%
#%% STN-Proto network
G = { ('STN', 'Proto'): -1.78,
  ('Proto', 'STN'): 1, 
  ('Proto', 'Proto'): 0 } # synaptic weight
G[('Proto', 'Proto')] = G[('STN', 'Proto')]
receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
pop_list = [1,2]

# receiving_pop_list = {('STN','1') : [('Proto', '1')],('Proto', '1'):[('STN','1')] }
# pop_list = [1]  
  
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'STN', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
nuclei_dict = {'Proto': Proto, 'STN' : STN}


receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real_STN_Proto_diverse, receiving_pop_list, nuclei_dict,t_list)
run(receiving_class_dict,t_list, dt, nuclei_dict)
fig = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, plot_start=1000,title_fontsize=15,
     title = r"$G_{SP}="+str(round(G[('Proto', 'STN')],2))+"$ "+", $G_{PS}=G_{PP}="+str(round(G[('STN', 'Proto')],2))+'$')
#g_list = np.linspace(-.6,-0.1, 20)
# n_half_cycle, G, nuclei_dict = find_oscillation_boundary_STN_GPe(g_list,nuclei_dict, A, A_mvt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base)
figname = 'STN-GPe loop with Proto-Proto stable'
fig.savefig(figname+'.png',dpi = 300)
fig.savefig(figname+'.pdf',dpi = 300)
#print(find_freq_of_pop_act_spec_window(STN[0],*duration_mvt))
temp_oscil_check(nuclei_dict['STN'][0].pop_act,oscil_peak_threshold['STN'], 3,dt,*duration_mvt)
# temp_oscil_check(nuclei_dict['STN'][0].pop_act,oscil_peak_threshold['STN'], 3,dt,*duration_base)
# temp_oscil_check(nuclei_dict['Proto'][0].pop_act,oscil_peak_threshold['Proto'], 3,dt,*duration_mvt)
#plt.title(r"$\tau_{GABA_A}$ = "+ str(round(x[n_plot],2))+r' $\tau_{GABA_B}$ ='+str(round(y[n_plot],2))+ r' $\tau_{Glut}$ ='+str(round(z[n_plot],2))+' f ='+str(round(c[n_plot],2)) , fontsize = 10)
#%%

#%% synaptic weight phase exploration only GP
# T[('Proto', 'Proto')]= 2
T[('Proto', 'Proto')]= 5
filename = 'data_synaptic_weight_GP_only_T_'+str(T[('Proto', 'Proto')])+'.pkl'
n = 50; if_plot = False
g_1_list = [1] #np.linspace(-2, 0, n, endpoint = True)
g_2_list = np.linspace(-6, 0, n, endpoint = True)
# synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')],synaptic_time_constant[('Proto', 'STN')]  =  [decay_time_scale['GABA-A']],[decay_time_scale['GABA-A']],[decay_time_scale['Glut']]
lim_n_cycle = [6,10]
receiving_pop_list = {('Proto','1') : [('Proto', '1')]}
pop_list = [1]  
G_dict = {('STN', 'Proto') : g_1_list, ('Proto', 'Proto') : g_2_list}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
# STN = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'STN', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
nuclei_dict = {'Proto': Proto}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
synaptic_weight_space_exploration(G.copy(),A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, receiving_class_dict,color_dict, if_plot)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
name = 'Proto'
color = 'perc_t_oscil_mvt' #mvt_f'
param = 'mvt_freq'
g_transient = data[name,'g_transient_boundary'][0]
# scatter_3d_wireframe_plot(data['g'][:,:,0],data['g'][:,:,1],data[(name,param)],data[(name,color)], name, ['STN-Proto', 'Proto-Proto', param,  color])
scatter_2d_plot(np.squeeze(data['g'][:,:,1]),np.squeeze(data[(name,param)]),np.squeeze(data[(name,color)]), 'only '+name,  ['G(Proto-Proto)', param, color] )
plt.axvline(g_transient[1], c = 'k')
#%%
#%% synaptic weight phase exploration only STN

n = 50; if_plot = False
g_1_list = [0.5] #np.linspace(-2, 0, n, endpoint = True)
g_2_list = np.linspace(-10, 0, n, endpoint = True)
# synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')],synaptic_time_constant[('Proto', 'STN')]  =  [decay_time_scale['GABA-A']],[decay_time_scale['GABA-A']],[decay_time_scale['Glut']]

lim_n_cycle = [6,10]
receiving_pop_list = {('STN','1') : [('Proto', '1')],('Proto', '1'):[('STN','1')] }
pop_list = [1]  
G_dict = {('Proto', 'STN') : g_1_list, ('STN', 'Proto') : g_2_list}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'STN', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
nuclei_dict = {'Proto': Proto, 'STN':STN}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
filename = 'data_synaptic_weight_STN_GP_only_STN.pkl'
synaptic_weight_space_exploration(G.copy(),A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, receiving_class_dict,color_dict)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

name = 'STN'
color = 'perc_t_oscil_mvt' #mvt_f'
param = 'mvt_freq'
g_transient = data[(name,'g_transient_boundary')][0]
# scatter_3d_wireframe_plot(data['g'][:,:,0],data['g'][:,:,1],data[(name,param)],data[(name,color)], name, ['STN-Proto', 'Proto-Proto', param,  color])
scatter_2d_plot(np.squeeze(data['g'][:,:,1]),np.squeeze(data[(name,param)]),np.squeeze(data[(name,color)]), name +' in STN-GP circuit G(Proto-STN) =' +str(G[('Proto','STN')]),  ['G(Proto-Proto)', param, color] )
plt.axvline(g_transient[1], c = 'k')
#%%
#%% Critical g Combine different circuits
g_cte_ind = [0,0]; g_ch_ind = [1,1]
fig = synaptic_weight_transition_multiple_circuits(['data_synaptic_weight_STN_GP_only_STN.pkl','data_synaptic_weight_GP_only_T_5.pkl'], 
                                                  ['STN', 'Proto'], ['STN-GPe', 'GPe-GPe'], ['k','r'],g_cte_ind,g_ch_ind,2*['mvt_freq'],2* ['perc_t_oscil_mvt'],'jet',x_axis='g_2')
# fig = synaptic_weight_transition_multiple_circuits(['data_synaptic_weight_GP_only_T_2.pkl','data_synaptic_weight_GP_only_T_5.pkl'], 
#                                                  ['Proto', 'Proto'], ['GP-GP (2ms)', 'GP-GP (5ms)'], ['k','r'],g_cte_ind,g_ch_ind,2*['mvt_freq'],2* ['perc_t_oscil_mvt'],'jet',x_axis='g_2')
fig.savefig('STN_GPe_synaptic_weight.png',dpi = 300)
fig.savefig('STN_GPe_synaptic_weight.pdf',dpi = 300)
#%%
#%% time scale space (GABA-a, GABA-b)

receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
#receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
#                    ('Proto','1') : [('Proto', '1'), ('STN', '1')],
#                    ('Proto','2') : [('Proto', '2'), ('STN', '2')]}
synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')],synaptic_time_constant[('Proto', 'STN')]  =  [decay_time_scale['GABA-A'],decay_time_scale['GABA-B']],[decay_time_scale['GABA-A'],decay_time_scale['GABA-B']],[decay_time_scale['Glut']]

pop_list = [1,2]  
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'STN', G, T, t_sim, dt, synaptic_time_constant,  receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
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
#%%
#%% STN-GPe time scale space GABA and glut

receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')],synaptic_time_constant[('Proto', 'STN')]  =  [10],[10],[decay_time_scale['Glut']]

pop_list = [1,2]  
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'STN', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
nuclei_dict = {'Proto': Proto, 'STN' : STN}

receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real_STN_Proto_diverse, receiving_pop_list, nuclei_dict,t_list)

n = 10;
syn_decay_dict = {'tau_1': {'tau_ratio':{('Proto', 'Proto'): 1, ('STN', 'Proto'): 1},'tau_list':np.linspace(5,30,n)},
                'tau_2':{'tau_ratio':{('Proto', 'STN'): 1},'tau_list':np.linspace(1,15,n)}}
G_ratio_dict = {('Proto', 'Proto'): 1, ('STN', 'Proto'): 1}
g_list = np.linspace(-20,-0.01, 150); g_ratio = 2
lim_n_cycle = [6,10] ; find_stable_oscill = True # to find stable oscillatory regime
filename = 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl'
# syn_t_specif_dict = {('Proto', 'Proto'): 1, ('STN', 'Proto'): 1}
sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)

pkl_file = open(filename, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
name = 'STN' ; color = 'trans_n_half_cycle'
color = 'freq'
g_transient = data[(name,'g_transient')]
g_transient = data[(name,'g_stable')]
x = data['tau'][:,:,0]
y = data['tau'][:,:,1]
z_transient = data[(name,'trans_mvt_freq')]
z_stable = data[(name, 'stable_mvt_freq')]
c = data[(name, 'trans_n_half_cycle')]
fig,ax = scatter_3d_wireframe_plot(x,y,z_stable, z_stable, name +' in STN-GPe circuit',[r'$\tau_{decay}^{inhibition}(ms)$',r'$\tau_{decay}^{excitaion}(ms)$','frequency(Hz)','frequency(Hz)'])
# scatter_3d_wireframe_plot(x,y,z_transient, c, name,[r'$\tau_{decay}^{inhibition}$',r'$\tau_{decay}^{excitaion}$','frequency(Hz)',color])
param = 'stable_mvt_freq'
x_spec =  data['tau'][:,:,0][:,0]
y_spec = data[(name, 'stable_mvt_freq')][:,3]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_spec,np.ones_like(x_spec)*y[0,3],y_spec, c = ['k']*len(y_spec),s = 80)
# scatter_2d_plot(x_spec,y_spec,y_spec, name +' in STN-GP circuit',  [r'$\tau_{decay}^{inhibition}(ms)$', 'Frequency(Hz)', 'Frequency(Hz)'] )
# plt.axvline(g_transient[1], c = 'k')
fig.savefig('STN_GPe_timescale_inh_excit_3d.png',dpi = 300)
fig.savefig('STN_GPe_timescale_inh_excit_3d.pdf',dpi = 300)

#%%
#%% Plot different G ratios for timescale plot of STN-GPe
# filename_list = ['data_FSI_D2_Proto_syn_t_scale_G_ratios_1_1_0-5.pkl','data_STN_GPe_syn_t_scale_g_ratio_1.pkl','data_Arky_D2_Proto_syn_t_scale_G_ratios_0-2_1_0-5.pkl']
# figname = 'All_circuits_timescale'
# label_list = ['FSI-D2-Proto','Arky-D2-Proto','STN-Proto+Proto-Proto']

# filename_list = ['data_FSI_D2_Proto_syn_t_scale_tau_3_1_1.pkl','data_FSI_D2_Proto_syn_t_scale_tau_2_1_1.pkl','data_FSI_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
# label_list = [r'$\tau_{PF}=\tau_{DP}=\dfrac{\tau_{FD}}{3}$',r'$\tau_{PF}=\tau_{DP}=\dfrac{\tau_{FD}}{3}$',r'$\tau_{PF}=\tau_{DP}=\tau_{FD}$']
# figname = 'FSI-D2-Proto_timescale_tau_g_stable'
# x_label = r'$\tau_{FP/DP}^{decay}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label

filename_list = ['data_Arky_D2_Proto_syn_t_scale_tau_3_1_1.pkl','data_Arky_D2_Proto_syn_t_scale_tau_2_1_1.pkl','data_Arky_D2_Proto_syn_t_scale_tau_1_1_1.pkl']
label_list = [r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{3}$',r'$\tau_{PA}=\tau_{DP}=\dfrac{\tau_{AD}}{3}$',r'$\tau_{PA}=\tau_{DP}=\tau_{AD}$']
figname = 'Arky-D2-Proto_timescale_g_stable'
x_label = r'$\tau_{AP/DP}^{decay}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label

# filename_list = ['data_FSI_D2_Proto_syn_t_scale_G_ratios_2_1_0-5.pkl','data_FSI_D2_Proto_syn_t_scale_G_ratios_1_1_0-5.pkl','data_FSI_D2_Proto_syn_t_scale_G_ratios_1_2_0-5.pkl']
# label_list = [r'$G_{PF}=\dfrac{G_{DP}}{2}=\dfrac{G_{FD}}{4}$',r'$G_{PF}=G_{DP}=2\times G_{FD}$',r'$G_{PF}=2\times G_{DP}=\dfrac{G_{FD}}{2}$']
# figname = 'FSI-D2-Proto_timescale'

# filename_list = ['data_Arky_D2_Proto_syn_t_scale_G_ratios_0-1_1_1.pkl','data_Arky_D2_Proto_syn_t_scale_G_ratios_1_1_1.pkl','data_Arky_D2_Proto_syn_t_scale_G_ratios_0-2_2_1.pkl']
# label_list = [r'$G_{PA}=G_{DP}=\dfrac{G_{AD}}{10}$',r'$G_{PA}=G_{DP}=G_{AD}$',r'$G_{PA}=2\times G_{DP}=\dfrac{G_{AD}}{10}$']
# figname = 'Arky-D2-Proto_timescale'

name_list = ['Proto']*3
y_list  = ['g_stable']*3
x_label = r'$\tau_{decay}^{inhibition}(ms)$' ; y_label = r'$G_{PA}$ at transition' ; c_label = y_label
# y_list  = ['stable_mvt_freq']*3
# x_label = r'$\tau_{decay}^{inhibition}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label

title = ''
g_tau_2_ind = 0
g_ratio_list = [1,1,1]

# filename_list = ['data_STN_GPe_syn_t_scale_g_ratio_2.pkl', 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl', 'data_STN_GPe_syn_t_scale_g_ratio_0-5.pkl']
# name_list = ['STN']*3
# y_list  = ['stable_mvt_freq']*3
# figname = 'STN_GPe_timescale'
# x_label = r'$\tau_{decay}^{inhibition}(ms)$' ; y_label = 'frequency(Hz)' ; c_label = y_label
# label_list = [r'$G_{PS}=2\times G_{PP}$',r'$G_{PS}=G_{PP}$',r'$G_{PS}=\dfrac{G_{PP}}{2}$']
# title = ''
# g_ratio_list = [1,1,1]
# g_tau_2_ind = 3

# filename_list = ['data_STN_GPe_syn_t_scale_g_ratio_2.pkl', 'data_STN_GPe_syn_t_scale_g_ratio_1.pkl', 'data_STN_GPe_syn_t_scale_g_ratio_0-5.pkl']
# y_list  = ['g_stable']*3
# figname = 'STN_GPe_g_stable'
# x_label = r'$\tau_{decay}^{inhibition}(ms)$' ; y_label = r'$G_{PS}$ at transition' ; c_label = y_label
# label_list = [r'$G_{PS}=2\times G_{PP}$',r'$G_{PS}=G_{PP}$',r'$G_{PS}=\dfrac{G_{PP}}{2}$']
# title = r'$G_{SP}=0.5$'
# g_ratio_list = [2,1,1]
# g_tau_2_ind = 0
# color_list = ['k','grey','lightgrey']
# c_list = y_list
# colormap = 'hot'


maxs = [] ; mins = []
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
for i in range(len(filename_list)):
    pkl_file = open(filename_list[i], 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    maxs.append(np.max(data[name_list[i],c_list[i]]))
    mins.append(np.min(data[name_list[i],c_list[i]]))
vmax = max(maxs) ; vmin = min(mins)

for i in range(len(filename_list)):
    pkl_file = open(filename_list[i], 'rb')
    data = pickle.load(pkl_file)
    x_spec =  data['tau'][:,:,0][:,0]
    y_spec = data[(name_list[i], y_list[i])][:,g_tau_2_ind]*g_ratio_list[i]
    c_spec = data[(name_list[i], c_list[i])][:,g_tau_2_ind]*g_ratio_list[i]
    ax.plot(x_spec,y_spec,c = color_list[i], lw = 3, label= label_list[i],zorder = 1)
    img = ax.scatter(x_spec,y_spec,vmin = vmin, vmax = vmax, c=c_spec, cmap=colormap,lw = 1,edgecolor = 'k',s =90,zorder = 2)
    # plt.axvline(g_transient, c = color_list[i])  # to get the circuit g which is the muptiplication
    ax.set_xlabel(x_label,fontsize = 20)
    ax.set_ylabel(y_label,fontsize = 20,labelpad=-8)
    ax.set_title(title,fontsize = 20)
    # ax.set_xlim(limits['x'])
    # ax.set_ylim(limits['y'])
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

clb = fig.colorbar(img)
clb.set_label(c_label, labelpad=20, y=.5, rotation=-90,fontsize = 20)
clb.ax.locator_params(nbins=4)

plt.legend(fontsize = 20)
plt.show()
fig.savefig(figname+'.png',dpi = 300)
fig.savefig(figname+'.pdf',dpi = 300)

#%%
#%% time scale space GABA-B

t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = int(t_sim/2); D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
G[('Proto','STN')] = 1
receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}

synaptic_time_constant[('Proto', 'Proto')], synaptic_time_constant[('STN', 'Proto')],synaptic_time_constant[('Proto', 'STN')]  =  [decay_time_scale['GABA-B']],[decay_time_scale['GABA-B']],[decay_time_scale['Glut']]

pop_list = [1,2]
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
STN = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, 'STN', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
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
#%% Scribble
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
    
#%%
    

    

# n =10  ; if_plot = False
# t_sim = 2000; t_list = np.arange(int(t_sim/dt))
# t_mvt = 1000 ; D_mvt = t_sim - t_mvt
# duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
# receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
#                     ('Proto','1') : [('D2', '1')],
#                     ('D2','1') : [('FSI','1')]}
# # synaptic_time_constant[('D2', 'FSI')], synaptic_time_constant[('FSI', 'Proto')],synaptic_time_constant[('Proto', 'D2')]  =  [30],[6],[10]

# pop_list = [1]  
# G_ratio_dict = {('D2', 'FSI') : 2, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 0.5}
# Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
# D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
# FSI = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
# lim_n_cycle = [6,10]
# nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
# syn_decay_dict = {'tau_1': {'tau_ratio':{('D2', 'FSI') : 3, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 1},'tau_list':np.linspace(5,15,n)},
#                 'tau_2':{'tau_ratio':{('Proto', 'Proto'): 1},'tau_list': [5]}}#np.linspace(5,15,n)}}
# g_list = np.linspace(-20,-0.01, 200); 
# lim_n_cycle = [6,10] ; find_stable_oscill = True # to find stable oscillatory regime

# filename = 'data_FSI_D2_Proto_syn_t_scale_G_ratios_'+str(G_ratio_dict[('D2', 'FSI')])+'_'+str(G_ratio_dict[('FSI', 'Proto')])+'_'+str(G_ratio_dict[('Proto', 'D2')])+'.pkl'
# sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)

# n = 10 ; if_plot = False
# t_sim = 2000; t_list = np.arange(int(t_sim/dt))
# t_mvt = 1000 ; D_mvt = t_sim - t_mvt
# duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
# receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
#                     ('Proto','1') : [('D2', '1')],
#                     ('D2','1') : [('FSI','1')]}
# # synaptic_time_constant[('D2', 'FSI')], synaptic_time_constant[('FSI', 'Proto')],synaptic_time_constant[('Proto', 'D2')]  =  [30],[6],[10]

# pop_list = [1]  
# G_ratio_dict = {('D2', 'FSI') : 1, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 0.5}
# Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
# D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
# FSI = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
# lim_n_cycle = [6,10]
# nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
# syn_decay_dict = {'tau_1': {'tau_ratio':{('D2', 'FSI') : 3, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 1},'tau_list':np.linspace(5,15,n)},
#                 'tau_2':{'tau_ratio':{('Proto', 'Proto'): 1},'tau_list': [5]}}#np.linspace(5,15,n)}}
# g_list = np.linspace(-20,-0.01, 150); 
# lim_n_cycle = [6,10] ; find_stable_oscill = True # to find stable oscillatory regime

# filename = 'data_FSI_D2_Proto_syn_t_scale_G_ratios_'+str(G_ratio_dict[('D2', 'FSI')])+'_'+str(G_ratio_dict[('FSI', 'Proto')])+'_'+str(G_ratio_dict[('Proto', 'D2')])+'.pkl'
# sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)

# n = 10 ; if_plot = False
# t_sim = 2000; t_list = np.arange(int(t_sim/dt))
# t_mvt = 1000 ; D_mvt = t_sim - t_mvt
# duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
# receiving_pop_list = {('FSI','1') : [('Proto', '1')], 
#                     ('Proto','1') : [('D2', '1')],
#                     ('D2','1') : [('FSI','1')]}
# # synaptic_time_constant[('D2', 'FSI')], synaptic_time_constant[('FSI', 'Proto')],synaptic_time_constant[('Proto', 'D2')]  =  [30],[6],[10]

# pop_list = [1]  
# G_ratio_dict = {('D2', 'FSI') : 1, ('FSI', 'Proto') : 2, ('Proto', 'D2'): 0.5}
# Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
# D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
# FSI = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'FSI', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
# lim_n_cycle = [6,10]
# nuclei_dict = {'Proto': Proto, 'D2' : D2, 'FSI':FSI}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
# syn_decay_dict = {'tau_1': {'tau_ratio':{('D2', 'FSI') : 3, ('FSI', 'Proto') : 1, ('Proto', 'D2'): 1},'tau_list':np.linspace(5,15,n)},
#                 'tau_2':{'tau_ratio':{('Proto', 'Proto'): 1},'tau_list': [5]}}#np.linspace(5,15,n)}}
# g_list = np.linspace(-20,-1, 250); 
# lim_n_cycle = [6,10] ; find_stable_oscill = True # to find stable oscillatory regime

# filename = 'data_FSI_D2_Proto_syn_t_scale_G_ratios_'+str(G_ratio_dict[('D2', 'FSI')])+'_'+str(G_ratio_dict[('FSI', 'Proto')])+'_'+str(G_ratio_dict[('Proto', 'D2')])+'.pkl'
# sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)
    
    
    

# n = 10 ; if_plot = False
# t_sim = 2000; t_list = np.arange(int(t_sim/dt))
# t_mvt = 1000 ; D_mvt = t_sim - t_mvt
# duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
# receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
#                     ('Proto','1') : [('D2', '1')],
#                     ('D2','1') : [('Arky','1')]}

# pop_list = [1]; lim_n_cycle = [6,10]
# G_ratio_dict = {('D2', 'Arky') : 0.2, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 0.5}
# Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
# D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
# Arky = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

# nuclei_dict = {'Proto': Proto, 'D2' : D2, 'Arky':Arky}
# receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
# filename = 'data_synaptic_weight_D2-P-A_tau_'+str(synaptic_time_constant[('D2', 'FSI')][0])+'_'+str(synaptic_time_constant[('FSI', 'Proto')][0])+'_'+str(synaptic_time_constant[('Proto', 'D2')][0])+'.pkl'
# syn_decay_dict = {'tau_1': {'tau_ratio':{('D2', 'Arky') : 3, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 1},'tau_list':np.linspace(5,15,n)},
#                 'tau_2':{'tau_ratio':{('Proto', 'Proto'): 1},'tau_list': [5]}}#np.linspace(5,15,n)}}
# g_list = np.linspace(-20,-0.01, 150); 
# find_stable_oscill = True # to find stable oscillatory regime

# filename = 'data_Arky_D2_Proto_syn_t_scale_G_ratios_'+str(G_ratio_dict[('D2', 'Arky')])+'_'+str(G_ratio_dict[('Arky', 'Proto')])+'_'+str(G_ratio_dict[('Proto', 'D2')])+'.pkl'
# sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)

n = 10 ; if_plot = False
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('Arky','1')]}

pop_list = [1]; lim_n_cycle = [6,10]
G_ratio_dict = {('D2', 'Arky') : 0.5, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 0.5}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
Arky = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

nuclei_dict = {'Proto': Proto, 'D2' : D2, 'Arky':Arky}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
filename = 'data_synaptic_weight_D2-P-A_tau_'+str(synaptic_time_constant[('D2', 'FSI')][0])+'_'+str(synaptic_time_constant[('FSI', 'Proto')][0])+'_'+str(synaptic_time_constant[('Proto', 'D2')][0])+'.pkl'
syn_decay_dict = {'tau_1': {'tau_ratio':{('D2', 'Arky') : 3, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 1},'tau_list':np.linspace(5,15,n)},
                'tau_2':{'tau_ratio':{('Proto', 'Proto'): 1},'tau_list': [5]}}#np.linspace(5,15,n)}}
g_list = np.linspace(-20,-1, 200); 
find_stable_oscill = True # to find stable oscillatory regime

filename = 'data_Arkt_D2_Proto_syn_t_scale_G_ratios_'+str(G_ratio_dict[('D2', 'Arky')])+'_'+str(G_ratio_dict[('Arky', 'Proto')])+'_'+str(G_ratio_dict[('Proto', 'D2')])+'.pkl'
sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)

n = 10 ; if_plot = False
t_sim = 2000; t_list = np.arange(int(t_sim/dt))
t_mvt = 1000 ; D_mvt = t_sim - t_mvt
duration_mvt = [int((t_mvt)/dt), int((t_mvt+D_mvt)/dt)] ; duration_base = [0, int(t_mvt/dt)]
receiving_pop_list = {('Arky','1') : [('Proto', '1')], 
                    ('Proto','1') : [('D2', '1')],
                    ('D2','1') : [('Arky','1')]}

pop_list = [1]; lim_n_cycle = [6,10]
G_ratio_dict = {('D2', 'Arky') : 0.5, ('Arky', 'Proto') : 1.5, ('Proto', 'D2'): 0.5}
Proto = [Nucleus(i, gain, threshold, ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Proto', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold) for i in pop_list]
D2 = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'D2', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]
Arky = [Nucleus(i, gain, threshold,ext_inp_delay,noise_variance, noise_amplitude, N, A,A_mvt, 'Arky', G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)for i in pop_list]

nuclei_dict = {'Proto': Proto, 'D2' : D2, 'Arky':Arky}
receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt, t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
filename = 'data_synaptic_weight_D2-P-A_tau_'+str(synaptic_time_constant[('D2', 'FSI')][0])+'_'+str(synaptic_time_constant[('FSI', 'Proto')][0])+'_'+str(synaptic_time_constant[('Proto', 'D2')][0])+'.pkl'
syn_decay_dict = {'tau_1': {'tau_ratio':{('D2', 'Arky') : 3, ('Arky', 'Proto') : 1, ('Proto', 'D2'): 1},'tau_list':np.linspace(5,15,n)},
                'tau_2':{'tau_ratio':{('Proto', 'Proto'): 1},'tau_list': [5]}}#np.linspace(5,15,n)}}
g_list = np.linspace(-20,-1, 200); 
find_stable_oscill = True # to find stable oscillatory regime

filename = 'data_Arkt_D2_Proto_syn_t_scale_G_ratios_'+str(G_ratio_dict[('D2', 'Arky')])+'_'+str(G_ratio_dict[('Arky', 'Proto')])+'_'+str(G_ratio_dict[('Proto', 'D2')])+'.pkl'
sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant.copy(), nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill)

    
    