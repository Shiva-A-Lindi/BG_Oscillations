import numpy as np
import matplotlib.pyplot as plt
import timeit
from numpy.fft import rfft,fft, fftfreq
from scipy import signal
from tempfile import TemporaryFile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#from scipy.ndimage.filters import generic_filter

N_sim = 500
population_list = ['STN', 'GPe']
N_sub_pop = 2
N = { 'STN': N_sim , 'GPe': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim}
# MSNs make up at least 95% of all striatal cells (Kemp and Powell, 1971)
N_Str = 2.79*10**6 # Oorschot 1998
N_real = { 'STN': 13560 , 'GPe': 34470, 'GPi': 3200, 'Str': N_Str, 'D2': int(0.475*N_Str/2), 'D1': int(0.475*N_Str) , 'FSI': int(0.02*N_Str)} # Oorschot 1998 , FSI-MSN: (Gerfen et al., 2010; Tepper, 2010)
A = { 'STN': 15 , 'GPe': 30,
     'FSI': 12.5, # FSI average firing rates:10–15 Hz. 60–80 Hz during behavioral tasks(Berke et al., 2004; Berke, 2008) or 18.5 Hz Berke et al 2010?
     'D1': 1.1, 'D2': 1.1, #Berke et al. 2010
     'GPi':26} # Benhamou & Cohen (201)
# mean firing rate from experiments
A_DD = { 'STN': 0 , 'GPe': 0,
     'FSI': 0, # FSI average firing rates:10–15 Hz. 60–80 Hz during behavioral tasks(Berke et al., 2004; Berke, 2008) or 18.5 Hz Berke et al 2010?
     'D1': 6.6, 'D2': 6.6, # Kita & Kita. 2011
     'GPi':0} 
A_mvt = { 'STN': 50 , 'GPe': 22, 'FSI': 70} # mean firing rate during movement from experiments
threshold = { 'STN': .1 ,'GPe': .1}
neuron_type = {'STN': 'Glut', 'GPe': 'GABA'}
gain = { 'STN': 1 ,'GPe': 1}

#K = { ('STN', 'GPe'): 475,
#      ('GPe', 'STN'): 161,
#      ('GPe', 'GPe'): 399}
Str_connec = {('D1', 'D2'): .28,('D2', 'D2'):.36,  ('D1', 'D1'):.26,  ('D2', 'D1'):.05, ('MSN','MSN'): 1350} # Taverna et al 2008
K_real = { ('STN', 'GPe'): 883, # Baufreton et al. (2009)
           ('GPe', 'STN'): 190, # Kita, H., and Jaeger, D. (2016)
           ('GPe', 'GPe'): 650, # Hegeman et al. (2017)
#           ('FSI', 'GPe'): int(800*15*N_real['GPe']/((0.04+0.457)*N_Str)*(0.75/(.75+.10+.54))), #800 boutons per GPe Guzman 2003, 15 contacts per bouton. 75%, 10%, 45%  connected to FSI, D1, NYP
           ('FSI', 'GPe'): 360, # averaging the FSI contacting of GPe boutons Bevan 1998
           ('D1', 'FSI'): int(36*2/(36+53)*240),#Guzman et al  (2003): 240 from one class interneuron to each MSI # 36% (FSI-D1) Gittis et al.2010
           ('D2', 'FSI'): int(53*2/(36+53)*240),#Guzman et al  (2003): 240 from one class interneuron to each MSI # 53% (FSI-D2) Gittis et al.2010
           ('FSI','FSI'): int(N_real['FSI']*0.58), # Gittis et al. (2010)
           ('GPe', 'D2'): int(N_Str/2*(1-np.power(0.13,1/(.1*N_Str/2)))), # Chuhma et al. 2011 --> 10% MSN activation leads to 87% proto activation
           ('D1', 'Ctx'):0,
           ('D2', 'Ctx'):0,
           ('FSI', 'Ctx'):0,
           ('GPi', 'D1'): 0,
           ('GPi', 'STN'): 0,
           ('GPi', 'GPe'): 0,
           ('D1', 'D2'): Str_connec[('MSN','MSN')]*Str_connec[('D1', 'D2')]/(Str_connec[('D1', 'D2')]+Str_connec[('D2', 'D2')]),
           ('D2', 'D2'): Str_connec[('MSN','MSN')]*Str_connec[('D2', 'D2')]/(Str_connec[('D1', 'D2')]+Str_connec[('D2', 'D2')]),
           ('D1', 'D1'): Str_connec[('MSN','MSN')]*Str_connec[('D1', 'D1')]/(Str_connec[('D1', 'D1')]+Str_connec[('D2', 'D1')]),  
           ('D2', 'D1'): Str_connec[('MSN','MSN')]*Str_connec[('D2', 'D1')]/(Str_connec[('D1', 'D1')]+Str_connec[('D2', 'D1')]), #Guzman et al (2003) based on Taverna et al (2008)
           ('D1', 'GPe'): int(N_real['GPe']*(1-np.power(64/81, 1/N_real['GPe'])))} # Klug et al 2018

K_real_DD = {('STN', 'GPe'):0, #883, # Baufreton et al. (2009)
           ('GPe', 'STN'): 0,#190, # Kita, H., and Jaeger, D. (2016)
           ('GPe', 'GPe'): 0,#650, # Hegeman et al. (2017)
#           ('FSI', 'GPe'): int(800*15*N_real['GPe']/((0.04+0.457)*N_Str)*(0.75/(.75+.10+.54))), #800 boutons per GPe Guzman 2003, 15 contacts per bouton. 75%, 10%, 45%  connected to FSI, D1, NYP
           ('FSI', 'GPe'): 0,#360, # averaging the FSI contacting of GPe boutons Bevan 1998
           ('D1', 'FSI'): int(36*2/(36+53)*240),#Guzman et al  (2003): 240 from one class interneuron to each MSI # 36% (FSI-D1) Gittis et al.2010
           ('D2', 'FSI'): 2*K_real[('D2', 'FSI')],
           ('FSI','FSI'): int(N_real['FSI']*0.58), # Gittis et al. (2010)
           ('GPe', 'D2'): int(N_Str/2*(1-np.power(0.13,1/(.1*N_Str/2)))), # Chuhma et al. 2011 --> 10% MSN activation leads to 87% proto activation
           ('D1', 'Ctx'):0,
           ('D2', 'Ctx'):0,
           ('FSI', 'Ctx'):0,
           ('GPi', 'D1'): 0,
           ('GPi', 'STN'): 0,
           ('GPi', 'GPe'): 0,
           ('D1', 'D2'): 0.7*K_real[('D1', 'D2')], #Damodaran et al 2015 based on Taverna et al. 2008
           ('D2', 'D2'): 0.5*K_real[('D2', 'D2')], #Damodaran et al 2015 based on Taverna et al. 2008
           ('D1', 'D1'): 0, #Damodaran et al 2015 based on Taverna et al. 2008
           ('D2', 'D1'): 0,
           ('D1', 'GPe'): int(N_real['GPe']*(1-np.power(64/81, 1/N_real['GPe'])))} # Klug et al 2018

K_real_STN_GPe_diverse = K_real.copy()
K_real_STN_GPe_diverse[('GPe', 'STN')] = K_real_STN_GPe_diverse[('GPe', 'STN')] / N_sub_pop # because one subpop in STN contacts all subpop in GPe
T = { ('STN', 'GPe'): 4, # Fujimoto & Kita (1993) - [firing rate]
      ('GPe', 'STN'): 2, # kita & Kitai (1991) - [firing rate]
      ('GPe', 'GPe'): 5,#  Ketzef & Silberberg (2020)- [IPSP]/ in the begining was 1.5
      ('GPe', 'D2'):  7.34, #ms proto Ketzef & Silberberg (2020) {in-vitro:striatal photostimulation recording at GPe}- [IPSP] /7ms Kita & Kitai (1991) - [IPSP]
      ('STN', 'Ctx'): 5.5, # kita & Kita (2011) [firing rate]/ Fujimoto & Kita 1993 say an early excitaion of 2.5
#      ('D2', 'Ctx'): 13.4 - 5, # short inhibition latency of MC--> GPe Kita & Kita (2011) - D2-GPe of Kita & Kitai (1991)
      ('D2', 'Ctx'): 10.5, # excitation of MC--> Str Kita & Kita (2011) - [firing rate]
      ('D1', 'Ctx'): 10.5,
      ('FSI', 'Ctx'): 8/12.5 * 10.5 ,# Kita & Kita (2011) x FSI/MSN latency in SW- Mallet et al. 2005
      ('GPi', 'D1'): 7.2, #  Kita et al. 2001 - [IPSP]
      ('GPi', 'STN'): 1.7, #  STN-EP Nakanishi et al. 1991 [EPSP] /1ms # STN-SNr Nakanishi et al 1987 / 6 - 5.5  (early excitaion latency of MC--> GPi Kita & Kita (2011) - Ctx-STN) - [firing rate]
      ('GPi', 'GPe'): 2.8, # Kita et al 2001 --> short latency of 2.8 and long latency 5.9 ms [IPSP]/ (4 - 2) ms Nakanishi et al. 1991: the IPSP following the EPSP with STN activation in EP, supposedly being due to STN-GPe-GPi circuit?
      ('Th', 'GPi'): 5, # estimate 
      ('Th', 'Ctx'): 5, # estimate
      ('FSI', 'GPe'): 6, #estimate
      ('D1' , 'FSI'): 0.84, #mice Gittis et al 2010
      ('D2' , 'FSI'): 0.93, # mice Gittis et al 2010
      ('FSI' , 'FSI'): 1, # estimate based on proximity
      ('D2', 'D1'): 1,
      ('D1', 'D1'): 1,
      ('D1', 'D2'): 1,
      ('D2', 'D2'): 1} 
    # transmission delay in ms
T_DD = {('D2', 'Ctx'): 5.5, # excitation of MC--> Str Kita & Kita (2011)  [firing rate]
        ('D1', 'Ctx'): 5.5,
        ('STN', 'Ctx'): 5.9} # kita & Kita (2011) [firing rate]
G = {('STN', 'GPe'): -2 ,
     ('GPe', 'STN'): 0.5 , 
     ('GPe', 'GPe'): 0,
     ('D2', 'Ctx'): 0,
     ('D1', 'Ctx'): 0,
     ('D2','GPe'): 0,
     ('D2', 'FSI'): 0, 
     ('FSI', 'GPe'): 0,
     ('FSI', 'FSI'): 0,
     ('D2','D2'): 0,
     ('D2','D1'): 0,
     ('D1','D2'): 0,
     ('D1', 'D1'): 0,
     ('GPi', 'GPe'): 0,
     ('Th', 'GPi') : 0
     } # synaptic weight
G[('GPe', 'GPe')] = 0.5* G[('STN', 'GPe')]
G[('D1', 'D1')] = 0.5* G[('D2', 'D2')]
G_DD = {('STN', 'GPe'): -2 ,
      ('GPe', 'STN'): 0.5 , 
      ('GPe', 'GPe'): 0,
      ('Str', 'Ctx'): 0,
      ('D2','GPe'): G[('D2','GPe')]*108/28} # IPSP amplitude in Ctr: 28pA, in DD: 108pA Corbit et al. (2016) [Is it due to increased connections or increased synaptic gain?]
G_DD[('GPe', 'GPe')] = 0.5* G_DD[('STN', 'GPe')]

tau = {'GABA-A' : 6, 'GABA-B': 200, 'Glut': 3.5} # Gerstner. synaptic time scale for excitation and inhibition
noise_variance = {'GPe' : 0.1, 'STN': 0.1}
noise_amplitude = {'GPe' : 1, 'STN': 1}
#rest_ext_input = { 'STN': A['STN']/gain['STN']-G[('STN', 'GPe')]*A['GPe'] + threshold['STN'] ,
#                   'GPe': A['GPe']/gain['GPe']-(G[('GPe', 'STN')]*A['STN'] + G[('GPe', 'GPe')]*A['GPe']) + threshold['GPe']} #  <Single pop> external input coming from Ctx and Str

#mvt_ext_input_dict = { ('STN', '1'): A_mvt['STN']/gain['STN']-G[('STN', 'GPe')]*A_mvt['GPe'] + threshold['STN'] -rest_ext_input['STN'],
#                   ('GPe', '1') : A_mvt['GPe']/gain['GPe']-(G[('GPe', 'STN')]*A_mvt['STN'] + G[('GPe', 'GPe')]*A_mvt['GPe']) + threshold['GPe'] -rest_ext_input['GPe']} # <single pop> external input coming from Ctx and Str

rest_ext_input = { 'STN': A['STN']/gain['STN']-G[('STN', 'GPe')]*A['GPe'] + threshold['STN'] ,
                   'GPe': A['GPe']/gain['GPe']-(G[('GPe', 'STN')]*A['STN']*2 + G[('GPe', 'GPe')]*A['GPe']) + threshold['GPe']} # <double pop> external input coming from Ctx and Str

mvt_ext_input_dict = { ('STN', '1'): A_mvt['STN']/gain['STN']-G[('STN', 'GPe')]*A_mvt['GPe'] + threshold['STN'] -rest_ext_input['STN'],
                   ('GPe', '1') : A_mvt['GPe']/gain['GPe']-(G[('GPe', 'STN')]*A_mvt['STN']*2 + G[('GPe', 'GPe')]*A_mvt['GPe']) + threshold['GPe'] -rest_ext_input['GPe']} # external input coming from Ctx and Str
mvt_ext_input_dict[('STN', '2')] = mvt_ext_input_dict[('STN', '1')] ; mvt_ext_input_dict[('GPe', '2')] = mvt_ext_input_dict[('GPe', '1')]  
pert_val = 10
mvt_selective_ext_input_dict = {('GPe','1') : pert_val, ('GPe','2') : -pert_val,
                                ('STN','1') : pert_val, ('STN','2') : -pert_val} # external input coming from Ctx and Str
dopamine_percentage = 100
t_sim = 1000 # simulation time in ms
dt = 0.5 # euler time step in ms
t_mvt = 200
D_mvt = 500
D_perturb = 500 # transient selective perturbation
d_Str = 200 # duration of external input to Str
t_list = np.arange(int(t_sim/dt))
duration_mvt = [int((t_mvt+ max(T[('GPe', 'D2')],T[('STN', 'Ctx')]))/dt), int((t_mvt+D_mvt+max(T[('GPe', 'D2')],T[('STN', 'Ctx')]))/dt)]
duration_base = [int((max(T[('GPe', 'STN')],T[('STN', 'GPe')]))/dt), int(t_mvt/dt)]

#%%
class Nucleus:

    def __init__(self,population_number, noise_variance, noise_amplitude, N, A, name, G, T, t_sim, dt, tau, n_trans_types, rest_ext_input, receiving_from_dict):
        
        self.n = N[name] # population size
        self.population_num = population_number
        self.name = name
        self.nat_firing = A[name]
        self.tau = dictfilt(tau, n_trans_types) # synaptic time scale based on neuron type
        self.transmission_delay = {k: v for k, v in T.items() if k[0]==name} # filter based on the receiving nucleus
        self.synaptic_weight = {k: v for k, v in G.items() if k[0]==name} # filter based on the receiving nucleus
        self.history_duration = max(self.transmission_delay.values()) # stored history in ms derived from the longest transmission delay of the projections
        self.output = np.zeros((self.n,int(self.history_duration/dt)))
        self.input = np.zeros((self.n))
        self.neuron_act = np.zeros((self.n))
        self.pop_act = np.zeros((int(t_sim/dt))) # time series of population activity
        self.receiving_from_dict = receiving_from_dict[(self.name, str(self.population_num))] 
#        self.receiving_from_list = receiving_from_list
        self.rest_ext_input = rest_ext_input[name]
        self.mvt_ext_input = np.zeros((int(t_sim/dt))) # external input mimicing movement
        self.external_inp_t_series = np.zeros((int(t_sim/dt)))
        self.avg_pop_act_mvt = None
        self.avg_pop_act_base = None
        self.noise_variance =  noise_variance[self.name ] #additive gaussian white noise with mean zero and variance sigma
        self.noise_amplitude =  noise_amplitude[self.name ] #additive gaussian white noise with mean zero and variance sigma
        self.connectivity_matrix = {}
        
    def calculate_input_and_inst_act(self, K, threshold, gain, t, dt, receiving_from_class_list, mvt_ext_inp):  
        
        syn_inputs = np.zeros((self.n,1)) # = Sum (G Jxm)
        
        for projecting in receiving_from_class_list:
#            print(np.matmul(J[(self.name, projecting.name)], projecting.output[:,int(-T[(self.name,projecting.name)]*dt)].reshape(-1,1)).shape)
            syn_inputs += self.synaptic_weight[(self.name, projecting.name)]*np.matmul(self.connectivity_matrix[(projecting.name,str(projecting.population_num))], 
                           projecting.output[:,-int(self.transmission_delay[(self.name,projecting.name)]/dt)].reshape(-1,1))/K[(self.name, projecting.name)]
        
#        print(self.noise_amplitude)
        self.input = syn_inputs + self.rest_ext_input  + mvt_ext_inp #+ noise_generator(self.noise_amplitude, self.noise_variance, self.n)
        self.neuron_act = transfer_func(threshold[self.name], gain[self.name], self.input)
        self.pop_act[t] = np.average(self.neuron_act)
        
    def update_output(self):
        new_output = self.output[:,-1].reshape(-1,1)
        for tau in self.tau.keys():
            new_output += dt*(-self.output[:,-1].reshape(-1,1)+self.neuron_act)/self.tau[tau]
        self.output = np.hstack((self.output[:,1:], new_output))
        
    def set_connections(self, K, N):
        ''' creat Jij connection matrix
        
        Parameters
        ----------
        
        connected_nuclei_list : list
            list of class instances that are connected to this nucleus
        
        Returns
        -------
        '''
        for projecting in self.receiving_from_dict:
            
            n_connections = K[(self.name, projecting[0])]
            self.connectivity_matrix[projecting] = build_connection_matrix(self.n, N[projecting[0]], n_connections)
#            J[(self.name, projecting)] = build_connection_matrix(self.n, N[projecting], n_connections)

def dopamine_effect(threshold, G, dopamine_percentage):
    ''' Change the threshold and synaptic weight depending on dopamine levels'''
    threshold['Str'] = -0.02 + 0.03*(1-(1.1/(1+0.1*np.exp(-0.03*(dopamine_percentage - 100)))))
    G[('Str','Ctx')] = 0.75/(1+np.exp(-0.09*(dopamine_percentage - 60)))
    return threshold, G

def noise_generator(amplitude, variance, n):
    return amplitude * np.random.normal(0,variance, n).reshape(-1,1)
def freq_from_fft(sig,dt):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
#    windowed = sig * signal.blackmanharris(len(sig))
#    f = rfft(windowed)
    N = len(sig)
    if N == 0:
        return 0
    else:
        f = rfft(sig)
        freq = fftfreq(N, dt)#[:N//2]
    #    plt.figure()
    #    plt.semilogy(freq[:N//2], f[:N//2])
        # Find the peak and interpolate to get a more accurate peak
        peak_freq = freq[np.argmax(abs(f))]  # Just use this for less-accurate, naive version
    #    true_i = parabolic(log(abs(f)), i)[0]
    
        # Convert to equivalent frequency
    #    return fs * true_i / len(windowed)

        return peak_freq

    
def freq_from_welch(sig,dt):
    """
    Estimate frequency with Welch method
    """
    fs = 1/(dt)
    [ff,pxx] = signal.welch(sig,axis=0,fs=fs,nperseg=int(fs))
#    plt.semilogy(ff,pxx)
    return ff[np.argmax(pxx)]

def run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuceli_dict):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.set_connections(K, N)
    
    GPe[0].external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,T[('GPe', 'D2')],mvt_ext_input_dict[('GPe','1')], t_list*dt)
    STN[0].external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,T[('STN', 'Ctx')],mvt_ext_input_dict[('STN','1')], t_list*dt)
     
    GPe[1].external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,T[('GPe', 'D2')],mvt_ext_input_dict[('GPe','2')], t_list*dt)
    STN[1].external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,T[('STN', 'Ctx')],mvt_ext_input_dict[('STN','2')], t_list*dt)
     

    start = timeit.default_timer()
    
    for t in t_list:
        for nuclei_list in nuclei_dict.values():
            k = 0
            for nucleus in nuclei_list:
                k += 1
        #        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement 
                mvt_ext_inp = np.ones((nucleus.n,1))*nucleus.external_inp_t_series[t] # movement added 
                nucleus.calculate_input_and_inst_act(K, threshold, gain, t, dt, receiving_class_dict[(nucleus.name,str(k))], mvt_ext_inp)
                nucleus.update_output()
        

    stop = timeit.default_timer()
    print("t = ", stop - start)
    return 
    
def plot( GPe, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_ob, title = "", n_subplots = 1):    
    plot_start = int(5/dt)
    if plot_ob == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot_ob
     
    line_type = ['-', '--']
    
    ax.plot(t_list[plot_start:]*dt,GPe[0].pop_act[plot_start:], line_type[0], label = "GPe" , c = 'r',lw = 1.5)
    ax.plot(t_list[plot_start:]*dt,STN[0].pop_act[plot_start:], line_type[0],label = "STN", c = 'k',lw = 1.5)
    
    ax.plot(t_list[plot_start:]*dt,GPe[1].pop_act[plot_start:], line_type[1], c = 'r',lw = 1.5)
    ax.plot(t_list[plot_start:]*dt,STN[1].pop_act[plot_start:], line_type[1], c = 'k',lw = 1.5)
    
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A['GPe'], '-.', c = 'r',lw = 1, alpha=0.8 )
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt['GPe'], '-.', c = 'r', alpha=0.2,lw = 1 )
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A['STN'], '-.', c = 'k', alpha=0.8,lw = 1 )
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt['STN'], '-.', c = 'k', alpha=0.2,lw = 1 )
    ax.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color='lightskyblue')
    
    plt.title(title, fontsize = 18)
    plt.xlabel("time (ms)", fontsize = 10)
    plt.ylabel("firing rate (spk/s)", fontsize = 10)
    plt.legend(fontsize = 10)
    ax.tick_params(axis='both', which='major', labelsize=10)
#def sweep_time_scales(GPe,STN,GABA_A, GABA_B, Glut, dt, filename,mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuceli_dict):
#
#    STN_freq = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut)))
#    GPe_freq = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut)))
#    tau_mat = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut),3))
#    count = 0
#    for gaba_b in GABA_B:
#        for gaba_a in GABA_A:
#            for glut in Glut:
#                
#                GPe.tau = {'GABA-A' : gaba_a, 'GABA-B' : gaba_b}
#                STN.tau = {'Glut': glut} 
#    
#                run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuceli_dict)
#                tau_mat[count,:] = [gaba_a, gaba_b, glut]
#                sig_STN = STN.pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(STN.pop_act[duration_mvt[0]:duration_mvt[1]])
##                STN_freq[count] = freq_from_welch(sig_STN[cut_plateau(sig_STN)],dt/1000)
#                STN_freq[count] = freq_from_fft(sig_STN[cut_plateau(sig_STN)],dt/1000)
#
#                sig_GPe = GPe.pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(GPe.pop_act[duration_mvt[0]:duration_mvt[1]])
##                GPe_freq [count] = freq_from_welch(sig_GPe[cut_plateau(sig_GPe)],dt/1000)
#                GPe_freq [count] = freq_from_fft(sig_GPe[cut_plateau(sig_GPe)],dt/1000)
#
#                count +=1
#                print(count, "from ", len(GABA_A)*len(GABA_B)*len(Glut))
#    
#    
#    np.savez(filename, tau_mat = tau_mat, GPe = GPe_freq, STN = STN_freq )
#    
#def sweep_time_scales_one_GABA(GPe, STN, inhibitory_trans,inhibitory_series, Glut, dt, filename,mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuceli_dict):
#
#    STN_freq = np.zeros((len(inhibitory_series)*len(Glut)))
#    GPe_freq = np.zeros((len(inhibitory_series)*len(Glut)))
#    tau_mat = np.zeros((len(inhibitory_series)*len(Glut),2))
#    count = 0
#
#    for gaba in inhibitory_series:
#        for glut in Glut:
#            
#            GPe.tau = {inhibitory_trans : gaba}
#            STN.tau = {'Glut': glut} 
#
#            run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuceli_dict)
#            tau_mat[count,:] = [gaba, glut]
#            sig_STN = STN[0].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(STN[0].pop_act[duration_mvt[0]:duration_mvt[1]]) + STN[1].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(STN[1].pop_act[duration_mvt[0]:duration_mvt[1]])
#            
##                STN_freq[count] = freq_from_welch(sig_STN[cut_plateau(sig_STN)],dt/1000)
#            STN_freq[count] = freq_from_fft(sig_STN[cut_plateau(sig_STN)],dt/1000)
#
#            sig_GPe = GPe[0].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(GPe[0].pop_act[duration_mvt[0]:duration_mvt[1]]) +GPe[1].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(GPe[1].pop_act[duration_mvt[0]:duration_mvt[1]])
##                GPe_freq [count] = freq_from_welch(sig_GPe[cut_plateau(sig_GPe)],dt/1000)
#            GPe_freq [count] = freq_from_fft(sig_GPe[cut_plateau(sig_GPe)],dt/1000)
#
#            count +=1
#            print(count, "from ", len(inhibitory_series)*len(Glut))
#
#    
#    np.savez(filename, tau_mat = tau_mat, GPe = GPe_freq, STN = STN_freq )
#    



def mvt_grad_ext_input(D_mvt, t_mvt, delay, H0, t_series):
    ''' a gradually increasing deacreasing input mimicing movement'''

    H = H0*np.cos(2*np.pi*(t_series-t_mvt)/D_mvt)**2
    ind = np.logical_or(t_series < t_mvt + delay, t_series > t_mvt + D_mvt + delay)
    H[ind] = 0
    return H
def mvt_step_ext_input(D_mvt,t_mvt,delay, H0, t_series):
    ''' step function as movement signal input '''
    
    H = H0*np.ones_like(t_series)
    ind = np.logical_or(t_series < t_mvt + delay, t_series > t_mvt + D_mvt + delay)
    H[ind] = 0
    return H
#plt.plot(t_list, mvt_ext_input(D_mvt,t_mvt,H0, t_list/dt))

def calculate_number_of_connections(N_sim,N_real,K_real):
    '''calculate number of connections in the scaled network.'''
    KK = K_real.copy()
    for k, v in K_real.items():
        KK[k] = int(1/(1/v-1/N_real[k[1]]+1/N_sim[k[0]]))
    return KK

def transfer_func(Threshold, gain, x):
    ''' a transfer function that grows linearly for positive values 
    of input higher than the threshold'''
    return gain* np.maximum(np.zeros_like(x), (x - Threshold))
    
def build_connection_matrix(n_receiving,n_projecting,n_connections):
    ''' return a matrix with Jij=0 or 1. 1 showing a projection from neuron j in projectin population to neuron i in receiving'''
    # produce a matrix listing received projections for each neuron in row i
    projection_list = np.random.rand(n_receiving, n_projecting).argpartition(n_connections,axis=1)[:,:n_connections]
#    print(projection_list)
    JJ = np.zeros((n_receiving, n_projecting))
    rows = ((np.ones((n_connections,n_receiving))*np.arange(n_receiving)).T).flatten().astype(int)
    cols = projection_list.flatten().astype(int)
    JJ[rows,cols] = 1
    return JJ

dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

def scatter_3d_plot(x,y,z,c, title, c_upper_limit, c_lower_limit, label):

    ind = np.logical_and(c<=c_upper_limit, c>=c_lower_limit)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x[ind], y[ind], z[ind], c=c[ind], cmap=plt.hot(),lw = 1,edgecolor = 'k')
    
#    ax = Axes3D(fig)
#    surf = ax.plot_trisurf(x[ind],y[ind],z[ind], cmap = cm.coolwarm)
#    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_zlabel(label[2])
    ax.set_title(title)
    clb = fig.colorbar(img)
    clb.set_label(label[3], labelpad=-40, y=1.05, rotation=0)
    plt.show()
    
#build_connection_matrix(4,10,2)
   
def max_non_empty_array(array):
    if len(array) == 0:
        return 0
    else:
        return np.max(array)

def if_oscillatory(sig, x_plateau, noise_amplitude):
    ''' detect if there are peaks with larger amplitudes than noise in mean subtracted data before plateau'''
    fluctuations = sig - np.average(sig[x_plateau:-1])
    peaks,_ = signal.find_peaks(fluctuations, height = noise_amplitude)
    
    if len(peaks) >0:
        return True
    else:
        return False
def synaptic_weight_space_exploration(g_inh_list, g_exit_list, GPe, STN, duration_mvt, duration_base):
    
    n = len(g_inh_list)
    m = len(g_exit_list)
    g_mat = np.zeros((int(n*m),3))
    STN_prop = {'base_f' : np.zeros((int(m*n))), 'mvt_f' : np.zeros((int(m*n))),
                'perc_t_oscil_base': np.zeros((int(m*n))), 'perc_t_oscil_mvt': np.zeros((int(m*n)))}
    GPe_prop = {'base_f' : np.zeros((int(m*n))), 'mvt_f' : np.zeros((int(m*n))),
                'perc_t_oscil_base': np.zeros((int(m*n))), 'perc_t_oscil_mvt': np.zeros((int(m*n)))}

    count  = 0
    fig = plt.figure()
    for g_inh in g_inh_list:
        for g_exit in g_exit_list:
            GPe.synaptic_weight[('GPe', 'STN')] = g_exit
            GPe.synaptic_weight[('GPe', 'GPe')] = g_inh*0.5
            STN.synaptic_weight[('STN','GPe')] = g_inh
            g_mat[count,:] = [g_exit, g_inh, g_inh*0.5]
            nuclei_dict = {'GPe': GPe, 'STN' : STN}
            run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
            GPe_test = GPe[0] ; STN_test = STN[0]
            
            x1_gp_mvt = np.argmax(GPe_test.pop_act[int(t_mvt/dt):int((t_mvt+D_mvt)/dt)])+int(t_mvt/dt)
            x1_stn_mvt = np.argmax(STN_test.pop_act[int(t_mvt/dt):int((t_mvt+D_mvt)/dt)])+int(t_mvt/dt)
            x2_mvt = duration_mvt[1]
            x1_gp_base = np.argmax(GPe_test.pop_act[0:int(t_mvt/dt)])
            x1_stn_base = np.argmax(STN_test.pop_act[0:int(t_mvt/dt)])
            x2_base = duration_base[1]
            
            sig_STN_mvt = STN_test.pop_act[x1_stn_mvt:x2_mvt] - np.average(STN_test.pop_act[x1_stn_mvt:x2_mvt])
            cut_sig_ind_mvt = cut_plateau(sig_STN_mvt)
            
            STN_prop[('perc_t_oscil_mvt')][count] = max_non_empty_array(cut_sig_ind_mvt)/len(sig_STN_mvt)*100
            STN_prop[('mvt_f')][count] = freq_from_fft(sig_STN_mvt[cut_sig_ind_mvt],dt/1000)

            sig_STN_base = STN_test.pop_act[x1_stn_base:x2_base] - np.average(STN_test.pop_act[x1_stn_base:x2_base])
            cut_sig_ind_base = cut_plateau(sig_STN_base)
            STN_prop[('perc_t_oscil_base')][count] = max_non_empty_array(cut_sig_ind_base)/len(sig_STN_base)*100
            STN_prop[('base_f')][count] = freq_from_fft(sig_STN_base[cut_sig_ind_base],dt/1000)

            sig_GPe_mvt = GPe_test.pop_act[x1_gp_mvt:x2_mvt] - np.average(GPe_test.pop_act[x1_gp_mvt:x2_mvt])
            cut_sig_ind_mvt = cut_plateau(sig_GPe_mvt)
            GPe_prop[('perc_t_oscil_mvt')][count] = max_non_empty_array(cut_sig_ind_mvt)/len(sig_GPe_mvt)*100
            GPe_prop[('mvt_f')][count] = freq_from_fft(sig_GPe_mvt[cut_sig_ind_mvt],dt/1000)

            sig_GPe_base = GPe_test.pop_act[x1_gp_base:x2_base] - np.average(GPe_test.pop_act[x1_gp_base:x2_base])
            cut_sig_ind_base = cut_plateau(sig_GPe_base)
            GPe_prop[('perc_t_oscil_base')][count] = max_non_empty_array(cut_sig_ind_base)/len(sig_GPe_base)*100
            GPe_prop[('base_f')][count] = freq_from_fft(sig_GPe_base[cut_sig_ind_base],dt/1000)
            
            ax = fig.add_subplot(n,m,count+1)
            plot(GPe_test, STN_test, dt, t_list, A, A_mvt, t_mvt, D_mvt,[fig, ax], title = r"$G_{STN-GPe}$ = "+ str(round(g_inh,2))+r' $G_{GPe-STN}$ ='+str(round(g_exit,2)), n_subplots = int(n*m))
            plt.title( r"$G_{STN-GPe}$ = "+ str(round(g_inh,2))+r' $G_{GPe-STN}$ ='+str(round(g_exit,2)), fontsize = 5)
            plt.xlabel("time (ms)", fontsize = 5)
            plt.ylabel("firing rate (spk/s)", fontsize = 5)
            plt.legend(fontsize = 2)
            ax.tick_params(axis='both', which='major', labelsize=2)
            count +=1
            print(count, "from", int(m*n))
            
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    return g_mat, GPe_prop, STN_prop
       
def cut_plateau(sig,epsilon_std = 10**(-2), epsilon = 10**(-2), window = 40):
    ''' return indices before a plateau '''
    ## filtering based on data variance from mean value
#    variation = (sig-np.average(sig))**2
#    ind = np.where(variation >= epsilon_std) 
#    return ind[0]
    ## filtering based on where the first and second derivatives are zero. Doesn't work with noise
#    derivative = np.diff(sig)
#    derivative_2 = np.diff(derivative)
#    ind = np.logical_and(np.abs(derivative[:-1])<epsilon,np.abs(derivative_2) < epsilon )
#    plateau_start = np.max(np.where(~ind))
#    return np.arange(plateau_start)
    ##
    rolling_var = np.var(rolling_window(sig, window), axis=-1)
    ind = np.where(rolling_var > epsilon)
    return ind[0]

def moving_average_array(X, n):
	'''Return the moving average over X with window n without changing dimesions of X'''

	z2= np.cumsum(np.pad(X, (n,0), 'constant', constant_values=0))
	z1 = np.cumsum(np.pad(X, (0,n), 'constant', constant_values=X[-1]))
	return (z1-z2)[(n-1):-1]/n

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def sweep_time_scales(GPe,STN,GABA_A, GABA_B, Glut, dt, filename,mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain):

    STN_freq = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut)))
    GPe_freq = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut)))
    tau_mat = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut),3))
    count = 0
    for gaba_b in GABA_B:
        for gaba_a in GABA_A:
            for glut in Glut:
                for k in range (len(GPe)):
                    GPe[k].tau = {'GABA-A' : gaba_a, 'GABA-B' : gaba_b}
                    STN[k].tau = {'Glut': glut} 
                nuclei_dict = {'GPe': GPe, 'STN' : STN}
                run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
                tau_mat[count,:] = [gaba_a, gaba_b, glut]
                sig_STN = STN[0].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(STN[0].pop_act[duration_mvt[0]:duration_mvt[1]])
#                STN_freq[count] = freq_from_welch(sig_STN[cut_plateau(sig_STN)],dt/1000)
                STN_freq[count] = freq_from_fft(sig_STN[cut_plateau(sig_STN)],dt/1000)

                sig_GPe = GPe[0].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(GPe[0].pop_act[duration_mvt[0]:duration_mvt[1]])
#                GPe_freq [count] = freq_from_welch(sig_GPe[cut_plateau(sig_GPe)],dt/1000)
                GPe_freq [count] = freq_from_fft(sig_GPe[cut_plateau(sig_GPe)],dt/1000)

                count +=1
                print(count, "from ", len(GABA_A)*len(GABA_B)*len(Glut))
    
    np.savez(filename, tau_mat = tau_mat, GPe = GPe_freq, STN = STN_freq )
    
def sweep_time_scales_one_GABA(GPe, STN, inhibitory_trans,inhibitory_series, Glut, dt, filename,mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain):

    STN_freq = np.zeros((len(inhibitory_series)*len(Glut)))
    GPe_freq = np.zeros((len(inhibitory_series)*len(Glut)))
    tau_mat = np.zeros((len(inhibitory_series)*len(Glut),2))
    count = 0

    for gaba in inhibitory_series:
        for glut in Glut:
            
            for k in range (len(GPe)):
                GPe[k].tau = {inhibitory_trans : gaba}
                STN[k].tau = {'Glut': glut} 

            nuclei_dict = {'GPe': GPe, 'STN' : STN}
            run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
            tau_mat[count,:] = [gaba, glut]
            sig_STN = STN[0].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(STN[0].pop_act[duration_mvt[0]:duration_mvt[1]]) + STN[1].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(STN[1].pop_act[duration_mvt[0]:duration_mvt[1]])
            
#                STN_freq[count] = freq_from_welch(sig_STN[cut_plateau(sig_STN)],dt/1000)
            STN_freq[count] = freq_from_fft(sig_STN[cut_plateau(sig_STN)],dt/1000)

            sig_GPe = GPe[0].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(GPe[0].pop_act[duration_mvt[0]:duration_mvt[1]]) +GPe[1].pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(GPe[1].pop_act[duration_mvt[0]:duration_mvt[1]])
#                GPe_freq [count] = freq_from_welch(sig_GPe[cut_plateau(sig_GPe)],dt/1000)
            GPe_freq [count] = freq_from_fft(sig_GPe[cut_plateau(sig_GPe)],dt/1000)

            count +=1
            print(count, "from ", len(inhibitory_series)*len(Glut))

    
    np.savez(filename, tau_mat = tau_mat, GPe = GPe_freq, STN = STN_freq )
    

#%% STN-GPe network
G = { ('STN', 'GPe'): -3 ,
  ('GPe', 'STN'): .62, 
  ('GPe', 'GPe'): 0} # synaptic weight
G[('GPe', 'GPe')] = 0.5* G[('STN', 'GPe')]

#K = calculate_number_of_connections(N,N_real,K_real)
receiving_pop_list = {('STN','1') : [('GPe', '1')], ('STN','2') : [('GPe', '2')],
                    ('GPe','1') : [('GPe', '1'), ('STN', '1'), ('STN', '2')],
                    ('GPe','2') : [('GPe', '2'), ('STN', '1'), ('STN', '2')]}
K = calculate_number_of_connections(N,N_real,K_real_STN_GPe_diverse)

GPe = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list)]
STN = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list)]
nuclei_dict = {'GPe': GPe, 'STN' : STN}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

run(mvt_selective_ext_input_dict, D_perturb,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
plot(GPe, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
#%% GPe-FSI-D2 network
receiving_pop_list = {('FSI','1') : [('GPe', '1')], ('FSI','2') : [('GPe', '2')],
                    ('GPe','1') : [('GPe', '1'), ('D2', '1')],
                    ('GPe','2') : [('GPe', '2'), ('D2', '1')],
                    ('D2','1') : [('FSI','1')], ('D2','2') : [('FSI','2')]}
K = calculate_number_of_connections(N,N_real,K_real_STN_GPe_diverse)

GPe = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list)]
D2 = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'D2', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'D2', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list)]
FSI = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'FSI', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'FSI', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list)]
nuclei_dict = {'GPe': GPe, 'D2' : D2}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

run(mvt_selective_ext_input_dict, D_perturb,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
plot(GPe, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
#%% synaptic weight phase exploration
n_inh = 8 ; n_exit = 8
g_inh_list = np.linspace(-3, -0.1, n_inh)
g_exit_list = np.linspace(0.1, 3, n_exit)

K_real = { ('STN', 'GPe'): 883, # Baufreton et al. (2009)
           ('GPe', 'STN'): 190, # Kita, H., and Jaeger, D. (2016)
           ('GPe', 'GPe'): 650} # Hegeman et al. (2017)
K_real_STN_GPe_diverse = K_real.copy()
K_real_STN_GPe_diverse[('GPe', 'STN')] = K_real_STN_GPe_diverse[('GPe', 'STN')] / N_sub_pop # because one subpop in STN contacts all subpop in GPe

receiving_pop_list = {('STN','1') : [('GPe', '1')], ('STN','2') : [('GPe', '2')],
                    ('GPe','1') : [('GPe', '1'), ('STN', '1'), ('STN', '2')],
                    ('GPe','2') : [('GPe', '2'), ('STN', '1'), ('STN', '2')]}
#receiving_pop_list = {('STN','1') : [('GPe', '1')], ('STN','2') : [('GPe', '2')],
#                    ('GPe','1') : [('STN', '1'), ('STN', '2')],
#                    ('GPe','2') : [('STN', '1'), ('STN', '2')]}
K = calculate_number_of_connections(N,N_real,K_real_STN_GPe_diverse)

GPe = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list)]
STN = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list)]
nuclei_dict = {'GPe': GPe, 'STN' : STN}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

g_mat, GPe_prop, STN_prop = synaptic_weight_space_exploration(g_inh_list, g_exit_list, GPe, STN, duration_mvt, duration_base)
scatter_3d_plot(g_mat[:,0],g_mat[:,1],STN_prop['perc_t_oscil_base'],STN_prop['perc_t_oscil_base'], 'STN', np.max(STN_prop['perc_t_oscil_base']), np.min(STN_prop['perc_t_oscil_base']), ['GPe-STN', 'STN-GPe', '% basal oscillatory period',  '% basal oscillatory period'])
scatter_3d_plot(g_mat[:,0],g_mat[:,1],STN_prop['perc_t_oscil_mvt'],STN_prop['perc_t_oscil_mvt'], 'STN', np.max(STN_prop['perc_t_oscil_mvt']), np.min(STN_prop['perc_t_oscil_mvt']), ['GPe-STN', 'STN-GPe', '% mvt oscillatory period', '% mvt oscillatory period'])
scatter_3d_plot(g_mat[:,0],g_mat[:,1],GPe_prop['perc_t_oscil_mvt'],GPe_prop['perc_t_oscil_mvt'], 'GPe', np.max(GPe_prop['perc_t_oscil_mvt']), np.min(GPe_prop['perc_t_oscil_mvt']), ['GPe-STN', 'STN-GPe', '% mvt oscillatory period', '% mvt oscillatory period'])
#scatter_3d_plot(g_mat[:,0],g_mat[:,1],STN_prop['mvt_f'],STN_prop['mvt_f'], 'STN', np.max(STN_prop['mvt_f']), np.min(STN_prop['mvt_f']), ['GPe-STN', 'STN-GPe', 'mvt freq', 'mvt freq'])
fig = plt.figure()
img = plt.scatter(g_mat[:,0],-g_mat[:,1], c = STN_prop['perc_t_oscil_base'], cmap=plt.hot(),lw = 1,edgecolor = 'k')
plt.xlabel(r'$G_{GPe-STN}$')
plt.ylabel(r'$G_{STN-GPe}$')
plt.title('STN')
clb = fig.colorbar(img)
clb.set_label('% basal oscillation period', labelpad=-40, y=1.05, rotation=0)
plt.show()

fig = plt.figure()
img = plt.scatter(g_mat[:,0],-g_mat[:,1], c = STN_prop['perc_t_oscil_mvt'], cmap=plt.hot(),lw = 1,edgecolor = 'k')
plt.xlabel(r'$G_{GPe-STN}$')
plt.ylabel(r'$G_{STN-GPe}$')
plt.title('STN')
clb = fig.colorbar(img)
clb.set_label('% mvt oscillation period', labelpad=-40, y=1.05, rotation=0)
plt.show()
#%% # time scale parameter space with frequency of transient oscillations at steady state
K_real = { ('STN', 'GPe'): 883, # Baufreton et al. (2009)
           ('GPe', 'STN'): 190, # Kita, H., and Jaeger, D. (2016)
           ('GPe', 'GPe'): 650} # Hegeman et al. (2017)
K_real_STN_GPe_diverse = K_real.copy()
K_real_STN_GPe_diverse[('GPe', 'STN')] = K_real_STN_GPe_diverse[('GPe', 'STN')] / N_sub_pop # because one subpop in STN contacts all subpop in GPe

receiving_pop_list = {('STN','1') : [('GPe', '1')], ('STN','2') : [('GPe', '2')],
                    ('GPe','1') : [('GPe', '1'), ('STN', '1'), ('STN', '2')],
                    ('GPe','2') : [('GPe', '2'), ('STN', '1'), ('STN', '2')]}
K = calculate_number_of_connections(N,N_real,K_real_STN_GPe_diverse)

GPe = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A', 'GABA-B'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A', 'GABA-B'], rest_ext_input, receiving_pop_list)]
STN = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list)]
nuclei_dict = {'GPe': GPe, 'STN' : STN}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

#run(mvt_selective_ext_input_dict, D_perturb,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
n = 7
GABA_A = np.linspace(5,20,n)
GABA_B = np.linspace(150,300,4)
Glut = np.linspace(0.5,12,n)
sweep_time_scales(GPe,STN, GABA_A, GABA_B, Glut, dt, 'data_GABA_A_B_Glut.npz',mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain)
file = np.load('data_GABA_A_B_Glut.npz')
x = file['tau_mat'][:,0]
y = file['tau_mat'][:,1]
z = file['tau_mat'][:,2]

scatter_3d_plot(x,y,z, file['STN'],'STN', np.max(file['STN']), np.min(file['STN']),['GABA_A','GABA_B','Glut','freq'])
scatter_3d_plot(x,y,z, file['STN'],'STN', 30, 20,['GABA_A','GABA_B','Glut','freq'])
scatter_3d_plot(x,y,z, file['STN'],'STN', 31, 27,['GABA_A','GABA_B','Glut','freq'])

scatter_3d_plot(x,y,z, file['GPe'],'GPe', np.max(file['GPe']), np.min(file['GPe']),['GABA_A','GABA_B','Glut','freq'])
scatter_3d_plot(x,y,z, file['GPe'],'GPe',  30, 20 ,['GABA_A','GABA_B','Glut','freq'])
scatter_3d_plot(x,y,z, file['GPe'],'GPe', 31, 27,['GABA_A','GABA_B','Glut','freq'])


######################################3 only GABA-A
GPe = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list)]
STN = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list)]
nuclei_dict = {'GPe': GPe, 'STN' : STN}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]
    
sweep_time_scales_one_GABA(GPe, STN, 'GABA_A', GABA_A, Glut, dt, 'data_GABA_A.npz',mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain)
file = np.load('data_GABA_A.npz')
x = file['tau_mat'][:,0]
y = file['tau_mat'][:,1]
z = file['STN']
scatter_3d_plot(x,y,z, file['STN'],'STN', np.max(file['STN']), np.min(file['STN']),['GABA_A','Glut','freq','freq'])

z = file['GPe']
scatter_3d_plot(x,y,z, file['GPe'],'GPe', np.max(z), np.min(z),['GABA_A','Glut','freq', 'freq'])


################################### only GABA-B

GPe = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-B'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-B'], rest_ext_input, receiving_pop_list)]
STN = [Nucleus(1, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list),
       Nucleus(2, noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list)]
nuclei_dict = {'GPe': GPe, 'STN' : STN}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]
GABA_B = np.linspace(150,300,8)
sweep_time_scales_one_GABA(GPe, STN, 'GABA_B', GABA_B, Glut, dt, 'data_GABA_B_Glut.npz',mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain)
file = np.load('data_GABA_B_Glut.npz')
x = file['tau_mat'][:,0]
y = file['tau_mat'][:,1]
z = file['STN']
scatter_3d_plot(x,y,z, file['STN'],'STN', np.max(file['STN']), np.min(file['STN']),['GABA_B','Glut','freq','freq'])

z = file['GPe']
scatter_3d_plot(x,y,z, file['GPe'],'GPe', np.max(z), np.min(z),['GABA_B','Glut','freq', 'freq'])

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
    


    

    

    
    
    
    
    