import numpy as np
import matplotlib.pyplot as plt
import timeit
from numpy.fft import rfft,fft
from scipy import signal

N_sim = 1000
population_list = ['STN', 'GPe']
N = { 'STN': N_sim , 'GPe': N_sim}
N_real = { 'STN': 13560 , 'GPe': 34470}

A = { 'STN': 15 , 'GPe': 30} # mean firing rate from experiments
A_mvt = { 'STN': 50 , 'GPe': 22} # mean firing rate during movement from experiments
threshold = { 'STN': .1 ,'GPe': .1}
neuron_type = {'STN': 'Glut', 'GPe': 'GABA'}
gain = { 'STN': 1 ,'GPe': 1}

#K = { ('STN', 'GPe'): 475,
#      ('GPe', 'STN'): 161,
#      ('GPe', 'GPe'): 399}
K_real = { ('STN', 'GPe'): 883,
           ('GPe', 'STN'): 190,
           ('GPe', 'GPe'): 650}
T = { ('STN', 'GPe'): 4, # Fujimoto & Kita (1993)
      ('GPe', 'STN'): 2, # kita & Kitai (1991)
      ('GPe', 'GPe'): 1.5} # transmission delay in ms
G = { ('STN', 'GPe'): -2 ,
      ('GPe', 'STN'): 1 , 
      ('GPe', 'GPe'): 0} # synaptic weight
G[('GPe', 'GPe')] = 0.5* G[('STN', 'GPe')]
tau = {'GABA-A' : 6, 'GABA-B': 200, 'Glut': 3.5} # Gerstner. synaptic time scale for excitation and inhibition

rest_ext_input = { 'STN': A['STN']/gain['STN']-G[('STN', 'GPe')]*A['GPe'] + threshold['STN'] ,
                   'GPe': A['GPe']/gain['GPe']-(G[('GPe', 'STN')]*A['STN'] + G[('GPe', 'GPe')]*A['GPe']) + threshold['GPe']} # external input coming from Ctx and Str

mvt_ext_input_dict = { 'STN': A_mvt['STN']/gain['STN']-G[('STN', 'GPe')]*A_mvt['GPe'] + threshold['STN'] -rest_ext_input['STN'],
                   'GPe': A_mvt['GPe']/gain['GPe']-(G[('GPe', 'STN')]*A_mvt['STN'] + G[('GPe', 'GPe')]*A_mvt['GPe']) + threshold['GPe'] -rest_ext_input['GPe']} # external input coming from Ctx and Str

J = {}
t_sim = 1000 # simulation time in ms
dt = 0.5 # euler time step in ms
t_mvt = 200
D_mvt = 500
d_Str = 200 # duration of external input to Str
t_ext_to_Ctx = (t_mvt-D_mvt/2,t_mvt+D_mvt/2)
t_ext_to_Str = (t_mvt-D_mvt/2,t_mvt-D_mvt/2+d_Str)
t_list = np.arange(int(t_sim/dt))

def freq_from_fft(sig):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * signal.blackmanharris(len(sig))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
#    true_i = parabolic(log(abs(f)), i)[0]

    # Convert to equivalent frequency
#    return fs * true_i / len(windowed)
    return i

def freq_from_welch(sig,dt):
    """
    Estimate frequency from peak of FFT
    """
    fs = 1/(dt/1000)
    [ff,pxx] = signal.welch(sig,axis=0,nperseg=int(fs),fs=fs)#,nfft=1024)
    
    return ff[np.argmax(pxx)]

def mvt_grad_ext_input(D_mvt,t_mvt,H0, t_series):
    ''' a gradually increasing deacreasing input mimicing movement'''

    H = H0*np.cos(2*np.pi*(t_series-t_mvt)/D_mvt)**2
    ind = np.logical_or(t_series<t_mvt, t_series>t_mvt+D_mvt)
    H[ind] = 0
    return H
def mvt_step_ext_input(D_mvt,t_mvt,H0, t_series):
    
    H = H0*np.ones_like(t_series)
    ind = np.logical_or(t_series<t_mvt, t_series>t_mvt+D_mvt)
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
#    print(x.shape)
#    print(np.maximum(np.zeros_like(x),x))
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
#build_connection_matrix(4,10,2)
class Nucleus:

    def __init__(self, N, A, name, T, t_sim, dt, tau, n_trans_types, rest_ext_input):
        self.n = N[name] # population size
        self.name = name
        self.nat_firing = A[name]
        self.tau = dictfilt(tau, n_trans_types) # synaptic time scale based on neuron type
        self.transmission_delays = {k: v for k, v in T.items() if k[1]==name} # filter based on the sending nucleus
        self.history_duration = max(self.transmission_delays.values()) # stored history in ms derived from the longest transmission delay of the projections
        self.output = np.zeros((self.n,int(self.history_duration/dt)))
        self.input = np.zeros((self.n))
        self.neuron_act = np.zeros((self.n))
        self.pop_act = np.zeros((int(t_sim/dt))) # time series of population activity
        self.receiving_from_list = [k[1] for k, v in T.items() if k[0]==name]
        self.rest_ext_input = rest_ext_input[name]
        self.mvt_ext_input = np.zeros((int(t_sim/dt))) # external input mimicing movement
        self.perturbation_in_time = np.zeros((int(t_sim/dt)))
#        connection_arr = np.random.choice(n, k, replace=False) # choose k neurons out of n to send input to neuron i 
        
    def fix_firing_rate(self,inputs):
        
        self.input = inputs
        
    def calculate_input_and_inst_act(self,threshold, gain, T, t, dt, receiving_from_class_list, mvt_ext_inp):    
        syn_inputs = np.zeros((self.n,1)) # = Sum (G Jxm)
        for projecting in receiving_from_class_list:
#            print(np.matmul(J[(self.name, projecting.name)], projecting.output[:,int(-T[(self.name,projecting.name)]*dt)].reshape(-1,1)).shape)
            syn_inputs += G[(self.name, projecting.name)]*np.matmul(J[(self.name, projecting.name)], 
                           projecting.output[:,-int(T[(self.name,projecting.name)]/dt)].reshape(-1,1))/K[(self.name, projecting.name)]
        self.input = syn_inputs + self.rest_ext_input  + mvt_ext_inp
#        self.input =  self.rest_ext_input* np.ones_like(syn_inputs) 

        self.neuron_act = transfer_func(threshold[self.name], gain[self.name], self.input)
#        self.neuron_act =  self.input

        self.pop_act[t] = np.average(self.neuron_act)
        
    def update_output(self):
        new_output = nucleus.output[:,-1].reshape(-1,1)
        for tau in self.tau.keys():
            new_output += dt*(-nucleus.output[:,-1].reshape(-1,1)+nucleus.neuron_act)/nucleus.tau[tau]
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
        for projecting in self.receiving_from_list:
            
            n_connections = K[(self.name, projecting)]
            J[(self.name, projecting)] = build_connection_matrix(self.n, N[projecting], n_connections)

#%%
K = calculate_number_of_connections(N,N_real,K_real)
#GPe = Nucleus(N, A, 'GPe', T, t_sim, dt, tau, ['GABA-B'], rest_ext_input)

GPe = Nucleus(N, A, 'GPe', T, t_sim, dt, tau, ['GABA-A','GABA-B'], rest_ext_input)
STN = Nucleus(N, A, 'STN', T, t_sim, dt, tau, ['Glut'], rest_ext_input)

GPe.set_connections(K, N)
STN.set_connections(K, N)

#GPe.perturbation_in_t =  mvt_grad_ext_input(D_mvt,t_mvt,mvt_ext_input_dict['GPe'], t_list*dt)
#STN.perturbation_in_t =  mvt_grad_ext_input(D_mvt,t_mvt,mvt_ext_input_dict['STN'], t_list*dt)
GPe.perturbation_in_t =  mvt_step_ext_input(D_mvt,t_mvt,mvt_ext_input_dict['GPe'], t_list*dt)
STN.perturbation_in_t =  mvt_step_ext_input(D_mvt,t_mvt,mvt_ext_input_dict['STN'], t_list*dt)
 
nuclei_list = [GPe, STN]
receiving_class_dict = {'STN': [GPe], 'GPe': [GPe, STN]} # points to the classes of nuclei each receive projections from

start = timeit.default_timer()

for t in t_list:
    for nucleus in nuclei_list:
#        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement 
        mvt_ext_inp = np.ones((nucleus.n,1))*nucleus.perturbation_in_t[t] # movement added 
        nucleus.calculate_input_and_inst_act(threshold,gain, T, t, dt, receiving_class_dict[nucleus.name],mvt_ext_inp)
        nucleus.update_output()
        
stop = timeit.default_timer()
print("t = ", stop - start)
plot_start = int(5/dt)
plt.figure(1)    
plt.plot(t_list[plot_start:]*dt,GPe.pop_act[plot_start:],label = "GPe", c = 'r')
plt.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A['GPe'], '--', c = 'r', alpha=0.8 )
plt.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt['GPe'], '--', c = 'r', alpha=0.15 )

#plt.title("GPe")
#plt.xlabel("time (ms)")
#plt.ylabel("firing rate (spk/s)")
#plt.figure(2)    
plt.plot(t_list[plot_start:]*dt,STN.pop_act[plot_start:], label = "STN", c = 'k')
plt.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A['STN'], '--', c = 'k', alpha=0.8 )
plt.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt['STN'], '--', c = 'k', alpha=0.15 )
plt.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color='lightskyblue')

#plt.title("STN")    
plt.xlabel("time (ms)")
plt.ylabel("firing rate (spk/s)")
plt.legend()
   
 
from scipy.ndimage.filters import generic_filter

t_freq = [int(t_mvt/dt), int((t_mvt+D_mvt/5)/dt)]
freq_from_welch(STN.pop_act[t_freq[0]:t_freq[1]],dt)

#tt = t_list[t_freq[0]:t_freq[1]]
#sig1  = np.sin(2*np.pi*10*tt)#*np.exp(-0.015*t_list[t_freq[0]:t_freq[1]])

sig2 = STN.pop_act[t_freq[0]:t_freq[1]]#-np.average(STN.pop_act[t_freq[0]:t_freq[1]])+10
sig_filtered = generic_filter(sig2, np.std, size=10)
#plt.plot(sig1)
plt.plot(sig2)
plt.plot(sig_filtered)
#fs = 1/(dt/1000)
#[ff,pxx] = signal.welch(sig2,axis=0,nperseg=int(fs),fs=fs)#,nfft=1024)
#plt.plot(ff,pxx)
#ff[np.argmax(pxx)]
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
    


    

    

    
    
    
    
    