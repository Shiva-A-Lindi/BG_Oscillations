import numpy as np
import matplotlib.pyplot as plt
import timeit
from numpy.fft import rfft,fft, fftfreq
from scipy import signal
from tempfile import TemporaryFile
from mpl_toolkits.mplot3d import Axes3D
#from scipy.ndimage.filters import generic_filter

N_sim = 1000
population_list = ['STN', 'GPe']
N_sub_pop = 2
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
K_real = { ('STN', 'GPe'): 883, # Baufreton et al. (2009)
           ('GPe', 'STN'): 190, # Kita, H., and Jaeger, D. (2016)
           ('GPe', 'GPe'): 650} # Hegeman et al. (2017)
K_real_STN_GPe_diverse = K_real.copy()
K_real_STN_GPe_diverse[('GPe', 'STN')] = K_real_STN_GPe_diverse[('GPe', 'STN')] / N_sub_pop # because one subpop in STN contacts all subpop in GPe
T = { ('STN', 'GPe'): 4, # Fujimoto & Kita (1993)
      ('GPe', 'STN'): 2, # kita & Kitai (1991)
      ('GPe', 'GPe'): 1.5, # estimate
      ('GPe', 'Str'): 5, # Kita & Kitai (1991)
      ('STN', 'Ctx'): 5.5} # kita & Kita (2011)
    # transmission delay in ms
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
duration_mvt = [int((t_mvt+ max(T[('GPe', 'Str')],T[('STN', 'Ctx')]))/dt), int((t_mvt+D_mvt+max(T[('GPe', 'Str')],T[('STN', 'Ctx')]))/dt)]
duration_base = [int((max(T[('GPe', 'STN')],T[('STN', 'GPe')]))/dt), int(t_mvt/dt)]

#%%
class Nucleus:

    def __init__(self,population_number, N, A, name, G, T, t_sim, dt, tau, n_trans_types, rest_ext_input):
        
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
        self.receiving_from_list = [k[1] for k, v in G.items() if k[0]==name]
#        self.receiving_from_list = receiving_from_list
        self.rest_ext_input = rest_ext_input[name]
        self.mvt_ext_input = np.zeros((int(t_sim/dt))) # external input mimicing movement
        self.external_inp_t_series = np.zeros((int(t_sim/dt)))
        self.avg_pop_act_mvt = None
        self.avg_pop_act_base = None

    def calculate_input_and_inst_act(self, J, K, threshold, gain, t, dt, receiving_from_class_list, mvt_ext_inp):  
        
        syn_inputs = np.zeros((self.n,1)) # = Sum (G Jxm)
        
        for projecting in receiving_from_class_list:
#            print(np.matmul(J[(self.name, projecting.name)], projecting.output[:,int(-T[(self.name,projecting.name)]*dt)].reshape(-1,1)).shape)
            syn_inputs += self.synaptic_weight[(self.name, projecting.name)]*np.matmul(J[(self.name, projecting.name)], 
                           projecting.output[:,-int(self.transmission_delay[(self.name,projecting.name)]/dt)].reshape(-1,1))/K[(self.name, projecting.name)]
        
        self.input = syn_inputs + self.rest_ext_input  + mvt_ext_inp
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
        for projecting in self.receiving_from_list:
            
            n_connections = K[(self.name, projecting)]
            J[(self.name, projecting)] = build_connection_matrix(self.n, N[projecting], n_connections)

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

def run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, J, K, threshold, gain):

    
    GPe.set_connections(K, N)
    STN.set_connections(K, N)
    
    GPe.external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,T[('GPe', 'Str')],mvt_ext_input_dict['GPe'], t_list*dt)
    STN.external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,T[('STN', 'Ctx')],mvt_ext_input_dict['STN'], t_list*dt)
     
    nuclei_list = [GPe, STN]
    
    
    start = timeit.default_timer()
    
    for t in t_list:
        for nucleus in nuclei_list:
    #        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement 
            mvt_ext_inp = np.ones((nucleus.n,1))*nucleus.external_inp_t_series[t] # movement added 
            nucleus.calculate_input_and_inst_act(J, K, threshold, gain, t, dt, receiving_class_dict[nucleus.name], mvt_ext_inp)
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
     
    ax.plot(t_list[plot_start:]*dt,GPe.pop_act[plot_start:],label = "GPe", c = 'r')
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A['GPe'], '--', c = 'r', alpha=0.8 )
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt['GPe'], '--', c = 'r', alpha=0.15 )
    
      
    ax.plot(t_list[plot_start:]*dt,STN.pop_act[plot_start:], label = "STN", c = 'k')
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A['STN'], '--', c = 'k', alpha=0.8 )
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt['STN'], '--', c = 'k', alpha=0.15 )
    ax.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color='lightskyblue')
    
    plt.title(title, fontsize = 18)
    plt.xlabel("time (ms)", fontsize = 10)
    plt.ylabel("firing rate (spk/s)", fontsize = 10)
    plt.legend(fontsize = 5)
    ax.tick_params(axis='both', which='major', labelsize=10)
def sweep_time_scales(GABA_A, GABA_B, Glut, dt, filename):

    STN_freq = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut)))
    GPe_freq = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut)))
    tau_mat = np.zeros((len(GABA_A)*len(GABA_B)*len(Glut),3))
    count = 0
    for gaba_b in GABA_B:
        for gaba_a in GABA_A:
            for glut in Glut:
                
                GPe.tau = {'GABA-A' : gaba_a, 'GABA-B' : gaba_b}
                STN.tau = {'Glut': glut} 
    
                run()
                tau_mat[count,:] = [gaba_a, gaba_b, glut]
                sig_STN = STN.pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(STN.pop_act[duration_mvt[0]:duration_mvt[1]])
#                STN_freq[count] = freq_from_welch(sig_STN[cut_plateau(sig_STN)],dt/1000)
                STN_freq[count] = freq_from_fft(sig_STN[cut_plateau(sig_STN)],dt/1000)

                sig_GPe = GPe.pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(GPe.pop_act[duration_mvt[0]:duration_mvt[1]])
#                GPe_freq [count] = freq_from_welch(sig_GPe[cut_plateau(sig_GPe)],dt/1000)
                GPe_freq [count] = freq_from_fft(sig_GPe[cut_plateau(sig_GPe)],dt/1000)

                count +=1
                print(count, "from ", len(GABA_A)*len(GABA_B)*len(Glut))
    
    
    np.savez(filename, tau_mat = tau_mat, GPe = GPe_freq, STN = STN_freq )
    
def sweep_time_scales_one_GABA(inhibitory_trans,inhibitory_series, Glut, dt, filename):

    STN_freq = np.zeros((len(inhibitory_series)*len(Glut)))
    GPe_freq = np.zeros((len(inhibitory_series)*len(Glut)))
    tau_mat = np.zeros((len(inhibitory_series)*len(Glut),2))
    count = 0

    for gaba in inhibitory_series:
        for glut in Glut:
            
            GPe.tau = {inhibitory_trans : gaba}
            STN.tau = {'Glut': glut} 

            run()
            tau_mat[count,:] = [gaba, glut]
            sig_STN = STN.pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(STN.pop_act[duration_mvt[0]:duration_mvt[1]])
#                STN_freq[count] = freq_from_welch(sig_STN[cut_plateau(sig_STN)],dt/1000)
            STN_freq[count] = freq_from_fft(sig_STN[cut_plateau(sig_STN)],dt/1000)

            sig_GPe = GPe.pop_act[duration_mvt[0]:duration_mvt[1]] - np.average(GPe.pop_act[duration_mvt[0]:duration_mvt[1]])
#                GPe_freq [count] = freq_from_welch(sig_GPe[cut_plateau(sig_GPe)],dt/1000)
            GPe_freq [count] = freq_from_fft(sig_GPe[cut_plateau(sig_GPe)],dt/1000)

            count +=1
            print(count, "from ", len(inhibitory_series)*len(Glut))

    
    np.savez(filename, tau_mat = tau_mat, GPe = GPe_freq, STN = STN_freq )
    



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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ind = np.logical_and(c<=c_upper_limit, c>=c_lower_limit)
#    print(ind)
    img = ax.scatter(x[ind], y[ind], z[ind], c=c[ind], cmap=plt.hot(),lw = 1,edgecolor = 'k')
    
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_zlabel(label[2])
    ax.set_title(title)
    clb = fig.colorbar(img)
    clb.set_label(label[3], labelpad=-40, y=1.05, rotation=0)
    plt.show()
#    fig = plt.figure(figsize=(6,6))
#    ax = Axes3D(fig)
#    surf = ax.plot_trisurf(x[ind],y[ind],z[ind], cmap = cm.coolwarm)
#    ax.set_xlabel('GABA-A')
#    ax.set_ylabel('GABA-B')
#    ax.set_zlabel('Glut')
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#build_connection_matrix(4,10,2)
   
def max_non_empty_array(array):
    if len(array) == 0:
        return 0
    else:
        return np.max(array)
    

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
            
            run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, J, K, threshold, gain)
            
            x1_gp_mvt = np.argmax(GPe.pop_act[int(t_mvt/dt):int((t_mvt+D_mvt)/dt)])+int(t_mvt/dt)
            x1_stn_mvt = np.argmax(STN.pop_act[int(t_mvt/dt):int((t_mvt+D_mvt)/dt)])+int(t_mvt/dt)
            x2_mvt = duration_mvt[1]
            x1_gp_base = np.argmax(GPe.pop_act[0:int(t_mvt/dt)])
            x1_stn_base = np.argmax(STN.pop_act[0:int(t_mvt/dt)])
            x2_base = duration_base[1]
            sig_STN_mvt = STN.pop_act[x1_stn_mvt:x2_mvt] - np.average(STN.pop_act[x1_stn_mvt:x2_mvt])
            cut_sig_ind_mvt = cut_plateau(sig_STN_mvt)
            
            STN_prop[('perc_t_oscil_mvt')][count] = max_non_empty_array(cut_sig_ind_mvt)/len(sig_STN_mvt)*100
            STN_prop[('mvt_f')][count] = freq_from_fft(sig_STN_mvt[cut_sig_ind_mvt],dt/1000)

            sig_STN_base = STN.pop_act[x1_stn_base:x2_base] - np.average(STN.pop_act[x1_stn_base:x2_base])
            cut_sig_ind_base = cut_plateau(sig_STN_base)
            STN_prop[('perc_t_oscil_base')][count] = max_non_empty_array(cut_sig_ind_base)/len(sig_STN_base)*100
            STN_prop[('base_f')][count] = freq_from_fft(sig_STN_base[cut_sig_ind_base],dt/1000)

            sig_GPe_mvt = GPe.pop_act[x1_gp_mvt:x2_mvt] - np.average(GPe.pop_act[x1_gp_mvt:x2_mvt])
            cut_sig_ind_mvt = cut_plateau(sig_GPe_mvt)
            GPe_prop[('perc_t_oscil_mvt')][count] = max_non_empty_array(cut_sig_ind_mvt)/len(sig_GPe_mvt)*100
            GPe_prop[('mvt_f')][count] = freq_from_fft(sig_GPe_mvt[cut_sig_ind_mvt],dt/1000)

            sig_GPe_base = GPe.pop_act[x1_gp_base:x2_base] - np.average(GPe.pop_act[x1_gp_base:x2_base])
            cut_sig_ind_base = cut_plateau(sig_GPe_base)
            GPe_prop[('perc_t_oscil_base')][count] = max_non_empty_array(cut_sig_ind_base)/len(sig_GPe_base)*100
            GPe_prop[('base_f')][count] = freq_from_fft(sig_GPe_base[cut_sig_ind_base],dt/1000)
            
            ax = fig.add_subplot(n,m,count+1)
            plot(GPe, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,[fig, ax], title = r"$G_{STN-GPe}$ = "+ str(round(g_inh,2))+r' $G_{GPe-STN}$ ='+str(round(g_exit,2)), n_subplots = int(n*m))
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

#%%  
#G = { ('STN', 'GPe'): g_mat[15,1] ,
#  ('GPe', 'STN'): g_mat[15,0], 
#  ('GPe', 'GPe'): 0} # synaptic weight
##G[('GPe', 'GPe')] = 0.5* G[('STN', 'GPe')]

K = calculate_number_of_connections(N,N_real,K_real)
GPe = Nucleus(1, N, A, 'GPe', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input)
#GPe = Nucleus(N, A, 'GPe', T, t_sim, dt, tau, ['GABA-A','GABA-B'], rest_ext_input)
STN = Nucleus(1, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input)
receiving_class_dict = {('STN','1'): [GPe], ('GPe','1'): [ GPe, STN]} # points to the classes of nuclei each receive projections from
#run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, J, K, threshold, gain)

#plot(GPe, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
n_inh = 10 ; n_exit = 10
g_inh_list = np.linspace(-3, -0.1, n_inh)
g_exit_list = np.linspace(0.1, 3, n_exit)
g_mat, GPe_prop, STN_prop = synaptic_weight_space_exploration(g_inh_list, g_exit_list, GPe, STN, duration_mvt, duration_base)
#scatter_3d_plot(g_mat[:,0],g_mat[:,1],STN_prop['perc_t_oscil_base'],STN_prop['perc_t_oscil_base'], 'STN', np.max(STN_prop['perc_t_oscil_base']), np.min(STN_prop['perc_t_oscil_base']), ['GPe-STN', 'STN-GPe', '% basal oscillatory period',  '% basal oscillatory period'])
#scatter_3d_plot(g_mat[:,0],g_mat[:,1],STN_prop['perc_t_oscil_mvt'],STN_prop['perc_t_oscil_mvt'], 'STN', np.max(STN_prop['perc_t_oscil_mvt']), np.min(STN_prop['perc_t_oscil_mvt']), ['GPe-STN', 'STN-GPe', '% mvt oscillatory period', '% mvt oscillatory period'])
#scatter_3d_plot(g_mat[:,0],g_mat[:,1],GPe_prop['perc_t_oscil_mvt'],GPe_prop['perc_t_oscil_mvt'], 'GPe', np.max(GPe_prop['perc_t_oscil_mvt']), np.min(GPe_prop['perc_t_oscil_mvt']), ['GPe-STN', 'STN-GPe', '% mvt oscillatory period', '% mvt oscillatory period'])
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
n = 8
GABA_A = np.linspace(5,20,n)
GABA_B = np.linspace(150,300,4)
Glut = np.linspace(0.5,12,n)
sweep_time_scales(GABA_A, GABA_B, Glut, dt, 'data_GABA_A_B_Glut.npz')
file = np.load('data_GABA_A_B_Glut.npz')
x = file['tau_mat'][:,0]
y = file['tau_mat'][:,1]
z = file['tau_mat'][:,2]

scatter_3d_plot(x,y,z, file['STN'],'STN', np.max(file['STN']), np.min(file['STN']),['GABA_A','GABA_B','Glut'])


sweep_time_scales_one_GABA('GABA_A', GABA_A, Glut, dt, 'data_GABA_A.npz')
file = np.load('data_GABA_A.npz')
x = file['tau_mat'][:,0]
y = file['tau_mat'][:,1]
z = file['STN']
scatter_3d_plot(x,y,z, file['STN'],'STN', np.max(file['STN']), np.min(file['STN']),['GABA_A','Glut','freq'])

z = file['GPe']
scatter_3d_plot(x,y,z, file['GPe'],'GPe', np.max(z), np.min(z),['GABA_A','Glut','freq'])

GABA_B = np.linspace(150,300,8)
sweep_time_scales_one_GABA('GABA_B', GABA_B, Glut, dt, 'data_GABA_B_Glut.npz')
file = np.load('data_GABA_B_Glut.npz')
x = file['tau_mat'][:,0]
y = file['tau_mat'][:,1]
z = file['STN']
scatter_3d_plot(x,y,z, file['STN'],'STN', np.max(file['STN']), np.min(file['STN']),['GABA_B','Glut','freq'])

z = file['GPe']
scatter_3d_plot(x,y,z, file['GPe'],'GPe', np.max(z), np.min(z),['GABA_B','Glut','freq'])

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
    


    

    

    
    
    
    
    