import numpy as np
import matplotlib.pyplot as plt
import timeit
from numpy.fft import rfft,fft, fftfreq
from scipy import signal
from tempfile import TemporaryFile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
import pickle
#from scipy.ndimage.filters import generic_filter

class Nucleus:

    def __init__(self, population_number,gain, threshold, noise_variance, noise_amplitude, N, A, name, G, T, t_sim, dt, tau, trans_types, rest_ext_input, receiving_from_dict,smooth_kern_window,oscil_peak_threshold):
        
        self.n = N[name] # population size
        self.population_num = population_number
        self.name = name
        self.basal_firing = A[name]
        self.mvt_firing = A_mvt[name]
        self.threshold = threshold[name]
        self.gain = gain[name]
        self.trans_types =  trans_types
        self.tau = dictfilt(tau, self.trans_types) # synaptic time scale based on neuron type
        self.transmission_delay = {k: v for k, v in T.items() if k[0]==name} # filter based on the receiving nucleus
        self.synaptic_weight = {k: v for k, v in G.items() if k[0]==name} # filter based on the receiving nucleus
        self.history_duration = max(self.transmission_delay.values()) # stored history in ms derived from the longest transmission delay of the projections
        self.output = np.zeros((self.n,int(self.history_duration/dt)))
        self.input = np.zeros((self.n))
        self.neuron_act = np.zeros((self.n))
        self.pop_act = np.zeros((int(t_sim/dt))) # time series of population activity
        self.receiving_from_dict = receiving_from_dict[(self.name, str(self.population_num))] 
#        self.max_delay = 
#        self.rest_ext_input = rest_ext_input[name]
        self.rest_ext_input = None
        self.mvt_ext_input = np.zeros((int(t_sim/dt))) # external input mimicing movement
        self.external_inp_t_series = np.zeros((int(t_sim/dt)))
        self.avg_pop_act_mvt = None
        self.avg_pop_act_base = None
        self.noise_variance =  noise_variance[self.name ] #additive gaussian white noise with mean zero and variance sigma
        self.noise_amplitude =  noise_amplitude[self.name ] #additive gaussian white noise with mean zero and variance sigma
        self.connectivity_matrix = {}
        self.noise_induced_basal_firing = None # the basal firing as a result of noise at steady state
        self.oscil_peak_threshold = oscil_peak_threshold[self.name]
        self.smooth_kern_window = smooth_kern_window[self.name]
        
    def calculate_input_and_inst_act(self, K, t, dt, receiving_from_class_list, mvt_ext_inp):  
        
        syn_inputs = np.zeros((self.n,1)) # = Sum (G Jxm)
        for projecting in receiving_from_class_list:
#            print(np.matmul(J[(self.name, projecting.name)], projecting.output[:,int(-T[(self.name,projecting.name)]*dt)].reshape(-1,1)).shape)
            syn_inputs += self.synaptic_weight[(self.name, projecting.name)]*np.matmul(self.connectivity_matrix[(projecting.name,str(projecting.population_num))], 
                           projecting.output[:,-int(self.transmission_delay[(self.name,projecting.name)]/dt)].reshape(-1,1))/K[(self.name, projecting.name)]
        
#        print((syn_inputs + self.rest_ext_input  + mvt_ext_inp)[0] )
#        print("noise", noise_generator(self.noise_amplitude, self.noise_variance, self.n)[0])
        self.input = syn_inputs + self.rest_ext_input  + mvt_ext_inp #+ noise_generator(self.noise_amplitude, self.noise_variance, self.n)
        self.neuron_act = transfer_func(self.threshold, self.gain, self.input)
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
    def clear_history(self):
        self.output = np.zeros_like(self.output)
        self.input = np.zeros_like(self.input)
        self.neuron_act = np.zeros_like(self.neuron_act)
        self.pop_act = np.zeros_like(self.pop_act) 
        self.mvt_ext_input = np.zeros_like(self.mvt_ext_input) 
        self.external_inp_t_series = np.zeros_like(self.external_inp_t_series) 
        
    def set_synaptic_weights(self):
        self.synaptic_weight = {k: v for k, v in G.items() if k[0]==self.name} # filter based on the receiving nucleus
    def set_ext_input(self,A):
        proj_list = [k[1] for k in list(self.synaptic_weight.keys())]
        self.rest_ext_input = self.basal_firing/self.gain - np.average([self.synaptic_weight[self.name,proj]*A[proj] for proj in proj_list]) + self.threshold
#        [k[0] for k in list(Proto[0].synaptic_weight.keys())]
def set_rest_ext_inp(A, gain, G, threshold):
    ''' find the amount of external input needed to obtain rest level of firing rate'''
    rest_ext_input = { 'STN': A['STN']/gain['STN']-G[('STN', 'Proto')]*A['Proto'] + threshold['STN'] ,
                   'Proto': A['Proto']/gain['Proto']-(G[('Proto', 'STN')]*A['STN']*2 + G[('Proto', 'Proto')]*A['Proto']) + threshold['Proto']} # <double pop> external input coming from Ctx and Str
    return rest_ext_input

def set_mvt_ext_inp(A_mvt, gain, G, threshold, rest_ext_input):
    ''' find the amount of external input needed to go from rest level of firing rate to firing rate at movement time'''
    mvt_ext_input_dict = { 'STN': A_mvt['STN']/gain['STN']-G[('STN', 'Proto')]*A_mvt['Proto'] + threshold['STN'] -rest_ext_input['STN'],
                   'Proto' : A_mvt['Proto']/gain['Proto']-(G[('Proto', 'STN')]*A_mvt['STN']*2 + G[('Proto', 'Proto')]*A_mvt['Proto']) + threshold['Proto'] -rest_ext_input['Proto']} # external input coming
    return mvt_ext_input_dict
     
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

def run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict):
    
    
    delay = {'Proto': T[('Proto', 'D2')], 'STN': T[('STN', 'Ctx')]}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.set_connections(K, N)
            nucleus.external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,delay[nucleus.name],mvt_ext_input_dict[nucleus.name], t_list*dt)

     
    start = timeit.default_timer()
    
    for t in t_list:
        for nuclei_list in nuclei_dict.values():
            k = 0
            for nucleus in nuclei_list:
                k += 1
        #        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement 
                mvt_ext_inp = np.ones((nucleus.n,1))*nucleus.external_inp_t_series[t] # movement added 
                nucleus.calculate_input_and_inst_act(K, t, dt, receiving_class_dict[(nucleus.name,str(k))], mvt_ext_inp)
                nucleus.update_output()

    stop = timeit.default_timer()
#    print("t = ", stop - start)
    return STN,Proto
    
def plot( Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_ob, title = "", n_subplots = 1):    
    plot_start =0# int(5/dt)
    if plot_ob == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot_ob
     
    line_type = ['-', '--']
    
    ax.plot(t_list[plot_start:]*dt,Proto[0].pop_act[plot_start:], line_type[0], label = "Proto" , c = 'r',lw = 1.5)
    ax.plot(t_list[plot_start:]*dt,STN[0].pop_act[plot_start:], line_type[0],label = "STN", c = 'k',lw = 1.5)
    
    ax.plot(t_list[plot_start:]*dt,Proto[1].pop_act[plot_start:], line_type[1], c = 'r',lw = 1.5)
    ax.plot(t_list[plot_start:]*dt,STN[1].pop_act[plot_start:], line_type[1], c = 'k',lw = 1.5)
    
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A['Proto'], '-.', c = 'r',lw = 1, alpha=0.8 )
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt['Proto'], '-.', c = 'r', alpha=0.2,lw = 1 )
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A['STN'], '-.', c = 'k', alpha=0.8,lw = 1 )
    ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt['STN'], '-.', c = 'k', alpha=0.2,lw = 1 )
    ax.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color='lightskyblue')
    
    plt.title(title, fontsize = 18)
    plt.xlabel("time (ms)", fontsize = 10)
    plt.ylabel("firing rate (spk/s)", fontsize = 10)
    plt.legend(fontsize = 10)
    ax.tick_params(axis='both', which='major', labelsize=10)

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

def scatter_3d_plot(x,y,z,c, title, c_upper_limit, c_lower_limit, label, limits = None):

    ind = np.logical_and(c<=c_upper_limit, c>=c_lower_limit)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x[ind], y[ind], z[ind], c=c[ind], cmap=plt.hot(),lw = 1,edgecolor = 'k')
#    X,Y = np.meshgrid(x[ind],y[ind])
#    print(X)
#    ax.plot_wireframe(X, Y, Z, c = 'k')
#    ax = Axes3D(fig)
#    surf = ax.plot_trisurf(x[ind],y[ind],z[ind], cmap = cm.coolwarm)
#    fig.colorbar(surf, shrink=0.5, aspect=5)
    if limits == None:
        limits = {'x':(min(x),max(x)), 'y':(min(y),max(y)), 'z':(min(z),max(z))}
        
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_zlabel(label[2])
    ax.set_title(title)
    ax.set_xlim(limits['x'])
    ax.set_ylim(limits['y'])
    ax.set_zlim(limits['z'])
    clb = fig.colorbar(img)
    clb.set_label(label[3], labelpad=-40, y=1.05, rotation=0)
    plt.show()
    
def scatter_3d_wireframe_plot(x,y,z,c, title, label, limits = None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color = 'grey', lw = 0.5 )
    img = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=c.flatten(), cmap=plt.hot(), s = 20, lw = 1,edgecolor = 'k')

    if limits == None:
        limits = {'x':(np.amin(x),np.amax(x)), 'y':(np.amin(y),np.amax(y)), 'z':(np.amin(z),np.max(z))}
        
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_zlabel(label[2])
    ax.set_title(title)
    ax.set_xlim(limits['x'])
    ax.set_ylim(limits['y'])
    ax.set_zlim(limits['z'])
    clb = fig.colorbar(img)
    clb.set_label(label[3], labelpad=-40, y=1.05, rotation=0)
    plt.show()
#build_connection_matrix(4,10,2)
   
def max_non_empty_array(array):
    if len(array) == 0:
        return 0
    else:
        return np.max(array)
    

     
    
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
    
def zero_crossing_freq_detect(sig,dt):
    ''' detect frequency from zero crossing distances'''
    zero_crossings = np.where(np.diff(np.sign(sig)))[0] # indices to elements before a crossing
#    print(zero_crossings)
    shifted = np.roll(zero_crossings,-1)
    
    half_lambda =  shifted[:-1] - zero_crossings[:-1]
#    print(half_lambda)
    n_half_cycles = len(half_lambda)
    if n_half_cycles > 1:
        frequency = 1/(np.average(half_lambda)*2*dt)
    else: 
        frequency = 0
    return n_half_cycles, frequency

def find_mean_of_signal(sig):
    ''' find the value which when subtracted the signal oscillates around zero'''
    cut_plateau_ind = cut_plateau(sig)
    len_not_plateau = len(cut_plateau_ind)
    if len_not_plateau > 0 and len_not_plateau < len(sig)*4/5:
        plateau_y = np.average(sig[max(cut_plateau_ind):])  
    else:
        plateau_y = np.average(sig)
    return cut_plateau_ind, plateau_y

def trim_start_end_sig_rm_offset(sig,start, end):
    ''' trim with max point at the start and the given end point'''
    trimmed = sig[start:end]
    _, plateau_y = find_mean_of_signal(trimmed)
    trimmed = trimmed - plateau_y # remove the offset determined with plateau level
#    print((sig - plateau_y)[max(cut_plateau(sig)):])
#    max_point = np.argmax(trimmed)
    max_value = np.max(trimmed)
    min_value = np.min(trimmed)
#    plt.figure()
#    plt.plot(trimmed)
    if abs(max_value) > abs(min_value):
#        print('max')
        max_point = np.max(np.where(trimmed == max_value))
    else:
        max_point = np.max(np.where(trimmed == min_value))
#    print('x_max = ',max_point)
    if max_point>len(trimmed)/2: # in case the amplitude increases over time take the whole signal #############################3 to do --> check for increasing amplitude in oscillations instead of naive implementation here
        return trimmed- np.average(trimmed)
    else:
        return(trimmed[max_point:] - np.average(trimmed[max_point:]))
#    return trimmed
    
def find_freq_of_pop_act_spec_window(nucleus,start, end):
    ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
    the plateau and in case it is oscillation determine the frequency '''
    sig = trim_start_end_sig_rm_offset(nucleus.pop_act,start, end)
    cut_sig_ind, plateau_y = find_mean_of_signal(sig)
#    print(cut_sig_ind)
#    plt.figure()
#    plt.plot(sig)
#    plt.plot(sig[cut_sig_ind])
    if len(cut_sig_ind) > 0: # if it's not all plateau from the beginning
        sig = sig - plateau_y
#        if if_oscillatory(sig, max(cut_sig_ind),nucleus.oscil_peak_threshold, nucleus.smooth_kern_window): # then check if there's oscillations
#            perc_oscil = max_non_empty_array(cut_sig_ind)/len(sig)*100
##            freq = freq_from_fft(sig[cut_sig_ind],dt/1000)
#            _,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)
#            return perc_oscil, freq
#        else:
#            return 0,0
        n_half_cycles,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)
#        print("freq ",len(cut_sig_ind), freq)
        if freq != 0: # then check if there's oscillations
            perc_oscil = max_non_empty_array(cut_sig_ind)/len(sig)*100            
            return n_half_cycles, perc_oscil, freq
        else:
            return 0,0,0
    else:
        return 0,0,0
        
def if_oscillatory(sig, x_plateau, peak_threshold, smooth_kern_window):
    ''' detect if there are peaks with larger amplitudes than noise in mean subtracted data before plateau'''
#    fluctuations = gaussian_filter1d(sig[:x_plateau],smooth_kern_window)
    peaks,_ = signal.find_peaks(sig, height = peak_threshold)
    troughs,_ = signal.find_peaks(-sig, height = peak_threshold)
#    plt.figure()
#    plt.plot(sig)
#    plt.axvline(x_plateau)
#    plt.plot(fluctuations, label = "Gaus kern smoothed")
#    plt.plot(peaks,sig[peaks],"x", markersize = 10,markeredgewidth = 2)
#    plt.plot(troughs,sig[troughs],"x", markersize = 10, markeredgewidth = 2)
#    plt.legend()
    if len(peaks)>0 and len(troughs)>0: # to have at least one maxima and one minima to count as oscillation
        return True
    else:
        return False
def temp_oscil_check(sig_in,peak_threshold, smooth_kern_window,start,end):
    def if_oscillatory(sig, x_plateau, peak_threshold, smooth_kern_window):
        ''' detect if there are peaks with larger amplitudes than noise in mean subtracted data before plateau'''
        fluctuations = gaussian_filter1d(sig[:x_plateau],smooth_kern_window)
        peaks,_ = signal.find_peaks(sig, height = peak_threshold)
        troughs,_ = signal.find_peaks(-sig, height = peak_threshold)
        plt.figure()
        plt.plot(sig)
#        plt.axhline(4*10**-4)
        plt.axvline(x_plateau)
        plt.plot(fluctuations, label = "Gaus kern smoothed")
        plt.plot(peaks,sig[peaks],"x", markersize = 10,markeredgewidth = 2)
        plt.plot(troughs,sig[troughs],"x", markersize = 10, markeredgewidth = 2)
        plt.legend()
        if len(peaks)>0 and len(troughs)>0: # to have at least one maxima and one minima to count as oscillation
            return True
        else:
            return False
    
    sig = trim_start_end_sig_rm_offset(sig_in,start,end)
    cut_sig_ind, plateau_y = find_mean_of_signal(sig)
    if len(cut_sig_ind) > 0: # if it's not all plateau from the beginning
        sig = sig - plateau_y
        if_oscillatory(sig, max(cut_sig_ind),peak_threshold, smooth_kern_window)
#        if if_oscillatory(sig, max(cut_sig_ind),peak_threshold, smooth_kern_window): # then check if there's oscillations
#            perc_oscil = max_non_empty_array(cut_sig_ind)/len(sig)*100
#            _,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)
#            freq = freq_from_fft(sig[cut_sig_ind],dt/1000)
#        else:
#            print("all plateau")
#        plt.figure()
#        plt.plot(sig)
        _,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)
        if freq != 0: # then check if there's oscillations
            perc_oscil = max_non_empty_array(cut_sig_ind)/len(sig)*100            
            print("% = ", perc_oscil, "f = ",freq)
        else:
            print("all plateau")

    else:
        print("no oscillation")
   
   
def sweep_time_scales(Proto,STN,GABA_A, GABA_B, Glut, dt, filename,mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain):

    data = {('STN','mvt_freq'): np.zeros((len(GABA_A)*len(GABA_B)*len(Glut))), ('STN','base_freq'): np.zeros((len(GABA_A)*len(GABA_B)*len(Glut))),
            ('Proto','mvt_freq'): np.zeros((len(GABA_A)*len(GABA_B)*len(Glut))), ('Proto','base_freq'): np.zeros((len(GABA_A)*len(GABA_B)*len(Glut))),
            ('STN','pop_act'): np.zeros((len(GABA_A)*len(GABA_B)*len(Glut),len(t_list))), ('Proto','pop_act'): np.zeros((len(GABA_A)*len(GABA_B)*len(Glut),len(t_list))),
            'tau':np.zeros((len(GABA_A)*len(GABA_B)*len(Glut),3))}
    count = 0
    for gaba_b in GABA_B:
        for gaba_a in GABA_A:
            for glut in Glut:
                for k in range (len(Proto)):
                    Proto[k].tau = {'GABA-A' : gaba_a, 'GABA-B' : gaba_b}
                    STN[k].tau = {'Glut': glut} 
                
                nuclei_dict = {'Proto': Proto, 'STN' : STN}
                run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
                data['tau'][count,:] = [gaba_a, gaba_b, glut]
                nucleus_list =[ Proto[0], STN[0]]
#                plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$\tau_{GABA_A}$ = "+ str(round(gaba_a,2))+r' $\tau_{GABA_B}$ ='+str(round(gaba_b,2)))
                for nucleus in nucleus_list:
                    data[(nucleus.name,'pop_act')][count,:] = nucleus.pop_act
                    _,_, data[nucleus.name,'mvt_freq'][count] = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt)
                    _,_, data[nucleus.name,'base_freq'][count] = find_freq_of_pop_act_spec_window(nucleus,*duration_base)
                count +=1
                print(count, "from ", len(GABA_A)*len(GABA_B)*len(Glut))

    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()
    
def sweep_time_scales_one_GABA(g_list, Proto, STN, inhibitory_trans,inhibitory_series, G,Glut, dt, filename,mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain):

    data = {('STN','mvt_freq'): np.zeros((len(inhibitory_series)*len(Glut))), ('STN','base_freq'): np.zeros((len(inhibitory_series)*len(Glut))),
            ('Proto','mvt_freq'): np.zeros((len(inhibitory_series)*len(Glut))), ('Proto','base_freq'): np.zeros((len(inhibitory_series)*len(Glut))),
             ('STN','pop_act'): np.zeros((len(inhibitory_series)*len(Glut),len(t_list))) ,('Proto','pop_act'): np.zeros((len(inhibitory_series)*len(Glut),len(t_list))),
            'tau':np.zeros((len(inhibitory_series)*len(Glut),2))}
    count = 0

    for gaba in inhibitory_series:
        for glut in Glut[3:]:
            for k in range (len(Proto)):
                Proto[k].tau = {inhibitory_trans : gaba}
                STN[k].tau = {'Glut': glut} 
            print('GABA = ', gaba, 'glut = ', glut)
            nuclei_dict = {'Proto': Proto, 'STN' : STN}
            n_half_cycle,G, nuclei_dict = find_oscillation_boundary(g_list,nuclei_dict, A, A_mvt, G, T, K, N, receiving_class_dict, gain, threshold, D_mvt, t_mvt, duration_mvt, duration_base)
            rest_ext_inp_dict = set_rest_ext_inp(A, gain, G, threshold)
            mvt_ext_input_dict =  set_mvt_ext_inp(A_mvt, gain, G, threshold, rest_ext_inp_dict)

            run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
            data['tau'][count,:] = [gaba, glut]
            nucleus_list =[ Proto[0], STN[0]]
#                plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$\tau_{GABA_A}$ = "+ str(round(gaba_a,2))+r' $\tau_{GABA_B}$ ='+str(round(gaba_b,2)))
            for nucleus in nucleus_list:
                data[(nucleus.name,'pop_act')][count,:] = nucleus.pop_act
                _,_, data[nucleus.name,'mvt_freq'][count] = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt)
                _,_, data[nucleus.name,'base_freq'][count] = find_freq_of_pop_act_spec_window(nucleus,*duration_base)
            count +=1
            for nuclei_list in nuclei_dict.values():
                for nucleus in nuclei_list:
                    nucleus.clear_history()
            print(count, "from ", len(inhibitory_series)*len(Glut))

    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()
def synaptic_weight_space_exploration(A, gain, G, threshold, g_1_list, g_2_list, Proto, STN, duration_mvt, duration_base, receiving_class_dict, t_list):
    
    n = len(g_1_list)
    m = len(g_2_list)
    g_mat = np.zeros((n,m,3))
    STN_prop = {'base_f' : np.zeros((n,m)), 'mvt_f' : np.zeros((n,m)),
                'perc_t_oscil_base': np.zeros((n,m)), 'perc_t_oscil_mvt': np.zeros((n,m))}
    Proto_prop = {'base_f' : np.zeros((n,m)), 'mvt_f' : np.zeros((n,m)),
                'perc_t_oscil_base': np.zeros((n,m)), 'perc_t_oscil_mvt': np.zeros((n,m))}

    count  = 0
    fig = plt.figure()
    i = 0 
    for g_1 in g_1_list:
        j = 0
        for g_2 in g_2_list:
#            Proto.synaptic_weight[('Proto', 'STN')] = g_exit
            G[('STN','Proto')] = g_1
            G[('Proto','Proto')] = g_2
            rest_ext_inp_dict = set_rest_ext_inp(A, gain, G, threshold)
            for k in range (len(Proto)):
                STN[k].clear_history(); Proto[k].clear_history()
                STN[k].synaptic_weight[('STN','Proto')] = g_1 
                Proto[k].synaptic_weight[('Proto','Proto')] = g_2 
                STN[k].rest_ext_input = rest_ext_inp_dict['STN']
                Proto[k].rest_ext_input = rest_ext_inp_dict['Proto']
                
            mvt_ext_input_dict =  set_mvt_ext_inp(A_mvt, gain, G, threshold, rest_ext_inp_dict)
            nuclei_dict = {'Proto': Proto, 'STN' : STN}
            run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
            Proto_test = Proto[0] ; STN_test = STN[0]
            
            
            g_mat[i,j,:] = [Proto_test.synaptic_weight[('Proto', 'STN')], g_1, g_2]
            _,STN_prop[('perc_t_oscil_mvt')][i,j], STN_prop[('mvt_f')][i,j]= find_freq_of_pop_act_spec_window(STN_test,*duration_mvt)
            _,STN_prop[('perc_t_oscil_base')][i,j], STN_prop[('base_f')][i,j]= find_freq_of_pop_act_spec_window(STN_test,*duration_base)
            _,Proto_prop[('perc_t_oscil_mvt')][i,j], Proto_prop[('mvt_f')][i,j]= find_freq_of_pop_act_spec_window(Proto_test,*duration_mvt)
            _,Proto_prop[('perc_t_oscil_base')][i,j], Proto_prop[('base_f')][i,j]= find_freq_of_pop_act_spec_window(Proto_test,*duration_base)

            ax = fig.add_subplot(n,m,count+1)
            plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,[fig, ax], title = r"$G_{STN-Proto}$ = "+ str(round(g_1,2))+r' $G_{Proto-Proto}$ ='+str(round(g_2,2)), n_subplots = int(n*m))
            plt.title( r"$G_{STN-Proto}$ = "+ str(round(g_1,2))+r' $G_{Proto-Proto}$ ='+str(round(g_2,2)), fontsize = 5)
            plt.xlabel("time (ms)", fontsize = 5)
            plt.ylabel("firing rate (spk/s)", fontsize = 5)
            plt.legend(fontsize = 2)
            ax.tick_params(axis='both', which='major', labelsize=2)
            count +=1
            j+=1
            print(count, "from", int(m*n))
        i+=1
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    return g_mat, Proto_prop, STN_prop 

def find_oscillation_boundary(g_list,nuclei_dict, A, A_mvt, G, T, K, N, receiving_class_dict, gain, threshold, D_mvt, t_mvt, duration_mvt, duration_base):
    ''' find the synaptic strength for a given set of parametes where you oscillations appear after increasing external input'''
    got_it = False
    for g in g_list:
        G[('STN', 'Proto')] = g
        G[('Proto', 'Proto')] = g
        rest_ext_inp_dict = set_rest_ext_inp(A, gain, G, threshold)
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.set_synaptic_weights(G)
        mvt_ext_input_dict =  set_mvt_ext_inp(A_mvt, gain, G, threshold, rest_ext_inp_dict)
        nuclei_dict = {'Proto': Proto, 'STN' : STN}
        run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
        Proto_test = nuclei_dict['Proto'][0] ; STN_test = nuclei_dict['STN'][0]
        n_half_cycles_mvt,perc_oscil_mvt, f_mvt = find_freq_of_pop_act_spec_window(STN_test,*duration_mvt)
        n_half_cycles_base, perc_oscil_base, f_base = find_freq_of_pop_act_spec_window(STN_test,*duration_base)
        print('g=',round(g,1), round(G['Proto','STN'],2),round(f_base,1),n_half_cycles_base, round(f_mvt,1), n_half_cycles_mvt)
#        if n_half_cycles_base <=4 and  n_half_cycles_mvt >= 4:
#        plot(nuclei_dict['Proto'], nuclei_dict['STN'], dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
        if n_half_cycles_mvt >= 6 and n_half_cycles_mvt <= 10:

#            plot(nuclei_dict['Proto'], nuclei_dict['STN'], dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
            got_it = True
            print("Gotcha!")
            break
    if not got_it:
        a = 1/0
    return n_half_cycles_mvt, G, nuclei_dict
#%% Constants 
if 1:
    N_sim = 100
    population_list = ['STN', 'Proto']
    N_sub_pop = 2
    N = { 'STN': N_sim , 'Proto': N_sim, 'Arky': N_sim, 'FSI': N_sim, 'D2': N_sim, 'D1': N_sim, 'GPi': N_sim, 'Th': N_sim}
    # MSNs make up at least 95% of all striatal cells (Kemp and Powell, 1971)
    N_Str = 2.79*10**6 # Oorschot 1998
    N_real = { 'STN': 13560 , 'Proto': 46000*0.70, 'Arky':46000*0.25, 'GPi': 3200, 'Str': N_Str, 'D2': int(0.475*N_Str), 'D1': int(0.475*N_Str) , 'FSI': int(0.02*N_Str),  # Oorschot 1998 , FSI-MSN: (Gerfen et al., 2010; Tepper, 2010)
              'Th': 10000} # check to find 
    A = { 'STN': 15 , 'Proto': 30, 'Arky': 18, # De la Crompe (2020)
         'FSI': 12.5, # FSI average firing rates:10–15 Hz. 60–80 Hz during behavioral tasks(Berke et al., 2004; Berke, 2008) or 18.5 Hz Berke et al 2010?
         'D1': 1.1, 'D2': 1.1, #Berke et al. 2010
         'GPi':26} # Benhamou & Cohen (201)
    # mean firing rate from experiments
    A_DD = { 'STN': 0 , 'Proto': 0,
         'FSI': 0, # FSI average firing rates:10–15 Hz. 60–80 Hz during behavioral tasks(Berke et al., 2004; Berke, 2008) or 18.5 Hz Berke et al 2010?
         'D1': 6.6, 'D2': 6.6, # Kita & Kita. 2011
         'GPi':0} 
    A_mvt = { 'STN': 50 , 'Proto': 22, 'FSI': 70} # mean firing rate during movement from experiments
    threshold = { 'STN': .1 ,'Proto': .1}
    neuron_type = {'STN': 'Glut', 'Proto': 'GABA'}
    gain = { 'STN': 1 ,'Proto': 1}
    
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
               ('D2', 'D2'): 1,
     } 
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
               ('FSI', 'Th'): 1} # find
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
         ('Proto', 'Proto'): 0.5,
         ('D2', 'Ctx'): 0,
         ('D1', 'Ctx'): 0,
         ('D2','Proto'): 0,
         ('D2', 'FSI'): 0, 
         ('FSI', 'Proto'): 0,
         ('FSI', 'FSI'): 0,
         ('D2','D2'): 0,
         ('D2','D1'): 0,
         ('D1','D2'): 0,
         ('D1', 'D1'): 0,
         ('GPi', 'Proto'): 0,
         ('Th', 'GPi') : 0
         } # synaptic weight
    
    G[('D1', 'D1')] = 0.5* G[('D2', 'D2')]
    G_DD = {('STN', 'Proto'): -3 ,
          ('Proto', 'STN'): 0.8 , 
          ('Proto', 'Proto'): 0, # become stronger (Bugaysen et al., 2013) 
          ('Str', 'Ctx'): 0,
          ('D2','Proto'): G[('D2','Proto')]*108/28} # IPSP amplitude in Ctr: 28pA, in DD: 108pA Corbit et al. (2016) [Is it due to increased connections or increased synaptic gain?]
    G_DD[('Proto', 'Proto')] = 0.5* G_DD[('STN', 'Proto')]
    
    tau = {'GABA-A' : 6, 'GABA-B': 200, 'Glut': 3.5} # Gerstner. synaptic time scale for excitation and inhibition
    noise_variance = {'Proto' : 0.1, 'STN': 0.1}
    noise_amplitude = {'Proto' : 10, 'STN': 10}
    oscil_peak_threshold = {'Proto' : 0.1, 'STN': 0.1}
    smooth_kern_window = {key: value * 30 for key, value in noise_variance.items()}
    #oscil_peak_threshold = {key: (gain[key]*noise_amplitude[key]*noise_variance[key]-threshold[key])/5 for key in noise_variance.keys()}
    #rest_ext_input = { 'STN': A['STN']/gain['STN']-G[('STN', 'Proto')]*A['Proto'] + threshold['STN'] ,
    #                   'Proto': A['Proto']/gain['Proto']-(G[('Proto', 'STN')]*A['STN'] + G[('Proto', 'Proto')]*A['Proto']) + threshold['Proto']} #  <Single pop> external input coming from Ctx and Str
    
    #mvt_ext_input_dict = { ('STN', '1'): A_mvt['STN']/gain['STN']-G[('STN', 'Proto')]*A_mvt['Proto'] + threshold['STN'] -rest_ext_input['STN'],
    #                   ('Proto', '1') : A_mvt['Proto']/gain['Proto']-(G[('Proto', 'STN')]*A_mvt['STN'] + G[('Proto', 'Proto')]*A_mvt['Proto']) + threshold['Proto'] -rest_ext_input['Proto']} # <single pop> external input coming from Ctx and Str
    
    rest_ext_input = { 'STN': A['STN']/gain['STN']-G[('STN', 'Proto')]*A['Proto'] + threshold['STN'] ,
                       'Proto': A['Proto']/gain['Proto']-(G[('Proto', 'STN')]*A['STN']*2 + G[('Proto', 'Proto')]*A['Proto']) + threshold['Proto']} # <double pop> external input coming from Ctx and Str
    
    mvt_ext_input_dict = { ('STN', '1'): A_mvt['STN']/gain['STN']-G[('STN', 'Proto')]*A_mvt['Proto'] + threshold['STN'] -rest_ext_input['STN'],
                       ('Proto', '1') : A_mvt['Proto']/gain['Proto']-(G[('Proto', 'STN')]*A_mvt['STN']*2 + G[('Proto', 'Proto')]*A_mvt['Proto']) + threshold['Proto'] -rest_ext_input['Proto']} # external input coming from Ctx and Str
    mvt_ext_input_dict[('STN', '2')] = mvt_ext_input_dict[('STN', '1')] ; mvt_ext_input_dict[('Proto', '2')] = mvt_ext_input_dict[('Proto', '1')]  
    pert_val = 10
    mvt_selective_ext_input_dict = {('Proto','1') : pert_val, ('Proto','2') : -pert_val,
                                    ('STN','1') : pert_val, ('STN','2') : -pert_val} # external input coming from Ctx and Str
    dopamine_percentage = 100
    t_sim = 400 # simulation time in ms
    dt = 0.5 # euler time step in ms
    t_mvt = 200
    D_mvt = 200
    D_perturb = 250 # transient selective perturbation
    d_Str = 200 # duration of external input to Str
    t_list = np.arange(int(t_sim/dt))
    duration_mvt = [int((t_mvt+ max(T[('Proto', 'D2')],T[('STN', 'Ctx')]))/dt), int((t_mvt+D_mvt)/dt)]
    duration_base = [int((max(T[('Proto', 'STN')],T[('STN', 'Proto')]))/dt), int(t_mvt/dt)]

#%% STN-Proto network
G = { ('STN', 'Proto'): -2.1,
  ('Proto', 'STN'): 0.5, 
  ('Proto', 'Proto'): -2.1 } # synaptic weight

K = calculate_number_of_connections(N,N_real,K_real_STN_Proto_diverse)

#K = calculate_number_of_connections(N,N_real,K_real)
receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
rest_ext_input = set_rest_ext_inp(A, gain, G, threshold) 
mvt_ext_input_dict = set_mvt_ext_inp(A_mvt, gain, G, threshold, rest_ext_input)
Proto = [Nucleus(1, gain, threshold, noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
STN = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
nuclei_dict = {'Proto': Proto, 'STN' : STN}

for k in range (len(Proto)):
    Proto[k].tau = {'GABA-A' : 5}
    STN[k].tau = {'Glut': 9.125} 

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

run(mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
plot(nuclei_dict['Proto'], nuclei_dict['STN'], dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
#g_list = np.linspace(-.6,-0.1, 20)
#n_half_cycle, G, nuclei_dict = find_oscillation_boundary(g_list,nuclei_dict, A, A_mvt, G, T, K, N, receiving_class_dict, gain, threshold, D_mvt, t_mvt, duration_mvt, duration_base)

print(find_freq_of_pop_act_spec_window(STN[0],*duration_mvt))
temp_oscil_check(nuclei_dict['STN'][0].pop_act,oscil_peak_threshold['STN'], 3,*duration_mvt)
temp_oscil_check(nuclei_dict['STN'][0].pop_act,oscil_peak_threshold['STN'], 3,*duration_base)

#temp_oscil_check(nuclei_dict['Proto'][0].pop_act,oscil_peak_threshold['Proto'], 3,*duration_base)
#plt.title(r"$\tau_{GABA_A}$ = "+ str(round(x[n_plot],2))+r' $\tau_{GABA_B}$ ='+str(round(y[n_plot],2))+ r' $\tau_{Glut}$ ='+str(round(z[n_plot],2))+' f ='+str(round(c[n_plot],2)) , fontsize = 10)


#%% Proto-FSI-D2 network
receiving_pop_list = {('FSI','1') : [('Proto', '1')], ('FSI','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('D2', '1')],
                    ('Proto','2') : [('Proto', '2'), ('D2', '1')],
                    ('D2','1') : [('FSI','1')], ('D2','2') : [('FSI','2')]}
K = calculate_number_of_connections(N,N_real,K_real_STN_Proto_diverse)

Proto = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
D2 = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'D2', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'D2', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
FSI = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'FSI', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'FSI', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
nuclei_dict = {'Proto': Proto, 'D2' : D2}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

run(mvt_selective_ext_input_dict, D_perturb,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
#%% synaptic weight phase exploration
n_1 = 10 ; n_2 = 10
g_inh_1_list = np.linspace(-2, -0.01, n_1)
g_inh_2_list = np.linspace(-2, -0.01, n_2)
#g_inh_1_list = [g_inh_1_list[0]] ; g_inh_2_list = [g_inh_2_list[2]]
G[('Proto','STN')] = 0.5

receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
K = calculate_number_of_connections(N,N_real,K_real_STN_Proto_diverse)

Proto = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
STN = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
nuclei_dict = {'Proto': Proto, 'STN' : STN}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

g_mat, Proto_prop, STN_prop = synaptic_weight_space_exploration(A, gain, G, threshold, g_inh_1_list, g_inh_2_list, Proto, STN, duration_mvt, duration_base,receiving_class_dict, t_list)
param = 'perc_t_oscil_mvt' #mvt_f'
freq = 'mvt_f'
scatter_3d_wireframe_plot(g_mat[:,:,1],g_mat[:,:,2],STN_prop[param],STN_prop[freq], 'STN', ['STN-Proto', 'Proto-Proto', param,  'frequency'])
scatter_3d_wireframe_plot(g_mat[:,:,1],g_mat[:,:,2],Proto_prop[param],Proto_prop[freq], 'Proto', ['STN-Proto', 'Proto-Proto', param, 'frequency'])



#%% # time scale parameter space with frequency of transient oscillations at steady state

receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
#receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
#                    ('Proto','1') : [('Proto', '1'), ('STN', '1')],
 #                   ('Proto','2') : [('Proto', '2'), ('STN', '2')]}
K = calculate_number_of_connections(N,N_real,K_real_STN_Proto_diverse)

Proto = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A', 'GABA-B'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A', 'GABA-B'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
STN = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
nuclei_dict = {'Proto': Proto, 'STN' : STN}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

#run(mvt_selective_ext_input_dict, D_perturb,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain, nuclei_dict)
n = 5
GABA_A = np.linspace(5,20,n)
GABA_B = np.linspace(150,300,3)
Glut = np.linspace(0.5,12,n)
sweep_time_scales(Proto,STN, GABA_A, GABA_B, Glut, dt, 'data_GABA_A_B_Glut.pkl',mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain)

pkl_file = open('data_GABA_A_B_Glut.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
x = data['tau'][:,0]
y = data['tau'][:,1]
z = data['tau'][:,2]
name = 'Proto' ; state = 'mvt_freq'
c = data[(name,state)]
ind = np.where(c>0)

scatter_3d_plot(x,y,z,c, name, np.max(c), np.min(c),['GABA_A','GABA_B','Glut','freq'],limits = {'x':(min(x),max(x)), 'y':(min(y),max(y)), 'z':(min(z),max(z))})
#scatter_3d_plot(x[ind],y[ind],z[ind],c[ind], name, np.max(c), np.min(c),['GABA_A','GABA_B','Glut','freq'],limits = {'x':(min(x),max(x)), 'y':(min(y),max(y)), 'z':(min(z),max(z))})
scatter_3d_plot(x[ind],y[ind],z[ind],c[ind], name, 30, 20,['GABA_A','GABA_B','Glut','freq'],limits = {'x':(min(x),max(x)), 'y':(min(y),max(y)), 'z':(min(z),max(z))})
#scatter_3d_plot(x[ind],y[ind],z[ind],c[ind], name, 31, 27,['GABA_A','GABA_B','Glut','freq'],limits = {'x':(min(x),max(x)), 'y':(min(y),max(y)), 'z':(min(z),max(z))})

n_plot = 64; plot_start = 0; line_type = ['-','--']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t_list[plot_start:]*dt,data['Proto','pop_act'][n_plot,plot_start:], line_type[0], label = "Proto" , c = 'r',lw = 1.5)
ax.plot(t_list[plot_start:]*dt,data['STN','pop_act'][n_plot,plot_start:], line_type[0],label = "STN", c = 'k',lw = 1.5)
ax.set_title(r"$\tau_{GABA_A}$ = "+ str(round(x[n_plot],2))+r' $\tau_{GABA_B}$ ='+str(round(y[n_plot],2))+ r' $\tau_{Glut}$ ='+str(round(z[n_plot],2))+' f ='+str(round(c[n_plot],2)) , fontsize = 10)
plt.legend(fontsize = 5)

temp_oscil_check(data['STN','pop_act'][n_plot,plot_start:],oscil_peak_threshold['STN'], 3,*duration_mvt)
temp_oscil_check(data['Proto','pop_act'][n_plot,plot_start:],oscil_peak_threshold['Proto'], 3,*duration_mvt)

plt.title(r"$\tau_{GABA_A}$ = "+ str(round(x[n_plot],2))+r' $\tau_{GABA_B}$ ='+str(round(y[n_plot],2))+ r' $\tau_{Glut}$ ='+str(round(z[n_plot],2))+' f ='+str(round(c[n_plot],2)) , fontsize = 10)

#%%
######################################3 only GABA-A
K = calculate_number_of_connections(N,N_real,K_real_STN_Proto_diverse)

#K = calculate_number_of_connections(N,N_real,K_real)
receiving_pop_list = {('STN','1') : [('Proto', '1')], ('STN','2') : [('Proto', '2')],
                    ('Proto','1') : [('Proto', '1'), ('STN', '1'), ('STN', '2')],
                    ('Proto','2') : [('Proto', '2'), ('STN', '1'), ('STN', '2')]}
rest_ext_input = set_rest_ext_inp(A, gain, G, threshold) 
mvt_ext_input_dict = set_mvt_ext_inp(A_mvt, gain, G, threshold, rest_ext_input)
Proto = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-A'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
STN = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list, smooth_kern_window,oscil_peak_threshold)]
nuclei_dict = {'Proto': Proto, 'STN' : STN}
#for k in range (len(Proto)):
#    Proto[k].tau = {'GABA-A' : 12, 'GABA-B' : 200}
#    STN[k].tau = {'Glut': 6} 
receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]

n = 5
Glut = np.linspace(0.5,12,n)
GABA_A = np.linspace(5,20,n)
g_list = np.linspace(-3,-0.01, 100)
sweep_time_scales_one_GABA(g_list, Proto, STN, 'GABA_A',GABA_A, G,Glut, dt, 'data_GABA_A.pkl',mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain)
pkl_file = open('data_GABA_A_B_Glut.pkl', 'rb')
freq = pickle.load(pkl_file)
pkl_file.close()
x = freq['tau'][:,0]
y = freq['tau'][:,1]
z = freq[('STN','mvt')]
name = 'STN'
scatter_3d_plot(x,y,z, z, name, np.max(z), np.min(z),['GABA_A','Glut','freq','freq'])
#%%
################################### only GABA-B

Proto = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-B'], rest_ext_input, receiving_pop_list,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'Proto', G, T, t_sim, dt, tau, ['GABA-B'], rest_ext_input, receiving_pop_list,oscil_peak_threshold)]
STN = [Nucleus(1, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list,oscil_peak_threshold),
       Nucleus(2, gain, threshold,noise_variance, noise_amplitude, N, A, 'STN', G, T, t_sim, dt, tau, ['Glut'], rest_ext_input, receiving_pop_list,oscil_peak_threshold)]
nuclei_dict = {'Proto': Proto, 'STN' : STN}

receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
for key in receiving_class_dict.keys():
    receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]
GABA_B = np.linspace(150,300,8)

sweep_time_scales_one_GABA(Proto, STN, 'GABA_B', GABA_B, Glut, dt, 'data_GABA_B_Glut.pkl',mvt_ext_input_dict, D_mvt,t_mvt,T, receiving_class_dict,t_list, K, N, threshold, gain)
pkl_file = open('ddata_GABA_B_Glut.pkl', 'rb')
freq = pickle.load(pkl_file)
pkl_file.close()
x = freq['tau'][:,0]
y = freq['tau'][:,1]
z = freq[('STN','mvt')]
name = 'STN'
scatter_3d_plot(x,y,z, z, name, np.max(z), np.min(z),['GABA_B','Glut','freq','freq'])

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
    


    

    

    
    
    
    
    