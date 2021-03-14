import numpy as np
import matplotlib.pyplot as plt
import timeit
import matplotlib
from numpy.fft import rfft,fft, fftfreq
from scipy import signal,stats
from tempfile import TemporaryFile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
import pickle
from matplotlib.ticker import FormatStrFormatter
# matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')
#from scipy.ndimage.filters import generic_filter

def find_sending_pop_dict(receiving_pop_list):
    sending_pop_list = {k: [] for k in receiving_pop_list.keys()}
    for k,v_list in receiving_pop_list.items():
        for v in v_list:
            sending_pop_list[v].append(k)
    return sending_pop_list
class Nucleus:

    def __init__(self, population_number,gain, threshold,neuronal_consts,tau, ext_inp_delay, noise_variance, noise_amplitude, 
        N, A,A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_from_list,smooth_kern_window,oscil_peak_threshold, neuronal_model = 'rate',poisson_prop = None):
        
        self.n = N[name] # population size
        self.population_num = population_number
        self.name = name
        self.basal_firing = A[name]
        self.mvt_firing = A_mvt[name]
        self.threshold = threshold[name]
        self.gain = gain[name]
        # self.trans_types =  trans_types
        self.synaptic_time_constant = {k: v for k, v in synaptic_time_constant.items() if k[1]==name} # filter based on the receiving nucleus# dictfilt(synaptic_time_constant, self.trans_types) # synaptic time scale based on neuron type
        self.transmission_delay = {k: v for k, v in T.items() if k[0]==name} # filter based on the receiving nucleus
        self.ext_inp_delay = ext_inp_delay        
        self.synaptic_weight = {k: v for k, v in G.items() if k[0]==name} # filter based on the receiving nucleus
        self.K_connections = None
        self.history_duration = max(self.transmission_delay.values()) # stored history in ms derived from the longest transmission delay of the projections
        self.receiving_from_list = receiving_from_list[(self.name, str(self.population_num))] 
        sending_to_dict = find_sending_pop_dict(receiving_from_list)
        self.sending_to_dict = sending_to_dict[(self.name, str(self.population_num))] 
        # self.output = np.zeros((self.n,int(self.history_duration/dt)))
        self.output = {k: np.zeros((self.n,int(T[k[0],self.name]/dt))) for k in self.sending_to_dict} 
        self.input = np.zeros((self.n))
        self.neuron_act = np.zeros((self.n))
        self.pop_act = np.zeros((int(t_sim/dt))) # time series of population activity
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

        if neuronal_model == 'spiking':
            self.spikes = np.zeros((self.n,int(t_sim/dt)))
            self.tau = {k: v for k, v in tau.items() if k[0]==name} # filter based on the receiving nucleus
            self.I_rise = {k: np.zeros((self.n,len(self.tau[self.name,k[0]]['decay']))) for k in self.receiving_from_list} # since every connection might have different rise and decay time, inputs must be updataed accordincg to where the input is coming from
            self.I_syn = {k: np.zeros((self.n,len(self.tau[self.name,k[0]]['decay']))) for k in self.receiving_from_list}
            self.I_syn['ext_pop','1'] = np.zeros((self.n))
            self.mem_potential = np.zeros((self.n)) # membrane potential
            self.neuronal_consts = neuronal_consts
            self.spike_thresh = np.random.normal(self.neuronal_consts['spike_thresh']['mean'],self.neuronal_consts['spike_thresh']['var'],self.n)
            self.membrane_time_constant = np.random.normal(self.neuronal_consts['membrane_time_constant']['mean'],self.neuronal_consts['membrane_time_constant']['var'],self.n)
            self.voltage_trace = np.zeros((int(t_sim/dt)))
            self.syn_inputs = {k: np.zeros((self.n,1)) for k in self.receiving_from_list}
            self.syn_inputs['ext_pop','1'] = np.zeros((self.n)) # synaptic inputs of the external population
            self.poisson_spikes = None # meant to store the poisson spike trains of the external population
            self.n_ext_population = poisson_prop[self.name]['n'] # external population size
            self.firing_of_ext_pop = poisson_prop[self.name]['firing'] 
            self.tau_ext_pop = np.random.normal(poisson_prop[self.name]['tau']['mean'], poisson_prop[self.name]['tau']['var'],self.n)# synaptic decay time of the external pop inputs
            self.syn_weight_ext_pop = poisson_prop[self.name]['g']
            self.representative_inp = {k: np.zeros((int(t_sim/dt),len(self.tau[self.name,k[0]]['decay']))) for k in self.receiving_from_list}
            self.representative_inp['ext_pop','1'] = np.zeros((int(t_sim/dt)))

    def calculate_input_and_inst_act(self, t, dt, receiving_from_class_list, mvt_ext_inp):  
        
        syn_inputs = np.zeros((self.n,1)) # = Sum (G Jxm)
        for projecting in receiving_from_class_list:

            syn_inputs += self.synaptic_weight[(self.name, projecting.name)]*np.matmul(self.connectivity_matrix[(projecting.name,str(projecting.population_num))], 
                           projecting.output[(self.name,str(self.population_num))][:,-int(self.transmission_delay[(self.name,projecting.name)]/dt)].reshape(-1,1))/self.K_connections[(self.name, projecting.name)]
        
        #        print("noise", noise_generator(self.noise_amplitude, self.noise_variance, self.n)[0])
        inputs = syn_inputs + self.rest_ext_input  + mvt_ext_inp #+ noise_generator(self.noise_amplitude, self.noise_variance, self.n)
        self.neuron_act = transfer_func(self.threshold, self.gain, inputs)
        self.pop_act[t] = np.average(self.neuron_act)

    def update_output(self,dt):
        new_output = {k: self.output[k][:,-1].reshape(-1,1) for k in self.output.keys()}
        for key in self.sending_to_dict:
            for tau in self.synaptic_time_constant[(key[0],self.name)]:
                new_output[key] += dt*(-self.output[key][:,-1].reshape(-1,1)+self.neuron_act)/tau
            self.output[key] = np.hstack((self.output[key][:,1:], new_output[key]))
    
    def cal_ext_inp(self,dt):
        poisson_spikes = possion_spike_generator(self.n,self.n_ext_population,self.firing_of_ext_pop,dt)
        self.syn_inputs['ext_pop','1'] =  np.sum(poisson_spikes,axis = 1)*self.membrane_time_constant*self.syn_weight_ext_pop
        self.I_syn['ext_pop','1'] += (-self.I_syn['ext_pop','1'] + self.syn_inputs['ext_pop','1'])*dt/self.tau_ext_pop

    def solve_EIF(self,t,dt,receiving_from_class_list,mvt_ext_inp):
        # self.rest_ext_input = np.ones((self.n)) * 100
        self.cal_ext_inp(dt)
        inputs = np.zeros((self.n))
        for projecting in receiving_from_class_list:
            num = str(projecting.population_num)
            self.syn_inputs[projecting.name,num] = self.synaptic_weight[(self.name, projecting.name)]*np.matmul(self.connectivity_matrix[(projecting.name,num)], 
                           projecting.spikes[:,int(t-self.transmission_delay[(self.name,projecting.name)]/dt)].reshape(-1,1))*self.membrane_time_constant.reshape(-1,1)
            i = 0
            for i in range(len(self.tau[self.name,projecting.name]['rise'])):
                self.I_rise[projecting.name,num][:,i] += (-self.I_rise[projecting.name,num][:,i] + self.syn_inputs[projecting.name,num])*dt/self.tau[(self.name,projecting.name)]['rise'][i]
                self.I_syn[projecting.name,num][:,i] += (-self.I_syn[projecting.name,num][:,i] + self.I_rise[projecting.name,num][:,i])*dt/self.tau[(self.name,projecting.name)]['decay'][i]
                self.representative_inp[projecting.name,num][t,i] = self.I_syn[projecting.name,num][5,i]
                inputs += self.I_syn[projecting.name,num][:,i]
                i += 1
        # inputs +=  self.rest_ext_input  #+ mvt_ext_inp #+ noise_generator(self.noise_amplitude, self.noise_variance, self.n)
        self.representative_inp['ext_pop','1'][t] = self.I_syn['ext_pop','1'][5]
        inputs +=  self.I_syn['ext_pop','1']  #+ mvt_ext_inp #+ noise_generator(self.noise_amplitude, self.noise_variance, self.n)
        ###### EIF
        # self.mem_potential += (self.neuronal_consts['u_rest'] - self.mem_potential+ inputs+ self.neuronal_consts['nonlin_sharpness'] *np.exp((self.mem_potential-
        #                         self.neuronal_consts['nonlin_thresh'])/self.neuronal_consts['nonlin_sharpness']))*dt/self.neuronal_consts['membrane_time_constant']
        ##### LIF
        self.mem_potential += (self.neuronal_consts['u_rest'] - self.mem_potential+ inputs)*dt/self.membrane_time_constant
        
        # spiking_ind = np.where(self.mem_potential > self.neuronal_consts['spike_thresh']['mean']) # homogeneous spike thresholds
        spiking_ind = np.where(self.mem_potential > self.spike_thresh) # gaussian distributed spike thresholds

        self.spikes[spiking_ind, t] = 1
        self.mem_potential[spiking_ind] = self.neuronal_consts['u_rest']
        self.pop_act[t] = np.sum(self.spikes[spiking_ind, t])
        self.voltage_trace[t] = self.mem_potential[1]

    def reset_ext_pop_properties(self,poisson_prop):
        '''reset the properties of the external poisson spiking population'''
        self.n_ext_population = poisson_prop[self.name]['n'] # external population size
        self.firing_of_ext_pop = poisson_prop[self.name]['firing'] 
        self.tau_ext_pop = poisson_prop[self.name]['tau'] # synaptic decay time of the external pop inputs
        self.syn_weight_ext_pop = poisson_prop[self.name]['g']

    def set_connections(self, K, N):
        ''' creat Jij connection matrix
        
        Parameters
        ----------
        
        connected_nuclei_list : list
            list of class instances that are connected to this nucleus
        
        Returns
        -------
        '''
        self.K_connections = {k: v for k, v in K.items() if k[0]==self.name}
        for projecting in self.receiving_from_list:
            
            n_connections = self.K_connections[(self.name, projecting[0])]
            self.connectivity_matrix[projecting] = build_connection_matrix(self.n, N[projecting[0]], n_connections)
#            J[(self.name, projecting)] = build_connection_matrix(self.n, N[projecting], n_connections)
    def clear_history(self):
        # self.output = np.zeros_like(self.output)
        self.output = {k: np.zeros_like(self.output[k]) for k in self.output.keys()}
        self.input = np.zeros_like(self.input)
        self.neuron_act = np.zeros_like(self.neuron_act)
        self.pop_act = np.zeros_like(self.pop_act) 
        self.mvt_ext_input = np.zeros_like(self.mvt_ext_input) 
        self.external_inp_t_series = np.zeros_like(self.external_inp_t_series) 
        
    def set_synaptic_weights(self,G):
        self.synaptic_weight = {k: v for k, v in G.items() if k[0]==self.name} # filter based on the receiving nucleus

    def set_synaptic_time_scales(self,synaptic_time_constant):
        self.synaptic_time_constant = {k: v for k, v in synaptic_time_constant.items() if k[1]==self.name}

    def set_ext_input(self,A, A_mvt, D_mvt,t_mvt, t_list, dt):
        proj_list = [k[0] for k in list(self.receiving_from_list)]

        self.rest_ext_input = self.basal_firing/self.gain - np.sum([self.synaptic_weight[self.name,proj]*A[proj] for proj in proj_list]) + self.threshold
        self.mvt_ext_input = self.mvt_firing/self.gain - np.sum([self.synaptic_weight[self.name,proj]*A_mvt[proj] for proj in proj_list]) + self.threshold - self.rest_ext_input
        self.external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,self.ext_inp_delay,self.mvt_ext_input, t_list*dt)

def find_ext_input_reproduce_nat_firing(g_series,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict):
    ''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''

    nucleus_name = list(nuclei_dict.keys())
    firing_prop = {k: {'mean_firing': np.zeros((len(nuclei_dict[nucleus_name[0]][0]))),'firing_var':np.zeros((len(nuclei_dict[nucleus_name[0]][0])))} for k in nucleus_name}
    for g_1 in g_1_list:
        for g_2 in g_2_list:
            poisson_prop[nucleus_name[0]]['g'] = g_1
            poisson_prop[nucleus_name[1]]['g'] = g_2
            for nuclei_list in nuclei_dict.values():
                for nucleus in nuclei_list:
                    nucleus.reset_ext_pop_properties(poisson_prop)
            nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
            for nuclei_list in nuclei_dict.values():
                for nucleus in nuclei_list:
                    firing_prop[nucleus.name]['mean_firing'][nucleus.population_num-1] = np.average[nucleus.pop_act[int(len(t_list)/2):]]
                    firing_prop[nucleus.name]['firing_var'][nucleus.population_num-1] = np.std[nucleus.pop_act[int(len(t_list)/2):]]
                    print(nucleus.name,'FR=',firing_prop[nucleus.name]['mean_firing'][nucleus.population_num-1],'std=',firing_prop[nucleus.name]['firing_var'][nucleus.population_num-1])
    return firing_prop

def dopamine_effect(threshold, G, dopamine_percentage):
    ''' Change the threshold and synaptic weight depending on dopamine levels'''
    threshold['Str'] = -0.02 + 0.03*(1-(1.1/(1+0.1*np.exp(-0.03*(dopamine_percentage - 100)))))
    G[('Str','Ctx')] = 0.75/(1+np.exp(-0.09*(dopamine_percentage - 60)))
    return threshold, G

def possion_spike_generator(n_pop,n_sending,r,dt):
    '''generate a times series of possion spikes for a population of size n, with firing rate r'''
    x = np.random.rand(n_pop,n_sending)
    ind = np.where(x <= (1-np.exp(-r*dt)))
    x[x <= r*dt] = 1 # spike with probability of rdt
    x[x != 1] = 0
    # print('n_spikes =', len(ind[0]))
    return x
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

def run(receiving_class_dict,t_list, dt, nuclei_dict):
    
    start = timeit.default_timer()
    
    # for t in t_list:
    for t in t_list[10:]:

        for nuclei_list in nuclei_dict.values():
            k = 0
            for nucleus in nuclei_list:
                k += 1
        #        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement 
                mvt_ext_inp = np.ones((nucleus.n,1))*nucleus.external_inp_t_series[t] # movement added 
                ######## rate model
                # nucleus.calculate_input_and_inst_act(t, dt, receiving_class_dict[(nucleus.name,str(k))], mvt_ext_inp)
                # nucleus.update_output(dt)
                ######## QIF
                nucleus.solve_EIF(t,dt,receiving_class_dict[(nucleus.name,str(k))],mvt_ext_inp)

    stop = timeit.default_timer()
#    print("t = ", stop - start)
    return nuclei_dict
       

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

def calculate_number_of_connections(N_sim,N_real,number_of_connection):
    '''calculate number of connections in the scaled network.'''
    KK = number_of_connection.copy()
    for k, v in number_of_connection.items():
#        print(k,v, N_real[k[1]],N_sim[k[0]])
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

def plot( nuclei_dict,color_dict,  dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_ob, title = "", n_subplots = 1,title_fontsize = 18,plot_start = 0,ylabelpad = 0):    

    if plot_ob == None:
        fig = plt.figure(figsize = (6,5))
        ax = fig.add_subplot(111)
    else:
        fig, ax = plot_ob
     
    line_type = ['-', '--']
    for nuclei_list in nuclei_dict.values():
        for nucleus in [nuclei_list[0]]:
            
            ax.plot(t_list[plot_start:]*dt, nucleus.pop_act[plot_start:], line_type[nucleus.population_num-1], label = nucleus.name, c = color_dict[nucleus.name],lw = 1.5)
            ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A[nucleus.name], '-.', c = color_dict[nucleus.name],lw = 1, alpha=0.8 )
            ax.plot(t_list[plot_start:]*dt, np.ones_like(t_list[plot_start:])*A_mvt[nucleus.name], '-.', c = color_dict[nucleus.name], alpha=0.2,lw = 1 )

    ax.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color='lightskyblue')
    plt.title(title, fontsize = title_fontsize)
    plt.xlabel("time (ms)", fontsize = 15)
    plt.ylabel("firing rate (spk/s)", fontsize = 15,labelpad=ylabelpad)
    plt.legend(fontsize = 15)
    # ax.tick_params(axis='both', which='major', labelsize=10)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    return fig

def scatter_2d_plot(x,y,c, title, label, limits = None,label_fontsize = 15):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y,'k', lw = 0.5)
    img = ax.scatter(x, y, c=c, cmap=plt.hot(),lw = 1,edgecolor = 'k', s = 50)

    if limits == None:
        limits = {'x':(min(x),max(x)), 'y':(min(y),max(y))}
        
    ax.set_xlabel(label[0], fontsize = label_fontsize)
    ax.set_ylabel(label[1], fontsize = label_fontsize)
    ax.set_title(title)
    # ax.set_xlim(limits['x'])
    # ax.set_ylim(limits['y'])
    clb = fig.colorbar(img)
    clb.set_label(label[2], labelpad=10, y=0.5, rotation=-90)
    clb.ax.locator_params(nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=4)
    plt.show()
    
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
        
    ax.w_xaxis.set_pane_color ((0., 0., 0., 0.))
    ax.w_yaxis.set_pane_color ((0., 0., 0., 0.))
    ax.w_zaxis.set_pane_color ((0., 0., 0., 0.))
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
    
def scatter_3d_wireframe_plot(x,y,z,c, title, label, limits = None,label_fontsize = 15):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color = 'grey', lw = 0.5 ,zorder = 1)
    img = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=c.flatten(), cmap=plt.hot(), s = 20, lw = 1,edgecolor = 'k',zorder=2)

    if limits == None:
        limits = {'x':(np.amin(x),np.amax(x)), 'y':(np.amin(y),np.amax(y)), 'z':(np.amin(z),np.max(z))}
      
    ax.w_xaxis.set_pane_color ((0., 0., 0., 0.))
    ax.w_yaxis.set_pane_color ((0., 0., 0., 0.))
    ax.w_zaxis.set_pane_color ((0., 0., 0., 0.))  
    ax.w_xaxis.gridlines.set_lw(.5)
    ax.w_yaxis.gridlines.set_lw(0.5)
    ax.w_zaxis.gridlines.set_lw(0.5)
    ax.set_xlabel(label[0], fontsize = label_fontsize)
    ax.set_ylabel(label[1],fontsize = label_fontsize)
    ax.set_zlabel(label[2],fontsize = label_fontsize,rotation = -90)
    ax.set_title(title)
    ax.set_xlim(limits['x'])
    ax.set_ylim(limits['y'])
    ax.set_zlim(limits['z'])
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))    
    clb = fig.colorbar(img,pad = 0.15)
    clb.set_label(label[3], labelpad=10, y=.5, rotation=-90)
    clb.ax.locator_params(nbins=5)
    plt.rcParams['grid.linewidth'] = 0.1
    # plt.locator_params(axis='y', nbins=6)
    # plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='z', nbins=5)
    plt.show()
    return fig,ax
    
def scatter_3d_wireframe_plot_2_data_series(x1,y1,z1,c_mark1,c_wire1, x2,y2,z2,c_mark2,c_wire2, title, labels, ax_label, limits = None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x1, y1, z1, color = c_wire1, lw = 0.5 )
    img = ax.scatter(x1.flatten(), y1.flatten(), z1.flatten(), c=c_mark1, s = 20, lw = 1,edgecolor = 'k', label = labels[0])
    ax.plot_wireframe(x2, y2, z2, color = c_wire2, lw = 0.5 )
    img = ax.scatter(x2.flatten(), y2.flatten(), z2.flatten(), c=c_mark2, s = 20, lw = 1,edgecolor = 'k', label = labels[1])

    # if limits == None:
    #     limits = {'x':(np.amin(x),np.amax(x)), 'y':(np.amin(y),np.amax(y)), 'z':(np.amin(z),np.max(z))}
        
    ax.set_xlabel(ax_label[0])
    ax.set_ylabel(ax_label[1])
    ax.set_zlabel(ax_label[2])
    ax.set_title(title)
    # ax.set_xlim(limits['x'])
    # ax.set_ylim(limits['y'])
    # ax.set_zlim(limits['z'])
    ax.legend()
    # clb = fig.colorbar(img,pad = 0.15)
    # clb.set_label(label[3], labelpad=-40, y=1.05, rotation=0)
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
    low_var_ind = np.where(rolling_var > epsilon)[0]
    if len(low_var_ind) == 0:
        return []
    else:
        return np.arange(np.max(low_var_ind))
        ##### Overthinking
        # continous_run_starts = np.where(np.diff(low_var_ind) != 1)[0] # find the starts of runs of continuous chunks
        # if len(continous_run_starts) != 0:
        #     # print(continous_run_starts)
        #     cut_plateau_ind = np.arange(low_var_ind[np.max(continous_run_starts)+1]) # make a continuous array up to the last run
        #     return cut_plateau_ind
        # else:
        #     return continous_run_starts

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
    # print(half_lambda)
    n_half_cycles = len(half_lambda)
    # print("half_lambda = ", half_lambda)
    if n_half_cycles > 1:
        frequency = 1/(np.average(half_lambda)*2*dt)
    else: 
        frequency = 0
    return n_half_cycles, frequency

def find_mean_of_signal(sig):
    ''' find the value which when subtracted the signal oscillates around zero'''
    cut_plateau_ind = cut_plateau(sig)
    len_not_plateau = len(cut_plateau_ind)
    # plt.figure()
    # plt.plot(sig)
    # plt.plot(sig[:max(cut_plateau_ind)])
    # plt.plot(sig[cut_plateau_ind])
    # print(len_not_plateau,len(sig))
    if len_not_plateau > 0 and len_not_plateau < len(sig)*4/5: # if more than 1/5th of signal is plateau
        plateau_y = np.average(sig[max(cut_plateau_ind):])  
    else: # if less than 1/5th is left better to average the whole signal for more confidence
        # peaks,properties = signal.find_peaks(sig)
        # troughs,properties = signal.find_peaks(-sig)
        plateau_y = np.average(sig)
    return cut_plateau_ind, plateau_y

def trim_start_end_sig_rm_offset(sig,start, end):
    ''' trim with max point at the start and the given end point'''
    trimmed = sig[start:end]
    _, plateau_y = find_mean_of_signal(trimmed)
    trimmed = trimmed - plateau_y # remove the offset determined with plateau level
    # print('trimmed = ',np.average(trimmed))
    max_value = np.max(trimmed)
    min_value = np.min(trimmed)
    # plt.figure()
    # plt.plot(trimmed)
    if abs(max_value) > abs(min_value): # find exterma, whether it's a minimum or maximum
        max_point = np.max(np.where(trimmed == max_value))
    else:
        max_point = np.max(np.where(trimmed == min_value))

    if max_point>len(trimmed)/2: # in case the amplitude increases over time take the whole signal #############################3 to do --> check for increasing amplitude in oscillations instead of naive implementation here
        return trimmed- np.average(trimmed)
    else:
        return(trimmed[max_point:] - np.average(trimmed[max_point:]))
    
def find_freq_of_pop_act_spec_window(nucleus, start, end, dt, peak_threshold = 0.1, smooth_kern_window= 3 , check_stability = False):
    ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
    the plateau and in case it is oscillation determine the frequency '''
    sig = trim_start_end_sig_rm_offset(nucleus.pop_act,start, end)
    cut_sig_ind, plateau_y = find_mean_of_signal(sig)
    # plt.plot(sig - plateau_y)
    # plt.axhline(plateau_y)
    # plt.plot(sig[cut_sig_ind])
    if_stable = False
    if len(cut_sig_ind) > 0: # if it's not all plateau from the beginning
        sig = sig - plateau_y
        # print('trimmed plateau removed', np.average(sig))
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
            if check_stability:
                if_stable = if_stable_oscillatory(sig, max(cut_sig_ind), peak_threshold, smooth_kern_window, amp_env_slope_thresh = - 0.05)
            return n_half_cycles, perc_oscil, freq, if_stable
        else:
            return 0,0,0, False
    else:
        return 0,0,0, False
        
def if_stable_oscillatory(sig,x_plateau, peak_threshold, smooth_kern_window, amp_env_slope_thresh = - 0.05, oscil_perc_as_stable = 0.9, last_first_peak_ratio_thresh = [0.92,1.1]):
    ''' detect if there's stable oscillation defined as a non-decaying wave'''
    # if len(sig) <= (x_plateau +2) and len(sig) >= (x_plateau-2) : # if the whole signal is oscillatory
    if  x_plateau > len(sig)*oscil_perc_as_stable : # if the whole signal is oscillatory

        sig = gaussian_filter1d(sig[:x_plateau],smooth_kern_window)
        peaks,properties = signal.find_peaks(sig, height = peak_threshold)
        # troughs,_ = signal.find_peaks(-sig, height = peak_threshold)
        # if len(peaks)-1 == 0: # if no peaks are found error will be raised, so we might as well plot to see the reason
            # troughs,_ = signal.find_peaks(-sig, height = peak_threshold)
            # plt.figure()
            # plt.plot(sig)
            # plt.axhline(np.average(sig))
        ################## amplitude envelope Slope thresholding method
        # slope, intercept, r_value, p_value, std_err = stats.linregress(peaks[1:],sig[peaks[1:]]) # discard the first peak because it's prone to errors
        # print('slope = ', slope)
        # if slope > amp_env_slope_thresh: 
        ################# relative first and last peak ratio thresholding
        if len(peaks)>1 : 
            last_first_peak_ratio = sig[peaks[-1]]/sig[peaks[1]]
            print(last_first_peak_ratio)
        else: return False
        if last_first_peak_ratio_thresh[0] < last_first_peak_ratio < last_first_peak_ratio_thresh[1]:
            # plt.figure()
            # plt.plot(sig)
            # plt.axvline(x_plateau)
            # plt.plot(peaks,slope*peaks+intercept,'-')
        #    plt.plot(fluctuations, label = "Gaus kern smoothed")
            # plt.plot(peaks,sig[peaks],"x", markersize = 10,markeredgewidth = 2)
            # plt.plot(troughs,sig[troughs],"x", markersize = 10, markeredgewidth = 2)
        #    plt.legend()
            # print('peaks, slope = ', slope)
            return True
        else:
            return False
    else: # it's transient
        return False
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
    

def synaptic_weight_space_exploration(G, A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, receiving_class_dict, color_dict, if_plot = False, G_ratio_dict = None):
    list_1  = list(G_dict.values())[0] ; list_2  = list(G_dict.values())[1]
    n = len(list_1) ;m = len(list_2)
    print(n,m)
    data = {} 
    for nucleus_list in nuclei_dict.values():
        nucleus = nucleus_list[0] # get only on class from each population
        data[(nucleus.name, 'mvt_freq')] = np.zeros((n,m))
        data[(nucleus.name, 'base_freq')] = np.zeros((n,m))
        data[(nucleus.name, 'perc_t_oscil_mvt')] = np.zeros((n,m))
        data[(nucleus.name, 'perc_t_oscil_base')] =np.zeros((n,m))
        data[(nucleus.name, 'n_half_cycles_mvt')] = np.zeros((n,m))
        data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n,m))
        data[(nucleus.name,'g_transient_boundary')] = []
        data[(nucleus.name,'g_stable_boundary')] = []
    data['g'] = np.zeros((n,m,2))
    count  = 0
    i = 0 
    if_stable_plotted = False
    if_trans_plotted = False
    if if_plot:
        fig = plt.figure()
    if np.average(list_1) < 0: list_1_copy = reversed(list_1) # to approach the boundary form the steady state
    for g_1 in list_1:
        j = 0
        found_g_transient = {k: False for k in nuclei_dict.keys()}
        found_g_stable = {k: False for k in nuclei_dict.keys()}
        if np.average(list_2) < 0: list_2_copy = reversed(list_2)
        for g_2 in list_2_copy:
            G[(tuple(G_dict.keys())[0])] = g_1 # returns keys as list, tuple is needed 
            G[(tuple(G_dict.keys())[1])] = g_2
            if G_ratio_dict != None: # if the circuit has more than 2 members
                for k,g_ratio in G_ratio_dict.items():
                    G[k] = g_2*g_ratio
            nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
            run(receiving_class_dict,t_list, dt, nuclei_dict)
            data['g'][i,j,:] = [g_1, g_2]
            nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
            for nucleus in nucleus_list:
                data[(nucleus.name, 'n_half_cycles_mvt')][i,j],data[(nucleus.name,'perc_t_oscil_mvt')][i,j], data[(nucleus.name,'mvt_freq')][i,j],if_stable_mvt= find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
                data[(nucleus.name, 'n_half_cycles_base')][i,j],data[(nucleus.name,'perc_t_oscil_base')][i,j], data[(nucleus.name,'base_freq')][i,j],if_stable_base= find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, smooth_kern_window = nucleus.smooth_kern_window, check_stability=True)
                print(nucleus.name,' g1 = ', round(g_1,2), ' g2 = ', round(g_2,2), 'n_cycles =', data[(nucleus.name, 'n_half_cycles_mvt')][i,j],round(data[(nucleus.name, 'perc_t_oscil_mvt')][i,j],2),'%',  'f = ', round(data[(nucleus.name,'mvt_freq')][i,j],2) )

                if not found_g_transient[nucleus.name]  and data[(nucleus.name, 'n_half_cycles_mvt')][i,j]> lim_n_cycle[0] and data[(nucleus.name, 'n_half_cycles_mvt')][i,j]< lim_n_cycle[1]:
                    data[(nucleus.name,'g_transient_boundary')].append([g_1,g_2]) # save the the threshold g to get transient oscillations
                    found_g_transient[nucleus.name] = True

                if not if_trans_plotted and data[(nucleus.name, 'n_half_cycles_mvt')][i,j]> lim_n_cycle[0] and data[(nucleus.name, 'n_half_cycles_mvt')][i,j]< lim_n_cycle[1]:
                    if_trans_plotted = True
                    print("transient plotted")
                    plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, title = r"$G_{"+list(G_dict.keys())[0][0]+"-"+list(G_dict.keys())[0][1]+"}$ = "+ str(round(g_1,2))+r"$G_{"+list(G_dict.keys())[1][0]+"-"+list(G_dict.keys())[1][1]+"}$ ="+str(round(g_2,2)),plot_ob = None)
                
                if not found_g_stable[nucleus.name] and if_stable_mvt: 
                    found_g_stable[nucleus.name] = True
                    data[(nucleus.name,'g_stable_boundary')].append([g_1,g_2])

                if not if_stable_plotted and if_stable_mvt:
                    if_stable_plotted = True
                    print("stable plotted")
                    plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, title = r"$G_{"+list(G_dict.keys())[0][0]+"-"+list(G_dict.keys())[0][1]+"}$ = "+ str(round(g_1,2))+r"  $G_{"+list(G_dict.keys())[1][0]+"-"+list(G_dict.keys())[1][1]+"}$ ="+str(round(g_2,2)),plot_ob = None)
            if if_plot:
                ax = fig.add_subplot(n,m,count+1)
                plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,[fig, ax], title = r"$G_{STN-Proto}$ = "+ str(round(g_1,2))+r' $G_{Proto-Proto}$ ='+str(round(g_2,2)), n_subplots = int(n*m))
                plt.title( r"$G_{STN-Proto}$ = "+ str(round(g_1,2))+r' $G_{Proto-Proto}$ ='+str(round(g_2,2)), fontsize = 10)    
                plt.xlabel("", fontsize = 10)
                plt.ylabel("", fontsize = 5)
                plt.legend(fontsize = 10)
            count +=1
            j+=1
            print(count, "from", int(m*n))
        i+=1
    if if_plot:
        fig.text(0.5, 0.01, 'time (ms)', ha='center')
        fig.text(0.01, 0.5, 'firing rate (spk/s)', va='center', rotation='vertical')
        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()


# def run_specific_g
def create_receiving_class_dict(receiving_pop_list, nuclei_dict):
    ''' make a list of classes that project to one class form the given list'''
    receiving_class_dict = {key: None for key in receiving_pop_list.keys()}
    for key in receiving_class_dict.keys():
        receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name,k in list(receiving_pop_list[key])]
    return receiving_class_dict

def set_connec_ext_inp(A, A_mvt, D_mvt, t_mvt,dt, N, N_real, K_real_STN_Proto_diverse, receiving_pop_list, nuclei_dict,t_list):
    '''find number of connections and build J matrix, set ext inputs as well'''
    #K = calculate_number_of_connections(N,N_real,K_real)
    K = calculate_number_of_connections(N,N_real,K_real_STN_Proto_diverse)
    receiving_class_dict= create_receiving_class_dict(receiving_pop_list, nuclei_dict)
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.set_ext_input(A, A_mvt, D_mvt,t_mvt, t_list, dt)
            nucleus.set_connections(K, N)
    return receiving_class_dict

def temp_oscil_check(sig_in,peak_threshold, smooth_kern_window,dt,start,end):
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
        print("stable osillatory regime?", if_stable_oscillatory(sig, max(cut_sig_ind),peak_threshold, smooth_kern_window))

#        if if_oscillatory(sig, max(cut_sig_ind),peak_threshold, smooth_kern_window): # then check if there's oscillations
#            perc_oscil = max_non_empty_array(cut_sig_ind)/len(sig)*100
#            _,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)
#            freq = freq_from_fft(sig[cut_sig_ind],dt/1000)
#        else:
#            print("all plateau")
#        plt.figure()
#        plt.plot(sig)
        n_cycles,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)
        if freq != 0: # then check if there's oscillations
            perc_oscil = max_non_empty_array(cut_sig_ind)/len(sig)*100            
            print("n cycles = ",round(n_cycles/2,2),"% = ", round(perc_oscil,2), "f = ",round(freq,2))
        else:
            print("all plateau")

    else:
        print("no oscillation")
    
def reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt):
    ''' Clear history of the population. Reinitialize the synaptic weights and corresponding external inputs '''
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            nucleus.clear_history()
            nucleus.set_synaptic_weights(G)
            nucleus.set_ext_input(A, A_mvt, D_mvt,t_mvt, t_list, dt)
    return nuclei_dict

def sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant, nuclei_dict, syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill=True):
    def set_time_scale( nuclei_dict, synaptic_time_constant):
        for nucleus_list in nuclei_dict.values():
            for nucleus in nucleus_list:
                nucleus.set_synaptic_time_scales(synaptic_time_constant) 
        return nuclei_dict
    t_decay_series_1 = list(syn_decay_dict['tau_1']['tau_list']) ; t_decay_series_2 = list(syn_decay_dict['tau_2']['tau_list'])
    data  = create_data_dict(nuclei_dict, [len(t_decay_series_1), len(t_decay_series_2)], 2,len(t_list))    
    count =0 ; i=0
    for t_decay_1 in t_decay_series_1:
        j = 0
        for t_decay_2 in t_decay_series_2:

            for key,v in syn_decay_dict['tau_1']['tau_ratio'].items():    synaptic_time_constant[key] = [syn_decay_dict['tau_1']['tau_ratio'][key] * t_decay_1]
            for key,v in syn_decay_dict['tau_2']['tau_ratio'].items():    synaptic_time_constant[key] = [syn_decay_dict['tau_2']['tau_ratio'][key] * t_decay_2]
            nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
            nuclei_dict = set_time_scale(nuclei_dict,synaptic_time_constant)
            n_half_cycle,g_transient,g_stable, nuclei_dict, if_stable = find_oscillation_boundary(g_list, nuclei_dict, G,G_ratio_dict, A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle =  lim_n_cycle , find_stable_oscill=find_stable_oscill)

            run(receiving_class_dict,t_list, dt, nuclei_dict)
            data['tau'][i,j,:] = [t_decay_1,t_decay_2]
            
            nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
    #                plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$\tau_{GABA_A}$ = "+ str(round(gaba_a,2))+r' $\tau_{GABA_B}$ ='+str(round(gaba_b,2)))
            for nucleus in nucleus_list:
                data[(nucleus.name, 'g_transient')][i,j] = g_transient
                data[(nucleus.name, 'g_stable')][i,j] = g_stable
                data[(nucleus.name,'trans_n_half_cycle')][i,j] = n_half_cycle
                data[(nucleus.name,'trans_pop_act')][i,j,:] = nucleus.pop_act
                _,_, data[nucleus.name,'trans_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
                _,_, data[nucleus.name,'trans_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
            
            if find_stable_oscill: # only run if you want to checkout the stable oscillatory regime
                for k,g_ratio in G_ratio_dict.items():
                    G[k] = g_stable*g_ratio
                # G[('STN','Proto')] = g_stable
                # G[('Proto','Proto')] = g_stable * g_ratio
                nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
                run(receiving_class_dict,t_list, dt, nuclei_dict)
                for nucleus in nucleus_list:
                    _,_, data[nucleus.name,'stable_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
                    _,_, data[nucleus.name,'stable_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)

            count +=1
            print(count, "from ", len(t_decay_series_1)*len(t_decay_series_2))
            j+=1
        i +=1
    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()

def find_oscillation_boundary(g_list,nuclei_dict, G, G_ratio_dict,A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle = [6,10], find_stable_oscill = False):
    ''' find the synaptic strength for a given set of parametes where you oscillations appear after increasing external input'''
    got_it = False ;g_stable = None; g_transient = None
    if np.average(g_list) < 0 : g_list = reversed(g_list) # always approach from low connection strength, no matter inhibitory or excitatory
    for g in g_list:
        for k,v in G_ratio_dict.items(): G[k] = g*v
        nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)

        run(receiving_class_dict,t_list, dt, nuclei_dict)
        test_1 = list(nuclei_dict.values())[0][0] ;  test = list(nuclei_dict.values())[0][0]#test = nuclei_dict['D2'][0]
        n_half_cycles_mvt,perc_oscil_mvt, f_mvt, if_stable_mvt = find_freq_of_pop_act_spec_window(test,*duration_mvt,dt, peak_threshold =test.oscil_peak_threshold, smooth_kern_window = test.smooth_kern_window, check_stability= find_stable_oscill)
        n_half_cycles_base, perc_oscil_base, f_base, if_stable_base = find_freq_of_pop_act_spec_window(test,*duration_base,dt, peak_threshold =test.oscil_peak_threshold, smooth_kern_window = test.smooth_kern_window, check_stability= find_stable_oscill)
        print('g=',round(g,1),round(f_base,1),n_half_cycles_base, round(f_mvt,1), n_half_cycles_mvt)
        if len(np.argwhere(test_1.pop_act[duration_mvt[0]:duration_mvt[1]] ==0 )) > (duration_mvt[1]-duration_mvt[0])/2 or len(np.argwhere(test.pop_act[duration_mvt[0]:duration_mvt[1]] ==0 )) > (duration_mvt[1]-duration_mvt[0])/2:
            print('zero activity')
       
        if n_half_cycles_mvt >= lim_n_cycle[0] and n_half_cycles_mvt <= lim_n_cycle[1]:
#            plot(nuclei_dict['Proto'], nuclei_dict['STN'], dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
            got_it = True
            
            if g_transient ==None:
                g_transient = g ; n_half_cycles = n_half_cycles_mvt
                print("Gotcha transient!")
            if  not find_stable_oscill:
                break
        if if_stable_mvt and find_stable_oscill:
            got_it = True
            g_stable = g
            print("Gotcha stable!")
            for k,v in G_ratio_dict.items(): 
                print(g_transient)
                G[k] = g_transient*v
            nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt) # return the nuclei at the oscillation boundary state
            break
            
    if not got_it:
        a = 1/0 # to bump a division zero error showing that oscillation couldn't be found in the g range
    return n_half_cycles,g_transient,g_stable, nuclei_dict,if_stable_mvt

def create_data_dict(nuclei_dict, iter_param_length_list, n_time_scale,n_timebins):
    '''build a data dictionary'''
    
    data = {} ; dimensions = iter_param_length_list.copy() ;
    print(n_timebins, dimensions) 
    pop_act_size = tuple(dimensions + [n_timebins])
    for nucleus_list in nuclei_dict.values():
        nucleus = nucleus_list[0] # get only on class from each population
        data[(nucleus.name, 'trans_mvt_freq')] = np.zeros(tuple(iter_param_length_list))
        data[(nucleus.name, 'trans_base_freq')] = np.zeros(tuple(iter_param_length_list))
        data[(nucleus.name, 'stable_mvt_freq')] = np.zeros(tuple(iter_param_length_list))
        data[(nucleus.name, 'stable_base_freq')] = np.zeros(tuple(iter_param_length_list))
        data[(nucleus.name, 'trans_n_half_cycle')] = np.zeros(tuple(iter_param_length_list))
        data[(nucleus.name, 'g_stable')] = np.zeros(tuple(iter_param_length_list))
        data[(nucleus.name, 'g_transient')] = np.zeros(tuple(iter_param_length_list))
        data[(nucleus.name, 'trans_pop_act')] = np.zeros(pop_act_size)
    data['tau'] = np.zeros(tuple(iter_param_length_list+[n_time_scale]))
    return data

def synaptic_weight_transition_multiple_circuits(filename_list, name_list, label_list, color_list, g_cte_ind, g_ch_ind, y_list, c_list,colormap,x_axis = 'multiply',title = "",x_label = "G"):
    maxs = [] ; mins = []
    fig = plt.figure(figsize=(10,8))
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
        if x_axis == 'multiply':
            g = np.squeeze(data['g'][:,:,0]*data['g'][:,:,1])
            g_transient = data[name_list[i],'g_transient_boundary'][0][g_ch_ind[i]]* data[name_list[i],'g_transient_boundary'][0][g_cte_ind[i]] 
            g_stable = data[name_list[i],'g_stable_boundary'][0][g_ch_ind[i]]* data[name_list[i],'g_stable_boundary'][0][g_cte_ind[i]] 
        else:
            g = np.squeeze(data['g'][:,:,g_ch_ind[i]])
            g_transient = data[name_list[i],'g_transient_boundary'][0][g_ch_ind[i]]
            print(data[name_list[i],'g_stable_boundary'])
            g_stable = data[name_list[i],'g_stable_boundary'][0][g_ch_ind[i]]

        # ax.plot(np.squeeze(data['g'][:,:,g_ch_ind[i]]), np.squeeze(data[(name_list[i],y_list[i])]),c = color_list[i], lw = 1, label= label_list[i])
        # img = ax.scatter(np.squeeze(data['g'][:,:,g_ch_ind[i]]), np.squeeze(data[(name_list[i],y_list[i])]),vmin = vmin, vmax = vmax, c=data[(name_list[i],c_list[i])], cmap=colormap,lw = 1,edgecolor = 'k')
        # plt.axvline(g_transient[g_ind[i]], c = color_list[i])
        ax.plot(g, np.squeeze(data[(name_list[i],y_list[i])]),c = color_list[i], lw = 3, label= label_list[i],zorder=1)
        img = ax.scatter(g, np.squeeze(data[(name_list[i],y_list[i])]),vmin = vmin, vmax = vmax, c=data[(name_list[i],c_list[i])], cmap=colormap,lw = 1,edgecolor = 'k',zorder=2,s=80)
        plt.axvline(g_transient, linestyle = '-.',c = color_list[i],alpha = 0.3,lw=2)  # to get the circuit g which is the muptiplication
        plt.axvline(g_stable, c = color_list[i],lw=2)  # to get the circuit g which is the muptiplication
    plt.text(g_stable-0.5, 0.6, 'Stable oscillations',fontsize=18, rotation = -90)
    plt.text(g_transient, 0.6, 'Oscillation appears',fontsize=18, rotation = -90)
    ax.set_xlabel(x_label,fontsize = 20)
    ax.set_ylabel('frequency(Hz)',fontsize=20)
    ax.set_title(title,fontsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    # ax.set_xlim(limits['x'])
    # ax.set_ylim(limits['y'])
    clb = fig.colorbar(img)
    clb.ax.locator_params(nbins=4)
    clb.set_label('% Oscillation', labelpad=15, y=.5, rotation=-90,fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    return fig

def multi_plot_as_f_of_timescale_shared_colorbar(data, y_list, c_list, label_list,g_ratio_list,name_list,filename_list,x_label,y_label,ylabelpad = -5):
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
        ax.set_ylabel(y_label,fontsize = 20,labelpad=ylabelpad)
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
    return fig
# def find_oscillation_boundary_Pallidostriatal(g_list,g_loop, g_ratio, nuclei_dict, G, A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle = [6,10], find_stable_oscill = False):
#     ''' find the synaptic strength for a given set of parametes where you oscillations appear after increasing external input'''
#     got_it = False ;g_stable = None; g_transient = None
#     Proto = nuclei_dict['Proto']; D2 = nuclei_dict['D2']; FSI = nuclei_dict['FSI']; 
#     for g in reversed(g_list):
#         G[('Proto','D2')] =  0
#         G[('D2', 'FSI')] =  -1
#         G[('FSI', 'Proto')] = -1
#         G[('Proto', 'Proto')] = -1
#         nuclei_dict = reinitialize_nuclei(nuclei_dict,G,A, A_mvt, D_mvt,t_mvt, t_list, dt)
#         run(receiving_class_dict,t_list, dt, nuclei_dict)
#         test_1 = nuclei_dict['Proto'][0] ; test = nuclei_dict['D2'][0]
#         n_half_cycles_mvt,perc_oscil_mvt, f_mvt, if_stable_mvt = find_freq_of_pop_act_spec_window(test,*duration_mvt,dt, peak_threshold =test.oscil_peak_threshold, smooth_kern_window = test.smooth_kern_window, check_stability= find_stable_oscill)
#         n_half_cycles_base, perc_oscil_base, f_base, if_stable_base = find_freq_of_pop_act_spec_window(test,*duration_base,dt, peak_threshold =test.oscil_peak_threshold, smooth_kern_window = test.smooth_kern_window, check_stability= find_stable_oscill)
#         print('g=',round(g,1), g_ratio,round(f_base,1),n_half_cycles_base, round(f_mvt,1), n_half_cycles_mvt)
#         if len(np.argwhere(test_1.pop_act[duration_mvt[0]:duration_mvt[1]] ==0 )) > (duration_mvt[1]-duration_mvt[0])/2 or len(np.argwhere(test.pop_act[duration_mvt[0]:duration_mvt[1]] ==0 )) > (duration_mvt[1]-duration_mvt[0])/2:
#             print('zero activity')

#         if n_half_cycles_mvt >= lim_n_cycle[0] and n_half_cycles_mvt <= lim_n_cycle[1]:
# #            plot(nuclei_dict['Proto'], nuclei_dict['STN'], dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
#             got_it = True
            
#             if g_transient ==None:
#                 g_transient = g ; n_half_cycles = n_half_cycles_mvt
#                 print("Gotcha transient!")
#             if  not find_stable_oscill:
#                 break
#         if if_stable_mvt and find_stable_oscill:
#             got_it = True
#             g_stable = g
#             #print('trans', g_transient)
#             nuclei_dict = initialize_pallidostriatal(nuclei_dict, g_transient,g_ratio,A, A_mvt, D_mvt,t_mvt, t_list, dt)
#             break
            
#     if not got_it:
#         a = 1/0 # to bump a division zero error showing that oscillation couldn't be found in the g range
#     return n_half_cycles,g_transient,g_stable, nuclei_dict,if_stable_mvt

# def find_oscillation_boundary_STN_GPe(g_list,g_ratio,nuclei_dict, G,A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle = [6,10], find_stable_oscill = False):
#     ''' find the synaptic strength for a given set of parametes where you oscillations appear after increasing external input'''
#     got_it = False ;g_stable = None; g_transient = None
#     if np.average(g_list) < 0 : g_list = reversed(g_list)
#     for g in g_list:
#         G[('STN','Proto')] = g
#         G[('Proto','Proto')] = g * g_ratio
#         nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)

#         run(receiving_class_dict,t_list, dt, nuclei_dict)
#         Proto_test = nuclei_dict['Proto'][0] ; STN_test = nuclei_dict['STN'][0]
#         n_half_cycles_mvt,perc_oscil_mvt, f_mvt, if_stable_mvt = find_freq_of_pop_act_spec_window(STN_test,*duration_mvt,dt, peak_threshold =STN_test.oscil_peak_threshold, smooth_kern_window = STN_test.smooth_kern_window, check_stability= find_stable_oscill)
#         n_half_cycles_base, perc_oscil_base, f_base, if_stable_base = find_freq_of_pop_act_spec_window(STN_test,*duration_base,dt, peak_threshold =STN_test.oscil_peak_threshold, smooth_kern_window = STN_test.smooth_kern_window, check_stability= find_stable_oscill)
#         print('g=',round(g,1), round(Proto_test.synaptic_weight['Proto','STN'],2),round(f_base,1),n_half_cycles_base, round(f_mvt,1), n_half_cycles_mvt)
#         if len(np.argwhere(Proto_test.pop_act[duration_mvt[0]:duration_mvt[1]] ==0 )) > (duration_mvt[1]-duration_mvt[0])/2 or len(np.argwhere(STN_test.pop_act[duration_mvt[0]:duration_mvt[1]] ==0 )) > (duration_mvt[1]-duration_mvt[0])/2:
#             print('zero activity')
       
#         if n_half_cycles_mvt >= lim_n_cycle[0] and n_half_cycles_mvt <= lim_n_cycle[1]:
# #            plot(nuclei_dict['Proto'], nuclei_dict['STN'], dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None)
#             got_it = True
            
#             if g_transient ==None:
#                 g_transient = g ; n_half_cycles = n_half_cycles_mvt
#                 print("Gotcha transient!")
#             if  not find_stable_oscill:
#                 break
#         if if_stable_mvt and find_stable_oscill:
#             got_it = True
#             g_stable = g
#             print("Gotcha stable!")
#             #print('trans', g_transient)
#             G[('STN','Proto')] = g_transient
#             G[('Proto','Proto')] = g_transient * g_ratio
#             nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt) # return the nuclei at the oscillation boundary state
#             break
            
#     if not got_it:
#         a = 1/0 # to throw a division zero error showing that oscillation couldn't be found in the g range
#     return n_half_cycles,g_transient,g_stable, nuclei_dict,if_stable_mvt

# def sweep_time_scales_STN_GPe(g_list,g_ratio,nuclei_dict, GABA_A,GABA_B, Glut, filename, G, A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill):

#     def set_time_scale(nuclei_dict, inhibitory_trans_1, inhibitory_trans_1_val, inhibitory_trans_2, inhibitory_trans_2_val, glut ):
#         for nucleus_list in nuclei_dict.values():
#             for nucleus in nucleus_list:
#                 if nucleus.name == 'Proto': nucleus.synaptic_time_constant = {inhibitory_trans_1 : inhibitory_trans_1_val, inhibitory_trans_2 : inhibitory_trans_2_val}
#                 if nucleus.name == 'STN': nucleus.tau = {'Glut': glut} 
#         return nuclei_dict
#     data  = create_data_dict(nuclei_dict, [len(GABA_A), len(GABA_B), len(Glut)], 3 ,len(t_list))    
#     count = 0
#     i = 0
#     for gaba_a in GABA_A:
#         j = 0
#         for gaba_b in GABA_B:
#             m = 0
#             for glut in Glut:
#                 set_time_scale(nuclei_dict, 'GABA_A', gaba_a,'GABA_B', gaba_b, glut )
#                 print('GABA-A',gaba_a, 'GABA-B = ', gaba_b, 'glut = ', glut)
#                 n_half_cycle,g_transient,g_stable, nuclei_dict, if_stable = find_oscillation_boundary_STN_GPe(g_list,g_ratio, nuclei_dict, G, A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle =  lim_n_cycle , find_stable_oscill=find_stable_oscill)
#                 run(receiving_class_dict,t_list, dt, nuclei_dict)
#                 data['tau'][i,j,m,:] = [gaba_a, gaba_b, glut]
#                 nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
#                 for nucleus in nucleus_list:
#                     data[(nucleus.name, 'g_transient')][i,j] = g_transient
#                     data[(nucleus.name, 'g_stable')][i,j] = g_stable
#                     data[(nucleus.name,'trans_n_half_cycle')][i,j,m] = n_half_cycle
#                     data[(nucleus.name,'trans_pop_act')][i,j,m,:] = nucleus.pop_act
#                     _,_, data[nucleus.name,'trans_mvt_freq'][i,j,m],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt ,dt,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                     _,_, data[nucleus.name,'trans_base_freq'][i,j,m],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
                
#                 if find_stable_oscill: # only run if you want to checkout the stable oscillatory regime
#                     G[('STN','Proto')] = g_stable
#                     G[('Proto','Proto')] = g_stable * g_ratio
#                     nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
#                     run(receiving_class_dict,t_list, dt, nuclei_dict)
#                     for nucleus in nucleus_list:
#                         _,_, data[nucleus.name,'stable_mvt_freq'][i,j,m],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt, dt,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                         _,_, data[nucleus.name,'stable_base_freq'][i,j,m],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                 count +=1
#                 print(count, "from ", len(GABA_A)*len(GABA_B)*len(Glut))
#                 m =+1
#             j += 1
#         i+=1
#     output = open(filename, 'wb')
#     pickle.dump(data, output)
#     output.close()
    
# def sweep_time_scales_one_GABA_STN_GPe(g_list, g_ratio, nuclei_dict, inhibitory_trans,inhibitory_series, Glut, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill):
#     def set_time_scale(gaba,glut, nuclei_dict, inhibitory_trans):
#         for nucleus_list in nuclei_dict.values():
#             for nucleus in nucleus_list:
#                 if nucleus.name == 'Proto': nucleus.tau = {inhibitory_trans : gaba}
#                 if nucleus.name == 'STN': nucleus.tau = {'Glut': glut} 
#         return nuclei_dict
#     data  = create_data_dict(nuclei_dict, [len(inhibitory_series), len(Glut)], 2,len(t_list))    
#     i = 0 ; count =0
#     for gaba in inhibitory_series:
#         j=0
#         for glut in Glut:
#             nuclei_dict = set_time_scale(gaba,glut, nuclei_dict, inhibitory_trans)
#             print('GABA = ', gaba, 'glut = ', glut)
#             n_half_cycle,g_transient,g_stable, nuclei_dict, if_stable = find_oscillation_boundary_STN_GPe(g_list,g_ratio, nuclei_dict, G, A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle =  lim_n_cycle , find_stable_oscill=find_stable_oscill)
#             run(receiving_class_dict,t_list, dt, nuclei_dict)
#             data['tau'][i,j,:] = [gaba, glut]
            
#             nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
# #                plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$\tau_{GABA_A}$ = "+ str(round(gaba_a,2))+r' $\tau_{GABA_B}$ ='+str(round(gaba_b,2)))
#             for nucleus in nucleus_list:
#                 data[(nucleus.name, 'g_transient')][i,j] = g_transient
#                 data[(nucleus.name, 'g_stable')][i,j] = g_stable
#                 data[(nucleus.name,'trans_n_half_cycle')][i,j] = n_half_cycle
#                 data[(nucleus.name,'trans_pop_act')][i,j,:] = nucleus.pop_act
#                 _,_, data[nucleus.name,'trans_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                 _,_, data[nucleus.name,'trans_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
            
#             if find_stable_oscill: # only run if you want to checkout the stable oscillatory regime
#                 G[('STN','Proto')] = g_stable
#                 G[('Proto','Proto')] = g_stable * g_ratio
#                 nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
#                 run(receiving_class_dict,t_list, dt, nuclei_dict)
#                 for nucleus in nucleus_list:
#                     _,_, data[nucleus.name,'stable_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                     _,_, data[nucleus.name,'stable_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)

#             count +=1
#             print(count, "from ", len(inhibitory_series)*len(Glut))
#             j+=1
#         i+=1
#     output = open(filename, 'wb')
#     pickle.dump(data, output)
#     output.close()
    
# def sweep_time_scale_and_g_one_GABA_Pallidostriatal(g_list, nuclei_dict, inhibitory_trans,inhibitory_series, Glut, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill):

#     data  = create_data_dict(nuclei_dict, [len(inhibitory_series), len(Glut)], 2,t_list)    
#     count = 0
#     i = 0
#     for gaba in inhibitory_series:
#         j=0
#         for g_ratio in g_ratio_list:
#             # G[('Proto','D2')] =  0
#             # G[('D2', 'FSI')] =  -1
#             # G[('FSI', 'Proto')] = -1
#             # G[('Proto', 'Proto')] = -1
#             nuclei_dict = reinitialize_nuclei(nuclei_dict, G,A, A_mvt, D_mvt,t_mvt, t_list, dt)            
#             print('GABA = ', gaba, 'glut = ', glut)
#             nuclei_dict = {'Proto': Proto, 'STN' : STN}
#             n_half_cycle,g_transient,g_stable, nuclei_dict, if_stable = find_oscillation_boundary_STN_GPe(g_list,g_ratio,nuclei_dict, G, A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle =  lim_n_cycle , find_stable_oscill=find_stable_oscill)

#             run(receiving_class_dict,t_list, dt, nuclei_dict)
#             data['tau'][i,j,:] = [gaba, glut]
#             nucleus_list =[v for v in nuclei_dict.values()[0]]
# #                plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$\tau_{GABA_A}$ = "+ str(round(gaba_a,2))+r' $\tau_{GABA_B}$ ='+str(round(gaba_b,2)))
#             for nucleus in nucleus_list:
#                 data[(nucleus.name,'trans_n_half_cycle')][i,j] = n_half_cycle
#                 data[(nucleus.name,'trans_pop_act')][i,j,:] = nucleus.pop_act
#                 _,_, data[nucleus.name,'trans_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                 _,_, data[nucleus.name,'trans_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
            
#             if find_stable_oscill: # only run if you want to checkout the stable oscillatory regime
#                 nuclei_dict = initialize_pallidostriatal(nuclei_dict, g_stable,g_ratio,A, A_mvt, D_mvt,t_mvt, t_list, dt) 
#                 run(receiving_class_dict,t_list, dt, nuclei_dict)
#                 for nucleus in nucleus_list:
#                     _,_, data[nucleus.name,'stable_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                     _,_, data[nucleus.name,'stable_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)

#             count +=1
#             # for nuclei_list in nuclei_dict.values():
#             #     for nucleus in nuclei_list:
#             #         nucleus.clear_history()
#             print(count, "from ", len(inhibitory_series)*len(Glut))
#             j+=1
#         i+=1
#     output = open(filename, 'wb')
#     pickle.dump(data, output)
#     output.close()