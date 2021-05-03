from __future__ import division
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
import decimal
from decimal import *
from scipy import optimize
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
from scipy.stats import truncexpon

# matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')
#from scipy.ndimage.filters import generic_filter
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = mticker.FuncFormatter(g)

def find_sending_pop_dict(receiving_pop_list):
    sending_pop_list = {k: [] for k in receiving_pop_list.keys()}
    for k,v_list in receiving_pop_list.items():
        for v in v_list:
            sending_pop_list[v].append(k)
    return sending_pop_list
class Nucleus:

    def __init__(self, population_number,gain, threshold,neuronal_consts,tau, ext_inp_delay, noise_variance, noise_amplitude, 
        N, A,A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_from_list,smooth_kern_window,oscil_peak_threshold, syn_input_integ_method = 'exp_rise_and_decay',
        neuronal_model = 'rate',poisson_prop = None,AUC_of_input = None, init_method = 'homogeneous', ext_inp_method = 'const+noise', der_ext_I_from_curve =False,
        bound_to_mean_ratio = [0.8 , 1.2], spike_thresh_bound_ratio = [1/20, 1/20], ext_input_integ_method = 'dirac_delta_input'):
        n_timebins = int(t_sim/dt)
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
        self.pop_act = np.zeros(( n_timebins )) # time series of population activity
        self.rest_ext_input = None
        self.mvt_ext_input = np.zeros(( n_timebins )) # external input mimicing movement
        self.external_inp_t_series = np.zeros(( n_timebins ))
        self.avg_pop_act_mvt = None
        self.avg_pop_act_base = None
        self.connectivity_matrix = {}
        self.noise_induced_basal_firing = None # the basal firing as a result of noise at steady state
        self.oscil_peak_threshold = oscil_peak_threshold[self.name]
        self.smooth_kern_window = smooth_kern_window[self.name]
        self.set_noise_param(noise_variance, noise_amplitude)
        self.init_method = init_method
        self.neuronal_model = neuronal_model
        self.frequency = None
        self.perc_oscil = None
        self.trim_sig_method_dict = {'spiking': 'simple', 'rate': 'neat'}
        if neuronal_model == 'spiking':
            self.spikes = np.zeros((self.n,int(t_sim/dt)),dtype = int)
            ## dt incorporated in tau for efficiency
            self.tau = {k: {kk: np.array(vv)/dt for kk, vv in tau[k].items()} for k, v in tau.items() if k[0]==name} # filter based on the receiving nucleus
            self.I_rise = {k: np.zeros((self.n,len(self.tau[self.name,k[0]]['decay']))) for k in self.receiving_from_list} # since every connection might have different rise and decay time, inputs must be updataed accordincg to where the input is coming from
            self.I_syn = {k: np.zeros((self.n,len(self.tau[self.name,k[0]]['decay']))) for k in self.receiving_from_list}
            self.I_syn['ext_pop','1'] = np.zeros(self.n,)
            self.I_rise['ext_pop','1'] = np.zeros(self.n,)
            self.neuronal_consts = neuronal_consts[self.name]
            self.u_rest = self.neuronal_consts['u_rest']
            self.mem_pot_before_spike = np.zeros(self.n)
            self.syn_inputs = {k: np.zeros((self.n,1)) for k in self.receiving_from_list}
            self.poisson_spikes = None # meant to store the poisson spike trains of the external population
            self.n_ext_population = poisson_prop[self.name]['n'] # external population size
            self.firing_of_ext_pop = poisson_prop[self.name]['firing'] 
            self.syn_weight_ext_pop = poisson_prop[self.name]['g']
            self.FR_ext = 0
            self.representative_inp = {k: np.zeros(( n_timebins ,len(self.tau[self.name,k[0]]['decay']))) for k in self.receiving_from_list}
            self.representative_inp['ext_pop','1'] = np.zeros( n_timebins )
            self.ext_input_all = np.zeros((self.n, n_timebins ))
            self.voltage_trace = np.zeros(n_timebins )
            self.AUC_of_input = AUC_of_input
            self.rest_ext_input = np.zeros(self.n)
            self.ext_inp_method = ext_inp_method
            self.der_ext_I_from_curve = der_ext_I_from_curve # if derive external input value from response curve
            self.sum_syn_inp_at_rest = None
            self.all_mem_pot = np.zeros((self.n, n_timebins ))
            self.ext_input_integ_method = ext_input_integ_method
            if init_method == 'homogeneous':
                self.initialize_homogeneously( poisson_prop, dt)
            elif init_method == 'heterogeneous':
                self.initialize_heterogeneously( poisson_prop, dt, spike_thresh_bound_ratio, * bound_to_mean_ratio )

            self.ext_inp_method_dict = {'Poisson': self.poissonian_ext_inp, 'const+noise' : self.constant_ext_input_with_noise, 'constant' : self.constant_ext_input}
            self.input_integ_method_dict = {'exp_rise_and_decay' : exp_rise_and_decay, 'instantaneus_rise_expon_decay' : instantaneus_rise_expon_decay , 'dirac_delta_input': _dirac_delta_input}
            self.syn_input_integ_method = syn_input_integ_method
            self.normalize_synaptic_weight()


    def initialize_heterogeneously(self, poisson_prop, dt, spike_thresh_bound_ratio, lower_bound_perc = 0.8, upper_bound_perc = 1.2):
        ''' cell properties and boundary conditions come from distributions'''

        # self.mem_potential = np.random.uniform(low = self.neuronal_consts['u_initial']['min'], high = self.neuronal_consts['u_initial']['max'], size = self.n) # membrane potential


        self.spike_thresh = truncated_normal_distributed ( self.neuronal_consts['spike_thresh']['mean'],
                                                            self.neuronal_consts['spike_thresh']['var'] , self.n,
                                                            scale_bound = scale_bound_with_arbitrary_value, scale = (self.neuronal_consts['spike_thresh']['mean'] - self.u_rest),
                                                            lower_bound_perc = spike_thresh_bound_ratio[0], upper_bound_perc = spike_thresh_bound_ratio[1] )
        self.mem_potential = np.random.uniform(low = self.u_rest, high = self.spike_thresh , size = self.n) # membrane potential
        # lower, upper, scale = 0 , self.neuronal_consts['spike_thresh']['mean'] - self.u_rest , 30 ### Doesn't work with linear interpolation of IF
        # X = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
        # self.mem_potential = self.neuronal_consts['spike_thresh']['mean'] - X.rvs(self.n) 

        # plt.figure()
        # plt.hist(self.mem_potential, bins = 50)
        # plt.title(self.name, fontsize = 15)
        # plt.xlabel(r'initial $V_{m}$', fontsize = 15)
        # plt.ylabel( 'Pr', fontsize = 15)
        self.all_mem_pot[:,0] = self.mem_potential.copy()
        self.membrane_time_constant = truncated_normal_distributed ( self.neuronal_consts['membrane_time_constant']['mean'], 
                                                             self.neuronal_consts['membrane_time_constant']['var'] , self.n,
                                                             lower_bound_perc = lower_bound_perc, upper_bound_perc = upper_bound_perc)

        ## dt incorporated in tau for efficiency
        self.tau_ext_pop = {'rise': truncated_normal_distributed ( poisson_prop[self.name]['tau']['rise']['mean'],
                                                                poisson_prop[self.name]['tau']['rise']['var'] , self.n,
                                                                lower_bound_perc = lower_bound_perc, upper_bound_perc = upper_bound_perc) / dt,
                            'decay':truncated_normal_distributed ( poisson_prop[self.name]['tau']['decay']['mean'],
                                                                poisson_prop[self.name]['tau']['decay']['var'] , self.n, 
                                                                lower_bound_perc = lower_bound_perc, upper_bound_perc = upper_bound_perc) / dt}
            
    def initialize_homogeneously(self, poisson_prop, dt):
        ''' cell properties and boundary conditions are constant for all cells'''

        # self.mem_potential = np.random.uniform(low = self.neuronal_consts['u_initial']['min'], high = self.neuronal_consts['u_initial']['max'], size = self.n) # membrane potential
        self.spike_thresh = np.full(self.n, self.neuronal_consts['spike_thresh']['mean'])
        self.mem_potential = np.random.uniform(low = self.u_rest, high = self.spike_thresh, size = self.n) # membrane potential
        self.all_mem_pot[:,0] = self.mem_potential

        self.membrane_time_constant = np.full(self.n ,  
                                             self.neuronal_consts['membrane_time_constant']['mean'])
        self.tau_ext_pop = {'rise': np.full(self.n,poisson_prop[self.name]['tau']['rise']['mean'])/dt,# synaptic decay time of the external pop inputs
                            'decay':np.full(self.n, poisson_prop[self.name]['tau']['decay']['mean'])/dt}

    def normalize_synaptic_weight(self):
        self.synaptic_weight = {k: v / (self.neuronal_consts['spike_thresh']['mean'] - self.u_rest) for k, v in self.synaptic_weight.items() if k[0]==self.name}

    def calculate_input_and_inst_act(self, t, dt, receiving_from_class_list, mvt_ext_inp):  
        ## to do

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

    def cal_ext_inp(self,dt,t):

        I_ext = self.ext_inp_method_dict[self.ext_inp_method](dt) # choose method of exerting external input from dictionary of methods

        self.I_syn['ext_pop','1'],self.I_rise['ext_pop','1'] = self.input_integ_method_dict[ self. ext_input_integ_method ] ( I_ext, 
                                                                            I_rise = self.I_rise['ext_pop','1'], 
                                                                            I = self.I_syn['ext_pop','1'],
                                                                            tau_rise = self.tau_ext_pop['rise'], 
                                                                            tau_decay = self.tau_ext_pop['decay'])
    def poissonian_ext_inp(self, dt):

        # poisson_spikes = possion_spike_generator(self.n,self.n_ext_population,self.firing_of_ext_pop,dt)
        poisson_spikes = possion_spike_generator(self.n , self.n_ext_population , self.FR_ext , dt )
        I_ext =  self.cal_input_from_poisson_spikes( poisson_spikes, dt )
        return I_ext

    def constant_ext_input_with_noise(self, dt ):

        return self.rest_ext_input + noise_generator(self.noise_amplitude, self.noise_variance, self.n).reshape(-1,)

    def constant_ext_input(self, dt ):

        return self.rest_ext_input 

    def cal_input_from_poisson_spikes(self, poisson_spikes, dt):

        return ( np.sum( poisson_spikes , axis = 1) / dt # normalize dirac delta spike amplitude
               * self.syn_weight_ext_pop 
               * self.membrane_time_constant 
               ).reshape(-1,)

    def cal_synaptic_input(self,dt,projecting, t):

        num = str(projecting.population_num)
        name = projecting.name
        self.syn_inputs[name,num] = (
                                    (self.synaptic_weight[(self.name,name)] * 
                                    np.matmul( 
                                        self.connectivity_matrix[(name,num)] , 
                                        projecting.spikes[ : , int ( t - self.transmission_delay[ (self.name, name) ] / dt ) ]
                                            ) / dt
                                    * self.membrane_time_constant).reshape(-1,)
                                   )     

    def sum_synaptic_input(self, receiving_from_class_list, dt,t):
        ''''''
        synaptic_inputs = np.zeros(self.n)
        for projecting in receiving_from_class_list:

            self.cal_synaptic_input(dt, projecting,t)
            synaptic_inputs = ( synaptic_inputs +  
                              self.sum_components_of_one_synapse(t, projecting.name, str(projecting.population_num), 
                                                                pre_n_components = len ( self.tau[self.name,projecting.name]['rise'] ) ))

        return synaptic_inputs

    def sum_synaptic_input_one_step_ahead_with_no_spikes(self, receiving_from_class_list, dt):
        ''''''
        synaptic_inputs = np.zeros(self.n)
        for projecting in receiving_from_class_list:

            synaptic_inputs = ( synaptic_inputs +  
                                self.sum_components_of_one_synapse_one_step_ahead_with_no_spikes( projecting.name, 
                                                                                                str(projecting.population_num), 
                                                                                                pre_n_components = len ( self.tau[self.name,projecting.name]['rise'] ) ))
                                                                                                 

        return synaptic_inputs

    def sum_components_of_one_synapse(self, t, pre_name, pre_num, pre_n_components = 1):

        i = 0 
        sum_components = np.zeros(self.n)
        for i in range(pre_n_components):
            self.I_syn[pre_name,pre_num][:,i], self.I_rise[pre_name,pre_num][:,i] = self.input_integ_method_dict [self.syn_input_integ_method](self.syn_inputs[pre_name,pre_num], 
                                                                                                        I_rise = self.I_rise[pre_name,pre_num][:,i], 
                                                                                                        I = self.I_syn[pre_name,pre_num][:,i], 
                                                                                                        tau_rise = self.tau[(self.name,pre_name)]['rise'][i],
                                                                                                        tau_decay = self.tau[(self.name,pre_name)]['decay'][i])
            self.representative_inp[pre_name,pre_num][t,i] = self.I_syn[pre_name,pre_num][0,i]
            sum_components = sum_components + self.I_syn[pre_name,pre_num][:,i]
            i += 1
        return sum_components

    def sum_components_of_one_synapse_one_step_ahead_with_no_spikes(self, pre_name, pre_num, pre_n_components = 1):
        '''Calculate I_syn(t+dt) assuming that there are no spikes between time t and t+dt '''

        i = 0 
        sum_components = np.zeros(self.n)
        for i in range(pre_n_components):

            I_syn_next_dt,_ = self.input_integ_method_dict [self.syn_input_integ_method]( 0   , 
                                                I_rise = self.I_rise[pre_name,pre_num][:,i], 
                                                I = self.I_syn[pre_name,pre_num][:,i], 
                                                tau_rise = self.tau[(self.name,pre_name)]['rise'][i],
                                                tau_decay = self.tau[(self.name,pre_name)]['decay'][i])

            sum_components = sum_components + I_syn_next_dt
            i += 1
        return sum_components

    def solve_IF_without_syn_input(self,t,dt,receiving_from_class_list,mvt_ext_inp = None):

        self.cal_ext_inp(dt,t)
        synaptic_inputs = np.zeros(self.n)
        self.update_potential(synaptic_inputs, dt, receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        # self.reset_potential(spiking_ind)
        self.reset_potential_with_interpolation(spiking_ind,dt)
        self.cal_population_activity(dt,t)

    def solve_IF(self,t,dt,receiving_from_class_list,mvt_ext_inp = None):

        self.cal_ext_inp(dt,t)
        synaptic_inputs = self.sum_synaptic_input(receiving_from_class_list,dt,t)
        self.update_potential(synaptic_inputs, dt, receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        self.reset_potential(spiking_ind)
        # self.reset_potential_with_interpolation(spiking_ind,dt)
        self.cal_population_activity(dt,t)
        self.update_representative_measures(t)
        self.all_mem_pot[:,t] = self.mem_potential

    def update_representative_measures(self,t):

        self.voltage_trace[t] = self.mem_potential[0]
        self.representative_inp['ext_pop','1'][t] = self.I_syn['ext_pop','1'][0]
        self.ext_input_all[:,t] = self.I_syn['ext_pop','1']

    def update_potential(self, synaptic_inputs, dt, receiving_from_class_list):

        ###### EIF
        #self.mem_potential += (-self.mem_potential+ inputs+ self.neuronal_consts['nonlin_sharpness'] *np.exp((self.mem_potential-
        #                       self.neuronal_consts['nonlin_thresh'])/self.neuronal_consts['nonlin_sharpness']))*dt/self.membrane_time_constant
        ###### LIF
        self.mem_pot_before_spike = self.mem_potential.copy()
        V_prime = f_LIF(self.membrane_time_constant, self.mem_potential, self.u_rest, self.I_syn['ext_pop','1'], synaptic_inputs)
        # self.mem_potential = fwd_Euler(dt, self.mem_potential, V_prime)
        I_syn_next_dt = self. sum_synaptic_input_one_step_ahead_with_no_spikes( receiving_from_class_list, dt)
        self.mem_potential = Runge_Kutta_second_order_LIF( dt, self.mem_potential, V_prime,  self.membrane_time_constant , I_syn_next_dt, self.u_rest, self.I_syn['ext_pop','1'])
    def find_spikes(self,t):

        # spiking_ind = np.where(self.mem_potential > self.neuronal_consts['spike_thresh']['mean']) # homogeneous spike thresholds
        spiking_ind = np.where(self.mem_potential > self.spike_thresh) # gaussian distributed spike thresholds
        self.spikes[spiking_ind, t] = 1
        return spiking_ind

    def reset_potential(self,spiking_ind):

        self.mem_potential[spiking_ind] = self.neuronal_consts['u_rest']

    def reset_potential_with_interpolation(self,spiking_ind,dt):
        ''' set the potential at firing times according to Hansel et. al. (1998)'''
        self.mem_potential[spiking_ind] = linear_interpolation( self.mem_potential[spiking_ind], self.spike_thresh[spiking_ind], 
                                                                dt, self.mem_pot_before_spike[spiking_ind], self.neuronal_consts['u_rest'], self.membrane_time_constant[spiking_ind])

    def cal_population_activity(self,dt,t, lower_bound_perc = 0.8, upper_bound_perc = 1.2):

        self.pop_act[t] = np.average(self.spikes[:, t],axis = 0)/(dt/1000)

    def reset_ext_pop_properties(self,poisson_prop,dt):
        '''reset the properties of the external poisson spiking population'''

        self.n_ext_population = poisson_prop[self.name]['n'] # external population size
        self.firing_of_ext_pop = poisson_prop[self.name]['firing'] 
        
        self.tau_ext_pop = {'rise': truncated_normal_distributed ( poisson_prop[self.name]['tau']['rise']['mean'],
                                                                poisson_prop[self.name]['tau']['rise']['var'] , self.n,
                                                                lower_bound_perc = lower_bound_perc, upper_bound_perc = upper_bound_perc) / dt,
                            'decay':truncated_normal_distributed ( poisson_prop[self.name]['tau']['decay']['mean'],
                                                                poisson_prop[self.name]['tau']['decay']['var'] , self.n, 
                                                                lower_bound_perc = lower_bound_perc, upper_bound_perc = upper_bound_perc) / dt}
        self.syn_weight_ext_pop = poisson_prop[self.name]['g']
        # self.representative_inp['ext_pop','1'] = np.zeros((int(t_sim/dt)))

    def set_noise_param(self, noise_variance, noise_amplitude):

        self.noise_variance =  noise_variance[self.name ] #additive gaussian white noise with mean zero and variance sigma
        self.noise_amplitude =  noise_amplitude[self.name ] #additive gaussian white noise with mean zero and variance sigma

    def set_connections(self, K, N):
        ''' creat Jij connection matrix'''

        self.K_connections = {k: v for k, v in K.items() if k[0]==self.name}
        for projecting in self.receiving_from_list:
            
            n_connections = self.K_connections[(self.name, projecting[0])]
            self.connectivity_matrix[projecting] = build_connection_matrix(self.n, N[projecting[0]], n_connections)

    def clear_history(self):

        self.pop_act = np.zeros_like(self.pop_act) 
        if self.neuronal_model == 'rate':
            # self.output = {k: np.zeros_like(self.output[k]) for k in self.output.keys()}
            # self.input = np.zeros_like(self.input)
            # self.neuron_act = np.zeros_like(self.neuron_act)
            # self.mvt_ext_input = np.zeros_like(self.mvt_ext_input) 
            # self.external_inp_t_series = np.zeros_like(self.external_inp_t_series) 
            for k in self.output.keys():
                self.output[k][:] = 0
            self.input[:] = 0
            self.neuron_act[:] = 0
            self.mvt_ext_input = np.zeros_like(self.mvt_ext_input) 
            self.external_inp_t_series[:] = 0

        if self.neuronal_model == 'spiking':
            for k in self.receiving_from_list:
                self.I_rise[k][:] = 0
                self.I_syn[k][:] = 0
                self.representative_inp[k][:] = 0
                self.syn_inputs[k][:] = 0
            self.pop_act[:] = 0
            self.I_syn['ext_pop','1'][:] = 0
            self.voltage_trace[:] = 0
            self.representative_inp['ext_pop','1'][:] = 0
            self.mem_potential = np.random.uniform(low= self.neuronal_consts['u_initial']['min'],high = self.neuronal_consts['u_initial']['max'],size = self.n)
            self.spikes[:,:] = 0

    def smooth_pop_activity(self, dt, window_ms = 5):
        self.pop_act = moving_average_array( self.pop_act, int(window_ms / dt) )

    def average_pop_activity(self, t_list, last_fraction = 1/2):
        average = np.average(self.pop_act [ int( len(t_list) * last_fraction) : ] )
        std = np.std( self.pop_act[ int( len(t_list) * last_fraction) : ] )
        return average, std

    def set_synaptic_weights(self,G):

        self.synaptic_weight = {k: v for k, v in G.items() if k[0]==self.name} # filter based on the receiving nucleus

    def set_synaptic_time_scales(self,synaptic_time_constant):

        self.synaptic_time_constant = {k: v for k, v in synaptic_time_constant.items() if k[1]==self.name}

    def set_ext_input(self,A, A_mvt, D_mvt,t_mvt, t_list, dt):

        proj_list = [k[0] for k in list(self.receiving_from_list)]

        if self.neuronal_model == 'rate':
            
            self.rest_ext_input = self.basal_firing/self.gain - np.sum([self.synaptic_weight[self.name,proj]*A[proj] for proj in proj_list]) + self.threshold
            self.mvt_ext_input = self.mvt_firing/self.gain - np.sum([self.synaptic_weight[self.name,proj]*A_mvt[proj] for proj in proj_list]) + self.threshold - self.rest_ext_input
            self.external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,self.ext_inp_delay,self.mvt_ext_input, t_list*dt)

        else: # for the firing rate model the ext input is reported as the firing rate of the ext pop needed.
            
            I_syn = np.sum([self.synaptic_weight[self.name,proj]*A[proj]/1000*self.K_connections[self.name,proj] for proj in proj_list])*self.membrane_time_constant
            print('I_syn', np.average(I_syn))
            self.sum_syn_inp_at_rest = I_syn
            if self.ext_inp_method == 'Poisson':
                self._set_ext_inp_poisson( I_syn)

            elif self.ext_inp_method == 'const+noise' or self.ext_inp_method == 'const' :
                self._set_ext_inp_const_plus_noise(I_syn)
            else: 
                raise ValueError('external input handling method not right!')

            # print('mean rest_ext_input', np.average(self.rest_ext_input))
            # self.rest_ext_input_of_mean = (self.neuronal_consts['spike_thresh']['mean'] - self.neuronal_consts['u_rest'])/ (1-exp)/(self.syn_weight_ext_pop*self.n_ext_population*self.neuronal_consts['membrane_time_constant']['mean'])

            # print('I_ext', np.average(self.rest_ext_input), 'without iput', np.average(
                # ((self.spike_thresh - self.u_rest)/ (1-exp) )/self.syn_weight_ext_pop/self.n_ext_population/self.membrane_time_constant), 'I_syn', np.average(I_syn))
            # print(self.name, np.average(I_syn), np.average(self.rest_ext_input))

            # temp = ((self.spike_thresh - self.u_rest))/self.syn_weight_ext_pop/self.n_ext_population/self.membrane_time_constant
            # exp = np.exp(-dt/self.neuronal_consts['membrane_time_constant']['mean'])
            # I_ext_stable = self.neuronal_consts['spike_thresh']['mean'] - self.neuronal_consts['u_rest']
            # self.rest_ext_input = I_ext_stable/self.n_ext_population/self.syn_weight_ext_pop*(1-exp)/(1-exp**(1/(self.basal_firing/1000*dt)+1))#*self.neuronal_consts['membrane_time_constant']['mean']
            # print('ext_inp=',np.average(self.rest_ext_input))

            # exp = np.exp(-self.membrane_time_constant*(self.mvt_firing/1000))
            # self.mvt_ext_input = ((self.spike_thresh - self.u_rest*exp)/ (1-exp) - 
            #     np.sum([self.synaptic_weight[self.name,proj]*A_mvt[proj]/dt/1000*self.K_connections[self.name,proj] for proj in proj_list]))/self.syn_weight_ext_pop/self.n_ext_population/1000

            ## non array
            # exp = np.exp(-1/(self.neuronal_consts['membrane_time_constant']['mean']*self.basal_firing/1000))
            # self.rest_ext_input = ((self.neuronal_consts['spike_thresh']['mean'] - self.neuronal_consts['u_rest'])/ (1-exp))/(self.syn_weight_ext_pop*self.n_ext_population*self.neuronal_consts['membrane_time_constant']['mean'])
            # print(self.name)
            # print(decimal.Decimal.from_float(self.rest_ext_input[0]))
            # print(decimal.Decimal.from_float(temp[0]))#, 'ext_inp=',self.rest_ext_input)

        # self.external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,self.ext_inp_delay,self.mvt_ext_input, t_list*dt)

    def _set_ext_inp_poisson(self, I_syn):
        exp = np.exp(-1/(self.membrane_time_constant*self.basal_firing/1000))
        self.rest_ext_input = ( (self.spike_thresh - self.u_rest) / (1-exp) - I_syn)
        self.FR_ext = self.rest_ext_input / self.syn_weight_ext_pop / self.n_ext_population / self.membrane_time_constant

    def _set_ext_inp_const_plus_noise(self, I_syn):
        if self.basal_firing > 25 : # linear regime:
            self._set_ext_inp_poisson( I_syn)
        else:
            self.rest_ext_input =  self.FR_ext * self.syn_weight_ext_pop * self.n_ext_population * self.membrane_time_constant - I_syn

    def estimate_needed_external_input(self, all_FR_list, dt, t_list, receiving_class_dict, if_plot = False, end_of_nonlinearity = 25,
                                        n_FR = 50, left_pad = 0.001, right_pad =0.001, maxfev = 5000):

        all_FR_list_2D = np.repeat(all_FR_list.reshape(-1,1), self.n, axis = 1) # provide the array for all neurons
        FR_sim = self.Find_threshold_of_firing( all_FR_list_2D, t_list, dt, receiving_class_dict )
        FR_list = find_FR_ext_range_for_each_neuron(FR_sim, all_FR_list, self.init_method, n_FR = n_FR, left_pad = left_pad, right_pad = right_pad)
        FR_sim = self.run_for_all_FR_ext( FR_list, t_list, dt, receiving_class_dict )
        self. set_FR_ext_each_neuron( FR_list, FR_sim, dt,  if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity, maxfev = maxfev)
        self.clear_history()

    def run_for_all_FR_ext(self, FR_list, t_list, dt, receiving_class_dict ):

        FR_sim = np.zeros((self.n, len(FR_list)))

        for i in range( FR_list.shape[0] ) :
            
            self.clear_history()
            self.rest_ext_input = FR_list [i,:] * self.membrane_time_constant * self.n_ext_population * self.syn_weight_ext_pop
            self. run(dt, t_list, receiving_class_dict)
            FR_sim[:,i] =  np.average(self.spikes[:,int(len(t_list)/2):] , axis = 1)/(dt/1000)
            print('FR = ', np.average(FR_list [i,:]) , np.average(FR_sim[:,i]))

        return FR_sim

    def Find_threshold_of_firing(self, FR_list, t_list, dt, receiving_class_dict ):
        FR_sim = np.zeros((self.n, len(FR_list)))

        for i in range( FR_list.shape[0] ) :
            
            self.clear_history()
            self.rest_ext_input = FR_list [i,:] * self.membrane_time_constant * self.n_ext_population * self.syn_weight_ext_pop
            self. run(dt, t_list, receiving_class_dict)
            FR_sim[:,i] =  np.average(self.spikes[:,int(len(t_list)/2):] , axis = 1)/(dt/1000)
            print('FR = ', np.average(FR_list [i,:]) , np.average(FR_sim[:,i]))
            if np.all(FR_sim[:,i] > 1):
                print('done!')
                break
        return FR_sim

    def run(self, dt, t_list, receiving_class_dict):
        for t in t_list: # run temporal dynamics
            self.solve_IF_without_syn_input(t,dt,receiving_class_dict[(self.name,str(self.population_num))])

    def set_FR_ext_each_neuron(self, FR_list, FR_sim, dt,  if_plot = False, end_of_nonlinearity = 25, maxfev = 5000):

        self.FR_ext = np.zeros((self.n))

        if self.init_method == 'homogeneous' :#and FR_list.shape[1] == 1:
            rep_FR_ext, _ = extrapolate_FR_ext_from_neuronal_response_curve ( FR_list[:,0] * 1000, np.average(FR_sim, axis = 0),
                                                                            self.basal_firing, if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity, maxfev = maxfev)
            self.FR_ext = np.full( self.n, rep_FR_ext ) 
                                    
        else :
            for i in range(self.n):
                self.FR_ext[i], _ = extrapolate_FR_ext_from_neuronal_response_curve ( FR_list[:,i] * 1000, FR_sim[i,:] , self.basal_firing, 
                                                                                if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity, maxfev = maxfev)
    def scale_synaptic_weight(self):
        self.synaptic_weight = {k: v/self.K_connections[k] for k, v in self.synaptic_weight.items() if k[0]==self.name}

    def find_freq_of_pop_act_spec_window(self, start, end, dt, peak_threshold = 0.1, smooth_kern_window= 3 , cut_plateau_epsilon = 0.1, check_stability = False, method = 'zero_crossing', if_plot = False):
        ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
            the plateau and in case it is oscillation determine the frequency '''
        if method not in ["fft", "zero_crossing"]:
            raise ValueError("mode must be either 'fft', or 'zero_crossing'")

        sig = trim_start_end_sig_rm_offset(self.pop_act,start, end, method = self.trim_sig_method_dict[ self.neuronal_model ]  )
        cut_sig_ind = cut_plateau ( sig,  epsilon= cut_plateau_epsilon)
        plateau_y = find_mean_of_signal(sig, cut_sig_ind)
        _plot_signal(if_plot, start, end, dt, sig, plateau_y, cut_sig_ind)
        if_stable = False
        if len(cut_sig_ind) > 0: # if it's not all plateau from the beginning

            sig = sig - plateau_y

            if method == 'zero_crossing':

                n_half_cycles, freq = zero_crossing_freq_detect(sig[cut_sig_ind] , dt / 1000)

            elif method == 'fft' : 

                freq = freq_from_fft (sig [ cut_sig_ind ] , dt / 1000 )
                n_half_cycles = None

            if freq != 0: # then check if there's oscillations

                perc_oscil = max_non_empty_array(cut_sig_ind) / len( sig ) * 100

                if check_stability:
                    if_stable = if_stable_oscillatory(sig, max(cut_sig_ind), peak_threshold, smooth_kern_window, amp_env_slope_thresh = - 0.05)

                return n_half_cycles, perc_oscil, freq, if_stable
            else:
                return 0,0,0, False

        else:
            return 0,0,0, False

def _plot_signal(if_plot, start, end, dt, sig, plateau_y, cut_sig_ind):
    if if_plot:
        plt.figure()
        t = np.arange(start,end) * dt
        plt.plot(t,sig - plateau_y)
        plt.axhline(plateau_y)
        plt.plot( t[cut_sig_ind], sig[cut_sig_ind] )
def find_FR_ext_range_for_each_neuron(FR_sim, all_FR_list, init_method,  n_FR = 50 , left_pad = 0.005, right_pad = 0.005):
    ''' put log spaced points in the nonlinear regime of each neuron'''
    n_pop = FR_sim.shape[ 0 ]
    if init_method == 'homogeneous':

        start_act = np.average([all_FR_list[np.min( np.where( FR_sim[i,:] > 1 )[0])] for i in range(n_pop)]) # setting according to one neurons would be enough
        FR_list = np.repeat(spacing_with_high_resolution_in_the_middle(n_FR, start_act - left_pad, start_act + right_pad).reshape(-1,1),n_pop,  axis = 1)
        
    else:
        
        FR_list = np.zeros( (n_FR-1, n_pop) )
        for i in range(n_pop):
        
            start_act = all_FR_list[np.min( np.where( FR_sim[i,:] > 1 )[0])]
            FR_list[:,i] = spacing_with_high_resolution_in_the_middle(n_FR, start_act - left_pad, start_act + right_pad).reshape(-1,)

    return FR_list

def extrapolate_FR_ext_from_neuronal_response_curve ( FR_ext, FR_sim , desired_FR, if_plot = False, end_of_nonlinearity = 25, maxfev = 5000):
    ''' All firing rates in Hz'''
    # plt.figure()
    # plt.plot( FR_ext, FR_sim, '-o')
    xdata, ydata = get_non_linear_part( FR_ext, FR_sim, end_of_nonlinearity =  end_of_nonlinearity)
    x, y = rescale_x_and_y ( xdata, ydata )
    coefs = fit_FR_as_a_func_of_FR_ext ( x, y, sigmoid, maxfev = maxfev)
    FR_ext = extrapolated_FR_ext_from_fitted_curve (x, y, desired_FR, coefs, sigmoid, inverse_sigmoid, 
                                                find_y_normalizing_factor(ydata), 
                                                find_x_mid_point_sigmoid( ydata, xdata)) 
    
    if if_plot: 
        plot_fitted_curve(xdata, ydata, x, coefs)

    return FR_ext / 1000 , coefs # FR_ext is in Hz, we want spk/ms

def set_connec_ext_inp(A, A_mvt, D_mvt, t_mvt,dt, N, N_real, K_real_STN_Proto_diverse, receiving_pop_list, nuclei_dict,t_list, 
                        all_FR_list = np.linspace(0.05,0.07,100) , n_FR = 50, if_plot = False, end_of_nonlinearity = 25, left_pad = 0.005, right_pad = 0.005, maxfev = 5000):
    '''find number of connections and build J matrix, set ext inputs as well'''
    #K = calculate_number_of_connections(N,N_real,K_real)
    K = calculate_number_of_connections(N,N_real,K_real_STN_Proto_diverse)
    receiving_class_dict= create_receiving_class_dict(receiving_pop_list, nuclei_dict)

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:

            nucleus.set_connections(K, N)

            if nucleus.neuronal_model == 'rate':
                nucleus.scale_synaptic_weight() # filter based on the receiving nucleus

            elif nucleus. der_ext_I_from_curve :
                # all_FR_list_2D = np.repeat(all_FR_list.reshape(-1,1), nucleus.n, axis = 1) # provide the array for all neurons
                # FR_sim = nucleus.Find_threshold_of_firing( all_FR_list_2D, t_list, dt, receiving_class_dict )
                # FR_list = find_FR_ext_range_for_each_neuron(FR_sim, all_FR_list, nucleus.init_method, n_FR = n_FR, left_pad = left_pad, right_pad = right_pad)
                nucleus.estimate_needed_external_input(all_FR_list, dt, t_list, receiving_class_dict, if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity, maxfev = maxfev,
                                                        n_FR = n_FR , left_pad = left_pad, right_pad = right_pad)
            
            nucleus.set_ext_input(A, A_mvt, D_mvt,t_mvt, t_list, dt)

    return receiving_class_dict

def plot_fitted_curve(xdata, ydata, x_scaled, coefs = []):
    plt.figure()
    plt.plot(xdata, ydata,'-o', label = 'data')
    if coefs != []:
        y = sigmoid(x_scaled ,*coefs)
        plt.plot(x_scaled + find_x_mid_point_sigmoid( ydata, xdata), y * find_y_normalizing_factor(ydata), label = 'fitted curve')
    plt.legend()

def scale_bound_with_mean(mean, lower_bound_perc , upper_bound_perc, scale = None):
    lower_bound = mean * lower_bound_perc 
    upper_bound = mean * upper_bound_perc
    return lower_bound, upper_bound

def scale_bound_with_arbitrary_value(mean, lower_bound_perc , upper_bound_perc, scale = 1):
    lower_bound = mean - scale* lower_bound_perc 
    upper_bound = mean + scale* upper_bound_perc
    return lower_bound, upper_bound

def truncated_normal_distributed ( mean, sigma, n, scale_bound = scale_bound_with_mean, scale = None, lower_bound_perc = 0.8, upper_bound_perc = 1.2):
    
    lower_bound, upper_bound = scale_bound(mean, lower_bound_perc, upper_bound_perc, scale = scale)
    return stats.truncnorm.rvs((lower_bound-mean)/sigma,(upper_bound-mean)/sigma, loc = mean, scale = sigma, size = int(n))

def find_FR_sim_vs_FR_ext(FR_list,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt):
    ''' find the simulated firing of population given different externalfiring rates'''

    nucleus_name = list(nuclei_dict.keys()); m = len(FR_list)
    firing_prop = {k: {'firing_mean': np.zeros((m,len(nuclei_dict[nucleus_name[0]]))),'firing_var':np.zeros((m,len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
    i = 0
    for FR in FR_list:

        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.clear_history()
                nucleus.FR_ext = FR
                nucleus.rest_ext_input = FR * nucleus.syn_weight_ext_pop * nucleus.n_ext_population * nucleus.membrane_time_constant
        nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.smooth_pop_activity(dt, window_ms = 5)
                FR_mean, FR_std = nucleus. average_pop_activity( t_list, last_fraction = 1/2)
                firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = FR_mean
                firing_prop[nucleus.name]['firing_var'][i,nucleus.population_num-1] = FR_std
                print(nucleus.name, np.round(np.average(nucleus.FR_ext), 3),
                    'FR=',FR_mean ,'std=',round(FR_std,2))
        i+=1
    return firing_prop

def instantaneus_rise_expon_decay( inputs, I = 0, I_rise = None, tau_decay = 5, tau_rise = None):

    return  I + (-I + inputs) / tau_decay, np.zeros_like(I) # dt incorporated in tau

def _dirac_delta_input (inputs, I_rise = None, I = None, tau_rise = None, tau_decay = None):

    return inputs, np.zeros_like(inputs)

def exp_rise_and_decay(inputs, I_rise = 0, I = 0 , tau_rise = 5 , tau_decay = 5):

    I_rise = I_rise + (-I_rise + inputs) / tau_rise # dt incorporated in tau
    I = I + ( -I  + I_rise ) / tau_decay # dt incorporated in tau
    return I, I_rise 

def fwd_Euler(dt, y, f):

    return y + dt * f

def f_LIF(tau, V, V_rest, I_ext, I_syn):
    ''' return dV/dt value for Leaky-integrate and fire neurons'''

    return ( -(V - V_rest) + I_ext + I_syn ) / tau

def Runge_Kutta_second_order_LIF(dt, V_t, f_t, tau, I_syn_next_dt, V_rest, I_ext):
    ''' Solve second order Runge-Kutta for a LIF at time t+dt (Mascagni & Sherman, 1997)'''

    V_next_dt = V_t + dt/2 * ( -(V_t + dt * f_t - V_rest) + I_syn_next_dt + I_ext ) / tau + f_t * dt / 2 
    return V_next_dt

def linear_interpolation( V_estimated_next_dt, V_thresh, dt, V_t, V_rest, tau):

    return  ( 
            (V_estimated_next_dt - V_thresh) * 
            ( 1 + dt / tau * (V_t - V_rest) / (V_estimated_next_dt - V_t) ) + V_rest
            )

def FR_ext_theory(V_thresh, V_rest, tau, g_ext, FR_list, N_ext):
    ''' calculate what external input is needed to get the desired firing of the list FR_list '''
    frac = (V_thresh - V_rest ) / ( FR_list * g_ext * N_ext * tau)
    return -1 / np.log ( 1 - frac ) / tau

def find_x_mid_point_sigmoid( y, x):
    ''' extrapolate the x of half max of y from the two nearest data points'''
    y_relative = y - np.max(y)/2
    signs = np.sign(y_relative)
    y_before = np.max(np.where ( signs < 0)[0])
    y_after = np.min( np.where (signs > 0)[0])
    return  ( x[y_before] + x[y_after] ) / 2

def expon(x, a, b, c):
    return a * np.exp(-b * x) + c

def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k*(x-x0)))
    
def inverse_sigmoid( y, x0, k):
    return -1/k * np.log ( (1 - y) / y) + x0

def fit_FR_as_a_func_of_FR_ext ( FR_ext, FR, estimating_func, maxfev=5000):
    popt, pcov = curve_fit(estimating_func, FR_ext.reshape(-1,), FR, method='dogbox', maxfev = maxfev)
    return popt

def extrapolated_FR_ext_from_fitted_curve (FR_ext, FR, desired_FR, coefs, estimating_func, inverse_estimating_func , FR_normalizing_factor , x_shift):
    
    return inverse_estimating_func( desired_FR / FR_normalizing_factor, *coefs) + x_shift

def find_y_normalizing_factor (y):
    return np.max (y)

def get_non_linear_part( x, y, end_of_nonlinearity = 25):
    ind =  np.where(y < end_of_nonlinearity)[0]
    return x[ind], y[ind]

def rescale_x_and_y ( x, y ):
    ydata = y / find_y_normalizing_factor(y)
    x_shift = find_x_mid_point_sigmoid( y, x)
    xdata = x - x_shift
    return xdata, ydata
    
def create_color_map(n_samples, colormap = plt.cm.viridis):
    colormap = colormap# LinearSegmentedColormap
    Ncolors = min(colormap.N,n_samples)
    mapcolors = [colormap(int(x*colormap.N/Ncolors)) for x in range(Ncolors)]
    return mapcolors

def create_sparse_matrix (matrix , end = None, start = 0):
    n_rows = matrix.shape [0]
    n_cols = matrix.shape [1]
    if end == None:
        end = n_cols
    return  np.array( [ np.where( matrix[i,int(start):int(end)] == 1 ) [0]  for i in range( n_rows )] ,dtype = object)

def raster_plot(ax, spikes_sparse, name, color_dict, labelsize=10, title_fontsize = 15):

    ax.eventplot(spikes_sparse, colors='k', linelengths=2, lw = 2, orientation='horizontal')
    ax.tick_params(axis = 'both', labelsize = labelsize)
    ax.set_title( name, c = color_dict[name], fontsize = title_fontsize)
    
def find_FR_sim_vs_FR_expected(FR_list,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict,A, A_mvt, D_mvt,t_mvt):
    ''' simulated FR vs. what we input as the desired firing rate'''

    nucleus_name = list(nuclei_dict.keys()); m = len(FR_list)
    firing_prop = {k: {'firing_mean': np.zeros((m,len(nuclei_dict[nucleus_name[0]]))),'firing_var':np.zeros((m,len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
    i = 0
    for FR in FR_list:

        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.clear_history()
                # nucleus.reset_ext_pop_properties(poisson_prop,dt)
                nucleus.basal_firing = FR
                nucleus.set_ext_input( A, A_mvt, D_mvt,t_mvt, t_list, dt)
        nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.pop_act = moving_average_array(nucleus.pop_act,50)
                firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                firing_prop[nucleus.name]['firing_var'][i,nucleus.population_num-1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                print(nucleus.name, np.round(np.average(nucleus.FR_ext), 3),
                    'FR=',firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] ,'std=',round(firing_prop[nucleus.name]['firing_var'][i,nucleus.population_num-1],2))
        i+=1
    return firing_prop


def find_ext_input_reproduce_nat_firing(tuning_param,list_1,list_2,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict):
    ''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''

    nucleus_name = list(nuclei_dict.keys()); m = len(list_1); n = len(list_2)
    firing_prop = {k: {'firing_mean': np.zeros((m,n,len(nuclei_dict[nucleus_name[0]]))),'firing_var':np.zeros((m,n,len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
    ext_firing = np.zeros((m,n,2))
    loss =  np.zeros((m,n))
    i = 0
    for g_1 in list_1:
        j = 0
        for g_2 in list_2:


            ext_firing[i,j] = [g_1,g_2]
            poisson_prop[nucleus_name[0]][tuning_param] = g_1
            poisson_prop[nucleus_name[1]][tuning_param] = g_2
            for nuclei_list in nuclei_dict.values():
                for nucleus in nuclei_list:
                    nucleus.clear_history()
                    nucleus.reset_ext_pop_properties(poisson_prop,dt)
            nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
            for nuclei_list in nuclei_dict.values():
                for nucleus in nuclei_list:

                    firing_prop[nucleus.name]['firing_mean'][i,j,nucleus.population_num-1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                    loss[i,j] += (firing_prop[nucleus.name]['firing_mean'][i,j,nucleus.population_num-1]- nucleus.basal_firing)**2
                    firing_prop[nucleus.name]['firing_var'][i,j,nucleus.population_num-1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                    print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop,3),
                        'FR=',firing_prop[nucleus.name]['firing_mean'][i,j,nucleus.population_num-1] ,'std=',round(firing_prop[nucleus.name]['firing_var'][i,j,nucleus.population_num-1],2))
            print('loss =',loss[i,j])
            j+=1
        i+=1
    return loss,ext_firing, firing_prop

def find_ext_input_reproduce_nat_firing_3_pop(tuning_param,list_1,list_2,list_3,poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict):
    ''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''

    nucleus_name = list(nuclei_dict.keys()); m = len(list_1); n = len(list_2); k = len(list_3)
    firing_prop = {kk: {'firing_mean': np.zeros((m,n,k,len(nuclei_dict[nucleus_name[0]]))),'firing_var':np.zeros((m,n,k,len(nuclei_dict[nucleus_name[0]])))} for kk in nucleus_name}
    ext_firing = np.zeros((m,n,k,3))
    loss =  np.zeros((m,n,k))
    i = 0
    for g_1 in list_1:
        j = 0
        for g_2 in list_2:
            l =0
            for g_3 in list_3:

                ext_firing[i,j,l] = [g_1,g_2,g_3]
                poisson_prop[nucleus_name[0]][tuning_param] = g_1
                poisson_prop[nucleus_name[1]][tuning_param] = g_2
                poisson_prop[nucleus_name[2]][tuning_param] = g_3
                for nuclei_list in nuclei_dict.values():
                    for nucleus in nuclei_list:
                        nucleus.clear_history()
                        nucleus.reset_ext_pop_properties(poisson_prop,dt)
                nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
                for nuclei_list in nuclei_dict.values():
                    for nucleus in nuclei_list:
                        firing_prop[nucleus.name]['firing_mean'][i,j,l,nucleus.population_num-1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                        loss[i,j,l] += (firing_prop[nucleus.name]['firing_mean'][i,j,l,nucleus.population_num-1]- nucleus.basal_firing)**2
                        firing_prop[nucleus.name]['firing_var'][i,j,l,nucleus.population_num-1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                        print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop,4),
                            'FR=',firing_prop[nucleus.name]['firing_mean'][i,j,l,nucleus.population_num-1] ,'std=',round(firing_prop[nucleus.name]['firing_var'][i,j,l,nucleus.population_num-1],2))
                print('loss =',loss[i,j,l])
                l+=1
            j+=1
        i+=1
    return loss,ext_firing, firing_prop

def find_ext_input_reproduce_nat_firing_relative(tuning_param,list_1, poisson_prop,receiving_class_dict,t_list, dt,nuclei_dict):
    ''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''

    nucleus_name = list(nuclei_dict.keys()); m = len(list_1)
    firing_prop = {k: {'firing_mean': np.zeros((m,len(nuclei_dict[nucleus_name[0]]))),'firing_var':np.zeros((m,len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
    ext_firing = np.zeros((m,len(nucleus_name)))
    loss = np.zeros(m)
    i =0
    for g in list_1:
        for j in range(len(nucleus_name)):
            poisson_prop[nucleus_name[j]][tuning_param] = g * nuclei_dict[nucleus_name[j]][0].FR_ext
            ext_firing[i,j] = poisson_prop[nucleus_name[j]][tuning_param]
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.clear_history()
                nucleus.reset_ext_pop_properties(poisson_prop,dt)
        nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                loss[i] += (firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1]- nucleus.basal_firing)**2
                firing_prop[nucleus.name]['firing_var'][i,nucleus.population_num-1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop,3),
                    'FR=',firing_prop[nucleus.name]['firing_mean'][i,nucleus.population_num-1] ,'std=',round(firing_prop[nucleus.name]['firing_var'][i,nucleus.population_num-1],2))
        print('loss = ',loss[i])
        i+=1
    return loss, ext_firing, firing_prop

def dopamine_effect(threshold, G, dopamine_percentage):
    ''' Change the threshold and synaptic weight depending on dopamine levels'''
    threshold['Str'] = -0.02 + 0.03*(1-(1.1/(1+0.1*np.exp(-0.03*(dopamine_percentage - 100)))))
    G[('Str','Ctx')] = 0.75/(1+np.exp(-0.09*(dopamine_percentage - 60)))
    return threshold, G

def possion_spike_generator(n_pop,n_sending,r,dt):
    '''generate a times series of possion spikes for a population of size n, with firing rate r'''
    x = np.random.rand(n_pop,n_sending)
    # poisson_thresh = r*dt
    # temp = (1-np.exp(-r*dt))
    temp = r*dt
    poisson_thresh = np.repeat(temp.reshape(-1,1),n_sending,axis =1)
    spikes = np.where(x <= poisson_thresh)
    # print(spikes)
    x = x*0
    x[spikes] = 1 # spike with probability of rdt
    # x[~spikes] = 0
    return x.astype(int)

def spacing_with_high_resolution_in_the_middle(n_points, start, end):
    '''return a series with lower spacing and higher resolution in the middle'''
    
    R = (start - end) / 2 
    x = R * np.linspace(-1, 1, n_points)
    y = np.sqrt(R ** 2 - x ** 2) 
    half_y = y [ : len(y) // 2 ]
    diff = - np.diff ( np.flip ( half_y ) )
    series = np.concatenate( (half_y, np.cumsum(diff) +  y [ len(y) // 2]) ) + start
    # print(series[ len(series) // 2])
    return series.reshape(-1,1)

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
        # sig = np.sin(50*np.arange(len(sig))*dt*2*np.pi)
        f = rfft(sig)
        freq = fftfreq(N, dt)#[:N//2]
        # plt.figure()
        # plt.semilogy(freq[:len(f)], f)
        # plt.semilogy(freq[:N//2], f[:N//2])

        # ind  = np.where(freq >= 0)
        # plt.plot(freq[:N//2], f[:N//2])
        # plt.xlabel('frequency (Hz)')
        # plt.ylabel('FFT power')      
        # Find the peak and interpolate to get a more accurate peak
        peak_freq = freq[np.argmax(abs(f[:N//2]))]  # Just use this for less-accurate, naive version
        # print(peak_freq)
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
    
    for t in t_list:

        for nuclei_list in nuclei_dict.values():
            k = 0
            for nucleus in nuclei_list:
                k += 1
        #        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement 
                mvt_ext_inp = np.ones((nucleus.n,1))*nucleus.external_inp_t_series[t] # movement added 
                if nucleus.neuronal_model == 'rate': ######## rate model
                    nucleus.calculate_input_and_inst_act(t, dt, receiving_class_dict[(nucleus.name,str(k))], mvt_ext_inp)
                    nucleus.update_output(dt)
                if nucleus.neuronal_model == 'spiking': ######## QIF
                    nucleus.solve_IF(t,dt,receiving_class_dict[(nucleus.name,str(k))],mvt_ext_inp)

    stop = timeit.default_timer()
    print("t = ", stop - start)
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
    JJ = np.zeros((n_receiving, n_projecting),dtype = int)
    rows = ((np.ones((n_connections,n_receiving))*np.arange(n_receiving)).T).flatten().astype(int)
    cols = projection_list.flatten().astype(int)
    JJ[rows,cols] = int(1)
    return JJ

dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

def plot( nuclei_dict,color_dict,  dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_ob, title = "", n_subplots = 1,title_fontsize = 18,plot_start = 0,ylabelpad = 0, plot_end = None, figsize = (6,5)):    

    if plot_ob == None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
    else:
        fig, ax = plot_ob
    if plot_end == None : plot_end = t_list [-1]
    else:
        plot_end = int(plot_end / dt)
    plot_start = int( plot_start / dt)
    line_type = ['-', '--']
    count = 0
    for nuclei_list in nuclei_dict.values():
        for nucleus in [nuclei_list[0]]:
            
            ax.plot(t_list[plot_start: plot_end]*dt, nucleus.pop_act[plot_start: plot_end], line_type[nucleus.population_num-1], label = nucleus.name, c = color_dict[nucleus.name],lw = 1.5)
            ax.plot(t_list[plot_start: plot_end]*dt, np.ones_like(t_list[plot_start: plot_end])*A[nucleus.name], '-.', c = color_dict[nucleus.name],lw = 1, alpha=0.8 )
            ax.plot(t_list[plot_start: plot_end]*dt, np.ones_like(t_list[plot_start: plot_end])*A_mvt[nucleus.name], '-.', c = color_dict[nucleus.name], alpha=0.2,lw = 1 )
            FR_mean, FR_std = nucleus. average_pop_activity( t_list, last_fraction = 1/2)
            txt =  r"$\overline{{FR_{{{0}}}}}$ ={1} $\pm$ {2}".format(nucleus.name,  round(FR_mean,2), round(FR_std,2) )

            fig.text(0.3, 0.8 - count * 0.05, txt, ha='left', va='center', rotation='horizontal',fontsize = 15, color = color_dict[nucleus.name])
            count = count + 1

    ax.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color='lightskyblue')
    plt.title(title, fontsize = title_fontsize)
    plt.xlabel("time (ms)", fontsize = 15)
    plt.ylabel("firing rate (spk/s)", fontsize = 15,labelpad=ylabelpad)
    plt.legend(fontsize = 15, loc = 'upper right')
    # ax.tick_params(axis='both', which='major', labelsize=10)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.xlim(plot_start * dt - 20, plot_end * dt + 20) 
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
    
def cut_plateau(sig, epsilon_std = 10**(-2), epsilon = 10**(-2), window = 40):
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
    shifted = np.roll(zero_crossings,-1)
    half_lambda =  shifted[:-1] - zero_crossings[:-1]
    n_half_cycles = len(half_lambda)
    if n_half_cycles > 1:
        frequency = 1 / (np.average (half_lambda ) * 2 * dt)
    else: 
        frequency = 0
    return n_half_cycles, frequency

def find_mean_of_signal(sig, cut_sig_ind, non_plateau_perc = 4/5):
    ''' find the value which when subtracted the signal oscillates around zero'''
    len_varying = len(cut_sig_ind)
    # plt.figure()
    # plt.plot(sig)
    # plt.plot(sig[:max(cut_plateau_ind)])
    # plt.plot(sig[cut_plateau_ind])
    # print(len_not_plateau,len(sig))
    if len_varying > 0 and len_varying < len(sig) * non_plateau_perc: # if more than 1/5th of signal is plateau
        plateau_y = np.average( sig[ max( cut_plateau_ind ) : ] )  
    else: # if less than 1/5th is left better to average the whole signal for more confidence
        # peaks,properties = signal.find_peaks(sig)
        # troughs,properties = signal.find_peaks(-sig)
        plateau_y = np.average(sig)
    return plateau_y

def trim_start_end_sig_rm_offset(sig,start, end, cut_plateau_epsilon = 0.1, method = 'neat'):
    ''' trim with max point at the start and the given end point'''
    if method not in ["simple", "neat"]:
        raise ValueError("mode must be either 'simple', or 'neat'")

    trimmed = sig[start:end]
    if method == 'simple' : 
        return trimmed
    elif method == 'complex':
        cut_sig_ind = cut_plateau(sig, epsilon= cut_plateau_epsilon)
        plateau_y = find_mean_of_signal(trimmed, cut_sig_ind)
        trimmed = trimmed - plateau_y # remove the offset determined with plateau level
        max_value = np.max(trimmed)
        min_value = np.min(trimmed)
        if abs(max_value) > abs(min_value): # find exterma, whether it's a minimum or maximum
            max_point = np.max(np.where(trimmed == max_value))
        else:
            max_point = np.max(np.where(trimmed == min_value))

        if max_point>len(trimmed)/2: # in case the amplitude increases over time take the whole signal #############################3 to do --> check for increasing amplitude in oscillations instead of naive implementation here
            return trimmed- np.average(trimmed)
        else:
            return(trimmed[max_point:] - np.average(trimmed[max_point:]))
        
def find_freq_of_pop_act_spec_window(nucleus, start, end, dt, peak_threshold = 0.1, smooth_kern_window= 3 , check_stability = False, cut_plateau_epsilon = 0.1):
    ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
    the plateau and in case it is oscillation determine the frequency '''
    sig = trim_start_end_sig_rm_offset(nucleus.pop_act,start, end)
    cut_sig_ind = cut_plateau( sig, epsilon= cut_plateau_epsilon)
    plateau_y = find_mean_of_signal(sig, cut_sig_ind)
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
            nuclei_dict = reinitialize_nuclei(nuclei_dict, G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
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


def temp_oscil_check(sig_in,peak_threshold, smooth_kern_window,dt,start,end, cut_plateau_epsilon = 0.1):
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
    cut_sig_ind = cut_plateau ( sig, epsilon= cut_plateau_epsilon)
    plateau_y = find_mean_of_signal(sig, cut_sig_ind)
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



def find_AUC_of_input(name,poisson_prop,gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude,
                      N, A, A_mvt,D_mvt,t_mvt, N_real, K_real,t_list,color_dict, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,if_plot = True):
    receiving_pop_list = {(name,'1') : []}
    pop_list = [1]  
    
    class Nuc_AUC(Nucleus):
        def cal_ext_inp(self,dt,t):
            ## to have exactly one spike the whole time
            poisson_spikes = np.zeros((self.n,self.n_ext_population))
            if t == 10:
                ind = np.random.randint(0,self.n_ext_population-1,size = self.n)
                poisson_spikes[(np.arange(self.n),ind)] = 1
            self.syn_inputs['ext_pop','1'] =  (np.sum(poisson_spikes,axis = 1) / dt * self.syn_weight_ext_pop * self.membrane_time_constant).reshape(-1,)
            # self.I_syn['ext_pop','1'] += np.true_divide((-self.I_syn['ext_pop','1'] + self.syn_inputs['ext_pop','1']),self.tau_ext_pop['decay']) # without rise
            self.I_rise['ext_pop','1'] += ((-self.I_rise['ext_pop','1'] + self.syn_inputs['ext_pop','1'])/self.tau_ext_pop['rise'])
            self.I_syn['ext_pop','1'] += np.true_divide((-self.I_syn['ext_pop','1'] + self.I_rise['ext_pop','1']),self.tau_ext_pop['decay'])
    nucleus = Nuc_AUC(1, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T,
                        t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking',poisson_prop =poisson_prop) 
    nuclei_dict = {name: [nucleus]}
    receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list)
    
    nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
    if if_plot:
        fig2, ax2 = plt.subplots(1, 1, sharex=True, sharey=True)
        
        for nucleus_list in nuclei_dict.values():
            for nucleus in nucleus_list:
                y = np.average(nucleus.ext_input_all,axis=0)
                std = np.std(nucleus.ext_input_all,axis=0)
                ax2.plot(t_list*dt,np.average(nucleus.ext_input_all,axis=0),c = color_dict[nucleus.name],label = nucleus.name)
                ax2.fill_between(t_list*dt, y-std, y+std, alpha = 0.5)
        ax2.set_title('external input',fontsize = 15)
        ax2.legend()
    AUC = np.average([np.trapz(nucleus.ext_input_all[i,:],x=t_list*dt) for i in range(N[name])],axis = 0)
    AUC_std = np.std([np.trapz(nucleus.ext_input_all[i,:],x=t_list*dt) for i in range(N[name])],axis = 0)
    print("AUC of one spike =",round(AUC,3),'+/-',round(AUC_std,3),"mV")
    return AUC,AUC_std



def plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, I_ext_range, neuronal_consts, start_epsilon = 10**(-10)):
    
    start_theory = (((neuronal_consts[name]['spike_thresh']['mean']-neuronal_consts[name]['u_rest'])
              / (poisson_prop[name]['g']*poisson_prop[name]['n']*neuronal_consts[name]['membrane_time_constant']['mean'])) + start_epsilon)
    x1 = np.linspace( start_theory, start_theory + 0.0001, 1000).reshape(-1,1)
    end = I_ext_range[name][1] / poisson_prop[name]['g'] / poisson_prop [name ]['n']
    x_theory = np.concatenate( ( x1, np.geomspace(x1 [ -1], end, 10)))
    y_theory = FR_ext_theory(neuronal_consts[name]['spike_thresh']['mean'], 
                              neuronal_consts[name]['u_rest'], 
                              neuronal_consts[name]['membrane_time_constant']['mean'], poisson_prop[name]['g'], x_theory, poisson_prop[name]['n'])
    plt.plot(x_theory * 1000, y_theory * 1000,label='theory', c= 'lightcoral' , markersize = 6, markeredgecolor = 'grey')
    plt.xlabel(r'$FR_{ext}$',fontsize=15)
    plt.ylabel(r'$FR$',fontsize=15)
    plt.xlim(I_ext_range[name][0] / poisson_prop[name]['g'] / poisson_prop [name ]['n'] * 1000, I_ext_range[name][1] / poisson_prop[name]['g'] / poisson_prop [name ]['n'] * 1000)
    plt.legend()

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


    # def solve_IF(self,t,dt,receiving_from_class_list,mvt_ext_inp):

    #     self.cal_ext_inp(dt,t)
    #     # inputs = np.zeros(self.n)
    #     synaptic_inputs = cal_synaptic_input(self,receiving_from_class_list,dt)

    #     self.representative_inp['ext_pop','1'][t] = self.I_syn['ext_pop','1'][0]
    #     self.input_all[:,t] = self.I_syn['ext_pop','1']/self.syn_weight_ext_pop

    #     # if not self.caught: ##### to check for voltage divergence
    #     #     ind = np.where(self.mem_potential < -10000)[0]
    #     #     if len(ind) >0:
    #     #         print('Divergence')
    #     #         self.ind = ind[0]
    #     #         self.caught = True
    #     # if self.caught:
    #     #     self.dumby_I_ext[t] = self.I_syn['ext_pop','1'] [self.ind]
    #     #     self.dumby_I_syn[t] = inputs[self.ind]
    #     #     self.dumby_V[t] = self.mem_potential[self.ind]
    #     # print(self.name,np.average( self.I_syn['ext_pop','1']))

    #     self.update_potential(synaptic_inputs)
    #     spiking_ind = self.find_spikes()
    #     self.reset_potential(spiking_ind)
    #     self.cal_population_activity()
    #     self.voltage_trace[t] = self.mem_potential[0]