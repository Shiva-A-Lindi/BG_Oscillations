from __future__ import division
import os
import sys
import subprocess
import timeit
import numpy as np
from numpy import inf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, FixedLocator, FixedFormatter
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.fft import rfft, fft, fftfreq, ifft, fftshift
from tempfile import TemporaryFile
import pickle
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import truncexpon, skewnorm
from scipy.signal import butter, sosfilt, sosfreqz, spectrogram, sosfiltfilt
from scipy import signal, stats
try:
    import jsonpickle
except ImportError or ModuleNotFoundError:
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'jsonpickle'])

# import jsonpickle
# import decimal
# from decimal import *
# from scipy import optimize


def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = mticker.FuncFormatter(g)
dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])


def extrapolate_FR_ext_from_neuronal_response_curve_high_act(FR_ext, FR_sim, desired_FR, if_plot=False, end_of_nonlinearity=None, maxfev=None, g_ext=0, N_ext=0, tau=0, ax=None, noise_var=0, c='grey'):
    ''' All firing rates in Hz'''
    slope, intercept = linear_regresstion(FR_ext, FR_sim)
    FR_ext_extrapolated = inverse_linear(desired_FR, slope, intercept)
    print(FR_ext_extrapolated * slope + intercept)
    if if_plot:

        plot_fitted_line(FR_ext, FR_sim, slope, intercept,  FR_to_I_coef=tau *
                         g_ext * N_ext / 1000, ax=ax, noise_var=noise_var, c=c)

    return FR_ext_extrapolated / 1000   # FR_ext is in Hz, we want spk/ms


def extrapolate_FR_ext_from_neuronal_response_curve(FR_ext, FR_sim, desired_FR, if_plot=False, end_of_nonlinearity=25, maxfev=5000, g_ext=0, N_ext=0, tau=0, ax=None, noise_var=0, c='grey'):
    ''' All firing rates in Hz'''

    xdata, ydata = get_non_linear_part(
        FR_ext, FR_sim, end_of_nonlinearity=end_of_nonlinearity)
    x, y = rescale_x_and_y(xdata, ydata, desired_FR)
    coefs = fit_FR_as_a_func_of_FR_ext(x, y, sigmoid, maxfev=maxfev)
    FR_ext = extrapolated_FR_ext_from_fitted_curve(x, y, desired_FR, coefs, sigmoid, inverse_sigmoid,
                                                find_y_normalizing_factor(ydata, desired_FR),
                                                find_x_mid_point_sigmoid(ydata, xdata))
    if if_plot:
        plot_fitted_sigmoid(xdata, ydata, x, desired_FR, coefs=coefs,
                            FR_to_I_coef=tau * g_ext * N_ext / 1000, 
                            ax=ax, noise_var=noise_var, c=c)
    if FR_ext == np.nan or FR_ext == np.inf:
        print(desired_FR, find_y_normalizing_factor(ydata, desired_FR))
        print('Corr FR_ext =', FR_ext)
        plot_fitted_sigmoid(xdata, ydata, x, desired_FR, coefs=coefs,
                            FR_to_I_coef=tau * g_ext * N_ext / 1000, ax=ax, noise_var=noise_var, c=c)
    return FR_ext / 1000  # FR_ext is in Hz, we want spk/ms


class Nucleus:

    def __init__(self, population_number, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
        synaptic_time_constant, receiving_from_list, smooth_kern_window, oscil_peak_threshold, syn_input_integ_method='exp_rise_and_decay', neuronal_model='rate',
        poisson_prop=None, AUC_of_input=None, init_method='homogeneous', ext_inp_method='const+noise', der_ext_I_from_curve=False, bound_to_mean_ratio=[0.8, 1.2],
        spike_thresh_bound_ratio=[1/20, 1/20], ext_input_integ_method='dirac_delta_input', path=None, mem_pot_init_method='uniform', plot_initial_V_m_dist=False, 
        set_input_from_response_curve=True, set_random_seed=False, keep_mem_pot_all_t=False, save_init=False, scale_g_with_N=True, syn_component_weight = None, 
        time_correlated_noise = True, noise_method = 'Gaussian', noise_tau = 10, keep_noise_all_t = False, state = 'rest', random_seed = 1996):

        if set_random_seed:
            self.random_seed = random_seed
            np.random.seed(self.random_seed)


        n_timebins = int(t_sim/dt)
        self.name = name
        self.population_num = population_number
        self.receiving_from_list = receiving_from_list[(self.name, str(self.population_num))]
        self.receiving_from_pop_name_list = [ pop[0] for pop in self.receiving_from_list]
        self.n = N[name]  # population size
        self.basal_firing = A[name]
        self.mvt_firing = A_mvt[name]
        self.threshold = threshold[name]
        self.gain = gain[name]
        # filter based on the receiving nucleus# dictfilt(synaptic_time_constant, self.trans_types) # synaptic time scale based on neuron type
        self.synaptic_time_constant = {
            k: v for k, v in synaptic_time_constant.items() if k[1] == name}
        # filter based on the receiving nucleus
        self.T_specs = {k: v for k, v in T.items() 
                          if k[0] == self.name and 
                             k[1] in self.receiving_from_pop_name_list}
        
        self.transmission_delay = {k: np.zeros( self.n ) 
                                   for k, v in self.T_specs.items()}
        
        # self.transmission_delay = {k: v for k, v in T.items() if k[0] == name}

        self.ext_inp_delay = ext_inp_delay
        
        self.synaptic_weight = {k: v for k, v in G.items() if k[0] == name} # filter based on the receiving nucleus
        self.K_connections = None
        # stored history in ms derived from the longest transmission delay of the projections

        sending_to_dict = find_sending_pop_dict(receiving_from_list)
        self.sending_to_dict = sending_to_dict[(self.name, str(self.population_num))]
        self.rest_ext_input = None
        self.avg_pop_act_mvt = None
        self.avg_pop_act_base = None
        self.connectivity_matrix = {}
        self.smooth_kern_window = smooth_kern_window[self.name]
        self.set_noise_param(noise_variance, noise_amplitude)
        self.init_method = init_method
        self.neuronal_model = neuronal_model
        self.frequency_basal = None
        self.perc_oscil_basal = None
        self.trim_sig_method_dict = {'spiking': 'simple', 'rate': 'neat'}
        self.path = path
        self.pop_act = np.zeros((n_timebins))  # time series of population activity
        self.external_inp_t_series = np.zeros((self.n, n_timebins))
        self.t_sim = t_sim
        self.sqrt_dt = np.sqrt(dt)
        self.half_dt = dt/2
        
        
        if neuronal_model == 'rate':
            
            self.output = {k: np.zeros((self.n, int( T[ k[0], self.name] / dt ))) 
                           for k in self.sending_to_dict}
            self.input = np.zeros((self.n))
            self.neuron_act = np.zeros((self.n))
            # external input mimicing movement
            self.mvt_ext_input = np.zeros((n_timebins))
            # the basal firing as a result of noise at steady state
            self.noise_induced_basal_firing = None
            self.oscil_peak_threshold = oscil_peak_threshold[self.name]
            self.scale_g_with_N = scale_g_with_N
            self. intialize_transmission_delays(dt)

        if neuronal_model == 'spiking':
            
            print('Initializing ', name)
            self.state = state
            self.spikes = np.zeros((self.n, int(t_sim/dt)), dtype=int)
            self.tau_specs = {k: v for k, v in tau.items() 
                              if k[0] == self.name and 
                              k[1] in self.receiving_from_pop_name_list}
            

            self.tau = {k: np.zeros((  len ( list( v['rise'].values()) [0]) , self.n )) 
                        for k, v in self.tau_specs.items()}
            
            # self.tau = {k: {kk: np.array(vv)/dt for kk, vv in tau[k].items()} 
            #             for k, v in tau.items() if k[0] == name}
            
            self.pre_n_components = {k[1]: len(v['rise']['mean']) for k,v in self.tau_specs.items()}
            
            if syn_component_weight != None:
                self.syn_component_weight = {k: v for k, v in syn_component_weight.items() if k[0] == name}
                
                
            # since every connection might have different rise and decay time, inputs must be updataed accordincg to where the input is coming from
            self.I_rise = {k: np.zeros((self.n, self.pre_n_components[k[0]])) for k in self.receiving_from_list}
            self.I_syn = {k: np.zeros((self.n, self.pre_n_components[k[0]])) for k in self.receiving_from_list}
            self.I_syn['ext_pop', '1'] = np.zeros(self.n,)
            self.I_rise['ext_pop', '1'] = np.zeros(self.n,)
            self.neuronal_consts = neuronal_consts[self.name]
            self.mem_pot_before_spike = np.zeros(self.n)
            self.syn_inputs = {k: np.zeros((self.n, 1))
                               for k in self.receiving_from_list}
            
            self.poisson_spikes = None                                       # meant to store the poisson spike trains of the external population
            self.n_ext_population = poisson_prop[self.name]['n']            # external population size
            self.firing_of_ext_pop = poisson_prop[self.name]['firing']
            self.syn_weight_ext_pop = poisson_prop[self.name]['g']
            self.representative_inp = {k: np.zeros( (n_timebins, self.pre_n_components[k[0]]) ) 
                                       for k in self.receiving_from_list}
            self.representative_inp['ext_pop', '1'] = np.zeros(n_timebins)
            self.ext_input_all_t = np.zeros((self.n, n_timebins))
            self.voltage_trace = np.zeros(n_timebins+1)
            self.AUC_of_input = AUC_of_input
            self.rest_ext_input = np.zeros(self.n)
            self.ext_inp_method = ext_inp_method
            self.der_ext_I_from_curve = der_ext_I_from_curve      # if derive external input value from response curve
            self.sum_syn_inp_at_rest = None
            self.keep_mem_pot_all_t = keep_mem_pot_all_t
            self.save_init = save_init
            self.set_input_from_response_curve = set_input_from_response_curve
            self.ext_input_integ_method = ext_input_integ_method
            self.mem_pot_init_method = mem_pot_init_method
            self.init_method = init_method
            self.spike_thresh_bound_ratio = spike_thresh_bound_ratio
            self.spike_rel_phase_hist = {}
            self.bound_to_mean_ratio = bound_to_mean_ratio
            self.pop_act_filtered = False
            self.syn_input_integ_method = syn_input_integ_method
            self.I_ext_0 = None
            self.noise = np.zeros(( self.n, 1))
            self.noise_method = noise_method
            self.noise_tau = noise_tau 
            
            if self.keep_mem_pot_all_t:
                self.all_mem_pot = np.zeros((self.n, n_timebins))
                
            if keep_noise_all_t : 
                self.noise_all_t = np.zeros((self.n, n_timebins))
                
            self.set_init_distribution( tau, poisson_prop, dt, t_sim,  plot_initial_V_m_dist = plot_initial_V_m_dist)
            self.normalize_synaptic_weight()
            
            self.ext_inp_method_dict = {'Poisson': self.poissonian_ext_inp,
                                        'const+noise': self.constant_ext_input_with_noise, 
                                        'constant': self.constant_ext_input}
            
            self.input_integ_method_dict = {'exp_rise_and_decay': exp_rise_and_decay,
                                            'instantaneus_rise_expon_decay': instantaneus_rise_expon_decay, 
                                            'dirac_delta_input': _dirac_delta_input}
            
        
            self.noise_generator_dict = { 'Gaussian' : noise_generator,
                                          'Ornstein-Uhlenbeck': OU_noise_generator}
            
    def set_init_distribution(self, tau, poisson_prop, dt, t_sim, plot_initial_V_m_dist = False):
        
        if self.init_method == 'homogeneous':
            
            self.initialize_homogeneously(poisson_prop, dt)
            self.FR_ext = 0
            
        elif self.init_method == 'heterogeneous':
            
            self.initialize_heterogeneously(tau, poisson_prop, dt, t_sim, self.spike_thresh_bound_ratio, 
                                            *self.bound_to_mean_ratio,  plot_initial_V_m_dist=plot_initial_V_m_dist)
            
            self.FR_ext = np.zeros(self.n)
          
    def intialize_membrane_time_constant(self, lower_bound_perc=0.8, upper_bound_perc=1.2, 
                                         plot_mem_tau_hist = False):
        
        ''' initialize membrane time constant either with a specified truncation value or
            truncation with percentage of the mean values '''
            
        if 'truncmin' in self.neuronal_consts['membrane_time_constant']:
            
            self.membrane_time_constant = truncated_normal_distributed(self.neuronal_consts['membrane_time_constant']['mean'],
                                                                       self.neuronal_consts['membrane_time_constant']['var'], self.n,
                                                                       truncmin=self.neuronal_consts['membrane_time_constant']['truncmin'],
                                                                       truncmax=self.neuronal_consts['membrane_time_constant']['truncmax'])
        else:
            
            self.membrane_time_constant = truncated_normal_distributed(self.neuronal_consts['membrane_time_constant']['mean'],
                                                                       self.neuronal_consts['membrane_time_constant']['var'], self.n,
                                                                       lower_bound_perc=lower_bound_perc, upper_bound_perc=upper_bound_perc)
        
        if plot_mem_tau_hist:
            plot_histogram(self.membrane_time_constant, bins = 25, title = self.name)
            
    def intialize_synaptic_time_constant(self, dt, tau, lower_bound_perc=0.8, upper_bound_perc=1.2,
                                          bins=50, color='grey', tc_plot = 'decay', syn_element_no = 0, 
                                          plot_syn_tau_hist = False):
        
        ''' initialize synaptic time constant with a truncated normal distribution
            Note: dt incorporated in tau for time efficiency
        '''
        if len(self.receiving_from_pop_name_list) > 0:
            
            tc_list = list (self.tau_specs[ list(self.tau_specs.keys()) [0] ].keys() ) 
            
            for key in list (self.tau_specs.keys()):
                
                self.tau[key] = {tc : np.array( [truncated_normal_distributed(self.tau_specs[key][tc]['mean'][i],
                                                                              self.tau_specs[key][tc]['sd'][i], self.n,
                                                                              truncmin = self.tau_specs[key][tc]['truncmin'][i],
                                                                              truncmax = self.tau_specs[key][tc]['truncmax'][i]) / dt 
                                                 for i in range( len(self.tau_specs[key][tc]['mean'] )) 
                                                 ] )
                                for tc in tc_list
                                }
                                       
                if plot_syn_tau_hist:
                    
                    self.plot_synaptic_time_scale_distribution(key, dt, ax=None, bins=bins, 
                                                               color=color, tc = tc_plot, 
                                                               syn_element_no = syn_element_no)
              
    def intialize_transmission_delays(self, dt, lower_bound_perc=0.8, upper_bound_perc=1.2,
                                          bins=50, color='grey',
                                          plot_T_hist = False):
        
        ''' initialize axonal transmission delays with a truncated normal distribution
            Note: dt incorporated in tau for time efficiency
        '''

        if len(self.receiving_from_pop_name_list) > 0:
            self.transmission_delay = {key: (truncated_normal_distributed(self.T_specs[key]['mean'],
                                                                          self.T_specs[key]['sd'], self.n,
                                                                          truncmin = self.T_specs[key]['truncmin'],
                                                                          truncmax = self.T_specs[key]['truncmax']) / dt ).astype(int)
                                       for key in list (self.T_specs.keys())}
            
            self.history_duration = get_max_value_dict(self.transmission_delay )

    def intialize_ext_synaptic_time_constant(self, poisson_prop, dt, lower_bound_perc=0.8, upper_bound_perc=1.2):
        
        tc_list = ['rise', 'decay']
        # dt incorporated for time efficiency

        self.tau_ext_pop = {tc: truncated_normal_distributed(poisson_prop[self.name]['tau'][tc]['mean'],
                                                             poisson_prop[self.name]['tau'][tc]['var'], self.n,
                                                             lower_bound_perc=lower_bound_perc, 
                                                             upper_bound_perc=upper_bound_perc) / dt
                            for tc in tc_list}
                  
    def initialize_spike_threshold(self, spike_thresh_bound_ratio, lower_bound_perc=0.8, upper_bound_perc=1.2):
        
        self.spike_thresh = truncated_normal_distributed(self.neuronal_consts['spike_thresh']['mean'],
                                                         self.neuronal_consts['spike_thresh']['var'], self.n,
                                                         scale_bound=scale_bound_with_arbitrary_value, scale=(
                                                         self.neuronal_consts['spike_thresh']['mean'] - self.neuronal_consts['u_rest']['mean']),
                                                         lower_bound_perc=spike_thresh_bound_ratio[0], upper_bound_perc=spike_thresh_bound_ratio[1])
    
    def initialize_resting_membrane_potential(self, plot_initial_V_m_dist = False):
    
        self.u_rest = truncated_normal_distributed(self.neuronal_consts['u_rest']['mean'],
                                               self.neuronal_consts['u_rest']['var'], self.n,
                                               truncmin=self.neuronal_consts['u_rest']['truncmin'],
                                               truncmax=self.neuronal_consts['u_rest']['truncmax'])

    def initialize_mem_potential(self, method = 'uniform', plot_initial_V_m_dist = False):
        
        if method not in ['uniform', 'constant', 'exponential', 'draw_from_data']:
            
            raise ValueError(
                " method must be either 'uniform', 'constant', 'exponential', or 'draw_from_data' ")
            
        if method == 'draw_from_data':

            data = np.load(os.path.join(self.path, 'all_mem_pot_' + self.name + '_tau_' + str(np.round(
                            self.neuronal_consts['membrane_time_constant']['mean'], 1)).replace('.', '-') + '_' + self.state + '.npy'))
            
            y_dist = data.reshape( int( data.shape[0] * data.shape[1] ), 1)
            
            self.mem_potential = draw_random_from_data_pdf(y_dist, self.n, bins=20)

        elif method == 'uniform':

            self.mem_potential = np.random.uniform( low=self.neuronal_consts['u_initial']['min'], 
                                                   high=self.neuronal_consts['u_initial']['max'], 
                                                   size=self.n)


        elif method == 'constant':
            
              self.mem_potential = np.full(self.n, self.neuronal_consts['u_rest']['mean'])

        elif method == 'exponential':  # Doesn't work with linear interpolation of IF, diverges
        
            lower, upper, scale = 0, self.neuronal_consts['spike_thresh']['mean'] - \
                                    self.u_rest, 30
            X = stats.truncexpon(b=(upper-lower) / scale, loc=lower, scale=scale)
            self.mem_potential = self.neuronal_consts['spike_thresh']['mean'] - X.rvs(
                                self.n)
        
        if self.keep_mem_pot_all_t:
            self.all_mem_pot[:, 0] = self.mem_potential.copy()
            
        if plot_initial_V_m_dist:
            self.plot_mem_potential_distribution_of_one_t(0, bins=50)
            
        self.voltage_trace[0] = self.mem_potential[0] 
        
    def initialize_heterogeneously(self, tau, poisson_prop, dt, t_sim, spike_thresh_bound_ratio, 
                                   lower_bound_perc=0.8, upper_bound_perc=1.2,
                                    plot_initial_V_m_dist=False, plot_mem_tau_hist = False,
                                    bins=50, color='grey', tc_plot = 'decay', syn_element_no = 0, 
                                    plot_syn_tau_hist = False):
        
        ''' cell properties and boundary conditions come from distributions'''
        
        self. initialize_resting_membrane_potential()
        
        self. intialize_transmission_delays(dt, 
                                            lower_bound_perc = lower_bound_perc, 
                                            upper_bound_perc = upper_bound_perc)
        
        self. initialize_spike_threshold( spike_thresh_bound_ratio,
                                         lower_bound_perc = lower_bound_perc, 
                                         upper_bound_perc = upper_bound_perc)
        
        self. initialize_mem_potential(method=self.mem_pot_init_method, 
                                       plot_initial_V_m_dist = plot_initial_V_m_dist)
        
        self. intialize_membrane_time_constant(lower_bound_perc = lower_bound_perc, 
                                               upper_bound_perc = upper_bound_perc,
                                               plot_mem_tau_hist = plot_mem_tau_hist)
        
        self. intialize_synaptic_time_constant(dt, tau, bins = bins, color= color, tc_plot = tc_plot, 
                                               syn_element_no = syn_element_no, 
                                               plot_syn_tau_hist = plot_syn_tau_hist,
                                               lower_bound_perc = lower_bound_perc, 
                                               upper_bound_perc = upper_bound_perc)
        
        self.intialize_ext_synaptic_time_constant(poisson_prop, dt, 
                                                  lower_bound_perc = lower_bound_perc, 
                                                  upper_bound_perc = upper_bound_perc)

        

    def initialize_homogeneously(self, poisson_prop, dt, keep_mem_pot_all_t=False):
        
        ''' cell properties and boundary conditions are constant for all cells'''

        self.transmission_delay = {key: ( np.full( self.n, self.T_specs[key]['mean']) 
                                         / dt ).astype(int)
                                   for key in list (self.T_specs.keys())}
        
        self.spike_thresh = np.full( self.n, 
                                    self.neuronal_consts['spike_thresh']['mean'])
        
        self.u_rest = np.full( self.n, 
                              self.neuronal_consts['u_rest']['mean'])
        
        self.mem_potential = np.random.uniform( low=self.neuronal_consts['u_rest']['mean'], 
                                                high=self.spike_thresh, size=self.n)  # membrane potential
        

        self.membrane_time_constant = np.full(self.n,
                                             self.neuronal_consts['membrane_time_constant']['mean'])
        
        self.tau_ext_pop = {'rise': np.full(self.n, poisson_prop[self.name]['tau']['rise']['mean']) / dt,  # synaptic decay time of the external pop inputs
                            'decay': np.full(self.n, poisson_prop[self.name]['tau']['decay']['mean']) / dt}
        
        tc_list = list (self.tau_specs[ list(self.tau_specs.keys()) [0] ].keys() ) 
            
        for key in list (self.tau_specs.keys()):
            
            self.tau[key] = {tc : np.array( [np.full( self.n, self.tau_specs[key][tc]['mean'][i]) / dt 
                                             for i in range( len(self.tau_specs[key][tc]['mean'] )) 
                                             ] )
                            for tc in tc_list
                            }
            # print( self.tau[key]['rise'] .shape)
            
            
        if self.keep_mem_pot_all_t:
            self.all_mem_pot[:, 0] = self.mem_potential.copy()
            
    def normalize_synaptic_weight(self):

        self.synaptic_weight = {k: v * (self.neuronal_consts['spike_thresh']['mean'] - self.neuronal_consts['u_rest']['mean'])
                                        for k, v in self.synaptic_weight.items() if k[0] == self.name}

    def calculate_input_and_inst_act(self, t, dt, receiving_from_class_list, mvt_ext_inp):
        ''' RATE MODEL: I = Sum (G * m * J) and then
            A = Transfer_f (I)'''

        syn_inputs = np.zeros((self.n, 1))  # = 
        for projecting in receiving_from_class_list:
            num = str(projecting.population_num)
            syn_inputs += ( 
                            self.synaptic_weight[(self.name, projecting.name)] * 
                           np.matmul(
                                   self.connectivity_matrix[ (projecting.name, num) ]  ,
                                   projecting.output[ (self.name, str(self.population_num))][:, - self.transmission_delay[(self.name, 
                                                                                                                           projecting.name)]].reshape(-1, 1)
                                   )
                           )

        inputs = syn_inputs + self.rest_ext_input + mvt_ext_inp
        self.neuron_act = transfer_func(self.threshold, 
                                        self.gain, inputs)
        self.pop_act[t] = np.average(self.neuron_act)

    def update_output(self, dt):
        
        ''' RATE MODEL: Update output m^{\beta\alpha} to each post synaptic population'''
        
        new_output = {k: self.output[k][:, -1].reshape(-1, 1) 
                      for k in self.output.keys()}
        
        for key in self.sending_to_dict:
            for tau in self.synaptic_time_constant[(key[0], self.name)]:
                
                new_output[key] += dt * \
                    (-self.output[key][:, -1].reshape(-1, 1) + self.neuron_act) / \
                        tau
            self.output[key] = np.hstack((self.output[key][:, 1:], new_output[key]))

    def cal_ext_inp(self, dt, t):

        # choose method of exerting external input from dictionary of methods
        I_ext = self.ext_inp_method_dict[self.ext_inp_method](dt) + self.external_inp_t_series[:, t]
        
        self.I_syn['ext_pop', '1'], self.I_rise['ext_pop', '1'] = self.input_integ_method_dict[self. ext_input_integ_method](I_ext, dt,
                                                                            I_rise = self.I_rise['ext_pop', '1'],
                                                                            I = self.I_syn['ext_pop', '1'],
                                                                            tau_rise = self.tau_ext_pop['rise'],
                                                                            tau_decay = self.tau_ext_pop['decay'])
        
    def constant_ext_input_with_noise(self, dt):
        self.noise =  self.noise_generator_dict [self.noise_method] (self.noise_amplitude, 
                                                                self.noise_std, 
                                                                self.n, 
                                                                dt,
                                                                self.sqrt_dt, 
                                                                tau = self.noise_tau, 
                                                                noise_dt_before = self.noise
                                                                )

        
        return self.rest_ext_input + self.noise.reshape(-1,)
    
    def poissonian_ext_inp(self, dt):

        poisson_spikes = possion_spike_generator(
            self.n, self.n_ext_population, self.FR_ext, dt)
        I_ext = self.cal_input_from_poisson_spikes(poisson_spikes, dt)
        
        return I_ext




    def constant_ext_input(self, dt):

        return self.rest_ext_input

    def cal_input_from_poisson_spikes(self, poisson_spikes, dt):

        return (np.sum(poisson_spikes, axis=1) / dt  # normalize dirac delta spike amplitude
               * self.syn_weight_ext_pop
               * self.membrane_time_constant
               ).reshape(-1,)

    def cal_synaptic_input(self, dt, projecting, t):
        ''' Calcculate synaptic input of the given projection'''
        
        num = str(projecting.population_num)
        name = projecting.name
        # print(projecting.spikes[:, t - self.transmission_delay[(self.name, name)]].shape)
        self.syn_inputs[name, num] = (self.synaptic_weight[(self.name, name)] *
                                    np.matmul(self.connectivity_matrix[(name, num)],
                                               np.diagonal( projecting.spikes[:, 
                                                   t - self.transmission_delay[(self.name, name)]])
                                            ) / dt * \
                                    self.membrane_time_constant).reshape(-1,)
                                    
        # print(name, np.average(np.matmul(self.connectivity_matrix[(name, num)],
        #                                         projecting.spikes[:, int(
        #                                             t - self.transmission_delay[(self.name, name)] / dt)]
        #                                     ) ))
        
    def sum_synaptic_input(self, receiving_from_class_list, dt, t):
        ''' Sum the synaptic input from all projections'''
        
        synaptic_inputs = np.zeros(self.n)
        
        for projecting in receiving_from_class_list:

            self.cal_synaptic_input(dt, projecting, t)
            synaptic_inputs = synaptic_inputs + \
                              self.sum_components_of_one_synapse(t, dt, 
                                                                 projecting.name, 
                                                                 str(projecting.population_num),
                                                                 pre_n_components = self.pre_n_components[projecting.name])
                              

        return synaptic_inputs

    def sum_synaptic_input_one_step_ahead_with_no_spikes(self, receiving_from_class_list, dt):
        
        '''Calculate I_syn(t+dt) one component (e.g. GABA-A or GABA-b) 
        assuming that there are no spikes between time t and t+dt '''
        
        synaptic_inputs = np.zeros(self.n)
        for projecting in receiving_from_class_list:

            synaptic_inputs =  synaptic_inputs + \
                                self.sum_components_of_one_synapse_one_step_ahead_with_no_spikes(dt, projecting.name,
                                                                                                str(projecting.population_num),
                                                                                                pre_n_components = 
                                                                                                self.pre_n_components[projecting.name]
                                                                                                )
                                

        return synaptic_inputs

    def sum_components_of_one_synapse(self, t, dt, pre_name, pre_num, pre_n_components=1):
        
        '''Calculate I_syn(t) as the sum of  all synaptic components  (e.g. GABA-A or GABA-b)  '''

        sum_components = np.zeros(self.n)
        
        for i in range(self.pre_n_components[pre_name]):
            
            (self.I_syn[pre_name, pre_num][:, i], 
             self.I_rise[pre_name, pre_num][:, i] ) = self.input_integ_method_dict[self.syn_input_integ_method](self.syn_inputs[pre_name, pre_num], dt, 
                                                                                                                I_rise = self.I_rise[pre_name,
                                                                                                                pre_num][:, i],
                                                                                                                I = self.I_syn[pre_name, pre_num][:, i],
                                                                                                                tau_rise = self.tau[(self.name, pre_name)]['rise'][i,:],
                                                                                                                tau_decay = self.tau[(self.name, pre_name)]['decay'][i,:])
            self.representative_inp[pre_name, pre_num][t,
                i] = self.I_syn[pre_name, pre_num][0, i]
            
            # print(self.name, pre_name, self.syn_component_weight[self.name, pre_name], i)

            sum_components =  sum_components + \
                              self.I_syn[pre_name, pre_num][:, i] * \
                              self.syn_component_weight[self.name, pre_name][i]
            
        return sum_components

    def sum_components_of_one_synapse_one_step_ahead_with_no_spikes(self, dt, pre_name, pre_num, pre_n_components=1):
        
        '''Calculate I_syn(t+dt) assuming that there are no spikes between time t and t+dt '''

        sum_components = np.zeros(self.n)
        for i in range(self.pre_n_components[pre_name]):

            I_syn_next_dt, _ = self.input_integ_method_dict[self.syn_input_integ_method](0, dt, 
                                                I_rise = self.I_rise[pre_name, pre_num][:, i],
                                                I = self.I_syn[pre_name, pre_num][:, i],
                                                tau_rise = self.tau[(self.name, pre_name)]['rise'][i,:],
                                                tau_decay = self.tau[(self.name, pre_name)]['decay'][i,:])

            sum_components = sum_components + I_syn_next_dt * self.syn_component_weight[self.name, pre_name][i]
            
        return sum_components

    def solve_IF_without_syn_input(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = np.zeros(self.n)
        self.update_potential(synaptic_inputs, dt, t, receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        # self.reset_potential(spiking_ind)
        self.reset_potential_with_interpolation(spiking_ind, dt)

    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = self.sum_synaptic_input(receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, t, receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        # self.reset_potential(spiking_ind)
        self.reset_potential_with_interpolation(spiking_ind,dt)
        
        # self.update_representative_measures(t)
        # if self.keep_mem_pot_all_t:
        #     self.all_mem_pot[:, t] = self.mem_potential
        
    def cal_coherence(self, dt, sampling_t_distance_ms = 1):
        
        ''' Measure of synchrony as defined in section <2.3> in Hansel et al. 1998'''
        half_t = int (self.all_mem_pot.shape[1] / 2)
        sampling_t_distance = int( sampling_t_distance_ms / dt)
        
        V_mean_t = self.average_mem_pot_over_n(start = half_t, sampling_t_distance = sampling_t_distance)
        Delta_N = np.average( np.power( V_mean_t , 2)) - \
                  np.power( np.average( V_mean_t ),
                            2)
        print("Delta_N", Delta_N)

        V_mean_n = self.average_mem_pot_over_t(start = half_t, sampling_t_distance = sampling_t_distance)
        V_2_mean_n = self.average_mem_pot_2_over_t(start = half_t, sampling_t_distance = sampling_t_distance)
        normalizing_factor = np.average( V_2_mean_n - \
                                       np.power( V_mean_n , 2)
                                       )
        print('normalizing factor', normalizing_factor)
        coherence = Delta_N / normalizing_factor
        
        return coherence
    
    def average_mem_pot_over_n(self, start  = 0, sampling_t_distance = 1):
        
        return np.average(self.all_mem_pot[:, start: -1 : sampling_t_distance], axis = 0)
    
    def average_mem_pot_over_t(self, start  = 0, sampling_t_distance = 1):
        
        return np.average(self.all_mem_pot[:, start: -1 : sampling_t_distance], axis = 1)
    
    def average_mem_pot_2_over_t(self, start  = 0, sampling_t_distance = 1):
       
        return np.average( np.power(
                                self.all_mem_pot[:, start: -1 : sampling_t_distance], 2),
                         axis = 1)
    
    def update_representative_measures(self, t):

        self.voltage_trace[t+1] = self.mem_potential[0]
        self.representative_inp['ext_pop', '1'][t] = self.I_syn['ext_pop', '1'][0]
        self.ext_input_all_t[:, t] = self.I_syn['ext_pop', '1']

    def update_potential(self, synaptic_inputs, dt, t, receiving_from_class_list):

        # EIF
        # self.mem_potential += (-self.mem_potential+ inputs+ self.neuronal_consts['nonlin_sharpness'] *np.exp((self.mem_potential-
        #                       self.neuronal_consts['nonlin_thresh'])/self.neuronal_consts['nonlin_sharpness']))*dt/self.membrane_time_constant
        # LIF
        self.mem_pot_before_spike = self.mem_potential.copy()
        
        V_prime = f_LIF(self.membrane_time_constant, 
                        self.mem_potential,
                        self.u_rest, 
                        self.I_syn['ext_pop', '1'], 
                        synaptic_inputs)
        self.voltage_trace[t] = self.mem_potential[0]

        # self.mem_potential = fwd_Euler(dt, self.mem_potential, V_prime)
        I_syn_next_dt = self. sum_synaptic_input_one_step_ahead_with_no_spikes( receiving_from_class_list, 
                                                                                dt)
        self.mem_potential = Runge_Kutta_second_order_LIF( dt, 
                                                          self.mem_potential, 
                                                          V_prime,  
                                                          self.membrane_time_constant, 
                                                          I_syn_next_dt, 
                                                          self.u_rest, 
                                                          self.I_syn['ext_pop', '1'],
                                                          self.half_dt)

    def find_spikes(self, t):

        spiking_ind = np.where(self.mem_potential > self.spike_thresh)
        self.spikes[spiking_ind, t] = 1
        
        return spiking_ind

    def reset_potential(self, spiking_ind):

        self.mem_potential[spiking_ind] = self.u_rest[spiking_ind]

    def reset_potential_with_interpolation(self, spiking_ind, dt):
        
        ''' set the potential at firing times according to Hansel et. al. (1998)'''
        
        self.mem_potential[spiking_ind] = linear_interpolation(self.mem_potential[spiking_ind], 
                                                               self.spike_thresh[spiking_ind],
                                                                dt, 
                                                                self.mem_pot_before_spike[spiking_ind], 
                                                                self.u_rest[spiking_ind], 
                                                                self.membrane_time_constant[spiking_ind])

    def cal_population_activity(self, dt, t, lower_bound_perc = 0.8, upper_bound_perc = 1.2):
        
        '''SNN: return pop activity as mean number of spikes per second == Hz'''
        self.pop_act[t] = np.average(self.spikes[:, t], axis=0)/ (dt/1000)

    def cal_population_activity_all_t(self, dt):
                self.pop_act = np.average(self.spikes, axis=0)/ (dt/1000)

    # def reset_ext_pop_properties(self, poisson_prop, dt):
    #     '''reset the properties of the external poisson spiking population'''

    #     # external population size
    #     self.n_ext_population = poisson_prop[self.name]['n']
    #     self.firing_of_ext_pop = poisson_prop[self.name]['firing']

    #     self.tau_ext_pop = {'rise': truncated_normal_distributed(poisson_prop[self.name]['tau']['rise']['mean'],
    #                                                             poisson_prop[self.name]['tau']['rise']['var'], self.n,
    #                                                             lower_bound_perc=lower_bound_perc, upper_bound_perc=upper_bound_perc) / dt,
    #                         'decay': truncated_normal_distributed(poisson_prop[self.name]['tau']['decay']['mean'],
    #                                                             poisson_prop[self.name]['tau']['decay']['var'], self.n,
    #                                                             lower_bound_perc=lower_bound_perc, upper_bound_perc=upper_bound_perc) / dt}
    #     self.syn_weight_ext_pop = poisson_prop[self.name]['g']

    def set_noise_param(self, noise_variance, noise_amplitude):
        
        self.noise_variance = noise_variance[self.name]
        self.noise_std = np.sqrt(noise_variance[self.name])
        self.noise_amplitude = noise_amplitude[self.name]

    def set_connections(self, K, N):
        ''' creat J_{ij} connection matrix

        '''
        same_pop = False
        self.K_connections = {k: v for k, v in K.items() if k[0] == self.name}
        for projecting in self.receiving_from_list:

            n_connections = self.K_connections[(self.name, projecting[0])]
            if self.name == projecting[0]:
                same_pop = True
            self.connectivity_matrix[projecting] = build_connection_matrix(
                self.n, N[projecting[0]], n_connections, same_pop = same_pop)


        
    def normalize_synaptic_weight_by_N(self):
        self.synaptic_weight = {
            k: v / self.K_connections[k] for k, v in self.synaptic_weight.items()}

    def clear_history(self, mem_pot_init_method=None):

        self.pop_act[:] = 0
        if self.neuronal_model == 'rate':

            for k in self.output.keys():
                self.output[k][:] = 0
            self.input[:] = 0
            self.neuron_act[:] = 0
            self.mvt_ext_input = np.zeros_like(self.mvt_ext_input)
            self.external_inp_t_series[:, :] = 0

        if self.neuronal_model == 'spiking':

            for k in self.receiving_from_list:

                self.I_rise[k][:] = 0
                self.I_syn[k][:] = 0
                self.representative_inp[k][:] = 0
                self.syn_inputs[k][:] = 0

            self.I_syn['ext_pop', '1'][:] = 0
            self.voltage_trace[:] = 0
            self.representative_inp['ext_pop', '1'][:] = 0

            if mem_pot_init_method == None:  # if not specified initialize as before
                mem_pot_init_method = self.mem_pot_init_method

            self.initialize_mem_potential(method=mem_pot_init_method)
            self.spikes[:, :] = 0

    def smooth_pop_activity(self, dt, window_ms=5):
        
        self.pop_act = moving_average_array(self.pop_act, int(window_ms / dt))


    def average_pop_activity(self,ind_start, ind_end):
        
        average = np.average(self.pop_act[ind_start : ind_end])
        std = np.std(self.pop_act[ind_start : ind_end])
        return average, std
    
    def set_synaptic_weights(self, G):

        # filter based on the receiving nucleus
        self.synaptic_weight = {k: v for k, v in G.items() if k[0] == self.name}

    def set_synaptic_time_scales(self, synaptic_time_constant):

        self.synaptic_time_constant = {
            k: v for k, v in synaptic_time_constant.items() if k[1] == self.name}

    def incoming_rest_I_syn(self, proj_list, A, dt):
        
        # I_syn = np.sum([self.synaptic_weight[self.name, proj] * A[proj] / 1000 * self.K_connections[self.name, proj]
        #                * len(self.tau[self.name, proj]['rise']) for proj in proj_list])*self.membrane_time_constant
        I_syn = np.sum([self.synaptic_weight[self.name, proj] * 
                        A[proj] / 1000  * 
                        self.K_connections[self.name, proj] *
                        np.sum(self.syn_component_weight[self.name, proj]) \
                           for proj in proj_list]) * \
                self.membrane_time_constant

        return I_syn

#     def incoming_rest_I_syn_multi_comp_synapses(self, proj_list, A):

#         I_syn = np.sum( [ self.synaptic_weight[self.name,proj] * A[proj] / 1000 * self.K_connections[self.name,proj] * len ( self.tau[self.name,proj]['rise'] )
#                           for proj in proj_list]) * self.membrane_time_constant

#         return I_syn

    def set_ext_input(self, A, A_mvt, D_mvt, t_mvt, t_list, dt, end_of_nonlinearity=25):

        proj_list = [k[0] for k in list(self.receiving_from_list)]

        if self.neuronal_model == 'rate':

            # self.rest_ext_input = self.basal_firing/self.gain - np.sum([self.synaptic_weight[self.name,proj]*A[proj] for proj in proj_list]) + self.threshold
            # self.mvt_ext_input = self.mvt_firing/self.gain - np.sum([self.synaptic_weight[self.name,proj]*A_mvt[proj] for proj in proj_list]) + self.threshold - self.rest_ext_input
            self.rest_ext_input = self.basal_firing/self.gain - \
                np.sum([self.synaptic_weight[self.name, proj]*A[proj] *
                       self.K_connections[self.name, proj] for proj in proj_list]) + self.threshold
            self.mvt_ext_input = self.mvt_firing/self.gain - \
                np.sum([self.synaptic_weight[self.name, proj]*A_mvt[proj] * self.K_connections[self.name, proj]
                       for proj in proj_list]) + self.threshold - self.rest_ext_input
            self.external_inp_t_series = mvt_step_ext_input(
                D_mvt, t_mvt, self.ext_inp_delay, self.mvt_ext_input, t_list * dt)

        else:  # for the firing rate model the ext input is reported as the firing rate of the ext pop needed to provide the desired firing rate.
            
            I_syn = self.incoming_rest_I_syn(proj_list, A, dt)

            # print('I_syn', np.average(I_syn))
            self.sum_syn_inp_at_rest = I_syn

            if self.ext_inp_method == 'Poisson':
                self._set_ext_inp_poisson(I_syn)

            elif self.ext_inp_method == 'const+noise' or self.ext_inp_method == 'const':
                self._set_ext_inp_const_plus_noise(I_syn, end_of_nonlinearity, dt)
                
            else:
                raise ValueError('external input handling method not right!')
            self._save_init()

        # self.external_inp_t_series =  mvt_step_ext_input(D_mvt,t_mvt,self.ext_inp_delay,self.mvt_ext_input, t_list*dt)

    def set_init_from_pickle(self, filepath, set_noise=True):

        f = load_pickle(filepath)
        self.FR_ext = f['FR_ext']
        self.spike_thresh = f['spike_thresh']
        self.membrane_time_constant = f['membrane_time_constant']
        self.tau_ext_pop = f['tau_ext_pop']
        if set_noise:
            try:
                self.noise_variance = f['noise_variance']
                self.noise_std = np.sqrt(self.noise_variance)
            except KeyError:
                print("Watchout couldn't set noise for the " +
                      self.name + ", set it manually!")
                pass

    def _save_init(self):
        
        if self.save_init:
            init = {'name': self.name,
                    'spike_thresh': self.spike_thresh,
                    'membrane_time_constant': self.membrane_time_constant,
                    'tau_ext_pop': self.tau_ext_pop,
                    'FR_ext': self.FR_ext,
                    'noise_variance': self.noise_variance}
            pickle_obj(init, os.path.join(self.path, 'tau_m_' + str(self.neuronal_consts['membrane_time_constant']['mean']).replace('.', '-') + '_' + self.name + '_A_' + str(self.basal_firing).replace('.', '-') + '_N_' +
                str(self.n) + '_T_' + str(self.t_sim) + '_noise_var_' + str(self.noise_variance).replace('.', '-') + '.pkl'))

    def _set_ext_inp_poisson(self, I_syn):
        
        exp = np.exp(-1/(self.membrane_time_constant * self.basal_firing/1000))
        self.rest_ext_input = ((self.spike_thresh - self.u_rest) / (1-exp) - I_syn)
        self.FR_ext = self.rest_ext_input / self.syn_weight_ext_pop / \
            self.n_ext_population / self.membrane_time_constant

    def _set_ext_inp_const_plus_noise(self, I_syn, end_of_nonlinearity, dt):
        # linear regime if decided not to derive from reponse curve (works for low noise levels)
        if self.basal_firing > end_of_nonlinearity and not self.set_input_from_response_curve:
            self._set_ext_inp_poisson(I_syn)
        else:
            self.rest_ext_input = (self.FR_ext * self.syn_weight_ext_pop * \
                                   self.n_ext_population * self.membrane_time_constant - 
                                   I_syn )#.reshape(-1,1)
            self.I_ext_0 =  np.average(self.rest_ext_input + I_syn)
            # print(self.name, '<I_ext + I_syn> = ', self.I_ext_0)
            
            # self.set_noise_param(self.I_ext_0/10, self.noise_amplitude)
    def set_ext_inp_const_plus_noise_collective(self, FR_range, t_list, dt, receiving_class_dict, n_FR = 50,
                                                 if_plot=False, end_of_nonlinearity=25, maxfev=5000, c='grey'):
        
        if self.basal_firing < end_of_nonlinearity:
            FR_list = spacing_with_high_resolution_in_the_middle(n_FR , * FR_range)
        else:
            FR_list = np.linspace(*FR_range, n_FR).reshape(-1,1)
        FR_sim_all_neurons = self.run_for_all_FR_ext(FR_list, t_list, dt, receiving_class_dict)
        FR_sim = np.average(FR_sim_all_neurons, axis = 0)
        if self.basal_firing > end_of_nonlinearity:
            
            self.FR_ext = extrapolate_FR_ext_from_neuronal_response_curve_high_act(FR_list.reshape(-1,) * 1000, FR_sim, self.basal_firing, if_plot= if_plot, 
                                                                                   end_of_nonlinearity=end_of_nonlinearity, maxfev=maxfev, 
                                                                                   tau= np.average(self.membrane_time_constant), g_ext=self.syn_weight_ext_pop, 
                                                                                   N_ext=self.n_ext_population, noise_var=self.noise_variance, c=c)
        else:
            
            self.FR_ext =  extrapolate_FR_ext_from_neuronal_response_curve(FR_list * 1000 , FR_sim, self.basal_firing, if_plot= if_plot, 
                                                                                   end_of_nonlinearity=end_of_nonlinearity, maxfev=maxfev, 
                                                                                   tau= np.average(self.membrane_time_constant), g_ext=self.syn_weight_ext_pop, 
                                                                                   N_ext=self.n_ext_population, noise_var=self.noise_variance, c=c)
        self.clear_history()
        print(self.name, "FR_ext = ", self.FR_ext)
        return self.FR_ext
    
    def estimate_needed_external_input(self, all_FR_list, dt, t_list, receiving_class_dict, if_plot=False, end_of_nonlinearity=25,
                                        n_FR=50, left_pad=0.001, right_pad=0.001, maxfev=5000, ax=None, c='grey'):
        start = timeit.default_timer()
        # provide the array for all neurons
        all_FR_list_2D = np.repeat(all_FR_list.reshape(-1, 1), self.n, axis=1)
        FR_sim = self.Find_threshold_of_firing(
            all_FR_list_2D, t_list, dt, receiving_class_dict)
        FR_list = find_FR_ext_range_for_each_neuron(
            FR_sim, all_FR_list, self.init_method, n_FR=n_FR, left_pad=left_pad, right_pad=right_pad)
        FR_sim = self.run_for_all_FR_ext(FR_list, t_list, dt, receiving_class_dict)
        self. set_FR_ext_each_neuron(FR_list, FR_sim, dt,  extrapolate=extrapolate_FR_ext_from_neuronal_response_curve,
                                    if_plot=if_plot, ax=ax, end_of_nonlinearity=end_of_nonlinearity, maxfev=maxfev, c=c)
        self.clear_history()
        stop = timeit.default_timer()
        print('t for I_ext init ' + self.name + ' =', round(stop - start, 2), ' s')

    def estimate_needed_external_input_high_act(self, FR_range, dt, t_list, receiving_class_dict, if_plot=False, n_FR=25, ax=None, c='grey', 
                                                set_FR_range_from_theory=True):

        start = timeit.default_timer()

        # if set_FR_range_from_theory: # find range based on theory. Only works for neurons bahaving close to tbe theory range
        #     FR_start =  FR_ext_of_given_FR_theory(self.spike_thresh, self.u_rest, self.membrane_time_constant, self.syn_weight_ext_pop, FR_range [0], self.n_ext_population)
        #     FR_end =   FR_ext_of_given_FR_theory(self.spike_thresh, self.u_rest, self.membrane_time_constant, self.syn_weight_ext_pop, FR_range [-1], self.n_ext_population)

        if set_FR_range_from_theory:  # scale range based on the homogeneous neurons
            FR_mean_start = FR_ext_of_given_FR_theory(self.neuronal_consts['spike_thresh']['mean'], self.neuronal_consts['u_rest']['mean'],
                                                      self.neuronal_consts['membrane_time_constant']['mean'], self.syn_weight_ext_pop, FR_range[0], self.n_ext_population)
            FR_mean_end = FR_ext_of_given_FR_theory(self.neuronal_consts['spike_thresh']['mean'], self.neuronal_consts['u_rest']['mean'],
                                                    self.neuronal_consts['membrane_time_constant']['mean'], self.syn_weight_ext_pop, FR_range[-1], self.n_ext_population)
            FR_start = (FR_ext_of_given_FR_theory(self.spike_thresh, self.u_rest, self.membrane_time_constant, self.syn_weight_ext_pop, FR_range[0], self.n_ext_population)
                        / FR_mean_start * FR_range[0])
            FR_end = (FR_ext_of_given_FR_theory(self.spike_thresh, self.u_rest, self.membrane_time_constant, self.syn_weight_ext_pop, FR_range[-1], self.n_ext_population)
                        / FR_mean_end * FR_range[-1])
        # else:
            # FR_start = FR_range[0]
            # FR_end = FR_range[-1]
        FR_list = np.repeat(np.linspace(*FR_range, n_FR).reshape(-1,1), self.n, axis = 1)
        # FR_list = np.linspace(*FR_range, n_FR).reshape(-1,1)
        FR_sim = self.run_for_all_FR_ext(FR_list, t_list, dt, receiving_class_dict)
        self. set_FR_ext_each_neuron(
            FR_list, FR_sim, dt, extrapolate=extrapolate_FR_ext_from_neuronal_response_curve_high_act, if_plot=if_plot, ax=ax, c=c)
        self.clear_history()
        stop = timeit.default_timer()
        print('t for I_ext init ' + self.name + ' =', round(stop - start, 2), ' s')
        
    def set_FR_ext_each_neuron(self, FR_list, FR_sim, dt, extrapolate=extrapolate_FR_ext_from_neuronal_response_curve,
                                if_plot=False, ax=None, end_of_nonlinearity=25, maxfev=5000, c='grey'):

        self.FR_ext = np.zeros((self.n))

        if self.init_method == 'homogeneous':  # and FR_list.shape[1] == 1:
            rep_FR_ext = extrapolate(FR_list[:, 0] * 1000, np.average(FR_sim, axis=0),
                                        self.basal_firing, ax=ax, if_plot=if_plot, end_of_nonlinearity=end_of_nonlinearity, maxfev=maxfev,
                                        tau=self.membrane_time_constant[0], g_ext=self.syn_weight_ext_pop, N_ext=self.n_ext_population, noise_var=self.noise_variance, c=c)
            self.FR_ext = np.full(self.n, rep_FR_ext)

        else:
            for i in range(self.n):
                self.FR_ext[i] = extrapolate(FR_list[:, i] * 1000, FR_sim[i, :], self.basal_firing, ax=ax,
                                            if_plot=if_plot, end_of_nonlinearity=end_of_nonlinearity, maxfev=maxfev,
                                            tau=self.membrane_time_constant[i], g_ext=self.syn_weight_ext_pop, N_ext=self.n_ext_population, noise_var=self.noise_variance, c=c)

    def run_for_all_FR_ext(self, FR_list, t_list, dt, receiving_class_dict):

        FR_sim = np.zeros((self.n, len(FR_list)))

        for i in range(FR_list.shape[0]):

            self.clear_history()
            self.rest_ext_input = FR_list[i, :]  * self.membrane_time_constant * \
                self.n_ext_population * self.syn_weight_ext_pop
            self. run(dt, t_list, receiving_class_dict)
            FR_sim[:, i] = np.average( self.spikes[:, int(len(t_list)/2):], 
                                      axis=1) /  (dt/1000)
            print('FR = ', np.average(FR_list[i, :]), np.average(FR_sim[:, i]))

        return FR_sim

    def plot_mem_potential_distribution_of_one_t(self, t,  ax=None, bins=50, color='gray'):
        
        fig, ax = get_axes(ax)
        ax.hist(self.all_mem_pot[:, t] / self.n, bins=bins,
                color=color, label=self.name, density=True, stacked=True)
        ax.set_xlabel('Membrane potential (mV)', fontsize=15)
        ax.set_ylabel(r'$Probability\ density$', fontsize=15)
        # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
        ax.legend(fontsize=15,  framealpha = 0.1, frameon = False)

    def plot_mem_potential_distribution_of_all_t(self, ax=None, bins=50, color='grey'):
        a = self.all_mem_pot.copy()
        fig, ax = get_axes(ax)
        ax.hist(a.reshape(int(a.shape[0] * a.shape[1]), 1), bins=bins,
                color=color, label=self.name, density=True, stacked=True, alpha = 0.2)
        ax.set_xlabel('Membrane potential (mV)', fontsize=15)
        ax.set_ylabel(r'$Probability$', fontsize=15)
        # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
        ax.legend(fontsize=15)
    def plot_mem_potential_distribution_of_all_t(self, key, dt, ax=None, bins=50, 
                                                 color='grey', tc = 'decay', syn_element_no = 0):

        fig, ax = get_axes(ax)
        ax.hist(self.tau[key][tc][syn_element_no,:] * dt, bins=bins,
                label='from '+ key[1] , density=True, stacked=True, color=color)
        ax.set_xlabel('synaptic ' + tc + ' time constant (ms)', fontsize=15)
        ax.set_ylabel(r'$Probability\ density$', fontsize=15)
        
        ax.set_title(self.name, fontsize=15)
        ax.legend(fontsize=15,  framealpha = 0.1, frameon = False)
        
    def Find_threshold_of_firing(self, FR_list, t_list, dt, receiving_class_dict):
        FR_sim = np.zeros((self.n, len(FR_list)))

        for i in range(FR_list.shape[0]):

            self.clear_history()
            self.rest_ext_input = FR_list[i, :]  * self.membrane_time_constant * \
                self.n_ext_population * self.syn_weight_ext_pop
            self. run(dt, t_list, receiving_class_dict)
            FR_sim[:, i] = np.average(
                self.spikes[:, int(len(t_list)/2):], axis=1)/(dt/1000)
            print('FR = ', np.average(FR_list[i, :]), np.average(FR_sim[:, i]))
            if np.all(FR_sim[:, i] > 1):
                print('done!')
                break
        return FR_sim

    def run(self, dt, t_list, receiving_class_dict):
        
        for t in t_list:  # run temporal dynamics
        
            self.solve_IF_without_syn_input(
                t, dt, receiving_class_dict[(self.name, str(self.population_num))])
            
    def find_spike_times_all_neurons(self, start = 0, neurons_ind = 'all'):
        
        if isinstance(neurons_ind, str)  and neurons_ind == 'all':    
            spike_times = np.where(self.spikes == 1)[1]
        elif hasattr(neurons_ind, '__len__') : # an actual list of neurons is given
            spike_times = np.where(self.spikes[neurons_ind,:] == 1)[1]
        else:
            raise("Invalid input for 'ind_neurons'. It has to be either 'all' or the list of indices of neurons to be included.")
            
        return spike_times[spike_times > start ] - start
      
    def find_peaks_of_pop_act(self, dt, low_f, high_f, filter_order = 6, height = 1, start = 0):
        act = self.butter_bandpass_filter_pop_act_not_modify( dt, low_f, high_f, order= filter_order)
        peaks,_ = signal.find_peaks(act[start:], height = height)
        return peaks
    

    def find_phase_hist_of_spikes(self, dt, low_f, high_f, filter_order = 6, height = 1,start = 0, ref_peaks = [], n_bins = 20, 
                                    total_phase = 360, ref_nuc_name = 'self', neurons_ind = 'all', troughs = False):
        
        all_spikes = self.find_spike_times_all_neurons(start = start, neurons_ind = neurons_ind)
        if len( ref_peaks ) == 0: # if peaks are given as argument phases are calculated relative to them
            ref_peaks = self.find_peaks_of_pop_act(dt, low_f, high_f, filter_order = filter_order, height = height, start = start)
        
        phase_all_spikes = np.zeros(len(all_spikes) * 2 + 1)
        count = 0
        left_peak_series, right_peak_series = set_peak_iterators(total_phase, ref_peaks, troughs = troughs)
        # print(total_phase, ref_peaks)
        for left_peak, right_peak in zip(left_peak_series, right_peak_series): # leave out the last peaks beacause sosfiltfilt affects the data boundaries
            # print(left_peak, right_peak)
            phase, n_spikes_of_this_cycle = find_phase_of_spikes_bet_2_peaks(left_peak, right_peak, all_spikes, total_phase = total_phase)
            phase_all_spikes[count : n_spikes_of_this_cycle + count] = phase
            count += n_spikes_of_this_cycle
          
        bins = np.linspace(0, total_phase, endpoint=True, num = n_bins)
        frq , edges = np.histogram(phase_all_spikes[:count], bins=bins)
        self.spike_rel_phase_hist[ref_nuc_name] = frq, edges[:-1]
        return self.spike_rel_phase_hist[ref_nuc_name]
    

    
    def scale_synaptic_weight(self):
        if self.scale_g_with_N:
            self.synaptic_weight = {
                k: v/self.K_connections[k] for k, v in self.synaptic_weight.items() if k[0] == self.name}

    def change_pop_firing_rate(self, FR_ext, A, A_mvt = None, D_mvt = None , t_mvt = None, t_list = None,
                                  dt = None, end_of_nonlinearity=30):
        self.FR_ext = FR_ext
        self.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list,
                                      dt, end_of_nonlinearity=end_of_nonlinearity)
    def change_basal_firing(self, A_new):
        self.basal_firing = A_new
        
    

    def find_freq_of_pop_act_spec_window(self, start, end, dt, peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False,
                                      method='zero_crossing', plot_sig=False, plot_spectrum=False, ax=None, c_spec='navy', fft_label='fft',
                                      spec_figsize=(6, 5), find_beta_band_power=False, fft_method='rfft', n_windows=6, include_beta_band_in_legend=False,
                                      divide_beta_band_in_power = False, normalize_spec = True, include_peak_f_in_legend = True, 
                                      low_beta_range = [12,20], high_beta_range = [20, 30], low_gamma_range = [30, 70],
                                      min_f = 100, max_f = 300, plot_sig_thresh = False,  n_std_thresh = 2, save_gamma = False):
        
        ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
            the plateau and in case it is oscillation determine the frequency '''
            
        if method not in ["fft", "zero_crossing"]:
            raise ValueError("method must be either 'fft', or 'zero_crossing'")

        sig = trim_start_end_sig_rm_offset(
            self.pop_act, start, end, method=self.trim_sig_method_dict[self.neuronal_model])
        # cut_sig_ind = cut_plateau(sig,  epsilon=cut_plateau_epsilon)
        # plateau_y = find_mean_of_signal(sig, cut_sig_ind)
        mean_sig = np.average(sig)
        # _plot_signal(plot_sig, start, end, dt, sig, plateau_y, cut_sig_ind)
        if_stable = False
        # if len(cut_sig_ind) > 0:  # if it's not all plateau from the beginning

        sig = sig - mean_sig

        # if method == 'zero_crossing':
            
        #     n_half_cycles, freq = zero_crossing_freq_detect(
        #         sig[cut_sig_ind], dt / 1000)

        # elif method == 'fft':

        f, pxx, freq = freq_from_fft(sig, dt / 1000, plot_spectrum=plot_spectrum, ax=ax, c=c_spec, label=fft_label, figsize=spec_figsize,
                                     method=fft_method, n_windows=n_windows, include_beta_band_in_legend=include_beta_band_in_legend,
                                     normalize_spec=normalize_spec, include_peak_f_in_legend = include_peak_f_in_legend,
                                     plot_sig_thresh = plot_sig_thresh, min_f = min_f, max_f = max_f, n_std_thresh = n_std_thresh)
        if find_beta_band_power:
            
            if divide_beta_band_in_power:
                
                low_beta_band_power = beta_bandpower(f, pxx, *low_beta_range)
                high_beta_band_power = beta_bandpower(f, pxx, *high_beta_range)
                if save_gamma:
                    low_gamma_band_power = beta_bandpower(f, pxx, * low_gamma_range)
            else:
        
                beta_band_power = beta_bandpower(f, pxx)
        
        else: beta_band_power = None
        
        n_half_cycles = None

        if freq != 0:  # then check if there's oscillations

            # perc_oscil = max_non_empty_array(cut_sig_ind) / len(sig) * 100

            # if check_stability:
            #     if_stable = if_stable_oscillatory(sig, max(
            #         cut_sig_ind), peak_threshold, smooth_kern_window, amp_env_slope_thresh=- 0.05)
            if divide_beta_band_in_power:
                if save_gamma:
                    return n_half_cycles, 0, freq, if_stable, [low_beta_band_power, high_beta_band_power, low_gamma_band_power], f, pxx
                else:
                    return n_half_cycles, 0, freq, if_stable, [low_beta_band_power, high_beta_band_power], f, pxx

            else:
                return n_half_cycles, 0, freq, if_stable, beta_band_power, f, pxx

        else:
            
            print("Freq = 0")
            return 0, 0, 0, False, None, f, pxx


    def butter_bandpass_filter_pop_act(self, dt, low, high, order=6):
        self.pop_act_filtered = True
        self.pop_act = butter_bandpass_filter(
            self.pop_act, low, high, 1 / (dt / 1000), order=order)
        
    def butter_bandpass_filter_pop_act_not_modify(self, dt, low, high, order=6):
        
        return butter_bandpass_filter(
            np.copy(self.pop_act), low, high, 1 / (dt / 1000), order=order)

    def additive_ext_input(self, ad_ext_inp):
        """ to add a certain external input to all neurons of the neucleus."""
        self.rest_ext_input = self.rest_ext_input + ad_ext_inp
       
def smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.smooth_pop_activity(dt, window_ms=window_ms)
            
    return nuclei_dict

def change_basal_firing_all_nuclei(A_new, nuclei_dict):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.change_basal_firing(A_new[nucleus.name])
            
    return nuclei_dict

def change_state_all_nuclei(new_state, nuclei_dict):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.state = new_state
            
    return nuclei_dict

def change_noise_all_nuclei( nuclei_dict, noise_variance, noise_amplitude):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.set_noise_param(noise_variance, noise_amplitude)
            
    return nuclei_dict

def plot_histogram(y, bins = 50, title = "", color = 'k'):
    fig, ax = plt.subplots()
    ax.hist(y, bins, color = color)
    ax.set_title(title, fontsize = 15)
    ax.set_xlabel(r'$\tau_{m}(ms)$', fontsize = 15)
    
# def wrap_angles(x):
#     ''' Wrap angles to [-pi, pi)'''
    # x = (x+np.pi) % (2*np.pi) 
    # x[x <= 0] = x[x <= 0] + (2*np.pi)
    # return x - np.pi

def set_peak_iterators(total_phase, ref_peaks, troughs = False):
    if total_phase == 360:
        left_peak_series = ref_peaks[:-2] 
        right_peak_series = ref_peaks[1:-1]
    elif total_phase == 720:
        left_peak_series = ref_peaks[:-2]
        right_peak_series = ref_peaks[2:]
        
    half_mean_cycle = np.average(np.diff(ref_peaks)) / 2
    
    if troughs:
        left_peak_series = left_peak_series - half_mean_cycle
        right_peak_series = right_peak_series - half_mean_cycle
        
    return left_peak_series, right_peak_series
def convert_angle_to_0_360_interval(angle):
   new_angle = np.arctan2(np.sin(angle), np.cos(angle))
   ind  = new_angle <= 0
   new_angle[ind] = abs(new_angle[ind]) + 2 * (np.pi - abs(new_angle[ind]))
   return new_angle * 180/np.pi
# convert_angle_to_0_360_interval(np.array([-np.pi/2, np.pi, np.pi * 2, 3 * np.pi / 2])) 

def circular_hist(ax, x, bins=16, density=True, fill = False, facecolor = 'grey', alpha = 0.8, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # transform to radians again
    x = x * np.pi/180 
    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)


    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,facecolor = facecolor,
                     edgecolor='C0', fill=fill, linewidth=1, alpha = alpha)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

def deg_to_rad(theta):
    return theta * np. pi / 180

def rad_to_deg(theta):
    return theta / np.pi * 180

def circular_bar_plot_as_hist(ax, frq, edges, density=True, fill = False, facecolor = 'grey', alpha = 0.8, offset=0):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    frq : array
        frequency observed in each bin

    edges : array
        bin edges of the hist

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    Returns
    -------

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """


    # transform to radians
    edges = edges * np.pi / 180 
    widths = np.diff(edges)
    widths = np.append(widths, widths[-1]) # only left edges are given
    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = frq / np.sum(frq)
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = frq

    # Plot data on ax
    patches = ax.bar(edges, radius, zorder=1, align='edge', width=widths,facecolor = facecolor,
                     edgecolor='C0', fill=fill, linewidth=1, alpha = alpha)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return edges, patches

def plot_phase_histogram_all_nuclei(nuclei_dict, dt, color_dict, low_f, high_f, filter_order = 6, height = 1, 
                                density = False, n_bins = 16, start = 0, ref_nuc_name = 'self', total_phase = 360, projection = None):
    
    n_plots = len(nuclei_dict)
    fig, axes = plt.subplots(n_plots,1,  subplot_kw=dict(projection= projection), figsize = (5, 10))
    find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = n_bins,
                                      height = height, ref_nuc_name = ref_nuc_name, start = start, 
                                      total_phase = total_phase, only_entrained_neurons = False)
    count = 0
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            ax = axes[count] 
            
            frq, edges = nucleus.spike_rel_phase_hist[ref_nuc_name]
            
            if projection == None:
                width=np.diff(edges) 
                ax.bar(edges, frq, width=np.append(width, width[-1]), align = 'edge', facecolor = color_dict[nucleus.name])
                
            elif projection == 'polar':
                
                circular_bar_plot_as_hist(ax, frq, edges, fill = True, alpha = 0.3, density = density,  facecolor = color_dict[nucleus.name])

            ax.set_title(nucleus.name + ' relative to ' + ref_nuc_name, fontsize = 15, color = color_dict[nucleus.name])
            count += 1
            
    if projection == None:
        fig.text(0.6, 0.01, 'Phase (deg)', ha='center',
                            va='center', fontsize=18)
        fig.text(0.03, 0.5, 'Spike count', ha='center',
                            va='center', rotation='vertical', fontsize=18)
    fig. tight_layout(pad=1.0)
    

    
def find_phase_of_spikes_bet_2_peaks(left_peak_time, right_peak_time, all_spikes, total_phase = 360):
    corr_spike_times = all_spikes[ 
                                                    np.logical_and(all_spikes >= left_peak_time, 
                                                                   all_spikes < right_peak_time) 
                                                   ]
    # print("left", left_peak_time * 0.25, "right", right_peak_time * 0.25)
    cycle_length = right_peak_time - left_peak_time
    n_spikes = len(corr_spike_times)
    phase = ( corr_spike_times - left_peak_time ) / cycle_length * total_phase
    # print(corr_spike_times * 0.25, n_spikes)
    # fig,ax = plt.subplots()
    # ax.hist(corr_spike_times, bins = 20)
    # ax.hist(phase, bins = np.linspace(0, 360, endpoint=True, num = 50))

    return phase, n_spikes

def create_FR_ext_filename_dict(nuclei_dict, path, dt):
    filename_dict = {}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            filename_dict[nucleus.name] = os.path.join(path, 'FR_ext_' + nucleus.name + 
                                                       '_noise_var_' + str( round(
                                                                           nucleus.noise_variance , 2)
                                                                           ).replace('.', '-') +
                                                       '_dt_' + str(0.1).replace('.', '-') +
                                                       '_A_' + str(nucleus.basal_firing).replace('.', '-') +
                                                       '.pkl')
    return filename_dict

def set_connec_ext_inp(path, A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, c='grey', scale_g_with_N=True,
                        all_FR_list=np.linspace(0.05, 0.07, 100), n_FR=50, if_plot=False, end_of_nonlinearity=None, left_pad=0.005,
                        right_pad=0.005, maxfev=5000, ax=None, set_FR_range_from_theory=True, method = 'single_neuron', FR_ext_all_nuclei_saved = None,
                        use_saved_FR_ext = False, save_FR_ext = True, normalize_G_by_N = False, state = 'rest'):
    
    '''find number of connections and build J matrix, set ext inputs as well
        Note: end_of_nonlinearity has been modified to be passed as dict (incompatible with single neuron setting)'''

    K = calculate_number_of_connections(N, N_real, K_real)
    receiving_class_dict = create_receiving_class_dict(
        receiving_pop_list, nuclei_dict)
    FR_ext_all_nuclei = {}
    FR_ext_filename_dict = create_FR_ext_filename_dict(nuclei_dict, path, dt)

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:

            nucleus.set_connections(K, N)
            
            if nucleus.neuronal_model == 'rate' and scale_g_with_N:
                
                nucleus.scale_synaptic_weight()
                nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list,
                                  dt)
            else:
                
                if nucleus. der_ext_I_from_curve:
                
                    if method == 'collective' and not use_saved_FR_ext:
                        print("external input is being set collectively for {0} at {1}...".format(nucleus.name, state))
                        FR_ext = nucleus.set_ext_inp_const_plus_noise_collective(all_FR_list[nucleus.name], t_list, dt, receiving_class_dict,
                                                                        if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity[nucleus.name][state],
                                                                        maxfev = maxfev, c=c, n_FR=n_FR)
                        FR_ext_all_nuclei[nucleus.name] = FR_ext
                        if save_FR_ext:
                            pickle_obj(nucleus.FR_ext, FR_ext_filename_dict[nucleus.name])
                    elif method == 'single_neuron':
                        if nucleus.basal_firing > end_of_nonlinearity:
                            nucleus.estimate_needed_external_input_high_act(all_FR_list[nucleus.name], dt, t_list, receiving_class_dict, if_plot=if_plot,
                                                                            n_FR=n_FR, ax=ax, c=c, set_FR_range_from_theory=set_FR_range_from_theory)
                            
                        else:
                            nucleus.estimate_needed_external_input(all_FR_list[nucleus.name], dt, t_list, receiving_class_dict, if_plot=if_plot, 
                                                                   end_of_nonlinearity=end_of_nonlinearity, maxfev=maxfev,
                                                                   n_FR=n_FR, left_pad=left_pad, right_pad=right_pad, ax=ax, c=c)
                    elif use_saved_FR_ext:
                        # print(os.path.split(FR_ext_filename_dict[nucleus.name])[1] )
                        nucleus.FR_ext = load_pickle(FR_ext_filename_dict[nucleus.name])
                    
                if normalize_G_by_N:
                    nucleus.normalize_synaptic_weight_by_N()
                nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list, dt, 
                                      end_of_nonlinearity = end_of_nonlinearity[nucleus.name][state])
                # print(nucleus.name, 
                #       {key: np.round(value,2) for key,value in nucleus.synaptic_weight.items()})

    return receiving_class_dict, nuclei_dict


def set_init_all_nuclei(nuclei_dict, list_of_nuc_with_trans_inp=None, filepaths=None):
    if list_of_nuc_with_trans_inp != None:
        filtered_nuclei_dict = {key: value for key, value in nuclei_dict.items(
        ) if key in list_of_nuc_with_trans_inp}

    else:
        filtered_nuclei_dict = nuclei_dict
    for nuclei_list in filtered_nuclei_dict.values():
        for nucleus in nuclei_list:
            if filepaths == None:
                filepath = os.path.join(nucleus.path, nucleus.name + '_N_' +
                                        str(nucleus.n) + '_T_' + str(nucleus.t_sim) + '.pkl')
            else:
                filepath = os.path.join(nucleus.path, filepaths[nucleus.name])
            nucleus.set_init_from_pickle(filepath)


def reinitialize_nuclei_SNN(nuclei_dict, tau, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, t_mvt, t_list, dt, state = 'rest', poisson_prop = None,
                            mem_pot_init_method=None, set_noise=True, end_of_nonlinearity=25, reset_init_dist = False, t_sim = None, normalize_G_by_N = False):
    

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(mem_pot_init_method=mem_pot_init_method)
            nucleus.set_synaptic_weights(G)
            nucleus.normalize_synaptic_weight()
            if normalize_G_by_N:
                nucleus.normalize_synaptic_weight_by_N()
            
            if set_noise:
                nucleus.set_noise_param(noise_variance, noise_amplitude)
            if reset_init_dist:
                nucleus.set_init_distribution( tau, poisson_prop, dt, t_sim,  plot_initial_V_m_dist = False)
            nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list,
                                  dt, end_of_nonlinearity=end_of_nonlinearity)
    return nuclei_dict

def set_ext_input_all_nuclei(nuclei_dict, A, A_mvt, D_mvt, t_mvt, t_list,
                                  dt, end_of_nonlinearity):
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list,
                                  dt, end_of_nonlinearity=end_of_nonlinearity)
def bandpower(f, pxx, fmin, fmax):
    ''' return the average power at the given range of frequency'''
    ind_min = np.argmin(f[f > fmin])
    ind_max = np.argmax(f[f < fmax])
    return np.trapz(pxx[ind_min: ind_max], f[ind_min: ind_max]) / (f[ind_max] - f[ind_min])

def bandpower_2d_pxx(f, pxx, fmin, fmax):
    ''' return the average power at the given range of frequency'''
    ind_min = np.argmin(f[f > fmin])
    ind_max = np.argmax(f[f < fmax])
    return np.trapz(pxx[:, ind_min: ind_max], f[ind_min: ind_max]) / (f[ind_max] - f[ind_min])

def beta_bandpower(f, pxx, fmin=13, fmax=30):
    return bandpower(f, pxx, fmin, fmax)


def find_sending_pop_dict(receiving_pop_list):

    sending_pop_list = {k: [] for k in receiving_pop_list.keys()}
    for k, v_list in receiving_pop_list.items():
        for v in v_list:
            sending_pop_list[v].append(k)
    return sending_pop_list

def create_a_list_of_entrianed_and_not(entrained_ind, n, n_entrained = 2, n_not_entrained = 2):
    not_entrained_ind = get_complement_ind(entrained_ind, n)
    neuron_list = np.concatenate( ( np.random.choice(entrained_ind, n_entrained, replace  = False), 
                                np.random.choice(not_entrained_ind, n_not_entrained, replace  = False)) , 
                             axis = 0)
    return neuron_list

def get_max_len_dict(dictionary):
    ''' return maximum length between items of a dictionary'''
    return max(len(v) for k, v in dictionary.items())

# def find_peaks_of_pop_act_all_nuclei(data, i, j, nuclei_dict, dt, low_f, high_f, filter_order = 6, height = 1, start = 0):
    
#     for nuclei_list in nuclei_dict.values():
#         for nucleus in nuclei_list:
#             peaks = nucleus.find_peaks_of_pop_act(dt, low_f, high_f, filter_order = filter_order, 
#                                                                                height = height, start = start)
#             data[(nucleus.name, 'ind_peaks')][i,j, :len(peaks)]  = peaks
#             data[(nucleus.name, 'n_peaks')][i,j] = len(peaks)
#     return data

def find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 90,
                                       height = 0, ref_nuc_name = 'self', start = 0, total_phase = 720,
                                       only_entrained_neurons = False, min_f_sig_thres = 0,window_mov_avg = 10, max_f = 250,
                                       n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2,
                                       min_f_AUC_thres = 7,  PSD_AUC_thresh = 10**-5, filter_based_on_AUC_of_PSD = False, troughs = False):
    if ref_nuc_name != 'self':
        ref_peaks = nuclei_dict[ref_nuc_name][0].find_peaks_of_pop_act(dt, low_f, high_f, filter_order = filter_order, 
                                                                              height = height, start = start)
    else:
        ref_peaks = []
        
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            if only_entrained_neurons:
                neurons_ind = significance_of_oscil_all_neurons( nucleus, dt, max_f = max_f, 
                                                          window_mov_avg = window_mov_avg, n_sd_thresh = n_sd_thresh, 
                                                           min_f_sig_thres = min_f_sig_thres, n_window_welch = n_window_welch, 
                                                           n_pts_above_thresh = n_pts_above_thresh,
                                                           min_f_AUC_thres = min_f_AUC_thres ,  PSD_AUC_thresh = PSD_AUC_thresh , 
                                                           filter_based_on_AUC_of_PSD = filter_based_on_AUC_of_PSD)
            else:
                neurons_ind = 'all'
            nucleus.find_phase_hist_of_spikes(dt, low_f, high_f, filter_order = filter_order, n_bins = n_bins,
                                              start = start, height = height, ref_peaks = ref_peaks, troughs = troughs,
                                              total_phase = total_phase, ref_nuc_name = ref_nuc_name, neurons_ind = neurons_ind)

def find_phase_sine_fit(x, y):
    ''' fit sine function derive phase'''
    A, w, p, c, f, fitfunc = fit_sine(x, y)
    if A > 0:
        phase_sine =  np.pi/ 2/ w + p 
    else:
        phase_sine = - np.pi/ 2/ w + p 
    phase_sine = equi_phase_in_0_360_range(phase_sine)
    return phase_sine, fitfunc, w

# def shift_small_phases_to_next_peak(phase, w, nuc_name, ref_nuc_name):
#     if nuc_name  != ref_nuc_name:
#         if phase < 30:
#             phase += 2 * np.pi / w
#     return phase

# def shift_large_phases_to_prev_peak(phase, w, nuc_name, ref_nuc_name):
#     ''' If outliers (minority) are on the second peak, shift them back'''
#     if nuc_name  != ref_nuc_name:
#         if phase > 340 :
#             print(nuc_name, 'shifting backward, phase before = ', phase)
#             phase -= 2 * np.pi / w
#             print('phase after = ', phase)
#     return phase

def shift_small_phases_to_next_peak(phases, ws, nuc_name, ref_nuc_name):
    mean_phases = np.average(phases)
    if nuc_name  != ref_nuc_name:
        if mean_phases > 100:
            ind = np.where(phases < 30)
            print(nuc_name, 'shifting forward, phase before = ', phases[ind])
            phases[ind] += 2 * np.pi / ws[ind]
            print('phase after = ', phases[ind])
    return phases

def shift_large_phases_to_prev_peak(phases, ws, nuc_name, ref_nuc_name):
    ''' If outliers (minority) are on the second peak, shift them back'''
    mean_phases = np.average(phases)
    if nuc_name  != ref_nuc_name:
        if mean_phases < 180:
            ind = np.where(phases > 320)
            print(nuc_name, 'shifting backward, phase before = ', phases[ind])
            phases[ind] -= 2 * np.pi / ws[ind]
            print('phase after = ', phases[ind])
    return phases

def shift_second_peak_phases_to_prev_peak(phase, w, nuc_name, ref_nuc_name):
    ''' If the second peak is detected shift to the previous'''
    if nuc_name  != ref_nuc_name:
        if phase > 360 :
            print(nuc_name, 'shifting to first peak, phase before = ', phase)
            phase -= 2 * np.pi / w
            print('phase after = ', phase)
    return phase

def find_phase_from_max(x, y):
    ''' find the phase of where the function maximises'''
    phase_max = x[ np.argmax( y ) ]
    phase_max = equi_phase_in_0_360_range(phase_max)
    
    return phase_max
            
def find_phase_from_sine_and_max(x,y, nuc_name, ref_nuc_name, shift_phase = None):
    ''' In case sine fit and maximum are close (90 deg) go with maximum, 
        Otherwise sine is to be trusted'''
        
    phase_sine,fitfunc, w = find_phase_sine_fit(x, y)
    phase_max = find_phase_from_max(x, y)
    
    phase = decide_bet_max_or_sine(phase_max, phase_sine, nuc_name, ref_nuc_name)
        
    phase = shift_second_peak_phases_to_prev_peak(phase, w, nuc_name, ref_nuc_name)
    
    # if shift_phase == 'backward':
    #     phase = shift_large_phases_to_prev_peak(phase, w, nuc_name, ref_nuc_name)
        
    # elif shift_phase == 'forward':
    #     phase = shift_small_phases_to_next_peak(phase, w, nuc_name, ref_nuc_name)
        
    return phase, fitfunc, w

def correct_phases(phases, ws, nuc_name, ref_nuc_name, shift_phase = None):
    
    if shift_phase == 'backward':
        phases = shift_large_phases_to_prev_peak(phases, ws, nuc_name, ref_nuc_name)
        
    elif shift_phase == 'forward':
        phases = shift_small_phases_to_next_peak(phases, ws, nuc_name, ref_nuc_name)
        
    elif shift_phase == 'both':
        phases = shift_small_phases_to_next_peak(phases, ws, nuc_name, ref_nuc_name)
        phases = shift_large_phases_to_prev_peak(phases, ws, nuc_name, ref_nuc_name)

    return phases

def decide_bet_max_or_sine(phase_max, phase_sine, nuc_name, ref_nuc_name):
    if abs( phase_sine - phase_max) < 90: # two methods are consistent, max is more accurate visually
        phase = phase_max
    else:
        if phase_sine < 30 and nuc_name != ref_nuc_name: #### if one phase is detected in the beginnig and they are >90 apart --> choose the other one (apprent peak in phase)
            phase = phase_max
            
        elif phase_max < 30 and nuc_name != ref_nuc_name:
            phase = phase_sine
        else:
            phase = phase_sine
    return phase

def mean_std_multiple_sets(means, stds, nums):
    mean_all = np.sum( means * nums ) / np.sum(nums)
    std_all = np.sqrt(np.sum( (nums - 1) * 
                      np.power(stds, 2) + 
                      nums  * 
                      np.power( means - mean_all, 2)) / \
                (np.sum(nums) - 1) )
    return mean_all, std_all

def equi_phase_in_0_360_range(phase):
    ''' bring phase into [0,360)'''
    if phase < 0 : phase += 360
    if phase >= 360 : phase -=360
    return phase

def get_centers_from_edges(edges):
    centers = edges + (edges[1] - edges[0]) / 2
    return centers

def save_phases_into_dataframe(nuclei_dict, data, i,j, ref_nuc_name, shift_phase = None):

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            frq , edges = nucleus.spike_rel_phase_hist[ref_nuc_name]
            data[(nucleus.name, 'rel_phase_hist')][i,j,0,:], data[(nucleus.name, 'rel_phase_hist')][i,j,1,:] = nucleus.spike_rel_phase_hist[ref_nuc_name]
            centers = get_centers_from_edges(edges)
            data[(nucleus.name, 'rel_phase')][i,j],_,_ = find_phase_from_sine_and_max(centers, frq, nucleus.name, ref_nuc_name, shift_phase = shift_phase)
    return data
 
def save_phases_into_dataframe_2d(nuclei_dict, data, i, m, j, ref_nuc_name, shift_phase = None):

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            frq , edges = nucleus.spike_rel_phase_hist[ref_nuc_name]
            data[(nucleus.name, 'rel_phase_hist')][i,m,j,0,:], data[(nucleus.name, 'rel_phase_hist')][i,m,j,1,:] = nucleus.spike_rel_phase_hist[ref_nuc_name]
            centers = get_centers_from_edges(edges)
            data[(nucleus.name, 'rel_phase')][i,m,j],_,_ = find_phase_from_sine_and_max(centers, frq, nucleus.name, ref_nuc_name, shift_phase = shift_phase)
    return data
def save_pxx_into_dataframe(f, pxx, name, data, i, m):
            
    data[(name, 'f')][i, m, :], data[(name, 'pxx')][i, m, :] = f, pxx

    return data

def save_pxx_into_dataframe_2d(f, pxx, name , data, i, m, j):
    

    data[(name, 'f')][i, m, j, :], data[(name, 'pxx')][i, m, j, :] = f, pxx
            
    return data

def save_pop_act_into_dataframe(nuclei_dict, start,data, i,m):

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            data[(nucleus.name, 'pop_act')][i, m, :] = nucleus.pop_act[start:]
    return data

def save_pop_act_into_dataframe_2d(nuclei_dict, start,data, i,m, j):

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            data[(nucleus.name, 'pop_act')][i, m, j, :] = nucleus.pop_act[start:]
            
    return data


def phase_summary(filename, name_list, color_dict, n_g_list, ref_nuc_name = 'Proto', total_phase = 720, 
                  n = 1000, set_ylim = True, shift_phase = None, y_max_series = None, xlabel_fontsize = 8,
                  ylabel_fontsize = 8, phase_txt_fontsize = 8, tick_label_fontsize = 8, 
                  ylabel = r'$ Mean \; neuron \; spike \; count/(4^{\circ} \; degrees)$',
                  xlabel = 'phase (deg)'):
    
    fig = plt.figure(figsize = (5, 15))
    outer = gridspec.GridSpec(len(n_g_list), 1, wspace=0.2, hspace=0.2)
    
    data = load_pickle(filename)
    
    n_run = data[(name_list[0], 'rel_phase')].shape[1] 
    
    for i, n_g in enumerate(n_g_list):
        inner = gridspec.GridSpecFromSubplotSpec(len(name_list), 1,
                subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        
        for j, name in enumerate(name_list):
            ax = plt.Subplot(fig, inner[j])
            
            edges = data[(name,'rel_phase_hist')][0,0,1,:]
            centers = get_centers_from_edges(edges)
            
            phase_hist_mean, phase_hist_std = get_mean_and_std_of_phase(data, n_g, name, n)
            plot_mean_phase_plus_std(phase_hist_mean, phase_hist_std, name, n_g, ax, color_dict, centers, lw = 0.5)
            phases = calculate_phase_all_runs(n_run, data, n_g, run , centers, name, ref_nuc_name, shift_phase = shift_phase)
            
            box_width = y_max_series[name] / 5
            box_y = y_max_series[name] / 3
            
            boxplot_phases(ax, phase_hist_mean, phase_hist_std, color_dict, phases, name, box_width, 
                           box_y, y_max_series[name] , phase_txt_fontsize = phase_txt_fontsize)
            
            ax.axvline(total_phase/2, ls = '--', c = 'k', dashes=(5, 10), lw = 1)
            ax.annotate(name, xy=(0.8,0.3),xycoords='axes fraction', color = color_dict[name],
                        fontsize= phase_txt_fontsize )
            fig.add_subplot(ax)
            ax.set_xticks([0,180,360,540,720])
            ax.yaxis.set_major_locator(MaxNLocator(2)) 
            ax.tick_params(axis='both', labelsize=tick_label_fontsize)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # ax.set_ylim(np.min(phase_hist_mean - phase_hist_std), 
            #             np.max(phase_hist_mean + phase_hist_std))
            y_max = ( np.min(phase_hist_mean - phase_hist_std) * 0.5 + 
                     np.max(phase_hist_mean + phase_hist_std) )
            if set_ylim:
                ax.set_ylim(0, y_max_series[name])
                        
            rm_ax_unnecessary_labels_in_subplots(j, len(name_list), ax)
            remove_frame(ax)
    fig.text(0.5, 0.02, xlabel, ha='center',
                 va='center', fontsize=xlabel_fontsize)
    fig.text(-0.1, 0.5, ylabel, ha='center', va='center',
                 rotation='vertical', fontsize=ylabel_fontsize)
    return fig

def save_pdf_png(fig, figname, size = (8,6)):
    fig.set_size_inches(size, forward=False)
    fig.savefig(figname + '.png', dpi = 500, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    fig.savefig(figname + '.pdf', dpi = 500, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
def normalize_PSDs(data, n_run, name, n_g):
    for run in range(n_run):
        AUC = np.trapz( data[(name,'pxx')][n_g, run,:])
        data[(name,'pxx')][n_g, run,:] = data[(name,'pxx')][n_g, run,:] / AUC * 100
        
    return data

def PSD_summary(filename, name_list, color_dict, n_g_list, xlim = None, inset_props = [0.65, 0.6, 0.3, 0.3],
                inset_yaxis_loc = 'right', inset_name = 'D2', err_plot = 'fill_between', legend_loc = 'upper right',
                plot_lines = False, tick_label_fontsize = 15, legend_font_size = 10, 
                normalize_PSD = True, include_AUC_ratio = False, x_y_label_size = 10,
                ylabel_norm = 'Norm. Power ' + r'$(\times 10^{-2})$',
                ylabel_PSD = 'PSD',
                xlabel = 'Frequency (Hz)', peak_f_sd = False):
    
    fig = plt.figure()    
    data = load_pickle(filename)
    
    n_run = data[(name_list[0], 'pxx')].shape[1] 
    for i, n_g in enumerate(n_g_list):
        
        max_pxx = 0
        ax = fig.add_subplot(len(n_g_list), 1, i + 1)
        
        for j, name in enumerate(name_list):
            
            if normalize_PSD:
                data = normalize_PSDs(data, n_run, name, i)
                
            f = data[(name,'f')][0,0].reshape(-1,)
            pxx_mean = np.average( data[(name,'pxx')][i,:,:], axis = 0)
            print(data[(name,'base_freq')][i,:])
            mean_peak_f =np.average( data[(name,'base_freq')][i,:], axis = 0)
            sd_peak_f = np.std( data[(name,'base_freq')][i,:], axis = 0)
            print('mean peak freq ', name, mean_peak_f)
            pxx_std = np.std( data[(name,'pxx')][i,:,:], axis = 0)
            all_std = np.std( data[(name,'pxx')][i,:,:].flatten() )
            significance = check_significance_of_PSD_peak(f, pxx_mean,  n_std_thresh = 2, min_f = 0, 
                                                                     max_f = 250, n_pts_above_thresh = 3, 
                                                                     ax = None, legend = 'PSD', c = 'k', if_plot = False, name = name)

            
            
                   
            if err_plot == 'fill_between':
                leg_label = name + ' f= '+\
                        r'$' + "{:.2f}". format(mean_peak_f)+ '$'
                if peak_f_sd:
                    leg_label += '$' +' \pm ' + "{:.2f}". format(sd_peak_f) + '$' + ' Hz'
                ax.plot(f, pxx_mean, color = color_dict[name], lw = .5, label = leg_label)
                ax.fill_between(f, pxx_mean - pxx_std ,
                                pxx_mean + pxx_std, color = color_dict[name], alpha = 0.2)
            if err_plot == 'errorbar':
                ax.errorbar(f, pxx_mean, yerr = pxx_std, color = color_dict[name], label = name, lw = .5)
                
            if plot_lines:
                ax.axhline(2 * all_std, 0, 200, ls = '--', color = color_dict[name], lw = 0.5, dashes=(10, 10))
                # ax.axvline(f[np.argmax(pxx_mean)], linestyle = '--', color = color_dict[name])
            
            if name == inset_name:
                plot_D2_as_inset(f, pxx_mean, pxx_std, color_dict,ax, name = 'D2', 
                                 inset_props = inset_props, inset_yaxis_loc = inset_yaxis_loc)
                
            max_pxx = max(np.max(pxx_mean + pxx_std), max_pxx)
            if include_AUC_ratio:
                ax.annotate(r'$\frac{AUC_{> 2SD}}{AUC} = ' + "{:.3f}".format(AUC_ratio) + '$', xy=(0.2 + 0.2*j,0.8), xycoords='axes fraction', color = color_dict[name],
                            fontsize=10)

        ax.set_ylim(-0, max_pxx)
        ax.yaxis.set_major_locator(MaxNLocator(4)) 
        rm_ax_unnecessary_labels_in_subplots(i, len(n_g_list), ax)
        ax.tick_params(axis='both', labelsize=tick_label_fontsize, pad=1)
        remove_frame(ax)
        if xlim == None:
            ax.set_xlim(8,60)
        else:
            ax.set_xlim(xlim)
            
        ax.legend(fontsize = legend_font_size, loc = legend_loc,  framealpha = 0.1, frameon = False)

    fig.text(0.5, -0.0005, xlabel, ha='center',
                 va='center', fontsize=x_y_label_size)
    if normalize_PSD:
        fig.text(0.001, 0.5, ylabel_norm, ha='center', va='center',
                     rotation='vertical', fontsize=x_y_label_size)
    else:
        fig.text(0.03, 0.5, ylabel_PSD, ha='center', va='center',
                     rotation='vertical', fontsize=x_y_label_size)
    return fig

def plot_D2_as_inset(f_mean, pxx_mean, pxx_std, color_dict,ax, name = 'D2', 
                     inset_props = [0.65, 0.6, 0.3, 0.3], inset_yaxis_loc = 'right'):
    
        axins = ax.inset_axes(
                   inset_props )
        axins.plot(f_mean, pxx_mean, color = color_dict[name], label = name)
        axins.fill_between(f_mean, pxx_mean - pxx_std ,
                            pxx_mean + pxx_std, color = color_dict[name], alpha = 0.2)
        axins.set_xlim(0, 70)
        axins.xaxis.set_major_locator(MaxNLocator(3)) 
        axins.set_yticklabels(labels = axins.get_yticks().tolist(), fontsize = 12)
        axins.set_xticklabels(labels = axins.get_xticks().tolist(), fontsize = 12)
        axins.yaxis.set_label_position(inset_yaxis_loc)

def get_mean_and_std_of_phase(data, n_g, name, n):
    
    phase_frq_rel_mean = np.average(data[(name,'rel_phase_hist')][n_g,:,0,:] / n, axis = 0)
    phase_frq_rel_std = np.std(data[(name,'rel_phase_hist')][n_g,:,0,:] / n, axis = 0)
    return phase_frq_rel_mean, phase_frq_rel_std

def plot_mean_phase_plus_std(phase_frq_rel_mean, phase_frq_rel_std, name, n_g, ax, color_dict, centers, lw = 1):

    ax.plot(centers, phase_frq_rel_mean, color = color_dict[name], lw = lw)
    ax.fill_between(centers, phase_frq_rel_mean - phase_frq_rel_std, 
                    phase_frq_rel_mean + phase_frq_rel_std, alpha=0.2, color = color_dict[name])
    
def calculate_phase_all_runs(n_run, data, n_g, run , centers, name, ref_nuc_name, shift_phase = None):
    
    phases= np.zeros(n_run)
    ws= np.zeros(n_run)

    for run in range(n_run):

        y = data[(name,'rel_phase_hist')][n_g,run,0,:]
        phases[run], fitfunc, ws[run] = find_phase_from_sine_and_max(centers, y, name, ref_nuc_name, shift_phase = shift_phase)
    phases = correct_phases(phases, ws, name, ref_nuc_name, shift_phase= shift_phase)
    return phases

def boxplot_phases(ax, phase_frq_rel_mean, phase_frq_rel_std, color_dict, phases,name, box_width, box_y, max_y, 
                   phase_txt_fontsize = 10):
    
    highest_point = np.max(phase_frq_rel_mean + phase_frq_rel_std)
    lowest_point = np.min(phase_frq_rel_mean - phase_frq_rel_std)
    middle = ( highest_point + lowest_point ) / 2
    width = ( highest_point - lowest_point)
    
    width = max_y
    bp = ax.boxplot(phases, positions = [box_y], vert=False,
                sym = '', widths = box_width, whis =  (0, 100))
    if np.average(phases) < 100:
        ax.annotate(r'$' + "{:.1f}". format(np.average(phases)) + ' ^{\circ} \pm ' + "{:.1f}". format(np.std(phases)) + '^{\circ}$', 
                xy=(np.average(phases), box_y -box_width), color = color_dict[name], fontsize = phase_txt_fontsize)
    else:
        ax.annotate(r'$' + "{:.1f}". format(np.average(phases)) + ' ^{\circ} \pm ' + "{:.1f}". format(np.std(phases)) + '^{\circ}$', 
                    xy=(np.average(phases) - 100, box_y - box_width), color = color_dict[name], fontsize = phase_txt_fontsize)

    bp = set_boxplot_prop(bp, [color_dict[name]])
        
def set_boxplot_prop(bp, color_list):
    # for patch, color in zip(bp['boxes'], colors): 
    #     patch.set_facecolor(color) 
       
    # changing color and linewidth of 
    # whiskers 
    for whisker,color in zip(bp['whiskers'], color_list): 
        whisker.set(color =color, 
                    linewidth = .5)               
    # changing color and linewidth of 
    # caps 
    for cap,color in zip(bp['caps'], color_list): 
        cap.set(color = color, 
                linewidth = .5) 
       
    # changing color and linewidth of 
    # medians 
    for median,color in zip(bp['medians'], color_list): 
        median.set(color =color, 
                   linewidth = 0.5) 
    return bp

def print_G_items(G_dict):
    
    print('G = \n')
    for k, values in G_dict.items():
    
        print(k, np.round( values, 2) )

def synaptic_weight_exploration_SNN(path, tau, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
    peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
    lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
    plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
    include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
    reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, N_real = None, N = None,
    receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
    spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
    phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, ref_nuc_name = 'Proto', plot_phase = False,
    total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = True,
    plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, save_pop_act = False):

    if set_seed:
        np.random.seed(1956)

    
    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = get_max_len_dict(G_dict)
    # print("n_inter = ", n_iter)
    data = {}
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0]  # get only on class from each population
        data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
        # data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
        # data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_run), dtype = bool) # stores the value of the PSD at the peak and the mean of the PSD elsewhere
        if save_pop_act :
            data[(nucleus.name, 'pop_act')] = np.zeros((n_iter, n_run, duration_base[1] - duration_base[0]))
        if find_phase:
            data[(nucleus.name, 'rel_phase_hist')] = np.zeros((n_iter, n_run, 2, n_phase_bins-1))
            data[(nucleus.name, 'rel_phase')] = np.zeros((n_iter, n_run))

        if divide_beta_band_in_power:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run, 2))
        else:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'f')] = np.zeros((n_iter, n_run, len_f_pxx))
        data[(nucleus.name, 'pxx')] = np.zeros((n_iter, n_run, len_f_pxx))

    data['g'] = G_dict
    count = 0
    G = dict.fromkeys(G_dict.keys(), None)

    if n_run > 1:  # don't plot all the runs
        plot_spectrum = False
        plot_firing = False
        plot_phase = False
        plot_raster = False
        
    if plot_firing:
        fig = plt.figure()

    if plot_spectrum:
        fig_spec = plt.figure()

    if plot_raster:
        fig_raster = plt.figure()
        outer = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
        
    if plot_phase:
        fig_phase = plt.figure()
        outer_phase = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
        
    for i in range(n_iter):
        
        start = timeit.default_timer()
        
        for k, values in G_dict.items():
            
            G[k] = values[i]
            print(k, np.round( values[i], 2) )

        if plot_spectrum:
            ax_spec = fig_spec.add_subplot(n_iter, 1, count+1)

        else: ax_spec = None

        title = G_element_as_txt(G_dict, i, display=display, decimal=decimal) 

        for j in range(n_run):
            
            print(' {} from {} runs'.format(j + 1 , n_run))
            nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, tau, G, noise_amplitude, noise_variance, A,
                                                  A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                                  reset_init_dist= reset_init_dist, poisson_prop = poisson_prop, 
                                                  normalize_G_by_N= True)  
            if reset_init_dist:
                receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = if_plot, 
                                                          end_of_nonlinearity = end_of_nonlinearity, 
                                                          set_FR_range_from_theory = False, method = 'collective', 
                                                          use_saved_FR_ext= use_saved_FR_ext,
                                                          normalize_G_by_N= False, save_FR_ext=False,
                                                          state = state)


            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            if save_pop_act:
                data = save_pop_act_into_dataframe(nuclei_dict, duration_base[0],data, i,j)
            if plot_raster:
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer[i], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)

            if find_phase:

                find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins,
                                              height = phase_thresh_h, ref_nuc_name = ref_nuc_name, start = start_phase, total_phase = 720, troughs = troughs)
                # find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins, troughs = troughs,
                #                               height = phase_thresh_h, ref_nuc_name = 'self', start = start_phase, total_phase = 360)
                data = save_phases_into_dataframe(nuclei_dict, data, i,j, ref_nuc_name)
                
            if plot_phase:
                fig_phase = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                                          density = False, ref_nuc_name = ref_nuc_name, total_phase = total_phase, 
                                                          projection = phase_projection, outer=outer_phase[i], fig= fig_phase,  title='', 
                                                          tick_label_fontsize=18, labelsize=15, title_fontsize=15, lw=1, linelengths=1, 
                                                          include_title=True, ax_label=True, nuc_order = nuc_order)
                
            data, nuclei_dict = find_freq_SNN(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
                                check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=plot_spectrum, ax=ax_spec,
                                c_spec=color_dict, spec_figsize=spec_figsize, n_windows=n_windows, fft_method=fft_method, find_beta_band_power=find_beta_band_power,
                                include_beta_band_in_legend=include_beta_band_in_legend, divide_beta_band_in_power = divide_beta_band_in_power, 
                                half_peak_range = 5, cut_off_freq = 100, check_peak_significance=check_peak_significance, 
                                save_pxx = save_pxx, len_f_pxx = len_f_pxx, normalize_spec=normalize_spec, 
                                plot_sig_thresh = plot_sig_thresh, plot_peak_sig = plot_peak_sig, min_f = min_f, 
                                max_f = max_f, n_std_thresh= n_std_thresh, AUC_ratio_thresh = AUC_ratio_thresh)

        if plot_spectrum:
            if fft_method == 'rfft':
                x_l = 10**9

            else:
                x_l = 5
                # ax_spec.axhline(x_l, ls='--', c='grey')

            # ax_spec.set_title(title, fontsize = 18)
            ax_spec.legend(fontsize=11, loc='upper center',
                           framealpha=0.1, frameon=False)
            ax_spec.set_xlim(spec_lim[0], spec_lim[1])
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax_spec)

        if plot_firing:
            ax = fig.add_subplot(n_iter, 1, count+1)
            plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title, include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim, include_FR = include_FR)
            ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
            ax.set_ylim(firing_ylim)
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)

        count += 1
        stop = timeit.default_timer()
        print(count, "from", int(n_iter), 'gs. t=', round(stop - start, 2))

    figs = []
    if plot_firing:
        fig.set_size_inches((15, 15), forward=False)
        fig.text(0.5, 0.05, 'time (ms)', ha='center', fontsize=18)
        fig.text(0.03, 0.5, 'firing rate (spk/s)',
                 va='center', rotation='vertical', fontsize=18)
        figs.append(fig)
    if plot_spectrum:
        fig_spec.set_size_inches((11, 15), forward=False)
        fig_spec.text(0.5, 0.05, 'frequency (Hz)', ha='center', fontsize=18)
        fig_spec.text(0.02, 0.5, 'fft Power', va='center',
                      rotation='vertical', fontsize=18)
        figs.append(fig_spec)

    if plot_raster:
        fig.set_size_inches((11, 15), forward=False)
        fig_raster.text(0.5, 0.05, 'time (ms)', ha='center',
                        va='center', fontsize=18)
        fig_raster.text(0.03, 0.5, 'neuron', ha='center',
                        va='center', rotation='vertical', fontsize=18)
        figs.append(fig_raster)
        fig_raster.show()
    if plot_phase:
        fig_phase.set_size_inches((11, 15), forward=False)
        # fig_phase.text(0.5, 0.05, 'Phase (deg)', ha='center', fontsize=18)
        # fig_phase.text(0.02, 0.5, 'Spike count', va='center',
        #               rotation='vertical', fontsize=18)
        figs.append(fig_phase)
    if save_pkl:
        pickle_obj(data, filepath)
    return figs, title, data

def reset_tau_all_nuclei(tau_dict, nuclei_dict, i, dt):
    
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            for k, v in tau_dict.items():
                if k[0] == nucleus.name:
                    nucleus.tau[k]['decay'] = np.array([tau_dict[k][ i ]  / dt ])
    return nuclei_dict

def reset_T_all_nuclei(T_dict, nuclei_dict, i, dt):
    
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            for k, v in T_dict.items():
                if k[0] == nucleus.name:
                    nucleus.transmission_delay[k] = T_dict[k][ i ] 
    return nuclei_dict

def synaptic_tau_exploration_SNN(path, tau, nuclei_dict, filepath, duration_base, G, tau_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
    peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
    lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
    plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
    include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
    reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, N_real = None, N = None,
    receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
    spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
    phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, ref_nuc_name = 'Proto', plot_phase = False,
    total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = True,
    plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, save_pop_act = False):

    if set_seed:
        np.random.seed(1956)

    
    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = get_max_len_dict(tau_dict)
    # print("n_inter = ", n_iter)
    data = {}
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0]  # get only on class from each population
        data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
        # data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
        # data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_run), dtype = bool) # stores the value of the PSD at the peak and the mean of the PSD elsewhere
        if save_pop_act :
            data[(nucleus.name, 'pop_act')] = np.zeros((n_iter, n_run, duration_base[1] - duration_base[0]))
        if find_phase:
            data[(nucleus.name, 'rel_phase_hist')] = np.zeros((n_iter, n_run, 2, n_phase_bins-1))
            data[(nucleus.name, 'rel_phase')] = np.zeros((n_iter, n_run))

        if divide_beta_band_in_power:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run, 2))
        else:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'f')] = np.zeros((n_iter, n_run, len_f_pxx))
        data[(nucleus.name, 'pxx')] = np.zeros((n_iter, n_run, len_f_pxx))

    data['tau'] = tau_dict
    count = 0

    if n_run > 1:  # don't plot all the runs
        plot_spectrum = False
        plot_firing = False
        plot_phase = False
        plot_raster = False
        
    if plot_firing:
        fig = plt.figure()

    if plot_spectrum:
        fig_spec = plt.figure()

    if plot_raster:
        fig_raster = plt.figure()
        outer = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
        
    if plot_phase:
        fig_phase = plt.figure()
        outer_phase = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
    G_copy = G.copy()    
    for i in range(n_iter):
        
        start = timeit.default_timer()
        nuclei_dict = reset_tau_all_nuclei(tau_dict, nuclei_dict, i, dt)
        
        if plot_spectrum:
            ax_spec = fig_spec.add_subplot(n_iter, 1, count+1)

        else: ax_spec = None

        # title = G_element_as_txt(tau_dict, i, display=display, decimal=decimal) 
        title = ''
        G.update({k: gg * (i  + 50) / 50  for k, gg in G_copy.items()})
        print_G_items(G)
        for j in range(n_run):
            
            print(' {} from {} runs'.format(j + 1 , n_run))
            nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, tau, G, noise_amplitude, noise_variance, A,
                                                  A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                                  reset_init_dist= reset_init_dist, poisson_prop = poisson_prop, 
                                                  normalize_G_by_N= True)  
            if reset_init_dist:
                receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = if_plot, 
                                                          end_of_nonlinearity = end_of_nonlinearity, 
                                                          set_FR_range_from_theory = False, method = 'collective', 
                                                          use_saved_FR_ext= use_saved_FR_ext,
                                                          normalize_G_by_N= False, save_FR_ext=False,
                                                          state = state)


            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            
            if save_pop_act:
                data = save_pop_act_into_dataframe(nuclei_dict, duration_base[0],data, i,j)
                
            if plot_raster:
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer[i], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)

            if find_phase:

                find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins,
                                              height = phase_thresh_h, ref_nuc_name = ref_nuc_name, start = start_phase, total_phase = 720, troughs = troughs)
                # find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins, troughs = troughs,
                #                               height = phase_thresh_h, ref_nuc_name = 'self', start = start_phase, total_phase = 360)
                save_phases_into_dataframe(nuclei_dict, data, i,j, ref_nuc_name)
                
            if plot_phase:
                fig_phase = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                                          density = False, ref_nuc_name = ref_nuc_name, total_phase = total_phase, 
                                                          projection = phase_projection, outer=outer_phase[i], fig= fig_phase,  title='', 
                                                          tick_label_fontsize=18, labelsize=15, title_fontsize=15, lw=1, linelengths=1, 
                                                          include_title=True, ax_label=True, nuc_order = nuc_order)
                
            data, nuclei_dict = find_freq_SNN(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
                                check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=plot_spectrum, ax=ax_spec,
                                c_spec=color_dict, spec_figsize=spec_figsize, n_windows=n_windows, fft_method=fft_method, find_beta_band_power=find_beta_band_power,
                                include_beta_band_in_legend=include_beta_band_in_legend, divide_beta_band_in_power = divide_beta_band_in_power, 
                                half_peak_range = 5,  cut_off_freq = 100, check_peak_significance=check_peak_significance, 
                                save_pxx = save_pxx, len_f_pxx = len_f_pxx, normalize_spec=normalize_spec, 
                                plot_sig_thresh = plot_sig_thresh, plot_peak_sig = plot_peak_sig, min_f = min_f, 
                                max_f = max_f, n_std_thresh= n_std_thresh, AUC_ratio_thresh = AUC_ratio_thresh)

        if plot_spectrum:
            if fft_method == 'rfft':
                x_l = 10**9

            else:
                x_l = 5
                ax_spec.axhline(x_l, ls='--', c='grey')

            # ax_spec.set_title(title, fontsize = 18)
            ax_spec.legend(fontsize=11, loc='upper center',
                           framealpha=0.1, frameon=False)
            ax_spec.set_xlim(spec_lim[0], spec_lim[1])
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax_spec)

        if plot_firing:
            ax = fig.add_subplot(n_iter, 1, count+1)
            plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title, include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim, include_FR = include_FR)
            ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
            ax.set_ylim(firing_ylim)
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)

        count += 1
        stop = timeit.default_timer()
        print(count, "from", int(n_iter), 'gs. t=', round(stop - start, 2))

    figs = []
    if plot_firing:
        fig.set_size_inches((15, 15), forward=False)
        fig.text(0.5, 0.05, 'time (ms)', ha='center', fontsize=18)
        fig.text(0.03, 0.5, 'firing rate (spk/s)',
                 va='center', rotation='vertical', fontsize=18)
        figs.append(fig)
    if plot_spectrum:
        fig_spec.set_size_inches((11, 15), forward=False)
        fig_spec.text(0.5, 0.05, 'frequency (Hz)', ha='center', fontsize=18)
        fig_spec.text(0.02, 0.5, 'fft Power', va='center',
                      rotation='vertical', fontsize=18)
        figs.append(fig_spec)

    if plot_raster:
        fig.set_size_inches((11, 15), forward=False)
        fig_raster.text(0.5, 0.05, 'time (ms)', ha='center',
                        va='center', fontsize=18)
        fig_raster.text(0.03, 0.5, 'neuron', ha='center',
                        va='center', rotation='vertical', fontsize=18)
        figs.append(fig_raster)
        fig_raster.show()
    if plot_phase:
        fig_phase.set_size_inches((11, 15), forward=False)
        # fig_phase.text(0.5, 0.05, 'Phase (deg)', ha='center', fontsize=18)
        # fig_phase.text(0.02, 0.5, 'Spike count', va='center',
        #               rotation='vertical', fontsize=18)
        figs.append(fig_phase)
    if save_pkl:
        pickle_obj(data, filepath)
    return figs, title, data

def synaptic_T_exploration_SNN(path, tau,  nuclei_dict, filepath, duration_base, G, T_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
    peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
    lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
    plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
    include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
    reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, N_real = None, N = None,
    receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
    spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
    phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, ref_nuc_name = 'Proto', plot_phase = False,
    total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = True,
    plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, save_pop_act = False):

    if set_seed:
        np.random.seed(1956)

    
    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = get_max_len_dict(T_dict)
    # print("n_inter = ", n_iter)
    data = {}
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0]  # get only on class from each population
        data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
        # data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
        # data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_run), dtype = bool) # stores the value of the PSD at the peak and the mean of the PSD elsewhere
        if save_pop_act :
            data[(nucleus.name, 'pop_act')] = np.zeros((n_iter, n_run, duration_base[1] - duration_base[0]))
        if find_phase:
            data[(nucleus.name, 'rel_phase_hist')] = np.zeros((n_iter, n_run, 2, n_phase_bins-1))
            data[(nucleus.name, 'rel_phase')] = np.zeros((n_iter, n_run))

        if divide_beta_band_in_power:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run, 2))
        else:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'f')] = np.zeros((n_iter, n_run, len_f_pxx))
        data[(nucleus.name, 'pxx')] = np.zeros((n_iter, n_run, len_f_pxx))

    data['T'] = T_dict
    count = 0

    if n_run > 1:  # don't plot all the runs
        plot_spectrum = False
        plot_firing = False
        plot_phase = False
        plot_raster = False
        
    if plot_firing:
        fig = plt.figure()

    if plot_spectrum:
        fig_spec = plt.figure()

    if plot_raster:
        fig_raster = plt.figure()
        outer = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
        
    if plot_phase:
        fig_phase = plt.figure()
        outer_phase = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
    G_copy = G.copy()  
    
    for i in range(n_iter):
        
        start = timeit.default_timer()
        nuclei_dict = reset_T_all_nuclei(T_dict, nuclei_dict, i, dt)
        
        if plot_spectrum:
            ax_spec = fig_spec.add_subplot(n_iter, 1, count+1)

        else: ax_spec = None

        # title = G_element_as_txt(tau_dict, i, display=display, decimal=decimal) 
        title = ''
        G.update({k: gg / ((i  + 10) / 10)  for k, gg in G_copy.items()})
        print_G_items(G)
        for j in range(n_run):
            
            print(' {} from {} runs'.format(j + 1 , n_run))
            nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, tau, G, noise_amplitude, noise_variance, A,
                                                  A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                                  reset_init_dist= reset_init_dist, poisson_prop = poisson_prop, 
                                                  normalize_G_by_N= True)  
            if reset_init_dist:
                receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = if_plot, 
                                                          end_of_nonlinearity = end_of_nonlinearity, 
                                                          set_FR_range_from_theory = False, method = 'collective', 
                                                          use_saved_FR_ext= use_saved_FR_ext,
                                                          normalize_G_by_N= False, save_FR_ext=False,
                                                          state = state)

            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            
            if save_pop_act:
                data = save_pop_act_into_dataframe(nuclei_dict, duration_base[0],data, i,j)
                
            if plot_raster:
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer[i], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)

            if find_phase:

                find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins,
                                              height = phase_thresh_h, ref_nuc_name = ref_nuc_name, start = start_phase, total_phase = 720, troughs = troughs)
                # find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins, troughs = troughs,
                #                               height = phase_thresh_h, ref_nuc_name = 'self', start = start_phase, total_phase = 360)
                data = save_phases_into_dataframe(nuclei_dict, data, i,j, ref_nuc_name)
                
            if plot_phase:
                fig_phase = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                                          density = False, ref_nuc_name = ref_nuc_name, total_phase = total_phase, 
                                                          projection = phase_projection, outer=outer_phase[i], fig= fig_phase,  title='', 
                                                          tick_label_fontsize=18, labelsize=15, title_fontsize=15, lw=1, linelengths=1, 
                                                          include_title=True, ax_label=True, nuc_order = nuc_order)
                
            data, nuclei_dict = find_freq_SNN(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
                                check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=plot_spectrum, ax=ax_spec,
                                c_spec=color_dict, spec_figsize=spec_figsize, n_windows=n_windows, fft_method=fft_method, find_beta_band_power=find_beta_band_power,
                                include_beta_band_in_legend=include_beta_band_in_legend, divide_beta_band_in_power = divide_beta_band_in_power, 
                                half_peak_range = 5,  cut_off_freq = 100, check_peak_significance=check_peak_significance, 
                                save_pxx = save_pxx, len_f_pxx = len_f_pxx, normalize_spec=normalize_spec, 
                                plot_sig_thresh = plot_sig_thresh, plot_peak_sig = plot_peak_sig, min_f = min_f, 
                                max_f = max_f, n_std_thresh= n_std_thresh, AUC_ratio_thresh = AUC_ratio_thresh)

        if plot_spectrum:
            if fft_method == 'rfft':
                x_l = 10**9

            else:
                x_l = 5
                ax_spec.axhline(x_l, ls='--', c='grey')

            # ax_spec.set_title(title, fontsize = 18)
            ax_spec.legend(fontsize=11, loc='upper center',
                           framealpha=0.1, frameon=False)
            ax_spec.set_xlim(spec_lim[0], spec_lim[1])
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax_spec)

        if plot_firing:
            ax = fig.add_subplot(n_iter, 1, count+1)
            plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title, include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim, include_FR = include_FR)
            ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
            ax.set_ylim(firing_ylim)
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)

        count += 1
        stop = timeit.default_timer()
        print(count, "from", int(n_iter), 'gs. t=', round(stop - start, 2))

    figs = []
    if plot_firing:
        fig.set_size_inches((15, 15), forward=False)
        fig.text(0.5, 0.05, 'time (ms)', ha='center', fontsize=18)
        fig.text(0.03, 0.5, 'firing rate (spk/s)',
                 va='center', rotation='vertical', fontsize=18)
        figs.append(fig)
    if plot_spectrum:
        fig_spec.set_size_inches((11, 15), forward=False)
        fig_spec.text(0.5, 0.05, 'frequency (Hz)', ha='center', fontsize=18)
        fig_spec.text(0.02, 0.5, 'fft Power', va='center',
                      rotation='vertical', fontsize=18)
        figs.append(fig_spec)

    if plot_raster:
        fig.set_size_inches((11, 15), forward=False)
        fig_raster.text(0.5, 0.05, 'time (ms)', ha='center',
                        va='center', fontsize=18)
        fig_raster.text(0.03, 0.5, 'neuron', ha='center',
                        va='center', rotation='vertical', fontsize=18)
        figs.append(fig_raster)
        fig_raster.show()
    if plot_phase:
        fig_phase.set_size_inches((11, 15), forward=False)
        # fig_phase.text(0.5, 0.05, 'Phase (deg)', ha='center', fontsize=18)
        # fig_phase.text(0.02, 0.5, 'Spike count', va='center',
        #               rotation='vertical', fontsize=18)
        figs.append(fig_phase)
    if save_pkl:
        pickle_obj(data, filepath)
    return figs, title, data
# def extract_g_from_dict(G, loop_key_lists):
    
#     for key,v in syn_decay_dict['tau_1']['tau_ratio'].items():    
#         synaptic_time_constant[key] = [syn_decay_dict['tau_1']['tau_ratio'][key] * t_decay_1]
        
#     if if_track_tau_2:
#         for key,v in syn_decay_dict['tau_2']['tau_ratio'].items():    
#             synaptic_time_constant[key] = [syn_decay_dict['tau_2']['tau_ratio'][key] * t_decay_2]
        
#     return synaptic_time_constant

def synaptic_weight_exploration_SNN_2d(loop_key_lists, tau, path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
    peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
    lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
    plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
    include_beta_band_in_legend=False, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
    reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, N_real = None, N = None,
    receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
    spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
    phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, ref_nuc_name = 'Proto', plot_phase = False,
    total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = True,
    plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, 
    save_pop_act = False, save_gamma = False):

    if set_seed:
        np.random.seed(1956)

    
    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = len(G_dict[loop_key_lists[0][0]])
    n_iter_2 = len(G_dict[loop_key_lists[1][0]])
    # print('size = ', n_iter, n_iter_2)
    data = {}
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0]  # get only on class from each population
        data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_iter_2, n_run))
        
        if save_pop_act :
            data[(nucleus.name, 'pop_act')] = np.zeros((n_iter, n_iter_2, n_run,  duration_base[1] - duration_base[0]))
                                                       
        if check_peak_significance:
            data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_iter_2, n_run), dtype = bool) # stores the value of the PSD at the peak and the mean of the PSD elsewhere
        
        if find_phase:
            data[(nucleus.name, 'rel_phase_hist')] = np.zeros((n_iter, n_iter_2, n_run, 2, n_phase_bins-1))
            data[(nucleus.name, 'rel_phase')] = np.zeros((n_iter, n_iter_2, n_run))

        if divide_beta_band_in_power:
            if save_gamma:
                data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_iter_2, n_run, 3))
            else:
                data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_iter_2, n_run, 2))
            
        else:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_iter_2,  n_run))
        data[(nucleus.name, 'f')] = np.zeros((n_iter, n_iter_2, n_run, len_f_pxx))
        data[(nucleus.name, 'pxx')] = np.zeros((n_iter, n_iter_2, n_run, len_f_pxx))
        
    data['g'] = G_dict
    count = 0
    G = dict.fromkeys(G_dict.keys(), None)

    if n_run > 1:  # don't plot all the runs
        plot_spectrum = False
        plot_firing = False
        plot_phase = False
        plot_raster = False
        
    if plot_firing:
        fig = plt.figure()

    if plot_spectrum:
        fig_spec = plt.figure()

    if plot_raster:
        fig_raster = plt.figure()
        outer = gridspec.GridSpec(n_iter, n_iter_2, wspace=0.2, hspace=0.2)
        
    if plot_phase:
        fig_phase = plt.figure()
        outer_phase = gridspec.GridSpec(n_iter, n_iter_2, wspace=0.2, hspace=0.2)
        
    for i in range(n_iter):
        
        start = timeit.default_timer()
        for k in loop_key_lists[0]:
            G[k] = G_dict[k][i]

        else: ax_spec = None

        
        for m in range(n_iter_2):
            # title =G_element_as_txt(G, i, display=display, decimal=decimal) 
            title = ''
            if plot_spectrum:
                ax_spec = fig_spec.add_subplot(n_iter, n_iter_2, count+1)
            
            for k in loop_key_lists[1]:
                G[k] = G_dict[k][m]
                 
            print_G_items(G)
            for j in range(n_run):
                
                print(' {} from {} runs'.format(j + 1 , n_run))
                nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, tau, G, noise_amplitude, noise_variance, A,
                                                      A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                                      reset_init_dist= reset_init_dist, poisson_prop = poisson_prop, 
                                                      normalize_G_by_N= True)  
                if reset_init_dist:
                    receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                                              all_FR_list = all_FR_list , n_FR =n_FR, if_plot = if_plot, 
                                                              end_of_nonlinearity = end_of_nonlinearity, 
                                                              set_FR_range_from_theory = False, method = 'collective', 
                                                              use_saved_FR_ext= use_saved_FR_ext,
                                                              normalize_G_by_N= False, save_FR_ext=False,
                                                              state = state)
    
    
                nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
                
                if save_pop_act:
                    data = save_pop_act_into_dataframe_2d(nuclei_dict, duration_base[0],data, i, m, j)
                
                if plot_raster:
                    fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer[i,m], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                        plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)
    
                if find_phase:
    
                    find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins,
                                                  height = phase_thresh_h, ref_nuc_name = ref_nuc_name, start = start_phase, total_phase = 720, troughs = troughs)
                    # find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins, troughs = troughs,
                    #                               height = phase_thresh_h, ref_nuc_name = 'self', start = start_phase, total_phase = 360)
                    data = save_phases_into_dataframe_2d(nuclei_dict, data, i, m, j, ref_nuc_name)
                    
                if plot_phase:
                    fig_phase = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                                              density = False, ref_nuc_name = ref_nuc_name, total_phase = total_phase, 
                                                              projection = phase_projection, outer=outer_phase[i,m], fig= fig_phase,  title='', 
                                                              tick_label_fontsize=18, labelsize=15, title_fontsize=15, lw=1, linelengths=1, 
                                                              include_title=True, ax_label=True, nuc_order = nuc_order)
                    
                data, nuclei_dict = find_freq_SNN(data, (i, m, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
                                    check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=plot_spectrum, ax=ax_spec,
                                    c_spec=color_dict, spec_figsize=spec_figsize, n_windows=n_windows, fft_method=fft_method, find_beta_band_power=find_beta_band_power,
                                    include_beta_band_in_legend=include_beta_band_in_legend, divide_beta_band_in_power = divide_beta_band_in_power, 
                                    half_peak_range = 5, cut_off_freq = 100, check_peak_significance=check_peak_significance, 
                                    save_pxx = save_pxx, len_f_pxx = len_f_pxx, normalize_spec=normalize_spec, plot_sig_thresh = plot_sig_thresh, 
                                    plot_peak_sig = plot_peak_sig, min_f = min_f, max_f = max_f, n_std_thresh= n_std_thresh, 
                                    AUC_ratio_thresh = AUC_ratio_thresh, save_gamma = save_gamma)
    
            if plot_spectrum:
                if fft_method == 'rfft':
                    x_l = 10**9
    
                else:
                    x_l = 5
                    ax_spec.axhline(x_l, ls='--', c='grey')
    
                # ax_spec.set_title(title, fontsize = 18)
                ax_spec.legend(fontsize=11, loc='upper center',
                               framealpha=0.1, frameon=False)
                ax_spec.set_xlim(spec_lim[0], spec_lim[1])
                rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax_spec)
    
            if plot_firing:
                ax = fig.add_subplot(n_iter, n_iter_2, count+1)
                plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title, include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                    n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim, include_FR = include_FR)
                ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
                ax.set_ylim(firing_ylim)
                rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)
    
            count += 1
            stop = timeit.default_timer()
            print(count, "from", int(n_iter * n_iter_2), 'gs. t=', round(stop - start, 2))

    figs = []
    if plot_firing:
        fig.set_size_inches((15, 15), forward=False)
        fig.text(0.5, 0.05, 'time (ms)', ha='center', fontsize=18)
        fig.text(0.03, 0.5, 'firing rate (spk/s)',
                 va='center', rotation='vertical', fontsize=18)
        figs.append(fig)
    if plot_spectrum:
        fig_spec.set_size_inches((11, 15), forward=False)
        fig_spec.text(0.5, 0.05, 'frequency (Hz)', ha='center', fontsize=18)
        fig_spec.text(0.02, 0.5, 'fft Power', va='center',
                      rotation='vertical', fontsize=18)
        figs.append(fig_spec)

    if plot_raster:
        fig.set_size_inches((11, 15), forward=False)
        fig_raster.text(0.5, 0.05, 'time (ms)', ha='center',
                        va='center', fontsize=18)
        fig_raster.text(0.03, 0.5, 'neuron', ha='center',
                        va='center', rotation='vertical', fontsize=18)
        figs.append(fig_raster)
        fig_raster.show()
    if plot_phase:
        fig_phase.set_size_inches((11, 15), forward=False)
        # fig_phase.text(0.5, 0.05, 'Phase (deg)', ha='center', fontsize=18)
        # fig_phase.text(0.02, 0.5, 'Spike count', va='center',
        #               rotation='vertical', fontsize=18)
        figs.append(fig_phase)
    if save_pkl:
        pickle_obj(data, filepath)
    return figs, title, data

def Coherence_single_pop_exploration_SNN(noise_dict, tau, path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
    peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
    lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
    plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
    include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
    reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, N_real = None, N = None,
    receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
    spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
    phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, ref_nuc_name = 'Proto', plot_phase = False,
    total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, sampling_t_distance_ms = 1, title_pad = None):

    if set_seed:
        np.random.seed(1956)

    n_iter = get_max_len_dict(noise_dict)
    print(n_iter)

    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    data = {}
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0]  # get only on class from each population
        data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))

        data[(nucleus.name, 'f')] = np.zeros((n_iter, n_run, len_f_pxx))
        data[(nucleus.name, 'pxx')] = np.zeros((n_iter, n_run, len_f_pxx))
        data[(nucleus.name, 'coherence')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'noise_sigma')] = np.zeros((n_iter))
        data[(nucleus.name, 'mean_noise')] = np.zeros((n_iter, n_run))


    if plot_firing:
        fig = plt.figure()

    if n_run > 1:  # don't plot all the runs
        plot_spectrum = False
    if plot_spectrum:
        fig_spec = plt.figure()

    if plot_raster:
        fig_raster = plt.figure()
        outer = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
        
    if plot_phase:
        fig_phase = plt.figure()
        outer_phase = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
    count = 0   
    G = dict.fromkeys(G_dict.keys(), None)

    for i in range(n_iter):
        start = timeit.default_timer()

        for nucleus_list in nuclei_dict.values():
            for nucleus in nucleus_list:
                nucleus.noise_std = np.sqrt( noise_dict[nucleus.name][i] )
                data[(nucleus.name, 'noise_sigma')][i] = noise_dict[nucleus.name][i]
                data[(nucleus.name, 'I_ext_0')] = np.average(nucleus.I_ext_0)

        for k, values in G_dict.items():
            G[k] = values[i]
            
        print_G_items(G)    
            

        if plot_spectrum:
            ax_spec = fig_spec.add_subplot(n_iter, 1, count+1)

        else: ax_spec = None

        title = G_element_as_txt(G_dict, i, display=display, decimal=decimal) 

        for j in range(n_run):
            
            print(' {} from {} runs'.format(j + 1 , n_run))
            nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, tau, G, noise_amplitude, noise_variance, A,
                                                  A_mvt, D_mvt, t_mvt, t_list, dt,  set_noise=False, 
                                                  reset_init_dist= reset_init_dist, poisson_prop = poisson_prop, 
                                                  normalize_G_by_N= True)  
            if reset_init_dist:
                receiving_class_dict = set_connec_ext_inp(path, A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = if_plot, 
                                                          end_of_nonlinearity = end_of_nonlinearity, 
                                                          set_FR_range_from_theory = False, method = 'collective', 
                                                          use_saved_FR_ext= use_saved_FR_ext,
                                                          normalize_G_by_N= False, save_FR_ext=False,
                                                          state = state)


            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            data[(nucleus.name, 'mean_noise')][i,j] = np.average( abs(nucleus.noise_all_t) )
            if plot_raster:
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer[i], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)
 
            data, nuclei_dict = find_freq_SNN(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
                                check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=plot_spectrum, ax=ax_spec,
                                c_spec=color_dict, spec_figsize=spec_figsize, n_windows=n_windows, fft_method=fft_method, find_beta_band_power=find_beta_band_power,
                                include_beta_band_in_legend=include_beta_band_in_legend, divide_beta_band_in_power = divide_beta_band_in_power, 
                                half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance=check_peak_significance, 
                                save_pxx = save_pxx, len_f_pxx = len_f_pxx)
            
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, fig_raster.gca())

            data = set_coherence_all_nuclei_to_data(nuclei_dict, data, i, j, dt, sampling_t_distance_ms=sampling_t_distance_ms)
        if plot_spectrum:
            ax_spec.legend(fontsize=11, loc='upper center',
                           framealpha=0.1, frameon=False)
            ax_spec.set_xlim(spec_lim[0], spec_lim[1])
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax_spec)

        if plot_firing:
            ax = fig.add_subplot(n_iter, 1, count+1)
            title = ( r'$ \langle I_{0} \rangle = $' + str( np.round(data[ref_nuc_name, 'I_ext_0'], 1))  + '   mV   ' + 
                      r'$\langle \eta \rangle = $' + str( np.round(data[ref_nuc_name, 'mean_noise'][i,j], 1))) + '  mV'
            plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title, 
                 include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                 n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, 
                 plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim, include_FR = include_FR,
                 plt_freq = False, title_pad = title_pad)
            
            ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)

        count += 1
        stop = timeit.default_timer()
        print(count, "from", int(n_iter), ' t=', round(stop - start, 2))

    figs = []
    if plot_firing:
        fig.set_size_inches((15, 15), forward=False)
        fig.text(0.5, 0.05, 'time (ms)', ha='center', fontsize=18)
        fig.text(0.03, 0.5, 'firing rate (spk/s)',
                 va='center', rotation='vertical', fontsize=18)
        figs.append(fig)
        fig = set_y_ticks(fig, [30, 50, 70])
    if plot_spectrum:
        fig_spec.set_size_inches((11, 15), forward=False)
        fig_spec.text(0.5, 0.05, 'frequency (Hz)', ha='center', fontsize=18)
        fig_spec.text(0.02, 0.5, 'fft Power', va='center',
                      rotation='vertical', fontsize=18)
        figs.append(fig_spec)

    if plot_raster:
        fig.set_size_inches((11, 15), forward=False)
        fig_raster.text(0.5, 0.005, 'time (ms)', ha='center',
                        va='center', fontsize=18)
        fig_raster.text(0.03, 0.5, 'neuron', ha='center',
                        va='center', rotation='vertical', fontsize=18)
        figs.append(fig_raster)
        fig_raster.show()
    if save_pkl:
        pickle_obj(data, filepath)
    return figs, title, data


def coherence_exploration(path, tau, nuclei_dict, G_dict, noise_amplitude, noise_variance, A, N, N_real, K_real, receiving_pop_list,
                          A_mvt, D_mvt, t_mvt, t_list, dt,  all_FR_list , n_FR, receiving_class_dict,
                          end_of_nonlinearity, color_dict,
                          poisson_prop, reset_init_dist = True, sampling_t_distance_ms = 1, if_plot = False):
    
    n_g = get_max_len_dict(G_dict)
    coherence_dict = {name: np.zeros(n_g) for name in list(nuclei_dict.keys())}
    
    
    for i in range(n_g):
        G = {}
        for k, values in G_dict.items():
            G[k] = values[i]
            print(k, values[i])
        nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, tau,  G, noise_amplitude, noise_variance, A,
                                              A_mvt, D_mvt, t_mvt, t_list, dt,  set_noise=False, 
                                              reset_init_dist= reset_init_dist, poisson_prop = poisson_prop, 
                                              normalize_G_by_N= True)  
        if reset_init_dist:
            receiving_class_dict = set_connec_ext_inp(path, A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                                      all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, 
                                                      end_of_nonlinearity = end_of_nonlinearity, 
                                                      set_FR_range_from_theory = False, method = 'collective', 
                                                      save_FR_ext=True, use_saved_FR_ext= True, 
                                                      normalize_G_by_N= False, state = 'rest')


        nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
        coherence_dict = set_coherence_all_nuclei_1d(nuclei_dict, coherence_dict, i, dt, 
                                                  sampling_t_distance_ms = sampling_t_distance_ms)
        
        if if_plot:
            fig = plot(nuclei_dict,color_dict, dt,  t_list, A, A_mvt, t_mvt, D_mvt, ax = None, 
                       title_fontsize=15, plot_start = 0, title = str(dt),
                       include_FR = False, include_std=False, plt_mvt=False,
                       legend_loc='upper right', ylim =None)
    coherence_dict['G'] = G_dict
        
    return coherence_dict

def set_coherence_all_nuclei_1d(nuclei_dict, coherence_dict, i, dt, sampling_t_distance_ms=1):
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            coherence_dict[nucleus.name][i] = nucleus.cal_coherence(dt, sampling_t_distance_ms= sampling_t_distance_ms)
    return coherence_dict

def set_coherence_all_nuclei_to_data(nuclei_dict, data, i, j, dt, sampling_t_distance_ms=1):
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            data[nucleus.name, 'coherence'][i, j] = nucleus.cal_coherence(dt, sampling_t_distance_ms= sampling_t_distance_ms)
    return data

def max_in_dict(dictionary):
    
    return np.max(
                np.array(
                        list(
                           dictionary.values()
                            )
                        ).flatten()
                )

def min_in_dict(dictionary):
    
    return np.min(
                np.array(
                        list(
                           dictionary.values()
                            )
                        ).flatten()
                )

def sinfunc(t, A, w, p, c):  return A * np.sin(w*(t - p)) + c

def fit_sine(t, y):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''

    f_init = np.fft.fftfreq(len(t), (t[1]-t[0]))   # assume uniform spacing
    Fy = abs(np.fft.fft(y))
    guess_freq = abs(f_init[np.argmax(Fy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    
    popt, pcov = curve_fit(sinfunc, t, y, p0=guess, maxfev=20000)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*(t - p)) + c
    # return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
    return A, w, p, c, f, fitfunc

def filter_pop_act_all_nuclei(nuclei_dict, dt,  lower_freq_cut, upper_freq_cut, order=6):
    
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
    
            nucleus.butter_bandpass_filter_pop_act(dt, lower_freq_cut, upper_freq_cut, order= order)
            
def find_freq_SNN(data, element_ind,  dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, 
                  cut_plateau_epsilon, check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, 
                  plot_spectrum=False, ax=None, c_spec='navy', spec_figsize=(6, 5), find_beta_band_power=False,
                  fft_method='rfft', n_windows=3, include_beta_band_in_legend=True, divide_beta_band_in_power = False, 
                  half_peak_range = 5, cut_off_freq = 100, check_peak_significance = False, len_f_pxx = 200, 
                  save_pxx = True, normalize_spec = True, plot_sig_thresh = False, plot_peak_sig = False, smooth = True,
                  min_f = 100, max_f = 300, n_std_thresh = 2, AUC_ratio_thresh = 0.8, save_gamma = False):

    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            
            if smooth:

                nucleus.smooth_pop_activity(dt, window_ms=smooth_window_ms)

            if low_pass_filter:
                nucleus.butter_bandpass_filter_pop_act(dt, lower_freq_cut, upper_freq_cut, order=6)

            (_ , _,
            data[(nucleus.name, 'base_freq')][element_ind],
            if_stable_base,
            data[(nucleus.name, 'base_beta_power')][element_ind],
            f, pxx) = nucleus.find_freq_of_pop_act_spec_window(*duration_base, dt,
                                                               peak_threshold=peak_threshold,
                                                               smooth_kern_window=smooth_kern_window,
                                                               cut_plateau_epsilon=cut_plateau_epsilon,
                                                               check_stability=check_stability,
                                                               method=freq_method,
                                                               plot_sig=plot_sig,
                                                               plot_spectrum=plot_spectrum,
                                                               ax=ax, c_spec=c_spec[nucleus.name],
                                                               fft_label=nucleus.name,
                                                               spec_figsize=spec_figsize,
                                                               find_beta_band_power=find_beta_band_power,
                                                               fft_method=fft_method,
                                                               n_windows=n_windows,
                                                               include_beta_band_in_legend=include_beta_band_in_legend,
                                                               divide_beta_band_in_power= divide_beta_band_in_power,
                                                               normalize_spec = normalize_spec,
                                                               plot_sig_thresh = plot_sig_thresh,
                                                               min_f = min_f, max_f = max_f, 
                                                               n_std_thresh = n_std_thresh,
                                                               save_gamma = save_gamma)
            if save_pxx:   
            
                data = save_pxx_to_df(f, pxx, len_f_pxx, element_ind, data, nucleus.name)
                
            if check_peak_significance:
                
                    data[(nucleus.name, 'peak_significance')][element_ind] = check_significance_of_PSD_peak(f, pxx, 
                                                                                                            n_std_thresh = n_std_thresh, 
                                                                                                            min_f = min_f, 
                                                                                                            max_f = max_f, 
                                                                                                            n_pts_above_thresh = 3,
                                                                                                            if_plot = plot_peak_sig,
                                                                                                            AUC_ratio_thresh = AUC_ratio_thresh)
                    
                                  
                

            nucleus.frequency_basal = data[(nucleus.name, 'base_freq')][element_ind]

            print(nucleus.name, 'f = ', round(data[(nucleus.name, 'base_freq')][element_ind], 2), 'beta_p =', data[(
                nucleus.name, 'base_beta_power')][element_ind], np.sum(data[(
                nucleus.name, 'base_beta_power')][element_ind]))

    return data, nuclei_dict

def save_pxx_to_df(f, pxx, len_f_pxx, element_ind, data, name):
    
    if len(pxx) >= len_f_pxx:            
        f_to_save, pxx_to_save = f[:len_f_pxx], pxx[:len_f_pxx]   
        
    else:

        nan_arr = np.full(len_f_pxx - len(pxx), np.nan)
        f_to_save, pxx_to_save = np.append(f, nan_arr), np.append(pxx, nan_arr)
        
    if len(element_ind) == 2:
        
        data = save_pxx_into_dataframe(f_to_save, pxx_to_save, name, data, *element_ind)
    
    else:
        
        data = save_pxx_into_dataframe_2d(f_to_save, pxx_to_save, name, data, *element_ind)
        
    return data



def find_freq_SNN_not_saving(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, 
                             smooth_kern_window, smooth_window_ms, 
                             cut_plateau_epsilon, check_stability, freq_method, plot_sig,
                             low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=False, 
                             ax=None, c_spec='navy', spec_figsize=(6, 5), find_beta_band_power=False,
                             fft_method='rfft', n_windows=3, include_beta_band_in_legend=True, smooth = False, 
                             normalize_spec = True, include_peak_f_in_legend = True,
                             check_significance = False, plot_sig_thresh = False, plot_peak_sig = False,
                             min_f = 100, max_f = 300, n_std_thresh = 2,AUC_ratio_thresh = 0.8):
    pxx = {}
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            
            if smooth:
                
                nucleus.smooth_pop_activity(dt, window_ms=smooth_window_ms)

            if low_pass_filter:
                nucleus.butter_bandpass_filter_pop_act(dt, lower_freq_cut, upper_freq_cut, order=6)

            (n_half_cycles, perc_oscil, freq, 
             if_stable, beta_band_power, f, pxx[nucleus.name]) = nucleus.find_freq_of_pop_act_spec_window(*duration_base, dt,
                                                                                                          peak_threshold=peak_threshold,
                                                                                                          smooth_kern_window=smooth_kern_window,
                                                                                                          cut_plateau_epsilon=cut_plateau_epsilon,
                                                                                                          check_stability=check_stability,
                                                                                                          method=freq_method,
                                                                                                          plot_sig=plot_sig,
                                                                                                          plot_spectrum=plot_spectrum,
                                                                                                          ax=ax, c_spec=c_spec[nucleus.name],
                                                                                                          fft_label=nucleus.name,
                                                                                                          spec_figsize=spec_figsize,
                                                                                                          find_beta_band_power=find_beta_band_power,
                                                                                                          fft_method=fft_method,
                                                                                                          n_windows=n_windows,
                                                                                                          include_beta_band_in_legend=include_beta_band_in_legend,
                                                                                                          normalize_spec = normalize_spec,
                                                                                                          include_peak_f_in_legend = include_peak_f_in_legend,
                                                                                                          plot_sig_thresh = plot_sig_thresh,
                                                                                                          min_f = min_f, max_f = max_f, 
                                                                                                          n_std_thresh = n_std_thresh)
            if check_significance:
                sig_bool = check_significance_of_PSD_peak(f, pxx[nucleus.name], 
                                                          n_std_thresh = n_std_thresh, 
                                                          min_f = min_f, 
                                                          max_f = max_f, 
                                                          n_pts_above_thresh = 3,
                                                          if_plot = plot_peak_sig, 
                                                          name = nucleus.name,
                                                          AUC_ratio_thresh = AUC_ratio_thresh)


    return freq, f, pxx


    
def rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    if count+1 < n_iter:
        ax.axes.xaxis.set_ticklabels([])


def G_element_as_txt(G_dict, i, display='normal', decimal=0):

    title = ''
    for j in range(len(G_dict)):
        if display == 'normal':
            title += r"$G_{"+list(G_dict.keys())[j][0]+"-"+list(G_dict.keys())[j][1]+"}$ = " + str(round(list(G_dict.values())[j][i], 2)) + ', '
        
        elif display == 'sci':
            title += r"$G_{"+list(G_dict.keys())[j][0]+"-"+list(G_dict.keys())[j][1]+"}$ = " + r"${0:s}$".format(as_si(list(G_dict.values())[j][i], decimal)) + ', '
        if (j+1) % 3 == 0:
            title += ' \n '  
    return title

def G_as_txt(G_dict, display='normal', decimal=0):

    G_txt = ''
    for j in range(len(G_dict)):
        if display == 'normal':
            G_txt += r"$G_{"+list(G_dict.keys())[j][0]+"-"+list(G_dict.keys())[j][1]+"}$ = " + str(round(list(G_dict.values())[j], 2)) + ', '
        
        elif display == 'sci':
            G_txt += r"$G_{"+list(G_dict.keys())[j][0]+"-"+list(G_dict.keys())[j][1]+"}$ = " + r"${0:s}$".format(as_si(list(G_dict.values())[j], decimal)) + ', '
        if (j+1) % 3 == 0:
            G_txt += ' \n '  
    return G_txt

def remove_frame(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def remove_whole_frame(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
def plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict, ax = None):
    fig, ax = get_axes(ax)
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            nucleus.plot_mem_potential_distribution_of_all_t(
                ax=ax, color=color_dict[nucleus.name], bins=100)
    remove_frame(ax)
    return fig, ax


def _plot_signal(if_plot, start, end, dt, sig, plateau_y, cut_sig_ind):
    if if_plot:
        ax = plt.figure()
        t = np.arange(start, end) * dt
        ax.plot(t, sig - plateau_y)
        ax.axhline(plateau_y)
        ax.plot(t[cut_sig_ind], sig[cut_sig_ind])
        return ax


def find_FR_ext_range_for_each_neuron(FR_sim, all_FR_list, init_method,  n_FR=50, left_pad=0.005, right_pad=0.005):
    ''' put log spaced points in the nonlinear regime of each neuron'''
    n_pop = FR_sim.shape[0]
    if init_method == 'homogeneous':

        start_act = np.average([all_FR_list[np.min(np.where(FR_sim[i, :] > 1)[0])]
                               for i in range(n_pop)])  # setting according to one neurons would be enough
        FR_list = np.repeat(spacing_with_high_resolution_in_the_middle(
            n_FR, start_act - left_pad, start_act + right_pad).reshape(-1, 1), n_pop,  axis=1)

    else:

        FR_list = np.zeros((n_FR-1, n_pop))
        for i in range(n_pop):

            start_act = all_FR_list[np.min(np.where(FR_sim[i, :] > 1)[0])]
            FR_list[:, i] = spacing_with_high_resolution_in_the_middle(
                n_FR, start_act - left_pad, start_act + right_pad).reshape(-1,)

    return FR_list


def find_FR_ext_range_for_each_neuron_high_act(FR_sim, all_FR_list, init_method,  n_FR=25, left_pad=0.005, right_pad=0.005, lin_start=35):
    ''' linear spaced FR for linear regime of response curve with high noise (i.e. Proto neurons)'''
    n_pop = FR_sim.shape[0]
    if init_method == 'homogeneous':

          # setting according to one neurons would be enough

        start_act = np.average([all_FR_list[np.min(np.where(FR_sim[i, :] > lin_start)[
                               0])] for i in range(n_pop)])  # setting according to one neurons would be enough

        FR_list = np.repeat(np.linspace(
            start_act, all_FR_list[-1], n_FR).rehsape(-1, 1), n_pop, axis=1)

    else:

        FR_list = np.zeros((n_FR, n_pop))
        for i in range(n_pop):

            start_act = all_FR_list[np.min(np.where(FR_sim[i, :] > lin_start)[0])]
            FR_list[:, i] = np.linspace(start_act, all_FR_list[-1], n_FR).reshape(-1,)

    return FR_list



def pickle_obj(obj, filepath):

    with open(filepath, "wb") as file_:
        pickle.dump(obj, file_)  # , -1)


def load_pickle(filepath):

    return pickle.load(open(filepath, "rb"))  # , -1)


def write_obj_to_json(obj, filepath):

    frozen = jsonpickle.encode(obj)
    with open(filepath, "w") as fh:
        fh.write(frozen)


def load_json_file_as_obj(filepath):
    o = open(filepath, 'r')
    file_ = o.read()
    return jsonpickle.decode(file_)


def ax_label_adjust(ax, fontsize=18, nbins=5, ybins=None):
    if ybins == None:
        ybins = nbins
    ax.locator_params(axis='y', nbins=ybins)
    ax.locator_params(axis='x', nbins=nbins)
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize


def plot_fitted_sigmoid(xdata, ydata, x_scaled, desired_FR, FR_to_I_coef=0, coefs=[], ax=None,  noise_var=0, c='grey'):
    fig, ax = get_axes(ax)
    ax.plot(xdata * FR_to_I_coef, ydata, 'o', label=r'$\sigma^{2} =$' + str(noise_var),
            c=c, markersize=7, markerfacecolor='none', markeredgewidth=1.5)
    if coefs != []:
        y = sigmoid(x_scaled, *coefs)
        I_ext = (x_scaled + find_x_mid_point_sigmoid(ydata, xdata)) * FR_to_I_coef
        FR = y * find_y_normalizing_factor(ydata, desired_FR)
        ax.plot(I_ext, FR, c=c, label='fitted curve', lw=2)
    ax.legend(framealpha=0.1, frameon=False)
    ax.set_xlabel(r'$I_{ext} \; (mV)$', fontsize=15)
    ax.set_ylabel('FR (Hz)', fontsize=15)
    ax_label_adjust(ax)
    remove_frame(ax)
    if ax != None:
        ax.legend()


def plot_fitted_line(x, y, slope, intercept, FR_to_I_coef=0, ax=None, noise_var=0, c='grey'):
    fig, ax = get_axes(ax)
    ax.plot(x * FR_to_I_coef, y, 'o', label=r'$\sigma^{2} =$' + str(noise_var),
            c=c, markersize=7, markerfacecolor='none', markeredgewidth=1.5)
    ax.plot(x * FR_to_I_coef, slope * x + intercept,
            c=c, label='fitted curve', lw=2)
    ax.set_xlabel(r'$I_{ext} \; (mV)$', fontsize=15)
    ax.set_ylabel('FR (Hz)', fontsize=15)
    ax_label_adjust(ax)
    remove_frame(ax)
    if ax != None:
        ax.legend(framealpha=0.1, frameon=False)


def scale_bound_with_mean(mean, lower_bound_perc, upper_bound_perc, scale=None):
    lower_bound = mean * lower_bound_perc
    upper_bound = mean * upper_bound_perc
    return lower_bound, upper_bound


def scale_bound_with_arbitrary_value(mean, lower_bound_perc, upper_bound_perc, scale=1):
    lower_bound = mean - scale * lower_bound_perc
    upper_bound = mean + scale * upper_bound_perc
    return lower_bound, upper_bound


def truncated_normal_distributed(mean, sigma, n, scale_bound=scale_bound_with_mean, scale=None, lower_bound_perc=0.8, upper_bound_perc=1.2, 
                                 truncmin=None, truncmax=None):
    
    if truncmin != None and truncmax != None:
        lower_bound, upper_bound = truncmin, truncmax
    else:
        lower_bound, upper_bound = scale_bound(
            mean, lower_bound_perc, upper_bound_perc, scale=scale)
    return stats.truncnorm.rvs((lower_bound-mean)/sigma, (upper_bound-mean)/sigma, loc=mean, scale=sigma, size=int(n))


def find_FR_sim_vs_FR_ext(FR_list, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict, A, A_mvt, D_mvt, t_mvt):
    ''' find the simulated firing of population given different externalfiring rates'''

    nucleus_name = list(nuclei_dict.keys()); m = len(FR_list)
    firing_prop = {k: {'firing_mean': np.zeros((m, len(nuclei_dict[nucleus_name[0]]))), 'firing_var': np.zeros(
        (m, len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
    i = 0
    for FR in FR_list:

        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.clear_history()
                nucleus.FR_ext = FR
                nucleus.rest_ext_input = FR * nucleus.syn_weight_ext_pop * \
                                        nucleus.n_ext_population * nucleus.membrane_time_constant
        nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.smooth_pop_activity(dt, window_ms=5)
                FR_mean, FR_std = nucleus. average_pop_activity(int(len(t_list) / 2) , len(t_list))
                firing_prop[nucleus.name]['firing_mean'][i,
                    nucleus.population_num-1] = FR_mean
                firing_prop[nucleus.name]['firing_var'][i,
                    nucleus.population_num-1] = FR_std
                print(nucleus.name, np.round(np.average(nucleus.FR_ext), 3),
                    'FR=', FR_mean, 'std=', round(FR_std, 2))
        i += 1
    return firing_prop

def plot_action_potentials(nucleus, n_neuron= 0, t_start = 0, t_end = 1000):
    fig, ax = plt.subplots()
    ax.plot(nucleus.all_mem_pot[n_neuron, t_start: t_end])      
    spikes = np.where(nucleus.spikes[n_neuron, t_start: t_end] == 1)[0]
    for sp in spikes:
        ax.axvline(sp, c = 'r')     
        

def instantaneus_rise_expon_decay(inputs, I=0, I_rise=None, tau_decay=5, tau_rise=None):

    # dt incorporated in tau
    return I + (-I + inputs) / tau_decay, np.zeros_like(I)


def _dirac_delta_input(inputs, dt, I_rise=None, I=None, tau_rise=None, tau_decay=None):
    return inputs , np.zeros_like(inputs)


def exp_rise_and_decay(inputs, dt,  I_rise=0, I=0, tau_rise=5, tau_decay=5):
    # print(I_rise.shape, inputs.shape)
    I_rise = I_rise + (-I_rise + inputs) / tau_rise  # dt incorporated in tau
    I = I + (-I + I_rise) / tau_decay  # dt incorporated in tau
    return I, I_rise


def fwd_Euler(dt, y, f):

    return y + dt * f


def f_LIF(tau, V, V_rest, I_ext, I_syn):
    ''' return dV/dt value for Leaky-integrate and fire neurons'''
    return (-(V - V_rest) + I_ext + I_syn) / tau


def save_all_mem_potential(nuclei_dict, path, state):
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            np.save(os.path.join(path, 'all_mem_pot_' + nucleus.name + '_tau_' + str(np.round(
                nucleus.neuronal_consts['membrane_time_constant']['mean'], 1)).replace('.', '-')) + '_' + state, nucleus.all_mem_pot)


def draw_random_from_data_pdf(data, n, bins=50, if_plot=False):

    hist, bins = np.histogram(data, bins=bins)

    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(n)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    if if_plot:
        plt.figure()
        plt.subplot(121)
        plt.hist(data, bins)
        plt.title('data pdf', fontsize=15)
        plt.subplot(122)
        plt.hist(random_from_cdf, bins)
        plt.title('drawn random variable pdf', fontsize=15)
        plt.show()
    return random_from_cdf


def Runge_Kutta_second_order_LIF(dt, V_t, f_t, tau, I_syn_next_dt, V_rest, I_ext, half_dt):
    ''' Solve second order Runge-Kutta for a LIF at time t+dt (Mascagni & Sherman, 1997)'''
    # print(np.isnan(np.sum(V_t)), np.isnan(np.sum(f_t)), np.isnan(np.sum(V_rest)), np.isnan(np.sum(I_syn_next_dt)), np.isnan(np.sum(I_ext)), np.isnan(np.sum(tau)))
    V_next_dt = V_t + half_dt * (-(V_t + dt * f_t - V_rest) +
                              I_syn_next_dt + I_ext) / tau + f_t * half_dt
    return V_next_dt


def linear_interpolation(V_estimated_next_dt, V_thresh, dt, V_t, V_rest, tau):

    return (
            (V_estimated_next_dt - V_thresh) *
            (1 + dt / tau * (V_t - V_rest) / (V_estimated_next_dt - V_t)) + V_rest
            )


def FR_ext_theory(V_thresh, V_rest, tau, g_ext, FR_list, N_ext):
    ''' calculate what external input is needed to get the desired firing of the list FR_list '''
    frac = (V_thresh - V_rest) / (FR_list * g_ext * N_ext * tau)
    return -1 / np.log(1 - frac) / tau


def FR_ext_of_given_FR_theory(V_thresh, V_rest, tau, g_ext, FR, N_ext):
    ''' calculate what external input is needed to get the desired firing of the list FR_list '''
    exp = np.exp(-1 / tau / FR)
    return (V_thresh - V_rest) / (1-exp) / tau / g_ext / N_ext


def find_x_mid_point_sigmoid(y, x):
    ''' extrapolate the x of half max of y from the two nearest data points'''
    y_relative = y - np.max(y)/2
    signs = np.sign(y_relative)
    y_before = np.max(np.where(signs < 0)[0])
    y_after = np.min(np.where(signs > 0)[0])
    return (x[y_before] + x[y_after]) / 2


def expon(x, a, b, c):
    return a * np.exp(-b * x) + c


def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k*(x-x0)))


def inverse_sigmoid(y, x0, k):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            output = -1/k * np.log((1 - y) / y) + x0

        except Warning as e:
                  print('error found:', e, y)

    return output
#     try:
#         return -1/k * np.log ( (1 - y) / y) + x0
#     except RuntimeWarning:
#         print('y  = ', y)


def inverse_linear(y, a, b):
    return (y - b) / a
#     try:
#         return ( y - b ) / a
#     except RuntimeWarning:
#         print('y  = {} \n b = {}, a = {}'.format(y, b, a))


def linear_regresstion(x, y):

    res = stats.linregress(x, y)
    return res.slope, res.intercept


def fit_FR_as_a_func_of_FR_ext(FR_ext, FR, estimating_func, maxfev=5000):
    popt, pcov = curve_fit(estimating_func, FR_ext.reshape(-1,),
                           FR, method='dogbox', maxfev=maxfev)
    return popt


def extrapolated_FR_ext_from_fitted_curve(FR_ext, FR, desired_FR, coefs, estimating_func, inverse_estimating_func, FR_normalizing_factor, x_shift):

    return inverse_estimating_func(desired_FR / FR_normalizing_factor, *coefs) + x_shift


def find_y_normalizing_factor(y, desired_FR, epsilon=0.2):
    y_max = np.max(y)
    if desired_FR >= y_max:  # if the maximum of the curve is the same as the desired FR add epsilon to it to avoid errors in log
        print('Oooops! max_sigmoid < desired_FR')
#         y_max = np.max(y) + epsilon
        return (desired_FR + epsilon)
    else:
        return y_max


# def find_y_normalizing_factor (y, desired_FR):
#     return np.max (y)

def get_non_linear_part(x, y, end_of_nonlinearity=25):
    ind = np.where(y < end_of_nonlinearity)[0]
    return x[ind], y[ind]


def rescale_x_and_y(x, y, desired_FR):
    ydata = y / find_y_normalizing_factor(y, desired_FR)
    x_shift = find_x_mid_point_sigmoid(y, x)
    xdata = x - x_shift
    return xdata, ydata


def create_color_map(n_samples, colormap=plt.cm.viridis):
    colormap = colormap  # LinearSegmentedColormap
    Ncolors = min(colormap.N, n_samples)
    mapcolors = [colormap(int(x*colormap.N/Ncolors)) for x in range(Ncolors)]
    return mapcolors


def create_sparse_matrix(matrix, end=None, start=0):
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    if end == None:
        end = n_cols
    return np.array([np.where(matrix[i, int(start):int(end)] == 1)[0] for i in range(n_rows)], dtype=object) + int(start)

def get_corr_key_to_val(mydict, value):
    
	""" return all the keys corresponding to the specified value"""
    
	return [k for k, v in mydict.items() if v == value]


def get_axes(ax, figsize=(6, 5)):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plt.gcf(), ax


def raster_plot(spikes_sparse, name, color_dict, color='k',  ax=None, tick_label_fontsize=10, title_fontsize=15, linelengths=2.5, lw=3,
                axvspan=False, span_start=None, span_end=None, axvspan_color='lightskyblue', orientation = 'horizontal',
                xlim=None, include_nuc_name = True, x_tick_length = 7, remove_ax_frame = False, y_tick_length = 2):
    
    fig, ax = get_axes(ax)

    ax.eventplot(spikes_sparse, colors=color_dict[name],
                 linelengths=linelengths, lw=lw, orientation= orientation)
    
    if include_nuc_name:
        ax.set_title(name, c=color_dict[name], fontsize=title_fontsize)
        
    if axvspan:
        ax.axvspan(span_start, span_end, alpha=0.2, color=axvspan_color)

    if xlim != None:
        ax.set_xlim(xlim)
        
    ax.legend(loc='upper right', framealpha=0.1, frameon=False)
    ax.tick_params(axis='y', length = y_tick_length)
    ax.tick_params(axis='x', length = x_tick_length)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    
    if remove_ax_frame:
        remove_frame(ax)
    ax_label_adjust(ax, fontsize=tick_label_fontsize, nbins=4)
    
    return ax



def phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, nuc_order = None,
                          density = False, ref_nuc_name = 'self', total_phase = 360, projection = None, legend_loc = 'upper right',
                          outer=None, fig=None,  title='', tick_label_fontsize=18, ylim = None, plot_mode = 'line',
                           labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, ax_label=False):
    fig_generated = False
    outer_generated = False
    if fig == None:
        fig_generated = True
        fig = plt.figure(figsize=(5, 8))
    if outer == None:
        outer_generated = True
        outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)[0]

    inner = gridspec.GridSpecFromSubplotSpec(len(nuclei_dict), 1,
                    subplot_spec=outer, wspace=0.1, hspace=0.1)
    j = 0
    
    if include_title:
        ax = plt.Subplot(fig, outer)
        ax.set_title(title, fontsize=15)
        ax.axis('off')
    if nuc_order == None:
        nuc_order = list(nuclei_dict.keys())

    for nuc_name in nuc_order:
        nucleus = nuclei_dict[nuc_name][0]
        # if not fig_generated and not outer_generated:
        #     ax = fig.axes[j]
        # else:
        ax = plt.Subplot(fig, inner[j])
        frq, edges = nucleus.spike_rel_phase_hist[ref_nuc_name]

        if projection == None:
            
            width=np.diff(edges) 
            if plot_mode == 'hist':
                ax.bar(edges, frq, width=np.append(width, width[-1]), align = 'edge', facecolor = color_dict[nucleus.name])
            elif plot_mode == 'line':
                ax.plot(get_centers_from_edges(edges), frq, color = color_dict[nucleus.name], label = nucleus.name)
            ax.axvline(total_phase/2, ls = '--', c = 'k')
        elif projection == 'polar':
            
            circular_bar_plot_as_hist(ax, frq, edges, fill = True, alpha = 0.3, density = density,  facecolor = color_dict[nucleus.name])

        # if fig_generated:
        fig.add_subplot(ax)
        rm_ax_unnecessary_labels_in_subplots(j, len(nuclei_dict), ax)
        j += 1
        ax.set_xticks([0,180,360,540,720])
        ax.annotate(nucleus.name, xy=(0.86,0.2),xycoords='axes fraction', color = color_dict[nucleus.name],
             fontsize=14)
        ax.set_xlim(0,total_phase)
        # ax.legend(fontsize = 12, loc = legend_loc)
    if ax_label:
        fig.text(0.5, 0.03, 'phase (deg)', ha='center',
                 va='center', fontsize=labelsize)
        fig.text(0.03, 0.5, 'spike count', ha='center', va='center',
                 rotation='vertical', fontsize=labelsize)
    return fig

def plot_exper_FR_distribution(xls, name_list, state_list, color_dict, bins = 'auto', 
                               alpha = 0.2, hatch = '/', zorder = 1, edgecolor = None):
    figs = {}
    FR_df = {}
    for state in state_list:
        fig, ax = plt.subplots()
        for name in name_list:
            figs[name] = fig
            try:
                FR_df[name] = pd.read_excel(xls, name, header = [0], skiprows = [0])
                col = name + "_" + state + "_FR"
                notna_ind = FR_df[name][col].notna()
                FR = FR_df[name][col][notna_ind]
                freq, edges = np.histogram(FR, bins = bins)
                width = np.diff(edges[:-1])
                ax.bar( edges[:-1], freq / len(FR) * 100,  
                       width=np.append(width, width[-1]), 
                       align = 'edge', facecolor = color_dict[name],
                        label = name + ' De La Crompe (2020)',  alpha = alpha, 
                        hatch = hatch, edgecolor = edgecolor,   
                        zorder = zorder, lw = 2)
                plt.rcParams['hatch.linewidth'] = 3
    
                ax.annotate( r'$ FR = {0} \pm {1} \; Hz$'.format( round (np.average( FR) , 2) , round( np.std( FR), 2) ),
                            xy=(0.5,0.7),xycoords='axes fraction', color = color_dict[name],
                            fontsize=14, alpha = alpha)
                print(name, state,
                      ' mean = ', np.round( np.average(FR), 2), 
                      ' std = ', np.round( np.std(FR), 2) )
            except ValueError:
                pass
        ax.set_title(state, fontsize =15)
        ax.set_xlabel('Firing Rate (spk/s)', fontsize=15)
        ax.set_ylabel('% of population', fontsize=15)
        # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
        ax.legend(fontsize=15,  framealpha = 0.1, frameon = False)
        
    return figs

def plot_FR_distribution(nuclei_dict, dt, color_dict, bins = 50, ax = None, zorder = 1, 
                         alpha = 0.2, start = 0, log_hist = False, box_plot = False, 
                         n_pts = 50, only_non_zero = False):
    
    ''' plot the firing rate distribution of neurons of different populations '''
    
    fig, ax = get_axes(ax)
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            FR_mean_neurons = np.average(nucleus.spikes[:,start:] , axis = 1) / (dt/1000)
            if only_non_zero:
                FR_mean_neurons = FR_mean_neurons[ FR_mean_neurons > 0]

            FR_std_neurons = np.std(FR_mean_neurons) 
            freq, edges = np.histogram(FR_mean_neurons, bins = bins)
            width = np.diff(edges[:-1])
            ax.annotate( r'$ FR = {0} \pm {1}\; Hz$'.format( round (np.average( FR_mean_neurons) , 2) , round( FR_std_neurons, 2) ),
                        xy=(0.5,0.55),xycoords='axes fraction', color = color_dict[nucleus.name],
             fontsize=14, alpha = alpha)
            
            
            if box_plot:
                bp = ax.boxplot(FR_mean_neurons, labels = [nucleus.name], patch_artist=True, whis = (0,100), 
                                widths = 0.6, zorder = 0 )
                for patch, color in zip(bp['boxes'], [color_dict[nucleus.name]]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.2)
                for median in (bp['medians']): 
                    median.set(color = 'k', 
                               linewidth = 0.5) 
                xs = np.random.normal(1, 0.04, n_pts)
                print(len(xs), len(FR_mean_neurons[:n_pts]))
                ax.scatter(xs, FR_mean_neurons[:n_pts], c= color_dict[nucleus.name], alpha=0.4, s = 10, ec = 'k', zorder = 1)
                ax.tick_params(axis='x', labelsize= 10)
                ax.tick_params(axis='y', labelsize= 12)
            else:
                ax.bar( edges[:-1], freq / nucleus.n * 100,  width=np.append(width, width[-1]), align = 'edge', facecolor = color_dict[nucleus.name],
                       label=nucleus.name,  alpha =alpha, zorder = zorder)
    if log_hist and not box_plot:
        ax.set_xscale("log")
    ax.set_xlabel('Firing Rate (spk/s)', fontsize=15)
    ax.set_ylabel('% of population', fontsize=15)
    # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
    ax.legend(fontsize=15,  framealpha = 0.1, frameon = False)
    
    return fig

def plot_ISI_distribution(nuclei_dict, dt, color_dict, bins = 50, ax = None, zorder = 1, 
                          alpha = 0.2, start = 0, log_hist = False, mean_neurons = False):
    
    ''' plot the interspike interval distribution of neurons of different populations '''
    
    fig, ax = get_axes(ax)
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            if mean_neurons:
                ISI, ISI_std = get_mean_sd_ISI_of_neurons(nucleus, start, dt)
                ylabel = '% populatoin'
            else:
                ISI, ISI_std = get_all_ISI_of_neurons(nucleus, start, dt)   
                ylabel = '% spike count'                           

            
            freq, edges = np.histogram(ISI, bins = bins)
            width = np.diff(edges[:-1])
            

            ax.bar( edges[:-1], freq / nucleus.n * 100,  width=np.append(width, width[-1]), align = 'edge', facecolor = color_dict[nucleus.name],
                    label=nucleus.name,  alpha =alpha, zorder = zorder)
            ax.annotate( r'$ ISI = {0} \pm {1}\; ms$'.format( round (np.average( ISI) , 2) , round( ISI_std, 2) ),
                        xy=(0.1,0.8),xycoords='axes fraction', color = color_dict[nucleus.name],
             fontsize=14, alpha = alpha)
            
    if log_hist:
        ax.set_xscale("log")
        
    ax.set_xlabel('ISI (ms)', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
    ax.legend(fontsize=15,  framealpha = 0.1, frameon = False)
    
    return fig

def get_mean_sd_ISI_of_neurons(nucleus, start, dt):
    
    ISI_mean_neurons = np.array (
                         [np.average( 
                                 np.diff(
                                     np.where(nucleus.spikes[i,start:] == 1)[0] 
                                         )
                                     ) for i in range(nucleus.n) if len(np.where(nucleus.spikes[i,start:] == 1)[0]) >= 2 ]
                            ) * dt
    ISI_std_neurons = np.std(ISI_mean_neurons) 
    
    return ISI_mean_neurons, ISI_std_neurons

def get_all_ISI_of_neurons(nucleus, start, dt):
    
    ISI_all_t_neurons = np.array([])
    for i in range(nucleus.n):
        if len(np.where(nucleus.spikes[i,start:] == 1)[0]) >= 2:
            ISIs = np.diff( np.where(nucleus.spikes[i,start:] == 1)[0] 
                                             ) * dt
            if len(ISIs) > 0:
                ISI_all_t_neurons = np.append( ISI_all_t_neurons, ISIs )

    ISI_all_t_neurons_sd = np.std(ISI_all_t_neurons) 
    
    return  ISI_all_t_neurons, ISI_all_t_neurons_sd

def plot_spike_amp_distribution(nuclei_dict, dt, color_dict, bins = 50):
    ''' plot the firing rate distribution of neurons of different populations '''
    fig, ax = plt.subplots()
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            spk_amplitude = nucleus.spike_thresh - nucleus.u_rest
            freq, edges = np.histogram(spk_amplitude, bins = bins)
            width = np.diff(edges[:-1])
            ax.bar( edges[:-1], freq / nucleus.n * 100,  width=np.append(width, width[-1]), align = 'edge', facecolor = color_dict[nucleus.name],
                    label=nucleus.name,  alpha = 0.2)
            
    ax.set_xlabel('Spike Amplitude (mV)', fontsize=15)
    ax.set_ylabel('% of population', fontsize=15)
    # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
    ax.legend(fontsize=15,  framealpha = 0.1, frameon = False)
    return fig

def set_axis_thickness(ax, linewidth  = 1):
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)
        
def get_str_of_nuclei_FR(nuclei_dict, name_list):
    As = ''
    for name in name_list:
        As += name + '_' + str(np.round(nuclei_dict[name][0].basal_firing, 2)).replace('.', '-') + '_'
    return As[:-1]

def get_str_of_A_with_state(name_list, Act, state):
    As = ''
    for name in name_list:
        As += name + '_' + str(np.round(Act[state][name], 2)).replace('.', '-') + '_'
    return As[:-1]

def raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=0, plot_end=None, tick_label_fontsize=18,
                            labelsize=15, title_fontsize=15, lw=1, linelengths=1, n_neuron=None, include_title=True, set_xlim=True,
                            axvspan=False, span_start=None, span_end=None, axvspan_color='lightskyblue', ax_label=False, neurons =[],
                            ylabel_x = 0.03, include_nuc_name = True, name_list = None, remove_ax_frame = True, y_tick_length = 2, 
                            x_tick_length = 5, axis_linewidth = 0.5):
    
    if name_list == None:
        name_list = list(nuclei_dict.keys())
    if outer == None:
        fig = plt.figure(figsize=(10, 8))
        outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)[0]

    inner = gridspec.GridSpecFromSubplotSpec(len(nuclei_dict), 1,
                    subplot_spec=outer, wspace=0.1, hspace=0.1)
    if include_title:
        ax = plt.Subplot(fig, outer)
        ax.set_title(title, fontsize=15)
        ax.axis('off')
        
    for j, name in enumerate(name_list):
        nucleus = nuclei_dict[name][0]
        
        if plot_end == None:
            plot_end = len(nucleus.pop_act)
            
        ax = plt.Subplot(fig, inner[j])
        
        if n_neuron == None:
            n_neuron = nucleus.n
            
        if neurons == []:
            neurons = np.random.choice(nucleus.n, n_neuron, replace=False)
            
        spikes_sparse = create_sparse_matrix(nucleus.spikes[neurons, :], end=(
                                            plot_end / dt), start=(plot_start / dt)) * dt
        if set_xlim:
            xlim = [plot_start, plot_end]
            
        else: 
            xlim = None
            
        c_dict = color_dict.copy()
        c_dict['Arky'] = 'darkorange'
        ax = raster_plot(spikes_sparse, nucleus.name, c_dict,  ax=ax, tick_label_fontsize=tick_label_fontsize, 
                         title_fontsize=title_fontsize, linelengths=linelengths, lw=lw, xlim=xlim,
                        axvspan=axvspan, span_start=span_start, span_end=span_end, axvspan_color=axvspan_color,
                        include_nuc_name = include_nuc_name, remove_ax_frame = remove_ax_frame, 
                        y_tick_length = y_tick_length, x_tick_length = x_tick_length)
        fig.add_subplot(ax)
        set_axis_thickness(ax, linewidth  = axis_linewidth)
        rm_ax_unnecessary_labels_in_subplots(j, len(nuclei_dict), ax)

    if ax_label:
        fig.text(0.5, 0.03, 'time (ms)', ha='center',
                 va='center', fontsize=labelsize)
        fig.text(ylabel_x, 0.5, 'neuron', ha='center', va='center',
                 rotation='vertical', fontsize=labelsize)
    return fig

def raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=0, plot_=None, tick_label_fontsize=18,
                            labelsize=15, title_fontsize=15, lw=1, linelengths=1, n_neuron=None, include_title=True, set_xlim=True,
                            axvspan=False, span_start=None, span_end=None, axvspan_color='lightskyblue', ax_label=False, n = 1000, 
                            t_transition = None, t_sim = None, ylabel_x = 0.03, include_nuc_name = True):
    
    neurons = np.random.choice(n, n_neuron, replace=False )
    
    fig_state_1 = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start, plot_end = t_transition - 10,
                            labelsize = labelsize, title_fontsize = title_fontsize, lw  = lw, linelengths = linelengths, n_neuron = n_neuron, include_title = include_title, set_xlim=True,
                            neurons = neurons, ax_label = True, tick_label_fontsize = tick_label_fontsize, ylabel_x = ylabel_x, include_nuc_name=include_nuc_name)
    
    fig_state_2 = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = t_sim - (t_transition-10-plot_start), plot_end = t_sim,
                            labelsize = labelsize, title_fontsize = title_fontsize, lw  = lw, linelengths = linelengths, n_neuron = n_neuron, include_title = include_title, set_xlim=True,
                            axvspan = True, span_start = t_transition, span_end = t_sim, axvspan_color = axvspan_color, include_nuc_name=include_nuc_name,
                            neurons = neurons, ax_label = True, tick_label_fontsize = tick_label_fontsize, ylabel_x = ylabel_x)
    
    return fig_state_1, fig_state_2

def find_FR_sim_vs_FR_expected(FR_list, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict, A, A_mvt, D_mvt, t_mvt):
    ''' simulated FR vs. what we input as the desired firing rate'''

    nucleus_name = list(nuclei_dict.keys()); m = len(FR_list)
    firing_prop = {k: {'firing_mean': np.zeros((m, len(nuclei_dict[nucleus_name[0]]))), 'firing_var': np.zeros(
        (m, len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
    i = 0
    for FR in FR_list:

        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.clear_history()
                # nucleus.reset_ext_pop_properties(poisson_prop,dt)
                nucleus.basal_firing = FR
                nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list, dt)
        nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.pop_act = moving_average_array(nucleus.pop_act, 50)
                firing_prop[nucleus.name]['firing_mean'][i, nucleus.population_num -
                    1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                firing_prop[nucleus.name]['firing_var'][i, nucleus.population_num -
                    1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                print(nucleus.name, np.round(np.average(nucleus.FR_ext), 3),
                    'FR=', firing_prop[nucleus.name]['firing_mean'][i, nucleus.population_num-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, nucleus.population_num-1], 2))
        i += 1
    return firing_prop


def find_ext_input_reproduce_nat_firing(tuning_param, list_1, list_2, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict):
    ''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''

    nucleus_name = list(nuclei_dict.keys()); m = len(list_1); n = len(list_2)
    firing_prop = {k: {'firing_mean': np.zeros((m, n, len(nuclei_dict[nucleus_name[0]]))), 'firing_var': np.zeros(
        (m, n, len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
    ext_firing = np.zeros((m, n, 2))
    loss = np.zeros((m, n))
    i = 0
    for g_1 in list_1:
        j = 0
        for g_2 in list_2:

            ext_firing[i, j] = [g_1, g_2]
            poisson_prop[nucleus_name[0]][tuning_param] = g_1
            poisson_prop[nucleus_name[1]][tuning_param] = g_2
            for nuclei_list in nuclei_dict.values():
                for nucleus in nuclei_list:
                    nucleus.clear_history()
                    nucleus.reset_ext_pop_properties(poisson_prop, dt)
            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            for nuclei_list in nuclei_dict.values():
                for nucleus in nuclei_list:

                    firing_prop[nucleus.name]['firing_mean'][i, j, nucleus.population_num -
                        1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                    loss[i, j] += (firing_prop[nucleus.name]['firing_mean']
                                   [i, j, nucleus.population_num-1] - nucleus.basal_firing)**2
                    firing_prop[nucleus.name]['firing_var'][i, j, nucleus.population_num -
                        1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                    print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop, 3),
                        'FR=', firing_prop[nucleus.name]['firing_mean'][i, j, nucleus.population_num-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, j, nucleus.population_num-1], 2))
            print('loss =', loss[i, j])
            j += 1
        i += 1
    return loss, ext_firing, firing_prop


def find_ext_input_reproduce_nat_firing_3_pop(tuning_param, list_1, list_2, list_3, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict):
    ''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''

    nucleus_name = list(nuclei_dict.keys()); m = len(
        list_1); n = len(list_2); k = len(list_3)
    firing_prop = {kk: {'firing_mean': np.zeros((m, n, k, len(nuclei_dict[nucleus_name[0]]))), 'firing_var': np.zeros(
        (m, n, k, len(nuclei_dict[nucleus_name[0]])))} for kk in nucleus_name}
    ext_firing = np.zeros((m, n, k, 3))
    loss = np.zeros((m, n, k))
    i = 0
    for g_1 in list_1:
        j = 0
        for g_2 in list_2:
            l = 0
            for g_3 in list_3:

                ext_firing[i, j, l] = [g_1, g_2, g_3]
                poisson_prop[nucleus_name[0]][tuning_param] = g_1
                poisson_prop[nucleus_name[1]][tuning_param] = g_2
                poisson_prop[nucleus_name[2]][tuning_param] = g_3
                for nuclei_list in nuclei_dict.values():
                    for nucleus in nuclei_list:
                        nucleus.clear_history()
                        nucleus.reset_ext_pop_properties(poisson_prop, dt)
                nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
                for nuclei_list in nuclei_dict.values():
                    for nucleus in nuclei_list:
                        firing_prop[nucleus.name]['firing_mean'][i, j, l, nucleus.population_num -
                            1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                        loss[i, j, l] += (firing_prop[nucleus.name]['firing_mean']
                                          [i, j, l, nucleus.population_num-1] - nucleus.basal_firing)**2
                        firing_prop[nucleus.name]['firing_var'][i, j, l,
                            nucleus.population_num-1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                        print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop, 4),
                            'FR=', firing_prop[nucleus.name]['firing_mean'][i, j, l, nucleus.population_num-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, j, l, nucleus.population_num-1], 2))
                print('loss =', loss[i, j, l])
                l += 1
            j += 1
        i += 1
    return loss, ext_firing, firing_prop


def find_ext_input_reproduce_nat_firing_relative(tuning_param, list_1, poisson_prop, receiving_class_dict, t_list, dt, nuclei_dict):
    ''' find the proper set of parameters for the external population of each nucleus that will give rise to the natural firing rates of all'''

    nucleus_name = list(nuclei_dict.keys()); m = len(list_1)
    firing_prop = {k: {'firing_mean': np.zeros((m, len(nuclei_dict[nucleus_name[0]]))), 
                       'firing_var': np.zeros((m, len(nuclei_dict[nucleus_name[0]])))} for k in nucleus_name}
	
    ext_firing = np.zeros((m, len(nucleus_name)))
    loss = np.zeros(m)
    i = 0
    for g in list_1:
        for j in range(len(nucleus_name)):
            poisson_prop[nucleus_name[j]][tuning_param] = g * \
			    nuclei_dict[nucleus_name[j]][0].FR_ext
            ext_firing[i, j] = poisson_prop[nucleus_name[j]][tuning_param]
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                nucleus.clear_history()
                nucleus.reset_ext_pop_properties(poisson_prop, dt)
        nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
        for nuclei_list in nuclei_dict.values():
            for nucleus in nuclei_list:
                firing_prop[nucleus.name]['firing_mean'][i, nucleus.population_num -
				    1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                loss[i] += (firing_prop[nucleus.name]['firing_mean']
				            [i, nucleus.population_num-1] - nucleus.basal_firing)**2
                firing_prop[nucleus.name]['firing_var'][i, nucleus.population_num -
				    1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop, 3),
					'FR=', firing_prop[nucleus.name]['firing_mean'][i, nucleus.population_num-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, nucleus.population_num-1], 2))
            print('loss = ', loss[i])
        i += 1
        return loss, ext_firing, firing_prop


def dopamine_effect(threshold, G, dopamine_percentage):
	''' Change the threshold and synaptic weight depending on dopamine levels'''
	threshold['Str'] = -0.02 + 0.03 * \
	    (1-(1.1/(1+0.1*np.exp(-0.03*(dopamine_percentage - 100)))))
	G[('Str', 'Ctx')] = 0.75/(1+np.exp(-0.09*(dopamine_percentage - 60)))
	return threshold, G


def possion_spike_generator(n_pop, n_sending, r, dt):
	'''generate a times series of possion spikes for a population of size n, with firing rate r'''
	x = np.random.rand(n_pop, n_sending)
	# poisson_thresh = r*dt
	# temp = (1-np.exp(-r*dt))
	temp = r*dt
	poisson_thresh = np.repeat(temp.reshape(-1, 1), n_sending, axis=1)
	spikes = np.where(x <= poisson_thresh)
	# print(spikes)
	x = x*0
	x[spikes] = 1  # spike with probability of rdt
	# x[~spikes] = 0
	return x.astype(int)

def pad_high_res_spacing_with_linspace(start_before, mid_start, n_before, mid_end, end_after,  n_after, n_high_res):
    
    linspace_before = np.linspace(start_before, mid_start, n_before)
    linspace_after = np.linspace(mid_end, end_after, n_after)
    high_res = spacing_with_high_resolution_in_the_middle(n_high_res, mid_start, mid_end).reshape(-1,)
    
    return np.concatenate((linspace_before, high_res, linspace_after), axis  = 0)

def pad_high_res_spacing_with_arange(start_before, mid_start, bin_before, mid_end, end_after,  bin_after, n_high_res):
    
    linspace_before = np.linspace(start_before, mid_start, int( ( mid_start - start_before) / bin_before) )
    linspace_after = np.linspace(mid_end, end_after, int( ( end_after - mid_end) / bin_after) )
    high_res = spacing_with_high_resolution_in_the_middle(n_high_res, mid_start, mid_end).reshape(-1,)
    
    return np.concatenate((linspace_before, high_res, linspace_after), axis  = 0)

def three_different_linspace_arrays(start_before, mid_start, n_before, mid_end, end_after,  n_after, n_mid):
    
    linspace_before = np.linspace(start_before, mid_start, n_before)
    linspace_after = np.linspace(mid_end, end_after, n_after)
    linspace_mid = np.linspace(mid_start, mid_end, n_mid)
    
    return np.concatenate((linspace_before, linspace_mid, linspace_after), axis  = 0)

def spacing_with_high_resolution_in_the_middle(n_points, start, end):
	'''return a series with lower spacing and higher resolution in the middle'''
    
	R = (start - end) / 2
	x = R * np.linspace(-1, 1, n_points)
	y = np.sqrt(R ** 2 - x ** 2)
	half_y = y[: len(y) // 2]
	diff = - np.diff(np.flip(half_y))
	series = np.concatenate((half_y, np.cumsum(diff) + y[len(y) // 2])) + start
# 	if len(series) < n_points: # doesn't work with odd number of points!!!!!!
# 		series = np.concatenate((series, series[-1]))

	return series.reshape(-1, 1)


def noise_generator(amplitude, std, n, dt, sqrt_dt, tau = 0, noise_dt_before = 0):
        
    return 1/sqrt_dt * amplitude * np.random.normal(0, std, n).reshape(-1, 1)

def OU_noise_generator(amplitude, std, n, dt, sqrt_dt, tau= 10,  noise_dt_before = 0):

    ''' Ornstein-Uhlenbeck process as time correlated noise generator'''
    
    noise_prime = -noise_dt_before / tau + \
                   std * np.sqrt(2 / tau) * noise_generator(amplitude, 1, n, dt, sqrt_dt)
    noise = fwd_Euler(dt, noise_dt_before, noise_prime)
    
    return noise

def plot_fft_spectrum(peak_freq, f, pxx, N, ax=None, c='navy', label='fft', figsize=(6, 5), 
                      include_beta_band_in_legend=False, tick_label_fontsize = 18, normalize = True,
                      include_peak_f_in_legend = True, plot_sig_thresh = False, 
                      min_f = 0, max_f = 250, n_std_thresh = 2, legend_loc = 'center right'):
    
    fig, ax = get_axes(ax, figsize=figsize)
	# plt.semilogy(freq[:N//2], f[:N//2])
    ylabel = 'FFT power'
    
    if include_peak_f_in_legend :    
	    label += ' f =' + str(round(peak_freq, 1)) + ' Hz'
        
    if include_beta_band_in_legend:
        beta_band_power = beta_bandpower(f, pxx)
        label += ' ' + r'$\overline{P}_{\beta}=$' + str(round(beta_band_power, 3))
        
    if normalize :
		
        pxx = normalize_PSD(pxx, f) * 100
        ylabel = 'Norm. Power ' + r'$(\times 10^{-2})$'
        
    
    ax.plot(f, pxx, c=c, label=label, lw=1.5)
    ax.set_xlabel('frequency (Hz)', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend(fontsize=15, loc=legend_loc, framealpha=0.1, frameon=False)
	# ax.tick_params(axis='both', which='major', labelsize=10)
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='x', nbins=5)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    remove_frame(ax)
    
    if plot_sig_thresh:
        
        sig_thresh = cal_sig_thresh_1d(f, pxx, min_f = min_f, max_f = max_f, n_std_thresh = n_std_thresh)
        ax.axhline(sig_thresh,0, max_f, ls = '--', color = c)

def normalize_PSD(pxx, f):
    
    ''' Normalize PSD to the AUC '''
    AUC = np.trapz(pxx, f)
    pxx = pxx / AUC 
    
    return pxx

def normalize_PSD_2d(pxx, f, axis = 0):
    print(pxx.shape, f.shape)
    ''' Normalize PSD to the AUC '''
    AUC = np.trapz(pxx, f, axis = axis)
    pxx = pxx / AUC 
    
    return pxx

def freq_from_fft(sig, dt, plot_spectrum=False, ax=None, c='navy', label='fft', figsize=(6, 5), 
                  method='rfft', n_windows=6, include_beta_band_in_legend=False, max_f = 200, min_f = 0,
                  normalize_spec = True, include_peak_f_in_legend = True, plot_sig_thresh = False, 
                  legend_loc = 'upper right',  n_std_thresh = 2):
	"""
	Estimate frequency from peak of FFT
	"""

	N = len(sig)

	if N == 0:
		return 0

	else:
		if method not in ["rfft", "Welch"]:
			raise ValueError("method must be either 'rff', or 'Welch'")
		if method == 'rfft':
			f, pxx, peak_freq = freq_from_rfft(sig, dt,  N)
		if method == 'Welch':
			f, pxx, peak_freq = freq_from_welch(sig, dt, n_windows=n_windows)
            
		# Find the peak and interpolate to get a more accurate peak

		if plot_spectrum:
			plot_fft_spectrum(peak_freq, f, pxx, N, ax=ax, c=c, label=label,
			                  figsize=figsize, normalize = normalize_spec,
                              include_beta_band_in_legend=include_beta_band_in_legend,
                              include_peak_f_in_legend = include_peak_f_in_legend, 
                              plot_sig_thresh = plot_sig_thresh, legend_loc = legend_loc,
                              min_f = min_f, max_f = max_f,  n_std_thresh = n_std_thresh)
		return f[f < max_f], pxx[f < max_f], peak_freq 



def freq_from_rfft(sig, dt, N):
    
	'''Estimate frequency with rfft method '''
    
	rf = rfft(sig)
	f = fftfreq(N, dt)[:N//2]
	pxx = np.abs(rf[:N//2]) ** 2
	# Just use this for less-accurate, naive version
	peak_freq = f[np.argmax(abs(rf[: N // 2]))]
    
	return f, pxx, peak_freq


def freq_from_welch(sig, dt, n_windows=6):
    
	"""
	Estimate frequency with Welch method
	"""
    
	fs = 1 / dt
	f, pxx = signal.welch(sig, axis=0, fs=fs, nperseg=int(len(sig) / n_windows))
	peak_freq = f[np.argmax(pxx)]
    
	return f, pxx, peak_freq

def autocorr_1d(x, method = 'numpy', mode = 'same'):
    
    if method == 'numpy':
        result = np.correlate(x, x, mode= mode)
        
    elif method == 'fft':
        result = signal.correlate(x, x, mode= mode, method = 'fft')
        
    return result[ result.size//2 : ]

def cut_PSD_2d(f, pxx, max_f = 250):
    
    f_to_keep_ind = f < max_f
    pxx = pxx[:, f_to_keep_ind] 
    
    return f[f_to_keep_ind] , pxx

def cut_PSD_1d(f, pxx, max_f = 250):
    
    f_to_keep_ind = f < max_f
    pxx = pxx[f_to_keep_ind] 
    
    return f[f_to_keep_ind] , pxx

def autocorr_2d(x):
    
  """FFT based autocorrelation function, which is faster than numpy.correlate"""
  
  # x is supposed to be an array of sequences, of shape (totalelements, length)
  length = x.shape[1]
  l = length * 2 - 1
  fftx = fft(x, n= l, axis=1)
  ret = ifft(fftx * np.conjugate(fftx), axis=1).real
  ret = fftshift(ret, axes=1)
  autocorr = ret[:, ret.shape[1]//2 : ] # take the latter half of data for positive lags
  return autocorr

def freq_from_welch_2d(sig, dt, n_windows=6):
    
 	"""
 	Estimate frequency with Welch method for all the rows of a 2d array
 	"""
     
 	fs = 1 / dt 
 	f, pxx = signal.welch(sig, axis=1, fs=fs, nperseg= int(sig.shape[1] / n_windows))
 	peak_freq = f[np.argmax(pxx, axis = 1)]
     
 	return f, pxx, peak_freq
 
def get_fft_autc_spikes(nucleus, dt, window_mov_avg, n_window_welch):
    
    spks = moving_average_array_2d(nucleus.spikes.copy(), int(window_mov_avg / dt))
    autc = autocorr_2d(spks)
    f, pxx, peak_f = freq_from_welch_2d(autc, dt/1000, n_windows= n_window_welch)
    
    return f, pxx, peak_f

def significance_of_oscil_all_neurons(nucleus, dt, window_mov_avg = 10, max_f = 250, min_f_sig_thres = 0,
                                      n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2,
                                      min_f_AUC_thres = 7,  PSD_AUC_thresh = 10**-5, filter_based_on_AUC_of_PSD = False):
    """ 
        Rerturn indice of the neurons that have <n_pts_above_thresh> points above significance threshold level
        in their PSD of autocorrelogram
    """
    f, pxx, peak_f = get_fft_autc_spikes(nucleus, dt, window_mov_avg, n_window_welch)
    f, pxx = cut_PSD_2d(f, pxx, max_f = max_f)
    signif_thresh =  cal_sig_thresh_2d(f, pxx, min_f = min_f_sig_thres, max_f = max_f, n_sd_thresh = n_sd_thresh)
    entrained_neuron_ind  = check_significance_neuron_autc_PSD( signif_thresh, f, pxx, nucleus.n, n_pts_above_thresh = n_pts_above_thresh,
                                                   fmin= min_f_AUC_thres, fmax = max_f, PSD_AUC_thresh = PSD_AUC_thresh, 
                                                   filter_based_on_AUC_of_PSD = filter_based_on_AUC_of_PSD)
    print(nucleus.name, len(entrained_neuron_ind), ' out of ', nucleus.n , ' entrained to oscillatin')
    return entrained_neuron_ind

def cal_sig_thresh_2d(f, pxx, n_std_thresh = 2, min_f = 0, max_f = 250):
    
    ind_f = np.logical_and( min_f < f, f < max_f)
    return ( np.average( pxx[: , ind_f], axis = 1) + 
             n_std_thresh * 
             np.std( pxx [:, ind_f] , axis = 1) )

def cal_sig_thresh_1d(f, pxx, n_std_thresh = 2, min_f = 0, max_f = 250):
    
    ind_f = np.logical_and( min_f < f, f < max_f)
    return ( np.average( pxx[ ind_f]) + 
             n_std_thresh * 
             np.std( pxx [ ind_f] ) )

def filter_based_on_PSD_AUC(f, pxx, fmin= 0, fmax = 200, PSD_AUC_thresh = 10 ** -5):
    PSD_integral = bandpower_2d_pxx(f, pxx, fmin, fmax)

    return np.where(PSD_integral > PSD_AUC_thresh)[0]
    
def check_significance_neuron_autc_PSD(signif_thresh, f, pxx, n, n_pts_above_thresh = 2, 
                           fmin= 8, fmax = 200, PSD_AUC_thresh = 10**-5, filter_based_on_AUC_of_PSD = False):
    
    if filter_based_on_AUC_of_PSD:
        above_PSD_AUC_thresh = filter_based_on_PSD_AUC(f, pxx, fmin= fmin, fmax = fmax, 
                                                   PSD_AUC_thresh = PSD_AUC_thresh)
        pxx[get_complement_ind(above_PSD_AUC_thresh, n), : ] = 0 # to remove the neurons that don't pass the AUC citeria
    above_thresh_neuron_ind, above_thresh_freq_ind = np.where(pxx >= signif_thresh.reshape(-1,1))

    n_above_thresh_each_neuron = np.bincount(above_thresh_neuron_ind)
    entrained_neuron_ind = np.where( n_above_thresh_each_neuron >= n_pts_above_thresh )[0]
    bool_neurons = np.zeros((n), dtype = bool)
    
    for neuron in entrained_neuron_ind:
        
        ind = above_thresh_neuron_ind == neuron
        freq_above_thresh = above_thresh_freq_ind[ind]
        longest_seq =longest_consecutive_chain_of_numbers( freq_above_thresh)
        if len(longest_seq) >= n_pts_above_thresh  :
            bool_neurons[neuron] = True

    return np.where(bool_neurons)[0]


def check_significance_of_PSD_peak(f, pxx,  n_std_thresh = 2, min_f = 0, max_f = 250, n_pts_above_thresh = 3, 
                                   ax = None, legend = 'PSD', c = 'k', if_plot = False, AUC_ratio_thresh = 0.8,
                                   xlim = [0, 80], name = ''):
    
    ''' Check significance of a peak in PSD by checking if the 
        <n_pts_above_thresh> consecutive points exceeds <n_std> 
        times the std of the rest of the PSD '''
    
    f, pxx = cut_PSD_1d(f, pxx, max_f = max_f)
    signif_thresh = cal_sig_thresh_1d(f, pxx, min_f = min_f, max_f = max_f, n_std_thresh = n_std_thresh)
    # above_thresh_ind = np.where(pxx >= signif_thresh)[0]
    
    pxx_rel_sig = pxx - signif_thresh
    # fig, ax = plt.subplots(1,1)
    # ax.plot(f, pxx_rel_sig.clip(min  = 0), '-o', label = legend, c = 'b')
    AUC_above_sig_thresh = np.trapz(pxx_rel_sig.clip(min  = 0), f)
    AUC = np.trapz(pxx, f)
    AUC_ratio = AUC_above_sig_thresh / AUC
    
    # longest_seq_abv_thresh = longest_consecutive_chain_of_numbers( above_thresh_ind)
    
    if if_plot:
        fig, ax = get_axes (ax)
        ax.plot(f, pxx, '-o', label = legend, c = c)
        ax.axhline(signif_thresh, min_f, max_f, ls = '--', color = c)
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_title(name)
    
    print( "AUC ratio = {}".format( AUC_ratio ), AUC_ratio > AUC_ratio_thresh)

    if AUC_ratio > AUC_ratio_thresh:

        return True
    
    else:
        return False
    
def oscillation_index(f,pxx, max_f, start_range, end_range):
    ''' calculate the oscillation index as the 
        AUC in f_range normalized to AUC of [0,max_f]
        
    '''

    ind = np.logical_and( f >= start_range, f <= end_range)
    AUC_range = np.trapz(pxx [ind], f[ind])
    AUC_total = np.trapz(pxx[f < max_f], f[ f < max_f])
    
    return AUC_range/ AUC_total

def get_complement_ind(indices, n):
    
    ''' return the complement of the indices from an array of 0 to n'''
    
    mask = np.zeros((n), dtype=bool)
    mask[indices] = True
    
    return np.where(~mask)[0]    


def longest_consecutive_chain_of_numbers(array ):
    
    return  max(np.split(
                        array , 
                         np.where(np.diff( array ) != 1)[0]+1
                         ), 
                key=len).tolist()  

def cal_population_activity_all_nuc_all_t(nuclei_dict, dt):
    ''' calculate the average FR as average of all spikes for all simulation duration and nuclei'''
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            nucleus.cal_population_activity_all_t(dt)
            
    return nuclei_dict


def run_rate_model(receiving_class_dict, t_list, dt, nuclei_dict):
    
    ''' run temporal dynimcas of the rate model'''
    
    for t in t_list:
        
        for nuclei_list in nuclei_dict.values():
            
            for k, nucleus in enumerate(nuclei_list):

		#        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement
                mvt_ext_inp = np.ones((nucleus.n, 1)) * \
				                      nucleus.external_inp_t_series[t]  # movement added
                nucleus.calculate_input_and_inst_act(
					    t, dt, receiving_class_dict[(nucleus.name, str(k + 1))], mvt_ext_inp)
                nucleus.update_output(dt)

    return nuclei_dict

def run_spiking_model(receiving_class_dict, t_list, dt, nuclei_dict):
    
    ''' run temporal dynimcas of the spiking neural net'''
    
    for t in t_list:
        
        for nuclei_list in nuclei_dict.values():
            
            for k, nucleus in enumerate(nuclei_list):

                nucleus.solve_IF(t, dt, receiving_class_dict[(
					    nucleus.name, str(k + 1))])
    
    nuclei_dict = cal_population_activity_all_nuc_all_t(nuclei_dict, dt)

    return nuclei_dict

run_dyn_func_dict = {'rate': run_rate_model, 'spiking': run_spiking_model}

def run(receiving_class_dict, t_list, dt, nuclei_dict):
    
    ''' run temporal dynamics based on the model'''
    
    start = timeit.default_timer()

    model = nuclei_dict[ list(nuclei_dict.keys()) [0]] [0]. neuronal_model
    print('running ', model, 'model...')
    nuclei_dict = run_dyn_func_dict[model](receiving_class_dict, t_list, dt, nuclei_dict)
    
    stop = timeit.default_timer()
    print("t = ", stop - start)
    
    return nuclei_dict



def reset_connec_ext_input_DD(nuclei_dict, K_DD, N, N_real, A_DD, A_mvt, D_mvt, t_mvt, t_list, dt):
    
	K = calculate_number_of_connections(N, N_real, K_DD)
	for nuclei_list in nuclei_dict.values():
		for nucleus in nuclei_list:
			nucleus.set_connections(K, N)
			# nucleus.set_synaptic_weights(G_DD)
			nucleus.set_ext_input(A_DD, A_mvt, D_mvt, t_mvt, t_list, dt)


def run_transition_to_DA_depletion(receiving_class_dict, t_list, dt, nuclei_dict, DD_init_filepaths, K_DD, N, N_real, 
                                   A_DD, A_mvt, D_mvt, t_mvt, t_transition=None):
    
	''' Transition to DD single neuron FR_ext initialization'''
    
	if t_transition == None:
		t_transition = t_list[int(len(t_list) / 3)]
        
	start = timeit.default_timer()

	for t in t_list:

		for nuclei_list in nuclei_dict.values():
			k = 0
			for nucleus in nuclei_list:
				k += 1
		#        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement
				mvt_ext_inp = np.ones((nucleus.n, 1)) * \
				                      nucleus.external_inp_t_series[t]  # movement added
				if nucleus.neuronal_model == 'rate':  # rate model
					nucleus.calculate_input_and_inst_act(
					    t, dt, receiving_class_dict[(nucleus.name, str(k))], mvt_ext_inp)
					nucleus.update_output(dt)
				if nucleus.neuronal_model == 'spiking':  # QIF
					nucleus.solve_IF(t, dt, receiving_class_dict[(
					    nucleus.name, str(k))], mvt_ext_inp)

		if t == t_transition:
			set_init_all_nuclei(nuclei_dict, filepaths=DD_init_filepaths)
			reset_connec_ext_input_DD(nuclei_dict, K_DD, N,
			                          N_real, A_DD, A_mvt, D_mvt, t_mvt, t_list, dt)

	stop = timeit.default_timer()
	print("t = ", stop - start)
	return nuclei_dict


def run_transition_to_movement(receiving_class_dict, t_list, dt, nuclei_dict, mvt_init_filepaths, 
							   N, N_real, A_mvt, D_mvt, t_mvt, t_transition=None):
	
	''' Transition to mvt single neuron FR_ext initialization'''

	if t_transition == None:
		t_transition = t_list[int(len(t_list) / 3)]
	start = timeit.default_timer()

	for t in t_list:

		for nuclei_list in nuclei_dict.values():
			k = 0
			for nucleus in nuclei_list:
				k += 1
		#        mvt_ext_inp = np.zeros((nucleus.n,1)) # no movement
				mvt_ext_inp = np.ones((nucleus.n, 1)) * \
				                      nucleus.external_inp_t_series[t]  # movement added
				if nucleus.neuronal_model == 'rate':  # rate model
					nucleus.calculate_input_and_inst_act(
					    t, dt, receiving_class_dict[(nucleus.name, str(k))], mvt_ext_inp)
					nucleus.update_output(dt)
				if nucleus.neuronal_model == 'spiking':  # QIF
					nucleus.solve_IF(t, dt, receiving_class_dict[(
					    nucleus.name, str(k))], mvt_ext_inp)

		if t == t_transition:
			set_init_all_nuclei(nuclei_dict, filepaths=mvt_init_filepaths)

			for nuclei_list in nuclei_dict.values():
				for nucleus in nuclei_list:
					nucleus.set_ext_input(A_mvt, A_mvt, D_mvt, t_mvt, t_list, dt)
	stop = timeit.default_timer()
	print("t = ", stop - start)
	return nuclei_dict


def run_transition_state_collective_setting(G, noise_variance,noise_amplitude,  path, receiving_class_dict, 
                                            receiving_pop_list, t_list, dt, nuclei_dict, Act, state_1, state_2, 
                                            K_all, N, N_real, A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, 
                                            end_of_nonlinearity, t_transition=None):
    
    ''' Transition from <state_1> to <state_2> collective FR_ext setting'''
    
    if t_transition == None:
        t_transition = t_list[int(len(t_list) / 3)]
        
    start = timeit.default_timer()
    
    for t in t_list:
            
        for nuclei_list in nuclei_dict.values():
            for k, nucleus in enumerate(nuclei_list):
                
                nucleus.solve_IF(t, dt, receiving_class_dict[(
					    nucleus.name, str(k + 1))])

        if t == t_transition:
            print('transitioning..')
            nuclei_dict = change_basal_firing_all_nuclei(Act[state_2], nuclei_dict)
            nuclei_dict = change_state_all_nuclei(state_2, nuclei_dict)
            nuclei_dict = change_noise_all_nuclei( nuclei_dict, noise_variance[state_2], noise_amplitude)
            
            if 'DD' in state_2 and ('Proto', 'Proto') in list(G.keys()): 
                nuclei_dict['Proto'][0].synaptic_weight[('Proto', 'Proto')] *= 2 

            receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_2], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_all[state_2], 
                                                                   receiving_pop_list, nuclei_dict, t_list,
                                                                   all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, 
                                                                   end_of_nonlinearity=end_of_nonlinearity,
                                                                   set_FR_range_from_theory=False, method='collective',  save_FR_ext=False,
                                                                   use_saved_FR_ext = True, normalize_G_by_N=False, state=state_2)
            
    nuclei_dict = cal_population_activity_all_nuc_all_t(nuclei_dict, dt)

    stop = timeit.default_timer()
    print("t = ", stop - start)
    return nuclei_dict


def cal_average_activity(nuclei_dict, n_run, avg_act):
    
    ''' calculate average population firing activity of multiple runs
    '''
    
    for nuclei_list in nuclei_dict.values():
            for k,nucleus in enumerate( nuclei_list) :
                avg_act[ nucleus.name][:, k] += nucleus.pop_act/n_run
                
    return avg_act



def plot_fr_response_from_experiment(FR_df, filename, color_dict, xlim = None, ylim = None, 
                     stim_duration = 10, ax = None, time_shift_dict = None, sheet_name_extra = '',
                     legend_loc = 'upper right'):
    
    fig, ax = get_axes( ax )
    
    if time_shift_dict == None:
        time_shift_dict = { key: 0 for key in list(FR_df.keys())}

    for name in list(FR_df.keys()):
        
        name_adj = name.split('_')[-1].replace( sheet_name_extra, '')
        time = FR_df[name]['Time'] * 1000 - time_shift_dict[name_adj]
        fr = FR_df[name].drop(columns = ['Time'])
        fr_mean = fr.mean(axis=1)
        # print( name, ' rest firing rate = ', np.average( fr_mean[time < 0] ))
        fr_std = fr.std(axis=1)
        n_cells = len(FR_df[name].columns) - 1
        ax.plot(time, fr_mean, c = color_dict[name_adj], label = name_adj + ' n =' + str(n_cells))
        ax.fill_between(time, fr_mean - fr_std/ np.sqrt(n_cells), fr_mean + fr_std/ np.sqrt(n_cells), 
                        color = color_dict[name_adj], alpha = 0.1)
        
    title = filename.split('.')[0].split('_')[-1] 
    ax.set_title(title, fontsize = 15)
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

def selective_additive_ext_input(nuclei_dict, list_of_nuc_with_trans_inp, ext_inp_dict):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            if nucleus.name in list_of_nuc_with_trans_inp:
                
                ext_inp = np.random.normal(ext_inp_dict[nucleus.name]['mean'] * np.average( nucleus.rest_ext_input), 
                                            ext_inp_dict[nucleus.name]['sigma'], nucleus.n)
                nucleus.additive_ext_input(ext_inp)
                
    return nuclei_dict


def run_with_transient_external_input_including_transmission_delay(receiving_class_dict, t_list, dt, nuclei_dict, rest_init_filepaths,
																   A, A_trans,  syn_trans_delay_dict, transient_init_filepaths = None,
																   t_transient=10, duration=10, inp_method='reset', ext_inp_dict = None):
    
    '''
    		run normaly til "t_transient" then exert an external transient input to the concerned nuclei then resume to normal state until the end of simulation.
    		Where the syn_trans_delay_dict contains the synaptic transmission delays of the input to different nuclei (e.g. MC to STN and MC to D2)
    '''
    min_syn_trans_delays = min(syn_trans_delay_dict, key=syn_trans_delay_dict. get)
    
    
    # synaptic trans delay relative to the nucleus with minimum delay.
    t_start_inp_dict = {k: t_transient + v -
	    syn_trans_delay_dict[min_syn_trans_delays] for k, v in syn_trans_delay_dict.items()}
	# synaptic trans delay relative to the nucleus with minimum delay.
    t_end_inp_dict = {k: duration + v for k, v in t_start_inp_dict.items()}

    start = timeit.default_timer()

    for t in t_list:
        # if it's the start of external input to (a) nucleus(ei)
        if t in list(t_start_inp_dict.values()):
            if inp_method == 'reset':
                A_trans_adjusted = A.copy()
                ### change only the firing rate of the nucleus with external input to it.
                A_trans_adjusted[get_corr_key_to_val(t_start_inp_dict, t)[0]] = A_trans[get_corr_key_to_val(t_start_inp_dict, t)[0]]
             
                selective_reset_ext_input(nuclei_dict, transient_init_filepaths, 
                                         get_corr_key_to_val(t_start_inp_dict, t), 
                                         A_trans_adjusted) 
            elif inp_method == 'add':
                selective_additive_ext_input(nuclei_dict, get_corr_key_to_val(t_start_inp_dict, t), ext_inp_dict)

                    
        # if it's the end of external input to (a) nucleus(ei)
        if t in list( t_end_inp_dict.values() ):
            
            selective_reset_ext_input(nuclei_dict, rest_init_filepaths, 
                                     get_corr_key_to_val(t_end_inp_dict, t), 
                                     A)

        for nuclei_list in nuclei_dict.values():
            for k, nucleus in enumerate(nuclei_list):
                k+=1
                nucleus.solve_IF(t,dt,receiving_class_dict[(nucleus.name,str(k))])
    
    stop = timeit.default_timer()
    print("t = ", stop - start)
    
    return nuclei_dict


def exp_rise_and_decay_transient_ext_inp_ChR2_like( trans_coef, mean_ext_inp, t_list, t_start, tau_rise, tau_decay, dt, n):
    
    ''' return an n by n_timebin matrix as the additive external input filled with 
        a normalized exponential rise and decay with  AUC equal to trans_coef * mean_ext_input
        timing of input is the same for all neurons mimicking ChR2 laser activation population wide
    '''
    
    t_list_trimmed = t_list[ t_start: ]
    
    f = (1- np.exp(-( t_list_trimmed - t_start) / (tau_rise / dt))) * \
        (np.exp(-( t_list_trimmed - t_start) / (tau_decay / dt))) 
        
    f_normalized = f / np.trapz(f, t_list_trimmed)
    ext_inp =  (trans_coef * mean_ext_inp * f_normalized).reshape(-1, 1).T
    # print(ext_inp.shape)
    return  np.repeat( ext_inp, n, axis = 0)

def step_like_norm_dist_transient_ext_inp_ChR2_like(trans_coef, mean_ext_inp, sigma, n, t_list, t_start, t_end):
    
    ''' return an n by n_timebin matrix as the additive external input filled with 
        normally distributed (for neurons) step like external inputs
        timing of input is the same for all neurons mimicking ChR2 laser activation population wide
    '''

    f = np.random.normal(trans_coef * mean_ext_inp, sigma, n ).reshape(-1, 1)
    ext_inp = np.zeros(( n, len (t_list[ t_start:]) ))
    ext_inp[:, :(t_end - t_start) ] = np.repeat(f, t_end - t_start, axis = 1)
    
    return ext_inp

def step_like_norm_dist_transient_ext_inp_projected(trans_coef, mean_ext_inp, sigma, n, t_list, t_start, t_end):
    
    ''' return an n by n_timebin matrix as the additive external input filled with 
        normally distributed (for neurons) step like external inputs
        timing of input is based on transmission delay distribution
    '''

    f = np.random.normal(trans_coef * mean_ext_inp, sigma, n ).reshape(-1, 1)
    # print(t_start)
    # t_size = len (t_list[ t_start:]) 

    ext_inp = np.array( [ np.hstack( ( np.zeros( t_start[i] ),  
                                       np.full( t_end[i] - t_start[i], f[i] ), 
                                       np.zeros(  len(t_list[t_start[i]:]) - ( t_end[i] - t_start[i] ))
                                    ) ) 
                         for i in range(n)])
    print(ext_inp.shape)
    return ext_inp

def exp_rise_and_decay_transient_ext_inp_projected( trans_coef, mean_ext_inp, t_list, t_start, tau_rise, tau_decay, dt, n):
    
    ''' return an n by n_timebin matrix as the additive external input filled with 
        a normalized exponential rise and decay with  AUC equal to trans_coef * mean_ext_input
        timing of input is based on transmission delay distribution
    '''
    
    t_list_trimmed = t_list[ t_start: ]
    
    f = (1- np.exp(-( t_list_trimmed - t_start) / (tau_rise / dt))) * \
        (np.exp(-( t_list_trimmed - t_start) / (tau_decay / dt))) 
        
    f_normalized = f / np.trapz(f, t_list_trimmed)
    ext_inp =  (trans_coef * mean_ext_inp * f_normalized).reshape(-1, 1).T
    # print(ext_inp.shape)
    return  np.repeat( ext_inp, n, axis = 0)




def selective_additive_ext_input_time_series(nuclei_dict, t_list, tau_rise, tau_decay, ext_inp_dict,
                                             t_start_inp_dict, t_end_inp_dict, dt, duration = 10, 
                                             plot = False, method = 'exponential', stim_method = 'ChR2'):
    
    ''' filling the values of the extrernal_inp_t_series with a transient input according to when the stimulus starts
        which is the same for all neurons.
    '''
    
    list_of_nuc_with_trans_inp = list(t_start_inp_dict.keys())

    for name in list_of_nuc_with_trans_inp:
        
        nucleus = nuclei_dict[name][0]

        t_start = t_start_inp_dict[name]
        t_end = t_end_inp_dict[name]
        
        if method == 'exponential':
            
            f_add = exp_rise_and_decay_transient_ext_inp_ChR2_like(ext_inp_dict[name]['mean'],
                                                                   np.average(nucleus.rest_ext_input),
                                                                   t_list, t_start, tau_rise, tau_decay, 
                                                                   dt, nucleus.n)
        elif method == 'step':
            if stim_method == 'ChR2':
            
                f_add = step_like_norm_dist_transient_ext_inp_ChR2_like( ext_inp_dict[nucleus.name]['mean'],
                                                                        np.average( nucleus.rest_ext_input), 
                                                                        ext_inp_dict[nucleus.name]['sigma'], 
                                                                        nucleus.n, t_list, t_start, t_end)
            else:
                
                f_add = step_like_norm_dist_transient_ext_inp_projected( ext_inp_dict[nucleus.name]['mean'],
                                                                        np.average( nucleus.rest_ext_input), 
                                                                        ext_inp_dict[nucleus.name]['sigma'], 
                                                                        nucleus.n, t_list, t_start, t_end)
        # nucleus.external_inp_t_series[ :, t_start:] = f_add
        nucleus.external_inp_t_series = f_add # change ChR2s to return the whole matrix
  
                                                                                
        if plot:
            
            fig, ax = plt.subplots()
            ax.plot(t_list * dt, nucleus.external_inp_t_series)
            ax.axvspan(t_start* dt, (t_start + duration) * dt, alpha=0.2, color='yellow')

    return nuclei_dict

def create_stim_start_end_dict(t_transient, duration, syn_trans_delay_dict):
    
    t_start_inp_dict = {k: t_transient + v 
                        for k, v in syn_trans_delay_dict.items()}
    t_end_inp_dict = {k: duration + v 
                      for k, v in t_start_inp_dict.items()}
    
    return t_start_inp_dict, t_end_inp_dict

def run_with_trans_ext_inp_with_axonal_delay_collective(receiving_class_dict, t_list, dt, nuclei_dict,
                                                        A, syn_trans_delay_dict, tau_rise = 50, tau_decay = 50,
                                                        t_transient=10, duration=10, ext_inp_dict = None, 
                                                        plot = False, ext_inp_method = 'exponential',
                                                        stim_method = 'ChR2'):
    
    '''
    		run normaly til "t_transient" then exert an external transient input ( as an exponential rise and decay) 
            to the concerned nuclei then resume to normal state until the end of simulation.
    		Where the syn_trans_delay_dict contains the synaptic transmission delays of the input to 
            different nuclei (e.g. MC to STN and MC to D2)
    '''
    # min_syn_trans_delays = min(syn_trans_delay_dict, key = syn_trans_delay_dict. get)
    
    # # synaptic trans delay relative to the nucleus with minimum delay.
    # t_start_inp_dict = {k: t_transient + v - syn_trans_delay_dict[min_syn_trans_delays] 
    #                     for k, v in syn_trans_delay_dict.items()}
    

    t_start_inp_dict, t_end_inp_dict = create_stim_start_end_dict(t_transient, duration, syn_trans_delay_dict)
    nuclei_dict = selective_additive_ext_input_time_series(nuclei_dict, t_list, tau_rise, tau_decay, ext_inp_dict,
                                                           t_start_inp_dict, t_end_inp_dict, dt,  
                                                           duration=10, plot = plot, method = ext_inp_method,
                                                           stim_method = stim_method)
    
    nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
    
    return nuclei_dict


def average_multi_run_collective(path, tau, receiving_pop_list, receiving_class_dict, t_list, dt, 
                                 nuclei_dict, A, G, N, N_real, K_real, syn_trans_delay_dict, poisson_prop,  
                                 n_FR, all_FR_list, end_of_nonlinearity, t_transient = 10, duration = 10, 
                                 n_run = 1, A_mvt = None, D_mvt = None, t_mvt = None, ext_inp_dict = None, 
                                 noise_amplitude = None, noise_variance = None, reset_init_dist = True, 
                                 color_dict = None, state = 'rest', exponential_ext_inp = False,
                                 tau_rise = 5, tau_decay = 10, plot = False, ext_inp_method = 'exponential',
                                 stim_method = 'ChR2'):
    
    avg_act = {nuc: np.zeros( ( len(t_list), len(nuclei_dict[nuc]) ) ) 
               for nuc in list( nuclei_dict.keys() ) }
    
    for i in range(n_run):

        nuclei_dict = run_with_trans_ext_inp_with_axonal_delay_collective(receiving_class_dict, 
                                                                            t_list, dt, nuclei_dict,
                                                                            A, syn_trans_delay_dict,
                                                                            t_transient = t_transient, 
                                                                            duration = duration, 
                                                                            ext_inp_dict = ext_inp_dict,
                                                                            tau_rise = tau_rise, 
                                                                            tau_decay = tau_decay, 
                                                                            plot = plot, 
                                                                            ext_inp_method = ext_inp_method,
                                                                            stim_method = stim_method)
        avg_act = cal_average_activity(nuclei_dict, n_run, avg_act)
        
        nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, tau, G, noise_amplitude, noise_variance, A,
                                              A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                              reset_init_dist= reset_init_dist, poisson_prop = poisson_prop, 
                                              normalize_G_by_N= True) 
        
        receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, 
                                                               receiving_pop_list, nuclei_dict,t_list, 
                                                               all_FR_list = all_FR_list , n_FR =n_FR, if_plot = False, 
                                                               end_of_nonlinearity = end_of_nonlinearity, 
                                                               set_FR_range_from_theory = False, method = 'collective', 
                                                               use_saved_FR_ext= True,
                                                               normalize_G_by_N= False, save_FR_ext=False,
                                                               state = state)


        print(i+1,'from',n_run)
        
    return avg_act

def average_multi_run(receiving_class_dict,t_list, dt, nuclei_dict, rest_init_filepaths,  A,
                      A_trans, list_of_nuc_with_trans_inp, t_transient = 10, duration = 10 ,n_run = 1, 
                      inp_method = 'reset',  ext_inp_dict = None, transient_init_filepaths = None):
    avg_act = {nuc: np.zeros((len(t_list),len(nuclei_dict[nuc]))) for nuc in list( nuclei_dict.keys() ) }
    for i in range(n_run):
        run_with_transient_external_input_including_transmission_delay(receiving_class_dict,t_list, dt, nuclei_dict, rest_init_filepaths, 
                                                                      A, A_trans, list_of_nuc_with_trans_inp, 
                                                                       t_transient = t_transient, duration = duration, inp_method = inp_method, 
                                                                       ext_inp_dict = ext_inp_dict, transient_init_filepaths = transient_init_filepaths)
        for nuclei_list in nuclei_dict.values():
                for k,nucleus in enumerate( nuclei_list) :
                    avg_act[ nucleus.name][:, k] += nucleus.pop_act/n_run
                    nucleus.clear_history()
        print(i+1,'from',n_run)
    return avg_act

def iterate_SNN(nuclei_dict, dt,receiving_class_dict, t_start = 0, t_end = 500):

    for t in range(t_start,t_end):
        for nuclei_list in nuclei_dict.values():
            for k, nucleus in enumerate(nuclei_list):
                nucleus.solve_IF(t,dt,receiving_class_dict[(nucleus.name,str(k + 1))])
                


def selective_reset_ext_input(nuclei_dict, init_filepaths, list_of_nuc_with_trans_inp, A, A_mvt = None, D_mvt = None, t_mvt = None, t_list = None, dt = None):

    set_init_all_nuclei(nuclei_dict, list_of_nuc_with_trans_inp =list_of_nuc_with_trans_inp, filepaths = init_filepaths) 

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:

            if nucleus.name in list_of_nuc_with_trans_inp:

                nucleus.set_ext_input(A, A_mvt, D_mvt,t_mvt, t_list, dt)

def selective_reset_ext_input_collective(nuclei_dict, list_of_nuc_with_trans_inp, A, 
                                         A_mvt = None, D_mvt = None, t_mvt = None, t_list = None, dt = None):

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:

            if nucleus.name in list_of_nuc_with_trans_inp:

                nucleus.set_ext_input(A, A_mvt, D_mvt,t_mvt, t_list, dt)

    return nuclei_dict

def mvt_grad_ext_input(D_mvt, t_mvt, delay, H0, t_series):
    ''' a gradually increasing deacreasing input mimicing movement'''

    H = H0*np.cos(2*np.pi*(t_series-t_mvt)/D_mvt)**2
    ind = np.logical_or(t_series < t_mvt + delay, t_series > t_mvt + D_mvt + delay)
    H[ind] = 0
    return H

def mvt_step_ext_input(D_mvt, t_mvt, delay, value, t_series):
    ''' step function returning external input during movment duration '''
    
    ext_add_inp = value * np.ones_like(t_series)
    ind = np.logical_or(t_series < t_mvt + delay, t_series > t_mvt + D_mvt + delay)
    ext_add_inp[ind] = 0
    return ext_add_inp

def calculate_number_of_connections(N_sim,N_real,number_of_connection):
    
    '''calculate number of connections in the scaled network.'''
    
    KK = number_of_connection.copy()
    
    for k, v in number_of_connection.items():
        KK[k] = int(1/(1/v-1/N_real[k[1]]+1/N_sim[k[0]]))
        
    return KK

def transfer_func(Threshold, gain, x):
    
    ''' a transfer function that grows linearly for positive values 
    of input higher than the threshold'''
    
    return gain* np.maximum(np.zeros_like(x), (x - Threshold))
    
# def build_connection_matrix(n_receiving,n_projecting,n_connections, same_pop = False):
#     ''' return a matrix with Jij=0 or 1. 1 showing a projection from neuron j in projectin population to neuron i in receiving'''
#     # produce a matrix listing received projections for each neuron in row i
#     projection_list = np.random.rand(n_receiving, n_projecting).argpartition(n_connections,axis=1)[:,:n_connections]
#     JJ = np.zeros((n_receiving, n_projecting),dtype = int)
#     rows = ((np.ones((n_connections,n_receiving))*np.arange(n_receiving)).T).flatten().astype(int)
#     cols = projection_list.flatten().astype(int)
#     JJ[rows,cols] = int(1)
#     return JJ

def build_connection_matrix(n_receiving,n_projecting,n_connections, same_pop = False):
    
    ''' return a matrix with Jij=0 or 1. 1 showing a projection from neuron j in projectin population to neuron i in receiving
        Arguments:
                same_pop: optional bool (default = False)
                    if the neuron type of pre and post are the same this value shows if they are in the same population as to avoid 
                    connecting a neuron to itself    
    '''
    # projection_list = np.random.rand(n_receiving, n_projecting).argpartition(n_connections,axis=1)[:,:n_connections] ### What the fuck? why this way? [Sep 2021]
    connection_prob = np.random.rand(n_receiving, n_projecting)
    
    if same_pop: # if connecting the population to itself, avoid autapses
        np.fill_diagonal(connection_prob, 0)
    projection_list = np.argsort(connection_prob, axis = 1)[::-1][:,:n_connections]
    
    JJ = np.zeros((n_receiving, n_projecting),dtype = int)
    rows = ((np.ones((n_connections,n_receiving))*np.arange(n_receiving)).T).flatten().astype(int)
    cols = projection_list.flatten().astype(int)
    JJ[rows,cols] = int(1)
    return JJ

def get_start_end_plot(plot_start, plot_end, dt, t_list):
    
    if plot_end == None : 
        
        plot_end = int( len(t_list) )
    
    else:
    
        plot_end = int(plot_end / dt)
        
    plot_start = int( plot_start / dt)
    
    return plot_start, plot_end

def plot( nuclei_dict,color_dict,  dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title = "", n_subplots = 1, title_fontsize = 12, plot_start = 0,
         ylabelpad = 0, include_FR = True, alpha_mvt = 0.2, plot_end = None, figsize = (6,5), plt_txt = 'vertical', plt_mvt = True, 
         plt_freq = False, ylim = None, include_std = True, round_dec = 2, legend_loc = 'upper right', 
         continuous_firing_base_lines = True, axvspan_color = 'lightskyblue', tick_label_fontsize = 18, plot_filtered = False,
         low_f = 8, high_f = 30, filter_order = 6, vspan = False, tick_length = 8, title_pad = None,
         ncol_legend = 1, line_type = ['-', '--'], alpha = 1, legend = True):    

    fig, ax = get_axes (ax)
    
    plot_start, plot_end  = get_start_end_plot(plot_start, plot_end, dt, t_list)
    count = 0
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in [nuclei_list[0]]:
            
            if plt_freq:
                
                label = nucleus.name + ' f=' + str(round(nucleus.frequency_basal,1)) + ' Hz'
            
            else: 
            
                label = nucleus.name
                
            if plot_filtered:
                
                act = nucleus.butter_bandpass_filter_pop_act_not_modify( dt, low_f, high_f, order= filter_order)
                peaks,_ =signal.find_peaks(act, height = 0)
                ax.plot(peaks * dt, act[peaks] + A[nucleus.name], 'x')
                ax.plot(t_list[plot_start: plot_end]*dt, act[plot_start: plot_end] + A[nucleus.name], line_type[nucleus.population_num-1], 
                        c = color_dict[nucleus.name],lw = 1.5, alpha = 0.4)
            
            else:
                ax.plot(t_list[plot_start: plot_end] * dt, nucleus.pop_act[plot_start: plot_end], 
                        line_type[nucleus.population_num-1], label = label, c = color_dict[nucleus.name],lw = 1.5, alpha = alpha)
                
            if continuous_firing_base_lines:
            
                ax.plot(t_list[plot_start: plot_end]*dt, np.ones_like(t_list[plot_start: plot_end])*A[nucleus.name], '--', c = color_dict[nucleus.name],lw = 1, alpha=0.8 )
            
            else:
            
                ax.plot(t_list[plot_start: int(t_mvt /dt)]*dt, np.ones_like(t_list[plot_start: int(t_mvt /dt)])*A[nucleus.name], '--', c = color_dict[nucleus.name],lw = 1, alpha=0.8 )

            if plt_mvt:
                
                if continuous_firing_base_lines:
                
                    ax.plot(t_list[plot_start: plot_end]*dt, np.ones_like(t_list[plot_start: plot_end])*A_mvt[nucleus.name], '--', c = color_dict[nucleus.name], alpha = alpha_mvt,lw = 1 )
                
                else:
                
                    ax.plot(t_list[int(t_mvt /dt): plot_end]*dt, np.ones_like(t_list[int(t_mvt /dt): plot_end])*A_mvt[nucleus.name], '--', c = color_dict[nucleus.name], alpha= alpha_mvt,lw = 1 )
            
            FR_mean, FR_std = nucleus. average_pop_activity(int(len(t_list) / 2) , len(t_list))
            
            if include_FR:
            
                if include_std:
                    txt =  r"$\overline{{FR_{{{0}}}}}$ ={1} $\pm$ {2}".format(nucleus.name,  round(FR_mean,round_dec), round(FR_std,round_dec) )
                else:
                    txt =  r"$\overline{{FR_{{{0}}}}}$ ={1}".format(nucleus.name,  round(FR_mean,round_dec))

            
                if plt_txt == 'horizontal' :
                    ax.text(0.05 + count * 0.22, 0.9 , txt, ha='left', va='center', rotation='horizontal',fontsize = 15, color = color_dict[nucleus.name], transform=ax.transAxes)

                elif plt_txt == 'vertical':
                    ax.text(0.2, 0.9 - count * 0.05, txt, ha='left', va='center', rotation='horizontal',fontsize = 15, color = color_dict[nucleus.name], transform=ax.transAxes)
            
            count = count + 1

    if vspan:
        ax.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color=axvspan_color)
        
    ax.set_title(title, fontsize = title_fontsize, pad = title_pad)
    ax.set_xlabel("time (ms)", fontsize = 15)
    ax.set_ylabel("firing rate (spk/s)", fontsize = 15,labelpad=ylabelpad)
    if legend:
        ax.legend(fontsize = 15, loc = legend_loc, framealpha = 0.1, frameon = False, ncol=ncol_legend)
    # ax.tick_params(axis='both', which='major', labelsize=10)
    ax_label_adjust(ax, fontsize = tick_label_fontsize, nbins = 5)
    ax.set_xlim(plot_start * dt - 10, plot_end * dt + 10) 
    
    ax.tick_params(axis='y', length = tick_length)
    ax.tick_params(axis='x', length = tick_length)
    if ylim != None:
        ax.set_ylim(ylim)
    remove_frame(ax)
    return fig

def _str_G_with_key(key):
    return r'$G_{' + list(key)[1] + '-' + list(key)[0] + r'}$'

def plot_multi_run_SNN( data,nuclei_dict,color_dict,  x, dt, t_list,  xlabel = 'G', title = "",title_fontsize = 18, figsize = (6,5)):    

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)

    count = 0
    for nuclei_list in nuclei_dict.values():
        for nucleus in [nuclei_list[0]]:
            std = np.std(data[nucleus.name, 'base_freq'], axis = 1)
            mean = np.average(data[nucleus.name, 'base_freq'], axis = 1)
            ax.plot(x, mean, '-o', label = nucleus.name, c = color_dict[nucleus.name], lw = 1.5)
            ax.fill_between(x,mean - std, mean + std, color = color_dict[nucleus.name], alpha = 0.3)
    if np.average(x) < 0 :
        ax.invert_xaxis()
    plt.title(title, fontsize = title_fontsize)
    plt.xlabel( xlabel, fontsize = 15)
    plt.ylabel("frequency", fontsize = 15)
    plt.legend(fontsize = 15, loc = 'upper right', framealpha = 0.1, frameon = False)
    # ax.tick_params(axis='both', which='major', labelsize=10)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=5)
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    fig.tight_layout()

    return fig

def scatter_2d_plot(x,y,c, title, label, limits = None,label_fontsize = 15, cmap = 'YlOrBr', ax = None):

    fig, ax = get_axes(ax)

    ax.plot(x, y,'k', lw = 0.5)
    img = ax.scatter(x, y, c=c, cmap=plt.get_cmap(cmap),lw = 1,edgecolor = 'k', s = 50)

    if limits == None:
        limits = {'x':(min(x),max(x)), 'y':(min(y),max(y))}
        
    ax.set_xlabel(label[0], fontsize = label_fontsize)
    ax.set_ylabel(label[1], fontsize = label_fontsize)
    ax.set_title(title)
    # ax.set_xlim(limits['x'])
    # ax.set_ylim(limits['y'])
    clb = fig.colorbar(img)
    clb.set_label(label[2], labelpad=10, y=0.5, rotation=-90)
    clb.ax.locator_params(nbins=5)
    ax_label_adjust(ax)   
    remove_frame(ax) 
    return fig

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
    
def scatter_3d_wireframe_plot(x,y,z,c, title, label, limits = None,label_fontsize = 15, cmap = 'hot', tick_label_fontsize = 15):
    
    fig = plt.figure(figsize = (8,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color = 'grey', lw = 0.5 ,zorder = 1)
    img = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=c.flatten(), cmap=plt.get_cmap(cmap), s = 20, lw = 1,edgecolor = 'k',zorder=2)

    if limits == None:
        limits = {'x':(np.amin(x),np.amax(x)), 'y':(np.amin(y),np.amax(y)), 'z':(np.amin(z),np.max(z))}
      
    ax.w_xaxis.set_pane_color ((0., 0., 0., 0.))
    ax.w_yaxis.set_pane_color ((0., 0., 0., 0.))
    ax.w_zaxis.set_pane_color ((0., 0., 0., 0.))  
    ax.w_xaxis.gridlines.set_lw(.5)
    ax.w_yaxis.gridlines.set_lw(0.5)
    ax.w_zaxis.gridlines.set_lw(0.5)
    ax.set_xlabel(label[0], fontsize = label_fontsize, labelpad = 10)
    ax.set_ylabel(label[1],fontsize = label_fontsize, labelpad = 10)
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
    ax_label_adjust(ax, fontsize = tick_label_fontsize)
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
# build_connection_matrix(4,10,2)
   
def max_non_empty_array(array):
    if len(array) == 0:
        return 0
    else:
        return np.max(array)
    
def cut_plateau(sig, epsilon_std = 10**(-2), epsilon = 10**(-2), window = 40):
    ''' return indices before a plateau '''
    # filtering based on data variance from mean value
#    variation = (sig-np.average(sig))**2
#    ind = np.where(variation >= epsilon_std) 
#    return ind[0]
    # filtering based on where the first and second derivatives are zero. Doesn't work with noise
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
        # Overthinking
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

def moving_average_array_2d(X, n):
    '''Return the moving average over axis = 1 of X with window n without changing dimesions of X'''
    z2 = np.cumsum(np.concatenate((np.zeros((X.shape[0], n)), X), axis = 1) ,
                   axis = 1)
    z1 = np.cumsum(np.concatenate(( X, np.repeat(X[:,-1].reshape(-1,1) , n, axis = 1 )), axis = 1) ,
                   axis = 1)

    return (z1-z2)[:,(n-1):-1]/n

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

def find_mean_of_signal(sig, cut_plateau_ind, non_plateau_perc = 4/5):
    ''' find the value which when subtracted the signal oscillates around zero'''
    len_varying = len(cut_plateau_ind)
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

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        ''' filtfilt is the forward-backward filter. It applies the filter twice, 
        once forward and once backward, resulting in zero phase delay.
        '''
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y

def trim_start_end_sig_rm_offset(sig,start, end, cut_plateau_epsilon = 0.1, method = 'neat'):
    ''' trim with max point at the start and the given end point'''
    if method not in ["simple", "neat"]:
        raise ValueError("method must be either 'simple', or 'neat'")

    trimmed = sig[start:end]
    if method == 'simple' : 
        return trimmed
    elif method == 'neat':
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
        
def find_freq_of_pop_act_spec_window(nucleus, start, end, dt, peak_threshold = 0.1, smooth_kern_window= 3 , 
                                     check_stability = False, cut_plateau_epsilon = 0.1):
    
    ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
    the plateau and in case it is oscillation determine the frequency '''
    sig = trim_start_end_sig_rm_offset(nucleus.pop_act,start, end)
    cut_sig_ind = cut_plateau( sig, epsilon= cut_plateau_epsilon)
    plateau_y = find_mean_of_signal(sig, cut_sig_ind)
    # fig, ax = plt.subplots()
    # ax.plot(sig - plateau_y)
    # ax.axhline(plateau_y)
    # ax.plot(sig[cut_sig_ind])
    if_stable = False
    if len(cut_sig_ind) > 0: # if it's not all plateau from the beginning
        sig = sig - plateau_y
        # print('trimmed plateau removed', np.average(sig))
#        if if_oscillatory(sig, max(cut_sig_ind),nucleus.oscil_peak_threshold, nucleus.smooth_kern_window): # then check if there's oscillations
#            perc_oscil = max_non_empty_array(cut_sig_ind)/len(sig)*100
# freq = freq_from_fft(sig[cut_sig_ind],dt/1000)
#            _,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)
#            return perc_oscil, freq
#        else:
#            return 0,0
        n_half_cycles,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)
        if freq != 0: # then check if there's oscillations
            perc_oscil = max_non_empty_array(cut_sig_ind)/len(sig)*100
            if check_stability:
                if_stable = if_stable_oscillatory(sig, max(cut_sig_ind), peak_threshold, smooth_kern_window, amp_env_slope_thresh = - 0.05)
            return n_half_cycles, perc_oscil, freq, if_stable
        else:
            return 0,0,0, False
    else:
        return 0,0,0, False

def if_stable_oscillatory(sig, x_plateau, peak_threshold, smooth_kern_window, amp_env_slope_thresh = - 0.05, 
                          oscil_perc_as_stable = 0.9, last_first_peak_ratio_thresh = [0.95,1.05]):
    
    ''' detect if there's stable oscillation defined as a non-decaying wave'''
    if  x_plateau > len(sig) * oscil_perc_as_stable : # if the whole signal is oscillatory

        # sig = gaussian_filter1d(sig[:x_plateau],smooth_kern_window)
        peaks, properties = signal.find_peaks(sig, height = peak_threshold)
        # troughs,_ = signal.find_peaks(-sig, height = peak_threshold)
        # if len(peaks)-1 == 0: # if no peaks are found error will be raised, so we might as well plot to see the reason
            # troughs,_ = signal.find_peaks(-sig, height = peak_threshold)
        # plt.figure()
        # plt.plot(sig)
        # plt.plot(peaks, sig[peaks], 'x')
            # plt.axhline(np.average(sig))
        # relative first and last peak ratio thresholding
        if len(peaks) > 5 : 
            last_first_peak_ratio = sig[peaks[-1]] / sig[peaks[5]]
            print('last_first_peak_ratio = ', last_first_peak_ratio)
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

        # amplitude envelope Slope thresholding method
        # slope, intercept, r_value, p_value, std_err = stats.linregress(peaks[1:],sig[peaks[1:]]) # discard the first peak because it's prone to errors
        # print('slope = ', slope)
        # if slope > amp_env_slope_thresh: 

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
    
def create_data_dict_G_sweep(nuclei_dict, G, n):
    data = {} 
    
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0] # get only on class from each population
        data[(nucleus.name, 'mvt_freq')] = np.zeros(n)
        data[(nucleus.name, 'base_freq')] = np.zeros(n)
        data[(nucleus.name, 'perc_t_oscil_mvt')] = np.zeros(n)
        data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros(n)
        data[(nucleus.name, 'n_half_cycles_mvt')] = np.zeros(n)
        data[(nucleus.name, 'n_half_cycles_base')] = np.zeros(n)
        data[(nucleus.name,'g_transient')] = []
        data[(nucleus.name,'g_stable')] = []
        
    
    data['g'] = {}    
    
    for k in list(G.keys()):
        
        data['g'][k] = np.zeros(n)
    return data
def save_freq_analysis_to_df(data, state, i, nucleus, dt, duration):
    
    (data[(nucleus.name, 'n_half_cycles_' + state)][i],
    data[(nucleus.name,'perc_t_oscil_' + state)][i], 
    data[(nucleus.name, state + '_freq')][i],
    if_stable )= find_freq_of_pop_act_spec_window(nucleus,*duration,dt, peak_threshold =nucleus.oscil_peak_threshold, 
                                                    smooth_kern_window = nucleus.smooth_kern_window, check_stability = True)
    return data, if_stable

def Product_G(data):
    
    g_keys = list( 
                data['g'].keys()
                )
    g = np.ones(
            len(data['g'][g_keys[0]])
                )
    for k in g_keys:
        g = g * data['g'][k]
    return g

def multiply_values_of_dict(dictionary):
    product = 1
    for val in dictionary.values():
        product = product * val
    return product

def multiply_G_by_key_ratio(g, G, G_ratio_dict):
    for key , v in G_ratio_dict.items(): 
        G[ key ] = g * v
    return G

def fill_Gs_in_data(data, i, G):
    for key , value in G.items(): 
        data['g'][key ][i] = value
    return data

def synaptic_weight_space_exploration(G, A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, 
                                      G_list, nuclei_dict, duration_mvt, duration_base, receiving_class_dict, 
                                      color_dict, if_plot = False, G_ratio_dict = None, plot_start_trans = 0, 
                                      plot_start_stable = 0, plot_duration = 600,
                                      legend_loc = 'upper left', vspan_stable = False):
    
    
    n = len(G_list)
    data = create_data_dict_G_sweep(nuclei_dict, G, n)
    if_stable_plotted = False
    if_trans_plotted = False
    
    if if_plot:
        fig = plt.figure()
        
    found_g_transient = {k: False for k in nuclei_dict.keys()}
    found_g_stable = {k: False for k in nuclei_dict.keys()}

    for i, g in enumerate( sorted(G_list, key=abs) ):

        G = multiply_G_by_key_ratio(g, G, G_ratio_dict)
        data = fill_Gs_in_data(data, i, G)
            
        nuclei_dict = reinitialize_nuclei(nuclei_dict, G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
        run(receiving_class_dict,t_list, dt, nuclei_dict)
        
        nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
        
        for nucleus in nucleus_list:


            data, if_stable_mvt = save_freq_analysis_to_df(data, 'mvt', i, nucleus, dt, duration_mvt) 
            data, if_stable_base = save_freq_analysis_to_df(data, 'base', i, nucleus, dt, duration_base)
                                                  

            print(nucleus.name,' g = ', round(multiply_values_of_dict(G), 2), 
                  'n_cycles =', data[(nucleus.name, 'n_half_cycles_mvt')][i],
                  round(data[(nucleus.name, 'perc_t_oscil_mvt')][i],2),
                  '%',  'f = ', round(data[(nucleus.name,'mvt_freq')][i],2) )

            if ( not found_g_transient[nucleus.name]  and 
                data[(nucleus.name, 'n_half_cycles_mvt')][i] > lim_n_cycle[0] and 
                data[(nucleus.name, 'n_half_cycles_mvt')][i]< lim_n_cycle[1] ):
                
                data[(nucleus.name,'g_transient')] = abs(g) # save the the threshold g to get transient oscillations
                found_g_transient[nucleus.name] = True
                data['g_loop_transient'] =  abs( multiply_values_of_dict(G) )
                
                if not if_trans_plotted:
                
                    if_trans_plotted = True
                    print("transient plotted")
                                      
                    fig_trans = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = plot_start_trans + plot_duration,
                                     include_FR = False, plot_start = plot_start_trans, legend_loc = legend_loc,
                                     title_fontsize = 15, title = 'Transient Oscillation', ax = None, continuous_firing_base_lines = False,
                                     vspan = True)
                    
            if found_g_stable[nucleus.name] == False and if_stable_mvt: 
                found_g_stable[nucleus.name] = True
                data[(nucleus.name,'g_stable')] = abs(g)
                data['g_loop_stable'] =  abs( multiply_values_of_dict(G) )
                
                if not if_stable_plotted :
                    
                    if_stable_plotted = True
                    print("stable plotted")
                    fig_stable1 = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = plot_start_stable + plot_duration,
                                      include_FR = False, plot_start = 0, legend_loc = legend_loc, 
                                      title_fontsize = 15, title = 'Stable Oscillation', ax = None, continuous_firing_base_lines = False,
                                      vspan = vspan_stable )
                    fig_stable2 = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = plot_start_trans + plot_duration,
                                      include_FR = False, plot_start = plot_start_trans, legend_loc = legend_loc, 
                                      title_fontsize = 15, title = 'Stable Oscillation', ax = None, continuous_firing_base_lines = False,
                                      vspan = True )
                    fig_stable3 = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = t_mvt + 200 + plot_duration,
                                      include_FR = False, plot_start =t_mvt + 200, legend_loc = legend_loc, 
                                      title_fontsize = 15, title = 'Stable Oscillation', ax = None, continuous_firing_base_lines = False,
                                      vspan = True )
                        
            if if_plot:
                
                ax = fig.add_subplot(n, 1, i + 1)
                plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt,[fig, ax], title = '', n_subplots = int(n))
                ax.set_title('', fontsize = 10)    
                ax.set_xlabel("", fontsize = 10)
                ax.set_ylabel("", fontsize = 5)
                ax.legend(fontsize = 10)
                
            print(i, "from", n)


    data['G_ratio'] = G_ratio_dict 
    if if_plot:
        fig.text(0.5, 0.01, 'time (ms)', ha='center')
        fig.text(0.01, 0.5, 'firing rate (spk/s)', va='center', rotation='vertical')
        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()
    return fig_trans, [fig_stable1, fig_stable2, fig_stable3]

def G_sweep_title(G_dict, g_1, g_2):
    title = ( r"$G_{" + list(G_dict.keys())[0][1] + "-" + 
                        list(G_dict.keys())[0][0] + "}$ = " + str(round(g_1, 2)) + r"$\; G_{" +
                        list(G_dict.keys())[1][1] + "-" +
                        list(G_dict.keys())[1][0] + "}$ =" + str(round(g_2, 2))
            )
    return title

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
            nucleus.scale_synaptic_weight() 
            nucleus.set_ext_input(A, A_mvt, D_mvt,t_mvt, t_list, dt)
    return nuclei_dict


    
def save_freq_analysis_to_df_2d(data, state, i, j, nucleus, dt, duration, check_stability = True):
    
    (data[(nucleus.name, 'n_half_cycle_' + state)][i, j], _,
    data[(nucleus.name, state + '_mvt_freq')][i, j], if_stable) = find_freq_of_pop_act_spec_window(nucleus,*duration,dt, 
                                                                                        peak_threshold = nucleus.oscil_peak_threshold, 
                                                                                        smooth_kern_window = nucleus.smooth_kern_window, 
                                                                                        check_stability = check_stability)
    print(nucleus.name, ' stability =', if_stable)                                                                                             
    return data

def create_data_dict_tau_sweep_2d(nuclei_dict, G, iter_param_length_list, n_time_scale, n_timebins, synaptic_time_constant,
                                  check_transient = False):
    '''build a data dictionary for 2d tau sweep '''
    
    data = {} ; 
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0] # get only on class from each population
        data[(nucleus.name, 'stable_mvt_freq')] = np.zeros(iter_param_length_list)
        data[(nucleus.name, 'n_half_cycle_stable')] = np.zeros(iter_param_length_list)
        
        if check_transient:
            data[(nucleus.name, 'trans_mvt_freq')] = np.zeros(iter_param_length_list)
            data[(nucleus.name, 'n_half_cycle_trans')] = np.zeros(iter_param_length_list)


    data['g_stable'] = np.zeros(iter_param_length_list)
    data['g_transient'] = np.zeros(iter_param_length_list)
    data['tau'] = {}
    
    for k in list(synaptic_time_constant.keys()):
        data['tau'][k] = np.zeros(iter_param_length_list)
        print(k)
    return data

def extract_syn_time_constant_from_dict(synaptic_time_constant, syn_decay_dict, t_decay_1, t_decay_2, if_track_tau_2 = True):
    
    for key,v in syn_decay_dict['tau_1']['tau_ratio'].items():    
        synaptic_time_constant[key] = [syn_decay_dict['tau_1']['tau_ratio'][key] * t_decay_1]
        
    if if_track_tau_2:
        for key,v in syn_decay_dict['tau_2']['tau_ratio'].items():    
            synaptic_time_constant[key] = [syn_decay_dict['tau_2']['tau_ratio'][key] * t_decay_2]
        
    return synaptic_time_constant

def fill_taus_and_Gs_in_data(data, i, j, synaptic_time_constant,  g_transient, g_stable):
    
    for k, v in synaptic_time_constant.items():
        data['tau'][k][i,j] = v[0]
        
    data['g_transient'][i, j], data['g_stable'][i, j] =  g_transient, g_stable
    return data

def set_time_scale( nuclei_dict, synaptic_time_constant):
    
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            nucleus.set_synaptic_time_scales(synaptic_time_constant) 
            
    return nuclei_dict
    
def run_and_derive_freq(data, receiving_class_dict, nuclei_dict, g_transient, duration, i, j , state,
                        G, G_ratio_dict, A, A_mvt, 
                        D_mvt,t_mvt, t_list, dt, check_stability = False):
    
    G = multiply_G_by_key_ratio(g_transient, G, G_ratio_dict)
    
    nuclei_dict = reinitialize_nuclei(nuclei_dict, G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
    run(receiving_class_dict,t_list, dt, nuclei_dict)

    nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
    
    for nucleus in nucleus_list:
        data = save_freq_analysis_to_df_2d(data, state, i, j, nucleus, dt, duration, check_stability = check_stability)
        
    return data

def sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant, nuclei_dict, 
                      syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict, 
                      t_list,dt, duration_base, duration_mvt, lim_n_cycle, find_stable_oscill=True, 
                      check_transient = False, if_track_tau_2 = True):
    
    
    t_decay_series_1 = list(syn_decay_dict['tau_1']['tau_list']) 
    t_decay_series_2 = list(syn_decay_dict['tau_2']['tau_list'])
    n_runs = len(t_decay_series_1)*len(t_decay_series_2)
    
    data  = create_data_dict_tau_sweep_2d(nuclei_dict, G, (len(t_decay_series_1), len(t_decay_series_2)), 
                                          2, len(t_list), synaptic_time_constant, check_transient = check_transient)    
    data['G_ratio'] = G_ratio_dict
    count = 0 
    
    for i, t_decay_1 in enumerate( t_decay_series_1 ) :
        
        for j, t_decay_2 in enumerate( t_decay_series_2 ) :
            
            synaptic_time_constant = extract_syn_time_constant_from_dict(synaptic_time_constant, syn_decay_dict, 
                                                                         t_decay_1, t_decay_2, if_track_tau_2 = if_track_tau_2 )
            print(synaptic_time_constant)
            nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
            nuclei_dict = set_time_scale(nuclei_dict, synaptic_time_constant)
            
            g_transient, g_stable = find_oscillation_boundary(g_list, nuclei_dict.copy(), G,
                                                              G_ratio_dict, A, A_mvt,t_list,dt, 
                                                              receiving_class_dict, D_mvt, 
                                                              t_mvt, duration_mvt, duration_base, 
                                                              lim_n_cycle = lim_n_cycle , 
                                                              find_stable_oscill = find_stable_oscill)
            
            data = fill_taus_and_Gs_in_data(data, i, j, synaptic_time_constant,  g_transient, g_stable)
            
            data = run_and_derive_freq(data, receiving_class_dict,nuclei_dict, g_stable, duration_mvt, i, j , 
                                       'stable', G, G_ratio_dict, A, A_mvt, 
                                       D_mvt,t_mvt, t_list, dt, check_stability = True)
            
            if check_transient:

                data = run_and_derive_freq(data, receiving_class_dict, nuclei_dict, g_transient, duration_mvt, i, j ,
                                       'trans', G, G_ratio_dict, A, A_mvt, 
                                       D_mvt,t_mvt, t_list, dt)
            count +=1
            print(count, "from ", n_runs)
            
    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()

def check_if_zero_activity(duration_mvt, nucleus):
    
        half_t_of_mvt =  middle_range(duration_mvt)
        ind_zero_act = np.argwhere(
                            nucleus.pop_act[ duration_mvt[0] : duration_mvt[1]] == 0 
                            )
        n_t_zero_act = len(ind_zero_act)
        if  n_t_zero_act > half_t_of_mvt  :
            print('zero activity')
            
def middle_range(range_ends):
    
    return (range_ends[1] - range_ends[0]) / 2

def find_oscillation_boundary(g_list,nuclei_dict, G, G_ratio_dict,A, A_mvt,t_list,dt, receiving_class_dict, 
                              D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle = [6,10], 
                              find_stable_oscill = False):
    
    ''' find the synaptic strength for a given set of parametes where you oscillations 
        appear after increasing g'''
    
    found_transient_g = False
    found_stable_g = False
    g_stable = None
    
    for g in sorted(g_list, key=abs):
        
        G = multiply_G_by_key_ratio(g, G, G_ratio_dict)
        
        nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
        run(receiving_class_dict,t_list, dt, nuclei_dict)
        
        nucleus = list(nuclei_dict.values())[0][0]
        
        n_half_cycles_mvt,perc_oscil_mvt, f_mvt, if_stable_mvt = find_freq_of_pop_act_spec_window(nucleus, *duration_mvt, dt, 
                                                                                                  peak_threshold = nucleus.oscil_peak_threshold,
                                                                                                  smooth_kern_window = nucleus.smooth_kern_window, 
                                                                                                  check_stability= find_stable_oscill)
        
        # n_half_cycles_base, perc_oscil_base, f_base, if_stable_base = find_freq_of_pop_act_spec_window(nucleus, *duration_base, dt, 
        #                                                                                                peak_threshold = nucleus.oscil_peak_threshold, 
        #                                                                                                smooth_kern_window = nucleus.smooth_kern_window, 
        #                                                                                                check_stability= find_stable_oscill)
        
        print('g = {}, f = {}, {} %'.format( round(g,1), 
                                            round(f_mvt , 1), 
                                            round(perc_oscil_mvt, 1) 
                                            )
              )
        
        # check_if_zero_activity(duration_mvt, nucleus)
       
        if lim_n_cycle[0] <= n_half_cycles_mvt <= lim_n_cycle[1] and not found_transient_g:
            
            found_transient_g = True
            g_transient = g 
            print("Gotcha transient!")
                
            if  not find_stable_oscill:
                break
            
        if find_stable_oscill and if_stable_mvt and not found_stable_g:
            
            found_stable_g = True
            g_stable = g
            print("Gotcha stable!")
            
            break
            
    if not found_transient_g:
        # raise ValueError("Transient oscillation couldn't be found in the given <g> range" )
        print("Transient oscillation couldn't be found in the given <g> range" )
        g_transient = 0
    if find_stable_oscill and not found_stable_g:
        raise ValueError ("Stable oscillation couldn't be found in the given <g> range" )
        
    return g_transient, g_stable
            

def synaptic_weight_transition_multiple_circuit_SNN(filename_list, name_list, label_list, color_list, g_cte_ind, g_ch_ind, y_list, 
                                                    c_list,colormap,x_axis = 'multiply',title = "",x_label = "G", key = (), 
                                                    param = 'all', y_line_fix = 4, clb_lower_lim = 0, clb_higher_lim = 50, 
                                                    clb_loc = 'center right', legend_loc = 'upper right', clb_borderpad = 8):
    maxs = [] ; mins = []
    fig, ax = plt.subplots(1,1,figsize=(8,6))

    vmax = 50; vmin = 0
    for i in range(len(filename_list)):
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        keys = list(data['g'].keys())
        if x_axis == 'multiply':
            g = np.ones( len( list (data ['g'].values()) [0] ))
            for v in data['g'].values():
                g *= v
            x_label = r'$G_{Loop}$'
            txt = []
            for k in keys:
                if k == key:
                    txt.append( _str_G_with_key(k) +'=[' + r"${0:s}$".format(as_si(data['g'][k][0],0)) + ',' + r"${0:s}$".format(as_si(data['g'][k][-1],0)) + ']')
                    continue
                txt.append( _str_G_with_key(k) +'=' + r"${0:s}$".format(as_si(data['g'][k][0],1)))
                title += ' ' + _str_G_with_key(k) +'=' + str(round(data['g'][k][0], 3))
        else:
            g = data['g'][key]
            x_label = _str_G_with_key(key)
            title = ''
            txt = []
            for k in keys:
                if k == key:
                    continue
                txt.append( _str_G_with_key(k) +'=' + str(round(data['g'][k][0], 3)))
                title += ' ' + _str_G_with_key(k) +'=' + str(round(data['g'][k][0], 3))
        if param == "all":
            y =  np.squeeze( np.average ( data[(name_list[i],y_list[i])] , axis = 1))
            std =  np.std ( data[(name_list[i],y_list[i])] , axis = 1)
            title = 'beta band (12-30 Hz)'
        elif param == 'low':
            y =  np.average ( data[(name_list[i],y_list[i])] , axis = 1)[:,0]
            std = np.std ( data[(name_list[i],y_list[i])] , axis = 1)[:,0]
            title = 'Low beta band (12-30 Hz)'
        elif param == 'high':
            y =  np.average ( data[(name_list[i],y_list[i])] , axis = 1)[:,1]
            std = np.std ( data[(name_list[i],y_list[i])] , axis = 1)[:,1]
            title = 'High beta band (20-30 Hz)'
        color = np.squeeze( np.average ( data[(name_list[i],c_list[i])] , axis = 1))
        where_are_NaNs = np.isnan(y)
        y[where_are_NaNs] = 0
        # ax.plot(g, y, c = color_list[i], lw = 3, label= label_list[i],zorder=1)
        ax.errorbar(g, y, yerr = std, c = color_list[i], lw = 3, label= label_list[i],zorder=1, capthick = 5, elinewidth = 1.5)

        img = ax.scatter(g, y ,vmin = vmin, vmax = vmax, c = color, 
                          cmap=plt.get_cmap(colormap),lw = 1,edgecolor = 'k',zorder=2, s=60)
    plt.axhline( y_line_fix, linestyle = '--', c = 'grey', lw=2)  # to get the circuit g which is the muptiplication
    plt.title(title, fontsize = 20)
    ax.set_xlabel(x_label,fontsize = 20)
    ax.set_ylabel('Beta Power (W/Hz)',fontsize=20)
    # ax.set_title(title,fontsize=20)
    # ax_label_adjust(ax, fontsize = 18, nbins = 6)
    axins1 = inset_axes(ax,
                    width="5%",  # width = 50% of parent_bbox width
                    height="50%",  # height : 5%
                    loc= clb_loc ,borderpad=clb_borderpad)#, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),)
    clb = fig.colorbar(img, cax=axins1, orientation="vertical")
    clb.ax.locator_params(nbins=4)
    clb.set_label('Frequency (Hz)', labelpad=20, y=.5, rotation=-90,fontsize=15)
    img.set_clim(clb_lower_lim , clb_higher_lim)
    ax.legend(fontsize=15, frameon = False, framealpha = 0.1, loc = legend_loc)
    remove_frame(ax)
    plt.show()
    for i in range( len(txt)):
        plt.gcf().text(0.5, 0.8- i*0.05,txt[i], ha='center',fontsize = 13)
    return fig

import warnings
warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")

def test(y):
    yy = np.copy(y)
    for i in range(y.shape[1]): # iterate over runs
        yy[:,i] = np.append(0, np.diff(y[:,i])) / y[:,i]
    return yy

def synaptic_weight_transition_multiple_circuit_SNN_Fr_inset(filename, name_list, label_list, color_list,  y_list, 
                                                             freq_list,colormap, x_axis = 'multiply', title = "", x_label = "G", key = (), 
                                                             param = 'all', y_line_fix = 4, inset_ylim = [0, 40],  legend_loc = 'upper right', 
                                                             include_Gs = True, double_xaxis = False, key_sec_ax = (), nuc_loop_lists = None, 
                                                             loops = 'single', plot_phase = True, new_tick_locations = np.array([0.2, 0.5, 0.8]),
                                                             second_axis_label = r"$G_{FSI-D2-P}$", inset_props = [0.02, 0.3, 0.35, 0.35],
                                                             ylim = None, plot_inset = True, markersize = 8):
    maxs = [] ; mins = []
    fig, ax = plt.subplots(figsize=[8, 6])
    if plot_inset:
        axins = ax.inset_axes(inset_props)
    else:
        axins = None
    vmax = 50; vmin = 0
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    
    for i in range( len(name_list) ):
        
        keys = list(data['g'].keys())
        g, txt = get_plot_Gs(x_axis, include_Gs, data, nuc_loop_lists, keys, key)
        
        freq = np.squeeze( np.average ( data[(name_list[i],freq_list[i])] , axis = 1))
        freq_std = np.squeeze( np.std ( data[(name_list[i],freq_list[i])] , axis = 1))
        title = 'Beta band (12-30 Hz)'
        beta_separate = beta_separate_or_not(data, name_list[i],y_list[i])
        if param == "all" and not beta_separate:
            
            y =  np.squeeze( np.average ( data[(name_list[i],y_list[i])] , axis = 1))
            std =  np.std ( data[(name_list[i],y_list[i])] , axis = 1)      
            ax.errorbar(abs(g), y, yerr = std, c = color_list[i], lw = 3, label= label_list[i], zorder=1, 
                        capthick = 5, elinewidth = 1.5, markersize = markersize,  marker = 'o')
        
        else:

            y, std, title_h_l = get_data_low_high_beta(data, name_list, i, y_list)
    
            if param == 'high_low':
                plot_beta_power(ax, g, y, std, 'high' , color_list[i], label_list[i], alpha = 1, markersize = markersize)
                plot_beta_power(ax, g, y, std, 'low' , color_list[i], '' , alpha = 0.4,  markersize = markersize)
                
            elif param == 'high' or param == 'low':
                plot_beta_power(ax, g, y, std, param , color_list[i], label_list[i], alpha = 1, markersize = markersize)
                title = title_h_l[param]
                
            else: ## beta is saved separatly but needs to be shown as summed
                y, std, title = sum_data_low_high_beta(data, name_list, i, y_list)
                
                ax.errorbar(abs(g), y, yerr = std, c = color_list[i], lw = 3, label= label_list[i], zorder=1, 
                            capthick = 5, elinewidth = 1.5, markersize= markersize, marker = 'o')
                ax.axhline(0, xmin = min(abs(g)), xmax = max(abs(g)), linestyle = '--', color = 'k')
        if plot_inset:
            axins.errorbar(abs(g), freq, yerr = freq_std,  c = color_list[i], lw = 1, label= label_list[i],zorder=1, 
                           capthick = 1, elinewidth = .7)
    
    
    set_ax_axins_prop(ax, title, x_label, inset_ylim, double_xaxis, data, key_sec_ax, new_tick_locations, 
                      second_axis_label, legend_loc, y_line_fix, ylim = ylim, plot_inset = plot_inset, axins = axins)

    
    if include_Gs:
        for i in range( len(txt)):
            plt.gcf().text(0.5, 0.8- i*0.05,txt[i], ha='center',fontsize = 13)
    return fig
# from scipy.stats import pearsonr
# def synaptic_weight_transition_multiple_circuit_SNN_Fr_inset_correlation(filename, name_list, label_list, color_list,  y_list, 
#                                                              freq_list,colormap, x_axis = 'multiply', title = "", x_label = "G", key = (), 
#                                                              param = 'all', y_line_fix = 4, inset_ylim = [0, 40],  legend_loc = 'upper right', 
#                                                              include_Gs = True, double_xaxis = False, key_sec_ax = (), nuc_loop_lists = None, 
#                                                              loops = 'single', plot_phase = True, new_tick_locations = np.array([0.2, 0.5, 0.8]),
#                                                              second_axis_label = r"$G_{FSI-D2-P}$", inset_props = [0.02, 0.3, 0.35, 0.35],
#                                                              ylim = None, plot_inset = True, markersize = 8):
#     maxs = [] ; mins = []
#     fig, ax = plt.subplots(figsize=[8, 6])
#     if plot_inset:
#         axins = ax.inset_axes(inset_props)
#     else:
#         axins = None
#     vmax = 50; vmin = 0
#     pkl_file = open(filename, 'rb')
#     data = pickle.load(pkl_file)
#     y_dict = {}
#     for i in range( len(name_list) ):
        
#         keys = list(data['g'].keys())
#         g, txt = get_plot_Gs(x_axis, include_Gs, data, nuc_loop_lists, keys, key)
        
#         freq = np.squeeze( np.average ( data[(name_list[i],freq_list[i])] , axis = 1))
#         freq_std = np.squeeze( np.std ( data[(name_list[i],freq_list[i])] , axis = 1))
#         title = 'Beta band (12-30 Hz)'
#         beta_separate = beta_separate_or_not(data, name_list[i],y_list[i])
#         if param == "all" and not beta_separate:
            
#             y =  np.squeeze( np.average ( data[(name_list[i],y_list[i])] , axis = 1))
#             std =  np.std ( data[(name_list[i],y_list[i])] , axis = 1)      
#             ax.errorbar(abs(g), y, yerr = std, c = color_list[i], lw = 3, label= label_list[i], zorder=1, 
#                         capthick = 5, elinewidth = 1.5, markersize = markersize,  marker = 'o')
        
#         else:

#             y, std, title_h_l = get_data_low_high_beta(data, name_list, i, y_list)
    
#             if param == 'high_low':
#                 plot_beta_power(ax, g, y, std, 'high' , color_list[i], label_list[i], alpha = 1, markersize = markersize)
#                 plot_beta_power(ax, g, y, std, 'low' , color_list[i], '' , alpha = 0.4,  markersize = markersize)
                
#             elif param == 'high' or param == 'low':
#                 plot_beta_power(ax, g, y, std, param , color_list[i], label_list[i], alpha = 1, markersize = markersize)
#                 title = title_h_l[param]
                
#             else: ## beta is saved separatly but needs to be shown as summed
#                 y, std, title = sum_data_low_high_beta(data, name_list, i, y_list)
#                 y_dict[name] = y
#                 ax.errorbar(abs(g), y, yerr = std, c = color_list[i], lw = 3, label= label_list[i], zorder=1, 
#                             capthick = 5, elinewidth = 1.5, markersize= markersize, marker = 'o')
#                 ax.axhline(0, xmin = min(abs(g)), xmax = max(abs(g)), linestyle = '--', color = 'k')
#         if plot_inset:
#             axins.errorbar(abs(g), freq, yerr = freq_std,  c = color_list[i], lw = 1, label= label_list[i],zorder=1, 
#                            capthick = 1, elinewidth = .7)
    
    
#     set_ax_axins_prop(ax, title, x_label, inset_ylim, double_xaxis, data, key_sec_ax, new_tick_locations, 
#                       second_axis_label, legend_loc, y_line_fix, ylim = ylim, plot_inset = plot_inset, axins = axins)

#     print(filename, pearsonr(y))
    
#     if include_Gs:
#         for i in range( len(txt)):
#             plt.gcf().text(0.5, 0.8- i*0.05,txt[i], ha='center',fontsize = 13)
#     return fig

def beta_separate_or_not(data, name, param):
    ''' if there is another dimension other than n_iter and n_run, the beta powers are saved separately'''
    if len( data[(name, param)].shape) == 3 : 
        return True
    else: 
        return False
def G_product_of_certain_loop(gs, nuc_loop_list):
    g = np.ones( len( list (gs.values()) [0] ))
    for key in list (gs.keys()):
        if set(key).issubset(nuc_loop_list): 
            g = g *gs[key]
    return g

def derive_G_multi_loop(gs, nuc_loop_lists):
    g = np.zeros( len( list (gs.values()) [0] ))
    for nuc_loop_list in nuc_loop_lists:
        g = g + G_product_of_certain_loop(gs, nuc_loop_list)
    return g

def multiply_Gs(lists_of_g_series):
    lists_of_g_series = list(lists_of_g_series)
    g_multiplication = np.ones( len( lists_of_g_series[0] ) )
                
    for v in lists_of_g_series:
        g_multiplication *= v
    return g_multiplication

def plot_beta_power(ax, g, y, std, param , color, label, alpha= 1, markersize = 8):
    
    ax.errorbar(abs(g), y[param], yerr = std[param], c = color, lw = 3, label= label, zorder=1, 
                capthick = 1, capsize = 3, elinewidth = 1.5, alpha = alpha, markersize = markersize,  marker = 'o')

def get_data_low_high_beta(data, name_list, i, y_list):
    y, std , title = {}, {}, {}
    y['high'] =  np.average ( data[(name_list[i],y_list[i])] , axis = 1)[:,0]
    std['high'] = np.std ( data[(name_list[i],y_list[i])] , axis = 1)[:,0]
    y['low'] =  np.average ( data[(name_list[i],y_list[i])] , axis = 1)[:,1]
    std['low'] = np.std ( data[(name_list[i],y_list[i])] , axis = 1)[:,1]
    title['high'] = 'High beta band (20-30 Hz)'
    title['low'] = 'Low beta band (12-30 Hz)'
    
    return y, std, title

def sum_data_low_high_beta(data, name_list, i, y_list):
    
    yy  = np.sum( data[(name_list[i],y_list[i])] , 
                 axis = 2)
    # yy = test(yy)
    
    y =  np.average ( yy ,
                    axis = 1)
    std = np.std ( yy ,
                    axis = 1)

    title = 'Beta band (12-30 Hz)'
    
    return y, std, title
def set_ax_axins_prop(ax, title, x_label, inset_ylim, double_xaxis, data, key_sec_ax, new_tick_locations, 
                      second_axis_label, legend_loc, y_line_fix, ylim = None, plot_inset = True, axins = None):
    if y_line_fix != None:
        ax.axhline( y_line_fix, linestyle = '--', c = 'grey', lw=2)  # to get the circuit g which is the muptiplication
        
    ax.set_title(title, fontsize = 20)
    ax.set_xlabel(r"$| G_{Loop} |$",fontsize = 20)
    ax.set_ylabel('Beta Power (W/Hz)',fontsize=20)
    ax_label_adjust(ax, fontsize = 18, nbins = 6)
    if ylim != None:
        ax.set_ylim(ylim)
    if plot_inset:
        axins.set_yticklabels(labels = axins.get_yticks().tolist(), fontsize = 12)
        axins.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        axins.set_ylim(inset_ylim)
        axins.set_ylabel("Frequency (Hz)", fontsize = 10)
        axins.yaxis.set_label_position("right")
        axins.yaxis.tick_right()
        axins.xaxis.set_major_locator(MaxNLocator(5)) 


        axins.set_xticklabels(labels = axins.get_xticks().tolist(), fontsize = 12)
        axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axins.set_xlabel(r"$| G+{Loop} |$", fontsize = 10)
    
    if double_xaxis:
        ax3 = ax.twiny()
        new_x = multiply_Gs(
            dictfilt(data['g'], key_sec_ax).values())
        scale = abs(new_x[-1] - new_x[0])
        ax3.set_xticks(new_tick_locations)
        ax3.set_xticklabels(["%.1f" % z for z in scale * new_tick_locations + scale * abs(new_x[0])])
        ax3.set_xlabel(second_axis_label, loc = 'right', fontsize = 15, labelpad = -20)
        # ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.legend(fontsize=15, frameon = False, framealpha = 0.1, loc = legend_loc)
    remove_frame(ax)
def get_plot_Gs(x_axis, include_Gs, data, nuc_loop_lists, keys, key):
    if x_axis == 'multiply':
        
        g = derive_G_multi_loop(data['g'], nuc_loop_lists)
        x_label = r"$| G+{Loop} |$"
        txt = []
        if include_Gs:
            
            for k in keys:
                if k in key:
                    txt.append( _str_G_with_key(k) +'=[' + r"${0}$".format(round(data['g'][k][0],1)) +
                               ',' + r"${0}$".format(round(data['g'][k][-1],1)) + ']')
                    continue
                
                txt.append( _str_G_with_key(k) +'=' + r"${0}$".format(round(data['g'][k][0],1)))
    
    else:
        g = data['g'][key]
        x_label = _str_G_with_key(key)
        title = ''
        txt = []
        for k in keys:
            if k == key:
                continue
            txt.append( _str_G_with_key(k) +'=' + str(round(data['g'][k][0], 3)))

    return g, txt

def get_extremes_from_all_dfs(filename_list, name_list, c_list):
    maxs = [] ; mins = []
    for i in range(len(filename_list)):
        
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        
        maxs.append(np.max(data[name_list[i],c_list[i]]))
        mins.append(np.min(data[name_list[i],c_list[i]]))
    vmax = max(maxs) ; vmin = min(mins)
    return vmax, vmin

def synaptic_weight_transition_multiple_circuits(filename_list, name_list, label_list, color_list, 
                                                 y_list,  marker_c_list = None,  colormap = 'hot', 
                                                 x_axis = 'multiply', title = "", edgecolor = 'k',
                                                 x_label = "G",  leg_loc = 'upper right', g_key = None,
                                                 vline_txt = True, colorbar = True, ylabel = 'frequency(Hz)',
                                                 vline_width = 2, lw = 1, xlim = None, markersize = 80,
                                                 alpha_transient = 1):
    
    
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)
    
    if colorbar:
        vmax, vmin = get_extremes_from_all_dfs(filename_list, name_list, marker_c_list)
    
    for i in range(len(filename_list)):
        
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        
        if x_axis == 'multiply':
            
            g = abs(Product_G(data))
            g_transient =  data['g_loop_transient']
            g_stable = data['g_loop_stable']
            x_label = r'$\vert G_{Loop} \vert$'
            
        else:
            g = data['g'][g_key]
            g_transient = data[name_list[i],'g_transient'] * data['G_ratio'][g_key]
            g_stable = data[name_list[i],'g_stable'] * data['G_ratio'][g_key]

        ax.plot(g , 
                np.squeeze(data[(name_list[i],y_list[i])]),
                c = color_list[i], lw = lw, 
                label= label_list[i], zorder=1)
        
        if colorbar:
            img = ax.scatter(g , 
                             np.squeeze(data[(name_list[i],y_list[i])]),
                             vmin = vmin, vmax = vmax, 
                             c=data[(name_list[i], marker_c_list[i])], 
                             cmap=plt.get_cmap(colormap), lw = 1, 
                             edgecolor = edgecolor, zorder = 2, s = markersize)
        else:
            img = ax.scatter(g , 
                             np.squeeze(data[(name_list[i],y_list[i])]),
                             c = color_list[i],  lw = 1, 
                              s = markersize, edgecolor = 'k')
        ax.axvline(g_transient , alpha = alpha_transient,
                   linestyle = '-.', c = color_list[i],
                    lw = vline_width) 
        
        ax.axvline(g_stable , 
                   c = color_list[i], lw = vline_width) 
        
    if vline_txt :
        shift = (g[-1] - g[0])/20
        ax.text(g_stable - shift, 0.6, 
                'Stable',
                fontsize=18, rotation = -90)
        ax.text(g_transient- shift , 0.6, 
                'Transient', 
                fontsize=18, rotation = -90)
        
    ax.set_xlabel(x_label,fontsize = 20)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.set_title(title,fontsize=20)
    ax_label_adjust(ax, fontsize = 18, nbins = 8)
    ax.legend(fontsize=15, frameon = False, framealpha = 0.1, loc = leg_loc)
    ax.tick_params(axis='x', length = 10)
    ax.tick_params(axis='y', length = 8)
    fig = set_y_ticks(fig, [0, 50, 100])
    remove_frame(ax)
    if xlim != None:
        ax.set_xlim(xlim)
    if colorbar:
        
        axins1 = inset_axes(ax,
                        width="5%",  # width = 50% of parent_bbox width
                        height="70%",  # height : 5%
                        loc='center right', 
                        borderpad=-1)#, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),)
        clb = fig.colorbar(img, cax=axins1, orientation="vertical")
        clb.ax.locator_params(nbins=4)
        clb.set_label('% Oscillation', 
                      labelpad=20, y=.5, 
                      rotation=-90,fontsize=15)
    
    return fig

def set_max_dec_tick(ax):
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
def set_n_ticks(ax, nx, ny):
    ax.xaxis.set_major_locator(MaxNLocator(nx)) 
    ax.yaxis.set_major_locator(MaxNLocator(ny)) 
    
def remove_tick_lines(ax):
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
def set_y_ticks(fig, label_list):
    
    for ax in fig.axes:
        y_formatter = FixedFormatter([str(x) for x in label_list])
        y_locator = FixedLocator(label_list)
        ax.yaxis.set_major_formatter(y_formatter)
        ax.yaxis.set_major_locator(y_locator)
        
    return fig

def set_y_lim_all_axis(fig, ylim):
    
    for ax in fig.axes:
        ax.set_ylim(ylim)
        
    return fig

def set_x_lim_all_axis(fig, ylim):
    
    for ax in fig.axes:
        ax.set_xlim(ylim)
        
    return fig

def remove_legend_all_axis(fig):
    
    for ax in fig.axes:
        ax.get_legend().remove()
        
    return fig

def remove_title_all_axis(fig):
    
    for ax in fig.axes:
        ax.set_title('')
        
    return fig

def set_x_ticks(fig, label_list):
    
    for ax in fig.axes:
        x_formatter = FixedFormatter([str(x) for x in label_list])
        x_locator = FixedLocator(label_list)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(x_locator)
        
    return fig

def remove_all_x_labels(fig):
    
    for ax in fig.axes:
        ax.axes.xaxis.set_ticklabels([])
        ax.set_xlabel('')
        
    return fig
   


def multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label, 
                                   key = None, tau_2_ind = None, ylabelpad = -5, title = '', c_label = '', ax = None):
    fig, ax = get_axes (ax)
    
    for i in range(len(filename_list)):

        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec =  data['tau'][key][:, tau_2_ind]

        y_spec = data[(name_list[i], y_list[i])][:,tau_2_ind]. reshape(-1,)
        ax.plot(x_spec,y_spec, '-o', c = color_list[i], lw = 3, label= label_list[i],zorder = 1)#, path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()])
        ax.set_xlabel(x_label,fontsize = 20)
        ax.set_ylabel(y_label,fontsize = 20,labelpad=ylabelpad)
        ax.set_title(title,fontsize = 20)
        ax_label_adjust(ax, fontsize = 20)
        remove_frame(ax)
    plt.legend(fontsize = 20)
    plt.show()
    return fig, ax

def multi_plot_as_f_of_timescale_shared_colorbar(y_list, color_list, c_list, label_list,name_list,filename_list,x_label,y_label, 
                                    g_tau_2_ind = None, g_ratio_list = [], ylabelpad = -5, colormap = 'hot', title = '', c_label = ''):
    
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(111)
    vmin, vmax = get_extremes_from_all_dfs(filename_list, name_list, c_list)
    
    for i in range(len(filename_list)):
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec =  data['tau'][:,:,0][:,0]
        y_spec = data[(name_list[i], y_list[i])][:,g_tau_2_ind]*g_ratio_list[i]
        c_spec = data[(name_list[i], c_list[i])][:,g_tau_2_ind]*g_ratio_list[i]
        ax.plot(x_spec,y_spec,c = color_list[i], lw = 3, label= label_list[i],zorder = 1)
        img = ax.scatter(x_spec,y_spec,vmin = vmin, vmax = vmax, c=c_spec, cmap=plt.get_cmap(colormap),lw = 1,edgecolor = 'k',s =90,zorder = 2)
        # plt.axvline(g_transient, c = color_list[i])  # to get the circuit g which is the muptiplication
        ax.set_xlabel(x_label,fontsize = 20)
        ax.set_ylabel(y_label,fontsize = 20,labelpad=ylabelpad)
        ax.set_title(title,fontsize = 20)
        # ax.set_xlim(limits['x'])
        # ax.set_ylim(limits['y'])
        ax_label_adjust(ax, fontsize = 20, nbins = 5, ybins = 6 )
    ax.margins(0)
    clb = fig.colorbar(img)
    clb.set_label(c_label, labelpad=20, y=.5, rotation=-90,fontsize = 20)
    clb.ax.locator_params(nbins=5)
    ax.legend(fontsize = 20, frameon=False)
    return fig



def find_AUC_of_input(name,path,poisson_prop,gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude,
                      N, A, A_mvt,D_mvt,t_mvt, N_real, K_real,t_list,color_dict, G, T, t_sim, dt, synaptic_time_constant, 
                      receiving_pop_list, smooth_kern_window,oscil_peak_threshold,syn_component_weight, 
                      end_of_nonlinearity,if_plot = True):
    
    receiving_pop_list = {(name,'1') : []}
    pop_list = [1]  
    
    class Nuc_AUC(Nucleus):
        
        def cal_ext_inp(self,dt,t):
            # to have exactly one spike the whole time
            poisson_spikes = np.zeros((self.n,self.n_ext_population))
            
            if t == 10:
                ind = np.random.randint(0, self.n_ext_population - 1, size = self.n)
                poisson_spikes[ (np.arange(self.n), ind) ] = 1
                print(np.sum(poisson_spikes, axis = 1))
                
            self.syn_inputs['ext_pop','1'] =  ( np.sum(poisson_spikes, axis = 1) / dt * \
                                               self.syn_weight_ext_pop * \
                                                   self.membrane_time_constant
                                              ).reshape(-1,)
            self.I_syn['ext_pop','1'], self.I_rise['ext_pop','1'] =  exp_rise_and_decay(self.syn_inputs['ext_pop','1'], 
                                                                                        I_rise = self.I_rise['ext_pop','1'], 
                                                                                        I = self.I_syn['ext_pop','1'], 
                                                                                        tau_rise = self.tau_ext_pop['rise'],
                                                                                        tau_decay = self.tau_ext_pop['decay'])
            
            # self.I_syn['ext_pop','1'], self.I_rise['ext_pop','1'] =  _dirac_delta_input(self.syn_inputs['ext_pop','1'], 
            #                                                                             I_rise = self.I_rise['ext_pop','1'], 
            #                                                                             I = self.I_syn['ext_pop','1'], 
            #                                                                             tau_rise = self.tau_ext_pop['rise'],
            #                                                                             tau_decay = self.tau_ext_pop['decay'])
            
    nucleus = Nuc_AUC(1, gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
               synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,neuronal_model ='spiking', set_input_from_response_curve = False,
               poisson_prop =poisson_prop, init_method = 'heterogeneous', der_ext_I_from_curve = False, mem_pot_init_method= 'draw_from_data',  keep_mem_pot_all_t =False,
               ext_input_integ_method='exp_rise_and_decay', syn_input_integ_method = 'exp_rise_and_decay', path = path, save_init = False,
               syn_component_weight=syn_component_weight)
    
    # nucleus.membrane_time_constant = np.full((nucleus.n,), 5)
    nuclei_dict = {name: [nucleus]}
    
    receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                           if_plot = False, end_of_nonlinearity = end_of_nonlinearity, 
                                          set_FR_range_from_theory=False, method = 'collective', return_saved_FR_ext= False, 
                                          use_saved_FR_ext= False, normalize_G_by_N= True)
    
    nuclei_dict = run(receiving_class_dict,t_list, dt, nuclei_dict)
    
    if if_plot:
        fig2, ax2 = plt.subplots(1, 1, sharex=True, sharey=True)
        
        for nucleus_list in nuclei_dict.values():
            for nucleus in nucleus_list:
                y = np.average(nucleus.ext_input_all_t,axis=0)
                std = np.std(nucleus.ext_input_all_t,axis=0)
                ax2.plot(t_list*dt,np.average(nucleus.ext_input_all_t,axis=0),c = color_dict[nucleus.name],label = nucleus.name)
                ax2.fill_between(t_list*dt, y-std, y+std, alpha = 0.5)
        ax2.set_title('external input',fontsize = 15)
        ax2.legend()
        
    AUC = np.average(
                    [np.trapz(nucleus.ext_input_all_t[i,:], x = t_list*dt) / nucleus.membrane_time_constant[i] for i in range(N[name]) ], 
                     axis = 0)
    AUC_std = np.std(
                    [np.trapz(nucleus.ext_input_all_t[i,:], x = t_list*dt) / nucleus.membrane_time_constant[i] for i in range(N[name])],
                     axis = 0)
    
    print("AUC of one spike = {:.3f} +/- {:.3f} mV".format( AUC, AUC_std))
    return AUC, AUC_std


def save_df_dict_to_excel_sheets(df_dict, filepath):
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')

    for df_name, df in df_dict.items():
        
        df.to_excel(writer, sheet_name=df_name)
        
    writer.save()

def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

def get_max_value_dict(dictionary):
    ''' return maximum length between items of a dictionary'''
    return max([ max(v) for k, v in dictionary.items()])

def read_sheets_of_xls_data(filepath, sheet_name_extra = 'Response'):
    
    FR_df = {}
    xls = pd.ExcelFile(filepath)
    sheet_name_list = xls.sheet_names
    for sheet_name in sheet_name_list:
        
        name = sheet_name.split('_')[-1].replace( sheet_name_extra, '')
        FR_df[name] = pd.read_excel(xls, sheet_name, header = [0])
        
    return FR_df

from itertools import chain

def get_max_min_from_column_in_df_dict(df_dict, colname):
    
    ''' get max and min within a certain column among dfs of a dictionary '''
    
    maximum = max(chain.from_iterable( df[colname] for df in df_dict.values()))
    minimum = min(chain.from_iterable( df[colname] for df in df_dict.values()))
    return np.array([minimum, maximum])

def parameterscape(x_list, y_list, name_list, markerstyle_list, freq_dict, color_dict, peak_significance,
                   size_list, xlabel, ylabel, title = '', label_fontsize = 18, cmap = 'jet', tick_size = 15,
                   annotate = True, ann_name = 'Proto', clb_tick_size  = 20, plot_acc_to_sig = True):
    
    fig, ax = plt.subplots(1, 1)
    for i, y in enumerate(y_list):
        
        for j, x in enumerate(x_list):
            
            for name, ms, s in zip(name_list, markerstyle_list, size_list):
                if plot_acc_to_sig:
                    if peak_significance[name][i,j] == True:
                        img = ax.scatter(x, y, marker = ms, c = color_dict[name][i,j], 
                                         s = s, cmap = cmap, edgecolors = 'k', 
                                         vmax = max_in_dict(color_dict), 
                                         vmin = min_in_dict(color_dict))
                    else:
                        
                        img = ax.scatter(x, y, marker = ms, c = 'grey', 
                                         s = s, edgecolors = 'k')
                else:
                    img = ax.scatter(x, y, marker = ms, c = color_dict[name][i,j], 
                                     s = s, cmap = cmap, edgecolors = 'k', 
                                     vmax = max_in_dict(color_dict), 
                                     vmin = min_in_dict(color_dict))
                if annotate:
                    ax.annotate(int(freq_dict[ann_name][i,j]), (x,y), color = 'k')
                
    ax.set_xlim(x_list[-1] + (x_list[1] - x_list[0]),
                 x_list[0] - (x_list[1] - x_list[0]))
    
    ax.set_ylim(y_list[-1] + (y_list[1] - y_list[0]),
                y_list[0] - (y_list[1] - y_list[0]))
    
    ax.set_xlabel(xlabel, fontsize = label_fontsize)
    ax.set_ylabel(ylabel, fontsize = label_fontsize)
    ax.set_title(title, fontsize = label_fontsize)
    set_n_ticks(ax, 4, 4)
    remove_tick_lines(ax)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    fig = set_y_ticks(fig, ax.get_xticks().tolist()[:-1])
    clb = fig.colorbar(img, shrink=0.5)
    clb.set_label(title, labelpad=-45, y=0.5, rotation=-90, fontsize = label_fontsize)
    clb.ax.tick_params(labelsize=clb_tick_size )
    set_max_dec_tick(ax)
    clb.ax.yaxis.tick_right()
    ax.invert_xaxis()
    remove_whole_frame(ax)
    
    return fig
    


def highlight_example_pts(fig, examples_ind, x_list, y_list, size_list, highlight_color = 'w', alpha = 0.5):
    ax = fig.gca()
    for key, ind in examples_ind.items():
        s = size_list[0] * 1.5
        x = x_list[ind[0]] 
        y = y_list[ind[1]] 
        shift_x = abs(x_list[1] - x_list[0])
        shift_y = abs(y_list[1] - y_list[0])
        
        ax.scatter(x, y, marker = 'o', c = highlight_color, alpha = alpha ,
                    s = s, edgecolors = None)
        ax.annotate(key, (x + shift_x/5, y - shift_y/15), color = 'w', size = 18)
    return fig


def plot_PSD_of_example_pts(data_all, examples_ind,  x_list, y_list, name_list, color_dict):
    
    
    for key, ind in examples_ind.items():
        fig, ax = plt.subplots()
        for name in name_list:
            
            f = data_all[(name, 'f')][ind[1], ind[0], 0, : ]
            pxx = np.average(data_all[(name, 'pxx')][ind[1], ind[0], :, : ], axis = 0)
            pxx = normalize_PSD( pxx, f)
            peak_freq = np.round(data_all[(name, 'peak_freq_all_runs')][ind[1], ind[0]] , 1)
            ax.plot(f, pxx, c = color_dict[name], label = name + ' ' +  str(peak_freq) + ' Hz', lw=1.5)
        
        ax.set_xlim(0,80)
        ax.legend()
        ax.set_title(key, fontsize = 15)
        
        
def plot_pop_act_and_PSD_of_example_pts(data, name_list, examples_ind, x_list, y_list, dt, color_dict, Act, state = 'awake_rest',
                                        plt_duration = 600, run_no = 0, window_ms = 5):
    
    n_exmp = len(examples_ind)
    fig = plt.figure( figsize=(12, 20) ) 
    outer = gridspec.GridSpec( n_exmp, 1, wspace=0.2, hspace=0.2)
    
    for i, (key, ind) in enumerate(examples_ind.items()):
    
        inner = gridspec.GridSpecFromSubplotSpec( 1, 2, width_ratios=[1, 3],
                                                 subplot_spec=outer[i], 
                                                 wspace=0.1, hspace=0.1)
        ax_PSD = plt.Subplot(fig, inner[0])
        ax_pop_act = plt.Subplot(fig, inner[1])
        
        for name in name_list:
        
            f = data[(name, 'f')][ind[1], ind[0], 0, : ]
            pxx = np.average(data[(name, 'pxx')][ind[1], ind[0], :, : ], axis = 0)
            pxx = normalize_PSD( pxx, f) * 100
            peak_freq = np.round(data[(name, 'peak_freq_all_runs')][ind[1], ind[0]] , 1)
            ax_PSD.plot(f, pxx, c = color_dict[name], label = name + ' ' +  str(peak_freq) + ' Hz', lw=1.5)
        
            duration = data[(name, 'pop_act')].shape[-1]
            
            pop_act = data[(name, 'pop_act')][ind[1], ind[0], run_no, duration - int( plt_duration/dt) : duration]
            pop_act = moving_average_array(pop_act, int(window_ms / dt))
            t_list = np.arange( duration - int( plt_duration/ dt), duration) * dt
            ax_pop_act.plot( t_list, pop_act, c = color_dict[name], lw = 1.5)
            ax_pop_act.plot(t_list, np.full_like(t_list, Act[state][name]), '--', 
                                                 c = color_dict[name],lw = 1, alpha=0.8 )

        ax_PSD.set_xlim(0, 80)
        ax_PSD.legend(fontsize = 8, frameon = False, loc = 'upper right')
        ax_PSD.set_ylabel(key, fontsize = 20, rotation = 0, labelpad = 10)
        # ax_pop_act.set_title(key, fontsize = 15)
        fig.add_subplot(ax_PSD)
        fig.add_subplot(ax_pop_act)
        fig.text(0.6, 0.085, 'Time (ms)', ha='center', fontsize = 15)
        fig.text(0.95, 0.5, 'firing rate (spk/s)', va='center', rotation='vertical', fontsize = 15)
        fig.text(0.2, 0.085, 'frequency', ha='center', fontsize = 15)
        fig.text(0.05, 0.5, 'Normalized Power' + r'$(\times 10^{-2})$', va='center', rotation='vertical',fontsize = 15)
        
    return fig


def plot_pop_act_and_PSD_of_example_pts_1d(data, name_list, examples_ind, x_list, y_list, dt, color_dict, Act, state = 'awake_rest',
                                        plt_duration = 600, run_no = 0, window_ms = 5):
    
    n_exmp = len(examples_ind)
    fig = plt.figure( figsize=(12, 20) ) 
    outer = gridspec.GridSpec( n_exmp, 1, wspace=0.2, hspace=0.2)
    
    for i, (key, ind) in enumerate(examples_ind.items()):
    
        inner = gridspec.GridSpecFromSubplotSpec( 1, 2, width_ratios=[1, 3],
                                                 subplot_spec=outer[i], 
                                                 wspace=0.1, hspace=0.1)
        ax_PSD = plt.Subplot(fig, inner[0])
        ax_pop_act = plt.Subplot(fig, inner[1])
        
        for name in name_list:
        
            f = data[(name, 'f')][ind, 0, : ]
            pxx = np.average(data[(name, 'pxx')][ind, :, : ], axis = 0)
            # print(data[(name, 'pxx')].shape, data[(name, 'pxx')][ind, :, : ].shape)
            pxx = normalize_PSD( pxx, f) * 100
            peak_freq = np.round( np.average(data[(name, 'base_freq')][ind, :]) , 1)
            ax_PSD.plot(f, pxx, c = color_dict[name], label = name + ' ' +  str(peak_freq) + ' Hz', lw=1.5)
        
            duration = data[(name, 'pop_act')].shape[-1]
            
            pop_act = data[(name, 'pop_act')][ind, run_no, duration - int( plt_duration/dt) : duration]
            pop_act = moving_average_array(pop_act, int(window_ms / dt))
            t_list = np.arange( duration - int( plt_duration/ dt), duration) * dt
            ax_pop_act.plot( t_list, pop_act, c = color_dict[name], lw = 1.5)
            ax_pop_act.plot(t_list, np.full_like(t_list, Act[state][name]), '--', 
                                                 c = color_dict[name],lw = 1, alpha=0.8 )

        ax_PSD.set_xlim(0, 80)
        ax_PSD.legend(fontsize = 8, frameon = False, loc = 'upper right')
        ax_PSD.set_ylabel(key, fontsize = 20, rotation = 0, labelpad = 10)
        # ax_pop_act.set_title(key, fontsize = 15)
        fig.add_subplot(ax_PSD)
        fig.add_subplot(ax_pop_act)
        fig.text(0.6, 0.05, 'Time (ms)', ha='center', fontsize = 15)
        fig.text(0.92, 0.5, 'firing rate (spk/s)', va='center', rotation=-90, fontsize = 15)
        fig.text(0.2, 0.05, 'frequency', ha='center', fontsize = 15)
        fig.text(0.05, 0.5, 'Normalized Power' + r'$(\times 10^{-2})$', va='center', rotation='vertical',fontsize = 15)
        
    return fig

def plot_surf(x_list,y_list, z_arr):
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X,Y = np.meshgrid( x_list,y_list)
    surf = ax.plot_surface(X, Y, z_arr, cmap = cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)#, location = 'left')
    return fig, ax

def plot_spec_as_surf(g_list, freq_list, pxx_2d_arr, normalize_PSD = True, 
                      xlabel = '', ylabel = '', zlabel = ''):
    if normalize_PSD:
        pxx_2d_arr = normalize_PSD_2d(pxx_2d_arr, freq_list, axis = 0)
    fig, ax = plot_surf(g_list, freq_list,pxx_2d_arr)
    ax.set_xlabel(xlabel, fontsize = 15)
    ax.set_ylabel(ylabel, fontsize = 15)
    ax.set_zlabel(zlabel, fontsize = 15)
    return fig, ax

def plot_theory_FR_sim_vs_FR_ext(name, poisson_prop, x_range, neuronal_consts, start_epsilon = 10**(-10), x_val = 'FR', ax = None, lw = 3):
    fig, ax = get_axes(ax)
    start_theory = (((neuronal_consts[name]['spike_thresh']['mean']-neuronal_consts[name]['u_rest'])
              / (poisson_prop[name]['g']*poisson_prop[name]['n']*neuronal_consts[name]['membrane_time_constant']['mean'])) + start_epsilon)
    x1 = np.linspace( start_theory, start_theory + 0.0001, 1000).reshape(-1,1)
    end = x_range[1] / poisson_prop[name]['g'] / poisson_prop [name ]['n']
    x_theory = np.concatenate( ( x1, np.geomspace(x1 [ -1], end, 10)))
    y_theory = FR_ext_theory(neuronal_consts[name]['spike_thresh']['mean'], 
                              neuronal_consts[name]['u_rest'], 
                              neuronal_consts[name]['membrane_time_constant']['mean'], poisson_prop[name]['g'], x_theory, poisson_prop[name]['n'])
    if x_val == 'FR':
        x = x_theory * 1000
        xlim = [x_range[0] / poisson_prop[name]['g'] / poisson_prop [name ]['n'] * 1000, x_range[1] / poisson_prop[name]['g'] / poisson_prop [name ]['n'] * 1000]
        ax.set_xlabel(r'$FR_{ext} \; (Hz)$',fontsize=15)
    elif x_val == 'I_ext':
        x = x_theory * poisson_prop[name]['g'] * poisson_prop [name ]['n'] * poisson_prop[name]['tau']['decay']['mean']
        xlim = [x_range[0] * poisson_prop[name]['tau']['decay']['mean'], x_range[1] * poisson_prop[name]['tau']['decay']['mean']]
        ax.set_xlabel(r'$I_{ext} \; (mV)$',fontsize=15)
    ax.plot(x, y_theory * 1000,label='theory', c= 'lightcoral' , markersize = 6, markeredgecolor = 'grey', lw = lw)
    ax.set_ylabel(r'$FR\; (Hz)$',fontsize=15)
    ax.set_xlim(xlim)
    ax_label_adjust(ax)
    remove_frame(ax)
    # ax.legend()


def _generate_filename_3_nuclei(nuclei_dict, G_dict, noise_variance, fft_method):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0],3)) + '--' + str(round(G[('D2', 'FSI')][-1],3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '--' + str(round(G[('Proto', 'D2')][-1],3)), 
          str(round(G[('FSI', 'Proto')][0],3)) + '--' + str(round(G[('FSI', 'Proto')][-1],3))]
    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    
# def synaptic_weight_transition_multiple_circuits(filename_list, name_list, label_list, color_list, g_cte_ind, g_ch_ind, 
#                                                  y_list, c_list, colormap = 'hot', x_axis = 'multiply', title = "", 
#                                                  x_label = "G", x_scale_factor = 1, leg_loc = 'upper right', 
#                                                  vline_txt = True, colorbar = True):
    
    
#     fig = plt.figure(figsize=(8,7))
#     ax = fig.add_subplot(111)
    
#     vmax, vmin = get_extremes_from_all_dfs(filename_list, name_list, c_list)
    
#     for i in range(len(filename_list)):
        
#         pkl_file = open(filename_list[i], 'rb')
#         data = pickle.load(pkl_file)
        
#         if x_axis == 'multiply':
            
#             g = np.squeeze(data['g'][:,:,0] * 
#                            data['g'][:,:,1])
#             g_transient = data[name_list[i],'g_transient_boundary'][0][g_ch_ind[i]]* data[name_list[i],'g_transient_boundary'][0][g_cte_ind[i]] 
#             g_stable = data[name_list[i],'g_stable_boundary'][0][g_ch_ind[i]]* data[name_list[i],'g_stable_boundary'][0][g_cte_ind[i]] 
#             x_label = r'$G_{Loop}$'
            
#         else:
#             g = np.squeeze(data['g'][:,:,g_ch_ind[i]])
#             g_transient = data[name_list[i],'g_transient_boundary'][0][g_ch_ind[i]]
#             g_stable = data[name_list[i],'g_stable_boundary'][0][g_ch_ind[i]]

#         ax.plot(g * x_scale_factor, 
#                 np.squeeze(data[(name_list[i],y_list[i])]),
#                 c = color_list[i], lw = 3, 
#                 label= label_list[i], zorder=1)
        
#         img = ax.scatter(g * x_scale_factor, 
#                          np.squeeze(data[(name_list[i],y_list[i])]),
#                          vmin = vmin, vmax = vmax, 
#                          c=data[(name_list[i],c_list[i])], 
#                          cmap=plt.get_cmap(colormap), lw = 1, 
#                          edgecolor = 'k', zorder = 2, s = 80)
        
#         ax.axvline(g_transient * x_scale_factor, 
#                    linestyle = '-.', c = color_list[i],
#                    alpha = 0.3, lw = 2) 
        
#         ax.axvline(g_stable * x_scale_factor, 
#                    c = color_list[i], lw = 2) 
        
#     if vline_txt :
#         ax.text(g_stable * x_scale_factor-0.5, 0.6, 
#                 'Stable oscillation',
#                 fontsize=18, rotation = -90)
#         ax.text(g_transient * x_scale_factor, 0.6, 
#                 'Transient Oscillation', 
#                 fontsize=18, rotation = -90)
        
#     ax.set_xlabel(x_label,fontsize = 20)
#     ax.set_ylabel('frequency(Hz)',fontsize=20)
#     ax.set_title(title,fontsize=20)
#     ax_label_adjust(ax, fontsize = 18, nbins = 8)
#     ax.legend(fontsize=15, frameon = False, framealpha = 0.1, loc = leg_loc)
#     remove_frame(ax)
    
#     if colorbar:
        
#         axins1 = inset_axes(ax,
#                         width="5%",  # width = 50% of parent_bbox width
#                         height="70%",  # height : 5%
#                         loc='center right')#,borderpad=-1)#, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),)
#         clb = fig.colorbar(img, cax=axins1, orientation="vertical")
#         clb.ax.locator_params(nbins=4)
#         clb.set_label('% Oscillation', 
#                       labelpad=20, y=.5, 
#                       rotation=-90,fontsize=15)
    
#     return fig


# def sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant, nuclei_dict, 
#                       syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict, 
#                       t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill=True):
    
#     def set_time_scale( nuclei_dict, synaptic_time_constant):
#         for nucleus_list in nuclei_dict.values():
#             for nucleus in nucleus_list:
#                 nucleus.set_synaptic_time_scales(synaptic_time_constant) 
#         return nuclei_dict
    
#     t_decay_series_1 = list(syn_decay_dict['tau_1']['tau_list']) ; t_decay_series_2 = list(syn_decay_dict['tau_2']['tau_list'])
#     data  = create_data_dict(nuclei_dict, [len(t_decay_series_1), len(t_decay_series_2)], 2,len(t_list))    
#     count =0 ; i=0
#     for t_decay_1 in t_decay_series_1:
#         j = 0
#         for t_decay_2 in t_decay_series_2:

            # synaptic_time_constant = extract_syn_time_constant_from_dict(synaptic_time_constant, syn_decay_dict, t_decay_1, t_decay_2)
#             nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
#             nuclei_dict = set_time_scale(nuclei_dict,synaptic_time_constant)
#             n_half_cycle,g_transient,g_stable, nuclei_dict, if_stable = find_oscillation_boundary(g_list, nuclei_dict, G,G_ratio_dict, A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle =  lim_n_cycle , find_stable_oscill=find_stable_oscill)

#             run(receiving_class_dict,t_list, dt, nuclei_dict)
#             data['tau'][i,j,:] = [t_decay_1,t_decay_2]
            
#             nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
#     #                plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$\tau_{GABA_A}$ = "+ str(round(gaba_a,2))+r' $\tau_{GABA_B}$ ='+str(round(gaba_b,2)))
#             for nucleus in nucleus_list:
#                 data[(nucleus.name, 'g_transient')][i,j] = g_transient
#                 data[(nucleus.name, 'g_stable')][i,j] = g_stable
#                 data[(nucleus.name,'trans_n_half_cycle')][i,j] = n_half_cycle
#                 data[(nucleus.name,'trans_pop_act')][i,j,:] = nucleus.pop_act
#                 _,_, data[nucleus.name,'trans_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                 _,_, data[nucleus.name,'trans_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
            
#             if find_stable_oscill: # only run if you want to checkout the stable oscillatory regime
            
#                 for k,g_ratio in G_ratio_dict.items():
#                     G[k] = g_stable * g_ratio
#                 # G[('STN','Proto')] = g_stable
#                 # G[('Proto','Proto')] = g_stable * g_ratio
#                 nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
#                 run(receiving_class_dict,t_list, dt, nuclei_dict)
                
#                 for nucleus in nucleus_list:
#                     _,_, data[nucleus.name,'stable_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                     _,_, data[nucleus.name,'stable_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)

#             count +=1
#             print(count, "from ", len(t_decay_series_1)*len(t_decay_series_2))
#             j+=1
#         i +=1
#     output = open(filename, 'wb')
#     pickle.dump(data, output)
#     output.close()
    
# def sweep_time_scales(g_list, G_ratio_dict, synaptic_time_constant, nuclei_dict, 
#                       syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict, 
#                       t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill=True):
    
#     def set_time_scale( nuclei_dict, synaptic_time_constant):
#         for nucleus_list in nuclei_dict.values():
#             for nucleus in nucleus_list:
#                 nucleus.set_synaptic_time_scales(synaptic_time_constant) 
#         return nuclei_dict
    
#     t_decay_series_1 = list(syn_decay_dict['tau_1']['tau_list']) ; t_decay_series_2 = list(syn_decay_dict['tau_2']['tau_list'])
#     data  = create_data_dict(nuclei_dict, [len(t_decay_series_1), len(t_decay_series_2)], 2,len(t_list))    
#     count =0 ; i=0
#     for t_decay_1 in t_decay_series_1:
#         j = 0
#         for t_decay_2 in t_decay_series_2:

            # synaptic_time_constant = extract_syn_time_constant_from_dict(synaptic_time_constant, syn_decay_dict, t_decay_1, t_decay_2)
#             nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
#             nuclei_dict = set_time_scale(nuclei_dict,synaptic_time_constant)
#             n_half_cycle,g_transient,g_stable, nuclei_dict, if_stable = find_oscillation_boundary(g_list, nuclei_dict, G,G_ratio_dict, A, A_mvt,t_list,dt, receiving_class_dict, D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle =  lim_n_cycle , find_stable_oscill=find_stable_oscill)

#             run(receiving_class_dict,t_list, dt, nuclei_dict)
#             data['tau'][i,j,:] = [t_decay_1,t_decay_2]
            
#             nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
#     #                plot(Proto, STN, dt, t_list, A, A_mvt, t_mvt, D_mvt,plot_ob = None, title = r"$\tau_{GABA_A}$ = "+ str(round(gaba_a,2))+r' $\tau_{GABA_B}$ ='+str(round(gaba_b,2)))
#             for nucleus in nucleus_list:
#                 data[(nucleus.name, 'g_transient')][i,j] = g_transient
#                 data[(nucleus.name, 'g_stable')][i,j] = g_stable
#                 data[(nucleus.name,'trans_n_half_cycle')][i,j] = n_half_cycle
#                 data[(nucleus.name,'trans_pop_act')][i,j,:] = nucleus.pop_act
#                 _,_, data[nucleus.name,'trans_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                 _,_, data[nucleus.name,'trans_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
            
#             if find_stable_oscill: # only run if you want to checkout the stable oscillatory regime
            
#                 for k,g_ratio in G_ratio_dict.items():
#                     G[k] = g_stable * g_ratio
#                 # G[('STN','Proto')] = g_stable
#                 # G[('Proto','Proto')] = g_stable * g_ratio
#                 nuclei_dict = reinitialize_nuclei(nuclei_dict,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
#                 run(receiving_class_dict,t_list, dt, nuclei_dict)
                
#                 for nucleus in nucleus_list:
#                     _,_, data[nucleus.name,'stable_mvt_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)
#                     _,_, data[nucleus.name,'stable_base_freq'][i,j],_ = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt ,peak_threshold = nucleus.oscil_peak_threshold, smooth_kern_window=nucleus.smooth_kern_window)

#             count +=1
#             print(count, "from ", len(t_decay_series_1)*len(t_decay_series_2))
#             j+=1
#         i +=1
#     output = open(filename, 'wb')
#     pickle.dump(data, output)
#     output.close()
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + '_G(FD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PF)= '  + gs[2] + 
              '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(nuclei_dict, figs, G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3):
    prefix = [ 'Firing_rate_', 'Power_spectrum_','Raster_' ]
    prefix = [pre_prefix[i] + prefix[i] for i in range( len(prefix))]
    prefix = ['Synaptic_weight_exploration_' + p for p in prefix]
    filename = _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method)
    for i in range( len (figs)):
        figs[i].set_size_inches(s [i], forward=False)
        figs[i].savefig(os.path.join(path, prefix[i] + filename + '.png'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        figs[i].savefig(os.path.join(path, prefix[i] + filename+ '.pdf'), dpi = 300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

def set_ylim_trans_stable_figs(fig_trans, fig_stable, ymax = [100, 100], ymin = [-4, -4]):
    
    for i, fig in enumerate( [fig_trans, fig_stable] ) :
    
        ax = fig.axes
        ax[0].set_ylim(ymin[i], ymax[i])
    
    return fig_trans , fig_stable

# def save_trans_stable_figs(fig_trans, fig_stable, path_rate, filename, figsize = (10,5)):
    
#     fig_trans.set_size_inches(figsize, forward=False)
        
#     fig_trans.savefig(os.path.join(path_rate, (filename + '_tansient_plot.png')),dpi = 300, facecolor='w', edgecolor='w',
#                     orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
#     fig_trans.savefig(os.path.join(path_rate, (filename + '_tansient_plot.pdf')),dpi = 300, facecolor='w', edgecolor='w',
#                     orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
#     fig_stable.set_size_inches(figsize, forward=False)
    
#     fig_stable.savefig(os.path.join(path_rate, (filename + '_stable_plot.png')),dpi = 300, facecolor='w', edgecolor='w',
#                     orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
#     fig_stable.savefig(os.path.join(path_rate, (filename + '_stable_plot.pdf')),dpi = 300, facecolor='w', edgecolor='w',
#                     orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

def save_trans_stable_figs(fig_trans, fig_stable_list, path_rate, filename, figsize = (10,5), ymax = [100, 100], ymin = [-4, -4]):
    

    fig_trans.set_size_inches(figsize, forward=False)
    
    for i, fig_stable in enumerate(fig_stable_list):
        fig_trans , fig_stable = set_ylim_trans_stable_figs(fig_trans, fig_stable, ymax = ymax , ymin = ymin)
        fig_stable.set_size_inches(figsize, forward=False)

        fig_stable.savefig(os.path.join(path_rate, (filename + '_stable_plot_' + str(i) + '.png')),dpi = 300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        
        fig_stable.savefig(os.path.join(path_rate, (filename + '_stable_plot_' + str(i) + '.pdf')),dpi = 300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
        fig_trans.savefig(os.path.join(path_rate, (filename + '_tansient_plot.png')),dpi = 300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        
        fig_trans.savefig(os.path.join(path_rate, (filename + '_tansient_plot.pdf')),dpi = 300, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        

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



# def run_with_transient_external_input(receiving_class_dict, t_list, dt, nuclei_dict, rest_init_filepaths,
# 									  transient_init_filepaths, A, A_trans, list_of_nuc_with_trans_inp,
# 									  t_transient=10, duration=10):
# 	'''
# 		run normaly til "t_transient" then exert an external transient input to "list_of_nuc_with_trans_inp" then resume to normal state until the end of simulation
# 	'''
# 	start = timeit.default_timer()
# 	# basal firing
# 	iterate_SNN(nuclei_dict, dt, receiving_class_dict,
# 	            t_start=0, t_end=t_transient)
# 	# transient external input to the network
# 	# , A_mvt, D_mvt, t_mvt, t_list, dt)
# 	selective_reset_ext_input(
# 	    nuclei_dict, transient_init_filepaths, list_of_nuc_with_trans_inp, A_trans)
# 	# interate through the duration of transient input
# 	iterate_SNN(nuclei_dict, dt, receiving_class_dict,
# 	            t_start=t_transient, t_end=t_transient + duration)
# 	# reset back to rest
# 	# , A_mvt, D_mvt, t_mvt, t_list, dt)
# 	selective_reset_ext_input(
# 	    nuclei_dict, rest_init_filepaths, list_of_nuc_with_trans_inp, A)
# 	# look at the decay of the transient input
# 	iterate_SNN(nuclei_dict, dt, receiving_class_dict,
# 	            t_start=t_transient + duration, t_end=t_list[-1])

# 	stop = timeit.default_timer()
# 	print("t = ", stop - start)
# 	return nuclei_dict