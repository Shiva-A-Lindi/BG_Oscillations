import os
import sys
import subprocess
import timeit
import numpy as np
import pandas as pd
import pickle
import scipy
import math
import imageio

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, FixedLocator, FixedFormatter, AutoMinorLocator
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from numpy.fft import rfft, fft, fftfreq, ifft, fftshift
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import truncexpon, skewnorm
from scipy.signal import butter, sosfilt, sosfreqz, spectrogram, sosfiltfilt, hilbert
from scipy import signal, stats
import scipy.sparse as sp

from tempfile import TemporaryFile
from pygifsicle import optimize
from PIL import Image
from astropy.stats import rayleightest




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
    
    
    if if_plot:

        plot_fitted_line(FR_ext, FR_sim, slope, intercept,  
                         FR_to_I_coef = tau * g_ext * N_ext / 1000, 
                         ax = ax, noise_var=noise_var, c=c)

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
        print('Corresponding FR_ext =', round( FR_ext, 2) )
        
        plot_fitted_sigmoid(xdata, ydata, x, desired_FR, coefs=coefs,
                            FR_to_I_coef=tau * g_ext * N_ext / 1000, 
                            ax=ax, noise_var=noise_var, c=c)
        
    return FR_ext / 1000  # FR_ext is in Hz, we want spk/ms


class Nucleus:

    def __init__(self, population_number, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance, noise_amplitude, N, A, A_mvt, name, G, T, t_sim, dt,
        synaptic_time_constant, receiving_from_list, smooth_kern_window, oscil_peak_threshold, syn_input_integ_method='exp_rise_and_decay', neuronal_model='rate',
        poisson_prop=None, AUC_of_input=None, init_method='homogeneous', ext_inp_method='const+noise', der_ext_I_from_curve=False, bound_to_mean_ratio=[0.8, 1.2],
        spike_thresh_bound_ratio= [1/20, 1/20], ext_input_integ_method='dirac_delta_input', path=None, mem_pot_init_method='uniform', plot_initial_V_m_dist=False, 
        set_input_from_response_curve=True, set_random_seed=False, keep_mem_pot_all_t=False, save_init=False, scale_g_with_N=True, syn_component_weight = None, 
        time_correlated_noise = True, noise_method = 'Gaussian', noise_tau = 10, keep_noise_all_t = False, state = 'rest', random_seed = 1996, FR_ext_specs = {},
        plot_spike_thresh_hist = False, plot_RMP_to_APth = False, external_input_bool = False, hetero_trans_delay = True,
        keep_ex_voltage_trace = False, hetero_tau = True, Act = None, upscaling_spk_counts = 2, spike_history = 'long-term',
        rtest_p_val_thresh = 0.05, refractory_period = 2, syn_w_dist = 'log-normal'):

        if set_random_seed:
            
            self.random_seed = random_seed
            np.random.seed(self.random_seed)


        n_timebins = int(t_sim/dt)
        self.name = name
        self.population_num = str(population_number)
        self.receiving_from_list = receiving_from_list[(self.name, self.population_num)]
        self.receiving_from_pop_name_list = [ pop[0] for pop in self.receiving_from_list]
        self.n = N[name]  # population size
        self.basal_firing = A[name]
        self.mvt_firing = A_mvt[name]
        self.threshold = threshold[name]
        self.gain = gain[name]
        
        # filter based on the receiving nucleus# dictfilt(synaptic_time_constant, self.trans_types) # synaptic time scale based on neuron type

        self.ext_inp_delay = ext_inp_delay
        self.K_connections = None
        sending_to_dict = find_sending_pop_dict(receiving_from_list)
        self.sending_to_dict = sending_to_dict[(self.name, self.population_num)]
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
        self.t_sim = t_sim
        self.n_timebins = n_timebins
        self.sqrt_dt = np.sqrt(dt)
        self.half_dt = dt/2
        self.synaptic_time_constant = {
            k: v for k, v in synaptic_time_constant.items() if k[1] == name}
        
        # filter based on the receiving nucleus
        self.T_specs = {k: v for k, v in T.items() 
                            if k[0] == self.name and 
                               k[1] in self.receiving_from_pop_name_list}
        

        
        self.G_heterogeneity = False
        self.synaptic_weight = {} 
        self.synaptic_weight_specs = {k: v for k, v in G.items() if k[0] == name} # filter based on the receiving nucleus
        self.create_syn_weight_mean_dict()

        if neuronal_model == 'rate':
            
            self.pop_act = np.zeros((n_timebins))  # time series of population activity

            self.external_inp_t_series = np.zeros(n_timebins)
            self.transmission_delay = {k: int( v['mean'] / dt )
                           for k, v in self.T_specs.items()}
            
            self.output = {k: np.zeros((self.n, int( T[ k[0], self.name]['mean'] / dt ) + 1)) 
                           for k in self.sending_to_dict}
            
            self.input = np.zeros((self.n))
            self.neuron_act = np.zeros((self.n))
            # external input mimicing movement
            self.mvt_ext_input = np.zeros((n_timebins))
            # the basal firing as a result of noise at steady state
            self.noise_induced_basal_firing = None
            self.oscil_peak_threshold = oscil_peak_threshold[self.name]
            self.scale_g_with_N = scale_g_with_N

        if neuronal_model == 'spiking':
            
            print('Initializing ', name)
            self.refractory_period = int(refractory_period / dt)
            self.pop_act = []  # time series of population activity
            self.syn_w_dist = syn_w_dist
            self.transmission_delay = {}
            self.tau = {}
            self.state = state
            self.tau_specs = {k: v for k, v in tau.items() 
                              if k[0] == self.name and 
                                  k[1] in self.receiving_from_pop_name_list}
            

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
            self.bound_to_mean_ratio = bound_to_mean_ratio
            self.pop_act_filtered = False
            self.syn_input_integ_method = syn_input_integ_method
            self.I_ext_0 = None
            self.noise = np.zeros(( self.n, 1))
            self.noise_method = noise_method
            self.noise_tau = noise_tau 
            self.external_input_bool = external_input_bool
            self.beta_stim = False
            self.beta_stim_method = None
            self.get_max_spike_history_t(T, dt)
            self.neuron_spike_phase_hist = {} # to store the mean phase over all cycles for each neuron size = (n_neuron x n_phase_bins)
            self.phase_bins = None
            self.rtest_p_val_thresh = rtest_p_val_thresh
            self.rtest_passed_neuron_ind = None
            self.spike_history = spike_history

            if spike_history == 'short-term':
                
                self.spikes = np.zeros((self.n, self.history_duration), dtype=int) ## limited spike history 
                max_FR = max( {state: FR[name] for state, FR in Act.items() if name in FR}.values())
                self.spike_times = np.zeros((self.n, 
                                             int(t_sim * dt * max_FR * upscaling_spk_counts)), dtype=int)
                self.ind_last_spike = np.zeros(self.n)
                
            elif spike_history == 'long-term':
                
                self.spikes = np.zeros((self.n, int(t_sim/dt)), dtype=int)
                
            elif spike_history == 'sparse':
                self.spikes = sp.lil_array((self.n, int(t_sim/dt)), dtype=int)
                
            if keep_ex_voltage_trace:
                
                self.voltage_trace = np.zeros(n_timebins + 1)
                self.representative_inp = {k: np.zeros( (n_timebins, self.pre_n_components[k[0]]) ) 
                                           for k in self.receiving_from_list}
                self.representative_inp['ext_pop', '1'] = np.zeros(n_timebins)
                self.ext_input_all_t = np.zeros((self.n, n_timebins))
                
            if external_input_bool :
                self.external_inp_t_series = np.zeros(n_timebins)

            if self.keep_mem_pot_all_t:
                self.all_mem_pot = np.zeros((self.n, n_timebins))
                
            if keep_noise_all_t : 
                self.noise_all_t = np.zeros((self.n, n_timebins))
                
                        
            self.set_init_distribution( self.T_specs, self.tau_specs, FR_ext_specs, poisson_prop, dt, t_sim,  
                                       plot_initial_V_m_dist = plot_initial_V_m_dist, 
                                       plot_spike_thresh_hist= plot_spike_thresh_hist,
                                       plot_RMP_to_APth = plot_RMP_to_APth,
                                       hetero_trans_delay = hetero_trans_delay,
                                       hetero_tau = hetero_tau )
            
            self.normalize_synaptic_weight()
            
            self.ext_inp_method_dict = {'Poisson': self.poissonian_ext_inp,
                                        'const+noise': self.constant_ext_input_with_noise, 
                                        'constant': self.constant_ext_input}
            
            self.input_integ_method_dict = {'exp_rise_and_decay': exp_rise_and_decay,
                                            'instantaneus_rise_expon_decay': instantaneus_rise_expon_decay, 
                                            'dirac_delta_input': _dirac_delta_input}
            
            self.cal_ext_inp_method_dict = {True: self.cal_ext_inp_with_additional_inp, 
                                            False: self.cal_ext_inp}
        
            self.noise_generator_dict = { 'Gaussian' : noise_generator,
                                          'Ornstein-Uhlenbeck': OU_noise_generator}
            
            self.save_new_spikes_method_dict = {'short-term': self.save_new_spikes_short_history,
                                                'long-term': self.save_new_spikes,
                                                'sparse': self.save_new_spikes}
            
            self.cal_synaptic_input_method_dict = {'short-term': self.cal_synaptic_input_short_history,
                                                   'long-term': self.cal_synaptic_input_long_history,
                                                   'sparse' : self.cal_synaptic_input_sparse_history}
            
            self.get_spike_times_of_neuron_method_dict = {'short-term': self.get_spike_times_of_neuron_short_history,
                                                          'long-term': self.get_spike_times_of_neuron_long_history,
                                                          'sparse' : self.get_spike_times_of_neuron_sparse_history}
    def get_max_spike_history_t(self, T, dt):
        
        max_transmission_delays = {k: v['truncmax'] for k,v in  T.items()}
        self.history_duration = int(
                max(max_transmission_delays.values()) 
                / dt)
        
    def create_syn_weight_mean_dict(self):
        
        if len(self.receiving_from_list) > 0:
            
            if type(self.synaptic_weight_specs[self.name, self.receiving_from_list[0][0]]) is not dict :
                
                self.synaptic_weight = self.synaptic_weight_specs
            else:
                self.G_heterogeneity = True
                self.synaptic_weight = {k: v['mean'] for k, v in self.synaptic_weight_specs.items()}
                
    def set_init_distribution(self, T_specs, tau_specs, FR_ext_specs, poisson_prop, dt, 
                              t_sim, plot_initial_V_m_dist = False,
                              plot_spike_thresh_hist = False, plot_RMP_to_APth = False,
                              keep_ex_voltage_trace = False, hetero_trans_delay = True,
                              hetero_tau = True):
        
        self.FR_ext_specs = FR_ext_specs

        if self.init_method == 'homogeneous':
            
            self.initialize_homogeneously( T_specs, tau_specs, poisson_prop, dt)
            self.FR_ext = 0
            
        elif self.init_method == 'heterogeneous':
            
            self.initialize_heterogeneously(poisson_prop, dt, t_sim, self.spike_thresh_bound_ratio, 
                                            *self.bound_to_mean_ratio,  plot_initial_V_m_dist=plot_initial_V_m_dist,
                                            plot_spike_thresh_hist= plot_spike_thresh_hist, 
                                            plot_RMP_to_APth = plot_RMP_to_APth, 
                                            keep_ex_voltage_trace = keep_ex_voltage_trace,
                                            hetero_trans_delay = hetero_trans_delay,
                                            hetero_tau = hetero_tau)
            
            self.FR_ext = np.zeros(self.n)

          
    def initialize_membrane_time_constant(self, lower_bound_perc=0.8, upper_bound_perc=1.2, 
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
            plot_histogram(self.membrane_time_constant, bins = 25, 
                           title = self.name, xlabel = r'$\tau_{m} \; (ms)$')
            
            
    def initialize_synaptic_time_constant(self, dt, tau_specs, lower_bound_perc=0.8, upper_bound_perc=1.2,
                                          bins=50, color='grey', tc_plot = 'decay', syn_element_no = 0, 
                                          plot_syn_tau_hist = False, hetero_tau = True):
        
        ''' initialize synaptic time constant with a truncated normal distribution
            Note: dt incorporated in tau for time efficiency
        '''
        
        if len(self.receiving_from_pop_name_list) > 0:
            
            
            if hetero_tau:
                
                self.initialize_synaptic_time_constant_heterogeneously( dt, tau_specs, bins=bins, color=color, 
                                                          tc_plot = tc_plot, syn_element_no = syn_element_no, 
                                                          plot_syn_tau_hist = plot_syn_tau_hist)
            else:
                
                self.initialize_synaptic_time_constant_homogeneously(dt, tau_specs)
                
                
    def initialize_synaptic_time_constant_homogeneously(self, dt, tau_specs):
        
        self.tau ={ proj: 
                   {tc: np.array(v['mean']) / dt for tc, v in tc_val.items()} 
                   for proj, tc_val in tau_specs.items()}
            
            
    def initialize_synaptic_time_constant_heterogeneously(self, dt, tau_specs, bins=50, color='grey', 
                                                          tc_plot = 'decay', syn_element_no = 0, 
                                                          plot_syn_tau_hist = False):
        
        tc_list = list (tau_specs[ list(tau_specs.keys()) [0] ].keys() ) 
        for key, val in tau_specs.items():
            
            self.tau[key] = {tc : np.array( [truncated_normal_distributed(val[tc]['mean'][i],
                                                                          val[tc]['sd'][i], self.n,
                                                                          truncmin = val[tc]['truncmin'][i],
                                                                          truncmax = val[tc]['truncmax'][i]) / dt 
                                             for i in range( len(tau_specs[key][tc]['mean'] )) 
                                             ] )
                            for tc in tc_list
                            }
                                   
            if plot_syn_tau_hist:
                
                plot_histogram(self.tau[key][tc_plot], bins = bins, 
                               title = self.name,
                               xlabel = r'$\tau_{' + tc_plot +'} \; (ms)$')     
        
        
    def initialize_transmission_delays(self, dt, T_specs, lower_bound_perc = 0.8, upper_bound_perc=1.2,
                                       bins=50, color='grey', plot_T_hist = False, hetero_trans_delay = True):
        
        ''' initialize axonal transmission delays with a truncated normal distribution
            Note: dt incorporated in tau for time efficiency
        '''

        if len(self.receiving_from_pop_name_list) > 0:
            
            if hetero_trans_delay:
                
                self.initialize_transmission_delays_heterogeneously(dt, T_specs)
                
            else:
                
                self.initialize_transmission_delays_homogeneously(dt, T_specs)

           
            
    def initialize_transmission_delays_heterogeneously(self, dt, T_specs):
        
        self.transmission_delay = {key: np.array(truncated_normal_distributed(v['mean'],
                                                                              v['sd'], self.n,
                                                                              truncmin = v['truncmin'],
                                                                              truncmax = v['truncmax']) / dt ).astype(int)
                                   for key, v in T_specs.items()}
        
    def initialize_transmission_delays_homogeneously(self, dt, T_specs):
        
        self.transmission_delay = {key: int( v['mean']/ dt )
                                   for key, v in self.T_specs.items()}
        
    def initialize_ext_synaptic_time_constant(self, poisson_prop, dt, lower_bound_perc=0.8, upper_bound_perc=1.2):
        
        tc_list = ['rise', 'decay']
        # dt incorporated for time efficiency

        self.tau_ext_pop = {tc: truncated_normal_distributed(poisson_prop[self.name]['tau'][tc]['mean'],
                                                             poisson_prop[self.name]['tau'][tc]['var'], self.n,
                                                             lower_bound_perc=lower_bound_perc, 
                                                             upper_bound_perc=upper_bound_perc) / dt
                            for tc in tc_list}
                  
    def initialize_spike_threshold(self, spike_thresh_bound_ratio, lower_bound_perc=0.8, 
                                   upper_bound_perc=1.2, plot_spike_thresh_hist = False,
                                   bins=20):
        
        self.spike_thresh = truncated_normal_distributed(self.neuronal_consts['spike_thresh']['mean'],
                                                         self.neuronal_consts['spike_thresh']['var'], 
                                                         self.n,
                                                         scale_bound = scale_bound_with_arbitrary_value, 
                                                         scale = ( self.neuronal_consts['spike_thresh']['mean'] - 
                                                                  self.neuronal_consts['u_rest']['mean']),
                                                         lower_bound_perc = spike_thresh_bound_ratio[0], 
                                                         upper_bound_perc = spike_thresh_bound_ratio[1])
        if plot_spike_thresh_hist:
            
            plot_histogram(self.spike_thresh, bins = bins, 
                           title = self.name,
                           xlabel = r'spike threshold (mV)')
            
    def initialize_resting_membrane_potential(self, plot_initial_V_m_dist = False):
    
        self.u_rest = truncated_normal_distributed(self.neuronal_consts['u_rest']['mean'],
                                                   self.neuronal_consts['u_rest']['var'], self.n,
                                                   truncmin=self.neuronal_consts['u_rest']['truncmin'],
                                                   truncmax=self.neuronal_consts['u_rest']['truncmax'])

    def initialize_mem_potential(self, method = 'uniform', plot_initial_V_m_dist = False, keep_ex_voltage_trace = False):
        
        if method not in ['uniform', 'constant', 'exponential', 'draw_from_data']:
            
            raise ValueError(
                " method must be either 'uniform', 'constant', 'exponential', or 'draw_from_data' ")
            
        if method == 'draw_from_data':

            data = np.load(os.path.join(self.path, 'all_mem_pot_' + self.name + '_tau_' + 
                                        str(np.round(
                                            self.neuronal_consts['membrane_time_constant']['mean'], 1)
                                            ).replace('.', '-') + '_' 
                                        + self.state + '.npy'))
            
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
            self.mem_potential = self.neuronal_consts['spike_thresh']['mean'] - X.rvs(self.n)
        
        if self.keep_mem_pot_all_t:
            self.all_mem_pot[:, 0] = self.mem_potential.copy()
            
        if plot_initial_V_m_dist:
            self.plot_mem_potential_distribution_of_one_t(0, bins=50)
            
        if keep_ex_voltage_trace:
            self.voltage_trace[0] = self.mem_potential[0] 
        
    def initialize_heterogeneously( self,  poisson_prop, dt, t_sim, spike_thresh_bound_ratio, 
                                    lower_bound_perc = 0.8, upper_bound_perc = 1.2,
                                    plot_initial_V_m_dist=False, plot_mem_tau_hist = False,
                                    bins=50, color='grey', tc_plot = 'decay', syn_element_no = 0, 
                                    plot_syn_tau_hist = False, plot_spike_thresh_hist = False,
                                    plot_RMP_to_APth = False, keep_ex_voltage_trace = False,
                                    hetero_trans_delay = True, hetero_tau = True):
        
        ''' cell properties and boundary conditions come from distributions'''
        
        self. initialize_resting_membrane_potential()
        
        self. initialize_transmission_delays(dt, self.T_specs,
                                            lower_bound_perc = lower_bound_perc, 
                                            upper_bound_perc = upper_bound_perc,
                                            hetero_trans_delay = hetero_trans_delay)
        
        self. initialize_spike_threshold( spike_thresh_bound_ratio,
                                         lower_bound_perc = lower_bound_perc, 
                                         upper_bound_perc = upper_bound_perc,
                                         plot_spike_thresh_hist= plot_spike_thresh_hist)
        
        self. initialize_mem_potential(method=self.mem_pot_init_method, 
                                       plot_initial_V_m_dist = plot_initial_V_m_dist,
                                       keep_ex_voltage_trace = keep_ex_voltage_trace )
        
        self. initialize_membrane_time_constant(lower_bound_perc = lower_bound_perc, 
                                               upper_bound_perc = upper_bound_perc,
                                               plot_mem_tau_hist = plot_mem_tau_hist)
        
        self. initialize_synaptic_time_constant(dt, self.tau_specs, bins = bins, color= color, tc_plot = tc_plot, 
                                               syn_element_no = syn_element_no, 
                                               plot_syn_tau_hist = plot_syn_tau_hist,
                                               lower_bound_perc = lower_bound_perc, 
                                               upper_bound_perc = upper_bound_perc,
                                               hetero_tau = hetero_tau )
        
        self.initialize_ext_synaptic_time_constant(poisson_prop, dt, 
                                                  lower_bound_perc = lower_bound_perc, 
                                                  upper_bound_perc = upper_bound_perc)

        if plot_RMP_to_APth:
            
            plot_histogram(self.spike_thresh - self.u_rest, bins = bins, 
                           title = self.name,
                           xlabel = r'spike threshold - RMP (mV)')
            
    def initialize_homogeneously(self, T_specs, tau_specs, poisson_prop, dt, keep_mem_pot_all_t=False):
        
        ''' cell properties and boundary conditions are constant for all cells'''

        self.initialize_transmission_delays_homogeneously(dt, T_specs)

        self.initialize_synaptic_time_constant_homogeneously(dt, tau_specs)

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
        
            
            
        if self.keep_mem_pot_all_t:
            self.all_mem_pot[:, 0] = self.mem_potential.copy()
          
    def reset_synaptic_weights(self, G, N):

        # filter based on the receiving nucleus
        self.synaptic_weight_specs = {k: v for k, v in G.items() if k[0] == self.name}
        self.create_syn_weight_mean_dict()
        
        if self.neuronal_model == 'spiking':
            
            self.normalize_synaptic_weight()
        
        self.revert_connectivity_mat_to_binary()
        self.multiply_connectivity_mat_by_G(N)
        
    def normalize_synaptic_weight(self):
        self.synaptic_weight = {key:  val * (self.neuronal_consts['spike_thresh']['mean'] - 
                                             self.neuronal_consts['u_rest']['mean']) 
                                for key, val in self.synaptic_weight.items() }
        
        if self.G_heterogeneity:
            
            self.synaptic_weight_specs = {key: 
                                              {k: v * (self.neuronal_consts['spike_thresh']['mean'] - 
                                                        self.neuronal_consts['u_rest']['mean']) 
                                                for k, v in self.synaptic_weight_specs[key].items() 
                                                }
                                              for key, val in self.synaptic_weight_specs.items() }
        else:
            self.synaptic_weight_specs = self.synaptic_weight


    def revert_connectivity_mat_to_binary(self):

        for k, v in self.connectivity_matrix.items():
            self.connectivity_matrix[k][np.nonzero(self.connectivity_matrix[k])] = 1
        
    def normalize_synaptic_weight_by_N(self):
        
        self.synaptic_weight = { key: val / self.K_connections[key] 
                                for key, val in self.synaptic_weight.items()}
        
        self.connectivity_matrix = { key: val / self.K_connections[self.name, key[0]] 
                                    for key, val in self.connectivity_matrix.items()}
        
        if self.G_heterogeneity:
            self.synaptic_weight_specs = { key: 
                                          { k: v / self.K_connections[key] for k,v in 
                                           self.synaptic_weight_specs[key].items()}
                                          for key, val in self.synaptic_weight_specs.items()}
        else:
            self.synaptic_weight_specs = self.synaptic_weight
            

        
    def set_connections(self, K, N, plot_syn_weight_hist = False):
        
        ''' 
            create J_{ij} connection matrix with values corresponding to synaptic weight

        '''
        self.K_connections = {k: v for k, v in K.items() if k[0] == self.name}
        
        for projecting in self.receiving_from_list:
            
            same_pop = False
            proj_name = projecting[0]
            
            n_connections = self.K_connections[(self.name, proj_name)]
            
            if self.name == proj_name:
                same_pop = True
                
            self.connectivity_matrix[projecting] = build_connection_matrix(self.n, N[proj_name], 
                                                                           n_connections, 
                                                                           same_pop = same_pop)
            
        self.multiply_connectivity_mat_by_G(N, plot_syn_weight_hist = plot_syn_weight_hist)
        
    def change_connectivity_mat_to_sparse(self):
        
        #### csc_array is faster that csr_array.
        self.connectivity_matrix = { k : 
                                    sp.csc_array(v)
                                    for k, v in self.connectivity_matrix.items()
                                    }
        
    def initialize_synaptic_weight(self, proj_name, N, plot_syn_weight_hist = False):
        
        key = (self.name, proj_name)
        
        if self.init_method == 'heterogeneous' and self.G_heterogeneity:
            
            if self.syn_w_dist == 'normal':
                
                synaptic_weights = truncated_normal_distributed(self.synaptic_weight_specs[key]['mean'],
                                                                  self.synaptic_weight_specs[key]['sd'], 
                                                                  (self.n, N[proj_name]), 
                                                                  truncmin = self.synaptic_weight_specs[key]['truncmin'],
                                                                  truncmax = self.synaptic_weight_specs[key]['truncmax'])
            elif self.syn_w_dist == 'log-normal':
                    
                
                if self.synaptic_weight_specs[key]['order_mag_sigma'] != None:
                    
                    sd = (10 ** self.synaptic_weight_specs[key]['order_mag_sigma'] - 1) / \
                         (10 ** self.synaptic_weight_specs[key]['order_mag_sigma'] + 1) \
                         * abs(self.synaptic_weight_specs[key]['mean'])
                    
                else:
                    
                    sd = self.synaptic_weight_specs[key]['sd']
                
                sign = np.sign(self.synaptic_weight_specs[key]['mean'])
                synaptic_weights = sign * truncated_lognormal_distributed(abs(self.synaptic_weight_specs[key]['mean']),
                                                                          sd, 
                                                                          (self.n, N[proj_name]), 
                                                                          truncmin = self.synaptic_weight_specs[key]['truncmin'],
                                                                          truncmax = self.synaptic_weight_specs[key]['truncmax'])
            else:
                
                raise ValueError ('synaptic weight distribution must be normal or log-normal')
                
        else:
            
            if self.G_heterogeneity:
                
                synaptic_weights = self.synaptic_weight_specs[key]['mean']
                
            else:
                
                synaptic_weights = self.synaptic_weight_specs[key]
        
        if plot_syn_weight_hist:
            plot_histogram(synaptic_weights.flatten(), bins = 25, xaxis = 'log', absolute = True,
                           title = self.name, xlabel = r'$G_{' + ' ' + self.name + '-' + proj_name + '}$')
        return synaptic_weights
    
    def multiply_connectivity_mat_by_G(self, N, plot_syn_weight_hist = False):
        
        for projecting in self.receiving_from_list:
            
            proj_name = projecting[0]
    
            synaptic_weights = self.initialize_synaptic_weight(proj_name, N, plot_syn_weight_hist = plot_syn_weight_hist)
                
            self.connectivity_matrix[proj_name,projecting[1]] = synaptic_weights * \
                                                                self.connectivity_matrix[proj_name,projecting[1]]
             
            self.change_connectivity_mat_to_sparse()       
                                             
            # if plot_syn_weight_dist:
                
            #      ind = np.nonzero(self.connectivity_matrix[proj_name,projecting[1]])
                 
            #      plot_histogram(self.connectivity_matrix[proj_name,projecting[1]][ind].flatten(), 
            #                     bins = 25, title =  proj_name + ' to '+ self.name ,
            #                     xlabel = 'normalizd G')                                              

    def calculate_input_and_inst_act(self, t, dt, receiving_from_class_list, mvt_ext_inp):
        
        ''' RATE MODEL: I = Sum (G * m * J) and then
            A = Transfer_f (I)'''

        syn_inputs = np.zeros((self.n, 1))  
        
        for projecting in receiving_from_class_list:
            

            
            syn_inputs += ( 
                           
                                   self.connectivity_matrix[ (projecting.name, projecting.population_num) ]  @ 
                                   projecting.output[ (self.name, 
                                                       self.population_num)][:, - self.transmission_delay[
                                                                                                       (self.name, 
                                                                                                        projecting.name)]].reshape(-1, 1)
                                   
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
                
                new_output[key] += dt \
                                    * (-self.output[key][:, -1].reshape(-1, 1) + self.neuron_act) \
                                    / tau
            self.output[key] = np.hstack((self.output[key][:, 1:], new_output[key]))

    def cal_ext_inp(self, dt, t):

        # choose method of exerting external input from dictionary of methods
        I_ext = self.ext_inp_method_dict[self.ext_inp_method](dt)

        ( self.I_syn['ext_pop', '1'], 
         self.I_rise['ext_pop', '1']) = self.input_integ_method_dict[self. ext_input_integ_method](I_ext, 
                                                                                                   dt,
                                                                                                   I_rise = self.I_rise['ext_pop', '1'],
                                                                                                   I = self.I_syn['ext_pop', '1'],
                                                                                                   tau_rise = self.tau_ext_pop['rise'],
                                                                                                   tau_decay = self.tau_ext_pop['decay'])
    def cal_ext_inp_with_additional_inp(self, dt, t):
        
        # choose method of exerting external input from dictionary of methods
        
        I_ext = self.ext_inp_method_dict[self.ext_inp_method](dt) + self.external_inp_t_series[t]
        
        ( self.I_syn['ext_pop', '1'], 
         self.I_rise['ext_pop', '1']) = self.input_integ_method_dict[self. ext_input_integ_method](I_ext, 
                                                                                                   dt,
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

            
                   
    def cal_synaptic_input_sparse_history(self, dt, projecting, t):
        
        ''' Calcculate synaptic input of the given projection 
            with heterogeneous transmission delays with sparse connectivity and spike matrices'''

        # print(type(self.connectivity_matrix[(projecting.name, projecting.population_num)]),
        #      type(projecting.spikes[
        #          (np.arange(self.n), 
        #           t  - self.transmission_delay[
        #                      (self.name, 
        #                       projecting.name)])].tocsc().T)
        #      )
        self.syn_inputs[projecting.name, 
                        projecting.population_num] =( (self.connectivity_matrix[(projecting.name, 
                                                                                  projecting.population_num)] @ 
                                                        projecting.spikes[
                                                            (np.arange(self.n), 
                                                            t  - self.transmission_delay[
                                                                        (self.name, 
                                                                        projecting.name)])].tocsr().T).toarray().reshape(-1,) * \
                                      self.membrane_time_constant  / dt).reshape(-1,)

        # self.syn_inputs[projecting.name, 
        #                 projecting.population_num] =( (self.connectivity_matrix[(projecting.name, 
        #                                                                          projecting.population_num)] @ 
        #                                                projecting.spikes.toarray()[
        #                                                    (np.arange(self.n), 
        #                                                     t  - self.transmission_delay[
        #                                                                (self.name, 
        #                                                                 projecting.name)])]).reshape(-1,) * \
        #                               self.membrane_time_constant  / dt).reshape(-1,)                                                                                 
    def cal_synaptic_input_long_history(self, dt, projecting, t):
        
        ''' Calcculate synaptic input of the given projection 
            with heterogeneous transmission delays with sparse connectivity and 
            a long spike history matrices'''

        
        self.syn_inputs[projecting.name, 
                        projecting.population_num] = (

                                          (self.connectivity_matrix[(projecting.name, projecting.population_num)] @
                                          projecting.spikes[(np.arange(self.n), 
                                                        t - self.transmission_delay[(self.name, projecting.name)])]

                                                ) * \
                                      self.membrane_time_constant  / dt).reshape(-1,)
                                                                             
    def cal_synaptic_input_short_history(self, dt, projecting, t):
        
        ''' Calcculate synaptic input of the given projection 
            with heterogeneous transmission delays with sparse connectivity and 
            short spike history matrices'''

        
        self.syn_inputs[projecting.name, 
                        projecting.population_num] = (

                                          (self.connectivity_matrix[(projecting.name, projecting.population_num)] @
                                          projecting.spikes[(np.arange(self.n), 
                                                        - self.transmission_delay[(self.name, projecting.name)])] ## limited spike history 
                                                        # t - self.transmission_delay[(self.name, projecting.name)])]

                                                ) * \
                                      self.membrane_time_constant  / dt).reshape(-1,)
                                                                             

    def find_and_save_new_spikes(self, t):

        spiking_ind = np.where(self.mem_potential > self.spike_thresh)[0]
        
        self.save_new_spikes_method_dict [self.spike_history] (spiking_ind, t)
        
        return spiking_ind

    def save_new_spikes(self, spiking_ind, t):
        
        self.spikes[spiking_ind, t] = 1 ## full spike history 

    def save_new_spikes_short_history(self, spiking_ind, t):
        
        self.shift_spikes_back(spiking_ind) ## limited spike history 
        self.ind_last_spike[spiking_ind] += 1
        self.spike_times[(spiking_ind, 
                          (self.ind_last_spike[spiking_ind] + 1).astype(int))] = t
    
    def save_new_spikes_long_history(self, spiking_ind, t):
        
        self.spikes[spiking_ind, t] = 1 ## full spike history 

    def shift_spikes_back(self, spiking_ind):
        
        result = np.zeros_like(self.spikes)
    
        result[spiking_ind,-1] = 1
        result[:, :-1] = self.spikes[:,1:]
        self.spikes = result

    def cal_population_activity_all_t(self, dt):
        
        if self.spike_history == 'short-term':
            spikes = np.zeros((self.n, self.n_timebins)) ## limited spike history 
            
            for n in range(self.n):
                spikes[n, self.spike_times[n, np.nonzero(self.spike_times[n,:])]] = 1
                
            self.pop_act = np.average(spikes, axis = 0)/ (dt/1000)
            
        
        elif self.spike_history == 'long-term':
            self.pop_act = np.average(self.spikes, axis=0)/ (dt/1000)
         
        elif self.spike_history == 'sparse':
            self.pop_act = np.average(self.spikes.toarray(), axis = 0) / (dt/1000)
                                                 
    def sum_synaptic_input(self, receiving_from_class_list, dt, t):
        
        ''' Sum the synaptic input from all projections'''
        
        synaptic_inputs = np.zeros(self.n)
        
        for projecting in receiving_from_class_list:

            self.cal_synaptic_input_method_dict [self.spike_history] (dt, projecting, t)
            synaptic_inputs = synaptic_inputs + \
                              self.sum_components_of_one_synapse(t, dt, 
                                                                 projecting.name, 
                                                                 projecting.population_num,
                                                                 pre_n_components = 
                                                                 self.pre_n_components[projecting.name])
                              

        return synaptic_inputs

    def sum_synaptic_input_one_step_ahead_with_no_spikes(self, receiving_from_class_list, dt):
        
        '''Calculate I_syn(t+dt) one component (e.g. GABA-A or GABA-b) 
        assuming that there are no spikes between time t and t+dt '''
        
        synaptic_inputs = np.zeros(self.n)
        
        for projecting in receiving_from_class_list:

            synaptic_inputs =  synaptic_inputs + \
                                self.sum_components_of_one_synapse_one_step_ahead_with_no_spikes(dt, projecting.name,
                                                                                                projecting.population_num,
                                                                                                pre_n_components = 
                                                                                                self.pre_n_components[projecting.name]
                                                                                                )
                                
        return synaptic_inputs


    def sum_components_of_one_synapse(self, t, dt, pre_name, pre_num, pre_n_components=1):
        
        '''Calculate I_syn(t) as the sum of  all synaptic components  (e.g. GABA-A or GABA-b)  '''

        sum_components = np.zeros(self.n)
        
        for i in range(self.pre_n_components[pre_name]):
            
            (self.I_syn[pre_name, pre_num][:, i], 
             self.I_rise[pre_name, pre_num][:, i] ) = self.input_integ_method_dict[self.syn_input_integ_method](
                                                                 self.syn_inputs[pre_name, pre_num], dt, 
                                                                 I_rise = self.I_rise[pre_name, pre_num][:, i],
                                                                 I = self.I_syn[pre_name, pre_num][:, i],
                                                                 tau_rise = self.tau[(self.name, pre_name)]['rise'][i],
                                                                 tau_decay = self.tau[(self.name, pre_name)]['decay'][i])
            # self.representative_inp[pre_name, pre_num][t,
            #     i] = self.I_syn[pre_name, pre_num][0, i]
            

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
                                                tau_rise = self.tau[(self.name, pre_name)]['rise'][i],
                                                tau_decay = self.tau[(self.name, pre_name)]['decay'][i])

            sum_components = sum_components + I_syn_next_dt * self.syn_component_weight[self.name, pre_name][i]
            
        return sum_components
    
    def solve_IF_without_syn_input(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp_method_dict [self.external_input_bool](dt, t)
        synaptic_inputs = np.zeros(self.n)
        self.update_potential(synaptic_inputs, dt, t, receiving_from_class_list)
        spiking_ind = self.find_and_save_new_spikes(t)
        # self.reset_potential(spiking_ind)
        self.reset_potential_with_interpolation(spiking_ind, dt)

    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp_method_dict [self.external_input_bool](dt, t)
        synaptic_inputs = self.sum_synaptic_input(receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, t, receiving_from_class_list)
        spiking_ind = self.find_and_save_new_spikes(t)
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
                  np.power( np.average( V_mean_t ), 2)

        V_mean_n = self.average_mem_pot_over_t(start = half_t, sampling_t_distance = sampling_t_distance)
        V_2_mean_n = self.average_mem_pot_2_over_t(start = half_t, sampling_t_distance = sampling_t_distance)
        normalizing_factor = np.average( V_2_mean_n - \
                                       np.power( V_mean_n , 2)
                                       )
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
        
        

        # self.mem_potential = fwd_Euler(dt, self.mem_potential, V_prime)
        I_syn_next_dt = self. sum_synaptic_input_one_step_ahead_with_no_spikes( receiving_from_class_list, 
                                                                                dt)
        ind_not_in_refractory_p = self.get_neurons_not_in_refractory_period(t)
        # print(len(ind_not_in_refractory_p))
        self.mem_potential[ind_not_in_refractory_p] = Runge_Kutta_second_order_LIF( dt, 
                                                          self.mem_potential[ind_not_in_refractory_p], 
                                                          V_prime[ind_not_in_refractory_p],  
                                                          self.membrane_time_constant[ind_not_in_refractory_p], 
                                                          I_syn_next_dt[ind_not_in_refractory_p], 
                                                          self.u_rest[ind_not_in_refractory_p], 
                                                          self.I_syn['ext_pop', '1'][ind_not_in_refractory_p],
                                                          self.half_dt)
        
        # self.mem_potential = Runge_Kutta_second_order_LIF( dt, 
        #                                                   self.mem_potential, 
        #                                                   V_prime,  
        #                                                   self.membrane_time_constant, 
        #                                                   I_syn_next_dt, 
        #                                                   self.u_rest, 
        #                                                   self.I_syn['ext_pop', '1'],
        #                                                   self.half_dt)
        # self.voltage_trace[t] = self.mem_potential[0]
        
    def get_neurons_not_in_refractory_period(self, t):
        
        """ return the indices of neurons that have not had any spikes
        for a duration of a refractory period"""
        return np.where(np.sum(self.spikes[:, 
                                    t - self.refractory_period : t], axis = 1)
                 == 0)[0]
        
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

    def clear_history(self, mem_pot_init_method=None):
        
        if len( self.pop_act ) > 0:
            self.pop_act[:] = 0
        
        if self.neuronal_model == 'rate':

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
                # self.representative_inp[k][:] = 0
                self.syn_inputs[k][:] = 0
                
            self.I_syn['ext_pop', '1'][:] = 0
            self.I_rise['ext_pop', '1'][:] = 0
            self.mem_pot_before_spike[:] = 0

            # self.voltage_trace[:] = 0
            # self.representative_inp['ext_pop', '1'][:] = 0
            
            if self.external_input_bool:
                self.external_inp_t_series[:] = 0
                
            if mem_pot_init_method == None:  # if not specified initialize as before
                mem_pot_init_method = self.mem_pot_init_method

            self.initialize_mem_potential(method=mem_pot_init_method)
            self.neuron_spike_phase_hist = {}
            self.noise[:, :] = 0
            
            if self.spike_history == 'short-term':
                
                self.ind_last_spike[:] = 0
                self.spike_times[:,:] = 0
                self.spikes[:, :] = 0

            elif self.spike_history == 'long-term':
                self.spikes[:, :] = 0
                
            elif self.spike_history == 'sparse':
                self.spikes = sp.lil_matrix(self.spikes.shape)
                
    def smooth_pop_activity(self, dt, window_ms=5):
        
        self.pop_act = moving_average_array(self.pop_act, int(window_ms / dt))


    def average_pop_activity(self,ind_start, ind_end):
        
        average = np.average(self.pop_act[ind_start : ind_end])
        std = np.std(self.pop_act[ind_start : ind_end])
        return average, std
    
    def set_synaptic_time_scales(self, synaptic_time_constant):

        self.synaptic_time_constant = {
            k: v for k, v in synaptic_time_constant.items() if k[1] == self.name}

    # def incoming_rest_I_syn(self, proj_list, A, dt):
        

    #     I_syn = np.sum(np.array([
    #                     np.sum(self.connectivity_matrix[proj, '1'], axis = 1) * 
    #                     A[proj] / 1000  * 
    #                     np.sum(self.syn_component_weight[self.name, proj]) \
    #                         for proj in proj_list]), axis = 0) * \
    #             self.membrane_time_constant

    #     return I_syn

    def incoming_rest_I_syn(self, proj_list, A, dt):
        
        I_syn = np.sum(np.array([
                        np.sum( self.connectivity_matrix[(proj, '1')], axis = 1).reshape(-1,) * 
                        A[proj] / 1000  * 
                        np.sum(self.syn_component_weight[self.name, proj]) \
                            for proj in proj_list]), axis = 0).reshape(-1,) * \
                self.membrane_time_constant

        return I_syn


    def set_ext_input(self, A, A_mvt, D_mvt, t_mvt, t_list, dt, end_of_nonlinearity = 25, change_states = True):

        proj_list = [ k[0] for k in list(self.receiving_from_list) ]

        if self.neuronal_model == 'rate':

            self.rest_ext_input = self.basal_firing / self.gain - \
                                  np.sum([self.synaptic_weight[self.name, proj] * A[proj] *
                                          self.K_connections[self.name, proj] for proj in proj_list]) + \
                                  self.threshold
            self.external_inp_t_series = np.zeros_like(t_list)

            if change_states:                       
                
                self.mvt_ext_input = self.mvt_firing / self.gain - \
                                        np.sum([self.synaptic_weight[self.name, proj]*A_mvt[proj] * \
                                                self.K_connections[self.name, proj]
                                                for proj in proj_list]) + self.threshold - self.rest_ext_input
                                            
                self.external_inp_t_series = mvt_step_ext_input( D_mvt, t_mvt, self.ext_inp_delay, 
                                                                 self.mvt_ext_input, t_list * dt)
        elif self.neuronal_model == 'spiking':  
            
            I_syn = self.incoming_rest_I_syn(proj_list, A, dt)

            self.sum_syn_inp_at_rest = I_syn

            if self.ext_inp_method == 'Poisson':
                
                self._set_ext_inp_poisson(I_syn)

            elif self.ext_inp_method == 'const+noise' or self.ext_inp_method == 'const':
                
                self._set_ext_inp_const_plus_noise(I_syn, end_of_nonlinearity, dt)
                
            else:
                raise ValueError('external input handling method is not right!')
                
            self._save_init()


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
            
            self.distribute_FR_ext_as_normal(self.FR_ext_specs['mean'])
            self.rest_ext_input = (self.FR_ext * self.syn_weight_ext_pop * \
                                   self.n_ext_population * self.membrane_time_constant - 
                                   I_syn )#.reshape(-1,1)
            self.I_ext_0 =  np.average(self.rest_ext_input + I_syn)
            
    def distribute_FR_ext_as_normal(self, FR_ext_mean ):
        
        if type (self.FR_ext_specs) is dict:

            self.FR_ext = truncated_normal_distributed(FR_ext_mean,
                                                       self.FR_ext_specs['sd'], 
                                                       self.n,  
                                                       truncmin = self.FR_ext_specs['truncmin'],
                                                       truncmax = self.FR_ext_specs['truncmax'])
            
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
        FR_sim = self.find_threshold_of_firing(
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
        self. set_FR_ext_each_neuron( FR_list, FR_sim, dt, 
                                     extrapolate = extrapolate_FR_ext_from_neuronal_response_curve_high_act, 
                                     if_plot = if_plot, ax=ax, c=c)
        
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
            
            self.distribute_FR_ext_as_normal(FR_list[i, 0])
            self.rest_ext_input = self.FR_ext  * self.membrane_time_constant * \
                                    self.n_ext_population * self.syn_weight_ext_pop
            self. run(dt, t_list, receiving_class_dict)
            
            FR_sim[:, i] = np.average( self.spikes[:, int(len(t_list)/2):], 
                                          axis=1) /  (dt/1000)
            
            print('FR = ', np.average(FR_list[i, :]), np.average(FR_sim[:, i]))

        return FR_sim
    
    def add_beta_ext_input(self, amplitude, dt, freq = 20, start = 0, end = None, 
                           mean = 0, plot = False, method = 'excitaion'):
        
        if end == None :
            end = self.n_timebins
            
        
        periodic_inp, t_list = generate_periodic_input(start, end, dt, amplitude, freq, mean, method)
        
        self.external_inp_t_series[start : end] =  periodic_inp
                                                               
        if plot:
            fig, ax = plt.subplots()
            ax.plot(t_list, periodic_inp.T.reshape(-1,1))
            

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
        
    def plot_syn_tau_dist(self, key, dt, ax=None, bins=50, 
                          color='grey', tc = 'decay', syn_element_no = 0):

        fig, ax = get_axes(ax)
        
        ax.hist(self.tau[key][tc][syn_element_no,:] * dt, bins=bins,
                label='from '+ key[1] , density=True, stacked=True, color=color)
        
        ax.set_xlabel('synaptic ' + tc + ' time constant (ms)', fontsize=15)
        ax.set_ylabel(r'$Probability\ density$', fontsize=15)
        ax.set_title(self.name, fontsize=15)
        ax.legend(fontsize=15,  framealpha = 0.1, frameon = False)
        
    def find_threshold_of_firing(self, FR_list, t_list, dt, receiving_class_dict):
        
        FR_sim = np.zeros((self.n, len(FR_list)))

        for i in range(FR_list.shape[0]):

            self.clear_history()
            self.rest_ext_input = FR_list[i, :]  * self.membrane_time_constant * \
                                   self.n_ext_population * self.syn_weight_ext_pop
                                   
            self. run(dt, t_list, receiving_class_dict)
            FR_sim[:, i] = np.average( self.spikes[:, int(len(t_list)/2):], 
                                       axis = 1) / ( dt /  1000 )
            
            print('FR = ', np.average(FR_list[i, :]), np.average(FR_sim[:, i]))
            
            if np.all(FR_sim[:, i] > 1):
                
                print('done!')
                
                break
            
        return FR_sim

    def run(self, dt, t_list, receiving_class_dict):
        
        for t in t_list:  # run temporal dynamics
        
            self.solve_IF_without_syn_input(
                t, dt, receiving_class_dict[(self.name, self.population_num)])
            

      
    def find_peaks_of_pop_act(self, dt, low_f, high_f, filter_order = 6, 
                              peak_threshold = None, start = 0, end = None, 
                              threshold_by_percentile = 75, ax = None):
        """
        

        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.
        low_f : TYPE
            DESCRIPTION.
        high_f : TYPE
            DESCRIPTION.
        filter_order : TYPE, optional
            DESCRIPTION. The default is 6.
        peak_threshold : float, optional
            if value entered the peaks of the detrended signal will be determined using 
            the provided threshold. The default is None.
        start : TYPE, optional
            DESCRIPTION. The default is 0.
        end : TYPE, optional
            DESCRIPTION. The default is None.
        threshold_by_percentile : float :math:`\in [0,1]`, optional
            if `peak_threshold=None` the `threshold_by_percentile` percentile of the analog signal will be used as 
            peak threshold. The default is 0.75.

        Returns
        -------
        peaks : TYPE
            DESCRIPTION.
        act : TYPE
            DESCRIPTION.

        """
        
        
        end = end or self.n_timebins
        
        act = self.butter_bandpass_filter_pop_act_not_modify( dt, low_f, high_f, order= filter_order)
        sig = act[start : end]
        amplitude_envelope = get_sig_amp_envelope(sig)
        peak_threshold = peak_threshold or np.percentile(amplitude_envelope, threshold_by_percentile)
        
        peaks_zero_thresh,_ = signal.find_peaks(sig, height = 0)
        peaks,_ = signal.find_peaks(sig, height = peak_threshold)
        
        print('ref peak threshold = ', round(peak_threshold, 3))

        print("n zero thresh peaks =", len(peaks_zero_thresh), 
              "n " + str(threshold_by_percentile), " percentile =",
              len(peaks))

        if ax != None:
            
            tlist = np.arange(start, end)
            ax.plot(act)
            ax.plot(tlist, sig)
            ax.plot(tlist, amplitude_envelope)
            ax.axhline(peak_threshold, xmin = 0, xmax = end, ls = '--')
            ax.scatter(tlist[peaks], sig[peaks], marker = 'x')
            ax.set_title(self.name)
            
        return peaks + start, act


########  population wise phase hist calculations

    # def find_spike_times_all_neurons(self, start = 0, neurons_ind = 'all'):
        
    #     if isinstance(neurons_ind, str)  and neurons_ind == 'all':  
            
    #         spike_times = np.where(self.spikes == 1)[1]
            
    #     elif hasattr(neurons_ind, '__len__') : # an actual list of neurons is given
        
    #         spike_times = np.where(self.spikes[neurons_ind,:] == 1) [1]
            
    #     else:
            
    #         raise( "Invalid input for 'ind_neurons'. \
    #               It has to be either 'all' or the list of indices of neurons to be included.")
            
    #     return spike_times[spike_times > start ] - start
    
    # def find_phase_hist_of_spikes(self, dt, low_f, high_f, filter_order = 6, height = 1,start = 0, 
    #                               ref_peaks = [], n_bins = 20, total_phase = 360, phase_ref = 'self', 
    #                               neurons_ind = 'all', troughs = False, end = None):
        
    #     ''' Note that it counts the spikes of all the neurons in the population '''
        
    #     if end == None:
    #         end = self.n_timebins
            
    #     all_spikes = self.find_spike_times_all_neurons(start = start, neurons_ind = neurons_ind)
        
    #     if len( ref_peaks ) == 0: # if peaks are given as argument, phases are calculated relative to them
        
    #         ref_peaks,_ = self.find_peaks_of_pop_act(dt, low_f, high_f, filter_order = filter_order, 
    #                                                height = height, start = start, end = end)
        
    #     phase_all_spikes = np.zeros( len(all_spikes) * 2 + 1)
        
        
    #     left_peak_series, right_peak_series = set_peak_iterators(total_phase, ref_peaks, troughs = troughs)
    #     n_cycles = len (left_peak_series)

    #     count = 0
    #     for left_peak, right_peak in zip( left_peak_series, right_peak_series ): # leave out the last peaks beacause sosfiltfilt affects the data boundaries
            
            
    #         phase, n_spikes_of_this_cycle = find_phase_of_spikes_bet_2_peaks(left_peak, right_peak, all_spikes, 
    #                                                                          total_phase = total_phase)
    #         phase_all_spikes[count : n_spikes_of_this_cycle + count] = phase
    #         count += n_spikes_of_this_cycle
          

    #     frq , edges = get_hist(phase_all_spikes[:count], end_bins = total_phase, n_bins = n_bins)
        
    #     self.neuron_spike_phase_hist[phase_ref] = frq / n_cycles / self.n, edges[:-1]
        
        
    
#### neuron wise phase histogram calculation
    def create_empty_phase_array(self, neurons_ind, n_bins, phase_ref):
        
        if isinstance(neurons_ind, str)  and neurons_ind == 'all':  
            
            self.neuron_spike_phase_hist[phase_ref] = np.zeros((self.n, n_bins -1))
            return np.arange(self.n)
        
        elif hasattr(neurons_ind, '__len__') : # an actual list of neurons is given
        
            self.neuron_spike_phase_hist[phase_ref] = np.zeros((len(neurons_ind), n_bins - 1))
            return neurons_ind
        
        else:
            
            raise( "Invalid input for 'ind_neurons'. \
                  It has to be either 'all' or the list of indices of neurons to be included.")
            return 0
        
    # def get_spike_times_of_neuron(self, n):
    #     if self.spike_history:
    #         return self.spike_times[n, :]
    #     else:
    #         return np.where(self.spikes[n,:] == 1)[0]
        
    def get_spike_times_of_neuron_short_history(self, n):
        
        return self.spike_times[n, :]
    
    def get_spike_times_of_neuron_long_history(self, n):
        
        return np.where(self.spikes[n,:] == 1)[0]

    def get_spike_times_of_neuron_sparse_history(self, n):
        
        return sp.find(self.spikes[n, :])[:2]
    

        
    def find_phase_hist_of_spikes(self, dt, low_f, high_f, filter_order = 6, start = 0, 
                                  ref_peaks = [], n_bins = 20, total_phase = 720, phase_ref = None, 
                                  neurons_ind = 'all', troughs = False, end = None, 
                                  keep_only_entrained = True, max_cycle_t = 83, ax = None, act = None):
        
        ''' find the average phases of each neuron over all cycles '''
        
            
        neurons_ind = self.create_empty_phase_array(neurons_ind, n_bins, phase_ref)
        self.phase_bins = np.linspace( 0, total_phase, endpoint = True, num = n_bins)
        
        rayleigh_test_p_values = np.zeros(self.n)
        
        left_peak_series, right_peak_series = set_peak_iterators(total_phase, ref_peaks, dt, 
                                                                 troughs = troughs, 
                                                                 max_cycle_t= max_cycle_t)
        n_cycles = len (left_peak_series)
        cycle_durations = right_peak_series - left_peak_series
        
        plot_filtered_ref_peaks(ax, left_peak_series, right_peak_series, act)
        
        for i, n in enumerate( neurons_ind ):
            
            spike_times = self.get_spike_times_of_neuron_method_dict [self.spike_history] (n)
            phases = np.empty(0)
            
            for (cycle_t, left_peak, right_peak) in zip(cycle_durations, left_peak_series, right_peak_series ): 


                this_cycle_spk_times, this_cycle_phases = find_phase_of_spikes_bet_2_peaks(
                                                                    cycle_t, left_peak, right_peak, 
                                                                    spike_times.copy(), total_phase = total_phase)

                phases = np.append(phases, this_cycle_phases)
                                                 
                                                      
            self.neuron_spike_phase_hist[phase_ref][n, :] = np.histogram( phases,  
                                                                         bins = self.phase_bins)[0]
            rayleigh_test_p_values[n] = self.Rayleigh_test_neuron_phase( phases)
            
        self.neuron_spike_phase_hist[phase_ref]  = self.neuron_spike_phase_hist[phase_ref] / n_cycles  
        
        if keep_only_entrained:
            
            self.filter_phase_on_rtest(rayleigh_test_p_values, phase_ref)
        
        if total_phase == 360:

            self.neuron_spike_phase_hist[phase_ref], self.phase_bins = double_phase_hist_cycle_2d(
                                                                                    self.neuron_spike_phase_hist[phase_ref], 
                                                                                    self.phase_bins)
    def filter_phase_on_rtest(self, rayleigh_test_p_values, phase_ref):
        

        
        self.rtest_passed_neuron_ind = np.where( rayleigh_test_p_values > self.rtest_p_val_thresh)[0]    
        self.neuron_spike_phase_hist[phase_ref] = self.neuron_spike_phase_hist[phase_ref][
                                                                    self.rtest_passed_neuron_ind]
        
        print( self.name, '{0} out of {1} passed Rayleigh test'.format(
                len(self.rtest_passed_neuron_ind),
                self.n))
        
    def Rayleigh_test_neuron_phase(self, phases):
        
        return rayleightest( phases )
    
    def shift_phase(self, theta, phase_ref, total_phase):
        
        n_bins = len( self.phase_bins )

        self.neuron_spike_phase_hist[phase_ref] = np.roll ( self.neuron_spike_phase_hist[phase_ref],  
                                                           int( theta / total_phase * n_bins) , axis = 1)
    
        
        
    def scale_synaptic_weight(self):
        
        if self.scale_g_with_N:
            
            self.synaptic_weight = {
                k: v / self.K_connections[k] for k, v in self.synaptic_weight.items()}
            
        
            self.connectivity_matrix = {
                k: v/self.K_connections[self.name, k[0]] for k, v in self.connectivity_matrix.items() }

    def change_pop_firing_rate(self, FR_ext, A, A_mvt = None, D_mvt = None , t_mvt = None, t_list = None,
                                  dt = None, end_of_nonlinearity=30):
        
        self.FR_ext = FR_ext
        self.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list,
                                      dt, end_of_nonlinearity=end_of_nonlinearity)
        
    def change_basal_firing(self, A_new):
        
        self.basal_firing = A_new
        
    def find_freq_of_pop_act_zero_crossing(self, sig, start, end, dt, peak_threshold = 0.1, smooth_kern_window= 3 , 
                                         check_stability = False, cut_plateau_epsilon = 0.001, plot_oscil = False):
        
        ''' RATE MODEL : 
            trim the beginning and end of the population activity of the nucleus 
            if necessary, cut the plateau and in case it is oscillation determine the frequency '''
        
        cut_sig_ind = cut_plateau( sig, epsilon= cut_plateau_epsilon)
        plateau_y = find_mean_of_signal(sig, cut_sig_ind)

        if_stable = False
        
        if len(cut_sig_ind) > 0: # if it's not all plateau from the beginning
        
            sig = sig - plateau_y

            n_half_cycles, freq = zero_crossing_freq_detect(sig[cut_sig_ind], dt / 1000)
            
            if freq != 0: # then check if there's oscillations
            
                perc_oscil = max_non_empty_array(cut_sig_ind)/ len(sig) * 100
                
                if check_stability:
                    
                    if_stable, last_first_peak_ratio = if_stable_oscillatory(sig,
                                                                             peak_threshold, dt = dt,
                                                                             x_plateau = max(cut_sig_ind),
                                                                             t_transition = start,
                                                                             smooth_kern_window = smooth_kern_window, 
                                                                             amp_env_slope_thresh = - 0.05,
                                                                             plot = plot_oscil)
                
                return n_half_cycles, last_first_peak_ratio , perc_oscil, freq, if_stable, None, None
            
            else:
                
                return 0, 0, 0, 0,  False, None, None
        else:
            
            return 0, 0, 0, 0, False, None, None

    def find_freq_of_pop_act_spec_window_fft(self, sig, start, end, dt,  smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False,
                                             plot_sig=False, plot_spectrum=False, ax=None, c_spec='navy', fft_label='fft',
                                             spec_figsize=(6, 5), find_beta_band_power=False, fft_method='fft', n_windows=6, include_beta_band_in_legend=False,
                                             divide_beta_band_in_power = False, normalize_spec = True, include_peak_f_in_legend = True, 
                                             low_beta_range = [12,20], high_beta_range = [20, 30], low_gamma_range = [30, 70],
                                             min_f = 0, max_f = 250, plot_sig_thresh = False,  n_std_thresh = 2, save_gamma = False):
        
        ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
            the plateau and in case it is oscillation determine the frequency '''
            

        # sig = signal.detrend(sig)
        # print(np.average(sig))
        # self.butter_bandpass_filter_pop_act_not_modify( dt, 0.5, 1000, order= 6)
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

        else: 

            beta_band_power = None
        
        n_half_cycles = None

        if freq != 0:  # then check if there's oscillations


            if divide_beta_band_in_power:

                if save_gamma:

                    return n_half_cycles, 0, freq, False, [low_beta_band_power, high_beta_band_power, low_gamma_band_power], f, pxx

                else:

                    return n_half_cycles, 0, freq, False, [low_beta_band_power, high_beta_band_power], f, pxx

            else:

                return n_half_cycles, 0, freq, False, beta_band_power, f, pxx

        else:
            
            print( " Freq = 0 " )

            return 0, 0, 0, False, None, f, pxx

    def find_freq_of_pop_act_spec_window(self, start, end, dt,  smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False,
                                      method='zero_crossing', plot_sig=False, plot_spectrum=False, ax=None, c_spec='navy', fft_label='fft',
                                      spec_figsize=(6, 5), find_beta_band_power=False, fft_method='welch', n_windows=6, include_beta_band_in_legend=False,
                                      divide_beta_band_in_power = False, normalize_spec = True, include_peak_f_in_legend = True, 
                                      low_beta_range = [12,20], high_beta_range = [20, 30], low_gamma_range = [30, 70], plot_oscil =False,
                                      min_f = 0, max_f = 300, plot_sig_thresh = False,  n_std_thresh = 2, save_gamma = False,
                                        peak_threshold = 0.1):
        ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
            the plateau and in case it is oscillation determine the frequency '''
        if method not in ["fft", "zero_crossing"]:
            
            raise ValueError(method, "is not an acceptable method. Method must be either 'fft', or 'zero_crossing'")

        sig = trim_start_end_sig_rm_offset( self.pop_act, start, end, 
                                            method = 
                                            self.trim_sig_method_dict[self.neuronal_model])

        if method == 'zero_crossing':
            return self.find_freq_of_pop_act_zero_crossing(sig, start, end, dt, peak_threshold = peak_threshold, smooth_kern_window= smooth_kern_window , 
                                                 check_stability = check_stability, cut_plateau_epsilon = cut_plateau_epsilon, plot_oscil = plot_oscil)
        elif method == 'fft':

            return self.find_freq_of_pop_act_spec_window_fft(sig, start, end, dt,  smooth_kern_window=smooth_kern_window, cut_plateau_epsilon=cut_plateau_epsilon, check_stability=check_stability,
                                             plot_sig=plot_sig, plot_spectrum=plot_spectrum, ax=ax, c_spec=c_spec, fft_label=fft_label,
                                             spec_figsize=spec_figsize, find_beta_band_power=find_beta_band_power, fft_method=fft_method, n_windows=n_windows, include_beta_band_in_legend=include_beta_band_in_legend,
                                             divide_beta_band_in_power = divide_beta_band_in_power, normalize_spec = normalize_spec, include_peak_f_in_legend = include_peak_f_in_legend, 
                                             low_beta_range = low_beta_range, high_beta_range = high_beta_range, low_gamma_range = low_gamma_range,
                                             min_f = min_f, max_f = max_f, plot_sig_thresh = plot_sig_thresh,  n_std_thresh = n_std_thresh, save_gamma = save_gamma)
        
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
 

def shift_phase_hist_all_nuclei(nuclei_dict, theta, phase_ref, total_phase = 720):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.shift_phase(theta, phase_ref, total_phase)
            
    return nuclei_dict

def shift_phase_hist(theta, hist, n_bins, total_phase = 720):
    return np.roll(hist,  
             int( theta / total_phase * n_bins) , axis = 1)
    
def plot_filtered_ref_peaks(ax, left_peak_series, right_peak_series, act):
    
    if ax != None:
        
        ax.scatter(left_peak_series, act[left_peak_series] + 1, marker = '*', c = 'purple')
        ax.scatter(right_peak_series, act[right_peak_series] + 3, marker = 's', c = 'navy')
# def find_phase_of_spikes_bet_2_peaks(left_peak_time, right_peak_time, all_spikes, total_phase = 360):
    
#     ''' For all neurons collectively'''
    
#     corr_spike_times = all_spikes[ np.logical_and(all_spikes >= left_peak_time, 
#                                                   all_spikes < right_peak_time) 
#                                                    ]
#     cycle_length = right_peak_time - left_peak_time
#     n_spikes = len(corr_spike_times)
#     phase = ( corr_spike_times - left_peak_time ) / cycle_length * total_phase

#     # fig,ax = plt.subplots()
#     # ax.hist(corr_spike_times, bins = 20)
#     # ax.hist(phase, bins = np.linspace(0, 360, endpoint=True, num = 50))

#     return phase, n_spikes

def find_phase_of_spikes_bet_2_peaks(cycle_t_duration, left_peak_time, right_peak_time, spike_times, total_phase = 360):
    
    corr_spike_times = spike_times[ np.logical_and( spike_times >= left_peak_time, 
                                                    spike_times < right_peak_time)]
    
    phases = ( corr_spike_times - left_peak_time ) / cycle_t_duration * total_phase
    # fig,ax = plt.subplots()
    # ax.hist(corr_spike_times, bins = 20)
    # ax.hist(phase, bins = np.linspace(0, 360, endpoint=True, num = 50))
    

    return corr_spike_times , phases
    
def get_hist(obs, end_bins = 720, n_bins = 360, bins = []):
    

    bins = np.linspace(0, end_bins, endpoint=True, num = n_bins)
        
    frq , edges = np.histogram(obs, bins = bins)
    
    # centers = get_centers_from_edges(edges)
    
    return frq, edges #, centers

# def shift_all_phases_all_nuclei(phase_ref):
    
#     for nuclei_list in nuclei_dict.values():
#         for nucleus in nuclei_list:
#             nucleus.shift_phase( theta, phase_ref, total_phase)
def shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def generate_periodic_input(start, end, dt, amplitude, freq, mean, method):
    
    t_list = np.arange(start, end) * dt/1000
    sine_function = sinfunc(t_list, amplitude, 2 * np.pi * freq, 0, mean).reshape(1,-1) 

    periodic_positive_inp = sine_function.clip( min = 0 )
    
    if method == 'excitation':
        periodic_inp = periodic_positive_inp
        
    elif method == 'inhibition':
        periodic_inp = - periodic_positive_inp
    
    else:
        raise ( " Beta induction method should be either excitation or inhibition!")
    return periodic_inp, t_list

# def set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2, order_mag_sigma = None):
    
#     for key, val in G.items():
        
#         sign = np.sign(val['mean'])
#         G[key]['sd'] = abs(val['mean'] ) * sd_to_mean_ratio
#         abs_truncmin = max(0, abs( val['mean']) - 
#                            n_sd_trunc * G[key]['sd'])
#         abs_truncmax = abs( val['mean'] ) + n_sd_trunc * G[key]['sd']
#         G[key]['truncmin'] = min( sign * abs_truncmin, sign * abs_truncmax, sign * 10**-10)
#         G[key]['truncmax'] = max( sign * abs_truncmin, sign * abs_truncmax)
#         G[key]['order_mag_sigma'] = order_mag_sigma
        
#     return G

def set_G_dist_specs(G, sd_to_mean_ratio = 0.5, trunc_with_sd = False, 
                     n_sd_trunc = 2, order_mag_sigma = None):
    
    for key, val in G.items():
        
        G[key]['sd'] = abs(val['mean'] ) * sd_to_mean_ratio

        if trunc_with_sd:

            G[key]['truncmin'] = max(0, abs( val['mean']) 
                                    - n_sd_trunc * G[key]['sd'])
            G[key]['truncmax'] = abs( val['mean']) \
                                    + n_sd_trunc * G[key]['sd']

        else:

            G[key]['truncmin'] = 10**-10
            G[key]['truncmax'] = 10**10

        G[key]['order_mag_sigma'] = order_mag_sigma
        
    return G

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

def plot_histogram(y, bins = 50, title = "", color = 'k', xlabel = 'control parameter', 
                   ax = None, plot_envelope = False, alpha = 1, frq = None, edges = None,
                   bar_hist = True, xaxis = 'linear', absolute  = False):
    
    fig, ax = get_axes(ax)
    ax.set_title(title, fontsize = 15)
    ax.set_xlabel(xlabel, fontsize = 15)
    if absolute:
        y = abs(y)

    if bar_hist and not xaxis == 'log':
        frq, edges,_ = ax.hist(y, bins, color = color, alpha = alpha)
    
    if xaxis == 'log':
        
        frq, edges = np.histogram(y, bins=bins)

        logbins = np.logspace(np.log10(edges[0]),np.log10(edges[-1]),len(edges))
        ax.hist(y, bins= logbins, color = color, alpha = alpha)
        ax.set_xscale('log')


    if plot_envelope :
        plot_hist_envelope(frq, edges, color, ax = ax)

    return ax

def plot_hist_envelope(frq, centers, color, ax = None,  lw = 1, alpha = 0.2):
    
    fig, ax = get_axes(ax)
    
    # if len(centers) != len (centers): 
    #     centers = centers[:-1]
        
    ax.plot(centers , frq, color = color,  lw = lw, alpha = alpha)
    
    return ax

# def wrap_angles(x):
#     ''' Wrap angles to [-pi, pi)'''
    # x = (x+np.pi) % (2*np.pi) 
    # x[x <= 0] = x[x <= 0] + (2*np.pi)
    # return x - np.pi

def iterate_with_step(array, step = 1):
    
    return array[:-step], array[step:]

def keep_beta_bursts(left_peak_series, right_peak_series, dt, max_cycle_t = 83):
    
    """
    remove the pairs of peaks between the left and right where the 
    distance exceeds `max_period`. Ultimately only calculating the phase by for the beta burst 
    durations.

    Parameters
    ----------
    left_peak_series : 1D-array
        first element of the peak pairs.
    right_peak_series : 1D-array
        second element of the peak pairs.
    max_cycle_t : float, optional
        maximum period of the acceptable cycle. The default is 83.

    Returns
    -------
    left_peak_series : 1D-array
        first element of the peak pairs filtered for the desired periods.
    right_peak_series : 1D-array
        second element of the peak pairs filtered for the desired periods.
    """
    ind_burst = np.where(right_peak_series - left_peak_series < max_cycle_t / dt)
    print("n all ref peaks =", len(right_peak_series),
          "n burst peaks = ", len(ind_burst[0]))
    return left_peak_series[ind_burst], right_peak_series[ind_burst]

def set_peak_iterators(total_phase, ref_peaks, dt, troughs = False, max_cycle_t = 83):
    
    n_cycles =  int(total_phase / 360)
    left_peak_series, right_peak_series = iterate_with_step(ref_peaks, step = n_cycles)
    left_peak_series, right_peak_series = keep_beta_bursts(left_peak_series, right_peak_series, 
                                                           dt, max_cycle_t = n_cycles * max_cycle_t)

    if troughs:
        
        half_mean_cycle = np.average(np.diff(ref_peaks)) / 2
        left_peak_series = left_peak_series - half_mean_cycle
        right_peak_series = right_peak_series - half_mean_cycle
        
    return left_peak_series, right_peak_series

def convert_angle_to_0_360_interval(angle):
   new_angle = np.arctan2(np.sin(angle), np.cos(angle))
   ind  = new_angle <= 0
   new_angle[ind] = abs(new_angle[ind]) + 2 * (np.pi - abs(new_angle[ind]))
   return new_angle * 180/np.pi
# convert_angle_to_0_360_interval(np.array([-np.pi/2, np.pi, np.pi * 2, 3 * np.pi / 2])) 


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

def create_gif_from_images(images_path, gif_filename, image_ext = 'png', fps = 24, 
                           optimize_gif = False, loop = None, ext = '.gif'):
    
    images =[f for f in os.listdir(images_path) if f.endswith(image_ext)]
    image_list = []
    kargs = { 'macro_block_size': None }

    for file_name in sorted(images):
        image_list.append(imageio.imread( os.path.join(images_path, file_name)))    
        
    if ext == '.gif':
        if loop != None:
            imageio.mimwrite( os.path.join(images_path, gif_filename + ext), image_list, fps = fps, loop = loop)
        else:
            imageio.mimwrite( os.path.join(images_path, gif_filename + ext), image_list, fps = fps)
        if optimize_gif:
        
            optimize(os.path.join(images_path, gif_filename + ext))
    else:
        imageio.mimwrite( os.path.join(images_path, gif_filename + ext), image_list, fps = fps)#, **kargs)


        
        
def find_duration_gif(img_obj):
    
    img_obj.seek(0)  # move to the start of the gif, frame 0
    tot_duration = 0
    # run a while loop to loop through the frames
    while True:
        try:
            frame_duration = img_obj.info['duration']  # returns current frame duration in milli sec.
            tot_duration += frame_duration
            # now move to the next frame of the gif
            img_obj.seek(img_obj.tell() + 1)  # image.tell() = current frame
        except EOFError:
            return tot_duration # this will return the tot_duration of the gif
        
    return tot_duration

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



def create_FR_ext_filename_dict(nuclei_dict, path, dt):
    
    filename_dict = {}
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            filename_dict[nucleus.name] = os.path.join(path, 
                                                       'FR_ext_' + nucleus.name + 
                                                       '_noise_var_' + str( round(
                                                                           nucleus.noise_variance , 2)
                                                                           ).replace('.', '-') +
                                                       '_dt_' + str(0.1).replace('.', '-') +
                                                       '_A_' + str(nucleus.basal_firing).replace('.', '-') +
                                                        '_tau_m_' + str( round(
                                                                          nucleus.neuronal_consts[
                                                                         'membrane_time_constant']['mean'], 2)
                                                                        ).replace('.', '-') +
                                                       '.pkl')
            
    return filename_dict

def reset_connections(nuclei_dict, K_real, N, N_real):
    
    K = calculate_number_of_connections(N, N_real, K_real)
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:

            nucleus.set_connections(K, N)

    return nuclei_dict

def set_connec_ext_inp(path, A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, 
                       nuclei_dict, t_list, c='grey', scale_g_with_N=True,
                        all_FR_list=np.linspace(0.05, 0.07, 100), n_FR=50, if_plot=False, end_of_nonlinearity=None, left_pad=0.005,
                        right_pad=0.005, maxfev=5000, ax=None, set_FR_range_from_theory=True, method = 'single_neuron', FR_ext_all_nuclei_saved = None,
                        use_saved_FR_ext = False, save_FR_ext = True, normalize_G_by_N = False, state = 'rest',
                        change_states = True, plot_syn_weight_hist = False):
    
    '''find number of connections and build J matrix, set ext inputs as well
        Note: end_of_nonlinearity has been modified to be passed as dict (incompatible with single neuron setting)'''

    K = calculate_number_of_connections(N, N_real, K_real)
    receiving_class_dict = create_receiving_class_dict(receiving_pop_list, nuclei_dict)
    
    FR_ext_all_nuclei = {}
    
    if nuclei_dict[ list( nuclei_dict.keys())[0]][0].neuronal_model != 'rate':
        
        FR_ext_filename_dict = create_FR_ext_filename_dict(nuclei_dict, path, dt)
        
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:

            nucleus.set_connections(K, N, plot_syn_weight_hist = plot_syn_weight_hist)
            
            if nucleus.neuronal_model == 'rate' and scale_g_with_N:
                
                nucleus.scale_synaptic_weight()
                nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list, dt, change_states = change_states)
                
            else:
                
                if nucleus. der_ext_I_from_curve:
                    
                    if method == 'collective' and not use_saved_FR_ext:
                        
                        der_ext_I_collective(nucleus, all_FR_list, t_list, dt, receiving_class_dict, state, FR_ext_filename_dict,
                                             FR_ext_all_nuclei, if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity,
                                             maxfev = maxfev, c = c, n_FR=n_FR, save_FR_ext = save_FR_ext )
                        
                    elif method == 'single_neuron':
                        
                        der_ext_I_single_neuron(nucleus, all_FR_list, t_list, dt, receiving_class_dict, state, FR_ext_filename_dict,
                                                FR_ext_all_nuclei, if_plot =  if_plot, end_of_nonlinearity =end_of_nonlinearity,
                                                maxfev = maxfev, c=c, n_FR=n_FR, set_FR_range_from_theory = set_FR_range_from_theory,
                                                left_pad= left_pad, right_pad=right_pad, ax = ax)

                    elif use_saved_FR_ext:
                        
                        nucleus.FR_ext_specs = load_pickle(FR_ext_filename_dict[nucleus.name])

                if normalize_G_by_N:
                    
                    nucleus.normalize_synaptic_weight_by_N()
                    
                nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list, dt, 
                                      end_of_nonlinearity = end_of_nonlinearity[nucleus.name][state])

    return receiving_class_dict, nuclei_dict

def der_ext_I_collective(nucleus, all_FR_list, t_list, dt, receiving_class_dict, state, FR_ext_filename_dict,
                         FR_ext_all_nuclei, if_plot = False, end_of_nonlinearity =None,
                         maxfev = 5000, c='grey', n_FR=50, save_FR_ext = True):
    
    print("external input is being set collectively for {0} at {1}...".format(nucleus.name, state))
    FR_ext = nucleus.set_ext_inp_const_plus_noise_collective(all_FR_list[nucleus.name], t_list, dt, receiving_class_dict,
                                                             if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity[nucleus.name][state],
                                                             maxfev = maxfev, c=c, n_FR=n_FR)
    FR_ext_all_nuclei[nucleus.name] = {'mean': FR_ext,
                                       'sd': nucleus.FR_ext_specs['sd'],
                                       'truncmin': nucleus.FR_ext_specs['truncmin'],
                                       'truncmax': nucleus.FR_ext_specs['truncmax']}
    nucleus.FR_ext_specs['mean']= FR_ext
     
    if save_FR_ext:
        
        pickle_obj(nucleus.FR_ext_specs, FR_ext_filename_dict[nucleus.name])
        
def der_ext_I_single_neuron(nucleus, all_FR_list, t_list, dt, receiving_class_dict, state, FR_ext_filename_dict,
                            FR_ext_all_nuclei, if_plot = False, end_of_nonlinearity =None,
                            maxfev = 5000, c='grey', n_FR=50, set_FR_range_from_theory = True,
                            left_pad=0.005, right_pad=0.005, ax = None):
    
    if nucleus.basal_firing > end_of_nonlinearity:
                            
        nucleus.estimate_needed_external_input_high_act(all_FR_list[nucleus.name], dt, t_list, receiving_class_dict, if_plot=if_plot,
                                                        n_FR=n_FR, ax=ax, c=c, set_FR_range_from_theory=set_FR_range_from_theory)
        
    else:
        
        nucleus.estimate_needed_external_input(all_FR_list[nucleus.name], dt, t_list, receiving_class_dict, if_plot=if_plot, 
                                               end_of_nonlinearity=end_of_nonlinearity, maxfev=maxfev,
                                               n_FR=n_FR, left_pad=left_pad, right_pad=right_pad, ax=ax, c=c)

def set_init_all_nuclei(nuclei_dict, list_of_nuc_with_trans_inp=None, filepaths=None):
    
    if list_of_nuc_with_trans_inp != None:
        filtered_nuclei_dict = {key: value for key, value in nuclei_dict.items() 
                                if key in list_of_nuc_with_trans_inp}

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

def reset_synaptic_weights_all_nuclei(nuclei_dict, G, N):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            nucleus.reset_synaptic_weights(G, N)
    
    return nuclei_dict

def reinitialize_nuclei_SNN( nuclei_dict, N, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, t_mvt, 
                             t_list, dt, state = 'rest', poisson_prop = None, mem_pot_init_method=None, 
                             set_noise=True, end_of_nonlinearity=25, reset_init_dist = False, t_sim = None, 
                             normalize_G_by_N = False):
    
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            nucleus.clear_history(mem_pot_init_method = mem_pot_init_method)
            
            if reset_init_dist:
                nucleus.set_init_distribution( nucleus.T_specs, nucleus.tau_specs, 
                                              nucleus.FR_ext_specs, poisson_prop, dt, t_sim)
                
            nucleus.reset_synaptic_weights(G, N)
            # nucleus.normalize_synaptic_weight()
            
            if normalize_G_by_N:
                nucleus.normalize_synaptic_weight_by_N()
            
            if set_noise:
                nucleus.set_noise_param(noise_variance, noise_amplitude)
            

            nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list, dt, 
                                  end_of_nonlinearity=end_of_nonlinearity)
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

def get_sig_amp_envelope(sig):
    
    analytic_signal = hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)
    
    return  amplitude_envelope

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


def find_beta_induction_nucleus(nuclei_dict):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            if nucleus.beta_stim :
                
                return True, nucleus.beta_stim_method, nucleus.name, nucleus.external_inp_t_series
            
    return False, None, None, None

def get_phase_ref_peaks( nuclei_dict, phase_ref, dt, low_f, high_f, filter_order = 6, 
                         peak_threshold = None, threshold_by_percentile = 75, 
                         start = 0, end = None, plot_ref_peaks = False,
                         shift_phases = False, shift_theta= 0, align_to_stim_onset = True, ax = None):
    
    beta_stim, beta_stim_method, beta_stim_nuc, stim_t_series = find_beta_induction_nucleus(nuclei_dict)
    
    print('aligning phases to {} peaks'.format(phase_ref))
    if beta_stim and phase_ref == 'stimulation':
        
        if beta_stim_method == 'excitation':
            
            ref_peaks, _ = signal.find_peaks( stim_t_series, height = 0)
            
        elif beta_stim_method == 'inhibition':
            
            ref_peaks, _ = signal.find_peaks( - stim_t_series, height = 0)
                        
        plot_ref_peaks_cyclic_stim(stim_t_series, ref_peaks, plot_ref_peaks)
        
        if align_to_stim_onset :
            
            shift_phases = True ; shift_theta = 90
        
        if ax != None:
            ax.plot(stim_t_series)
        act = stim_t_series
        
    elif phase_ref in list( nuclei_dict.keys()):
        
        
        ref_peaks, act = nuclei_dict[phase_ref][0].find_peaks_of_pop_act(dt, low_f, high_f, 
                                                                       filter_order = filter_order, 
                                                                       peak_threshold = peak_threshold, 
                                                                       start = start, end = end,
                                                                       threshold_by_percentile = threshold_by_percentile,
                                                                       ax = ax)
        
    else: 
        
        raise(" Check input for phase reference!")
        
    if shift_phases:
        
        print(' shifting stimulation reference peaks {} degrees'.format(shift_theta))
        ref_peaks = shift_ref_peaks(ref_peaks, theta = shift_theta)
    
    return ref_peaks, act

def shift_ref_peaks(ref_peaks, theta = 90):
    
    n_cycles = len(ref_peaks) - 1
    cycle_period = ( ref_peaks[-1] - ref_peaks[0] ) / n_cycles
    
    return (ref_peaks - theta / 360 * cycle_period ).astype(int)

def plot_ref_peaks_cyclic_stim(stim_t_series, ref_peaks, plot_ref_peaks):
    
    if plot_ref_peaks:
    
        fig, ax = plt.subplots()
        ax.plot(stim_t_series)
        ax.plot(ref_peaks, stim_t_series[ref_peaks], 'x')
        
def find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = 90,
                                       peak_threshold = None , phase_ref = None, start = 0, total_phase = 360,
                                       only_PSD_entrained_neurons = False, min_f_sig_thres = 0,window_mov_avg = 10, max_f = 250,
                                       n_window_welch = 6, n_sd_thresh = 2, n_pts_above_thresh = 2, end = None,
                                       min_f_AUC_thres = 7,  PSD_AUC_thresh = 10**-5, filter_based_on_AUC_of_PSD = False, 
                                       troughs = False, plot_ref_peaks = False, shift_ref_theta = 0, shift_ref_phases = False,
                                       align_to_stim_onset = True, shift_phase_deg = None,
                                       only_rtest_entrained = True, threshold_by_percentile = 75,
                                       plot = False, ax = None):
     
    if plot: fig, ax = plt.subplots()
    ref_peaks, act = get_phase_ref_peaks(nuclei_dict, phase_ref, dt, low_f, high_f, filter_order = filter_order, 
                                    peak_threshold = peak_threshold, start = start, end = end, plot_ref_peaks = plot_ref_peaks,
                                    shift_theta = shift_ref_theta, shift_phases = shift_ref_phases, 
                                    align_to_stim_onset= align_to_stim_onset,
                                    threshold_by_percentile = threshold_by_percentile,
                                    ax = ax)
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            if only_PSD_entrained_neurons:
                
                neurons_ind = significance_of_oscil_all_neurons( nucleus, dt, max_f = max_f, 
                                                                window_mov_avg = window_mov_avg, n_sd_thresh = n_sd_thresh, 
                                                                min_f_sig_thres = min_f_sig_thres, n_window_welch = n_window_welch, 
                                                                n_pts_above_thresh = n_pts_above_thresh,
                                                                min_f_AUC_thres = min_f_AUC_thres ,  PSD_AUC_thresh = PSD_AUC_thresh , 
                                                                filter_based_on_AUC_of_PSD = filter_based_on_AUC_of_PSD)
            else:
                
                neurons_ind = 'all'
                
            nucleus.find_phase_hist_of_spikes(dt, low_f, high_f, filter_order = filter_order, n_bins = n_bins,
                                              start = start, ref_peaks = ref_peaks, 
                                              total_phase = total_phase, phase_ref = phase_ref, 
                                              neurons_ind = neurons_ind, end = end, 
                                              keep_only_entrained = only_rtest_entrained,
                                              act = act, ax = ax)

    if shift_phase_deg != None:
        nuclei_dict = shift_phase_hist_all_nuclei(nuclei_dict, shift_phase_deg, phase_ref, total_phase = 720)
                # nucleus.shift_phase( shift_all_theta, phase_ref, total_phase)
                
            # elif phase_ref in list( nuclei_dict.keys()) :
                
                # nucleus.shift_phase( 180, phase_ref, total_phase)
                
    return nuclei_dict
                  
def find_phase_sine_fit(x, y):
    
    ''' fit sine function derive phase'''
    
    A, w, p, c, f, fitfunc = fit_sine(x, y)
    
    if A > 0:
        
        phase_sine =  np.pi/ 2/ w + p 
        
    else:
        
        phase_sine = - np.pi/ 2/ w + p 
        
    phase_sine = equi_phase_in_0_360_range(phase_sine)
    
    return phase_sine, fitfunc, w

# def shift_small_phases_to_next_peak(phase, w, nuc_name, phase_ref):
#     if nuc_name  != phase_ref:
#         if phase < 30:
#             phase += 2 * np.pi / w
#     return phase

# def shift_large_phases_to_prev_peak(phase, w, nuc_name, phase_ref):
#     ''' If outliers (minority) are on the second peak, shift them back'''
#     if nuc_name  != phase_ref:
#         if phase > 340 :
#             print(nuc_name, 'shifting backward, phase before = ', phase)
#             phase -= 2 * np.pi / w
#             print('phase after = ', phase)
#     return phase

def shift_small_phases_to_next_peak(phases, nuc_name, phase_ref, ws = None):
    
    mean_phases = np.average(phases)
    
    if nuc_name  != phase_ref:
        if mean_phases > 100:
            
            ind = np.where(phases < 30)
            
            print(nuc_name, 'shifting forward, phase before = ', phases[ind])
            # phases[ind] += 2 * np.pi / ws[ind]
            phases[ind] += 360
            print('phase after = ', phases[ind])
            
    return phases

def shift_large_phases_to_prev_peak(phases, nuc_name, phase_ref, ws = None):
    
    ''' If outliers (minority) are on the second peak, shift them back'''
    
    std_phases = np.std(phases)
    
    # if nuc_name  != phase_ref:
        
    if std_phases > 100:
        
        ind = np.where(phases > 300)
        print(nuc_name, 'shifting backward, phase before = ', phases[ind])
        # phases[ind] -= 2 * np.pi / ws[ind]
        phases[ind] -= 360
        print('phase after = ', phases[ind])
        
    return phases

def shift_second_peak_phases_to_prev_peak(phase, nuc_name, phase_ref, w = None):
    
    ''' If the second peak is detected shift to the previous'''
    
    if nuc_name  != phase_ref:
        
        if phase > 360 :
            
            print(nuc_name, 'shifting to first peak, phase before = ', phase)
            phase -= 360
            # phase -= 2 * np.pi / w
            print('phase after = ', phase)
            
    return phase

def find_phase_from_max(x, y):
    
    ''' find the phase of where the function maximises'''
    
    phase_max = x[ np.argmax( y ) ]
    phase_max = equi_phase_in_0_360_range(phase_max)
    
    return phase_max
            
def find_phase_from_sine_and_max(x,y, nuc_name, phase_ref, shift_phase = None):
    
    ''' In case sine fit and maximum are close (90 deg) go with maximum, 
        Otherwise sine is to be trusted'''
        
    # phase_sine,fitfunc, w = find_phase_sine_fit(x, y)
    phase_max = find_phase_from_max(x, y)
    # print( nuc_name, 'sine p: ', np.round( phase_sine, 1) , 'max p:', np.round( phase_max, 2))
    # phase = decide_bet_max_or_sine(phase_max, phase_sine, nuc_name, phase_ref)
    phase = phase_max
    phase = shift_second_peak_phases_to_prev_peak(phase, nuc_name, phase_ref)

    # return phase, fitfunc, w
    return phase, 0, 0

def correct_phases(phases, ws, nuc_name, phase_ref, shift_phase = None):
    
    if shift_phase == 'backward':

        phases = shift_large_phases_to_prev_peak(phases, nuc_name, phase_ref)
        
    elif shift_phase == 'forward':
        
        phases = shift_small_phases_to_next_peak(phases, nuc_name, phase_ref)
        
    elif shift_phase == 'both':
        
        phases = shift_small_phases_to_next_peak(phases, nuc_name, phase_ref)
        phases = shift_large_phases_to_prev_peak(phases, nuc_name, phase_ref)

    return phases

def decide_bet_max_or_sine(phase_max, phase_sine, nuc_name, phase_ref, preferred_when_consistent = 'sine'):
    
    # if abs( phase_sine - phase_max) < 90: # two methods are consistent, 
    
    #     if preferred_when_consistent == 'sine':
            
    #         phase = phase_sine #     sine is more accurate for high degree resolutions
            
    #     elif preferred_when_consistent == 'max':
            
    #         phase = phase_max #     max is more accurate  for low degree resolutions

    # else:
        
    #     if phase_sine < 30 and nuc_name != phase_ref: #### if one phase is detected in the beginnig and they are >90 apart --> choose the other one (apprent peak in phase)
    #         phase = phase_max
            
    #     elif phase_max < 30 and nuc_name != phase_ref:
    #         phase = phase_sine
            
    #     else:
    #         phase = phase_sine
    phase = phase_max
    return phase

def mean_std_multiple_sets(means, stds, nums, sem_instead_of_std = False):
    
    if sem_instead_of_std:
        
        stds = stds * np.sqrt(nums)
        
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
    if phase >= 360 : phase -= 360
    
    return phase

def get_centers_from_edges(edges): 
    
    centers = edges + (edges[1] - edges[0]) / 2
    
    return centers

def save_phases_into_dataframe(nuclei_dict, data, i,j, phase_ref, shift_phase = None):

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            frq  = nucleus.neuron_spike_phase_hist[phase_ref]
            edges = nucleus.phase_bins
            
            #### average phase of all neurons

            ( data[(nucleus.name, 'rel_phase_hist')][i,j,:frq.shape[0],:], 
              data[(nucleus.name, 'rel_phase_hist_bins')] ) = frq , edges
            
            centers = get_centers_from_edges(edges[:-1])
            
            data[(nucleus.name, 'rel_phase')][i,j],_,_ = find_phase_from_sine_and_max(centers, np.average( frq, axis = 0), 
                                                                                         nucleus.name, 
                                                                                         phase_ref, shift_phase = shift_phase)
    
    return data
 
def save_phases_into_dataframe_2d(nuclei_dict, data, i, m, j, phase_ref, shift_phase = None):

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            frq = nucleus.neuron_spike_phase_hist[phase_ref]
            edges = nucleus.phase_bins
            
            ( data[(nucleus.name, 'rel_phase_hist')][i,m, j,:,:], 
              data[(nucleus.name, 'rel_phase_hist_bins')] ) = frq , edges
            
            centers = get_centers_from_edges(edges[:-1])
            
            data[(nucleus.name, 'rel_phase')][i,m,j],_,_ = find_phase_from_sine_and_max(centers, 
                                                                                           np.average( frq, axis = 0), 
                                                                                           nucleus.name, 
                                                                                           phase_ref, shift_phase = shift_phase)
    
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

def set_minor_locator(ax, n = 2, axis = 'x'):
    
    minor_locator = AutoMinorLocator(n)
    
    if axis == 'y':
        ax.yaxis.set_minor_locator(minor_locator)
        
    if axis == 'x':
        ax.xaxis.set_minor_locator(minor_locator)
        
    if axis == 'both':
        ax.yaxis.set_minor_locator(minor_locator)
        minor_locator = AutoMinorLocator(n)
        ax.xaxis.set_minor_locator(minor_locator)
        
def set_minor_locator_all_axes(fig, n_x = 2, n_y = 2,  n = 2, axis = 'x'):
    
    
    for ax in fig.axes:
        
        
        if axis == 'y':
            minor_locator = AutoMinorLocator(n)
            ax.yaxis.set_minor_locator(minor_locator)
            
        if axis == 'x':
            minor_locator = AutoMinorLocator(n)
            ax.xaxis.set_minor_locator(minor_locator)
            
        if axis == 'both':
            
            minor_locator_y = AutoMinorLocator(n_y)
            ax.yaxis.set_minor_locator(minor_locator_y)
            minor_locator_x = AutoMinorLocator(n_x)
            ax.xaxis.set_minor_locator(minor_locator_x)
   
def create_FR_dict_from_sim(nuclei_dict, start, stop, state = 'rest'):

    FR_dict = {name: 
               { state : 
                    {'mean': np.average( nuc[0]. pop_act [start:stop ]),
                     'SEM': stats.sem( nuc[0]. pop_act [start:stop ])}
                }
               for name, nuc in nuclei_dict.items()}
    
    return FR_dict

def plot_mean_FR_in_phase_hist(FR_dict, name, ax, angles, color_dict, 
                               state, lw = 1, alpha = 0.1, plot_FR = False):
    
    if plot_FR:
        
        try:
            ax.plot(angles, np.full(len(angles),
                                    FR_dict[name][state]['mean']), 
                    '-.' ,
                    color = color_dict[name], lw = lw)
            ax.fill_between(angles, 
                            np.full ( len(angles), 
                                             FR_dict[name][state]['mean'] - FR_dict[name][state]['SEM']), 
                            np.full ( len(angles), 
                                             FR_dict[name][state]['mean'] + FR_dict[name][state]['SEM']),
                           alpha = alpha , color = color_dict[name])
        except:
            ax.plot(angles, np.full(len(angles),
                                    FR_dict[name]), 
                    '-.' ,
                    color = color_dict[name], lw = lw)
def extract_spike_phases_Brice(filepath, shift_phase_deg = 0, total_phase = 360):
    
    '''extract the spike phases reported for a single neuron for each laser stimulation '''
    
    df = pd.read_table(filepath, skiprows=[0], header = [0])
    df = df [ df ['Times'].notna() ].reset_index()
    
    n_sweeps = int(np.max(df [ df ['Sweep'].notna() ] ['Sweep'].values))
    
    corr_laser_sweep_inds = np.where( df['Events'].notna() )[0]
    spike_phases = np.remainder( df['Times'] + 90 + shift_phase_deg, total_phase) ## laser max is at phase = 0
    
    return spike_phases, corr_laser_sweep_inds, n_sweeps

def double_phase_hist_cycle(frq, edges):
    
    frq_doubled = np.concatenate( (frq, frq), axis = 0)
    edges_doubled = np.concatenate( (edges, edges[1:] + 360), axis = 0)
    return frq_doubled, edges_doubled

def double_phase_hist_cycle_2d(frq, edges):

    frq_doubled = np.concatenate( (frq, frq), axis = 1)
    edges_doubled = np.concatenate( (edges, edges[1:] + 360), axis = 0)
    return frq_doubled, edges_doubled


def look_at_one_neuron_laser_Brice(filepath, total_phase, n_bins, name, color_dict):

    spike_phases, corr_laser_sweep_inds, n_sweeps = extract_spike_phases_Brice(filepath)
    frq, edges = get_hist(spike_phases, end_bins = total_phase, n_bins = n_bins)
    
    ax = plot_histogram(spike_phases, bins = edges, title = "", color = color_dict[name], xlabel = 'Phase (deg)', 
                        ax = None, plot_envelope = True, alpha = 0.2, frq = frq, edges = edges)
    raster_plot_Brice(spike_phases , corr_laser_sweep_inds, color_dict[name], ax = ax)
    

def cal_neuron_phase_hist_Brice(spike_phases, bins, n_sweeps):
    
    frq , edges = np.histogram(spike_phases, bins = bins)
    frq = frq / n_sweeps 
    frq , edges = double_phase_hist_cycle(frq, edges)
        
    return frq, edges

def integrate_Brice_single_neuron_phases(path, name, total_phase, n_bins, shift_phase_deg = 0):
    """ extract single neuron phases provided in different .txt files and 
        integrate them returning the phase histogram of all neurons stacked
    """
    # fig, ax = get_axes(ax)
    
    filepath_list = list_files_of_interest_in_path(os.path.join( path, name + '_Phase'), 
                                                   extensions = ['txt'])
    bins = np.linspace(0, 360, endpoint=True, num = n_bins)
    
    n_neurons = len(filepath_list)
    frq_all_neurons = np.zeros(( n_neurons, len(bins) * 2 - 2))
    
    for i, filepath in enumerate( filepath_list ):
        spike_phases, corr_laser_sweep_inds, n_sweeps = extract_spike_phases_Brice(filepath, shift_phase_deg)
        # print('n sweeps = ', n_sweeps)
        frq_all_neurons[i, :], edges = cal_neuron_phase_hist_Brice(spike_phases, bins, n_sweeps)
    
    
    try :
        
        centers = get_centers_from_edges(edges[:-1])
        return frq_all_neurons, centers, n_neurons
    
    except: ## no spikes 
        
        return frq_all_neurons, [0] * ((n_bins - 1) * 2), 0
    
def integrate_Asier_single_neuron_phases(path, name, total_phase, n_bins, f_stim, color_dict,  
                                          scale_count_to_FR = False, ax = None, 
                                          align_to = 'Laser', n_sweeps = 1200):
    
    """ extract single neuron phases provided together as columns in a single .csv file and 
        integrate them returning the phase histogram of all neurons stacked
    """
    print(name)
    stim_name = os.path.basename(path).split('_')[1]
    df_n_stim = pd.read_excel(os.path.join(path, stim_name  + '_stim_beta.xlsx'), 
                              sheet_name = name, header = [0])
    filepath = list_files_of_interest_in_path(os.path.join( path, name + '_Phase'), 
                                                    extensions = [ align_to + '.csv'])[0]
    
    df = pd.read_csv(filepath, header = [0])
    
    df = df.fillna(0)
    neurons = list(df.columns)
    neurons.remove('Degree')
    n_neurons = len(neurons)
    frq_all_neurons = df[neurons].values.T / (df_n_stim['Stim #'].values.reshape(-1,1) * 30 * f_stim)
    edges = np.append(df['Degree'].values, 360)
    bin_deg = edges[1] - edges[0]
    n_shift = int(90 / bin_deg)  
    
    if align_to == 'Laser':
        frq_all_neurons = np.roll(frq_all_neurons, n_shift) # assuming that hist is aligned to laser max.

    frq_all_neurons, edges = double_phase_hist_cycle_2d(frq_all_neurons, edges)
    centers = get_centers_from_edges(edges[:-1])
    
    return frq_all_neurons, centers, n_neurons

def scale_phase_spike_count(count, err, total_phase, n_bins, coef = 1):
    
    if total_phase == 720:
        n_bins += 2
    bin_size_in_deg = total_phase / n_bins
    print('bin size in deg = ', bin_size_in_deg)
    return count * coef / bin_size_in_deg, err * coef / bin_size_in_deg


def phase_plot_experiment_laser_aligned( experiment_protocol, name_list, color_dict, path, y_max_series, n_bins = 36, f_stim = 20, scale_count_to_FR = False, 
                                    total_phase = 360,  box_plot = False, set_ylim = False,title_fontsize = 15,
                                    print_stat_phase = False, coef = 1, phase_text_x_shift = 150, phase_txt_fontsize = 8, 
                                    phase_txt_yshift_coef = 1.4, lw = 0.5, name_fontsize = 8, tick_label_fontsize = 8,
                                    name_ylabel_pad = [0,0,0], name_side = 'right', name_place = 'ylabel', alpha = 0.15, 
                                    alpha_single_neuron = 0.1, title = '', lw_single_neuron = 1,
                                    xlabel_y = 0.05, ylabel_x = -0.1, n_fontsize = 8, plot_FR = False, n_decimal = 0,
                                    n_minor_tick_y = 4, n_minor_tick_x = 4, xlabel_fontsize = 8, FR_dict = None, 
                                    ylabel_fontsize = 8, xlabel = 'phase (deg)',strip_plot = False, state  = 'OFF',
                                    plot_single_neuron_hist = True, smooth_hist = False, hist_smoothing_wind = 5, 
                                    shift_phase = None, plot_mean_FR = True, align_to = 'Laser', 
                                    shift_phase_deg =  0, n_neurons_to_plot = 10, random_seed = None):

    n_subplots = len(name_list)
    xy = {name :(0.7, 0.7) for name in name_list}

    fig = plt.figure(figsize = (3, 1.5 * n_subplots))
    outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)

    inner = gridspec.GridSpecFromSubplotSpec(n_subplots, 1,
            subplot_spec=outer[0], wspace=0.1, hspace=0.1)
                
    for j, name in enumerate(name_list):
        
        ax = plt.Subplot(fig, inner[j])
        fig.add_subplot(ax)
        
        if 'mouse' in experiment_protocol: # data is coming from Asier's experiments, will adjust to that data format
            
            hists , centers, n_neurons =  integrate_Asier_single_neuron_phases(path, name, total_phase, n_bins, 
                                                                                     f_stim, color_dict, ax = ax,
                                                                                     scale_count_to_FR = scale_count_to_FR,
                                                                                     align_to = align_to)
            
        else:
            
            hists , centers, n_neurons =  integrate_Brice_single_neuron_phases(path, name, total_phase, n_bins,
                                                                               shift_phase_deg = shift_phase_deg)
            
        count = np.average(hists, axis = 0)
        err = stats.sem(hists, axis = 0)
        count, err = scale_phase_spike_count(count, err, total_phase, len(centers), coef = coef)

        phases = calculate_phase_all_runs(n_neurons, hists, centers, name, None, 
                                              shift_phase = shift_phase)

            
        hists =  smooth_hists(hists, hist_smoothing_wind, smooth_hist = smooth_hist)
        make_phase_plot(count, err, name, ax, centers, 
                        color_dict, phases,  coef, y_max_series,  plot_FR = plot_FR, state = state,
                        f_stim = f_stim, scale_count_to_FR = scale_count_to_FR, lw = lw, alpha =alpha ,
                        print_stat_phase = print_stat_phase , box_plot = box_plot, FR_dict = FR_dict, 
                        phase_text_x_shift = phase_text_x_shift , phase_txt_fontsize = phase_txt_fontsize , 
                        phase_txt_yshift_coef = phase_txt_yshift_coef, total_phase = 720,
                        single_neuron_traces  = plot_single_neuron_hist,
                        plot_mean_FR = plot_mean_FR)
         
        annotate_txt(ax, 'n=' + str( n_neurons) , name, color_dict[name], n_fontsize, xy = xy)
        
        if plot_single_neuron_hist:
            
            plot_single_neuron_hists(name, hists, n_neurons_to_plot,
                                    coef, f_stim, centers, total_phase, ax, scale_count_to_FR,
                                    color_dict, alpha = alpha_single_neuron, random_seed = random_seed, lw = lw_single_neuron)
            
        set_ax_prop_phase(ax, name, color_dict, name_ylabel_pad, name_fontsize, name_place, name_side,
                          y_max_series, 720, n_decimal, tick_label_fontsize, n_subplots, j, set_ylim)
        
    set_fig_prop_phase(fig,  n_x = n_minor_tick_x, n_y = n_minor_tick_y, strip_plot = strip_plot, xlabel_y = xlabel_y, ylabel_x = ylabel_x,
                       xlabel_fontsize = xlabel_fontsize, ylabel_fontsize = ylabel_fontsize,  xlabel = xlabel, 
                       title = title, title_fontsize = title_fontsize,
                       n_bins = len(centers), total_phase = total_phase, coef = coef, 
                       scale_count_to_FR = scale_count_to_FR)
    
    return fig

def up_scale_mean_err(mean, err, coef):
    
    return mean * coef, err * coef

def phase_plot_Brice_EcoG_aligned(phase_dict, angles, name_list, color_dict, phase_ref = None, total_phase = 720, 
                                  n = 1000, set_ylim = True, shift_phase = None, y_max_series = None, xlabel_fontsize = 8,
                                  ylabel_fontsize = 8, phase_txt_fontsize = 8, tick_label_fontsize = 8, 
                                  ylabel = None, xlabel = 'phase (deg)', coef = 1000, lw = 0.5, name_fontsize = 8, 
                                  name_ylabel_pad = 4, name_place = 'ylabel', alpha = 0.15, title = '',
                                  xlabel_y = 0.05, ylabel_x = -0.1, n_fontsize = 8, state = 'OFF',
                                  phase_txt_yshift_coef = 1.4, name_side = 'right', n_decimal = 0, title_fontsize = 8,
                                  print_stat_phase = True, strip_plot = False, box_plot = True, FR_dict = None,
                                  phase_text_x_shift = 150, f_stim = 20, scale_count_to_FR = False, plot_FR = False,
                                  n_minor_tick_y = 4, n_minor_tick_x = 4):
    
    n_subplots = len(name_list)
    fig = plt.figure(figsize = (3, 1.5 * n_subplots))
    outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)

    inner = gridspec.GridSpecFromSubplotSpec(n_subplots, 1,
            subplot_spec=outer[0], wspace=0.1, hspace=0.1)
    
  
    for j, name in enumerate(name_list):
        
        ax = plt.Subplot(fig, inner[j])
        fig.add_subplot(ax)
        
        count, err = phase_dict[name]['count'] , phase_dict[name]['SEM']
        count, err = scale_phase_spike_count(count, err, total_phase, len(angles), coef = coef)

        make_phase_plot(count, err, name, ax, angles, color_dict,  phase_dict[name]['phase'], 
                        coef, y_max_series,  plot_FR = plot_FR, state = state, 
                        f_stim = f_stim, scale_count_to_FR = scale_count_to_FR, lw = lw, alpha =alpha ,
                        print_stat_phase = print_stat_phase , box_plot = box_plot, FR_dict = FR_dict,
                        phase_text_x_shift = phase_text_x_shift , phase_txt_fontsize = phase_txt_fontsize , 
                        phase_txt_yshift_coef = phase_txt_yshift_coef, total_phase = total_phase)

       
        annotate_txt(ax, 'n=' + str( phase_dict[name]['n'] ) , name, color_dict[name], n_fontsize)

        
        set_ax_prop_phase(ax, name, color_dict, name_ylabel_pad, name_fontsize, name_place, name_side,
                              y_max_series, total_phase, n_decimal, tick_label_fontsize, n_subplots, j, set_ylim)

    
    set_fig_prop_phase(fig,  n_x = n_minor_tick_x, n_y = n_minor_tick_y, strip_plot = strip_plot, 
                       xlabel_y = xlabel_y, ylabel_x = ylabel_x, 
                       xlabel_fontsize = xlabel_fontsize, ylabel_fontsize = ylabel_fontsize,  xlabel = xlabel, 
                       title = title, title_fontsize = title_fontsize,
                       n_bins = len(angles), total_phase = total_phase, coef = coef, 
                       scale_count_to_FR = scale_count_to_FR)
    return fig


def read_Brice_EcoG_aligned_phase_hist(filename, name_list, fig_ind_hist, fig_ind_phase, angle_header, coef = 1000, sheet_name = 'Fig 3'):
    

    name_list_Brice = ['STN', 'Arkypallidal', 'Prototypic']
    
    xls = pd.ExcelFile(filename)
    data = pd.read_excel(xls, sheet_name, header = [0, 1, 2])#, skiprows = [0])
    
    angles = data[fig_ind_hist, angle_header, 'Angle(in °)'].values
    angles = angles[~ np.isnan( angles) ]
    n_angles = len(angles)
    
    phase_dict = {name: {'count' : data[fig_ind_hist, name_k, 'MEAN count/degree'].values [:n_angles] * 10 * coef, # x 10 because binsize is 10 degrees but value is count/deg
                         'SEM': data[fig_ind_hist, name_k, 'SEM'].values [:n_angles] * 10* coef,
                         'phase': data[fig_ind_phase, name_k, 'Phase Angle'].values [
                                  ~ np.isnan(data[fig_ind_phase, name_k, 'Phase Angle'].values)] ,
                              
                         'n' : int( data[fig_ind_hist, name_k, 'NB'].values[0]) }
                  for name_k, name  in zip(name_list_Brice, name_list)}
    return angles, phase_dict

def read_Brice_FR_states(filename, name_list, first_header, sheet_name = 'Fig 3'):
    

    name_list_Brice = ['STN', 'Arkypallidal', 'Prototypic']
    state_list = ['ON', 'OFF']#, 'Park']
    xls = pd.ExcelFile(filename)
    data = pd.read_excel(xls, sheet_name, header = [0, 1, 2])#, skiprows = [0])


    FR_dict = {name: {state: {'mean' : np.average(data[first_header, name_k + ' (Firing rate in Spk/s)', state].values [
                                                ~ np.isnan(data[first_header, name_k + ' (Firing rate in Spk/s)', state].values)]),
                              'SEM': stats.sem(data[first_header, name_k + ' (Firing rate in Spk/s)', state].values [~ np.isnan(
                                            data[first_header, name_k + ' (Firing rate in Spk/s)', state].values)])}
                          for state in state_list}
                  for name_k, name  in zip(name_list_Brice, name_list)}
                  
    for name, name_k in zip( name_list, name_list_Brice):
        FR_dict[ name]['DD'] = {'mean' : np.average(data[first_header, name_k + ' (Firing rate in Spk/s)', name + ' Park'].values [
                                                ~ np.isnan(data[first_header, name_k + ' (Firing rate in Spk/s)', name + ' Park'].values)]),
                              'SEM': stats.sem(data[first_header, name_k + ' (Firing rate in Spk/s)', name + ' Park'].values [~ np.isnan(
                                            data[first_header, name_k + ' (Firing rate in Spk/s)', name + ' Park'].values)])}
    return FR_dict

def make_phase_plot(count, err, name, ax, angles, color_dict, phases, 
                    coef, y_max_series,  plot_FR = False, 
                    f_stim = 20, scale_count_to_FR = False, lw = 0.5, alpha = 0.15,
                    print_stat_phase = True, box_plot = True, FR_dict = None,
                    phase_text_x_shift = 150, phase_txt_fontsize = 8, 
                    phase_txt_yshift_coef = 1.4, total_phase = 720, state = 'OFF',
                    single_neuron_traces  = False,
                    plot_mean_FR = True):
    
    if scale_count_to_FR:

        
        count, err = scale_spk_count_to_FR(count, err, f_stim,
                                           coef = coef)
        print('Mean FR-ON = ',  str(round(np.average(count), 2)))
        
        if plot_mean_FR:
            plot_mean_FR_in_phase_hist(FR_dict, name, ax, angles, color_dict, 
                                       state, lw = lw, alpha = alpha, plot_FR = plot_FR)
    print(name, 
          'min FR = ', round(min(count), 1),
          'max FR = ', round(max(count), 1))
    
    plot_mean_phase_plus_std(count, err, name, ax, 
                             color_dict, angles, lw = lw, alpha = alpha)
    
        
    if box_plot:
        
        box_width = y_max_series[name] / 5
        box_y = y_max_series[name] / 3
        
        if single_neuron_traces:
            box_y = y_max_series[name] * 0.8
            
        boxplot_phases(ax, color_dict, phases, name, box_width, 
                       box_y, y_max_series[name] , phase_txt_fontsize = phase_txt_fontsize,
                       phase_txt_yshift_coef = phase_txt_yshift_coef ,
                       print_stat_phase = print_stat_phase,
                       text_x_shift = phase_text_x_shift)
        
def get_y_label_phase_hist(n_bins, total_phase = 720, coef = 1, scale_count_to_FR = False):
    
    if scale_count_to_FR:
        
        return r'$ Mean \; Firing \; rate \; (spk/s)$'
    else:
        return  (r'$ Mean \; spike \; count\;/ \; degree (.10 ^{-' + 
                str( int( math.log10(coef))) +
                '})$' )
    # elif total_phase == n_bins :
        
    #     return  (r'$ Mean \; spike \; count\;/ \; degree (.10 ^{-' + 
    #             str( int( math.log10(coef))) +
    #             '})$' )
    
    # else:
        
    #     return  (r'$ Mean \; spike \; count\;/ \; ' + 
    #              str(int( total_phase / n_bins) )  + 
    #             ' \; degrees \; (.10 ^{-' + 
    #             str( int( math.log10(coef))) +
    #             '})$' )
    
def phase_plot(filename, name_list, color_dict, n_g_list, phase_ref = 'Proto', total_phase = 720, 
                  n = 1000, set_ylim = True, shift_phase = None, y_max_series = None, xlabel_fontsize = 8,
                  ylabel_fontsize = 8, phase_txt_fontsize = 8, tick_label_fontsize = 8, 
                  ylabel = None, xlabel = 'phase (deg)', coef = 1, lw = 0.5, name_fontsize = 8, 
                  name_ylabel_pad = 4, name_place = 'ylabel', alpha = 0.1, alpha_single_neuron = 0.1, f_stim = 20,
                  xlabel_y = 0.05, ylabel_x = -0.1, phase_txt_yshift_coef = 1.5, title = '', title_fontsize = 8,
                  name_side = 'right', print_stat_phase = True, strip_plot = False, lw_single_neuron = 1,
                  box_plot = True, phase_text_x_shift = 150, n_decimal = 0, scale_count_to_FR = False,
                  state = 'OFF', FR_dict = None, plot_FR = False,
                  n_minor_tick_y = 2, n_minor_tick_x = 4, plot_single_neuron_hist = False,
                  n_neuron_hist = 10 , hist_smoothing_wind = 5, plot_mean_FR = True,
                  smooth_hist = True, shift_phase_deg = None, random_seed = None):
    
    xy = {name :(0.6, 0.8) for name in name_list}
    n_subplots = len(name_list)
    fig = plt.figure(figsize = (3, 1.5 * n_subplots))
    outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
    
    data = load_pickle(filename)
    
    
    n_run = data[(name_list[0], 'rel_phase_hist')].shape[1] 
    n_neuron = data[(name_list[0], 'rel_phase_hist')].shape[2] 
    name_ylabel_pad = handle_label_pads(name_list, name_ylabel_pad)
    
    for i, n_g in enumerate(n_g_list):
        
        inner = gridspec.GridSpecFromSubplotSpec(n_subplots, 1,
                subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        
        for j, name in enumerate(name_list):
            
            ax = plt.Subplot(fig, inner[j])
            fig.add_subplot(ax)
            
            edges = data[(name,'rel_phase_hist_bins')]
            
            centers = get_centers_from_edges(edges[:-1])
            

           
            count, std, err, hists = get_stats_of_phase(data, n_g, name, n, n_run, 
                                                        centers, shift_phase_deg = shift_phase_deg)
            
            hists =  smooth_hists(hists, hist_smoothing_wind, smooth_hist = smooth_hist)
            
            phases = calculate_phase_all_runs(n_neuron, hists, centers, name, phase_ref, 
                                              shift_phase = shift_phase)
            count, err = scale_phase_spike_count(count, err, total_phase, len(centers), coef = coef)

            make_phase_plot(count, err, name, ax, centers, color_dict,  phases, 
                            coef, y_max_series,  plot_FR = plot_FR, state = state,
                            f_stim = f_stim, scale_count_to_FR = scale_count_to_FR, lw = lw, alpha =alpha ,
                            print_stat_phase = print_stat_phase , box_plot = box_plot, FR_dict = FR_dict,
                            phase_text_x_shift = phase_text_x_shift , phase_txt_fontsize = phase_txt_fontsize , 
                            phase_txt_yshift_coef = phase_txt_yshift_coef, total_phase = total_phase,
                            single_neuron_traces  = plot_single_neuron_hist,
                            plot_mean_FR = plot_mean_FR)
            
            perc_entrained = int(hists.shape[0] / n_neuron / n_run * 100 )
            annotate_txt(ax, str( perc_entrained ) + ' %', name, color_dict[name], name_fontsize/1.5, xy)
            
            set_ax_prop_phase(ax, name, color_dict, name_ylabel_pad, name_fontsize, name_place, name_side,
                              y_max_series, total_phase, n_decimal, tick_label_fontsize, n_subplots, j, set_ylim)
            
            if plot_single_neuron_hist:
                
                plot_single_neuron_hists(name, hists, n_neuron_hist,
                                        coef, f_stim, centers, total_phase, ax, scale_count_to_FR,
                                        color_dict, lw = lw_single_neuron, random_seed = random_seed, alpha = alpha_single_neuron)
    
    set_fig_prop_phase(fig, n_x = n_minor_tick_x,  n_y = n_minor_tick_y , strip_plot = strip_plot, xlabel_y = xlabel_y, ylabel_x = ylabel_x,
                       xlabel_fontsize = xlabel_fontsize, ylabel_fontsize = ylabel_fontsize,  xlabel = xlabel, 
                       ylabel = ylabel, title = title, title_fontsize = title_fontsize,
                       n_bins = len(centers), total_phase = total_phase, coef = coef, scale_count_to_FR = scale_count_to_FR)
    return fig

def plot_single_neuron_hists( name, hists, n_neuron_hist, coef, 
                             f_stim, centers, total_phase, ax,scale_count_to_FR,
                             color_dict, alpha = 0.2, random_seed = None, lw = 1):
    
    # neuron_id = np.random.choice(n_neuron, size = n_neuron_hist, replace = False)
    # run_id = np.random.choice(n_run, size = n_neuron_hist)
    # frq_single_neurons = data[(name,'rel_phase_hist')][n_g, run_id, neuron_id, :].copy()
    
    n_neuron = hists.shape[0]
    n_neuron_hist = min(n_neuron, n_neuron_hist)
    if isinstance(random_seed, int):
        print('seed set')
        rng = np.random.RandomState(random_seed)
        neuron_id = rng.choice(n_neuron, size = n_neuron_hist, replace = False)

    else:
        
        neuron_id = np.random.choice(n_neuron, size = n_neuron_hist, replace = False)
    frq_single_neurons = hists[neuron_id, :].copy()
    frq_single_neurons, _ = scale_phase_spike_count(frq_single_neurons, np.empty((0,1)), total_phase, len(centers), coef  = coef)

    if scale_count_to_FR:
        
        scale = cal_scale_of_spk_count_to_FR(f_stim, 
                                             coef = coef)
        frq_single_neurons = frq_single_neurons * scale
    
    for n in range(n_neuron_hist):    
                                                                 
        plot_hist_envelope( 
            frq_single_neurons[n,:] , 
            centers, color_dict[ name ], ax = ax,  lw = lw, alpha = alpha)
        
def set_ax_prop_phase(ax, name, color_dict, name_ylabel_pad, name_fontsize, name_place, name_side,
                      y_max_series, total_phase, n_decimal, tick_label_fontsize, n_subplots, j, set_ylim):
    
    print_pop_name(ax, name, color_dict[name], name_ylabel_pad[j], name_fontsize, 
                   name_place, name_side)
    
    if set_ylim:
        ax.set_ylim(0, y_max_series[name]  + y_max_series[name] * .1)
        set_y_ticks_one_ax(ax, [0, y_max_series[name]])
    
    ax.axvline(total_phase/2, ls = '--', c = 'k', dashes=(5, 10), lw = 1)
    set_x_ticks_one_ax(ax, [0, 360, 720])
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.' + str(n_decimal) + 'f'))
    rm_ax_unnecessary_labels_in_subplots(j, n_subplots, ax, axis = 'x')
    remove_frame(ax)
    
def set_fig_prop_phase(fig, n_x = 4, n_y = 4, strip_plot = False, xlabel_y = 0.05, ylabel_x = -0.1,
                       xlabel_fontsize = 8, ylabel_fontsize = 8,  xlabel = 'phase (deg)', 
                       ylabel = None, scale_count_to_FR = False,
                       title = '', title_fontsize = 15, n_bins = 720,
                       total_phase = 720, coef = 1):
    
    set_minor_locator_all_axes(fig, n_x = n_x, 
                                    n_y = n_y, 
                                    axis = 'both') 
    
    
    ylabel = ylabel or get_y_label_phase_hist(n_bins, total_phase = 720, coef = coef, 
                                              scale_count_to_FR = scale_count_to_FR)
    if strip_plot:
        
        fig = remove_all_labels(fig)
        
    else:
        
        fig.text(0.5, xlabel_y, xlabel, ha='center',
                     va='center', fontsize=xlabel_fontsize)
        fig.text(ylabel_x, 0.5, ylabel, ha='center', va='center',
                     rotation='vertical', fontsize=ylabel_fontsize)
        
    fig.suptitle(title, fontsize = title_fontsize)
    
def create_figs_for_phase_subplots(fig, outer, nuc_order, nuclei_dict, figsize = (3, 7.5)):
    
    fig_generated = False
    outer_generated = False
    
    if fig == None:
        
        fig_generated = True
        fig = plt.figure(figsize = figsize)
        
    if outer == None:
        
        outer_generated = True
        outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)[0]

    inner = gridspec.GridSpecFromSubplotSpec(len(nuclei_dict), 1,
                    subplot_spec=outer, wspace=0.1, hspace=0.1)
    
        
    if nuc_order == None:
        
        nuc_order = list(nuclei_dict.keys())
    
    return fig_generated, outer_generated, fig, outer, inner, nuc_order

def cal_scale_of_spk_count_to_FR(f_stim, coef = 1):

    '''scaling parameter to go from spike count per degree to Hertz'''

    return 360 * f_stim / coef 


def scale_spk_count_to_FR(spk_count, err_spk_count, f_stim, coef = 1):
    
    ''' scale spike counts in phase hist to FR assuming one oscillation per cycle''' 
    
    scale  = cal_scale_of_spk_count_to_FR(f_stim, coef = coef)
    spk_Hz = spk_count * scale
    spk_Hz_err = err_spk_count * scale
    
    return spk_Hz, spk_Hz_err

def phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, nuc_order = None,
                                  density = False, phase_ref = 'self', total_phase = 360, projection = None,
                                  outer=None, fig=None,  title='',  ylim = None, plot_mode = 'line',
                                  labelsize=15, title_fontsize = 8, lw=1, linelengths=1, include_title=True, ax_label=False,
                                  scale_count_to_FR = False, f_stim = 20, legend_loc = 'upper right',
                                  xlabel_fontsize = 13, ylabel_fontsize = 13, phase_txt_fontsize = 8, tick_label_fontsize = 10, 
                                  plot_FR = False, FR_dict = None, alpha = 0.2, state = 'rest', y_max_series = None,
                                  name_fontsize = 10, name_side = 'right', set_ylim = False, n_decimal = 0, 
                                  name_ylabel_pad = [0,0,0,0,0], name_place = 'ylabel', coef = 1,
                                  n_minor_tick_y = 4, n_minor_tick_x = 4):
    
    n_subplots = len( nuclei_dict.keys())
    fig_generated, outer_generated, fig, outer, inner, nuc_order = create_figs_for_phase_subplots(fig, outer, nuc_order, 
                                                                                                  nuclei_dict,
                                                                                                  figsize = 
                                                                                                  (3, 1.5 * n_subplots))
    # if shift_phase_deg != None:
        
    #     nuclei_dict = shift_phase_hist_all_nuclei(nuclei_dict, shift_phase_deg, phase_ref, total_phase = total_phase)
    for j, nuc_name in enumerate(nuc_order):
        
        ax = plt.Subplot(fig, inner[j])
        fig.add_subplot(ax)
        
        nucleus = nuclei_dict[nuc_name][0]
        
        count = np.average( nucleus.neuron_spike_phase_hist[phase_ref], axis = 0)
        err = stats.sem( nucleus.neuron_spike_phase_hist[phase_ref], axis = 0)
        edges = nucleus.phase_bins
        centers = get_centers_from_edges(edges [:-1])
        count, err = scale_phase_spike_count(count, err, total_phase, len(centers), coef = coef)

        phases = None
        make_phase_plot(count, err, nuc_name, ax, centers, color_dict, phases, 
                        coef, y_max_series,  plot_FR = plot_FR,
                        f_stim = 20, scale_count_to_FR = scale_count_to_FR, lw = 0.5, alpha = 0.15,
                        print_stat_phase = False, box_plot = False, FR_dict = FR_dict,
                        phase_text_x_shift = 150, phase_txt_fontsize = 8, 
                        phase_txt_yshift_coef = 1.4, total_phase = total_phase, state = state)
        
        if projection == 'polar':
            
            circular_bar_plot_as_hist(ax, count, edges, fill = True, alpha = 0.3, density = density,  
                                      facecolor = color_dict[nucleus.name])


        set_ax_prop_phase(ax, nuc_name, color_dict, name_ylabel_pad, name_fontsize, name_place, name_side,
                      y_max_series, total_phase, n_decimal, tick_label_fontsize, n_subplots, j, set_ylim)
    

    set_fig_prop_phase(fig, n_x = n_minor_tick_x,  n_y = n_minor_tick_y , strip_plot = False, xlabel_y = 0.05, ylabel_x = -0.1,
                       xlabel_fontsize = xlabel_fontsize, ylabel_fontsize = ylabel_fontsize,  xlabel = 'phase (deg)', 
                       n_bins = len(nucleus.phase_bins), total_phase = total_phase, coef = coef, 
                       scale_count_to_FR = scale_count_to_FR,
                       title = '', title_fontsize = title_fontsize)
    return fig

def handle_label_pads(name_list, name_ylabel_pad):
    
    if type(name_ylabel_pad) == int:
        name_ylabel_pad = [name_ylabel_pad] * len(name_list)
        
    if len(name_ylabel_pad) < len(name_list):
        raise('unequal labelpad lengths as nuclei')

    return name_ylabel_pad

def print_pop_name(ax, name, color, label_pad, name_fontsize, name_place, name_side):
    
    if name_place == 'in plot':
        
        ax.annotate(name, xy = (0.8,0.05), xycoords='axes fraction', color = color,
                    fontsize = name_fontsize )
        
    elif name_place == 'ylabel':
        if name_side == 'right':
            
            ax.yaxis.set_label_position("right")
        ax.set_ylabel(name, fontsize = name_fontsize, color = color, labelpad = label_pad)
       
def annotate_txt(ax, txt, name, color, name_fontsize, xy = {'STN' : (0.7, 0.8),
                                                             'Proto': (0.7, 0.4),
                                                             'Arky': (0.7, 0.8)}):

        
    ax.annotate(txt, xy = xy[name], xycoords='axes fraction', color = color,
                    fontsize = name_fontsize )

def print_average_phase(ax, phases , text_x_shift,  phase_txt_fontsize, color, y = 0):

    text = r'$' + "{:.1f}". format(np.average(phases)) + \
            ' ^{\circ} \pm ' + "{:.1f}". format(np.std(phases)) + \
            '^{\circ}$'
            
    x = np.average(phases)
    
    # if np.average(phases) > 100:
    #     x = x - text_x_shift

    ax.annotate( text,  xy = (x, y), 
                    color = color, fontsize = phase_txt_fontsize)
        
def remove_non_entrained_hists(data, name, n_g):
    
    hists = data[(name,'rel_phase_hist')][n_g, :, :, :] 
    ind_entrained = np.where ( hists.any (axis = -1) ) 

    entrained_hists = hists[ ind_entrained[0], ind_entrained[1] , :]
    
    print( name, '{} out of {} are entrained'. format
                                  ( len (ind_entrained[0] ), hists.shape[0] * hists.shape[1] ))
    
    return entrained_hists

def check_if_some_not_entrained(data, n_g, name):
    
    hists =  data[(name,'rel_phase_hist')][n_g, :, :, :] 
    ind_non_entrained = np.where ( ~hists.any (axis = -1) ) 

    if len(ind_non_entrained [0]) > 0 :
        
        return True
    
    else:
        
        return False
    
def get_stats_of_phase(data, n_g, name, n_neuron, n_run, centers, shift_phase_deg = None):
        
    # phase_hist_per_neuron = data[(name,'rel_phase_hist')][n_g, :, :, :] * coef  
    some_non_entrained = check_if_some_not_entrained(data, n_g, name) 
    
    if some_non_entrained:
                          
        phase_hist_per_neuron =  remove_non_entrained_hists(data, name, n_g) 
        
    else:
        ### collapse the run and neuron axes
        phase_hist_per_neuron = data[(name,'rel_phase_hist')][n_g, :, :, :].reshape( 
                                                                        int(n_run * n_neuron), -1)  
        
    if shift_phase_deg != None:

        phase_hist_per_neuron = shift_phase_hist(shift_phase_deg, phase_hist_per_neuron, 
                                                 len(centers), total_phase = 720)
    n_neuron = phase_hist_per_neuron.shape[0]
    
    phase_frq_rel_mean = np.average( phase_hist_per_neuron, axis = 0)
    phase_frq_rel_std = np.std( phase_hist_per_neuron, axis = 0)
    # phase_frq_rel_sem = stats.sem( phase_hist_per_neuron, axis = (0,1))
    phase_frq_rel_sem = np.std( phase_hist_per_neuron, axis = 0) / np.sqrt(n_neuron)

    return phase_frq_rel_mean, phase_frq_rel_std, phase_frq_rel_sem, phase_hist_per_neuron

def plot_mean_phase_plus_std(phase_frq_rel_mean, phase_frq_rel_std, name,ax, color_dict, 
                             centers, lw = 1, alpha = 0.1):

    ax.plot(centers, phase_frq_rel_mean, color = color_dict[name], lw = lw)
    ax.fill_between(centers, phase_frq_rel_mean - phase_frq_rel_std, 
                    phase_frq_rel_mean + phase_frq_rel_std, alpha = alpha , color = color_dict[name])
    
def calculate_phase_all_runs(n_neuron, hists, centers, name, phase_ref, 
                             shift_phase = None):
    
    n_neuron = hists.shape [0]
    phases = np.zeros(n_neuron)
    ws = np.zeros( n_neuron)
    
    
    
    for n in range(n_neuron):
        
        phases[n], fitfunc, ws[n] = find_phase_from_sine_and_max(centers, hists[n, :], name, 
                                                                 phase_ref, shift_phase = shift_phase)

    phases = correct_phases(phases, ws, name, phase_ref, shift_phase= shift_phase)
    
    return phases

def smooth_hists(hists, hist_smoothing_wind,
                smooth_hist = True):
    
    if smooth_hist :
        
        hists = moving_average_array_2d(hists, hist_smoothing_wind )
        
    return  hists
    
def plot_phase_histogram_all_nuclei(nuclei_dict, dt, color_dict, low_f, high_f, filter_order = 6, peak_threshold = 1, 
                                density = False, n_bins = 16, start = 0, end = None, phase_ref = 'self',
                                total_phase = 360, projection = None):

    n_plots = len(nuclei_dict)
    fig, axes = plt.subplots(n_plots, 1,  subplot_kw=dict(projection= projection), figsize = (5, 10))
    
    find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = 6, n_bins = n_bins,
                                      peak_threshold = peak_threshold, phase_ref = phase_ref, start = start, 
                                      total_phase = 360, only_PSD_entrained_neurons = False, end = end)

    for count, nuclei_list in enumerate( nuclei_dict.values() ):
        
        for nucleus in nuclei_list:
            
            ax = axes[count] 
            
            frq, edges = nucleus.neuron_spike_phase_hist[phase_ref]
            
            if projection == None:
                
                width = np.diff(edges) 
                ax.bar(edges, frq, width=np.append(width, width[-1]), align = 'edge', facecolor = color_dict[nucleus.name])
                
            elif projection == 'polar':
                
                circular_bar_plot_as_hist(ax, frq, edges, fill = True, alpha = 0.3, density = density,  
                                          facecolor = color_dict[nucleus.name])

            ax.set_title(nucleus.name + ' relative to ' + phase_ref, fontsize = 15, color = color_dict[nucleus.name])
            count += 1
            
    if projection == None:
        
        fig.text(0.6, 0.01, 'Phase (deg)', ha='center',
                            va='center', fontsize=18)
        fig.text(0.03, 0.5, 'Spike count', ha='center',
                            va='center', rotation='vertical', fontsize=18)
        
    fig. tight_layout(pad=1.0)
    
def pairwise(iterable):
    ''' iterate through list pair by pair'''
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    
    a = iter(iterable)
    
    return zip(a, a)

def set_boxplot_prop(bp, color_list, 
                     face_color = 'white',
                     linewidths = {'box': 0.8, 'whiskers': 0.5,
                                   'caps': 0.5, 'median': 2}):
    for color, patch in zip( color_list, bp['boxes']): 
        patch.set(color = face_color, linewidth = linewidths['box']) 
        patch.set_edgecolor(color=color)

    # whiskers
    for color, whiskers in zip(color_list, pairwise(bp['whiskers'])):
        for whisker in list( whiskers) : 
            whisker.set(color = color, 
                        linewidth = linewidths['whiskers'])               

    # caps 
    for color, caps in zip(color_list, pairwise(bp['caps'])):    
        for cap in caps: 
            cap.set(color = color, 
                    linewidth = linewidths['caps']) 
           
    # medians 
    for color, median in zip( color_list, bp['medians']): 
        median.set(color =color, 
                    linewidth =linewidths['median']) 

    return bp

def boxplot_phases(ax, color_dict, phases,name, box_width, box_y, width, 
                   phase_txt_fontsize = 10, phase_txt_yshift_coef = 1.5,
                   text_x_shift = 50, print_stat_phase = True):
    
    
    bp = ax.boxplot(phases, positions = [box_y], vert=False, patch_artist = True,
                    sym = '', widths = box_width, whis =  (0, 100))
    if print_stat_phase:
        
        print_average_phase(ax, phases , text_x_shift,
                            phase_txt_fontsize, color_dict[name], 
                            y = box_y - box_width * phase_txt_yshift_coef )
    
    bp = set_boxplot_prop(bp, [color_dict[name]])
        
def save_pdf_png(fig, figname, size = (8,6)):
    fig.set_size_inches(size, forward=False)
    fig.savefig(figname + '.png', dpi = 500, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    fig.savefig(figname + '.pdf', dpi = 500, #facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
def save_png(fig, figname, size = (8,6)):
    fig.set_size_inches(size, forward=False)
    fig.savefig(figname + '.png', dpi = 500, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
def save_pdf(fig, figname, size = (8,6)):
    fig.set_size_inches(size, forward=False)
    fig.savefig(figname + '.pdf', dpi = 500, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

def set_x_tick_colors(ax, color_list):
    
    [t.set_color(i) for (i,t) in
     zip(color_list,ax.xaxis.get_ticklabels())]
    
    
def shift_x_ticks(ax, shift_to ='right'):
    
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment(shift_to)
        

def PSD_summary(filename, name_list, color_dict, n_g_list, xlim = None, ylim = None,
                inset_props = [0.65, 0.6, 0.3, 0.3],
                inset_yaxis_loc = 'right', inset_name = 'D2', err_plot = 'fill_between', legend_loc = 'upper right',
                plot_lines = False, tick_label_fontsize = 15, legend_font_size = 10, 
                normalize_PSD = True, include_AUC_ratio = False, x_y_label_size = 10,
                ylabel_norm = 'Norm. Power ' , log_scale = 0,  f_decimal = 1,
                ylabel_PSD = 'PSD', f_in_leg = True, axvspan_color = 'grey', vspan = False,
                xlabel = 'Frequency (Hz)', peak_f_sd = False, legend = True, tick_length = 8,
                leg_lw = 2.5, span_beta = True, x_ticks = None,
                y_ticks = None, xlabel_y = -0.05, xaxis_invert =False):
    
    fig = plt.figure()    
    data = load_pickle(filename)
    
    n_run = data[(name_list[0], 'pxx')].shape[1] 
    n_subplots = len(n_g_list)
    
    for i, n_g in enumerate(n_g_list):
        
        max_pxx = 0
        ax = fig.add_subplot(n_subplots, 1, i + 1)
        
        for j, name in enumerate(name_list):
            
            if normalize_PSD:
                data = norm_PSDs(data, n_run, name, i, log_scale =log_scale)
                
            f = data[(name,'f')][0,0].reshape(-1,)
            pxx_mean = np.average( data[(name,'pxx')][n_g,:,:], axis = 0) 
            mean_peak_f =np.average( data[(name,'base_freq')][n_g,:], axis = 0)
            sd_peak_f = np.std( data[(name,'base_freq')][n_g,:], axis = 0)
            pxx_std = np.std( data[(name,'pxx')][n_g,:,:], axis = 0)
            all_std = np.std( data[(name,'pxx')][n_g,:,:].flatten() )
            significance = check_significance_of_PSD_peak(f, pxx_mean,  n_std_thresh = 2, min_f = 0, 
                                                          max_f = 250, n_pts_above_thresh = 3, 
                                                          ax = None, legend = 'PSD', c = 'k', 
                                                          if_plot = False, name = name)
            
            leg_label = name 
            if f_in_leg:
                
                leg_label += (' f= '
                              + r'$' + "{:."
                              + str(f_decimal) + "f}"). format(mean_peak_f)+ '\;Hz$'
            if peak_f_sd:
                leg_label += '$' +' \pm ' + "{:.1f}". format(sd_peak_f) + '$' + ' Hz'
            if err_plot == 'fill_between':

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
                
        if span_beta:
            
            ax.axvspan(12, 30, color='lightgrey', alpha=0.5, zorder=0)
        
        if x_ticks != None:
            ax.set_xticks(x_ticks)
        if ylim != None and y_ticks == None:
            y_ticks = ylim
        y_ticks = y_ticks or [0, int(max_pxx)]
        ax.set_yticks(y_ticks)
        rm_ax_unnecessary_labels_in_subplots(i, n_subplots, ax)
        remove_frame(ax)
        
        ax.tick_params(which = 'major', axis='both', labelsize=tick_label_fontsize, pad=1, length = tick_length)
        ax.tick_params(which = 'minor', axis='both', labelsize=tick_label_fontsize, pad=1, length = tick_length/2)
        ylim = ylim or [-0.01, max_pxx]
        xlim = xlim or (6, 80)
        manage_ax_lims(xlim, xlim , ax, ylim, ylim)
        
        if xaxis_invert:
            ax.invert_xaxis()
        if legend:
            
            leg = ax.legend(fontsize = legend_font_size , loc = legend_loc,  framealpha = 0.1, frameon = False)
            [l.set_linewidth(leg_lw) for l in leg.get_lines()]

    set_minor_locator_all_axes(fig, n = 2, axis = 'both')

    fig.text(0.5, xlabel_y, xlabel, ha='center',
                 va='center', fontsize=x_y_label_size)
    
    if normalize_PSD:
        if log_scale > 0 :
           ylabel_norm + r'$(\times 10^{-' + str(log_scale) + '})$'
        fig.text(-0.05, 0.5, ylabel_norm, ha='center', va='center',
                     rotation='vertical', fontsize=x_y_label_size)
    else:
        fig.text(0.01, 0.5, ylabel_PSD, ha='center', va='center',
                     rotation='vertical', fontsize=x_y_label_size)
    if vspan:
        ax.axvspan(*ax.get_xlim(), alpha=0.2, color=axvspan_color)

    return fig

def manage_ax_lims(input_xlim, default_xlim, ax, input_ylim, default_ylim):
    
    manage_xlim(input_xlim, default_xlim, ax)
    manage_ylim(input_ylim, default_ylim, ax)
    
def manage_xlim(input_xlim, default_xlim, ax):
    
    if input_xlim == None:
        
        ax.set_xlim( default_xlim)
        
    else:
        
        ax.set_xlim(input_xlim)
        
def manage_ylim(input_ylim, default_ylim, ax):
    
    if input_ylim == None:
        
        ax.set_ylim( default_ylim)
        
    else:
        
        ax.set_ylim(input_ylim)
        
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


def print_G_items(G_dict):
    
    print('G = \n')
    for k, values in G_dict.items():
    
        print(k, np.round( values, 2) )

def generate_figures_multi_run(plot_firing, plot_spectrum, plot_raster, plot_phase, n_run, n_iter):
    
    if n_run > 1:  # don't plot all the runs
        plot_spectrum = False
        plot_firing = False
        plot_phase = False
        plot_raster = False
        
    if plot_firing:
        fig_FR = plt.figure()
    else:
        fig_FR = None
        
    if plot_spectrum:
        fig_spec = plt.figure()
    else:
        fig_spec = None
        
    if plot_raster:
        fig_raster = plt.figure()
        outer_raster = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
    else:
        fig_raster = None ; outer_raster = None
        
    if plot_phase:
        fig_phase = plt.figure()
        outer_phase = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
    else:
        fig_phase = None ; outer_phase = None
        
    return fig_FR, fig_raster, fig_spec, fig_phase, outer_raster, outer_phase

def manage_figs_multi_run(fig_FR, fig_spec, fig_raster, fig_phase,
                          plot_firing, plot_spectrum, plot_raster, plot_phase):

    figs = []
    
    if plot_firing:
        fig_FR.set_size_inches((15, 15), forward=False)
        fig_FR.text(0.5, 0.05, 'time (ms)', ha='center', fontsize=18)
        fig_FR.text(0.03, 0.5, 'firing rate (spk/s)',
                 va='center', rotation='vertical', fontsize=18)
        figs.append(fig_FR)
        
    if plot_spectrum:
        fig_spec.set_size_inches((11, 15), forward=False)
        fig_spec.text(0.5, 0.05, 'frequency (Hz)', ha='center', fontsize=18)
        fig_spec.text(0.02, 0.5, 'fft Power', va='center',
                      rotation='vertical', fontsize=18)
        figs.append(fig_spec)

    if plot_raster:
        fig_raster.set_size_inches((11, 15), forward=False)
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
        
    return figs

def synaptic_weight_exploration_SNN(path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, 
                                    t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
                                    peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
                                    lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
                                    plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
                                    include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
                                    reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, 
                                    K_real = None, N_real = None, N = None, K_all = None, 
                                    receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
                                    spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
                                    phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, phase_ref = 'Proto', plot_phase = False,
                                    total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = True,
                                    plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, save_pop_act = True,
                                    state = 'rest', end_phase = None, threshold_by_percentile = 75,
                                    only_rtest_entrained = True):

    if set_seed:
        np.random.seed(1956)

    
    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = get_max_len_dict({k : v['mean'] for k, v in G_dict.items()} )
    G = dict.fromkeys(G_dict.keys(), None)

    data = create_df_for_iteration_SNN(nuclei_dict, G_dict, duration_base, n_iter, n_run, n_phase_bins = n_phase_bins-1, 
                                       len_f_pxx = len_f_pxx, save_pop_act = save_pop_act, find_phase = find_phase, 
                                       divide_beta_band_in_power = divide_beta_band_in_power, iterating_name = 'g')
    

    fig_FR, fig_raster, fig_spec, fig_phase, outer_raster, outer_phase = generate_figures_multi_run(plot_firing, 
                                                                                                    plot_spectrum, 
                                                                                                    plot_raster, 
                                                                                                    plot_phase, 
                                                                                                    n_run, n_iter)
    count = 0
    for i in range(n_iter):
        
        start = timeit.default_timer()
        
        for k, values in G_dict.items():
            G[k] = {}
            G[k]['mean'] = values['mean'][i]
            print(k, np.round( values['mean'][i], 2) )
        
        G = set_G_dist_specs(G, order_mag_sigma = 1)


        if plot_spectrum:
            ax_spec = fig_spec.add_subplot(n_iter, 1, count+1)

        else: ax_spec = None

        for j in range(n_run):
            
            print(' {} from {} runs'.format(j + 1 , n_run))
            

            receiving_class_dict, nuclei_dict = reinit_and_reset_connec_SNN(path, nuclei_dict, N,  N_real, G, noise_amplitude, noise_variance, Act,
                                                                        A_mvt, D_mvt, t_mvt, t_list, dt, K_all, receiving_pop_list, all_FR_list,
                                                                        end_of_nonlinearity, reset_init_dist= reset_init_dist, poisson_prop = poisson_prop,
                                                                        use_saved_FR_ext = use_saved_FR_ext, if_plot = if_plot, n_FR = 20,
                                                                        normalize_G_by_N= True,  set_noise=False, state = state)
                    

            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            
            if save_pop_act:
                data = save_pop_act_into_dataframe(nuclei_dict, duration_base[0], data, i,j)
            if find_phase:

                find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = int(n_phase_bins/2),
                                              peak_threshold = phase_thresh_h, phase_ref = phase_ref, start = start_phase, 
                                              only_rtest_entrained = only_rtest_entrained,
                                               total_phase = 360, troughs = troughs, threshold_by_percentile = threshold_by_percentile)
                data = save_phases_into_dataframe(nuclei_dict, data, i,j, phase_ref)
                
            if plot_raster:
                
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer_raster[i], title = '', fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)


            if plot_phase:
                
                fig_phase = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                                          density = False, phase_ref = phase_ref, total_phase = total_phase, 
                                                          projection = phase_projection, outer=outer_phase[i], fig= fig_phase,  title='', 
                                                          tick_label_fontsize=18, labelsize=15, title_fontsize=15, lw=1, linelengths=1, 
                                                          include_title=True, ax_label=True, nuc_order = nuc_order)
                


                # fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, coef = 1, scale_count_to_FR = True,
                #                                     density = False, phase_ref= phase_ref, total_phase=720, projection=None,
                #                                     outer=None, fig=None,  title='', tick_label_fontsize=18, n_decimal = 0,
                #                                     labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, 
                #                                     ax_label=False, nuc_order = [ 'FSI', 'D2', 'STN', 'Arky', 'Proto'])
                
                data = save_phases_into_dataframe(nuclei_dict, data, i,j, phase_ref)
                

            data, nuclei_dict = find_freq_all_nuclei_and_save(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
                                                              check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=plot_spectrum,
                                                              c_spec=color_dict, spec_figsize=spec_figsize, n_windows=n_windows, fft_method=fft_method, find_beta_band_power=find_beta_band_power,
                                                              include_beta_band_in_legend=include_beta_band_in_legend, divide_beta_band_in_power = divide_beta_band_in_power, 
                                                              half_peak_range = 5, cut_off_freq = 100, check_peak_significance=check_peak_significance, 
                                                              save_pxx = save_pxx, len_f_pxx = len_f_pxx, normalize_spec=normalize_spec, 
                                                              plot_sig_thresh = plot_sig_thresh, plot_peak_sig = plot_peak_sig, min_f = min_f, 
                                                              max_f = max_f, n_std_thresh= n_std_thresh, AUC_ratio_thresh = AUC_ratio_thresh)
            


        if plot_spectrum:

            ax_spec.legend(fontsize=11, loc='upper center',
                           framealpha=0.1, frameon=False)
            ax_spec.set_xlim(spec_lim[0], spec_lim[1])
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax_spec)

        if plot_firing:
            
            ax = fig_FR.add_subplot(n_iter, 1, count+1)
            plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax, '', 
                 include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                 n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, 
                 plot_start = plot_start, plot_end = plot_end, ylim=firing_ylim, include_FR = include_FR)
            
            ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
            ax.set_ylim(firing_ylim)
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)

        count += 1
        stop = timeit.default_timer()
        print(count, "from", int(n_iter), 'gs. t=', round(stop - start, 2))
    
    figs = manage_figs_multi_run(fig_FR, fig_spec, fig_raster, fig_phase,
                                 plot_firing, plot_spectrum, plot_raster, plot_phase)

        
    if save_pkl:
        pickle_obj(data, filepath)
        
    return figs, '', data



def multi_run_transition(
                 path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, 
                 t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
                 peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
                 lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
                 plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
                 include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
                 reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25,  K_real = None, N_real = None, N = None,
                 receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
                 spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
                 phase_thresh_h = 0, filter_order = 6, low_f = 12, high_f = 30, n_phase_bins = 70, start_phase = 0, phase_ref = None, plot_phase = False,
                 total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = True,
                 plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, save_pop_act = False,
                 state_1 = 'rest', state_2 = 'DD_anesth', K_all = None,  state_change_func = None,
                 beta_induc_name_list = ['D2'], amplitude_dict = None , freq_dict = None, induction_method = 'excitation',
                 start_dict = None, end_dict = None, mean_dict = None,  end_phase = None,
                 shift_phase_deg = None, only_rtest_entrained = True, threshold_by_percentile = 75):

    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = get_max_len_dict({k : v['mean'] for k, v in G_dict.items()} )

    data = create_df_for_iteration_SNN(nuclei_dict, G_dict, duration_base, n_iter, n_run, n_phase_bins = n_phase_bins-1, 
                                       len_f_pxx = len_f_pxx, save_pop_act = save_pop_act, find_phase = find_phase, 
                                       divide_beta_band_in_power = divide_beta_band_in_power, iterating_name = 'g')
    count = 0
    G = dict.fromkeys(G_dict.keys(), None)

    for i in range(n_iter):
                
        for k, values in G_dict.items():
            G[k] = {}
            G[k]['mean'] = values['mean'][i]
        
        G = set_G_dist_specs(G, order_mag_sigma = 1)
        for j in range(n_run):
            
            print(' {} from {} runs'.format(j + 1 , n_run))

            nuclei_dict = reset_connections(nuclei_dict, K_all[state_1], N, N_real)
            nuclei_dict =  reset_synaptic_weights_all_nuclei(nuclei_dict, G, N)
            
            receiving_class_dict, nuclei_dict = reinit_and_reset_connec_SNN(
                                                    path, nuclei_dict, N,  N_real, G, noise_amplitude, noise_variance, Act,
                                                    A_mvt, D_mvt, t_mvt, t_list, dt, K_all, receiving_pop_list, all_FR_list,
                                                    end_of_nonlinearity, reset_init_dist= reset_init_dist, poisson_prop = poisson_prop,
                                                    use_saved_FR_ext = use_saved_FR_ext, if_plot = if_plot, n_FR = 20,
                                                    normalize_G_by_N= True,  set_noise=False, state = state_1)
            
            if state_2 == 'induction':
                
                nuclei_dict = induce_beta_to_nuclei( nuclei_dict, beta_induc_name_list, dt, amplitude_dict, freq_dict, 
                                                    start_dict, end_dict, mean_dict, method = induction_method)
            else:
                
                nuclei_dict = change_network_states(G, noise_variance, noise_amplitude,  path, receiving_class_dict, 
                                            receiving_pop_list, t_list, dt, nuclei_dict, Act, state_1, state_2, 
                                            K_all, N, N_real, A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, 
                                            end_of_nonlinearity )
                
            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            nuclei_dict = smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5)

            # fig = plot(nuclei_dict, color_dict, dt, t_list, 
            #            Act[state_1], Act[state_2], plot_start, D_mvt)
            if save_pop_act:
                
                data = save_pop_act_into_dataframe(nuclei_dict, duration_base[0], data, i, j)

            if find_phase:

                nuclei_dict = find_phase_hist_of_spikes_all_nuc(nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = int(n_phase_bins / 2),
                                                                peak_threshold = phase_thresh_h, phase_ref = phase_ref, start = start_phase, 
                                                                end = end_phase, total_phase = 360, troughs = False,
                                                                only_rtest_entrained = only_rtest_entrained,
                                                                threshold_by_percentile =threshold_by_percentile)

                # fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, coef = 1, scale_count_to_FR = True,
                #                                     density = False, phase_ref= phase_ref, total_phase=720, projection=None,
                #                                     outer=None, fig=None,  title='', tick_label_fontsize=18, n_decimal = 0,
                #                                     labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, 
                #                                     ax_label=False, nuc_order = [ 'FSI', 'D2', 'STN', 'Arky', 'Proto'])
                
                data = save_phases_into_dataframe(nuclei_dict, data, i,j, phase_ref)
                

            data, nuclei_dict = find_freq_all_nuclei_and_save(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
                                              check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=plot_spectrum,
                                              c_spec=color_dict, spec_figsize=spec_figsize, n_windows=n_windows, fft_method=fft_method, find_beta_band_power=find_beta_band_power,
                                              include_beta_band_in_legend=include_beta_band_in_legend, divide_beta_band_in_power = divide_beta_band_in_power, 
                                              half_peak_range = 5, cut_off_freq = 100, check_peak_significance=check_peak_significance, 
                                              save_pxx = save_pxx, len_f_pxx = len_f_pxx, normalize_spec=normalize_spec, 
                                              plot_sig_thresh = plot_sig_thresh, plot_peak_sig = plot_peak_sig, min_f = min_f, 
                                              max_f = max_f, n_std_thresh= n_std_thresh, AUC_ratio_thresh = AUC_ratio_thresh)
            


        count += 1
        print(count, "from", int(n_iter), 'gs.')

    if save_pkl:
        pickle_obj(data, filepath)
        
    return  data




def reset_tau_specs_all_nuclei(tau_dict, nuclei_dict, i, dt):
    
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            for k, v in tau_dict.items():
                
                if k[0] == nucleus.name:
                    
                    nucleus.tau_specs[k]['decay'] = { key : [ val[i] ] for key, val in tau_dict[k].items() }
                    
    return nuclei_dict

def reset_T_specs_all_nuclei(T_dict, nuclei_dict, i, dt):
    
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            for k, v in T_dict.items():
                if k[0] == nucleus.name:
                    nucleus.T_specs[k]['decay'] = { key : [ val[i] ] for key, val in T_dict[k].items() }
    return nuclei_dict

def update_tau_dict(tau, tau_dict, i):
    
    ''' update the tau according to the ith element of the existing keys within tau_dict '''
    
    for k, val in tau_dict.items():
        
        tau[k]['decay']['mean'] = np.array([tau_dict[k]['mean'][ i ] ])
        tau[k]['decay']['sd'] = np.array([tau_dict[k]['sd'][ i ] ])
        
    return tau

def create_df_for_iteration_SNN(nuclei_dict, iterating_param_dict, duration_base, n_iter, n_run, n_phase_bins = 180, 
                                len_f_pxx = 200, save_pop_act = False, 
                                find_phase = False, divide_beta_band_in_power = False, iterating_name = 'g'):
    
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
        
            data[(nucleus.name, 'rel_phase_hist')] = np.zeros((n_iter, n_run, nucleus.n, n_phase_bins-1))
            data[(nucleus.name, 'rel_phase_hist_bins')] = np.zeros( n_phase_bins-1)

            data[(nucleus.name, 'rel_phase')] = np.zeros((n_iter, n_run))

        if divide_beta_band_in_power:
            
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run, 2))
        
        else:
            
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
        
        data[(nucleus.name, 'f')] = np.zeros((n_iter, n_run, len_f_pxx))
        data[(nucleus.name, 'pxx')] = np.zeros((n_iter, n_run, len_f_pxx))

    data[iterating_name] = iterating_param_dict
    
    return data



def synaptic_tau_exploration_SNN(path, tau, nuclei_dict, filepath, duration_base, G, tau_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
                                 peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
                                 lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
                                 plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
                                 include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
                                 reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, K_all = None, N_real = None, N = None,
                                 receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
                                 spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
                                 phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, end_phase = None, phase_ref = 'Proto', plot_phase = False,
                                 total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = True,
                                 plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, save_pop_act = False,
                                 threshold_by_percentile = 75):

    if set_seed:
        np.random.seed(1956)

    
    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = get_max_len_dict({k : v['mean'] for k, v in tau_dict.items()} )
    # print("n_inter = ", n_iter)
    data = create_df_for_iteration_SNN(nuclei_dict, tau_dict, duration_base, n_iter, n_run, n_phase_bins = n_phase_bins, 
                                       len_f_pxx = len_f_pxx, save_pop_act = save_pop_act, find_phase = find_phase, 
                                       divide_beta_band_in_power = divide_beta_band_in_power, iterating_name = 'tau')
    count = 0

    fig_FR, fig_raster, fig_spec, fig_phase, outer_raster, outer_phase = generate_figures_multi_run(plot_firing, 
                                                                                                    plot_spectrum, 
                                                                                                    plot_raster, 
                                                                                                    plot_phase, 
                                                                                                    n_run, n_iter)
    G_copy = G.copy()    
    for i in range(n_iter):
        
        start = timeit.default_timer()
        nuclei_dict = reset_tau_specs_all_nuclei(tau_dict, nuclei_dict, i, dt)
        
        if plot_spectrum:
            
            ax_spec = fig_spec.add_subplot(n_iter, 1, count+1)

        else: ax_spec = None

        # title = G_element_as_txt(tau_dict, i, display=display, decimal=decimal) 
        title = ''
        
        if nuclei_dict[ list(nuclei_dict.keys() )[0]] [0].G_heterogeneity:
            
            G.update({k: {'mean': gg['mean'] * (i  + 50) / 50}  for k, gg in G_copy.items()})
            G = set_G_dist_specs(G, sd_to_mean_ratio = 0.5, n_sd_trunc = 2)

        else:
            
            G.update({k: gg * (i  + 50) / 50  for k, gg in G_copy.items()})
            print_G_items(G)
        
        for j in range(n_run):
            
            

            print(' {} from {} runs'.format(j + 1 , n_run))
            receiving_class_dict, nuclei_dict = reinit_and_reset_connec_SNN(path, nuclei_dict, N,  N_real, G, noise_amplitude, noise_variance, Act,
                                                                    A_mvt, D_mvt, t_mvt, t_list, dt, K_all, receiving_pop_list, all_FR_list,
                                                                    end_of_nonlinearity, reset_init_dist= reset_init_dist, poisson_prop = poisson_prop,
                                                                    use_saved_FR_ext = True, if_plot = False, n_FR = n_FR,
                                                                    normalize_G_by_N= True,  set_noise=False, state = state)
 

            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            
            if save_pop_act:
                data = save_pop_act_into_dataframe(nuclei_dict, duration_base[0],data, i,j)
                
            if plot_raster:
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer_raster[i], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)

            if find_phase:

                find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins,
                                              peak_threshold = phase_thresh_h, phase_ref = phase_ref, start = start_phase, 
                                              end_phase = end_phase, total_phase = 360, troughs = troughs,
                                              threshold_by_percentile = threshold_by_percentile)
                save_phases_into_dataframe(nuclei_dict, data, i,j, phase_ref)
                
            if plot_phase:
                fig_phase = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                                          density = False, phase_ref = phase_ref, total_phase = total_phase, 
                                                          projection = phase_projection, outer=outer_phase[i], fig= fig_phase,  title='', 
                                                          tick_label_fontsize=18, labelsize=15, title_fontsize=15, lw=1, linelengths=1, 
                                                          include_title=True, ax_label=True, nuc_order = nuc_order)
                
            data, nuclei_dict = find_freq_all_nuclei_and_save(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
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
            ax = fig_FR.add_subplot(n_iter, 1, count+1)
            plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax, title, include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim, include_FR = include_FR)
            ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
            ax.set_ylim(firing_ylim)
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)

        count += 1
        stop = timeit.default_timer()
        print(count, "from", int(n_iter), 'gs. t=', round(stop - start, 2))

    figs = manage_figs_multi_run(fig_FR, fig_spec, fig_raster, fig_phase,
                                 plot_firing, plot_spectrum, plot_raster, plot_phase)
    
    if save_pkl:
        pickle_obj(data, filepath)
    return figs, title, data

def synaptic_T_exploration_SNN(path, tau,  nuclei_dict, filepath, duration_base, G, T_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
                               peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
                               lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
                               plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
                               include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
                               reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, K_all = None,  N_real = None, N = None,
                               receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
                               spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
                               phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, end_phase = None, phase_ref = 'Proto', plot_phase = False,
                               total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = True,
                               plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, save_pop_act = False,
                               threshold_by_percentile = 75):

    if set_seed:
        np.random.seed(1956)

    
    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = get_max_len_dict({k : v['mean'] for k, v in T_dict.items()} )

    data = create_df_for_iteration_SNN(nuclei_dict, T_dict, duration_base, n_iter, n_run, n_phase_bins = n_phase_bins, 
                                       len_f_pxx = len_f_pxx, save_pop_act = save_pop_act, find_phase = find_phase, 
                                       divide_beta_band_in_power = divide_beta_band_in_power, iterating_name = 'T')
    count = 0

    fig_FR, fig_raster, fig_spec, fig_phase, outer_raster, outer_phase = generate_figures_multi_run(plot_firing, 
                                                                                                    plot_spectrum, 
                                                                                                    plot_raster, 
                                                                                                    plot_phase, 
                                                                                                    n_run, n_iter)
    G_copy = G.copy()  
    
    for i in range(n_iter):

        start = timeit.default_timer()
        nuclei_dict = reset_T_specs_all_nuclei(T_dict, nuclei_dict, i, dt)
        
        if plot_spectrum:
            ax_spec = fig_spec.add_subplot(n_iter, 1, count+1)

        else: ax_spec = None

        # title = G_element_as_txt(tau_dict, i, display=display, decimal=decimal) 
        title = ''
        G.update({k: gg / ((i  + 10) / 10)  for k, gg in G_copy.items()})
        print_G_items(G)
        for j in range(n_run):
            
            print(' {} from {} runs'.format(j + 1 , n_run))
            receiving_class_dict, nuclei_dict = reinit_and_reset_connec_SNN(path, nuclei_dict, N,  N_real, G, noise_amplitude, noise_variance, Act,
                                                                        A_mvt, D_mvt, t_mvt, t_list, dt, K_all, receiving_pop_list, all_FR_list,
                                                                        end_of_nonlinearity, reset_init_dist= reset_init_dist, poisson_prop = poisson_prop,
                                                                        use_saved_FR_ext = use_saved_FR_ext, if_plot = if_plot, n_FR = 20,
                                                                        normalize_G_by_N= True,  set_noise=False, state = state)
        
            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
            
            if save_pop_act:
                data = save_pop_act_into_dataframe(nuclei_dict, duration_base[0],data, i,j)
                
            if plot_raster:
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer_raster[i], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)

            if find_phase:

                find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins,
                                              peak_threshold = phase_thresh_h, phase_ref = phase_ref, start = start_phase, 
                                              end_phase = end_phase, total_phase = 360, troughs = troughs,
                                              threshold_by_percentile = threshold_by_percentile)
                # find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = n_phase_bins, troughs = troughs,
                #                               height = phase_thresh_h, phase_ref = 'self', start = start_phase, total_phase = 360)
                data = save_phases_into_dataframe(nuclei_dict, data, i,j, phase_ref)
                
            if plot_phase:
                fig_phase = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                                          density = False, phase_ref = phase_ref, total_phase = total_phase, 
                                                          projection = phase_projection, outer=outer_phase[i], fig= fig_phase,  title='', 
                                                          tick_label_fontsize=18, labelsize=15, title_fontsize=15, lw=1, linelengths=1, 
                                                          include_title=True, ax_label=True, nuc_order = nuc_order)
                
            data, nuclei_dict = find_freq_all_nuclei_and_save(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
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
            ax = fig_FR.add_subplot(n_iter, 1, count+1)
            plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title, include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim, include_FR = include_FR)
            ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
            ax.set_ylim(firing_ylim)
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)

        count += 1
        stop = timeit.default_timer()
        print(count, "from", int(n_iter), 'gs. t=', round(stop - start, 2))

    figs = manage_figs_multi_run(fig_FR, fig_spec, fig_raster, fig_phase,
                                 plot_firing, plot_spectrum, plot_raster, plot_phase)
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


def synaptic_weight_exploration_SNN_2d(loop_key_lists, path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, Act, A_mvt, t_mvt, D_mvt, 
                                       receiving_class_dict, noise_amplitude, noise_variance,
                                       peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
                                       lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
                                       plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
                                       include_beta_band_in_legend=False, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
                                       reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, K_all = None, N_real = None, N = None,
                                       receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
                                       spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
                                       phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0,  end_phase = None, phase_ref = 'Proto', plot_phase = False,
                                       total_phase = 720, phase_projection = None, troughs = False, nuc_order = None, save_pxx = True, len_f_pxx = 200, normalize_spec = False,
                                       plot_sig_thresh = False, plot_peak_sig = False, min_f = 100, max_f = 300, n_std_thresh= 2, AUC_ratio_thresh = 0.8, 
                                       save_pop_act = False, save_gamma = False, threshold_by_percentile = 75, only_rtest_entrained = True):

    if set_seed:
        np.random.seed(1956)

    
    max_freq = 100; max_n_peaks = int ( t_list[-1] * dt / 1000 * max_freq ) # maximum number of peaks aniticipated for the duration of the simulation
    n_iter = len(G_dict[loop_key_lists[0][0]]['mean'])
    n_iter_2 = len(G_dict[loop_key_lists[1][0]]['mean'])
    print(n_iter, n_iter_2)
    data = create_df_for_iteration_2d(nuclei_dict, G_dict, duration_base, n_iter, n_iter_2, n_run, 
                                          n_phase_bins = n_phase_bins, 
                                       len_f_pxx = len_f_pxx, save_pop_act = save_pop_act, find_phase = find_phase, 
                                       divide_beta_band_in_power = divide_beta_band_in_power, iterating_name = 'g',
                                       check_peak_significance=check_peak_significance, save_gamma = save_gamma)
    count = 0
    G = dict.fromkeys(G_dict.keys(), None)

    if n_run > 1:  # don't plot all the runs
        plot_spectrum = False
        plot_firing = False
        plot_phase = False
        plot_raster = False
        
    if plot_firing:
        fig_FR = plt.figure()

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
            G[k] = {}
            G[k]['mean'] = G_dict[k]['mean'][i]



        
        for m in range(n_iter_2):
            
            # title =G_element_as_txt(G, i, display=display, decimal=decimal) 
            title = ''
            
            if plot_spectrum:
                
                ax_spec = fig_spec.add_subplot(n_iter, n_iter_2, count+1)
                
            else: ax_spec = None

            for k in loop_key_lists[1]:
                
                G[k] = {}
                G[k]['mean'] = G_dict[k]['mean'][m]                 
            G = set_G_dist_specs(G, order_mag_sigma = 1)
            for j in range(n_run):
                
                print(' {} from {} runs'.format(j + 1 , n_run))
    
                nuclei_dict = reset_connections(nuclei_dict, K_all[state], N, N_real)
                nuclei_dict =  reset_synaptic_weights_all_nuclei(nuclei_dict, G, N)
                
                receiving_class_dict, nuclei_dict = reinit_and_reset_connec_SNN(
                                                        path, nuclei_dict, N,  N_real, G, noise_amplitude, noise_variance, Act,
                                                        A_mvt, D_mvt, t_mvt, t_list, dt, K_all, receiving_pop_list, all_FR_list,
                                                        end_of_nonlinearity, reset_init_dist= reset_init_dist, poisson_prop = poisson_prop,
                                                        use_saved_FR_ext = use_saved_FR_ext, if_plot = if_plot, n_FR = 20,
                                                        normalize_G_by_N= True,  set_noise=False, state = state)
                nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
                
                if save_pop_act:
                    data = save_pop_act_into_dataframe_2d(nuclei_dict, duration_base[0], data, i, m, j)
                
                if plot_raster:
                    fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer[i,m], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                        plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)
    
                if find_phase:
    
                    nuclei_dict = find_phase_hist_of_spikes_all_nuc( nuclei_dict, dt, low_f, high_f, filter_order = filter_order, n_bins = int(n_phase_bins / 2),
                                                      peak_threshold = phase_thresh_h, phase_ref = phase_ref, start = start_phase, 
                                                      end =  end_phase, total_phase = 360, troughs = False,
                                                      threshold_by_percentile = threshold_by_percentile,
                                                      only_rtest_entrained = only_rtest_entrained)
                    data = save_phases_into_dataframe_2d(nuclei_dict, data, i, m, j, phase_ref)


                    # fig = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, coef = 1, scale_count_to_FR = True,
                    #                                     density = False, phase_ref= phase_ref, total_phase=720, projection=None,
                    #                                     outer=None, fig=None,  title='', tick_label_fontsize=18, n_decimal = 0,
                    #                                     labelsize=15, title_fontsize=15, lw=1, linelengths=1, include_title=True, 
                    #                                     ax_label=False, nuc_order = [ 'FSI', 'D2', 'STN', 'Arky', 'Proto'])
                    
                    data = save_phases_into_dataframe(nuclei_dict, data, i,j, phase_ref)
                    
                if plot_phase:
                    fig_phase = phase_plot_all_nuclei_in_grid(nuclei_dict, color_dict, dt, 
                                                              density = False, phase_ref = phase_ref, total_phase = total_phase, 
                                                              projection = phase_projection, outer=outer_phase[i,m], fig= fig_phase,  title='', 
                                                              tick_label_fontsize=18, labelsize=15, title_fontsize=15, lw=1, linelengths=1, 
                                                              include_title=True, ax_label=True, nuc_order = nuc_order)
                    
                data, nuclei_dict = find_freq_all_nuclei_and_save(data, (i, m, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
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
                ax = fig_FR.add_subplot(n_iter, n_iter_2, count+1)
                plot(nuclei_dict, color_dict, dt, t_list, Act[state], A_mvt, t_mvt, D_mvt, ax, title, include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                    n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim, include_FR = include_FR)
                ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
                ax.set_ylim(firing_ylim)
                rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax)
    
            count += 1
            stop = timeit.default_timer()
            print(count, "from", int(n_iter * n_iter_2), 'gs. t=', round(stop - start, 2))
    if save_pkl:
        
        pickle_obj(data, filepath)
            
    if plot_firing and plot_spectrum and plot_raster:
        figs = manage_figs_multi_run(fig_FR, fig_spec, fig_raster, fig_phase,
                                     plot_firing, plot_spectrum, plot_raster, plot_phase)

        return figs, title, data
    else:
        return None, None, data

def Coherence_single_pop_exploration_SNN(noise_dict, tau, path, nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
    peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
    lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
    plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
    include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_FR = False, include_std=True, round_dec=2, legend_loc='upper right', display='normal', decimal=0,
    reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, N_real = None, N = None,
    receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False,
    spec_lim = [0, 55],  half_peak_range = 5, n_std = 2, cut_off_freq = 100, check_peak_significance = False, find_phase = False,
    phase_thresh_h = 0, filter_order = 6, low_f = 10, high_f = 30, n_phase_bins = 70, start_phase = 0, phase_ref = 'Proto', plot_phase = False,
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


    fig_FR, fig_raster, fig_spec, fig_phase, outer_raster, outer_phase = generate_figures_multi_run(plot_firing, 
                                                                                                    plot_spectrum, 
                                                                                                    plot_raster, 
                                                                                                    plot_phase, 
                                                                                                    n_run, n_iter)
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
            nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, N, G, noise_amplitude, noise_variance, A,
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
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer_raster[i], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)
 
            data, nuclei_dict = find_freq_all_nuclei_and_save(data, (i, j), dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
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
            ax = fig_FR.add_subplot(n_iter, 1, count+1)
            title = ( r'$ \langle I_{0} \rangle = $' + str( np.round(data[phase_ref, 'I_ext_0'], 1))  + '   mV   ' + 
                      r'$\langle \eta \rangle = $' + str( np.round(data[phase_ref, 'mean_noise'][i,j], 1))) + '  mV'
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

    figs = manage_figs_multi_run(fig_FR, fig_spec, fig_raster, fig_phase,
                                 plot_firing, plot_spectrum, plot_raster, plot_phase)
    if save_pkl:
        pickle_obj(data, filepath)
    return figs, title, data


def coherence_exploration(path, tau, nuclei_dict, G_dict, noise_amplitude, noise_variance, Act, 
                          N, N_real, K_real, K_all, receiving_pop_list,
                          A_mvt, D_mvt, t_mvt, t_list, dt,  all_FR_list , n_FR, receiving_class_dict,
                          end_of_nonlinearity, color_dict, 
                          poisson_prop, reset_init_dist = True, sampling_t_distance_ms = 1,
                          if_plot = False, state= 'rest'):
    
    n_g = get_max_len_dict(G_dict)
    coherence_dict = {name: np.zeros(n_g) for name in list(nuclei_dict.keys())}
    
    
    for i in range(n_g):
        G = {}
        for k, values in G_dict.items():
            G[k] = values[i]
            print(k, values[i])
            
        receiving_class_dict, nuclei_dict = reinit_and_reset_connec_SNN(path, nuclei_dict, N,  N_real, G, noise_amplitude, noise_variance, Act,
                                                                        A_mvt, D_mvt, t_mvt, t_list, dt, K_all, receiving_pop_list, all_FR_list,
                                                                        end_of_nonlinearity, reset_init_dist= reset_init_dist, poisson_prop = poisson_prop,
                                                                        use_saved_FR_ext = use_saved_FR_ext, if_plot = if_plot, n_FR = 20,
                                                                        normalize_G_by_N= True,  set_noise=False, state = state)
        

        nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
        coherence_dict = set_coherence_all_nuclei_1d(nuclei_dict, coherence_dict, i, dt, 
                                                  sampling_t_distance_ms = sampling_t_distance_ms)
        
        if if_plot:
            fig = plot(nuclei_dict,color_dict, dt,  t_list, Act[state], A_mvt, t_mvt, D_mvt, ax = None, 
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
    '''Fit sine to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''

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
            
def find_freq_all_nuclei_and_save(data, element_ind,  dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, 
                                  cut_plateau_epsilon, check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, 
                                  plot_spectrum=False, ax=None, c_spec='navy', spec_figsize=(6, 5), find_beta_band_power=False,
                                  fft_method='welch', n_windows=3, include_beta_band_in_legend=True, divide_beta_band_in_power = False, 
                                  half_peak_range = 5, cut_off_freq = 100, check_peak_significance = False, len_f_pxx = 200, 
                                  save_pxx = True, normalize_spec = False, plot_sig_thresh = False, plot_peak_sig = False, smooth = True,
                                  min_f = 0, max_f = 300, n_std_thresh = 2, AUC_ratio_thresh = 0.2, save_gamma = False,
                                  print_AUC_ratio = False):

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
                                                                                                            peak_threshold = peak_threshold,
                                                                                                            name = nucleus.name,
                                                                                                            n_std_thresh = n_std_thresh, 
                                                                                                            min_f = min_f, 
                                                                                                            max_f = max_f, 
                                                                                                            n_pts_above_thresh = 3,
                                                                                                            if_plot = plot_peak_sig,
                                                                                                            AUC_ratio_thresh = AUC_ratio_thresh,
                                                                                                            print_AUC_ratio = print_AUC_ratio)
                    
                                  
                

            nucleus.frequency_basal = data[(nucleus.name, 'base_freq')][element_ind]

            # print(nucleus.name, 'f = ', round(data[(nucleus.name, 'base_freq')][element_ind], 2), 'beta_p =', data[(
                # nucleus.name, 'base_beta_power')][element_ind], np.sum(data[(
                # nucleus.name, 'base_beta_power')][element_ind]))

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


def find_freq_all_nuclei(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold,
                             smooth_kern_window, smooth_window_ms,
                             cut_plateau_epsilon, check_stability, freq_method, plot_sig,
                             low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=False, 
                             ax=None, c_spec='navy', spec_figsize=(6, 5), find_beta_band_power=False,
                             fft_method='rfft', n_windows=3, include_beta_band_in_legend=True, smooth = False, 
                             normalize_spec = True, include_peak_f_in_legend = True,
                             check_significance = False, plot_sig_thresh = False, plot_peak_sig = False,
                             min_f = 100, max_f = 300, n_std_thresh = 2,AUC_ratio_thresh = 0.8, print_AUC_ratio = False):
    pxx = {}
    if plot_spectrum:
        ax = ax or plt.subplots()[1]
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            
            if smooth:
                
                nucleus.smooth_pop_activity(dt, window_ms=smooth_window_ms)

            if low_pass_filter:
                nucleus.butter_bandpass_filter_pop_act(dt, lower_freq_cut, upper_freq_cut, order=6)

            (n_half_cycles, perc_oscil, freq, 
             if_stable, beta_band_power, f, pxx[nucleus.name]) = nucleus.find_freq_of_pop_act_spec_window(*duration_base, dt,
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
                                                          peak_threshold = peak_threshold,
                                                          n_std_thresh = n_std_thresh, 
                                                          min_f = min_f, 
                                                          max_f = max_f, 
                                                          n_pts_above_thresh = n_std_thresh,
                                                          if_plot = plot_peak_sig, 
                                                          name = nucleus.name,
                                                          AUC_ratio_thresh = AUC_ratio_thresh,
                                                          print_AUC_ratio = print_AUC_ratio )


    return freq, f, pxx


    
def rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax, axis = 'both'):
    
    if axis == 'both':
    
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    elif axis == 'x':
        
        ax.set_xlabel("")
        
    elif axis == 'y':
        
        ax.set_ylabel("")
        
    if count+1 < n_iter:
        # remove the x tick labels except for the bottom plot
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
            nucleus.plot_mem_potential_distribution_of_all_t( ax = ax, 
                                                             color=color_dict[nucleus.name], 
                                                             bins=100)
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


def truncated_normal_distributed(mean, sigma, size, scale_bound=scale_bound_with_mean, 
                                 scale=None, lower_bound_perc=0.8, upper_bound_perc=1.2, 
                                 truncmin=None, truncmax=None):
    if sigma == 0:
    
        return np.full(size, mean)
    
    elif truncmin != None and truncmax != None:
        
        lower_bound, upper_bound = truncmin, truncmax
        
    else:
        
        lower_bound, upper_bound = scale_bound(mean, lower_bound_perc, 
                                               upper_bound_perc, scale=scale)

    return stats.truncnorm.rvs((lower_bound-mean)/sigma, 
                               (upper_bound-mean)/sigma, 
                               loc=mean, scale=sigma, size=size)


def truncated_lognormal_distributed(mu, sigma, size, scale_bound=scale_bound_with_mean, 
                                 scale=None, lower_bound_perc=0.8, upper_bound_perc=1.2, 
                                 truncmin=None, truncmax=None, mu_norm = None, sigma_norm = None):
    
    """draw samples from a truncated lognoraml distribution"""

    if mu == 0: # if meant to be disconnected bypass the log normal estimation
        return np.zeros(size)
    else:
            
        sigma_norm = sigma_norm or np.sqrt(np.log(1 + (sigma / mu) ** 2))
        mu_norm = mu_norm or np.log( mu ** 2 / np.sqrt(mu ** 2 + sigma ** 2))
        norm_sample = truncated_normal_distributed(mu_norm, sigma_norm, size, scale_bound=scale_bound_with_mean, 
                                                   scale=scale, lower_bound_perc = lower_bound_perc, upper_bound_perc = lower_bound_perc, 
                                                   truncmin = np.log(truncmin), truncmax = np.log(truncmax))
        
        lognorm_sample = np.exp(norm_sample)
        
        return lognorm_sample


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
                    int(nucleus.population_num)-1] = FR_mean
                firing_prop[nucleus.name]['firing_var'][i,
                    int(nucleus.population_num)-1] = FR_std
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
            np.save(os.path.join(nucleus.path, 'all_mem_pot_' + nucleus.name + '_tau_' + str(np.round(
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


def extrapolated_FR_ext_from_fitted_curve(FR_ext, FR, desired_FR, coefs, estimating_func, 
                                          inverse_estimating_func, FR_normalizing_factor, x_shift):

    return inverse_estimating_func(desired_FR / FR_normalizing_factor, *coefs) + x_shift


def find_y_normalizing_factor(y, desired_FR, epsilon=0.2):
    
    y_max = np.max(y)
    
    if desired_FR >= y_max:  # if the maximum of the curve is the same as the desired FR add epsilon to it to avoid errors in log
        
        print('Oooops! max_sigmoid < desired_FR')
        return (desired_FR + epsilon)
    
    else:
        
        return y_max



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
    return np.array([np.where(matrix[i, int(start) : int(end)] == 1)[0] 
                     for i in range(n_rows)], 
                    dtype=object)  \
            + int(start)

def get_corr_key_to_val(mydict, value):
    
	""" return all the keys corresponding to the specified value"""
    
	return [k for k, v in mydict.items() if v == value]


def get_axes(ax, figsize=(6, 5)):
    
    if ax == None:
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    return plt.gcf(), ax


def plot_exper_FR_distribution(xls, name_list, state_list, color_dict, bins = 'auto', 
                               alpha = 0.2, hatch = '/', zorder = 1, edgecolor = None,
                               annotate_fontsize = 14, nbins = 4, ticklabel_fontsize = 12,
                               title_fontsize = 18):
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
                        label = 'De La Crompe (2020)',  alpha = alpha, 
                        hatch = hatch, edgecolor = edgecolor,   
                        zorder = zorder, lw = 2)
                plt.rcParams['hatch.linewidth'] = 3
                
                ax.annotate( r'$ FR = {0} \pm {1} \; Hz$'.format( round (np.average( FR) , 2) , round( np.std( FR), 2) ),
                            xy=(0.5,0.6),xycoords='axes fraction', color = color_dict[name],
                            fontsize= annotate_fontsize, alpha = alpha)
                print(name, state,
                      ' mean = ', np.round( np.average(FR), 2), 
                      ' std = ', np.round( np.std(FR), 2),
                      ' n = ', len(FR))
            except ValueError:
                pass
        ax.set_title(state + ' ' + name, fontsize =15)
        ax.set_xlabel('Firing Rate (spk/s)', fontsize=15)
        ax.set_ylabel('% of population', fontsize=15)
        ax.locator_params(axis='y', nbins=nbins)
        ax.locator_params(axis='x', nbins=nbins)
        ax.tick_params(axis='both', labelsize=ticklabel_fontsize)
        # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
        ax.legend(fontsize=15,  framealpha = 0.1, frameon = False, loc = 'upper right')
        remove_frame(ax)
    return figs

def plot_FR_distribution(nuclei_dict, dt, color_dict, bins = 50, ax = None, zorder = 1, 
                         alpha = 0.2, start = 0, log_hist = False, box_plot = False, 
                         n_pts = 50, only_non_zero = False, legend_fontsize = 15, 
                         label_fontsize = 18, ticklabel_fontsize = 12,
                         annotate_fontsize = 14, nbins = 4, title_fontsize = 18, state = 'rest'):
    
    ''' plot the firing rate distribution of neurons of different populations '''
    
    fig, ax = get_axes(ax)
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            
            FR_mean_neurons_all = np.average(nucleus.spikes[:,start:], axis = 1) / (dt/1000)
                                                                                                          
            if only_non_zero:
                FR_mean_neurons = FR_mean_neurons_all[ FR_mean_neurons_all > 0]
                print('percentage of spontaneously active = ', 
                      np.round( len(FR_mean_neurons) / nucleus.n * 100 , 2), 
                      ' %')
                
                print(r'non-silent only FR = {0} ±  {1}  Hz'.format( round (np.average( FR_mean_neurons) , 2) , 
                                                                    round( np.std(FR_mean_neurons) , 2) ))
            else: FR_mean_neurons = FR_mean_neurons_all
            FR_std_neurons = np.std(FR_mean_neurons) 
            freq, edges = np.histogram(FR_mean_neurons, bins = bins)
            width = np.diff(edges[:-1])
            ax.annotate( r'$ FR = {0} \pm {1}\; Hz$'.format( round (np.average( FR_mean_neurons) , 2) , 
                                                             round( FR_std_neurons, 2) ),
                        xy=(0.5,0.5),xycoords='axes fraction', color = color_dict[nucleus.name],
             fontsize= annotate_fontsize, alpha = alpha)
            
            print(r' FR = {0} ±  {1}  Hz'.format( round (np.average( FR_mean_neurons_all) , 2) , 
                                                  round(np.std(FR_mean_neurons_all) , 2) ))
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
                ax.scatter(xs, FR_mean_neurons[:n_pts], c= color_dict[nucleus.name], alpha=0.4, s = 10, ec = 'k', zorder = 1)

            else:
                ax.bar( edges[:-1], freq / nucleus.n * 100,  width=np.append(width, width[-1]), align = 'edge', facecolor = color_dict[nucleus.name],
                       label='Simulation',  alpha =alpha, zorder = zorder)
                
    if log_hist and not box_plot:
        ax.set_xscale("log")
        
    ax.set_title(state + ' ' + nucleus.name, fontsize = title_fontsize)
    ax.set_xlabel('Firing Rate (spk/s)', fontsize=label_fontsize)
    ax.set_ylabel('% of population', fontsize= label_fontsize)
    # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
    ax.legend(fontsize=legend_fontsize,  framealpha = 0.1, frameon = False, loc = 'upper right')
    ax.locator_params(axis='y', nbins=nbins)
    ax.locator_params(axis='x', nbins=nbins)
    ax.tick_params(axis='both', labelsize=ticklabel_fontsize)
    remove_frame(ax)
    
    if only_non_zero:
        ax.set_title(' Only spontaneously active' , fontsize=15)
        
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
    
    ''' plot the spike amplitude distribution of neurons of different populations '''
    
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

def raster_plot(spikes_sparse, name, color_dict, color='k',  ax=None, tick_label_fontsize=5, title_fontsize=15, linelengths=2.5, lw=3,
                axvspan=False, span_start=None, span_end=None, axvspan_color='lightskyblue', orientation = 'horizontal',
                xlim=None, include_nuc_name = True, y_tick_length = 2, x_tick_length = 5, remove_ax_frame = False, 
                remove_whole_ax_frame = False, y_ticks = None, axis_linewidth = 0.5):
    
    fig, ax = get_axes(ax)

    ax.eventplot(spikes_sparse, colors=color_dict[name],
                 linelengths=linelengths, lw=lw, orientation= orientation)

    if include_nuc_name:
        ax.set_title(name, c=color_dict[name], fontsize=title_fontsize)
        
    if axvspan:
        ax.axvspan(span_start, span_end, alpha=0.2, color=axvspan_color)

    if xlim != None:
        ax.set_xlim(xlim)
    if y_ticks == None:
        ax.set_yticks([])
    else:
        set_y_ticks_one_ax(ax, y_ticks)

    n_neuron = int(len(spikes_sparse))
    ax.legend(loc='upper right', framealpha=0.1, frameon=False)
    ax.tick_params(which = 'both', axis='x', length = x_tick_length, labelsize=tick_label_fontsize)
    ax.tick_params(which = 'both', axis='y', length = y_tick_length, labelsize=tick_label_fontsize)
    set_axis_thickness(ax, linewidth  = axis_linewidth)
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from matplotlib.transforms import Bbox

    scalebar = AnchoredSizeBar(ax.transData,
                               100, '100 ms', 'lower right', 
                               pad=0.1,
                               color='k',
                               frameon=False,
                               size_vertical=0.1, 
                               fontproperties = matplotlib.font_manager.FontProperties(size = 10))
    ax.add_artist(scalebar)

    ax.set_xticks([])
    if remove_ax_frame:
        remove_frame(ax)
    if remove_whole_ax_frame:
        remove_whole_frame(ax)
    
    return ax

def raster_plot_Brice(spike_times, corr_laser_sweep_inds, color, ax = None):

    fig, ax = get_axes(ax)
    spikes_sparse = np.array([ spike_times[l: r]
                           for l, r in zip( *iterate_with_step(corr_laser_sweep_inds, step = 1)) ], 
                             dtype=object)
    ax.eventplot(spikes_sparse, colors = color,
                 linelengths=2, lw=2, orientation= 'horizontal')
    
    
def rm_ax_unnecessary_labels_in_fig(fig):
    
    n = len(fig.axes)
    for j, ax in enumerate(fig.axes):
        rm_ax_unnecessary_labels_in_subplots(j, n, ax)
        
def raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=0, plot_end=None, tick_label_fontsize=18,
                            labelsize=15, title_fontsize=15, lw=1, linelengths=1, n_neuron=None, include_title=True, set_xlim=True,
                            axvspan=False, span_start=None, span_end=None, axvspan_color='lightskyblue', ax_label=False, neurons =[],
                            ylabel_x = 0.03, include_nuc_name = True, name_list = None, remove_ax_frame = True, remove_whole_ax_frame = False,
                            tick_length = 10, axis_linewidth = 0.5, t_shift = 0,  x_tick_length = 5, y_tick_length = 2):
    
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
                                            plot_end / dt), start=(plot_start / dt)) * dt - t_shift
        if set_xlim:
            xlim = [plot_start - t_shift, plot_end - t_shift]
            
        else: 
            xlim = None
            
        if span_end != None:
            span_end_n = span_end - t_shift
            
        else: span_end_n = span_end
        
        if span_start != None:
            span_start_n = span_start - t_shift
            
        else:  span_start_n = span_start
        
        c_dict = color_dict.copy()
        c_dict['Arky'] = 'darkorange'
        ax = raster_plot(spikes_sparse, nucleus.name, c_dict,  ax=ax, tick_label_fontsize=tick_label_fontsize, 
                         title_fontsize=title_fontsize, linelengths=linelengths, lw=lw, xlim=xlim,
                        axvspan=axvspan, span_start = span_start_n, span_end = span_end_n, axvspan_color=axvspan_color,
                        include_nuc_name = include_nuc_name, remove_ax_frame = remove_ax_frame, y_ticks = [n_neuron],
                         x_tick_length = x_tick_length, y_tick_length = y_tick_length, remove_whole_ax_frame = remove_whole_ax_frame)
        
        fig.add_subplot(ax)
        rm_ax_unnecessary_labels_in_subplots(j, len(nuclei_dict), ax)

    set_minor_locator_all_axes(fig, n = 2, axis = 'x')

    
    if ax_label:
        fig.text(0.5, 0.03, 'time (ms)', ha='center',
                 va='center', fontsize=labelsize)
        fig.text(ylabel_x, 0.5, 'neuron', ha='center', va='center',
                 rotation='vertical', fontsize=labelsize)
    return fig

def list_files_of_interest_in_path(path, extensions = ['csv']):
    
    '''get all the files with extention in the path where you want to search'''
    
    files = [x for x in os.listdir(path) if not x.startswith('.')]
    files.sort()
    files_of_interest = [ os.path.join(path, fi) for fi in files if fi.endswith( tuple(extensions)) ]
    
    return files_of_interest

def raster_plot_all_nuclei_transition(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='',  plot_=None, tick_label_fontsize=18,
                            labelsize=15, title_fontsize=15, lw=1, linelengths=1, n_neuron=None, include_title=True, set_xlim=True,
                            axvspan=False, span_start=None, span_end=None, axvspan_color='lightskyblue', ax_label=False, n = 1000, 
                            t_transition = None, t_sim = None, ylabel_x = 0.03, include_nuc_name = True,
                            plot_start_state_1=0, plot_end_state_1=0, plot_start_state_2=0, plot_end_state_2=0):
    
    neurons = np.random.choice(n, n_neuron, replace=False )
    
    fig_state_1 = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start_state_1, plot_end = plot_end_state_1,
                                         labelsize = labelsize, title_fontsize = title_fontsize, lw  = lw, linelengths = linelengths, n_neuron = n_neuron, include_title = include_title, set_xlim=set_xlim,
                                         neurons = neurons, ax_label = True, tick_label_fontsize = tick_label_fontsize, ylabel_x = ylabel_x, include_nuc_name=include_nuc_name,
                                         t_shift = plot_start_state_1)
    
    fig_state_2 = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer = None, fig = None,  title = '', plot_start = plot_start_state_2, plot_end = plot_end_state_2,
                            labelsize = labelsize, title_fontsize = title_fontsize, lw  = lw, linelengths = linelengths, n_neuron = n_neuron, include_title = include_title, set_xlim=set_xlim,
                            axvspan = True, span_start = plot_start_state_2, span_end = plot_end_state_2 , axvspan_color = axvspan_color, include_nuc_name=include_nuc_name,
                            neurons = neurons, ax_label = True, tick_label_fontsize = tick_label_fontsize, ylabel_x = ylabel_x,
                            t_shift = plot_start_state_2)
    
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
                firing_prop[nucleus.name]['firing_mean'][i, int(nucleus.population_num) -
                    1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                firing_prop[nucleus.name]['firing_var'][i, int(nucleus.population_num) -
                    1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                print(nucleus.name, np.round(np.average(nucleus.FR_ext), 3),
                    'FR=', firing_prop[nucleus.name]['firing_mean'][i, int(nucleus.population_num)-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, int(nucleus.population_num)-1], 2))
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

                    firing_prop[nucleus.name]['firing_mean'][i, j, int(nucleus.population_num) -
                        1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                    loss[i, j] += (firing_prop[nucleus.name]['firing_mean']
                                   [i, j, int(nucleus.population_num)-1] - nucleus.basal_firing)**2
                    firing_prop[nucleus.name]['firing_var'][i, j, int(nucleus.population_num) -
                        1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                    print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop, 3),
                        'FR=', firing_prop[nucleus.name]['firing_mean'][i, j, int(nucleus.population_num)-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, j, int(nucleus.population_num)-1], 2))
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
                        firing_prop[nucleus.name]['firing_mean'][i, j, l, int(nucleus.population_num) -
                            1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                        loss[i, j, l] += (firing_prop[nucleus.name]['firing_mean']
                                          [i, j, l, int(nucleus.population_num)-1] - nucleus.basal_firing)**2
                        firing_prop[nucleus.name]['firing_var'][i, j, l,
                            int(nucleus.population_num)-1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                        print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop, 4),
                            'FR=', firing_prop[nucleus.name]['firing_mean'][i, j, l, int(nucleus.population_num)-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, j, l, int(nucleus.population_num)-1], 2))
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
                firing_prop[nucleus.name]['firing_mean'][i, int(nucleus.population_num) -
				    1] = np.average(nucleus.pop_act[int(len(t_list)/2):])
                loss[i] += (firing_prop[nucleus.name]['firing_mean']
				            [i, int(nucleus.population_num)-1] - nucleus.basal_firing)**2
                firing_prop[nucleus.name]['firing_var'][i, int(nucleus.population_num) -
				    1] = np.std(nucleus.pop_act[int(len(t_list)/2):])
                print(tuning_param, nucleus.name, round(nucleus.firing_of_ext_pop, 3),
					'FR=', firing_prop[nucleus.name]['firing_mean'][i, int(nucleus.population_num)-1], 'std=', round(firing_prop[nucleus.name]['firing_var'][i, int(nucleus.population_num)-1], 2))
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

def pad_high_res_spacing_with_linspace(start_before, mid_start, n_before, mid_end, end_after,  n_after, n_high_res, base = 1.1):
    
    linspace_before = np.linspace(start_before, mid_start, n_before)
    linspace_after = np.linspace(mid_end, end_after, n_after)
    high_res = spacing_with_high_resolution_in_the_middle(n_high_res, mid_start, mid_end, base = base).reshape(-1,)
    
    return np.concatenate((linspace_before, high_res, linspace_after), axis  = 0)

def pad_high_res_spacing_with_arange(start_before, mid_start, bin_before, mid_end, end_after,  bin_after, n_high_res, base = 1.1):
    
    linspace_before = np.linspace(start_before, mid_start, int( ( mid_start - start_before) / bin_before) )
    linspace_after = np.linspace(mid_end, end_after, int( ( end_after - mid_end) / bin_after) )
    high_res = spacing_with_high_resolution_in_the_middle(n_high_res, mid_start, mid_end, base = base).reshape(-1,)
    
    return np.concatenate((linspace_before, high_res, linspace_after), axis  = 0)

def three_different_linspace_arrays(start_before, mid_start, n_before, mid_end, end_after,  n_after, n_mid):
    
    linspace_before = np.linspace(start_before, mid_start, n_before)
    linspace_after = np.linspace(mid_end, end_after, n_after)
    linspace_mid = np.linspace(mid_start, mid_end, n_mid)
    
    return np.concatenate((linspace_before, linspace_mid, linspace_after), axis  = 0)

def spacing_with_high_resolution_in_the_middle(n_points, start, end, base = 1.1):
    '''return a series with lower spacing and higher resolution in the middle'''    
    R = (end - start) / 2
    x = R * np.linspace(-1, 0, int(n_points/2))
    # y = np.sqrt(R ** 2 - x ** 2)
    # fig, ax = plt.subplots()
    # ax.plot(y, np.ones_like(y), 'o')    
# 	if len(series) < n_points: # doesn't work with odd number of points!!!!!!
# 		series = np.concatenate((series, series[-1]))

    y = 1- np.logspace(0, -10 , int( n_points /2) , base = base, endpoint = True) 
    y = y / y[-1] * R # scale because of base
    diff = - np.diff(np.flip(y))
    series = np.concatenate((y, np.cumsum(diff) + y[-1])) + start
    
    # ax.plot(series, np.ones_like(series) + 1, 'o')
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
    ylabel = 'PSD' + r'$(V^{2}/Hz)$'
    
    if include_peak_f_in_legend :    
	    label += ' f =' + str(round(peak_freq, 1)) + ' Hz'
        
    if include_beta_band_in_legend:
        beta_band_power = beta_bandpower(f, pxx)
        label += ' ' + r'$\overline{P}_{\beta}=$' + str(round(beta_band_power, 3))
        
    if normalize :
		
        pxx = norm_PSD(pxx, f) * 100
        ylabel = '{\tiny{Norm}. PSD} ' + r'$(V^{2}/Hz \times 10^{-2})$'
        
    
    ax.plot(f, pxx, c=c, label=label, lw=1.5)
    ax.set_xlabel('frequency (Hz)', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend(fontsize=15, loc=legend_loc, framealpha=0.1, frameon=False)
	# ax.tick_params(axis='both', which='major', labelsize=10)
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='x', nbins=5)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.set_xlim(0, 80)
    remove_frame(ax)
    
    if plot_sig_thresh:
        
        sig_thresh = cal_sig_thresh_1d(f, pxx, min_f = min_f, max_f = max_f, n_std_thresh = n_std_thresh)
        ax.axhline(sig_thresh,0, max_f, ls = '--', color = c)

def norm_PSD(pxx, f):
    
    ''' Normalize PSD to the AUC '''
    AUC = np.trapz(pxx, f)
    pxx = pxx / AUC 
    return pxx

def norm_PSD_2d(pxx, f, axis = 0):

    ''' Normalize PSD to the AUC '''
    AUC = np.trapz(pxx, f, axis = axis)
    pxx = pxx / AUC 
    
    return pxx

def norm_PSDs(data, n_run, name, n_g, log_scale = 0):
    
    for run in range(n_run):
        # AUC = np.trapz( data[(name,'pxx')][n_g, run,:])
        data[(name,'pxx')][n_g, run,:] = data[(name,'pxx')][n_g, run,:]/ np.var(data[(name,'pop_act')][n_g, run,:]) * 10** log_scale
        
    return data

def freq_from_fft(sig, dt, plot_spectrum=False, ax=None, c='navy', label='fft', figsize=(6, 5), 
                  method='rfft', n_windows=6, include_beta_band_in_legend=False, max_f = 200, min_f = 0,
                  normalize_spec = True, include_peak_f_in_legend = True, plot_sig_thresh = False, 
                  legend_loc = 'upper right',  n_std_thresh = 2):
    """
    Estimate frequency from peak of FFT
    """
    N = len(sig)

    if N == 0:
        return 0 , 0, 0, 0
    else:
        if method not in ["rfft", "Welch"]:
            raise ValueError("method must be either 'rff', or 'Welch'")            
        if method == 'rfft':
            f, pxx, peak_freq = freq_from_rfft(sig, dt,  N)
            
        if method == 'Welch':
            f, pxx, peak_freq = freq_from_welch(sig, dt, n_windows=n_windows)

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


def freq_from_welch(sig, dt, n_windows=6, detrend  = False, window = 'hann'):# detrend = 'constant'):
    
    """	Estimate frequency with Welch method
    """
    fs = 1 / dt
    sig = signal.detrend(sig)
    f, pxx = signal.welch(sig, axis=0, fs=fs, nperseg=int(len(sig) / n_windows), 
                       window = window, detrend = detrend)
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
    signif_thresh =  cal_sig_thresh_2d(f, pxx, min_f = min_f_sig_thres, max_f = max_f, n_std_thresh = n_sd_thresh)
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


def check_significance_of_PSD_peak(f, pxx,  n_std_thresh = 2, min_f = 0, max_f = 250, n_pts_above_thresh = 2, 
                                   ax = None, legend = 'PSD', c = 'k', if_plot = True, AUC_ratio_thresh = 0.2,
                                   xlim = [0, 80], name = '', print_AUC_ratio = False, peak_threshold  = 0.7, f_cut = 1):
    
    ''' Check significance of a peak in PSD by thresholding the ratio of AUC above n_std_thresh
        relative to the absolute AUC, as well as thresholding the peak of the power spec
    '''

    f, pxx = cut_PSD_1d(f, pxx, max_f = max_f)
    f, pxx = f[f > f_cut], pxx[f > f_cut]
    pxx_norm = norm_PSD(pxx, f)
    peaks, _ = scipy.signal.find_peaks(pxx_norm, height = peak_threshold)
    
    n_peaks = len(peaks)
    signif_thresh = cal_sig_thresh_1d(f, pxx, min_f = min_f, max_f = max_f, n_std_thresh = n_std_thresh)
    # above_thresh_ind = np.where(pxx >= signif_thresh)[0]
    
    pxx_rel_sig = pxx - signif_thresh
    # fig, ax = plt.subplots(1,1)
    # ax.plot(f, pxx_rel_sig.clip(min  = 0), '-o', label = legend, c = 'b')
    AUC_above_sig_thresh = np.trapz(pxx_rel_sig.clip(min  = 0), f)
    AUC = np.trapz(pxx, f)
    AUC_ratio = AUC_above_sig_thresh / AUC
    
    # longest_seq_abv_thresh = longest_consecutive_chain_of_numbers( above_thresh_ind)
    # if_plot = True
    if if_plot:
        
        fig, ax = get_axes (ax)
        ax.plot(f, pxx_norm, '-o', label = legend, c = c)
        # ax.axhline(signif_thresh, min_f, max_f, ls = '--', color = c)
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_title(name)
    
    if print_AUC_ratio:
        print(name,
              "AUC ratio = {}".format( AUC_ratio ), 
              " n peaks = ", n_peaks)
        # if n_peaks > 0:
        #     print("max peak = ", max(pxx_norm[peaks]))
            
    if AUC_ratio > AUC_ratio_thresh and n_peaks > 0:

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

                ext_inp = np.ones((nucleus.n, 1)) * \
				                      nucleus.external_inp_t_series[t]  # movement added

                nucleus.calculate_input_and_inst_act(t, dt, receiving_class_dict[(nucleus.name, str(k + 1))], ext_inp)
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
			# nucleus.reset_synaptic_weights(G_DD)
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

def change_synaptic_weight( nuclei_dict, name, projection_name, multiply_by = 1):

    if nuclei_dict[name][0].G_heterogeneity:
        
        nuclei_dict[name][0].synaptic_weight_specs[(name, projection_name)]['mean'] *= multiply_by  
        
    else:
        
        nuclei_dict[name][0].synaptic_weight_specs[(name, projection_name)] *= multiply_by 
        
    nuclei_dict[name][0].create_syn_weight_mean_dict()
    
    return nuclei_dict
    
def change_network_states(G, noise_variance,noise_amplitude,  path, receiving_class_dict, 
                  receiving_pop_list, t_list, dt, nuclei_dict, Act, state_1, state_2, 
                  K_all, N, N_real, A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, 
                  end_of_nonlinearity ):
    
    print('transitioning..')
    nuclei_dict = change_basal_firing_all_nuclei(Act[state_2], nuclei_dict)
    nuclei_dict = change_state_all_nuclei(state_2, nuclei_dict)
    nuclei_dict = change_noise_all_nuclei( nuclei_dict, noise_variance[state_2], noise_amplitude)
    
    if 'DD' in state_2 :
        
        if ('Proto', 'Proto') in list(G.keys()): 
            
            nuclei_dict = change_synaptic_weight( nuclei_dict, 'Proto', 'Proto', multiply_by = 2)

            
        if ('STN', 'Proto') in list(G.keys()): 
        
            nuclei_dict = change_synaptic_weight( nuclei_dict, 'STN', 'Proto', multiply_by = 2)
            
        # if ('D2', 'Arky') in list(G.keys()): 
        
        #     nuclei_dict = change_synaptic_weight( nuclei_dict, 'D2', 'Arky', multiply_by = 2.25)
                
        # if ('Proto', 'STN') in list(G.keys()): 
        
        #     nuclei_dict = change_synaptic_weight( nuclei_dict, 'Proto', 'STN', multiply_by = 1.35)
            
    
    receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state_2], A_mvt, D_mvt, t_mvt, dt, N, N_real, 
                                                           K_all[state_2], receiving_pop_list, nuclei_dict, t_list,
                                                           all_FR_list=all_FR_list, n_FR=n_FR, if_plot=False, 
                                                           end_of_nonlinearity=end_of_nonlinearity,
                                                           set_FR_range_from_theory=False, method='collective',  
                                                           save_FR_ext = False, use_saved_FR_ext = True, 
                                                           normalize_G_by_N = False, state=state_2)
    
    return nuclei_dict

def print_syn_weights(nuclei_dict):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            print(nucleus.synaptic_weight_specs , '\n')
      
def print_syn_weights_in_connectivity_mat(nuclei_dict):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            for proj in nucleus.receiving_from_pop_name_list:
                print( proj, 
                      np.mean( nucleus.connectivity_matrix [proj, '1'][
                          np.nonzero( nucleus.connectivity_matrix [proj, '1'] )] ) , '\n')
            
def run_transition_state_collective_setting(G, noise_variance, noise_amplitude, path, receiving_class_dict, 
                                            receiving_pop_list, t_list, dt, nuclei_dict, Act, state_1, state_2, 
                                            K_all, N, N_real, A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, 
                                            end_of_nonlinearity, t_transition=None):
    
    ''' Transition from <state_1> to <state_2> collective FR_ext setting'''
    
    if t_transition == None:
        t_transition = t_list[int(len(t_list) / 3)]
        
    nuclei_dict = run(receiving_class_dict, t_list[:t_transition], dt, nuclei_dict)

    nuclei_dict = change_network_states(G, noise_variance,noise_amplitude,  path, receiving_class_dict, 
                                receiving_pop_list, t_list, dt, nuclei_dict, Act, state_1, state_2, 
                                K_all, N, N_real, A_mvt, D_mvt, t_mvt, all_FR_list, n_FR, 
                                end_of_nonlinearity )

    nuclei_dict = run(receiving_class_dict, t_list[ t_transition: ], dt, nuclei_dict)
    nuclei_dict = cal_population_activity_all_nuc_all_t(nuclei_dict, dt)

    return nuclei_dict

def induce_beta_to_nuclei( nuclei_dict, name_list, dt, amplitude_dict, 
                           freq_dict, start_dict, end_dict, mean_dict, method = 'excitation' ):
    
    for name in name_list:
        nuclei_dict[name][0].beta_stim = True,
        nuclei_dict[name][0].beta_stim_method = method
        nuclei_dict[name][0].add_beta_ext_input(amplitude_dict[name], dt, freq = freq_dict[name], 
                                                start = start_dict[name], end = end_dict[name], 
                                                mean = mean_dict[name], method = method)
    return nuclei_dict

def run_transition_to_beta_induction( receiving_class_dict,  name_list, dt, amplitude_dict, freq_dict, 
                                     start_dict, end_dict, mean_dict,
                                     t_list, nuclei_dict, t_transition=None, 
                                     method = 'excitation'):
    
    ''' Transition from <state_1> to <state_2> collective FR_ext setting'''
    
    if t_transition == None:
        t_transition = t_list[int(len(t_list) / 3)]
        
    nuclei_dict = run(receiving_class_dict, t_list[:t_transition], dt, nuclei_dict)
    nuclei_dict = induce_beta_to_nuclei( nuclei_dict, name_list, dt, amplitude_dict, freq_dict, 
                                        start_dict, end_dict, mean_dict, method = method)
    nuclei_dict = run(receiving_class_dict, t_list[ t_transition: ], dt, nuclei_dict)
    nuclei_dict = cal_population_activity_all_nuc_all_t(nuclei_dict, dt)

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
    return fig, ax, title

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

def exp_rise_and_decay_f(t_series, tau_rise, tau_decay, dt):
    
    f = (1- np.exp(- t_series / (tau_rise / dt))) * \
            (np.exp(- t_series/ (tau_decay / dt))) 
    return f / np.trapz(f, t_series)
            
def exp_rise_and_decay_transient_ext_inp_ChR2_like( trans_coef, mean_ext_inp, t_list, t_start_all_n, 
                                                   tau_rise, tau_decay, dt, n):
    
    ''' return an n by n_timebin matrix as the additive external input filled with 
        a normalized exponential rise and decay with  AUC equal to trans_coef * mean_ext_input
        timing of input is the same for all neurons mimicking ChR2 laser activation population wide
    '''
    t_start = t_start_all_n[0] # cause activation onset is homogeneous among neurons 
    
    t_list_trimmed = t_list[ t_start: ]

    f = exp_rise_and_decay_f(t_list_trimmed - t_start, tau_rise, tau_decay, dt) 
        
    ext_inp =  (trans_coef * mean_ext_inp * f).reshape(-1, 1).T
    ext_inp_at_onset = np.repeat( ext_inp, n, axis = 0)
    
    return  np.pad(ext_inp_at_onset, ( (0,0), (t_start,0)), constant_values=0)

def step_like_norm_dist_transient_ext_inp_ChR2_like(trans_coef, mean_ext_inp, sigma, n, t_list, 
                                                    t_start_all_n, t_end_all_n, homogeneous = False):
    
    ''' return an n by n_timebin matrix as the additive external input filled with 
        normally distributed (for neurons) step like external inputs.
        Note that timing of input is the same for all neurons mimicking ChR2 laser activation population wide
    '''
    t_start = t_start_all_n[0] # cause activation onset is homogeneous among neurons 
    t_end = t_end_all_n[0]
    
    if not homogeneous:
        
        f = np.random.normal(trans_coef * mean_ext_inp, sigma, n ).reshape(-1, 1)
        ext_inp = np.zeros(( n, len (t_list) ))
        
        ext_inp[:, t_start : t_start + (t_end - t_start) ] = np.repeat(f, t_end - t_start, axis = 1)
    
    else:
        
        ext_inp = np.hstack( ( np.zeros( t_start ),  
                                       np.full( t_end - t_start, trans_coef * mean_ext_inp ), 
                                       np.zeros(  len(t_list[t_start:]) - ( t_end - t_start ))
                                    ) ).reshape(-1, )
    return ext_inp

def step_like_norm_dist_transient_ext_inp_projected(trans_coef, mean_ext_inp, sigma, n, t_list, t_start, t_end):
    
    ''' return an n by n_timebin matrix as the additive external input filled with 
        normally distributed (for neurons) step like external inputs
        timing of input is based on transmission delay distribution
    '''

    f = np.random.normal(trans_coef * mean_ext_inp, sigma, n ).reshape(-1, 1)

    ext_inp = np.array( [ np.hstack( ( np.zeros( t_start[i] ),  
                                       np.full( t_end[i] - t_start[i], f[i] ), 
                                       np.zeros(  len(t_list[t_start[i]:]) - ( t_end[i] - t_start[i] ))
                                    ) ) 
                         for i in range(n)])

    return ext_inp

def exp_rise_and_decay_transient_ext_inp_projected( trans_coef, mean_ext_inp, t_list, t_start, tau_rise, tau_decay, dt, n):
    
    ''' return an n by n_timebin matrix as the additive external input filled with 
        a normalized exponential rise and decay with  AUC equal to trans_coef * mean_ext_input.
        Note that timing of input is based on transmission delay distribution
    '''
    
    
    ext_inp = np.array( [ np.hstack( ( np.zeros( t_start[i] ),  
                                       trans_coef * mean_ext_inp * \
                                       exp_rise_and_decay_f(t_list[ t_start[i]: ] - t_start[i], tau_rise, tau_decay, dt)
                                   ) ) 
                     for i in range(n)])

    return  ext_inp




def selective_additive_ext_input_time_series(nuclei_dict, t_list,  ext_inp_dict,
                                             t_start_inp_dict, t_end_inp_dict, dt, duration = 10, 
                                             plot = False, method = 'exponential', stim_method = 'ChR2',
                                             homogeneous = False):
    
    ''' filling the values of the extrernal_inp_t_series with a transient input according to when the stimulus starts
        which is the same for all neurons.
    '''
    
    list_of_nuc_with_trans_inp = list(t_start_inp_dict.keys())

    for name in list_of_nuc_with_trans_inp:
        
        nucleus = nuclei_dict[name][0]

        t_start = t_start_inp_dict[name]
        t_end = t_end_inp_dict[name]
        
        if method == 'exponential':
            if stim_method == 'ChR2':

                f_add = exp_rise_and_decay_transient_ext_inp_ChR2_like(ext_inp_dict[name]['mean'],
                                                                       np.average(nucleus.rest_ext_input),
                                                                       t_list, t_start, ext_inp_dict[name]['tau_rise'], 
                                                                       ext_inp_dict[name]['tau_decay'], 
                                                                       dt, nucleus.n)
            else:
                
                f_add = exp_rise_and_decay_transient_ext_inp_projected(ext_inp_dict[name]['mean'],
                                                                       np.average(nucleus.rest_ext_input),
                                                                       t_list, t_start, ext_inp_dict[name]['tau_rise'], 
                                                                       ext_inp_dict[name]['tau_decay'], 
                                                                       dt, nucleus.n)
        elif method == 'step':
            if stim_method == 'ChR2':
            
                f_add = step_like_norm_dist_transient_ext_inp_ChR2_like( ext_inp_dict[nucleus.name]['mean'],
                                                                        np.average( nucleus.rest_ext_input), 
                                                                        ext_inp_dict[nucleus.name]['sigma'], 
                                                                        nucleus.n, t_list, t_start, t_end,
                                                                        homogeneous = homogeneous)
            else:
                
                f_add = step_like_norm_dist_transient_ext_inp_projected( ext_inp_dict[nucleus.name]['mean'],
                                                                        np.average( nucleus.rest_ext_input), 
                                                                        ext_inp_dict[nucleus.name]['sigma'], 
                                                                        nucleus.n, t_list, t_start, t_end)

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

def filter_transmission_delay_for_downstream_projection(T, list_of_receiving_nuclei, projecting = 'Ctx'):

    syn_trans_delay_dict = {
        k[0]: v  for k, v in T.items() if k[0] in list_of_receiving_nuclei and k[1] == projecting}

    return syn_trans_delay_dict

def syn_trans_delay_heterogeneous( syn_trans_delay_dict, dt, n ):
    
    ''' Return a normally distributed array of size n
        for each transmission delay specs given in <syn_trans_delay_dict> 
    '''
    syn_trans_delay_dict = {key: (truncated_normal_distributed(v['mean'],
                                                                v['sd'], n,
                                                                truncmin = v['truncmin'],
                                                                truncmax = v['truncmax']) / dt ).astype(int)
                                for key, v in syn_trans_delay_dict.items()}
    return syn_trans_delay_dict

def syn_trans_delay_homogeneous( syn_trans_delay_dict, dt, n ):
    
    ''' 
        Return a uniformly distributed array of size n
        for each transmission delay specs given in <syn_trans_delay_dict> 
    '''
    syn_trans_delay_dict = {key: (np.full(n, v['mean']) / dt ).astype(int)
                                for key, v in syn_trans_delay_dict.items()}
    return syn_trans_delay_dict

def run_with_trans_ext_inp_with_axonal_delay_collective(receiving_class_dict, t_list, dt, nuclei_dict,
                                                        A, syn_trans_delay_dict,
                                                        t_transient=10, duration=10, ext_inp_dict = None, 
                                                        plot = False, ext_inp_method = 'exponential',
                                                        stim_method = 'ChR2', homogeneous = False):
    
    '''
    		run normaly til "t_transient" then exert an external transient input ( as an exponential rise and decay) 
            to the concerned nuclei then resume to normal state until the end of simulation.
    		Where the syn_trans_delay_dict contains the synaptic transmission delays of the input to 
            different nuclei (e.g. MC to STN and MC to D2)
    '''
    
    t_start_inp_dict, t_end_inp_dict = create_stim_start_end_dict(t_transient, duration, syn_trans_delay_dict)
    nuclei_dict = selective_additive_ext_input_time_series(nuclei_dict, t_list, ext_inp_dict,
                                                           t_start_inp_dict, t_end_inp_dict, dt,  
                                                           duration=10, plot = plot, method = ext_inp_method,
                                                           stim_method = stim_method,
                                                           homogeneous= homogeneous)
    
    nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
    
    return nuclei_dict

def plot_extermums_FR(nuclei_dict, peak_jiggle_dict, t_transient, dt, color_dict, fig,  
                      alpha = 1, smooth_kern_window = 10, plot_gaus_smoothed = True):
    
    for nuclei_list in nuclei_dict.values():
        for k, nucleus in enumerate(nuclei_list):
            
            peak_threshold =  peak_jiggle_dict[nucleus.name] * nucleus.basal_firing
            
            pop_act = gaussian_filter1d(nucleus.pop_act[ int(t_transient /dt): ] - 
                                        np.average(nucleus.pop_act[int(t_transient /dt):]), smooth_kern_window)
            
            troughs,_ = signal.find_peaks(-pop_act, height =peak_threshold )
            peaks,_ = signal.find_peaks(pop_act, height = peak_threshold)
            
            if plot_gaus_smoothed:
                fig.gca().plot(np.arange(len(nucleus.pop_act[int(t_transient /dt):])) * dt , pop_act, 
                               '-o', markersize = 2, c = color_dict[nucleus.name], alpha = alpha)
            
            fig.gca().plot(peaks * dt , nucleus.pop_act[peaks + int(t_transient /dt)], 
                           'x', markersize = 10, c = color_dict[nucleus.name], alpha = alpha)
            fig.gca().plot(troughs * dt , nucleus.pop_act[troughs + int(t_transient /dt)], 
                           'x', markersize = 10, c = color_dict[nucleus.name], alpha = alpha)
            print( nucleus.name, 'peak thesh = ', peak_threshold, ' peaks at :', peaks * dt)
            print( nucleus.name, ' troughs at :', troughs * dt)
            
    return fig

def reinit_and_reset_connec_SNN(path, nuclei_dict, N,  N_real, G, noise_amplitude, noise_variance, Act,
                                A_mvt, D_mvt, t_mvt, t_list, dt, K_all, receiving_pop_list, all_FR_list,
                                end_of_nonlinearity, reset_init_dist= True, poisson_prop = None,
                                use_saved_FR_ext = True, if_plot = False, n_FR = 20,
                                normalize_G_by_N= True,  set_noise=False, state = 'rest'):
    
    nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, N, G, noise_amplitude, noise_variance[state], Act[state],
                                          A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                          reset_init_dist= reset_init_dist, poisson_prop = poisson_prop, 
                                          normalize_G_by_N= True)  

 
    if reset_init_dist:
        receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, 
                                                               N_real, K_all[state], receiving_pop_list, 
                                                               nuclei_dict, t_list, all_FR_list = all_FR_list , 
                                                               n_FR =n_FR, if_plot = if_plot, 
                                                               end_of_nonlinearity = end_of_nonlinearity, 
                                                               set_FR_range_from_theory = False, method = 'collective', 
                                                               use_saved_FR_ext= use_saved_FR_ext,
                                                               normalize_G_by_N= False, save_FR_ext=False,
                                                               state = state)
    return receiving_class_dict, nuclei_dict

def average_multi_run_collective(path,  receiving_pop_list, receiving_class_dict, t_list, dt, 
                                 nuclei_dict, Act, G, N, N_real, K_real, K_all, syn_trans_delay_dict, poisson_prop,  
                                 n_FR, all_FR_list, end_of_nonlinearity, t_transient = 10, duration = 10, 
                                 n_run = 1, A_mvt = None, D_mvt = None, t_mvt = None, ext_inp_dict = None, 
                                 noise_amplitude = None, noise_variance = None, reset_init_dist = True, 
                                 color_dict = None, state = 'rest', plot = False, ext_inp_method = 'exponential',
                                 stim_method = 'ChR2', homogeneous = False):
    
    avg_act = {nuc: np.zeros( ( len(t_list), len(nuclei_dict[nuc]) ) ) 
               for nuc in list( nuclei_dict.keys() ) }
    
    for i in range(n_run):

        nuclei_dict = run_with_trans_ext_inp_with_axonal_delay_collective(receiving_class_dict, 
                                                                            t_list, dt, nuclei_dict,
                                                                            Act[state], syn_trans_delay_dict,
                                                                            t_transient = t_transient, 
                                                                            duration = duration, 
                                                                            ext_inp_dict = ext_inp_dict,
                                                                            plot = plot, 
                                                                            ext_inp_method = ext_inp_method,
                                                                            stim_method = stim_method,
                                                                            homogeneous= homogeneous)
        avg_act = cal_average_activity(nuclei_dict, n_run, avg_act)
        
        
        receiving_class_dict, nuclei_dict = reinit_and_reset_connec_SNN(path, nuclei_dict, N,  N_real, G, noise_amplitude, noise_variance, Act,
                                                                    A_mvt, D_mvt, t_mvt, t_list, dt, K_all, receiving_pop_list, all_FR_list,
                                                                    end_of_nonlinearity, reset_init_dist= reset_init_dist, poisson_prop = poisson_prop,
                                                                    use_saved_FR_ext = True, if_plot = plot, n_FR = 20,
                                                                    normalize_G_by_N= True,  set_noise=False, state = state)

        print(i + 1, 'from', n_run)
        
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

def calculate_number_of_connections(N_sim, N_real, number_of_connection):
    
    '''calculate number of connections in the scaled network.'''
    
    KK = number_of_connection.copy()
    
    for k, v in number_of_connection.items():
        KK[k] = round(1 / ( 1 / v - 
                          1 / N_real[ k[1] ] + 
                          1 / N_sim[ k[1] ] 
                         ) 
                    )
        
    return KK

def transfer_func(Threshold, gain, x):
    
    ''' a transfer function that grows linearly for positive values 
    of input higher than the threshold'''
    
    return gain* np.maximum(np.zeros_like(x), (x - Threshold))
    


def build_connection_matrix(n_receiving, n_projecting, n_connections, same_pop = False):
    
    ''' 
        return a matrix with Jij=0 or 1. 1 showing a projection from neuron j in 
        projection population to neuron i in receiving
        Arguments:
                same_pop: optional bool (default = False)
                    if the neuron type of pre and post are the same this value shows if they 
                    are in the same population as to avoid connecting a neuron to itself    
    '''
    connection_prob = np.random.rand(n_receiving, n_projecting)
    
    if same_pop: # if connecting the population to itself, avoid autapses
        np.fill_diagonal(connection_prob, 0)
        
    projection_list = np.argsort(connection_prob, axis = 1)[::-1][:, :n_connections]
    
    rows = ((np.ones((n_connections, n_receiving)) * np.arange(n_receiving)).T).flatten().astype(int)
    cols = projection_list.flatten().astype(int)
    
    ### not sparse
    JJ = np.zeros( (n_receiving, n_projecting), dtype = int)
    JJ[(rows, cols)] = np.ones_like(rows)
    
    #### sparse
    # data = np.ones_like(rows)
    # JJ = sp.csc_matrix((data, (rows, cols)), shape = (n_receiving, n_projecting))
    
    return JJ

def get_start_end_plot(plot_start, plot_end, dt, t_list):
    
    if plot_end == None : 
        
        plot_end = int( len(t_list) )
    
    else:
    
        plot_end = int(plot_end / dt)
        
    plot_start = int( plot_start / dt)
    
    return plot_start, plot_end

def plot( nuclei_dict, color_dict,  dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title = "", n_subplots = 1, 
         title_fontsize = 12, plot_start = 0, ylabelpad = 0, include_FR = True, alpha_mvt = 0.2, plot_end = None, 
         figsize = (6,5), plt_txt = 'vertical', plt_mvt = True, plt_freq = False, ylim = None, include_std = True, 
         round_dec = 2, legend_loc = 'upper right', continuous_firing_base_lines = True, axvspan_color = 'lightskyblue', 
         tick_label_fontsize = 18, plot_filtered = False, low_f = 8, high_f = 30, filter_order = 6, vspan = False, 
         tick_length = 8, title_pad = None, ncol_legend = 1, line_type = ['-', '--'], alpha = 1, legend = True,
         xlim = None, lw = 1.5, legend_fontsize = 14, label_fontsize = 12, threshold_by_percentile = 75,
         xlabel = "time (ms)", peak_threshold = None, leg_lw = 2, y_ticks = None):    

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
                

                peaks, act = nucleus.find_peaks_of_pop_act( dt, low_f, high_f, filter_order = filter_order, 
                                                           peak_threshold = peak_threshold, start = 0, end = None, 
                                                          threshold_by_percentile = threshold_by_percentile)
                peaks_base = peaks[(peaks > plot_start) & (peaks < int(t_mvt /dt))]
                peaks_mvt = peaks[(peaks > int(t_mvt /dt)) & (peaks < plot_end)]
                
                ax.plot(t_list[peaks_base] * dt, act[peaks_base] + A[nucleus.name], 'x', color = color_dict[nucleus.name])
                ax.plot(t_list[peaks_mvt] * dt, act[peaks_mvt] + A_mvt[nucleus.name], 'x', color = color_dict[nucleus.name])
                ax.plot(t_list[plot_start: int(t_mvt /dt)]*dt, act[plot_start: int(t_mvt /dt)] + A[nucleus.name], line_type[int(nucleus.population_num)-1], 
                        c = color_dict[nucleus.name],lw = 1.5, alpha = 0.4)
                ax.plot(t_list[int(t_mvt /dt): plot_end] * dt, act[int(t_mvt /dt): plot_end] + A_mvt[nucleus.name], line_type[int(nucleus.population_num)-1], 
                        c = color_dict[nucleus.name],lw = 1.5, alpha = 0.4)
            
            else:
                
                ax.plot(t_list[plot_start: plot_end] * dt, nucleus.pop_act[plot_start: plot_end], 
                        line_type[int(nucleus.population_num)-1], label = label, c = color_dict[nucleus.name],
                        lw = lw, alpha = alpha)
                
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


    ax.set_title(title, fontsize = title_fontsize, pad = title_pad)
    ax.set_xlabel(xlabel, fontsize = label_fontsize)
    ax.set_ylabel("firing rate (spk/s)", fontsize = label_fontsize,labelpad=ylabelpad)
    ax_label_adjust(ax, fontsize = tick_label_fontsize, nbins = 5)   
    set_minor_locator(ax, n = 4, axis = 'y')
    ax.tick_params(which = 'major', axis='both', length = tick_length)
    ax.tick_params(which = 'minor', axis='both', length = tick_length * .6)
    if y_ticks != None:
        ax.set_yticks(y_ticks, fontsize = tick_label_fontsize)

    remove_frame(ax)
    
    if vspan:
        ax.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color=axvspan_color)
        
    if legend:
        leg = ax.legend(fontsize = legend_fontsize, loc = legend_loc, framealpha = 0.1, frameon = False, ncol=ncol_legend)
        [l.set_linewidth(leg_lw) for l in leg.get_lines()]
    if ylim != None:
        ax.set_ylim(ylim)
        
    if xlim != None:
        ax.set_xlim(xlim)
        
    else:
        ax.set_xlim(plot_start * dt - 10, plot_end * dt + 10) 
    
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
    
    zero_crossings = np.where(
                        np.diff(
                                np.sign(sig)
                                )
                             )[0] # indices to elements before a crossing
    
    shifted = np.roll(zero_crossings, -1)
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

def trim_start_end_sig_rm_offset(sig, start, end, cut_plateau_epsilon = 0.1, method = 'neat'):
    
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

        if max_point > len(trimmed)/2: # in case the amplitude increases over time take the whole signal #############################3 to do --> check for increasing amplitude in oscillations instead of naive implementation here
            
            return trimmed- np.average(trimmed)
        
        else:
        
            return(trimmed[max_point:] - np.average(trimmed[max_point:]))
        
# def find_freq_of_pop_act_spec_window(nucleus, start, end, dt, peak_threshold = 0.1, smooth_kern_window= 3 , 
#                                      check_stability = False, cut_plateau_epsilon = 0.001, plot_oscil = False):
    
#     ''' RATE MODEL : 
#         trim the beginning and end of the population activity of the nucleus 
#         if necessary, cut the plateau and in case it is oscillation determine the frequency '''
    
#     sig = trim_start_end_sig_rm_offset(nucleus.pop_act, start, end)
#     cut_sig_ind = cut_plateau( sig, epsilon= cut_plateau_epsilon)
#     plateau_y = find_mean_of_signal(sig, cut_sig_ind)

#     if_stable = False
    
#     if len(cut_sig_ind) > 0: # if it's not all plateau from the beginning
    
#         sig = sig - plateau_y

#         n_half_cycles, freq = zero_crossing_freq_detect(sig[cut_sig_ind], dt / 1000)
        
#         if freq != 0: # then check if there's oscillations
        
#             perc_oscil = max_non_empty_array(cut_sig_ind)/ len(sig) * 100
            
#             if check_stability:
                
#                 if_stable, last_first_peak_ratio = if_stable_oscillatory(sig, max(cut_sig_ind), peak_threshold, start, dt,
#                                                                          smooth_kern_window, amp_env_slope_thresh = - 0.05,
#                                                                          plot = plot_oscil)
            
#             return n_half_cycles, last_first_peak_ratio , perc_oscil, freq, if_stable
        
#         else:
            
#             return 0, 0, 0, 0,  False
#     else:
        
#         return 0, 0, 0, 0, False

def if_stable_oscillatory(sig, peak_threshold, dt = 0.1, smooth_kern_window = 5, amp_env_slope_thresh = - 0.05, 
                          oscil_perc_as_stable = 0.9, last_first_peak_ratio_thresh = [0.995,1.1], multiple_peaks = False,
                          plot = False, t_transition = 0, x_plateau = None, smooth = True, n_ref_peak = 1):
    
    ''' detect if there's stable oscillation defined as a non-decaying wave'''
    
    peaks, properties = signal.find_peaks(sig, height = peak_threshold)
    
    if plot:
        
        x_plateau = x_plateau or len(sig)
        
        if smooth:
            fluctuations = gaussian_filter1d(sig[:x_plateau], smooth_kern_window)
        else: fluctuations = []
        
        troughs,_ = signal.find_peaks(-sig, height = peak_threshold)
        plot_oscillations(sig, x_plateau, peaks, troughs, t_transition, dt, fluctuations = fluctuations)

    # relative first and last peak ratio thresholding
    if len(peaks) > n_ref_peak + 2: # was 8 for STN-Proto 2 March 
        if not multiple_peaks:
            last_first_peak_ratio = sig[peaks[-1]] / sig[peaks[n_ref_peak]],
        else:
            last_first_peak_ratio = max(sig[peaks[-3:-1]]) / max(sig[peaks[n_ref_peak - 2: n_ref_peak]])
        
    else: 
        
        return False, 0
    
    if last_first_peak_ratio_thresh[0] < last_first_peak_ratio < last_first_peak_ratio_thresh[1]:

    # amplitude envelope Slope thresholding method
    # slope, intercept, r_value, p_value, std_err = stats.linregress(peaks[1:],sig[peaks[1:]]) # discard the first peak because it's prone to errors
    # print('slope = ', slope)
    # if slope > amp_env_slope_thresh: 

        return True, last_first_peak_ratio

    else:
        
        return False, last_first_peak_ratio


def if_oscillatory(sig, x_plateau, peak_threshold, smooth_kern_window, plot = False):
    
    ''' detect if there are peaks with larger amplitudes than 
        noise in mean subtracted data before plateau'''
    
    fluctuations = gaussian_filter1d(sig[:x_plateau], smooth_kern_window)
    peaks,_ = signal.find_peaks(sig, height = peak_threshold)
    troughs,_ = signal.find_peaks(-sig, height = peak_threshold)
    
    if plot:

        plot_oscillations(sig, x_plateau, peaks, troughs, fluctuations = fluctuations)
        
    if len(peaks) > 0 and len(troughs) > 0: # to have at least one maxima and one minima to count as oscillation
    
        return True
    
    else:
    
        return False
    
def plot_oscillations(sig, x_plateau, peaks, troughs, t_transition, dt, fluctuations = [] ):
    
    ''' Plot signal, detected peaks and troughs as well as where the signal is cut'''
    fig, ax = plt.subplots()
    t_list = ( np.arange(len(sig)) + t_transition) * dt
    ax.plot(t_list, sig)
    ax.axvline(( x_plateau + t_transition) * dt)
    if len(fluctuations) != 0:
        ax.plot(t_list[:x_plateau], fluctuations, label = "Gaus kern smoothed")
    ax.plot( ( peaks + t_transition) * dt, sig[peaks],"x", markersize = 10,markeredgewidth = 2)
    ax.plot( ( troughs + t_transition) * dt,sig[troughs],"x", markersize = 10, markeredgewidth = 2)
    ax.legend()
    
def temp_oscil_check(sig_in,peak_threshold, smooth_kern_window,dt,start,end, cut_plateau_epsilon = 0.1):
    

    
    sig = trim_start_end_sig_rm_offset(sig_in,start,end)
    cut_sig_ind = cut_plateau ( sig, epsilon= cut_plateau_epsilon)
    plateau_y = find_mean_of_signal(sig, cut_sig_ind)
    
    if len(cut_sig_ind) > 0: # if it's not all plateau from the beginning
    
        sig = sig - plateau_y
        if_oscillatory(sig, max(cut_sig_ind),peak_threshold, smooth_kern_window)
        if_stable = if_stable_oscillatory(sig, peak_threshold, dt = dt, smooth_kern_window = smooth_kern_window, x_plateau = max(cut_sig_ind))
        n_cycles,freq = zero_crossing_freq_detect(sig[cut_sig_ind],dt/1000)

        print("stable osillatory regime?", if_stable)

        if freq != 0: # then check if there's oscillations
        
            perc_oscil = max_non_empty_array(cut_sig_ind) / len(sig) * 100            
            print("n cycles = ",round(n_cycles/2,2),"% = ", round(perc_oscil,2), "f = ",round(freq,2))
            
        else:
            
            print("all plateau")

    else:
        
        print("no oscillation")
        
def create_df_for_iteration_RM(nuclei_dict, G, n):
    
    data = {} 
    
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0] # get only on class from each population
        data[(nucleus.name, 'mvt_freq')] = np.zeros(n)
        data[(nucleus.name, 'base_freq')] = np.zeros(n)
        data[(nucleus.name, 'last_first_peak_ratio_mvt')] = np.zeros(n)
        data[(nucleus.name, 'perc_oscil_mvt')] = np.zeros(n)
        data[(nucleus.name, 'perc_oscil_base')] = np.zeros(n)
        data[(nucleus.name, 'last_first_peak_ratio_base')] = np.zeros(n)
        data[(nucleus.name, 'n_half_cycles_mvt')] = np.zeros(n)
        data[(nucleus.name, 'n_half_cycles_base')] = np.zeros(n)
        data[(nucleus.name,'g_transient')] = []
        data[(nucleus.name,'g_stable')] = []
        
    
    data['g'] = {}    
    
    for k in list(G.keys()):
        
        data['g'][k] = np.zeros(n)
    return data

def create_df_for_iteration_2d(nuclei_dict, iterating_param_dict, duration, n_iter, n_iter_2, n_run, n_phase_bins = 180, 
                                len_f_pxx = 200, save_pop_act = False, find_phase = False, divide_beta_band_in_power = False, 
                                iterating_name = 'g', check_peak_significance = False, save_gamma = False, pop_act_start = 0):
    
    data = {}
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0]  # get only on class from each population
        data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_iter_2, n_run))
        
        if save_pop_act :
            data[(nucleus.name, 'pop_act')] = np.zeros((n_iter, n_iter_2, n_run,  duration[1] - pop_act_start ))
                                                       
        if check_peak_significance:
            data[(nucleus.name, 'peak_significance')] = np.zeros((n_iter, n_iter_2, n_run), dtype = bool) # stores the value of the PSD at the peak and the mean of the PSD elsewhere
        
        if find_phase and nucleus.neuronal_model == 'spiking':
            
            data[(nucleus.name, 'rel_phase_hist')] = np.zeros((n_iter,  n_iter_2, n_run, nucleus.n, n_phase_bins-2))
            data[(nucleus.name, 'rel_phase_hist_bins')] = np.zeros( n_phase_bins-2 )
            
        if divide_beta_band_in_power:
            if save_gamma:
                data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_iter_2, n_run, 3))
            else:
                data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_iter_2, n_run, 2))
            
        else:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_iter_2,  n_run))
        data[(nucleus.name, 'f')] = np.zeros((n_iter, n_iter_2, n_run, len_f_pxx))
        data[(nucleus.name, 'pxx')] = np.zeros((n_iter, n_iter_2, n_run, len_f_pxx))
        
    data[iterating_name] = iterating_param_dict
    return data


def save_freq_analysis_to_df(data, state, i, nucleus, dt, duration, freq_method = 'zero_crossing'):
    
    (data[(nucleus.name, 'n_half_cycles_' + state)][i],
    data[(nucleus.name,'last_first_peak_ratio_' + state)][i], 
    data[(nucleus.name,'perc_oscil_' + state)][i], 
    data[(nucleus.name, state + '_freq')][i],
    if_stable, _, _ )= nucleus.find_freq_of_pop_act_spec_window(*duration,dt, peak_threshold =nucleus.oscil_peak_threshold, 
                                                    smooth_kern_window = nucleus.smooth_kern_window, check_stability = True,
                                                    method = freq_method)
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

def plot_unconnected_network(G, G_ratio_dict, receiving_class_dict, nuclei_dict, 
                             N, N_real, K_real, A, A_mvt, D_mvt,t_mvt, t_list, dt,
                             color_dict, plot_start_trans, plot_duration,
                             legend_loc, legend_fontsize = 12):
    
    G = multiply_G_by_key_ratio(0, G, G_ratio_dict)
    nuclei_dict = reinitialize_nuclei(nuclei_dict, N, N_real, K_real,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
    nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
    
    fig_unconnected = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = plot_start_trans + plot_duration,
                            include_FR = False, plot_start = plot_start_trans, legend_loc = legend_loc,
                            title_fontsize = 15, title = 'Unconnected', ax = None, continuous_firing_base_lines = False,
                            vspan = True, legend_fontsize = legend_fontsize)
    return fig_unconnected

def synaptic_weight_exploration_RM(N, N_real, K_real, G, A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, 
                                      G_list, nuclei_dict, duration_mvt, duration_base, receiving_class_dict, 
                                      color_dict, if_plot = False, G_ratio_dict = None, plot_start_trans = 0, 
                                      plot_start_stable = 0, plot_duration = 600,
                                      legend_loc = 'upper left', vspan_stable = False, path = None, 
                                      legend_fontsize = 12, loop_name = 'STN-Proto', ylim  = [-4, 76],
                                       legend_fontsize_rec= 18, freq_method = 'zero_crossing'):
    
    
    n = len(G_list)
    data = create_df_for_iteration_RM(nuclei_dict, G, n)
    if_stable_plotted = False
    if_trans_plotted = False
    
    if if_plot:
        fig = plt.figure()
        
    fig_trans, ax_trans = plt.subplots()
    fig_stable, ax_stable = plt.subplots()
    found_g_transient = {k: False for k in nuclei_dict.keys()}
    found_g_stable = {k: False for k in nuclei_dict.keys()}
    fig_unconnected = None
    # fig_unconnected = plot_unconnected_network(G, G_ratio_dict, receiving_class_dict, nuclei_dict, 
    #                                            N, N_real, K_real, A, A_mvt, D_mvt,t_mvt, t_list, dt,
    #                                            color_dict, plot_start_trans, plot_duration,
    #                                            legend_loc, legend_fontsize=legend_fontsize)
    for i, g in enumerate( sorted(G_list, key=abs) ):

        G = multiply_G_by_key_ratio(g, G, G_ratio_dict)
        data = fill_Gs_in_data(data, i, G)
            
        nuclei_dict = reinitialize_nuclei(nuclei_dict, N, N_real, K_real,G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
        nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
        
        nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
        
        for nucleus in nucleus_list:


            data, if_stable_mvt = save_freq_analysis_to_df(data, 'mvt', i, nucleus, dt, duration_mvt, freq_method = freq_method) 
            data, if_stable_base = save_freq_analysis_to_df(data, 'base', i, nucleus, dt, duration_base, freq_method = freq_method)
                                                  

            print(nucleus.name,' Loop G = ', round(multiply_values_of_dict(G), 2), 
                  'n_cycles =', data[(nucleus.name, 'n_half_cycles_mvt')][i],
                  'A ratio =', round(data[(nucleus.name, 'last_first_peak_ratio_mvt')][i],2),
                    'f = ', round(data[(nucleus.name,'mvt_freq')][i],2) )

            if ( not found_g_transient[nucleus.name]  and 
                data[(nucleus.name, 'n_half_cycles_mvt')][i] > lim_n_cycle[0] and 
                data[(nucleus.name, 'n_half_cycles_mvt')][i]< lim_n_cycle[1] 
                ):
                
                data[(nucleus.name,'g_transient')] = abs(g) # save the the threshold g to get transient oscillations
                found_g_transient[nucleus.name] = True
                data['g_loop_transient'] =  abs( multiply_values_of_dict(G) )
                
                if not if_trans_plotted:
                
                    if_trans_plotted = True
                    print("transient plotted")
                                      
                    fig_trans = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = plot_start_trans + plot_duration,
                                     include_FR = False, plot_start = plot_start_trans, legend_loc = legend_loc,
                                     title_fontsize = 15, title = 'Transient Oscillation', continuous_firing_base_lines = False,
                                     vspan = True, legend_fontsize=legend_fontsize)
                    
            if found_g_stable[nucleus.name] == False and if_stable_mvt: 
                
                found_g_stable[nucleus.name] = True
                data[(nucleus.name,'g_stable')] = abs(g)
                data['g_loop_stable'] =  abs( multiply_values_of_dict(G) )
                
                if not if_stable_plotted :
                    
                    if_stable_plotted = True
                    print("stable plotted")
                    
                    # fig_stable1 = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = plot_start_stable + plot_duration,
                    #                   include_FR = False, plot_start = 0, legend_loc = legend_loc, 
                    #                   title_fontsize = 15, title = 'Stable Oscillation', ax = None, continuous_firing_base_lines = False,
                    #                   vspan = vspan_stable )
                    # fig_stable2 = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = plot_start_trans + plot_duration,
                    #                   include_FR = False, plot_start = plot_start_trans, legend_loc = legend_loc, 
                    #                   title_fontsize = 15, title = 'Stable Oscillation', ax = None, continuous_firing_base_lines = False,
                    #                   vspan = True )
                    fig_stable = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, plot_end = t_mvt + 200 + plot_duration,
                                      include_FR = False, plot_start =t_mvt + 200, legend_loc = legend_loc, 
                                      title_fontsize = 15, title = 'Stable Oscillation', continuous_firing_base_lines = False,
                                      vspan = True , legend_fontsize=legend_fontsize)
                        
        if if_plot:# and i > 0:
            
            fig_t = plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, 
                         plot_end = t_list[-1] * dt,# ax = fig_t.gca(),
                         include_FR = False, plot_start = plot_start_trans, legend_loc = legend_loc,
                         title_fontsize = 24, title = r'$G_{Loop}=$' + str(round(multiply_values_of_dict(G), 2)), 
                         continuous_firing_base_lines = False,
                         vspan = True, legend_fontsize= legend_fontsize_rec, label_fontsize = 25, ylim = ylim)
            
            fig_t.set_size_inches((7, 5), forward=False)
            fig_t.savefig( os.path.join(path, loop_name + '_n_{:03d}.pdf'.format(i)), dpi = 300, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

            plt.close(fig_t)
            
        print(i, "from", n)


    data['G_ratio'] = G_ratio_dict 
    
    # if if_plot:
    #     fig.text(0.5, 0.01, 'time (ms)', ha='center')
    #     fig.text(0.01, 0.5, 'firing rate (spk/s)', va='center', rotation='vertical')
    #     fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    # pickle_obj(data, filename)

            
    return fig_unconnected, fig_trans, fig_stable

def synaptic_weight_exploration_RM_2d(N, N_real, K_real, G_ratio_dict, A, A_mvt, D_mvt, t_mvt, t_list, dt, filename,
                                      loop_key_lists, nuclei_dict, duration, receiving_class_dict, 
                                      color_dict, plot_firing = False, G_dict = None,
                                      legend_loc = 'upper left', vspan_stable = False, path = None, 
                                      legend_fontsize = 12, loop_name = 'STN-Proto', ylim  = [-4, 76],
                                       legend_fontsize_rec= 18, n_windows = 3, check_stability = True,
                                       lower_freq_cut = 8, upper_freq_cut = 60, freq_method = 'fft', fft_method = 'Welch',
                                       peak_threshold=0.1, smooth_kern_window=3, smooth_window_ms = 5,
                                       cut_plateau_epsilon=0.1, plot_sig = False, low_pass_filter = False,
                                       lim_oscil_perc=10, plot_spectrum = True, normalize_spec = False,
                                       plot_sig_thresh = False, min_f = 0, max_f = 200, n_std_thresh = 2,
                                       save_pxx = True, len_f_pxx = 150, AUC_ratio_thresh = 0.1, plot_peak_sig = False,
                                       check_peak_significance = True, print_AUC_ratio = False,
                                       change_states = False, save_pop_act = True, n_run = 1, pop_act_start = None
                                       ):
    
    n_iter_1 = len(G_dict[loop_key_lists[0][0]])
    n_iter_2 = len(G_dict[loop_key_lists[1][0]])
    pop_act_start = pop_act_start or 0
    
    data = create_df_for_iteration_2d(nuclei_dict, G_dict, duration, n_iter_1, n_iter_2, n_run,
                                         len_f_pxx = len_f_pxx, save_pop_act = True, divide_beta_band_in_power = False, 
                                         iterating_name = 'g', check_peak_significance = check_peak_significance,
                                         pop_act_start = pop_act_start)
    figs = []
    if plot_firing:
        
        fig_FR = plt.figure()
        figs.append(fig_FR)
        
    if plot_spectrum:
        
        fig_spec = plt.figure()
        figs.append(fig_spec)

    
    count = 0
    G = {}
    
    for r in range(n_run):
        for i in range(n_iter_1):
            
            
            for k in loop_key_lists[0]:
                
                G[k] = G_dict[k][i] *G_ratio_dict[k]
    
            for m in range(n_iter_2):
            
                for k in loop_key_lists[1]:
                    
                    G[k] = G_dict[k][m] *G_ratio_dict[k]
                nuclei_dict = reinitialize_nuclei_RM(nuclei_dict, N, N_real, K_real, 
                                                  G, A, A_mvt, D_mvt, t_mvt, t_list, dt, 
                                                  change_states = change_states)
                nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)
                
                if plot_spectrum:
                    
                    ax_spec = fig_spec.add_subplot(n_iter_1, n_iter_2, count + 1)
                    
                else: ax_spec = None
                
        
                if save_pop_act:
                    data = save_pop_act_into_dataframe_2d(nuclei_dict, pop_act_start, data, i, m, r)
                    
                data, nuclei_dict = find_freq_all_nuclei_and_save(data, (i, m, r), dt, nuclei_dict, duration, 
                                                                  lim_oscil_perc, peak_threshold, smooth_kern_window, 
                                                                  smooth_window_ms, cut_plateau_epsilon,
                                                                  check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, 
                                                                  upper_freq_cut, plot_spectrum=plot_spectrum, ax=ax_spec,
                                                                  c_spec=color_dict, n_windows=n_windows, 
                                                                  fft_method= fft_method, find_beta_band_power= True,
                                                                  divide_beta_band_in_power = False, include_beta_band_in_legend=False,
                                                                  half_peak_range = 5, cut_off_freq = 100, 
                                                                  check_peak_significance=check_peak_significance, 
                                                                  save_pxx = save_pxx, len_f_pxx = len_f_pxx, 
                                                                  normalize_spec=normalize_spec, plot_sig_thresh = plot_sig_thresh, 
                                                                  plot_peak_sig = plot_peak_sig, min_f = min_f, max_f = max_f, 
                                                                  n_std_thresh= n_std_thresh, 
                                                                  AUC_ratio_thresh = AUC_ratio_thresh,
                                                                  print_AUC_ratio = print_AUC_ratio)                               
           
                if plot_firing:
    
                    ax_FR = fig_FR.add_subplot(n_iter_1, n_iter_2, count + 1)
                    plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax_FR, '', include_std=False, 
                         round_dec=True,  plt_txt='horizontal', plt_mvt=False, plt_freq=True, 
                         plot_start=duration[0] * dt, plot_end = duration[1] * dt, include_FR = False)
                    # ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
                    # ax.set_ylim(firing_ylim)
                    rm_ax_unnecessary_labels_in_subplots(count, n_iter_1, ax_FR)
                    
                count += 1
                print(count, "from", (n_iter_1 * n_iter_2 * n_run))


    
    pickle_obj(data, filename)
    return figs
            
    
def G_sweep_title(G_dict, g_1, g_2):
    
    title = ( r"$G_{" + list(G_dict.keys())[0][1] + "-" + 
                        list(G_dict.keys())[0][0] + "}$ = " + 
                        str(round(g_1, 2)) + r"$\; G_{" +
                        list(G_dict.keys())[1][1] + "-" +
                        list(G_dict.keys())[1][0] + "}$ =" + 
                        str(round(g_2, 2))
            )
    return title

def create_receiving_class_dict(receiving_pop_list, nuclei_dict):
    
    ''' make a list of nuclei classes that project to one 
        nuclei class form the given list of pop names'''
    
    receiving_class_dict = {}
    
    for key in receiving_pop_list.keys():
        
        receiving_class_dict[key] = [nuclei_dict[name][int(k)-1] for name, k in list(receiving_pop_list[key])]
        
    return receiving_class_dict


    
def reinitialize_nuclei_RM(nuclei_dict, N, N_real, K_real, G, A, A_mvt, D_mvt,t_mvt, t_list, dt, change_states = True):
    
    ''' Clear history of the population. Reinitialize the synaptic 
        weights and corresponding external inputs '''
        
    K = calculate_number_of_connections(N, N_real, K_real)

    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            nucleus.clear_history()
            nucleus.set_connections(K, N)
            nucleus.reset_synaptic_weights(G, N)
            nucleus.scale_synaptic_weight() 
            nucleus.set_ext_input(A, A_mvt, D_mvt,t_mvt, t_list, dt, change_states = change_states)
            
    return nuclei_dict


    
def save_freq_analysis_to_df_2d(data, state, i, j, nucleus, dt, duration, check_stability = True, method = 'zero_crossing'):
    
    (data[(nucleus.name, 'n_half_cycle_' + state)][i, j], _, _,
    data[(nucleus.name, state + '_freq')][i, j], if_stable, _, _) = nucleus.find_freq_of_pop_act_spec_window(*duration, dt, 
                                                                                        peak_threshold = nucleus.oscil_peak_threshold, 
                                                                                        smooth_kern_window = nucleus.smooth_kern_window, 
                                                                                        check_stability = check_stability, method = method)
    print(nucleus.name, ' stability =', if_stable)                                                                                             
    return data

def create_df_tau_sweep_2d(nuclei_dict, G, iter_param_length_list, n_time_scale, n_timebins, synaptic_time_constant,
                                  check_transient = False):
    '''build a data dictionary for 2d tau sweep '''
    
    data = {} ; 
    for nucleus_list in nuclei_dict.values():
        
        nucleus = nucleus_list[0] # get only on class from each population
        data[(nucleus.name, 'stable_freq')] = np.zeros(iter_param_length_list)
        data[(nucleus.name, 'n_half_cycle_stable')] = np.zeros(iter_param_length_list)
        
        if check_transient:
            data[(nucleus.name, 'trans_freq')] = np.zeros(iter_param_length_list)
            data[(nucleus.name, 'n_half_cycle_trans')] = np.zeros(iter_param_length_list)


    data['g_stable'] = np.zeros(iter_param_length_list)
    data['g_transient'] = np.zeros(iter_param_length_list)
    data['tau'] = {}
    
    for k in list(synaptic_time_constant.keys()):
        data['tau'][k] = np.zeros(iter_param_length_list)
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
    
def run_and_derive_freq(data, N, N_real, K_real, receiving_class_dict, 
                        nuclei_dict, g_transient, duration, i, j , state,
                        G, G_ratio_dict, A, A_mvt, D_mvt,t_mvt, t_list, dt, 
                        check_stability = False, freq_method = 'zero_crossing'):
    
    G = multiply_G_by_key_ratio(g_transient, G, G_ratio_dict)
    
    nuclei_dict = reinitialize_nuclei_RM(nuclei_dict, N, N_real, K_real, G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
    run(receiving_class_dict,t_list, dt, nuclei_dict)

    nucleus_list = [nucleus_list[0] for nucleus_list in nuclei_dict.values()]
    for nucleus in nucleus_list:
        data = save_freq_analysis_to_df_2d(data, state, i, j, nucleus, dt, duration, 
                                           check_stability = check_stability, method = freq_method)
        
    return data

def sweep_time_scales_2d(g_list, G_ratio_dict, synaptic_time_constant, nuclei_dict, N, N_real, K_real,
                      syn_decay_dict, filename, G,A,A_mvt, D_mvt,t_mvt, receiving_class_dict, 
                      t_list,dt, duration_base, duration_mvt, lim_n_cycle, find_stable_oscill=True, 
                      check_transient = False, if_track_tau_2 = True, freq_method = 'zero_crossing', change_states = True):
    
    
    t_decay_series_1 = list(syn_decay_dict['tau_1']['tau_list']) 
    t_decay_series_2 = list(syn_decay_dict['tau_2']['tau_list'])
    n_runs = len(t_decay_series_1)*len(t_decay_series_2)
    
    data  = create_df_tau_sweep_2d(nuclei_dict, G, (len(t_decay_series_1), len(t_decay_series_2)), 
                                          2, len(t_list), synaptic_time_constant, check_transient = check_transient)    
    data['G_ratio'] = G_ratio_dict
    count = 0 
    
    for i, t_decay_1 in enumerate( t_decay_series_1 ) :
        
        for j, t_decay_2 in enumerate( t_decay_series_2 ) :
            
            synaptic_time_constant = extract_syn_time_constant_from_dict(synaptic_time_constant, syn_decay_dict, 
                                                                         t_decay_1, t_decay_2, if_track_tau_2 = if_track_tau_2 )
            nuclei_dict = reinitialize_nuclei_RM(nuclei_dict, N, N_real, K_real, G, 
                                              A, A_mvt, D_mvt,t_mvt, t_list, dt, change_states=change_states)
            nuclei_dict = set_time_scale(nuclei_dict, synaptic_time_constant)
            
            g_transient, g_stable = find_oscillation_boundary(g_list, nuclei_dict.copy(),  
                                                              N, N_real, K_real, G,
                                                              G_ratio_dict, A, A_mvt,t_list,dt, 
                                                              receiving_class_dict, D_mvt, 
                                                              t_mvt, duration_mvt, duration_base, 
                                                              lim_n_cycle = lim_n_cycle , 
                                                              find_stable_oscill = find_stable_oscill)
            
            data = fill_taus_and_Gs_in_data(data, i, j, synaptic_time_constant,  g_transient, g_stable)
            
            data = run_and_derive_freq(data,  N, N_real, K_real, receiving_class_dict,
                                       nuclei_dict, g_stable, duration_mvt, i, j , 
                                       'stable', G, G_ratio_dict, A, A_mvt, 
                                       D_mvt,t_mvt, t_list, dt, 
                                       check_stability = True, freq_method = freq_method)
            
            if check_transient:

                data = run_and_derive_freq(data,  N, N_real, K_real, receiving_class_dict,
                                           nuclei_dict, g_transient, duration_mvt, i, j ,
                                           'trans', G, G_ratio_dict, A, A_mvt, 
                                           D_mvt,t_mvt, t_list, dt, freq_method = freq_method)
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

def find_oscillation_boundary(g_list,nuclei_dict,N, N_real, K_real, G, G_ratio_dict,A, A_mvt,t_list,dt, receiving_class_dict, 
                              D_mvt, t_mvt, duration_mvt, duration_base, lim_n_cycle = [6,10], 
                              find_stable_oscill = False):
    
    ''' find the synaptic strength for a given set of parametes where you oscillations 
        appear after increasing g'''
    
    found_transient_g = False
    found_stable_g = False
    g_stable = None
    
    for g in sorted(g_list, key=abs):
        
        G = multiply_G_by_key_ratio(g, G, G_ratio_dict)
        
        nuclei_dict = reinitialize_nuclei_RM(nuclei_dict, N, N_real, K_real, G, A, A_mvt, D_mvt,t_mvt, t_list, dt)
        run(receiving_class_dict,t_list, dt, nuclei_dict)
        
        nucleus = list(nuclei_dict.values())[0][0]
        n_half_cycles, last_first_peak_ratio , perc_oscil, f, if_stable, _, _ = nucleus.find_freq_of_pop_act_spec_window(*duration_mvt, dt, 
                                                                                                  peak_threshold = nucleus.oscil_peak_threshold,
                                                                                                  smooth_kern_window = nucleus.smooth_kern_window, 
                                                                                                  check_stability= find_stable_oscill, method = "zero_crossing")
        # n_half_cycles_base, perc_oscil_base, f_base, if_stable_base,_,_ = nucleus.find_freq_of_pop_act_spec_window(*duration_base, dt, 
        #                                                                                                peak_threshold = nucleus.oscil_peak_threshold, 
        #                                                                                                smooth_kern_window = nucleus.smooth_kern_window, 
        #                                                                                                check_stability= find_stable_oscill)
        
        print('g = {}, f = {}, {} %, P-ratio = {}'.format(round(g,1), 
                                            round(f , 1), 
                                            round(perc_oscil, 1),
                                            round(last_first_peak_ratio, 2))
              )
        
        # check_if_zero_activity(duration_mvt, nucleus)
       
        if lim_n_cycle[0] <= n_half_cycles <= lim_n_cycle[1] and not found_transient_g:
            
            found_transient_g = True
            g_transient = g 
            print("Gotcha transient!")
                
            if  not find_stable_oscill:
                break
            
        if find_stable_oscill and if_stable and not found_stable_g:
            
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
                                                 alpha_transient = 1, mark_pts_with = 'line'):
    
    
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

    ax =  mark_transient_stable_oscillation_pts(ax, mark_pts_with, g_transient, g_stable, data, i, name_list, 
                                          y_list, color_list, vline_width, alpha_transient)
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

def mark_transient_stable_oscillation_pts(ax, mark_pts_with, g_transient, g_stable, data, i, name_list, 
                                          y_list, color_list, vline_width, alpha_transient):
    
    if mark_pts_with == 'line':
                
        ax.axvline(g_transient , alpha = alpha_transient,
                   linestyle = '-.', c = color_list[i],
                    lw = vline_width) 
        
        ax.axvline(g_stable , 
                   c = color_list[i], lw = vline_width) 
        
    if mark_pts_with == 'arrow':
        
        trans_i = np.where( abs(Product_G(data)) == g_transient)[0][0]

        ax.arrow(g_transient,  np.squeeze(data[(name_list[i],y_list[i])])[trans_i] + 0.2, 
                 0, -0.12, head_length = 0.05, width = 0.05, edgecolor = None, alpha = 0.5,
                 facecolor = 'k', head_starts_at_zero = True, length_includes_head = True)
        
        stab_i = np.where( abs(Product_G(data)) == g_stable )[0][0]

        ax.arrow(g_stable,  np.squeeze(data[(name_list[i],y_list[i])])[stab_i] + 0.2, 
                 0, -0.12, head_length = 0.05, width = 0.05, edgecolor = None, alpha = 1,
                 facecolor = 'k', head_starts_at_zero = True, length_includes_head = True)
    return ax

def set_max_dec_tick(ax, n_decimal = 1):
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.'+ str(n_decimal) + 'f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.'+ str(n_decimal) + 'f'))
    
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

def set_y_ticks_one_ax(ax, label_list):
    
    
    y_formatter = FixedFormatter([str(x) for x in label_list])
    y_locator = FixedLocator(label_list)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.set_major_locator(y_locator)
    
    return ax

def set_x_ticks_one_ax(ax, label_list):
    
    
    x_formatter = FixedFormatter([str(x) for x in label_list])
    x_locator = FixedLocator(label_list)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(x_locator)
    
    return ax

def set_xy_ticks_one_ax(ax, x_label_list, y_label_list):
    
    set_x_ticks_one_ax(ax, x_label_list)
    set_y_ticks_one_ax(ax, y_label_list)
    
    return ax

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
  
def remove_all_y_labels(fig):
    
    for ax in fig.axes:
        ax.axes.yaxis.set_ticklabels([])
        ax.set_ylabel('')
        
    return fig

def remove_all_labels(fig):
    
    fig  = remove_all_x_labels(fig)
    fig  = remove_all_y_labels(fig)
    
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


def eval_averaged_PSD_peak_significance(data, x_list, y_list, name_list, 
                                        AUC_ratio_thresh = 0.2, plot = False, 
                                        smooth = False, n_ref_peak = 6,
                                        examples_ind  = None, dt = 0.1,
                                        last_first_peak_ratio_thresh = [0.9,1.19]):
    
    peak = {name: np.zeros((len(x_list), len(y_list))) for name in name_list}
    peak_significance = {name: np.zeros((len(x_list), len(y_list)), dtype = bool) for name in name_list}
    last_first_peak_ratio = {name: np.zeros((len(x_list), len(y_list))) for name in name_list}
    for i, x in enumerate(x_list):
        
        for j, y in enumerate(y_list):
            
            for name in name_list:
                                
                pxx = np.nanmean(data[(name, 'pxx')][i, j], axis = 0)
                pxx = drop_nan(pxx)
                f = drop_nan(data[name, 'f'][i,j, 0])
                peak[name][i,j] = data[name, 'f'][i,j, 0][np.argmax(pxx)]
                pop_act = signal.detrend(data[(name, 'pop_act')][i, j, 0])
                # if peak[name][i,j] > 0:
                #     print(peak[name][i,j])
                #     pop_act = butter_bandpass_filter(pop_act, max(peak[name][i,j] - 5, 1), peak[name][i,j] + 5, 1/dt/1000, order=5)
                (if_stable, last_first_peak_ratio[name][i,j]) = if_stable_oscillatory(pop_act, 0, plot = plot, 
                                                                         smooth = smooth, n_ref_peak = n_ref_peak, multiple_peaks = True,
                                                                         last_first_peak_ratio_thresh = last_first_peak_ratio_thresh)
                peak_significance[name][i,j] = if_stable
                
                if (i,j) in list(examples_ind.values()):
                    print(i,j, name, last_first_peak_ratio[name][i,j])

                # peak_significance[name][i,j] = check_significance_of_PSD_peak(f, pxx,  n_std_thresh = 2, 
                #                                                                  min_f = 0, max_f = 200, n_pts_above_thresh = 2, 
                #                                                                   AUC_ratio_thresh = AUC_ratio_thresh,if_plot = False,
                #                                                                  xlim = [0, 80], name = '', print_AUC_ratio = False, f_cut = 1)

    return peak, peak_significance,  last_first_peak_ratio

def parameterscape(x_list, y_list, name_list, markerstyle_list, freq_dict, color_dict, peak_significance,
                   size_list, xlabel, ylabel, clb_title = '', label_fontsize = 18, cmap = 'jet', tick_size = 15,
                   annotate = True, ann_name = 'Proto', clb_tick_size  = 20, only_significant = True,
                   y_ticks = None, multirun = False):
    
    """Plot the frequency as a colormap with different neural populations as different 
        markers tiling the plot.
    """
    fig, ax = plt.subplots(1, 1)
    cm = plt.cm.get_cmap(cmap)
    # vmax = max_in_dict(color_dict)
    # vmin = min_in_dict(color_dict)
    vmax = 70 # to be constistent in all parameterscapes
    vmin = 0
    for i, x in enumerate(x_list):
        
        for j, y in enumerate(y_list):
            
            for name, ms, s in zip(name_list, markerstyle_list, size_list):
                
                if only_significant:
                    
                    
                    if peak_significance[name][i,j]:
                        
                        img = ax.scatter(x, y, marker = ms, c = color_dict[name][i,j], 
                                         s = s, cmap = cm, edgecolors = 'k', 
                                         vmax = vmax, 
                                         vmin = vmin)
                        
                    else:
                        
                        ax.scatter(x, y, marker = ms, c = 'grey', 
                                         s = s, edgecolors = 'k')
                else:
                    
                    img = ax.scatter(x, y, marker = ms, c = color_dict[name][i,j], 
                                     s = s, cmap = cm, edgecolors = 'k', 
                                     vmax = vmax, 
                                     vmin = vmin)
                if annotate:
                    
                    ax.annotate(int(freq_dict[ann_name][i,j]), (x,y), color = 'k')
           
    ax.set_xlim(x_list[0] - (x_list[1] - x_list[0]),
                x_list[-1] + (x_list[1] - x_list[0]))
    
    ax.set_ylim(y_list[0] - (y_list[1] - y_list[0]),
                y_list[-1] + (y_list[1] - y_list[0]))
                
    ax.set_xlabel(xlabel, fontsize = label_fontsize)
    ax.set_ylabel(ylabel, fontsize = label_fontsize)
    # ax.set_title(title, fontsize = label_fontsize)
    # ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    y_ticks = y_ticks or ax.get_yticks().tolist()[:-1]
    fig = set_y_ticks(fig, y_ticks)
    
    clb = fig.colorbar(img, shrink=0.5, ax = ax)
    clb.set_label(clb_title, labelpad=-60, y=0.5, rotation=-90, fontsize = label_fontsize)
    clb.ax.tick_params(labelsize=clb_tick_size )
    set_max_dec_tick(ax)
    clb.ax.yaxis.tick_right()

    remove_whole_frame(ax)
    # set_n_ticks(ax, 4, 4)
    remove_tick_lines(ax)
    
    return fig

def and_bools_multiple_keys(dictionary):
    
    summed = np.ones_like(list(dictionary.values()[0]), dtype = bool)
    for key, val in dictionary.items():
        summed = np.logical_and(summed, val)
        
    return summed

def parameterscape_imshow(x_list, y_list, name_list, markerstyle_list, freq_dict, color_dict, peak_significance,
                   size_list, xlabel, ylabel, clb_title = r'$Frequency\; (Hz)$', label_fontsize = 16, cmap = 'jet',
                   tick_size = 15,
                   annotate = True, ann_name = 'Proto', clb_tick_size  = 25, only_significant = True, x_ticks = None,
                   y_ticks = None, multirun = False, name = 'Proto', figsize = (7,7), n_decimal = 0, tick_length = 10):
    
    """Plot the frequency as a colormap with different neural populations as different 
        markers tiling the plot.
    """
    fig, ax = plt.subplots(figsize = figsize)
    cm = plt.cm.get_cmap(cmap)
    

    masked = np.ma.masked_where(~peak_significance[name].T, color_dict[name].T)
    
    cm.set_bad(color='white')
    n_x = len(x_list) - 1
    n_y = len(y_list) - 1
    binsize_x = (x_list[-1]- x_list[0]) / n_x
    binsize_y = (y_list[-1]- y_list[0]) / n_y
    img = ax.imshow(masked, cmap = cm, vmin = 0, vmax = 70, 
                    extent=[x_list[0] - binsize_x/2, x_list[-1] + binsize_x/2, 
                            y_list[-1] + binsize_y/2, y_list[0] - binsize_y/2],
                    aspect='auto')

    print(x_list[0] - binsize_x/2, x_list[-1] + binsize_x/2, 
        y_list[-1] + binsize_y/2, y_list[0] - binsize_y/2)
    ax.add_patch(matplotlib.patches.Rectangle((x_list[-4] - binsize_x/2, y_list[0] - binsize_y/2), 
                                   binsize_x * 4, binsize_y * 2, hatch='//', fill=False, 
                                   snap=False, linewidth=1))
    ax.set_xlabel(xlabel, fontsize = label_fontsize)
    ax.set_ylabel(ylabel, fontsize = label_fontsize)
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=tick_size, length = tick_length)
    y_ticks = y_ticks or ax.get_yticks().tolist()[:-1]
    fig = set_y_ticks(fig,y_ticks)
    
    x_ticks = x_ticks or ax.get_xticks().tolist()[:-1]
    fig = set_x_ticks(fig, x_ticks)
    clb = fig.colorbar(img, shrink=0.8, ax = ax)
    clb.set_ticks([0, 20, 40, 60])
    clb.set_ticklabels([0, 20, 40, 60])
    clb.set_label(clb_title, labelpad=40, y=0.5, rotation=-90, fontsize = label_fontsize)
    clb.ax.tick_params(labelsize=clb_tick_size, length = tick_length)
    set_max_dec_tick(ax, n_decimal = n_decimal)
    clb.ax.yaxis.tick_right()

    fig.tight_layout()
    return fig

def highlight_example_pts(fig, examples_ind, x_list, y_list, size_list, text_size = 25,
                          highlight_color = 'w', alpha = 0.5, annotate_shift = 0.1):
    
    ax = fig.gca()
    
    for key, ind in examples_ind.items():
        
        s = size_list[0] * 1.5
        x = x_list[ind[0]] 
        y = y_list[ind[1]] 
        x_span = abs(x_list[1] - x_list[0])
        y_span = abs(y_list[1] - y_list[0])
        
        ax.scatter(x, y, marker = 'o', c = highlight_color, 
                   alpha = alpha, s = s, edgecolors = 'k')
        txt = ax.annotate(key, (x - x_span * annotate_shift , y - y_span * annotate_shift), color = 'k', size = text_size)
        txt.set_path_effects([pe.withStroke(linewidth=5, foreground='w')])

    return fig


def plot_PSD_of_example_pts(data_all, examples_ind,  x_list, y_list, name_list, color_dict):
    
    
    for key, ind in examples_ind.items():
        fig, ax = plt.subplots()
        for name in name_list:
            
            f = data_all[(name, 'f')][ind[1], ind[0], 0, : ]
            pxx = np.average(data_all[(name, 'pxx')][ind[1], ind[0], :, : ], axis = 0)
            pxx = norm_PSD( pxx, f)
            peak_freq = np.round(data_all[(name, 'peak_freq_all_runs')][ind[1], ind[0]] , 1)
            ax.plot(f, pxx, c = color_dict[name], label = name + ' ' +  str(peak_freq) + ' Hz', lw=1.5)
        
        ax.set_xlim(0, 60)
        ax.legend()
        ax.set_title(key, fontsize = 15)
        
        
def plot_pop_act_and_PSD_of_example_pts(data, name_list, examples_ind, x_list, y_list, dt, color_dict, 
                                        Act, state = 'awake_rest',smooth = True, tick_size = 15,
                                        plt_duration = 600, run_no = 0, window_ms = 5, tick_length = 5):
    
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
            pxx = norm_PSD( pxx, f) * 100
            peak_freq = np.round(data[(name, 'peak_freq_all_runs')][ind[1], ind[0]] , 1)
            ax_PSD.plot(f, pxx, c = color_dict[name], label = name + ' ' +  str(peak_freq) + ' Hz', lw=1.5)
        
            duration = data[(name, 'pop_act')].shape[-1]
            
            pop_act = data[(name, 'pop_act')][ind[1], ind[0], run_no, duration - int( plt_duration/dt) : duration]
            
            if smooth:
                pop_act = moving_average_array(pop_act, int(window_ms / dt))
                
            t_list = np.arange( duration - int( plt_duration/ dt), duration) * dt / 1000
            ax_pop_act.plot(t_list, pop_act, c = color_dict[name], lw = 1.5)
            ax_pop_act.plot(t_list, np.full_like(t_list, Act[state][name]), '--', 
                                                 c = color_dict[name],lw = 1, alpha=0.8 )

        set_minor_locator(ax_PSD, n = 3, axis = 'y')
        set_minor_locator(ax_pop_act, n = 2, axis = 'both')
        ax_pop_act.tick_params(axis='both', which='major', labelsize=tick_size,  length = tick_length)
        ax_PSD.tick_params(axis='both', which='major', labelsize=tick_size, length = tick_length)
        ax_pop_act.tick_params(axis='both', which='minor', labelsize=tick_size,  length = tick_length/2)
        ax_PSD.tick_params(axis='both', which='minor', labelsize=tick_size, length = tick_length/2)
        ax_PSD.set_xlim(0, 80)
        ax_PSD.legend(fontsize = 8, frameon = False, loc = 'upper right')
        ax_PSD.set_ylabel(key, fontsize = 20, rotation = 0, labelpad = 10)
        # ax_pop_act.set_title(key, fontsize = 15)
        fig.add_subplot(ax_PSD)
        fig.add_subplot(ax_pop_act)
        fig.text(0.6, 0.085, 'Time (ms)', ha='center', fontsize = 15)
        fig.text(0.9, 0.5, 'firing rate (spk/s)', va='center', rotation=-90, fontsize = 15)
        fig.text(0.2, 0.085, 'frequency', ha='center', fontsize = 15)
        fig.text(0.05, 0.5, 'Normalized Power' + r'$(\times 10^{-2})$', va='center', rotation='vertical',fontsize = 15)
        
    return fig

def drop_nan(x):
    
    return x[~np.isnan(x)]

    
def reeval_PSD_peak_significance(data, x_list, y_list, name_list, AUC_ratio_thresh = 0.2):
    for i, x in enumerate(x_list):
        
        for j, y in enumerate(y_list):
            
            for name in name_list:
                
                n_run = data[name, 'f'].shape[-2]
                
                for r in range(n_run):
                    
                    data[name, 'peak_significance'][i,j, r] = check_significance_of_PSD_peak(data[name, 'f'][i,j, r], data[name,'pxx'][i,j, r],  n_std_thresh = 2, 
                                                                                             min_f = 0, max_f = 200, n_pts_above_thresh = 2, 
                                                                                             ax = None, legend = 'PSD', c = 'k', if_plot = False, AUC_ratio_thresh = AUC_ratio_thresh,
                                                                                             xlim = [0, 80], name = '', print_AUC_ratio = False, f_cut = 1)
                    
    return data

def plot_pop_act_and_PSD_of_example_pts_RM(data, name_list, examples_ind, x_list, y_list, dt, color_dict, Act, PSD_duration = None,
                                           state = 'rest', plt_duration = 600, run_no = 0, window_ms = 5, 
                                           act_ylim = (-5, 90), PSD_ylim = None, PSD_xlim = (0, 80), ylabel = 'Normalized Power',
                                           PSD_y_labels = [0, 40, 80, 120], act_y_labels = [0, 40, 80], normalize_spec = False,
                                           PSD_x_labels = [0, 20, 40, 60], act_x_labels = [4000, 4150, 4300], run = 0,
                                           last_first_peak_ratio = None, unit_variance_PSD = False, f_in_PSD_label = False,
                                           tick_size = 12, tick_length = 5, highlight_key_size = 18):
    
    n_exmp = len(examples_ind)
    fig = plt.figure( figsize=(12, 20) ) 
    outer = gridspec.GridSpec( n_exmp, 1, wspace=0.2, hspace=0.2)
    
    for i, (key, ind) in enumerate(examples_ind.items()):
    
        inner = gridspec.GridSpecFromSubplotSpec( 1, 2, width_ratios=[1, 2],
                                                 subplot_spec=outer[i], 
                                                 wspace=0.1, hspace=0.1)
        ax_PSD = plt.Subplot(fig, inner[0])
        ax_pop_act = plt.Subplot(fig, inner[1])
        
        for name in name_list:

            f = data[(name, 'f')][ind[0], ind[1], run, : ]
            pxx = np.nanmean(data[(name, 'pxx')][ind[0], ind[1]], axis = 0)
            f = drop_nan(f)
            pxx = drop_nan(pxx)

            if normalize_spec :

                pxx = norm_PSD(pxx, f) * 100
                ylabel = 'Norm. Power ' + r'$(\times 10^{-2})$'
                


            peak_freq = np.round(np.average(data[(name, 'base_freq')][ind[0], ind[1]]) , 1)

            length = data[(name, 'pop_act')].shape[-1]
            pop_act = data[(name, 'pop_act')][ind[0], ind[1], run, length - int( plt_duration/dt) : length]
            
            if unit_variance_PSD:
                PSD_duration = PSD_duration or length
                pxx = pxx/ np.var(data[(name, 'pop_act')][ind[0], ind[1], run, length - PSD_duration:])
            
            label = name 
            if f_in_PSD_label:
                label += ' ' +  str(int(round(peak_freq,0))) + ' Hz'

            ax_PSD.plot(f, pxx,'-', c = color_dict[name], 
                        label = label, lw=1.5)
        
            # t_list = np.arange( length - int( plt_duration/ dt), length) * dt 
            t_list = np.arange(int( plt_duration/ dt)) * dt 

            ax_pop_act.plot( t_list, pop_act, c = color_dict[name], lw = 1.5)
            ax_pop_act.plot(t_list, np.full_like(t_list, Act[state][name]), '--', 
                                                 c = color_dict[name],lw = 1, alpha=0.8 )
        set_minor_locator(ax_PSD, n = 3, axis = 'y')
        set_minor_locator(ax_pop_act, n = 2, axis = 'x')
        set_minor_locator(ax_pop_act, n = 3, axis = 'y')

        ax_pop_act.tick_params(axis='both', which='major', labelsize=tick_size,  length = tick_length)
        ax_PSD.tick_params(axis='both', which='major', labelsize=tick_size, length = tick_length)
        ax_pop_act.tick_params(axis='both', which='minor', labelsize=tick_size,  length = tick_length/2)
        ax_PSD.tick_params(axis='both', which='minor', labelsize=tick_size, length = tick_length/2)

        ax_pop_act.set_ylim(act_ylim)
        ax_PSD.set_ylim(PSD_ylim or ax_PSD.get_ylim())
        ax_PSD.set_xlim(PSD_xlim)
        ax_PSD = set_xy_ticks_one_ax(ax_PSD, PSD_x_labels, PSD_y_labels)
        ax_pop_act = set_xy_ticks_one_ax(ax_pop_act, act_x_labels, act_y_labels)
        remove_frame(ax_PSD)
        remove_frame(ax_pop_act)
        ax_PSD.legend(fontsize = 10, frameon = False, loc = 'upper right')
        ax_PSD.set_ylabel(key, fontsize = highlight_key_size, rotation = 0, labelpad = -5)
        # ax_pop_act.set_title(key, fontsize = 15)
        rm_ax_unnecessary_labels_in_subplots(i, n_exmp, ax_PSD, axis = 'x')
        rm_ax_unnecessary_labels_in_subplots(i, n_exmp, ax_pop_act, axis = 'x')
        fig.add_subplot(ax_PSD)
        fig.add_subplot(ax_pop_act)
    fig.text(0.7, 0.04, 'Time (ms)', ha='center', fontsize = 15)
    fig.text(0.9, 0.5, 'Firing rate (spk/s)', va='center', rotation=-90, fontsize = 18)
    fig.text(0.25, 0.04, 'Frequency (Hz)', ha='center', fontsize = 15)
    fig.text(0.0, 0.5, ylabel, va='center', rotation='vertical',fontsize = 18)
    fig.tight_layout()
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
            pxx = norm_PSD( pxx, f) * 100
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

        ax_PSD.legend(fontsize = 13, frameon = False, loc = 'upper right')
        ax_PSD.set_ylabel(key, fontsize = 20, rotation = 0, labelpad = 10)
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
        pxx_2d_arr = norm_PSD_2d(pxx_2d_arr, freq_list, axis = 0)
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
#     data  = create_df(nuclei_dict, [len(t_decay_series_1), len(t_decay_series_2)], 2,len(t_list))    
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
#     data  = create_df(nuclei_dict, [len(t_decay_series_1), len(t_decay_series_2)], 2,len(t_list))    
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

def set_ylim_trans_stable_figs(figs, ymax = [100, 100], ymin = [-4, -4]):
    
    for i, fig in enumerate( figs ) :
    
        ax = fig.axes
        ax[0].set_ylim(ymin[i], ymax[i])
    
    return figs

def save_trans_stable_figs( figs, states, path_rate, filename, figsize = (10,5), 
                            ymax = [100, 100], ymin = [-4, -4]):
    
    figs = set_ylim_trans_stable_figs(figs, ymax = ymax , ymin = ymin)
    for (fig, state) in zip( figs, states):
        save_pdf_png(fig, filename + '_' + state + '_plot', size = figsize)

# def save_trans_stable_figs(fig_trans, fig_stable_list, path_rate, filename, figsize = (10,5), ymax = [100, 100], ymin = [-4, -4]):
    

#     fig_trans.set_size_inches(figsize, forward=False)
    
#     for i, fig_stable in enumerate(fig_stable_list):
#         fig_trans , fig_stable = set_ylim_trans_stable_figs([fig_trans, fig_stable], ymax = ymax , ymin = ymin)
#         fig_stable.set_size_inches(figsize, forward=False)

#         fig_stable.savefig(os.path.join(path_rate, (filename + '_stable_plot_' + str(i) + '.png')),dpi = 300, facecolor='w', edgecolor='w',
#                         orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        
#         fig_stable.savefig(os.path.join(path_rate, (filename + '_stable_plot_' + str(i) + '.pdf')),dpi = 300, facecolor='w', edgecolor='w',
#                         orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
#         fig_trans.savefig(os.path.join(path_rate, (filename + '_tansient_plot.png')),dpi = 300, facecolor='w', edgecolor='w',
#                         orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        
#         fig_trans.savefig(os.path.join(path_rate, (filename + '_tansient_plot.pdf')),dpi = 300, facecolor='w', edgecolor='w',
#                         orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        

# def sweep_time_scales_STN_GPe(g_list,g_ratio,nuclei_dict, GABA_A,GABA_B, Glut, filename, G, A,A_mvt, D_mvt,t_mvt, receiving_class_dict,t_list,dt, duration_base, duration_mvt, lim_n_cycle,find_stable_oscill):

#     def set_time_scale(nuclei_dict, inhibitory_trans_1, inhibitory_trans_1_val, inhibitory_trans_2, inhibitory_trans_2_val, glut ):
#         for nucleus_list in nuclei_dict.values():
#             for nucleus in nucleus_list:
#                 if nucleus.name == 'Proto': nucleus.synaptic_time_constant = {inhibitory_trans_1 : inhibitory_trans_1_val, inhibitory_trans_2 : inhibitory_trans_2_val}
#                 if nucleus.name == 'STN': nucleus.tau = {'Glut': glut} 
#         return nuclei_dict
#     data  = create_df(nuclei_dict, [len(GABA_A), len(GABA_B), len(Glut)], 3 ,len(t_list))    
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
#     data  = create_df(nuclei_dict, [len(inhibitory_series), len(Glut)], 2,len(t_list))    
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

#     data  = create_df(nuclei_dict, [len(inhibitory_series), len(Glut)], 2,t_list)    
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