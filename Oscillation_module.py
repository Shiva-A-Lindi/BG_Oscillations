from __future__ import division
import os
import sys
import subprocess
import timeit
import numpy as np
from numpy import inf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.fft import rfft, fft, fftfreq
from tempfile import TemporaryFile
import pickle
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import truncexpon, skewnorm
from scipy.signal import butter, sosfilt, sosfreqz, spectrogram
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


def extrapolate_FR_ext_from_neuronal_response_curve_high_act(FR_ext, FR_sim, desired_FR, if_plot=False, end_of_nonlinearity=None, maxfev=None, g_ext=0, N_ext=0, tau=0, ax=None, noise_var=0, c='grey'):
    ''' All firing rates in Hz'''

    slope, intercept = linear_regresstion(FR_ext, FR_sim)
    FR_ext_extrapolated = inverse_linear(desired_FR, slope, intercept)
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
        spike_thresh_bound_ratio=[1/20, 1/20], ext_input_integ_method='dirac_delta_input', path=None, mem_pot_init_method='uniform', plot_initial_V_m_dist=False, set_input_from_response_curve=True,
        set_random_seed=False, keep_mem_pot_all_t=False, save_init=False, scale_g_with_N=True):

        if set_random_seed:
            self.random_seed = 1996
            np.random.seed(self.random_seed)
        else:
            np.random.seed()

        n_timebins = int(t_sim/dt)
        self.n = N[name]  # population size
        self.population_num = population_number
        self.name = name
        self.basal_firing = A[name]
        self.mvt_firing = A_mvt[name]
        self.threshold = threshold[name]
        self.gain = gain[name]
        # filter based on the receiving nucleus# dictfilt(synaptic_time_constant, self.trans_types) # synaptic time scale based on neuron type
        self.synaptic_time_constant = {
            k: v for k, v in synaptic_time_constant.items() if k[1] == name}
        # filter based on the receiving nucleus
        self.transmission_delay = {k: v for k, v in T.items() if k[0] == name}
        self.ext_inp_delay = ext_inp_delay
        # filter based on the receiving nucleus
        self.synaptic_weight = {k: v for k, v in G.items() if k[0] == name}
        self.K_connections = None
        # stored history in ms derived from the longest transmission delay of the projections
        self.history_duration = max(self.transmission_delay.values())
        self.receiving_from_list = receiving_from_list[(
            self.name, str(self.population_num))]
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
        self.external_inp_t_series = np.zeros((n_timebins))
        self.t_sim = t_sim
        if neuronal_model == 'rate':
            self.output = {k: np.zeros(
                (self.n, int(T[k[0], self.name]/dt))) for k in self.sending_to_dict}
            self.input = np.zeros((self.n))
            self.neuron_act = np.zeros((self.n))
            # external input mimicing movement
            self.mvt_ext_input = np.zeros((n_timebins))
            # the basal firing as a result of noise at steady state
            self.noise_induced_basal_firing = None
            self.oscil_peak_threshold = oscil_peak_threshold[self.name]
            self.scale_g_with_N = scale_g_with_N
        if neuronal_model == 'spiking':
            self.spikes = np.zeros((self.n, int(t_sim/dt)), dtype=int)
            # dt incorporated in tau for efficiency
            # filter based on the receiving nucleus
            self.tau = {k: {kk: np.array(
                vv)/dt for kk, vv in tau[k].items()} for k, v in tau.items() if k[0] == name}
            # since every connection might have different rise and decay time, inputs must be updataed accordincg to where the input is coming from
            self.I_rise = {k: np.zeros((self.n, len(
                self.tau[self.name, k[0]]['decay']))) for k in self.receiving_from_list}
            self.I_syn = {k: np.zeros((self.n, len(
                self.tau[self.name, k[0]]['decay']))) for k in self.receiving_from_list}
            self.I_syn['ext_pop', '1'] = np.zeros(self.n,)
            self.I_rise['ext_pop', '1'] = np.zeros(self.n,)
            self.neuronal_consts = neuronal_consts[self.name]
            self.u_rest = self.neuronal_consts['u_rest']
            self.mem_pot_before_spike = np.zeros(self.n)
            self.syn_inputs = {k: np.zeros((self.n, 1))
                                           for k in self.receiving_from_list}
            # meant to store the poisson spike trains of the external population
            self.poisson_spikes = None
            # external population size
            self.n_ext_population = poisson_prop[self.name]['n']
            self.firing_of_ext_pop = poisson_prop[self.name]['firing']
            self.syn_weight_ext_pop = poisson_prop[self.name]['g']

            self.representative_inp = {k: np.zeros((n_timebins, len(
                self.tau[self.name, k[0]]['decay']))) for k in self.receiving_from_list}
            self.representative_inp['ext_pop', '1'] = np.zeros(n_timebins)
            self.ext_input_all = np.zeros((self.n, n_timebins))
            self.voltage_trace = np.zeros(n_timebins)
            self.AUC_of_input = AUC_of_input
            self.rest_ext_input = np.zeros(self.n)
            self.ext_inp_method = ext_inp_method
            # if derive external input value from response curve
            self.der_ext_I_from_curve = der_ext_I_from_curve
            self.sum_syn_inp_at_rest = None
            self.keep_mem_pot_all_t = keep_mem_pot_all_t
            self.save_init = save_init
            self.set_input_from_response_curve = set_input_from_response_curve
            if self.keep_mem_pot_all_t:
                self.all_mem_pot = np.zeros((self.n, n_timebins))
            self.ext_input_integ_method = ext_input_integ_method
            self.mem_pot_init_method = mem_pot_init_method
            self.init_method = init_method
            self.spike_thresh_bound_ratio = spike_thresh_bound_ratio
            self.bound_to_mean_ratio = bound_to_mean_ratio
            
            self.set_init_distribution( poisson_prop, dt, t_sim,  plot_initial_V_m_dist = plot_initial_V_m_dist)
            
            self.ext_inp_method_dict = {'Poisson': self.poissonian_ext_inp,
                'const+noise': self.constant_ext_input_with_noise, 'constant': self.constant_ext_input}
            self.input_integ_method_dict = {'exp_rise_and_decay': exp_rise_and_decay,
                'instantaneus_rise_expon_decay': instantaneus_rise_expon_decay, 'dirac_delta_input': _dirac_delta_input}
            self.syn_input_integ_method = syn_input_integ_method
            self.normalize_synaptic_weight()
            
    def set_init_distribution(self, poisson_prop, dt, t_sim,  plot_initial_V_m_dist = False):
        if self.init_method == 'homogeneous':
            self.initialize_homogeneously(poisson_prop, dt)
            self.FR_ext = 0
        elif self.init_method == 'heterogeneous':
            self.initialize_heterogeneously(poisson_prop, dt, t_sim, self.spike_thresh_bound_ratio,
                                            *self.bound_to_mean_ratio,  plot_initial_V_m_dist=plot_initial_V_m_dist)
            self.FR_ext = np.zeros(self.n)
            
    def initialize_heterogeneously(self, poisson_prop, dt, t_sim, spike_thresh_bound_ratio, lower_bound_perc=0.8, upper_bound_perc=1.2,
                                    plot_initial_V_m_dist=False):
        ''' cell properties and boundary conditions come from distributions'''
        self.spike_thresh = truncated_normal_distributed(self.neuronal_consts['spike_thresh']['mean'],
                                                            self.neuronal_consts['spike_thresh']['var'], self.n,
                                                            scale_bound=scale_bound_with_arbitrary_value, scale=(
                                                                self.neuronal_consts['spike_thresh']['mean'] - self.u_rest),
                                                            lower_bound_perc=spike_thresh_bound_ratio[0], upper_bound_perc=spike_thresh_bound_ratio[1])

        self.initialize_mem_potential(method=self.mem_pot_init_method)
        if self.keep_mem_pot_all_t:
            self.all_mem_pot[:, 0] = self.mem_potential.copy()
        if plot_initial_V_m_dist:
            self.plot_mem_potential_distribution_of_one_t(0, bins=50)

        if 'truncmin' in self.neuronal_consts['membrane_time_constant']:
            self.membrane_time_constant = truncated_normal_distributed(self.neuronal_consts['membrane_time_constant']['mean'],
                                                             self.neuronal_consts['membrane_time_constant']['var'], self.n,
                                                             truncmin=self.neuronal_consts['membrane_time_constant']['truncmin'],
                                                             truncmax=self.neuronal_consts['membrane_time_constant']['truncmax'])
        else:
            self.membrane_time_constant = truncated_normal_distributed(self.neuronal_consts['membrane_time_constant']['mean'],
                                                                 self.neuronal_consts['membrane_time_constant']['var'], self.n,
                                                                 lower_bound_perc=lower_bound_perc, upper_bound_perc=upper_bound_perc)

        # dt incorporated in tau for efficiency
        self.tau_ext_pop = {'rise': truncated_normal_distributed(poisson_prop[self.name]['tau']['rise']['mean'],
                                                                poisson_prop[self.name]['tau']['rise']['var'], self.n,
                                                                lower_bound_perc=lower_bound_perc, upper_bound_perc=upper_bound_perc) / dt,
                            'decay': truncated_normal_distributed(poisson_prop[self.name]['tau']['decay']['mean'],
                                                                poisson_prop[self.name]['tau']['decay']['var'], self.n,
                                                                lower_bound_perc=lower_bound_perc, upper_bound_perc=upper_bound_perc) / dt}

    def initialize_homogeneously(self, poisson_prop, dt, keep_mem_pot_all_t=False):
        ''' cell properties and boundary conditions are constant for all cells'''

        # self.mem_potential = np.random.uniform(low = self.neuronal_consts['u_initial']['min'], high = self.neuronal_consts['u_initial']['max'], size = self.n) # membrane potential
        self.spike_thresh = np.full(
            self.n, self.neuronal_consts['spike_thresh']['mean'])
        self.mem_potential = np.random.uniform(
            low=self.u_rest, high=self.spike_thresh, size=self.n)  # membrane potential
        if self.keep_mem_pot_all_t:
            self.all_mem_pot[:, 0] = self.mem_potential.copy()

        self.membrane_time_constant = np.full(self.n,
                                             self.neuronal_consts['membrane_time_constant']['mean'])
        self.tau_ext_pop = {'rise': np.full(self.n, poisson_prop[self.name]['tau']['rise']['mean'])/dt,  # synaptic decay time of the external pop inputs
                            'decay': np.full(self.n, poisson_prop[self.name]['tau']['decay']['mean'])/dt}

    def normalize_synaptic_weight(self):

        self.synaptic_weight = {k: v * (self.neuronal_consts['spike_thresh']['mean'] - self.u_rest)
                                        for k, v in self.synaptic_weight.items() if k[0] == self.name}

    def calculate_input_and_inst_act(self, t, dt, receiving_from_class_list, mvt_ext_inp):
        # to do

        syn_inputs = np.zeros((self.n, 1))  # = Sum (G Jxm)
        for projecting in receiving_from_class_list:

            syn_inputs += self.synaptic_weight[(self.name, projecting.name)]*np.matmul(self.connectivity_matrix[(projecting.name, str(projecting.population_num))],
                           projecting.output[(self.name, str(self.population_num))][:, -int(self.transmission_delay[(self.name, projecting.name)]/dt)].reshape(-1, 1))  # /self.K_connections[(self.name, projecting.name)]

        # + noise_generator(self.noise_amplitude, self.noise_variance, self.n)
        inputs = syn_inputs + self.rest_ext_input + mvt_ext_inp
        self.neuron_act = transfer_func(self.threshold, self.gain, inputs)
        self.pop_act[t] = np.average(self.neuron_act)

    def update_output(self, dt):

        new_output = {k: self.output[k][:, -
            1].reshape(-1, 1) for k in self.output.keys()}
        for key in self.sending_to_dict:
            for tau in self.synaptic_time_constant[(key[0], self.name)]:
                new_output[key] += dt * \
                    (-self.output[key][:, -1].reshape(-1, 1)+self.neuron_act)/tau
            self.output[key] = np.hstack((self.output[key][:, 1:], new_output[key]))

    def cal_ext_inp(self, dt, t):

        # choose method of exerting external input from dictionary of methods
        I_ext = self.ext_inp_method_dict[self.ext_inp_method](dt)

        self.I_syn['ext_pop', '1'], self.I_rise['ext_pop', '1'] = self.input_integ_method_dict[self. ext_input_integ_method](I_ext,
                                                                            I_rise=self.I_rise['ext_pop', '1'],
                                                                            I=self.I_syn['ext_pop', '1'],
                                                                            tau_rise=self.tau_ext_pop['rise'],
                                                                            tau_decay=self.tau_ext_pop['decay'])

    def poissonian_ext_inp(self, dt):

        # poisson_spikes = possion_spike_generator(self.n,self.n_ext_population,self.firing_of_ext_pop,dt)
        poisson_spikes = possion_spike_generator(
            self.n, self.n_ext_population, self.FR_ext, dt)
        I_ext = self.cal_input_from_poisson_spikes(poisson_spikes, dt)
        return I_ext

    def constant_ext_input_with_noise(self, dt):

        return self.rest_ext_input + noise_generator(self.noise_amplitude, self.noise_variance, self.n).reshape(-1,)

    def constant_ext_input(self, dt):

        return self.rest_ext_input

    def cal_input_from_poisson_spikes(self, poisson_spikes, dt):

        return (np.sum(poisson_spikes, axis=1) / dt  # normalize dirac delta spike amplitude
               * self.syn_weight_ext_pop
               * self.membrane_time_constant
               ).reshape(-1,)

    def cal_synaptic_input(self, dt, projecting, t):

        num = str(projecting.population_num)
        name = projecting.name
        self.syn_inputs[name, num] = (
                                    (self.synaptic_weight[(self.name, name)] *
                                    np.matmul(self.connectivity_matrix[(name, num)],
                                               projecting.spikes[:, int(
                                                   t - self.transmission_delay[(self.name, name)] / dt)]
                                            ) / dt
                                    * self.membrane_time_constant).reshape(-1,)
                                    )

    def sum_synaptic_input(self, receiving_from_class_list, dt, t):
        ''''''
        synaptic_inputs = np.zeros(self.n)
        for projecting in receiving_from_class_list:

            self.cal_synaptic_input(dt, projecting, t)
            synaptic_inputs = (synaptic_inputs +
                              self.sum_components_of_one_synapse(t, projecting.name, str(projecting.population_num),
                                                                pre_n_components=len(self.tau[self.name, projecting.name]['rise'])))

        return synaptic_inputs

    def sum_synaptic_input_one_step_ahead_with_no_spikes(self, receiving_from_class_list, dt):
        ''''''
        synaptic_inputs = np.zeros(self.n)
        for projecting in receiving_from_class_list:

            synaptic_inputs = (synaptic_inputs +
                                self.sum_components_of_one_synapse_one_step_ahead_with_no_spikes(projecting.name,
                                                                                                str(projecting.population_num),
                                                                                                pre_n_components=len(self.tau[self.name, projecting.name]['rise'])))

        return synaptic_inputs

    def sum_components_of_one_synapse(self, t, pre_name, pre_num, pre_n_components=1):

        i = 0
        sum_components = np.zeros(self.n)
        for i in range(pre_n_components):
            self.I_syn[pre_name, pre_num][:, i], self.I_rise[pre_name, pre_num][:, i] = self.input_integ_method_dict[self.syn_input_integ_method](self.syn_inputs[pre_name, pre_num],
                                                                                                                                               I_rise=self.I_rise[pre_name,
                                                                                                                                                   pre_num][:, i],
                                                                                                                                               I=self.I_syn[pre_name, pre_num][:, i],
                                                                                                                                               tau_rise=self.tau[(
                                                                                                                                                   self.name, pre_name)]['rise'][i],
                                                                                                                                                tau_decay=self.tau[(self.name, pre_name)]['decay'][i])
            self.representative_inp[pre_name, pre_num][t,
                i] = self.I_syn[pre_name, pre_num][0, i]
            sum_components = sum_components + self.I_syn[pre_name, pre_num][:, i]
            i += 1
        return sum_components

    def sum_components_of_one_synapse_one_step_ahead_with_no_spikes(self, pre_name, pre_num, pre_n_components=1):
        '''Calculate I_syn(t+dt) assuming that there are no spikes between time t and t+dt '''

        i = 0
        sum_components = np.zeros(self.n)
        for i in range(pre_n_components):

            I_syn_next_dt, _ = self.input_integ_method_dict[self.syn_input_integ_method](0,
                                                I_rise=self.I_rise[pre_name, pre_num][:, i],
                                                I=self.I_syn[pre_name, pre_num][:, i],
                                                tau_rise=self.tau[(self.name, pre_name)]['rise'][i],
                                                tau_decay=self.tau[(self.name, pre_name)]['decay'][i])

            sum_components = sum_components + I_syn_next_dt
            i += 1
        return sum_components

    def solve_IF_without_syn_input(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = np.zeros(self.n)
        self.update_potential(synaptic_inputs, dt, receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        # self.reset_potential(spiking_ind)
        self.reset_potential_with_interpolation(spiking_ind, dt)
        self.cal_population_activity(dt, t)

    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):

        self.cal_ext_inp(dt, t)
        synaptic_inputs = self.sum_synaptic_input(receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, receiving_from_class_list)
        spiking_ind = self.find_spikes(t)
        self.reset_potential(spiking_ind)
        # self.reset_potential_with_interpolation(spiking_ind,dt)
        self.cal_population_activity(dt, t)
        self.update_representative_measures(t)
        if self.keep_mem_pot_all_t:
            self.all_mem_pot[:, t] = self.mem_potential

    def update_representative_measures(self, t):

        self.voltage_trace[t] = self.mem_potential[0]
        self.representative_inp['ext_pop', '1'][t] = self.I_syn['ext_pop', '1'][0]
        self.ext_input_all[:, t] = self.I_syn['ext_pop', '1']

    def update_potential(self, synaptic_inputs, dt, receiving_from_class_list):

        # EIF
        # self.mem_potential += (-self.mem_potential+ inputs+ self.neuronal_consts['nonlin_sharpness'] *np.exp((self.mem_potential-
        #                       self.neuronal_consts['nonlin_thresh'])/self.neuronal_consts['nonlin_sharpness']))*dt/self.membrane_time_constant
        # LIF
        self.mem_pot_before_spike = self.mem_potential.copy()
        V_prime = f_LIF(self.membrane_time_constant, self.mem_potential,
                        self.u_rest, self.I_syn['ext_pop', '1'], synaptic_inputs)
        # self.mem_potential = fwd_Euler(dt, self.mem_potential, V_prime)
        I_syn_next_dt = self. sum_synaptic_input_one_step_ahead_with_no_spikes(
            receiving_from_class_list, dt)
        self.mem_potential = Runge_Kutta_second_order_LIF(
            dt, self.mem_potential, V_prime,  self.membrane_time_constant, I_syn_next_dt, self.u_rest, self.I_syn['ext_pop', '1'])

    def find_spikes(self, t):

        # spiking_ind = np.where(self.mem_potential > self.neuronal_consts['spike_thresh']['mean']) # homogeneous spike thresholds
        # gaussian distributed spike thresholds
        spiking_ind = np.where(self.mem_potential > self.spike_thresh)
        self.spikes[spiking_ind, t] = 1
        return spiking_ind

    def reset_potential(self, spiking_ind):

        self.mem_potential[spiking_ind] = self.neuronal_consts['u_rest']

    def reset_potential_with_interpolation(self, spiking_ind, dt):
        ''' set the potential at firing times according to Hansel et. al. (1998)'''
        self.mem_potential[spiking_ind] = linear_interpolation(self.mem_potential[spiking_ind], self.spike_thresh[spiking_ind],
                                                                dt, self.mem_pot_before_spike[spiking_ind], self.neuronal_consts['u_rest'], self.membrane_time_constant[spiking_ind])

    def cal_population_activity(self, dt, t, lower_bound_perc=0.8, upper_bound_perc=1.2):

        self.pop_act[t] = np.average(self.spikes[:, t], axis=0)/(dt/1000)

    def reset_ext_pop_properties(self, poisson_prop, dt):
        '''reset the properties of the external poisson spiking population'''

        # external population size
        self.n_ext_population = poisson_prop[self.name]['n']
        self.firing_of_ext_pop = poisson_prop[self.name]['firing']

        self.tau_ext_pop = {'rise': truncated_normal_distributed(poisson_prop[self.name]['tau']['rise']['mean'],
                                                                poisson_prop[self.name]['tau']['rise']['var'], self.n,
                                                                lower_bound_perc=lower_bound_perc, upper_bound_perc=upper_bound_perc) / dt,
                            'decay': truncated_normal_distributed(poisson_prop[self.name]['tau']['decay']['mean'],
                                                                poisson_prop[self.name]['tau']['decay']['var'], self.n,
                                                                lower_bound_perc=lower_bound_perc, upper_bound_perc=upper_bound_perc) / dt}
        self.syn_weight_ext_pop = poisson_prop[self.name]['g']
        # self.representative_inp['ext_pop','1'] = np.zeros((int(t_sim/dt)))

    def set_noise_param(self, noise_variance, noise_amplitude):

        # additive gaussian white noise with mean zero and variance sigma
        self.noise_variance = noise_variance[self.name]
        # additive gaussian white noise with mean zero and variance sigma
        self.noise_amplitude = noise_amplitude[self.name]

    def set_connections(self, K, N):
        ''' creat Jij connection matrix'''

        self.K_connections = {k: v for k, v in K.items() if k[0] == self.name}
        for projecting in self.receiving_from_list:

            n_connections = self.K_connections[(self.name, projecting[0])]
            self.connectivity_matrix[projecting] = build_connection_matrix(
                self.n, N[projecting[0]], n_connections)

    def initialize_mem_potential(self, method='uniform'):
        np.random.seed()
        if method not in ['uniform', 'constant', 'exponential', 'draw_from_data']:
            raise ValueError(
                " method must be either 'uniform', 'constant', 'exponential', or 'draw_from_data' ")

        if method == 'uniform':

            self.mem_potential = np.random.uniform(
                low=self.neuronal_consts['u_initial']['min'], high=self.neuronal_consts['u_initial']['max'], size=self.n)

        elif method == 'draw_from_data':

            data = np.load(os.path.join(self.path, 'all_mem_pot_' + self.name + '_tau_' + str(np.round(
                self.neuronal_consts['membrane_time_constant']['mean'], 1)).replace('.', '-') + '.npy'))
            y_dist = data.reshape(int(data.shape[0] * data.shape[1]), 1)
            self.mem_potential = draw_random_from_data_pdf(y_dist, self.n, bins=20)

        elif method == 'constant':
              self.mem_potential = np.full(self.n, self.neuronal_consts['u_rest'])

        elif method == 'exponential':  # Doesn't work with linear interpolation of IF, diverges
            lower, upper, scale = 0, self.neuronal_consts['spike_thresh']['mean'] - \
                self.u_rest, 30
            X = stats.truncexpon(b=(upper-lower) / scale, loc=lower, scale=scale)
            self.mem_potential = self.neuronal_consts['spike_thresh']['mean'] - X.rvs(
                self.n)

    def normalize_synaptic_weight_by_N(self):
        self.synaptic_weight = {
            k: v / self.K_connections[k] for k, v in self.synaptic_weight.items()}

    def clear_history(self, mem_pot_init_method=None):

        self.pop_act[:] = 0
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

            self.I_syn['ext_pop', '1'][:] = 0
            self.voltage_trace[:] = 0
            self.representative_inp['ext_pop', '1'][:] = 0

            if mem_pot_init_method == None:  # if not specified initialize as before
                mem_pot_init_method = self.mem_pot_init_method

            self.initialize_mem_potential(method=mem_pot_init_method)
            self.spikes[:, :] = 0

    def smooth_pop_activity(self, dt, window_ms=5):
        self.pop_act = moving_average_array(self.pop_act, int(window_ms / dt))

    def average_pop_activity(self, t_list, last_fraction=1/2):
        average = np.average(self.pop_act[int(len(t_list) * last_fraction):])
        std = np.std(self.pop_act[int(len(t_list) * last_fraction):])
        return average, std

    def set_synaptic_weights(self, G):

        # filter based on the receiving nucleus
        self.synaptic_weight = {k: v for k, v in G.items() if k[0] == self.name}

    def set_synaptic_time_scales(self, synaptic_time_constant):

        self.synaptic_time_constant = {
            k: v for k, v in synaptic_time_constant.items() if k[1] == self.name}

    def incoming_rest_I_syn(self, proj_list, A):

        I_syn = np.sum([self.synaptic_weight[self.name, proj] * A[proj] / 1000 * self.K_connections[self.name, proj]
                       * len(self.tau[self.name, proj]['rise']) for proj in proj_list])*self.membrane_time_constant

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
            I_syn = self.incoming_rest_I_syn(proj_list, A)

            # print('I_syn', np.average(I_syn))
            self.sum_syn_inp_at_rest = I_syn

            if self.ext_inp_method == 'Poisson':
                self._set_ext_inp_poisson(I_syn)

            elif self.ext_inp_method == 'const+noise' or self.ext_inp_method == 'const':
                self._set_ext_inp_const_plus_noise(I_syn, end_of_nonlinearity)
            else:
                raise ValueError('external input handling method not right!')
            self._save_init()
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

            # non array
            # exp = np.exp(-1/(self.neuronal_consts['membrane_time_constant']['mean']*self.basal_firing/1000))
            # self.rest_ext_input = ((self.neuronal_consts['spike_thresh']['mean'] - self.neuronal_consts['u_rest'])/ (1-exp))/(self.syn_weight_ext_pop*self.n_ext_population*self.neuronal_consts['membrane_time_constant']['mean'])
            # print(self.name)
            # print(decimal.Decimal.from_float(self.rest_ext_input[0]))
            # print(decimal.Decimal.from_float(temp[0]))#, 'ext_inp=',self.rest_ext_input)

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

    def _set_ext_inp_const_plus_noise(self, I_syn, end_of_nonlinearity):
        # linear regime if decided not to derive from reponse curve (works for low noise levels)
        if self.basal_firing > end_of_nonlinearity and not self.set_input_from_response_curve:
            self._set_ext_inp_poisson(I_syn)
        else:
            self.rest_ext_input = self.FR_ext * self.syn_weight_ext_pop * \
                self.n_ext_population * self.membrane_time_constant - I_syn
                
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
            FR_mean_start = FR_ext_of_given_FR_theory(self.neuronal_consts['spike_thresh']['mean'], self.u_rest,
                                                      self.neuronal_consts['membrane_time_constant']['mean'], self.syn_weight_ext_pop, FR_range[0], self.n_ext_population)
            FR_mean_end = FR_ext_of_given_FR_theory(self.neuronal_consts['spike_thresh']['mean'], self.u_rest,
                                                    self.neuronal_consts['membrane_time_constant']['mean'], self.syn_weight_ext_pop, FR_range[-1], self.n_ext_population)
            FR_start = (FR_ext_of_given_FR_theory(self.spike_thresh, self.u_rest, self.membrane_time_constant, self.syn_weight_ext_pop, FR_range[0], self.n_ext_population)
                        / FR_mean_start * FR_range[0])
            FR_end = (FR_ext_of_given_FR_theory(self.spike_thresh, self.u_rest, self.membrane_time_constant, self.syn_weight_ext_pop, FR_range[-1], self.n_ext_population)
                        / FR_mean_end * FR_range[-1])
        # else:
            # FR_start = FR_range[0]
            # FR_end = FR_range[-1]
        FR_list = np.linspace(*FR_range, n_FR)
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
            self.rest_ext_input = FR_list[i, :] * self.membrane_time_constant * \
                self.n_ext_population * self.syn_weight_ext_pop
            self. run(dt, t_list, receiving_class_dict)
            FR_sim[:, i] = np.average(
                self.spikes[:, int(len(t_list)/2):], axis=1)/(dt/1000)
            print('FR = ', np.average(FR_list[i, :]), np.average(FR_sim[:, i]))

        return FR_sim

    def plot_mem_potential_distribution_of_one_t(self, t,  ax=None, bins=50, color='gray'):
        fig, ax = get_axes(ax)
        ax.hist(self.all_mem_pot[:, t] / self.n, bins=bins,
                color=color, label=self.name, density=True, stacked=True)
        ax.set_xlabel('Membrane potential (mV)', fontsize=15)
        ax.set_ylabel(r'$Probability\ density$', fontsize=15)
        # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
        ax.legen(fontsize=15)

    def plot_mem_potential_distribution_of_all_t(self, ax=None, bins=50, color='grey'):
        a = self.all_mem_pot.copy()
        fig, ax = get_axes(ax)
        ax.hist(a.reshape(int(a.shape[0] * a.shape[1]), 1), bins=bins,
                color=color, label=self.name, density=True, stacked=True)
        ax.set_xlabel('Membrane potential (mV)', fontsize=15)
        ax.set_ylabel(r'$Probability$', fontsize=15)
        # ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
        ax.legend(fontsize=15)

    def Find_threshold_of_firing(self, FR_list, t_list, dt, receiving_class_dict):
        FR_sim = np.zeros((self.n, len(FR_list)))

        for i in range(FR_list.shape[0]):

            self.clear_history()
            self.rest_ext_input = FR_list[i, :] * self.membrane_time_constant * \
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


    def scale_synaptic_weight(self):
        if self.scale_g_with_N:
            self.synaptic_weight = {
                k: v/self.K_connections[k] for k, v in self.synaptic_weight.items() if k[0] == self.name}

    def find_freq_of_pop_act_spec_window(self, start, end, dt, peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False,
                                      method='zero_crossing', plot_sig=False, plot_spectrum=False, ax=None, c_spec='navy', fft_label='fft',
                                      spec_figsize=(6, 5), find_beta_band_power=False, fft_method='rfft', n_windows=6, include_beta_band_in_legend=False,
                                      divide_beta_band_in_power = False):
        ''' trim the beginning and end of the population activity of the nucleus if necessary, cut
            the plateau and in case it is oscillation determine the frequency '''
        if method not in ["fft", "zero_crossing"]:
            raise ValueError("method must be either 'fft', or 'zero_crossing'")

        sig = trim_start_end_sig_rm_offset(
            self.pop_act, start, end, method=self.trim_sig_method_dict[self.neuronal_model])
        cut_sig_ind = cut_plateau(sig,  epsilon=cut_plateau_epsilon)
        plateau_y = find_mean_of_signal(sig, cut_sig_ind)
        _plot_signal(plot_sig, start, end, dt, sig, plateau_y, cut_sig_ind)
        if_stable = False
        if len(cut_sig_ind) > 0:  # if it's not all plateau from the beginning

            sig = sig - plateau_y

            if method == 'zero_crossing':

                n_half_cycles, freq = zero_crossing_freq_detect(
                    sig[cut_sig_ind], dt / 1000)

            elif method == 'fft':

                f, pxx, freq = freq_from_fft(sig[cut_sig_ind], dt / 1000, plot_spectrum=plot_spectrum, ax=ax, c=c_spec, label=fft_label, figsize=spec_figsize,
                                             method=fft_method, n_windows=n_windows, include_beta_band_in_legend=include_beta_band_in_legend)
                if find_beta_band_power:
                    if divide_beta_band_in_power:
                        
                        low_beta_band_power = beta_bandpower(f, pxx, fmin = 12, fmax = 20)
                        high_beta_band_power = beta_bandpower(f, pxx, fmin = 20, fmax = 30)
                    else:
                        beta_band_power = beta_bandpower(f, pxx)
                else: beta_band_power = None
                n_half_cycles = None

            if freq != 0:  # then check if there's oscillations

                perc_oscil = max_non_empty_array(cut_sig_ind) / len(sig) * 100

                if check_stability:
                    if_stable = if_stable_oscillatory(sig, max(
                        cut_sig_ind), peak_threshold, smooth_kern_window, amp_env_slope_thresh=- 0.05)
                if divide_beta_band_in_power:
                    return n_half_cycles, perc_oscil, freq, if_stable, [low_beta_band_power, high_beta_band_power], f, pxx
                else:
                    return n_half_cycles, perc_oscil, freq, if_stable, beta_band_power, f, pxx

            else:
                print("Freq = 0")
                return 0, 0, 0, False, None, f, pxx

        else:
            print("all plateau")
            return 0, 0, 0, False, None, [], []

    def low_pass_filter(self, dt, low, high, order=6):
        self.pop_act = butter_bandpass_filter(
            self.pop_act, low, high, 1 / (dt / 1000), order=order)

    def additive_ext_input(self, ad_ext_inp):
        """ to add a certain external input to all neurons of the neucleus."""
        self.rest_ext_input = self.rest_ext_input + ad_ext_inp


def set_connec_ext_inp(A, A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, receiving_pop_list, nuclei_dict, t_list, c='grey', scale_g_with_N=True,
                        all_FR_list=np.linspace(0.05, 0.07, 100), n_FR=50, if_plot=False, end_of_nonlinearity=25, left_pad=0.005,
                        right_pad=0.005, maxfev=5000, ax=None, set_FR_range_from_theory=True, method = 'single_neuron', FR_ext_all_nuclei_saved = None,
                        use_saved_FR_ext = False, return_saved_FR_ext = False):
    '''find number of connections and build J matrix, set ext inputs as well'''
    # K = calculate_number_of_connections(N,N_real,K_real)
    K = calculate_number_of_connections(N, N_real, K_real)
    receiving_class_dict = create_receiving_class_dict(
        receiving_pop_list, nuclei_dict)
    FR_ext_all_nuclei = {}
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:

            nucleus.set_connections(K, N)
            
            if nucleus.neuronal_model == 'rate' and scale_g_with_N:
                
                nucleus.scale_synaptic_weight()

            elif nucleus. der_ext_I_from_curve:
                
                if method == 'collective' and not use_saved_FR_ext:
                    print("external input is being set collectively for {}...".format(nucleus.name))
                    FR_ext = nucleus.set_ext_inp_const_plus_noise_collective(all_FR_list[nucleus.name], t_list, dt, receiving_class_dict,
                                                                    if_plot = if_plot, end_of_nonlinearity = end_of_nonlinearity,
                                                                    maxfev = maxfev, c=c, n_FR=n_FR)
                    FR_ext_all_nuclei[nucleus.name] = FR_ext
                elif method == 'single_neuron':
                    if nucleus.basal_firing > end_of_nonlinearity:
                        nucleus.estimate_needed_external_input_high_act(all_FR_list[nucleus.name], dt, t_list, receiving_class_dict, if_plot=if_plot,
                                                                        n_FR=n_FR, ax=ax, c=c, set_FR_range_from_theory=set_FR_range_from_theory)
                        
                    else:
                        nucleus.estimate_needed_external_input(all_FR_list[nucleus.name], dt, t_list, receiving_class_dict, if_plot=if_plot, 
                                                               end_of_nonlinearity=end_of_nonlinearity, maxfev=maxfev,
                                                               n_FR=n_FR, left_pad=left_pad, right_pad=right_pad, ax=ax, c=c)
                elif use_saved_FR_ext:
                    nucleus.FR_ext = FR_ext_all_nuclei_saved[nucleus.name]
            nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list,
                                  dt, end_of_nonlinearity=end_of_nonlinearity)
    if return_saved_FR_ext:
        return receiving_class_dict, FR_ext_all_nuclei
    else:
        return receiving_class_dict

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


def reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A, A_mvt, D_mvt, t_mvt, t_list, dt, poisson_prop = None,
                            mem_pot_init_method=None, set_noise=True, end_of_nonlinearity=25, reset_init_dist = False, t_sim = None):
    

    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.clear_history(mem_pot_init_method=mem_pot_init_method)
            nucleus.set_synaptic_weights(G)
            nucleus.normalize_synaptic_weight()
            
            if set_noise:
                nucleus.set_noise_param(noise_variance, noise_amplitude)
            if reset_init_dist:
                nucleus.set_init_distribution( poisson_prop, dt, t_sim,  plot_initial_V_m_dist = False)
            nucleus.set_ext_input(A, A_mvt, D_mvt, t_mvt, t_list,
                                  dt, end_of_nonlinearity=end_of_nonlinearity)
    return nuclei_dict


def bandpower(f, pxx, fmin, fmax):
    ''' return the average power at the given range of frequency'''
    ind_min = scipy.argmin(f[f > fmin])
    ind_max = scipy.argmax(f[f < fmax])

    return scipy.trapz(pxx[ind_min: ind_max], f[ind_min: ind_max]) / (f[ind_max] - f[ind_min])


def beta_bandpower(f, pxx, fmin=13, fmax=30):
    return bandpower(f, pxx, fmin=fmin, fmax=fmax)


def find_sending_pop_dict(receiving_pop_list):

    sending_pop_list = {k: [] for k in receiving_pop_list.keys()}
    for k, v_list in receiving_pop_list.items():
        for v in v_list:
            sending_pop_list[v].append(k)
    return sending_pop_list


def get_max_len_dict(dictionary):
    ''' return maximum length between items of a dictionary'''
    return max(len(v) for k, v in dictionary.items())


def synaptic_weight_exploration_SNN(nuclei_dict, filepath, duration_base, G_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, receiving_class_dict, noise_amplitude, noise_variance,
    peak_threshold=0.1, smooth_kern_window=3, cut_plateau_epsilon=0.1, check_stability=False, freq_method='fft', plot_sig=False, n_run=1,
    lim_oscil_perc=10, plot_firing=False, smooth_window_ms=5, low_pass_filter=False, lower_freq_cut=1, upper_freq_cut=2000, set_seed=False, firing_ylim=[0, 80],
    plot_spectrum=False, spec_figsize=(6, 5), plot_raster=False, plot_start=0, plot_start_raster=0, plot_end=None, find_beta_band_power=False, n_windows=6, fft_method='rfft',
    include_beta_band_in_legend=True, n_neuron=None, save_pkl=False, include_std=True, round_dec=2, legend_loc='upper right', display='sci', decimal=0,
    reset_init_dist = False, all_FR_list = None , n_FR =  20, if_plot = False, end_of_nonlinearity = 25, state = 'rest', K_real = None, N_real = None, N = None,
    receiving_pop_list = None, poisson_prop = None, use_saved_FR_ext= False, FR_ext_all_nuclei_saved = {}, return_saved_FR_ext= False, divide_beta_band_in_power= False):

    if set_seed:
        np.random.seed(1956)
    else:
        np.random.seed()
    nn = 200
    n_iter = get_max_len_dict(G_dict)
    print("n_inter = ", n_iter)
    data = {}
    for nucleus_list in nuclei_dict.values():
        nucleus = nucleus_list[0]  # get only on class from each population
        # data[(nucleus.name, 'mvt_freq')] = np.zeros((n,m))
        data[(nucleus.name, 'base_freq')] = np.zeros((n_iter, n_run))
        # data[(nucleus.name, 'perc_t_oscil_mvt')] = np.zeros((n,m))
        data[(nucleus.name, 'perc_t_oscil_base')] = np.zeros((n_iter, n_run))
        # data[(nucleus.name, 'n_half_cycles_mvt')] = np.zeros((n,m))
        data[(nucleus.name, 'n_half_cycles_base')] = np.zeros((n_iter, n_run))
        if divide_beta_band_in_power:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run, 2))
        else:
            data[(nucleus.name, 'base_beta_power')] = np.zeros((n_iter, n_run))
        data[(nucleus.name, 'f')] = np.empty((n_iter, n_run, 0))
        data[(nucleus.name, 'pxx')] = np.empty((n_iter, n_run, 0))

    data['g'] = G_dict
    count = 0
    G = dict.fromkeys(G_dict.keys(), None)

    if plot_firing:
        fig = plt.figure()

    if n_run > 1:  # don't plot all the runs
        plot_spectrum = False
    if plot_spectrum:
        fig_spec = plt.figure()

    if plot_raster:
        fig_raster = plt.figure()
        outer = gridspec.GridSpec(n_iter, 1, wspace=0.2, hspace=0.2)
    for i in range(n_iter):
        start = timeit.default_timer()
        for k, values in G_dict.items():
            G[k] = values[i]
            print(k, values[i])

        if plot_spectrum:
            ax_spec = fig_spec.add_subplot(n_iter, 1, count+1)

        else: ax_spec = None

        title = _get_title(G_dict, i, display=display, decimal=decimal)

        for j in range(n_run):

            nuclei_dict = reinitialize_nuclei_SNN(nuclei_dict, G, noise_amplitude, noise_variance, A,
                                                  A_mvt, D_mvt, t_mvt, t_list, dt, set_noise=False, 
                                                  reset_init_dist= reset_init_dist, poisson_prop = poisson_prop)  # , mem_pot_init_method = 'uniform')
            if reset_init_dist:
                receiving_class_dict = set_connec_ext_inp(A, A_mvt,D_mvt,t_mvt,dt, N, N_real, K_real, receiving_pop_list, nuclei_dict,t_list, 
                                                          all_FR_list = all_FR_list , n_FR =n_FR, if_plot = if_plot, 
                                                          end_of_nonlinearity = end_of_nonlinearity[nucleus.name][state], 
                                                          set_FR_range_from_theory = False, method = 'collective', use_saved_FR_ext= use_saved_FR_ext,
                                                          FR_ext_all_nuclei_saved = FR_ext_all_nuclei_saved, return_saved_FR_ext= return_saved_FR_ext)
                
            nuclei_dict = run(receiving_class_dict, t_list, dt, nuclei_dict)

            if plot_raster:
                fig_raster = raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=outer[i], title=title, fig=fig_raster, plot_start=plot_start_raster,
                                                    plot_end=plot_end, labelsize=10, title_fontsize=15, lw=1.8, linelengths=1, n_neuron=n_neuron)

            data = find_freq_SNN(data, i, j, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon,
                                check_stability, freq_method, plot_sig, low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=plot_spectrum, ax=ax_spec,
                                c_spec=color_dict, spec_figsize=spec_figsize, n_windows=n_windows, fft_method=fft_method, find_beta_band_power=find_beta_band_power,
                                include_beta_band_in_legend=include_beta_band_in_legend, divide_beta_band_in_power = divide_beta_band_in_power)

        if plot_spectrum:
            if fft_method == 'rfft':
                x_l = 10**9

            else:
                x_l = 8
                ax_spec.axhline(x_l, ls='--', c='grey')

            # ax_spec.set_title(title, fontsize = 18)
            ax_spec.legend(fontsize=11, loc='upper center',
                           framealpha=0.1, frameon=False)
            ax_spec.set_xlim(5, 55)
            rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax_spec)

        if plot_firing:
            ax = fig.add_subplot(n_iter, 1, count+1)
            plot(nuclei_dict, color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, ax, title, include_std=include_std, round_dec=round_dec, legend_loc=legend_loc,
                n_subplots=int(n_iter), plt_txt='horizontal', plt_mvt=False, plt_freq=True, plot_start=plot_start, plot_end=plot_end, ylim=firing_ylim)
            ax.legend(fontsize=13, loc=legend_loc, framealpha=0.1, frameon=False)
            # ax.set_ylim(*plt_ylim)
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
    if save_pkl:
        pickle_obj(data, filepath)
    return figs, title, data

def find_freq_SNN(data, i, j, dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, check_stability, freq_method, plot_sig,
                low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=False, ax=None, c_spec='navy', spec_figsize=(6, 5), find_beta_band_power=False,
                fft_method='rfft', n_windows=6, include_beta_band_in_legend=True, divide_beta_band_in_power = False):

    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:

            nucleus.smooth_pop_activity(dt, window_ms=smooth_window_ms)

            if low_pass_filter:
                nucleus.low_pass_filter(dt, lower_freq_cut, upper_freq_cut, order=6)

            (data[(nucleus.name, 'n_half_cycles_base')][i, j],
            data[(nucleus.name, 'perc_t_oscil_base')][i, j],
            data[(nucleus.name, 'base_freq')][i, j],
            if_stable_base,
            data[(nucleus.name, 'base_beta_power')][i, j],
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
                                                                    include_beta_band_in_legend=include_beta_band_in_legend)
            # try:
            #     data[(nucleus.name, 'f')][i, j, :], data[(nucleus.name, 'pxx')][i, j, :] = f, pxx
            # except ValueError:
            #     data[(nucleus.name, 'f')][i, j, :len(f)], data[(nucleus.name, 'pxx')][i, j, :len(pxx)] = f, pxx
            #     data[(nucleus.name, 'f')][i, j, :len(f)], data[(nucleus.name, 'pxx')][i, j, len(pxx):] = np.nan
                
            nucleus.frequency_basal = data[(nucleus.name, 'base_freq')][i, j]

            print(nucleus.name, 'f = ', round(data[(nucleus.name, 'base_freq')][i, j], 2), 'beta_p =', data[(
                nucleus.name, 'base_beta_power')][i, j])

    return data


def find_freq_SNN_not_saving(dt, nuclei_dict, duration_base, lim_oscil_perc, peak_threshold, smooth_kern_window, smooth_window_ms, cut_plateau_epsilon, check_stability, freq_method, plot_sig,
                low_pass_filter, lower_freq_cut, upper_freq_cut, plot_spectrum=False, ax=None, c_spec='navy', spec_figsize=(6, 5), find_beta_band_power=False,
                fft_method='rfft', n_windows=6, include_beta_band_in_legend=True):

    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:

            nucleus.smooth_pop_activity(dt, window_ms=smooth_window_ms)

            if low_pass_filter:
                nucleus.low_pass_filter(dt, lower_freq_cut, upper_freq_cut, order=6)

            nucleus.find_freq_of_pop_act_spec_window(*duration_base, dt,
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
                                                                    include_beta_band_in_legend=include_beta_band_in_legend)




def rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    if count+1 < n_iter:
        ax.axes.xaxis.set_ticklabels([])


# def _get_title(G_dict, i, display='normal', decimal=0):
#     if display == 'normal':

#         title = (r"$G_{"+list(G_dict.keys())[0][0]+"-"+list(G_dict.keys())[0][1]+"}$ = " + str(round(list(G_dict.values())[0][i], 2)) +
#                 r"  $G_{"+list(G_dict.keys())[1][0]+"-"+list(G_dict.keys())[1][1]+"}$ ="+str(round(list(G_dict.values())[1][i], 2)) +
#                 r"  $G_{"+list(G_dict.keys())[2][0]+"-"+list(G_dict.keys())[2][1]+"}$ ="+str(round(list(G_dict.values())[2][i], 2)))
#     if display == 'sci':
#         title = (r"$G_{"+list(G_dict.keys())[0][0]+"-"+list(G_dict.keys())[0][1]+"}$ = " + r"${0:s}$".format(as_si(list(G_dict.values())[0][i], decimal)) +
#                 r"  $G_{"+list(G_dict.keys())[1][0]+"-"+list(G_dict.keys())[1][1]+"}$ ="+r"${0:s}$".format(as_si(list(G_dict.values())[1][i], decimal)) +
#                 r"  $G_{"+list(G_dict.keys())[2][0]+"-"+list(G_dict.keys())[2][1]+"}$ ="+r"${0:s}$".format(as_si(list(G_dict.values())[2][i], decimal)))
#     return title

def _get_title(G_dict, i, display='normal', decimal=0):

    title = ''
    for j in range(len(G_dict)):
        if display == 'normal':
            title += r"$G_{"+list(G_dict.keys())[j][0]+"-"+list(G_dict.keys())[j][1]+"}$ = " + str(round(list(G_dict.values())[j][i], 2)) + ' '
        
        elif display == 'sci':
            title += r"$G_{"+list(G_dict.keys())[j][0]+"-"+list(G_dict.keys())[j][1]+"}$ = " + r"${0:s}$".format(as_si(list(G_dict.values())[j][i], decimal)) + ' '
        if (j+1) % 3 == 0:
            title += ' \n '  
    return title

            
def remove_frame(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict):
    fig, ax = plt.subplots()
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
    ax.plot(xdata * FR_to_I_coef, ydata, 'o', label=r'$\sigma =$' + str(noise_var),
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
    ax.plot(x * FR_to_I_coef, y, 'o', label=r'$\sigma =$' + str(noise_var),
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


def truncated_normal_distributed(mean, sigma, n, scale_bound=scale_bound_with_mean, scale=None, lower_bound_perc=0.8, upper_bound_perc=1.2, truncmin=None, truncmax=None):

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
                FR_mean, FR_std = nucleus. average_pop_activity(t_list, last_fraction=1/2)
                firing_prop[nucleus.name]['firing_mean'][i,
                    nucleus.population_num-1] = FR_mean
                firing_prop[nucleus.name]['firing_var'][i,
                    nucleus.population_num-1] = FR_std
                print(nucleus.name, np.round(np.average(nucleus.FR_ext), 3),
                    'FR=', FR_mean, 'std=', round(FR_std, 2))
        i += 1
    return firing_prop


def smooth_pop_activity_all_nuclei(nuclei_dict, dt, window_ms=5):
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            nucleus.smooth_pop_activity(dt, window_ms=window_ms)


def instantaneus_rise_expon_decay(inputs, I=0, I_rise=None, tau_decay=5, tau_rise=None):

    # dt incorporated in tau
    return I + (-I + inputs) / tau_decay, np.zeros_like(I)


def _dirac_delta_input(inputs, I_rise=None, I=None, tau_rise=None, tau_decay=None):

    return inputs, np.zeros_like(inputs)


def exp_rise_and_decay(inputs, I_rise=0, I=0, tau_rise=5, tau_decay=5):

    I_rise = I_rise + (-I_rise + inputs) / tau_rise  # dt incorporated in tau
    I = I + (-I + I_rise) / tau_decay  # dt incorporated in tau
    return I, I_rise


def fwd_Euler(dt, y, f):

    return y + dt * f


def f_LIF(tau, V, V_rest, I_ext, I_syn):
    ''' return dV/dt value for Leaky-integrate and fire neurons'''

    return (-(V - V_rest) + I_ext + I_syn) / tau


def save_all_mem_potential(nuclei_dict, path):
    for nucleus_list in nuclei_dict.values():
        for nucleus in nucleus_list:
            np.save(os.path.join(path, 'all_mem_pot_' + nucleus.name + '_tau_' + str(np.round(
                nucleus.neuronal_consts['membrane_time_constant']['mean'], 1)).replace('.', '-')), nucleus.all_mem_pot)


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


def Runge_Kutta_second_order_LIF(dt, V_t, f_t, tau, I_syn_next_dt, V_rest, I_ext):
    ''' Solve second order Runge-Kutta for a LIF at time t+dt (Mascagni & Sherman, 1997)'''
    # print(np.isnan(np.sum(V_t)), np.isnan(np.sum(f_t)), np.isnan(np.sum(V_rest)), np.isnan(np.sum(I_syn_next_dt)), np.isnan(np.sum(I_ext)), np.isnan(np.sum(tau)))
    V_next_dt = V_t + dt/2 * (-(V_t + dt * f_t - V_rest) +
                              I_syn_next_dt + I_ext) / tau + f_t * dt / 2
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


def get_axes(ax, figsize=(6, 5)):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plt.gcf(), ax


def raster_plot(spikes_sparse, name, color_dict, color='k',  ax=None, labelsize=10, title_fontsize=15, linelengths=2.5, lw=3, xlim=None,
                axvspan=False, span_start=None, span_end=None, axvspan_color='lightskyblue'):
    fig, ax = get_axes(ax)
    c_dict = color_dict.copy()
    c_to_ch = {v: k for k, v in c_dict.items()}['grey']
    c_dict[c_to_ch] = 'k'
    ax.eventplot(spikes_sparse, colors=c_dict[name],
                 linelengths=linelengths, lw=lw, orientation='horizontal')
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_title(name, c=color_dict[name], fontsize=title_fontsize)
    if axvspan:
        ax.axvspan(span_start, span_end, alpha=0.2, color=axvspan_color)
    remove_frame(ax)
    ax_label_adjust(ax, fontsize=labelsize, nbins=4)
    if xlim != None:
        ax.set_xlim(xlim)
    ax.legend(loc='upper right', framealpha=0.1, frameon=False)
    return ax


def raster_plot_all_nuclei(nuclei_dict, color_dict, dt, outer=None, fig=None,  title='', plot_start=0, plot_end=None, tick_label_fontsize=18,
                            labelsize=15, title_fontsize=15, lw=1, linelengths=1, n_neuron=None, include_title=True, set_xlim=True,
                            axvspan=False, span_start=None, span_end=None, axvspan_color='lightskyblue', ax_label=False):
    if outer == None:
        fig = plt.figure(figsize=(10, 8))
        outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)[0]

    inner = gridspec.GridSpecFromSubplotSpec(len(nuclei_dict), 1,
                    subplot_spec=outer, wspace=0.1, hspace=0.1)
    j = 0
    if include_title:
        ax = plt.Subplot(fig, outer)
        ax.set_title(title, fontsize=15)
        ax.axis('off')
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:

            if plot_end == None:
                plot_end = len(nucleus.pop_act)
            ax = plt.Subplot(fig, inner[j])
            if n_neuron == None:
                n_neuron = nucleus.n
            neurons = np.random.choice(nucleus.n, n_neuron, replace=False)
            spikes_sparse = create_sparse_matrix(nucleus.spikes[neurons, :], end=(
                plot_end / dt), start=(plot_start / dt)) * dt
            if set_xlim:
                xlim = [plot_start, plot_end]
            else: xlim = None
            ax = raster_plot(spikes_sparse, nucleus.name, color_dict,  ax=ax, labelsize=tick_label_fontsize, title_fontsize=title_fontsize, linelengths=linelengths, lw=lw, xlim=xlim,
                            axvspan=axvspan, span_start=span_start, span_end=span_end, axvspan_color=axvspan_color)
            fig.add_subplot(ax)

            rm_ax_unnecessary_labels_in_subplots(j, len(nuclei_dict), ax)
            j += 1
    if ax_label:
        fig.text(0.5, 0.03, 'time (ms)', ha='center',
                 va='center', fontsize=labelsize)
        fig.text(0.03, 0.5, 'neuron', ha='center', va='center',
                 rotation='vertical', fontsize=labelsize)
    return fig


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


def spacing_with_high_resolution_in_the_middle(n_points, start, end):
	'''return a series with lower spacing and higher resolution in the middle'''

	R = (start - end) / 2
	x = R * np.linspace(-1, 1, n_points)
	y = np.sqrt(R ** 2 - x ** 2)
	half_y = y[: len(y) // 2]
	diff = - np.diff(np.flip(half_y))
	series = np.concatenate((half_y, np.cumsum(diff) + y[len(y) // 2])) + start
	# print(series[ len(series) // 2])
	return series.reshape(-1, 1)


def noise_generator(amplitude, variance, n):

	return amplitude * np.random.normal(0, variance, n).reshape(-1, 1)


def plot_fft_spectrum(peak_freq, f, pxx, N, ax=None, c='navy', label='fft', figsize=(6, 5), include_beta_band_in_legend=False):

	fig, ax = get_axes(ax, figsize=figsize)
	# plt.semilogy(freq[:N//2], f[:N//2])
	label = label + ' f =' + str(round(peak_freq, 0))
	if include_beta_band_in_legend:
		beta_band_power = beta_bandpower(f, pxx)
		label += ' ' + r'$\overline{P}_{\beta}=$' + str(round(beta_band_power, 3))
	ax.plot(f, pxx, c=c, label=label, lw=1.5)
	ax.set_xlabel('frequency (Hz)', fontsize=15)
	ax.set_ylabel('FFT power', fontsize=15)
	ax.legend(fontsize=15, loc='center right', framealpha=0.1, frameon=False)
	# ax.tick_params(axis='both', which='major', labelsize=10)
	ax.locator_params(axis='y', nbins=5)
	ax.locator_params(axis='x', nbins=5)
	plt.rcParams['xtick.labelsize'] = 18
	plt.rcParams['ytick.labelsize'] = 18
	remove_frame(ax)


def freq_from_fft(sig, dt, plot_spectrum=False, ax=None, c='navy', label='fft', figsize=(6, 5), method='rfft', n_windows=6, include_beta_band_in_legend=False):
	"""
	Estimate frequency from peak of FFT
	"""
	# Compute Fourier transform of windowed signal
#    windowed = sig * signal.blackmanharris(len(sig))

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
			                  figsize=figsize, include_beta_band_in_legend=include_beta_band_in_legend)
		ind_max = 200
		return f[:ind_max], pxx[:ind_max], peak_freq


def freq_from_rfft(sig, dt, N):
	'''Estimate frequency with rfft method '''
	rf = rfft(sig)
	f = fftfreq(N, dt)[:N//2]
	# print(f)
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


def run(receiving_class_dict, t_list, dt, nuclei_dict):

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

	stop = timeit.default_timer()
	print("t = ", stop - start)
	return nuclei_dict

	stop = timeit.default_timer()
	print("t = ", stop - start)
	return nuclei_dict


def run_transition_to_DA_depletion(receiving_class_dict, t_list, dt, nuclei_dict, DD_init_filepaths, K_DD, N, N_real, A_DD, A_mvt, D_mvt, t_mvt, t_transition=None):
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


def reset_connec_ext_input_DD(nuclei_dict, K_DD, N, N_real, A_DD, A_mvt, D_mvt, t_mvt, t_list, dt):
	K = calculate_number_of_connections(N, N_real, K_DD)
	for nuclei_list in nuclei_dict.values():
		for nucleus in nuclei_list:
			nucleus.set_connections(K, N)
			# nucleus.set_synaptic_weights(G_DD)
			nucleus.set_ext_input(A_DD, A_mvt, D_mvt, t_mvt, t_list, dt)


def run_transition_to_movement(receiving_class_dict, t_list, dt, nuclei_dict, mvt_init_filepaths, N, N_real, A_mvt, D_mvt, t_mvt, t_transition=None):
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


def run_with_transient_external_input(receiving_class_dict, t_list, dt, nuclei_dict, rest_init_filepaths,
									  transient_init_filepaths, A, A_trans, list_of_nuc_with_trans_inp,
									  t_transient=10, duration=10):
	'''
		run normaly til "t_transient" then exert an external transient input to "list_of_nuc_with_trans_inp" then resume to normal state until the end of simulation
	'''
	start = timeit.default_timer()
	# basal firing
	iterate_SNN(nuclei_dict, dt, receiving_class_dict,
	            t_start=0, t_end=t_transient)
	# transient external input to the network
	# , A_mvt, D_mvt, t_mvt, t_list, dt)
	selective_reset_ext_input(
	    nuclei_dict, transient_init_filepaths, list_of_nuc_with_trans_inp, A_trans)
	# interate through the duration of transient input
	iterate_SNN(nuclei_dict, dt, receiving_class_dict,
	            t_start=t_transient, t_end=t_transient + duration)
	# reset back to rest
	# , A_mvt, D_mvt, t_mvt, t_list, dt)
	selective_reset_ext_input(
	    nuclei_dict, rest_init_filepaths, list_of_nuc_with_trans_inp, A)
	# look at the decay of the transient input
	iterate_SNN(nuclei_dict, dt, receiving_class_dict,
	            t_start=t_transient + duration, t_end=t_list[-1])

	stop = timeit.default_timer()
	print("t = ", stop - start)
	return nuclei_dict


def get_corr_key_to_val(mydict, value):
	""" return all the keys corresponding to the specified value"""
	return [k for k, v in mydict.items() if v == value]


def run_with_transient_external_input_including_transmission_delay(receiving_class_dict, t_list, dt, nuclei_dict, rest_init_filepaths,
																   transient_init_filepaths, A, A_trans,  syn_trans_delay_dict,
																   t_transient=10, duration=10):
	'''
		run normaly til "t_transient" then exert an external transient input to the concerned nuclei then resume to normal state until the end of simulation.
		Where the syn_trans_delay_dict contains the synaptic transmission delays of the input to different nuclei (e.g. MC to STN and MC to D2)
	'''

	min_syn_trans_delays = min(
	    syn_trans_delay_dict, key=syn_trans_delay_dict. get)
	# synaptic trans delay relative to the nucleus with minimum delay.
	t_start_inp_dict = {k: t_transient + v -
	    syn_trans_delay_dict[min_syn_trans_delays] for k, v in syn_trans_delay_dict.items()}
	# synaptic trans delay relative to the nucleus with minimum delay.
	t_end_inp_dict = {k: duration + v for k, v in t_start_inp_dict.items()}

# 	print('stimulation start times = ', t_start_inp_dict*dt)
# 	print('stimulation end times = ', t_end_inp_dict)

	start = timeit.default_timer()

	for t in t_list:
		# if it's the start of external input to (a) nucleus(ei)
		if t in list(t_start_inp_dict.values()):
 			# print("stim at {} for {}".format(t*dt, get_corr_key_to_val(t_start_inp_dict, t)))
 			selective_reset_ext_input(nuclei_dict, transient_init_filepaths,
									 get_corr_key_to_val(t_start_inp_dict, t),
									 A_trans)
		# if it's the end of external input to (a) nucleus(ei)
		if t in list(t_end_inp_dict.values()):
# 			print("stim end at {} for {}".format(t*dt, get_corr_key_to_val(t_end_inp_dict, t)))
			selective_reset_ext_input(nuclei_dict, rest_init_filepaths,
									 get_corr_key_to_val(t_end_inp_dict, t),
									 A)
		for nuclei_list in nuclei_dict.values():
			for k, nucleus in enumerate(nuclei_list):
				k += 1
				nucleus.solve_IF(t, dt, receiving_class_dict[(nucleus.name, str(k))])

	stop = timeit.default_timer()
	print("t = ", stop - start)

	return nuclei_dict


##### seting ext input with respect to all corresponding nuclei going to A of transient (Seems wrong because the external input should be independent of the synaptic input)
# def run_with_transient_external_input_including_transmission_delay(receiving_class_dict, t_list, dt, nuclei_dict, rest_init_filepaths,
# 																   A, A_trans,  syn_trans_delay_dict, transient_init_filepaths = None,
# 																   t_transient=10, duration=10, inp_method='reset', ext_inp_dict = None):
    
#     '''
#     		run normaly til "t_transient" then exert an external transient input to the concerned nuclei then resume to normal state until the end of simulation.
#     		Where the syn_trans_delay_dict contains the synaptic transmission delays of the input to different nuclei (e.g. MC to STN and MC to D2)
#     '''
#     min_syn_trans_delays = min(syn_trans_delay_dict, key=syn_trans_delay_dict. get)
    
    
#     # synaptic trans delay relative to the nucleus with minimum delay.
#     t_start_inp_dict = {k: t_transient + v -
# 	    syn_trans_delay_dict[min_syn_trans_delays] for k, v in syn_trans_delay_dict.items()}
# 	# synaptic trans delay relative to the nucleus with minimum delay.
#     t_end_inp_dict = {k: duration + v for k, v in t_start_inp_dict.items()}

# #     print('stimulation start times = ', t_start_inp_dict*dt)
# #     print('stimulation end times = ', t_end_inp_dict)

#     start = timeit.default_timer()

#     for t in t_list:
#         # if it's the start of external input to (a) nucleus(ei)
#         if t in list(t_start_inp_dict.values()):
#             if inp_method == 'reset':
#              # print("stim at {} for {}".format(t*dt, get_corr_key_to_val(t_start_inp_dict, t)))             
#              selective_reset_ext_input(nuclei_dict, transient_init_filepaths, 
#                                          get_corr_key_to_val(t_start_inp_dict, t), 
#                                          A_trans) 
#             elif inp_method == 'add':
#                 selective_additive_ext_input(nuclei_dict, get_corr_key_to_val(t_start_inp_dict, t), ext_inp_dict)

                    
#         # if it's the end of external input to (a) nucleus(ei)
#         if t in list( t_end_inp_dict.values() ):
# #             print("stim end at {} for {}".format(t*dt, get_corr_key_to_val(t_end_inp_dict, t)))
#             # if inp_method == 'reset':
#             selective_reset_ext_input(nuclei_dict, rest_init_filepaths, 
#                                      get_corr_key_to_val(t_end_inp_dict, t), 
#                                      A)
#             # elif inp_method == 'add':
#             #     ext_inp_dict_end = {k: {k1: -v1 for k1, v1 in v.items()} for k,v in ext_inp_dict.items()}
#             #     selective_additive_ext_input(nuclei_dict, get_corr_key_to_val(t_start_inp_dict, t), ext_inp_dict_end)

#         for nuclei_list in nuclei_dict.values():
#             for k, nucleus in enumerate(nuclei_list):
#                 k+=1
#                 nucleus.solve_IF(t,dt,receiving_class_dict[(nucleus.name,str(k))])

#     stop = timeit.default_timer()
#     print("t = ", stop - start)
    
#     return nuclei_dict


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
                k+=1
                nucleus.solve_IF(t,dt,receiving_class_dict[(nucleus.name,str(k))])
                
def selective_additive_ext_input(nuclei_dict, list_of_nuc_with_trans_inp, ext_inp_dict):
    
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            if nucleus.name in list_of_nuc_with_trans_inp:
                ext_inp = np.random.normal(ext_inp_dict[nucleus.name]['mean'], ext_inp_dict[nucleus.name]['sigma'], nucleus.n)
                nucleus.additive_ext_input(ext_inp)

def selective_reset_ext_input(nuclei_dict, init_filepaths, list_of_nuc_with_trans_inp, A, A_mvt = None, D_mvt = None, t_mvt = None, t_list = None, dt = None):
    set_init_all_nuclei(nuclei_dict, list_of_nuc_with_trans_inp =list_of_nuc_with_trans_inp, filepaths = init_filepaths) 
    for nuclei_list in nuclei_dict.values():
        for nucleus in nuclei_list:
            if nucleus.name in list_of_nuc_with_trans_inp:
                nucleus.set_ext_input(A, A_mvt, D_mvt,t_mvt, t_list, dt)

def mvt_grad_ext_input(D_mvt, t_mvt, delay, H0, t_series):
    ''' a gradually increasing deacreasing input mimicing movement'''

    H = H0*np.cos(2*np.pi*(t_series-t_mvt)/D_mvt)**2
    ind = np.logical_or(t_series < t_mvt + delay, t_series > t_mvt + D_mvt + delay)
    H[ind] = 0
    return H

def mvt_step_ext_input(D_mvt, t_mvt, delay, H0, t_series):
    ''' step function returning external input during movment duration '''
    
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

def plot( nuclei_dict,color_dict,  dt, t_list, A, A_mvt, t_mvt, D_mvt, ax = None, title = "", n_subplots = 1,title_fontsize = 12,plot_start = 0,ylabelpad = 0, include_FR = True, alpha_mvt = 0.2,
         plot_end = None, figsize = (6,5), plt_txt = 'vertical', plt_mvt = True, plt_freq = False, ylim = None, include_std = True, round_dec = 2, legend_loc = 'upper right', 
         continuous_firing_base_lines = True, axvspan_color = 'lightskyblue', tick_label_fontsize = 18):    

    fig, ax = get_axes (ax)
    if plot_end == None : plot_end = t_list [-1]
    else:
        plot_end = int(plot_end / dt)
    plot_start = int( plot_start / dt)
    line_type = ['-', '--']
    count = 0
    for nuclei_list in nuclei_dict.values():
        for nucleus in [nuclei_list[0]]:
            if plt_freq:
                label = nucleus.name + ' f=' + str(round(nucleus.frequency_basal,1)) + ' Hz'
            else: 
                label = nucleus.name
            ax.plot(t_list[plot_start: plot_end]*dt, nucleus.pop_act[plot_start: plot_end], line_type[nucleus.population_num-1], label = label, c = color_dict[nucleus.name],lw = 1.5)
            if continuous_firing_base_lines:
                ax.plot(t_list[plot_start: plot_end]*dt, np.ones_like(t_list[plot_start: plot_end])*A[nucleus.name], '--', c = color_dict[nucleus.name],lw = 1, alpha=0.8 )
            else:
                ax.plot(t_list[plot_start: int(t_mvt /dt)]*dt, np.ones_like(t_list[plot_start: int(t_mvt /dt)])*A[nucleus.name], '--', c = color_dict[nucleus.name],lw = 1, alpha=0.8 )

            if plt_mvt:
                if continuous_firing_base_lines:
                    ax.plot(t_list[plot_start: plot_end]*dt, np.ones_like(t_list[plot_start: plot_end])*A_mvt[nucleus.name], '--', c = color_dict[nucleus.name], alpha = alpha_mvt,lw = 1 )
                else:
                    ax.plot(t_list[int(t_mvt /dt): plot_end]*dt, np.ones_like(t_list[int(t_mvt /dt): plot_end])*A_mvt[nucleus.name], '--', c = color_dict[nucleus.name], alpha= alpha_mvt,lw = 1 )
            FR_mean, FR_std = nucleus. average_pop_activity( t_list, last_fraction = 1/2)
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

    ax.axvspan(t_mvt, t_mvt+D_mvt, alpha=0.2, color=axvspan_color)
    ax.set_title(title, fontsize = title_fontsize)
    ax.set_xlabel("time (ms)", fontsize = 15)
    ax.set_ylabel("firing rate (spk/s)", fontsize = 15,labelpad=ylabelpad)
    ax.legend(fontsize = 15, loc = legend_loc, framealpha = 0.1, frameon = False)
    # ax.tick_params(axis='both', which='major', labelsize=10)
    ax_label_adjust(ax, fontsize = tick_label_fontsize, nbins = 5)

    ax.set_xlim(plot_start * dt - 20, plot_end * dt + 20) 
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
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
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
        # relative first and last peak ratio thresholding
        if len(peaks)>1 : 
            last_first_peak_ratio = sig[peaks[-1]]/sig[peaks[1]]
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
    

def synaptic_weight_space_exploration(G, A, A_mvt, D_mvt, t_mvt, t_list, dt,filename, lim_n_cycle, G_dict, nuclei_dict, duration_mvt, duration_base, receiving_class_dict, color_dict, if_plot = False, G_ratio_dict = None, plt_start = 0):
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

                (data[(nucleus.name, 'n_half_cycles_mvt')][i,j],
                data[(nucleus.name,'perc_t_oscil_mvt')][i,j], 
                data[(nucleus.name,'mvt_freq')][i,j],
                if_stable_mvt )= find_freq_of_pop_act_spec_window(nucleus,*duration_mvt,dt, peak_threshold =nucleus.oscil_peak_threshold, 
                                                                smooth_kern_window = nucleus.smooth_kern_window, check_stability = True)
                (data[(nucleus.name, 'n_half_cycles_base')][i,j],
                    data[(nucleus.name,'perc_t_oscil_base')][i,j], 
                    data[(nucleus.name,'base_freq')][i,j],
                    if_stable_base ) = find_freq_of_pop_act_spec_window(nucleus,*duration_base,dt, peak_threshold =nucleus.oscil_peak_threshold, 
                                                                        smooth_kern_window = nucleus.smooth_kern_window, check_stability = True)
                print(nucleus.name,' g1 = ', round(g_1,2), ' g2 = ', round(g_2,2), 'n_cycles =', data[(nucleus.name, 'n_half_cycles_mvt')][i,j],round(data[(nucleus.name, 'perc_t_oscil_mvt')][i,j],2),'%',  'f = ', round(data[(nucleus.name,'mvt_freq')][i,j],2) )

                if not found_g_transient[nucleus.name]  and data[(nucleus.name, 'n_half_cycles_mvt')][i,j]> lim_n_cycle[0] and data[(nucleus.name, 'n_half_cycles_mvt')][i,j]< lim_n_cycle[1]:
                    data[(nucleus.name,'g_transient_boundary')].append([g_1,g_2]) # save the the threshold g to get transient oscillations
                    found_g_transient[nucleus.name] = True

                if not if_trans_plotted and data[(nucleus.name, 'n_half_cycles_mvt')][i,j]> lim_n_cycle[0] and data[(nucleus.name, 'n_half_cycles_mvt')][i,j]< lim_n_cycle[1]:
                    if_trans_plotted = True
                    print("transient plotted")
                    fig_trans = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, include_FR = False, plot_start = plt_start, legend_loc = 'upper left',title_fontsize = 15,
                        title = r"$G_{"+list(G_dict.keys())[0][1]+"-"+list(G_dict.keys())[0][0]+"}$ = "+ str(round(g_1,2))+r"$\; G_{"+list(G_dict.keys())[1][1]+"-"+list(G_dict.keys())[1][0]+"}$ ="+str(round(g_2,2)),
                        ax = None)
                
                if not found_g_stable[nucleus.name] and if_stable_mvt: 
                    found_g_stable[nucleus.name] = True
                    data[(nucleus.name,'g_stable_boundary')].append([g_1,g_2])

                if not if_stable_plotted and if_stable_mvt:
                    if_stable_plotted = True
                    print("stable plotted")
                    fig_stable = plot(nuclei_dict,color_dict, dt, t_list, A, A_mvt, t_mvt, D_mvt, include_FR = False, plot_start = plt_start, legend_loc = 'upper left', title_fontsize = 15,
                        title = r"$G_{"+list(G_dict.keys())[0][1]+"-"+list(G_dict.keys())[0][0]+"}$ = "+ str(round(g_1,2))+r"$\; G_{"+list(G_dict.keys())[1][1]+"-"+list(G_dict.keys())[1][0]+"}$ ="+str(round(g_2,2)), 
                        ax = None)
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
    return fig_trans, fig_stable

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
            nucleus.scale_synaptic_weight() 
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
def synaptic_weight_transition_multiple_circuit_SNN(filename_list, name_list, label_list, color_list, g_cte_ind, g_ch_ind, y_list, 
                                                    c_list,colormap,x_axis = 'multiply',title = "",x_label = "G", key = (), param =None):
    maxs = [] ; mins = []
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    # for i in range(len(filename_list)):
    #     pkl_file = open(filename_list[i], 'rb')
    #     data = pickle.load(pkl_file)
    #     pkl_file.close()
    #     maxs.append(np.max(data[name_list[i],c_list[i]]))
    #     mins.append(np.min(data[name_list[i],c_list[i]]))
    # vmax = max(maxs) ; vmin = min(mins)
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
#         print('g = ', g)
        if param == None:
            y =  np.squeeze( np.average ( data[(name_list[i],y_list[i])] , axis = 1))
            title = ''
        elif param == 'low':
            y =  np.squeeze( np.average ( data[(name_list[i],y_list[i])][:,:,0] , axis = 1))
            title = 'Low beta band (12-30 Hz)'
        elif param == 'high':
            y =  np.squeeze( np.average ( data[(name_list[i],y_list[i])][:,:,1] , axis = 1))
            title = 'High beta band (20-30 Hz)'
        color = np.squeeze( np.average ( data[(name_list[i],c_list[i])] , axis = 1))
        where_are_NaNs = np.isnan(y)
        y[where_are_NaNs] = 0
#         print('beta_power = ', y)
#         print('freq = ', color)
        ax.plot(g, y, c = color_list[i], lw = 3, label= label_list[i],zorder=1)
        img = ax.scatter(g, y ,vmin = vmin, vmax = vmax, c = color, 
                         cmap=plt.get_cmap(colormap),lw = 1,edgecolor = 'k',zorder=2, s=100)
    plt.axhline( 4, linestyle = '--', c = 'grey', lw=2)  # to get the circuit g which is the muptiplication
    plt.title(title, fontsize = 20)
    ax.set_xlabel(x_label,fontsize = 20)
    ax.set_ylabel('Beta Power (W/Hz)',fontsize=20)
    # ax.set_title(title,fontsize=20)
    ax_label_adjust(ax, fontsize = 18, nbins = 6)
    axins1 = inset_axes(ax,
                    width="5%",  # width = 50% of parent_bbox width
                    height="50%",  # height : 5%
                    loc='center right',borderpad=8)#, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),)
    clb = fig.colorbar(img, cax=axins1, orientation="vertical")
    clb.ax.locator_params(nbins=4)
    clb.set_label('Frequency (Hz)', labelpad=20, y=.5, rotation=-90,fontsize=15)
    img.set_clim(0,50)
    ax.legend(fontsize=15, frameon = False, framealpha = 0.1, loc = 'upper right')
    remove_frame(ax)
    plt.show()
    for i in range( len(txt)):
        plt.gcf().text(0.5, 0.8- i*0.05,txt[i], ha='center',fontsize = 13)
    return fig

def synaptic_weight_transition_multiple_circuits(filename_list, name_list, label_list, color_list, g_cte_ind, g_ch_ind, y_list, c_list,colormap = 'hot',
                                                    x_axis = 'multiply',title = "",x_label = "G", x_scale_factor = 1, leg_loc = 'upper right', vline_txt = True):
    maxs = [] ; mins = []
    fig = plt.figure(figsize=(8,7))
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
            x_label = r'$G_{Loop}$'
        else:
            g = np.squeeze(data['g'][:,:,g_ch_ind[i]])
            g_transient = data[name_list[i],'g_transient_boundary'][0][g_ch_ind[i]]
            print(data[name_list[i],'g_stable_boundary'])
            g_stable = data[name_list[i],'g_stable_boundary'][0][g_ch_ind[i]]

        # ax.plot(np.squeeze(data['g'][:,:,g_ch_ind[i]]), np.squeeze(data[(name_list[i],y_list[i])]),c = color_list[i], lw = 1, label= label_list[i])
        # img = ax.scatter(np.squeeze(data['g'][:,:,g_ch_ind[i]]), np.squeeze(data[(name_list[i],y_list[i])]),vmin = vmin, vmax = vmax, c=data[(name_list[i],c_list[i])], cmap=colormap,lw = 1,edgecolor = 'k')
        # plt.axvline(g_transient[g_ind[i]], c = color_list[i])
        ax.plot(g * x_scale_factor, np.squeeze(data[(name_list[i],y_list[i])]),c = color_list[i], lw = 3, label= label_list[i],zorder=1)
        img = ax.scatter(g * x_scale_factor, np.squeeze(data[(name_list[i],y_list[i])]),vmin = vmin, vmax = vmax, c=data[(name_list[i],c_list[i])], cmap=plt.get_cmap(colormap),lw = 1,edgecolor = 'k',zorder=2,s=80)
        ax.axvline(g_transient * x_scale_factor, linestyle = '-.',c = color_list[i],alpha = 0.3,lw=2)  # to get the circuit g which is the muptiplication
        ax.axvline(g_stable * x_scale_factor, c = color_list[i],lw=2)  # to get the circuit g which is the muptiplication
    if vline_txt :
        ax.text(g_stable * x_scale_factor-0.5, 0.6, 'Stable oscillations',fontsize=18, rotation = -90)
        ax.text(g_transient * x_scale_factor, 0.6, 'Oscillation appears',fontsize=18, rotation = -90)
    ax.set_xlabel(x_label,fontsize = 20)
    ax.set_ylabel('frequency(Hz)',fontsize=20)
    ax.set_title(title,fontsize=20)
    ax_label_adjust(ax, fontsize = 18, nbins = 8)

    # ax.set_xlim(limits['x'])
    # ax.set_ylim(limits['y'])
    axins1 = inset_axes(ax,
                    width="5%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='center right')#,borderpad=-1)#, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),)
    clb = fig.colorbar(img, cax=axins1, orientation="vertical")
    clb.ax.locator_params(nbins=4)
    clb.set_label('% Oscillation', labelpad=20, y=.5, rotation=-90,fontsize=15)
    ax.legend(fontsize=15, frameon = False, framealpha = 0.1, loc = leg_loc)
    remove_frame(ax)

    return fig

def multi_plot_as_f_of_timescale(y_list, color_list, label_list, name_list, filename_list, x_label, y_label, 
                                    g_tau_2_ind = None, ylabelpad = -5, title = '', c_label = '', ax = None):
    fig, ax = get_axes (ax)
    
    for i in range(len(filename_list)):
        pkl_file = open(filename_list[i], 'rb')
        data = pickle.load(pkl_file)
        x_spec =  data['tau'][:,:,0][:,0]

        y_spec = data[(name_list[i], y_list[i])][:,g_tau_2_ind]. reshape(-1,)
        print(x_spec.shape, y_spec.shape)
        ax.plot(x_spec,y_spec, '-o', c = color_list[i], lw = 3, label= label_list[i],zorder = 1)#, path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()])
        ax.set_xlabel(x_label,fontsize = 20)
        ax.set_ylabel(y_label,fontsize = 20,labelpad=ylabelpad)
        ax.set_title(title,fontsize = 20)
        # ax.set_xlim(limits['x'])
        # ax.set_ylim(limits['y'])
        ax_label_adjust(ax, fontsize = 20)
        remove_frame(ax)
    plt.legend(fontsize = 20)
    plt.show()
    return fig, ax
def multi_plot_as_f_of_timescale_shared_colorbar(y_list, color_list, c_list, label_list,name_list,filename_list,x_label,y_label, 
                                    g_tau_2_ind = None, g_ratio_list = [], ylabelpad = -5, colormap = 'hot', title = '', c_label = ''):
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



def find_AUC_of_input(name,poisson_prop,gain, threshold, neuronal_consts,tau,ext_inp_delay,noise_variance, noise_amplitude,
                      N, A, A_mvt,D_mvt,t_mvt, N_real, K_real,t_list,color_dict, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, smooth_kern_window,oscil_peak_threshold,if_plot = True):
    receiving_pop_list = {(name,'1') : []}
    pop_list = [1]  
    
    class Nuc_AUC(Nucleus):
        def cal_ext_inp(self,dt,t):
            # to have exactly one spike the whole time
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


def _generate_filename_3_nuclei(nuclei_dict, G, noise_variance, fft_method):
    G = G_dict
    names = [list(nuclei_dict.values())[i][0].name for i in range(len(nuclei_dict))]
    gs = [str(round(G[('D2', 'FSI')][0],3)) + '--' + str(round(G[('D2', 'FSI')][-1],3)), 
          str(round(G[('Proto', 'D2')][0],3)) + '--' + str(round(G[('Proto', 'D2')][-1],3)), 
          str(round(G[('FSI', 'Proto')][0],3)) + '--' + str(round(G[('FSI', 'Proto')][-1],3))]
    gs = [gs[i].replace('.','-') for i in range( len (gs))]
    nucleus = nuclei_dict[names[0]][0]
    
    filename = (  names[0] + '_' + names[1] + '_'+  names[2] + '_G(FD)=' + gs[0]+ '_G(DP)=' +gs[1] + '_G(PF)= '  + gs[2] + 
              '_' + nucleus.init_method + '_' + nucleus.ext_inp_method + '_noise=' + 'input_integ_ext_' + nucleus.ext_input_integ_method + '_syn_' + nucleus.syn_input_integ_method+ '_' +
              str(noise_variance[names[0]]) + '_' + str(noise_variance[names[1]]) + '_' + str(noise_variance[names[2]]) 
            + '_N=' + str(nucleus.n) +'_T' + str(nucleus.t_sim) + '_' + fft_method  ) 
    
    return filename

def save_figs(figs, G, noise_variance, path, fft_method, pre_prefix = ['']*3, s= [(15,15)]*3):
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

def save_trans_stable_figs(fig_trans, fig_stable, path_rate, filename):
    fig_trans.savefig(os.path.join(path_rate, (filename + '_tansient_plot.png')),dpi = 300, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    fig_trans.savefig(os.path.join(path_rate, (filename + '_tansient_plot.pdf')),dpi = 300, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    fig_stable.savefig(os.path.join(path_rate, (filename + '_stable_plot.png')),dpi = 300, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    fig_stable.savefig(os.path.join(path_rate, (filename + '_stable_plot.pdf')),dpi = 300, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
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
