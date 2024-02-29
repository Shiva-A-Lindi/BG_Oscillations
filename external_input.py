# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:43:00 2024

@author: Shiva
"""

# Deriving F_ext from response curve of population firing rate in heterogeneous mode


from constants import *

plt.close('all')

########################################## Choose nucleus and state to fit I_ext for ##########################################
name = 'D2'
# name = 'FSI'
# name = 'STN'
# name = 'Proto'
# name = 'Arky'

state = 'rest'
# state = 'awake_rest'
# state = 'DD_anesth'
# state = 'mvt'
# state = 'trans_Nico_mice'
# state = 'trans_Kita_rat'
# state = 'induction_STN_excitation'
# state = 'induction_Proto_inhibition'



######################################## Parameters to change ###################################
#################################################################################################
mem_pot_init_method = 'uniform' # USE FIRST ROUND WHEN NO INIT SAVED
# mem_pot_init_method = 'draw_from_data' # USE AFTER YOU SAVED A COPY OF INIT 

save_mem_pot_dist = True 
# save_mem_pot_dist = False

use_saved_FR_ext = False  # USE FOR THE FIRST ROUND WHEN WITH NO ESTIMATE OF F_EXT SAVED
# use_saved_FR_ext = True  # USE TO REFINE THE SAVED ESTIMATE OF F_EXT

if_plot = True # IF YOU  WANT TO SEE FIT GRAPHICALLY

t_sim = 2000 # INCREASE FOR BETTER ACCURACY (NOTE: IT WILL INCREASE THE SAVED MEM_DIST FILESIZE)


root = r'C:\Users\Shiva\BG_Oscillations' # set directory
path = root
path_lacie = root
nicos_data_path = os.path.join(root,'Modeling_Data_Nico','Brice_paper', 'FR_Brice_data.xlsx')
##################################################################################################
##################################################################################################

print('desired activity =', Act[state][name])


N_sim = 1000
N = dict.fromkeys(N, N_sim)
dt = 0.1    
t_list = np.arange(int(t_sim/dt))
duration = [int(t_sim/dt/2), int(t_sim/dt)]
t_mvt = t_sim
D_mvt = t_sim - t_mvt

G = {}
receiving_pop_list = {(name, '1'): []}
pop_list = [1]
g = -0.01



init_method = 'heterogeneous'
keep_mem_pot_all_t = True
keep_noise_all_t = True
set_FR_range_from_theory = False
set_input_from_response_curve = True
der_ext_I_from_curve = True
noise_method = 'Ornstein-Uhlenbeck'
syn_input_integ_method = 'exp_rise_and_decay'
ext_input_integ_method = 'dirac_delta_input'
ext_inp_method = 'const+noise'
FSI_on_log = False
poisson_prop = {name: {'n': 10000, 'firing': 0.0475, 'tau': {
    'rise': {'mean': 1, 'var': .5}, 'decay': {'mean': 5, 'var': 3}}, 'g': 0.01}}

###### JN  REVISIONS 
########################################### Uncomment If you want to change the time constant ##########################################

# ta_m = np.linspace(5.13, 13, endpoint = True, num = 4)
# neuronal_consts['STN']['membrane_time_constant']= {'mean': ta_m[-1], 'var': 0.6 , 'truncmin': 2, 'truncmax': 25}  # for JN review process

###################################################### If no noise ######################################################################
# noise_variance[state][name] = 0

# range of I_ext to explore for fitting 
# rest
# FR_ext_range['STN'][state] = np.array([12/1000, 16/1000])
# FR_ext_range['Proto'][state] = np.array([8/1000, 10/1000])
# FR_ext_range['Arky'][state] = np.array([7/1000, 10/1000])
# FR_ext_range['FSI'][state] = np.array([68/1000, 75/1000])
# FR_ext_range['D2'][state] = np.array([32/1000, 43/1000])

#DD
# FR_ext_range['STN'][state] = np.array([14/1000, 17/1000])
# FR_ext_range['Proto'][state] = np.array([8/1000, 10/1000])
# FR_ext_range['Arky'][state] = np.array([7/1000, 10/1000])
# FR_ext_range['FSI'][state] = np.array([68/1000, 75/1000])
# FR_ext_range['D2'][state] = np.array([38/1000, 47/1000])

##################################################################################################



class Nuc_keep_V_m(Nucleus):

    def solve_IF(self, t, dt, receiving_from_class_list, mvt_ext_inp=None):
        
        self.cal_ext_inp_method_dict [self.external_input_bool](dt, t)
        synaptic_inputs = self.sum_synaptic_input(receiving_from_class_list, dt, t)
        self.update_potential(synaptic_inputs, dt, t, receiving_from_class_list)
        spiking_ind = self.find_and_save_new_spikes(t)
        # self.reset_potential(spiking_ind)
        self.reset_potential_with_interpolation(spiking_ind,dt)
        self.all_mem_pot[:, t] = self.mem_potential




nuc = [Nuc_keep_V_m(i, gain, threshold, neuronal_consts, tau, ext_inp_delay, noise_variance[state], noise_amplitude, 
                    N, Act[state], A_mvt, name, G, T, t_sim, dt, synaptic_time_constant, receiving_pop_list, 
                    smooth_kern_window, oscil_peak_threshold, neuronal_model='spiking', set_input_from_response_curve=set_input_from_response_curve, 
                    state = state, poisson_prop=poisson_prop, init_method=init_method, der_ext_I_from_curve=der_ext_I_from_curve, 
                    mem_pot_init_method=mem_pot_init_method, keep_mem_pot_all_t=keep_mem_pot_all_t, 
                    ext_input_integ_method=ext_input_integ_method, syn_input_integ_method=syn_input_integ_method,
                    path=path_lacie, save_init=False, noise_method=noise_method, keep_noise_all_t=keep_noise_all_t,
                    Act  = Act, FR_ext_specs = FR_ext_specs[name][state],
                    plot_spike_thresh_hist= False, plot_RMP_to_APth = False) for i in pop_list]



nuclei_dict = {name: nuc}
nucleus = nuc[0]

# plot_action_potentials(nucleus, n_neuron=1, t_end=5000)


n_FR = 20
all_FR_list = {name: FR_ext_range[name][state]
               for name in list(nuclei_dict.keys())}

##### THIS FUNCTION FITS THE EXTERNAL INPUT CURVE AFTER GENERATING DATA POINTS WITH RUNNING SIMULATIONS 
receiving_class_dict, nuclei_dict = set_connec_ext_inp(path, Act[state], A_mvt, D_mvt, t_mvt, dt, N, N_real, K_real, 
                                                       receiving_pop_list, nuclei_dict, t_list, all_FR_list=all_FR_list,
                                                        n_FR=n_FR, if_plot=if_plot, end_of_nonlinearity=end_of_nonlinearity,
                                                       set_FR_range_from_theory=False, method='collective', save_FR_ext=True,
                                                       use_saved_FR_ext=use_saved_FR_ext, normalize_G_by_N=True, state=state, time = True)

nuclei_dict = run(receiving_class_dict, t_list, dt,  {name: nuc})

if save_mem_pot_dist:
    save_all_mem_potential(nuclei_dict, path, state)
    
############################################### Plot results of fit ###################################################
for name in list(nuclei_dict.keys()):
    print('mean I0 ', name, np.round( np.average( nuclei_dict[name][0].rest_ext_input)) , 2) 
    print('mean Noise ', name, np.average(
        abs(nuclei_dict[name][0].noise_all_t)))
    print('std Noise ', name, np.std(nuclei_dict[name][0].noise_all_t))
    print('mean firing =', np.round( np.average(nuclei_dict[name][0].pop_act[int(t_sim / 2):]), 2 ) )
          # , '± ', np.round( np.std(nuclei_dict[name][0].pop_act[int(t_sim / 2):]), 2 ) ) 
    print('coherence = ', nuclei_dict[name][0].cal_coherence(
        dt, sampling_t_distance_ms=1))
    
if name in ['STN', 'Arky', 'Proto'] and state in ['rest', 'DD_anesth']:

    state_dict = {'rest': 'CTRL', 'DD_anesth': 'Park'}
    xls = pd.ExcelFile(nicos_data_path)
    if name == 'STN':
        col = 'w'
    else:
        col = 'k'

    figs = plot_exper_FR_distribution(xls, [name], [state_dict[state]],
                                      color_dict, bins=np.arange(
                                          0, bins[name][state]['max'], bins[name][state]['step']),
                                      hatch='/', edgecolor=col, alpha=0.2, zorder=1,
                                      annotate_fontsize = 20)
    

    fig_FR_dist = plot_FR_distribution(nuclei_dict, dt, color_dict, bins=np.arange(0, bins[name][state]['max'], bins[name][state]['step']),
                                    ax=figs[name].gca(), alpha=1, zorder=0, start=int(t_sim / dt / 2),
                                     legend_fontsize = 18, label_fontsize = 20, ticklabel_fontsize = 20,
                                     annotate_fontsize = 20, nbins = 4, state = state, path = path)

elif state in list (bins[name].keys() ):
    if name == '':
        
        only_non_zero= True; box_plot =False
    
    else:
        
        only_non_zero= False; box_plot =False
        
    if name == 'FSI' and FSI_on_log:
        log_hist = True
        _bins = np.logspace(-2, 2, 50)
    
    else:
        log_hist = False
        _bins = np.arange(0, bins[name][state]['max'], bins[name][state]['step'])
        
    fig_FR_dist = plot_FR_distribution(nuclei_dict, dt, color_dict,
                                        bins = _bins,
                                        ax = None, alpha = 1, zorder = 0, start = int(t_sim / dt / 2),
                                        log_hist = log_hist, only_non_zero= only_non_zero, box_plot =box_plot,
                                        legend_fontsize = 18, label_fontsize = 20, ticklabel_fontsize = 20,
                                        annotate_fontsize = 20, nbins = 4, state = state, path = path)
try:
    save_pdf_png(fig_FR_dist, os.path.join(path, name + '_FR_dist_' + state + '_'),
              size=(6, 5))
except NameError:
    pass

# fig_ISI_dist = plot_ISI_distribution(nuclei_dict, dt, color_dict, bins=np.logspace(0, 4, 50),
#                                      ax=None, alpha=1, zorder=0, start=int(t_sim / dt / 2), log_hist=True)

# save_pdf_png(fig_ISI_dist, os.path.join(path, name + '_ISI_dist_' + state + '_'),
#              size=(6, 5))

# plot_spike_amp_distribution(nuclei_dict, dt, color_dict, bins = 50)


status = 'set_FR'
# fig, ax = plot_mem_pot_dist_all_nuc(nuclei_dict, color_dict)

######################################## Plot average population firing rate time course derived from the fit ##########################################################
nucleus.smooth_pop_activity(dt, window_ms=5)
fig = plot(nuclei_dict, color_dict, dt,  t_list, Act[state], A_mvt, t_mvt, D_mvt, ax=None,
            title_fontsize=15, plot_start=int(t_sim / 2), title=str(dt),
            include_FR=False, include_std=False, plt_mvt=False,
            legend_loc='upper right', ylim=None)

# # save_pdf_png(fig, os.path.join(path, name + '_Firing_'),
# #              size=(12, 4))
