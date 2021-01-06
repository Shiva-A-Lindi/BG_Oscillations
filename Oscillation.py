import numpy as np
import matplotlib.pyplot as plt
import timeit

N_sim = 50
population_list = ['STN', 'GPe']
N = { 'STN': N_sim , 'GPe': N_sim}
N_real = { 'STN': 13560 , 'GPe': 34470}
rest_ext_input = { 'STN': 30 , 'GPe': -20} # external input coming from Ctx and Str
A = { 'STN': 34 , 'GPe': 14} # mean firing rate from experiments
threshold = { 'STN': -0.1 ,'GPe': 0.1}
neuron_type = {'STN': 'Glut', 'GPe': 'GABA'}
gain = { 'STN': 1 ,'GPe': 1}

#K = { ('STN', 'GPe'): 475,
#      ('GPe', 'STN'): 161,
#      ('GPe', 'GPe'): 399}
K_real = { ('STN', 'GPe'): 883,
           ('GPe', 'STN'): 190,
           ('GPe', 'GPe'): 650}
T = { ('STN', 'GPe'): 1,
      ('GPe', 'STN'): 2.5, 
      ('GPe', 'GPe'): 1} # transmission delay in ms
G = { ('STN', 'GPe'): -0.2 ,
      ('GPe', 'STN'): 0.6 , 
      ('GPe', 'GPe'): -0.1} # synaptic weight
tau = {'GABA' : 5, 'Glut': 1} # synaptic time scale for excitation and inhibition
J = {}
t_sim = 100 # simulation time in ms
dt = 0.5 # euler time step in ms
t_mvt = 3 
D_mvt = 5
d_Str = 200 # duration of external input to Str
t_ext_to_Ctx = (t_mvt-D_mvt/2,t_mvt+D_mvt/2)
t_ext_to_Str = (t_mvt-D_mvt/2,t_mvt-D_mvt/2+d_Str)
t_list = np.arange(int(t_sim/dt))

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

#build_connection_matrix(4,10,2)
class Nucleus:

    def __init__(self, N, A, name, T, t_sim, dt, tau, neuron_type, rest_ext_input):
        self.n = N[name] # population size
        self.name = name
        self.nat_firing = A[name]
        self.tau = tau[neuron_type] # synaptic time scale based on neuron type
        self.transmission_delays = {k: v for k, v in T.items() if k[1]==name} # filter based on the sending nucleus
        self.history_duration = max(self.transmission_delays.values()) # stored history in ms derived from the longest transmission delay of the projections
        self.output = np.zeros((self.n,int(self.history_duration/dt)))
        self.input = np.zeros((self.n))
        self.neuron_act = np.zeros((self.n))
        self.pop_act = np.zeros((int(t_sim/dt))) # time series of population activity
        self.receiving_from_list = [k[1] for k, v in T.items() if k[0]==name]
        self.rest_ext_input = rest_ext_input[name]
        self.mvt_ext_input = np.zeros((int(t_sim/dt))) # external input mimicing movement
#        connection_arr = np.random.choice(n, k, replace=False) # choose k neurons out of n to send input to neuron i 
        
    def fix_firing_rate(self,inputs):
        
        self.input = inputs
        
    def calculate_input_and_inst_act(self,threshold, gain, T, t, dt, receiving_from_class_list):    
        syn_inputs = np.zeros((self.n,1)) # = Sum (G Jxm)
        for projecting in receiving_from_class_list:
#            print(np.matmul(J[(self.name, projecting.name)], projecting.output[:,int(-T[(self.name,projecting.name)]*dt)].reshape(-1,1)).shape)
            syn_inputs += G[(self.name, projecting.name)]*np.matmul(J[(self.name, projecting.name)], 
                           projecting.output[:,-int(T[(self.name,projecting.name)]/dt)].reshape(-1,1))
        self.input = syn_inputs + self.rest_ext_input 
#        self.input =  self.rest_ext_input* np.ones_like(syn_inputs) 

        self.neuron_act = transfer_func(threshold[self.name], gain[self.name], self.input)
#        self.neuron_act =  self.input

        self.pop_act[t] = np.average(self.neuron_act)
        
    def update_output(self, new_output):
        
        new = np.hstack((self.output[:,1:], new_output))
        self.output = new
        
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

K = calculate_number_of_connections(N,N_real,K_real)
GPe = Nucleus(N, A, 'GPe', T, t_sim, dt, tau, 'GABA', rest_ext_input)
STN = Nucleus(N, A, 'STN', T, t_sim, dt, tau, 'Glut', rest_ext_input)
nuclei_list = [GPe, STN]

GPe.set_connections(K, N)
STN.set_connections(K, N)

receiving_class_dict = {'STN': [GPe], 'GPe': [GPe, STN]} # points to the classes of nuclei each receive projections from

start = timeit.default_timer()
for t in t_list:
    for nucleus in nuclei_list:
#        external_input = np.ones((nucleus.n,1)) * nucleus.rest_ext_input
        nucleus.calculate_input_and_inst_act(threshold,gain, T, t, dt, receiving_class_dict[nucleus.name])
#        print(nucleus.neuron_act.shape)
        new_output = nucleus.output[:,-1].reshape(-1,1) + dt*(-nucleus.output[:,-1].reshape(-1,1)+nucleus.neuron_act)/nucleus.tau
        nucleus.update_output(new_output)
stop = timeit.default_timer()
print("t = ", stop - start)
plot_start = int(5/dt)
plt.figure(1)    
plt.plot(t_list[plot_start:]*dt,GPe.pop_act[plot_start:])
plt.title("GPe")
plt.xlabel("time (ms)")
plt.ylabel("firing rate (spk/s)")
#plt.figure(2)    
plt.plot(t_list[plot_start:]*dt,STN.pop_act[plot_start:])
plt.title("STN")    
plt.xlabel("time (ms)")
plt.ylabel("firing rate (spk/s)")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    