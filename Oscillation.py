import numpy as np
import matplotlib.pyplot as plt


N_sim = 100
population_list = ['STN', 'GPe']
N = { 'STN': N_sim , 'GPe': N_sim}
N_real = { 'STN': 13560 , 'GPe': 34470}
external_input_dict = { 'STN': 10 , 'GPe': 10} # external input coming from Ctx and Str
A = { 'STN': 34 , 'GPe': 14} # mean firing rate from experiments
threshold = { 'STN': 1 ,'GPe': 1}
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
G = { ('STN', 'GPe'): 1 ,
      ('GPe', 'STN'): 1 , 
      ('GPe', 'GPe'): 1} # synaptic weight
tau = {'GABA' : 5, 'Glut': 1} # synaptic time scale for excitation and inhibition
J = {}
t = 10 # simulation time in ms
dt = 0.5 # euler time step in ms
t_list = np.linspace(0,t,num = int(t/dt))

def calculate_number_of_connections(N_sim,N_real,K_real):
    KK = K_real.copy()
    for k, v in K_real.items():
        KK[k] = int(1/(1/v-1/N_real[k[1]]+1/N_sim[k[0]]))
    return KK

def transfer_func(Threshold, gain, x):
    ''' a transfer function that grows linearly for positive values 
    of input higher than the threshold'''
#    print(x.shape)
#    print(np.maximum(np.zeros_like(x),x))
    return gain* np.maximum(np.zeros_like(x), (np.maximum(np.zeros_like(x),x) - Threshold))
    
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

    def __init__(self, N, A, name, T, dt, tau, neuron_type):
        self.n = N[name] # population size
        self.name = name
        self.nat_firing = A[name]
        self.tau = tau[neuron_type] # synaptic time scale base on neuron type
        self.transmission_delays = {k: v for k, v in T.items() if k[1]==name} # filter based on the sending nucleus
        self.history_duration = max(self.transmission_delays.values()) # stored history in ms derived from the longest transmission delay of the projections
#        print(self.history_duration/dt)
        self.output = np.zeros((self.n,int(self.history_duration/dt)))
        self.input = np.zeros((self.n,1))
        self.act = np.zeros((self.n,1))
        self.receiving_from_list = [k[1] for k, v in T.items() if k[0]==name]
#        connection_arr = np.random.choice(n, k, replace=False) # choose k neurons out of n to send input to neuron i 
        
    def fix_firing_rate(self,inputs):
        
        self.input = inputs
        
    def calculate_input(self,threshold, gain, T, dt, external_input, receiving_from_class_list):    
        syn_inputs = np.zeros((self.n,1)) # = Sum (G Jxm)
        for projecting in receiving_from_class_list:
#            print(np.matmul(J[(self.name, projecting.name)], projecting.output[:,int(-T[(self.name,projecting.name)]*dt)].reshape(-1,1)).shape)
            syn_inputs += G[(self.name, projecting.name)]*np.matmul(J[(self.name, projecting.name)], projecting.output[:,int(-T[(self.name,projecting.name)]*dt)].reshape(-1,1))
        self.input = syn_inputs + external_input 
        self.act = transfer_func(threshold[self.name], gain[self.name], self.input)
        
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
GPe = Nucleus(N, A, 'GPe', T, dt, tau, 'GABA')
STN = Nucleus(N, A, 'STN', T, dt, tau, 'Glut')
nuclei_list = [GPe, STN]

GPe.set_connections(K, N)
STN.set_connections(K, N)

receiving_class_dict = {'STN': [GPe], 'GPe': [GPe, STN]}
for t in t_list:
    for nucleus in nuclei_list:
        external_input = np.ones((nucleus.n,1)) * external_input_dict[nucleus.name]
        nucleus.calculate_input(threshold,gain, T, dt, external_input, receiving_class_dict[nucleus.name])
        new_output = nucleus.output[:,-1] + dt*(-nucleus.output[:,-1]+nucleus.act)/nucleus.tau
        nucleus.update_output(new_output)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    