import numpy as np
import matplotlib.pyplot as plt


N_GPe = 1000
N_STN = 1000
A_STN = 34
A_GPe = 14
K_GPe_STN = 161
K_STN_GPe = 475

t = 1000 #ms
dt = 0.5 
class GPe:

    def __init__(self,n,nat_firing):
        self.n = n # population size
        self.nat_firing = nat_firing
        self.act = np.zeros((self.n,1))
        self.input = np.zeros((self.n,1))
        connection_arr = np.random.choice(n, k, replace=False) # choose k neurons out of n to send input to neuron i 
        self.J_STN = np.zeros
        
    def fix_firing_rate(inputs):
        
        self.input = inputs
        
    
GPe = GPe(N_GPe,A_GPe)
        
t_list = np.linspace(0,t,num = t/dt)
for t in t_list:
    
    update(network)