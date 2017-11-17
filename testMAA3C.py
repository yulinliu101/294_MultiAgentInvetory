import numpy as np
import tensorflow as tf
import logz
import os
import time
import inspect
from MASimulator import MASimulator as Simulator
import random
#============================================================================================#
# Utilities
#============================================================================================#

#========================================================================================#
# Env setup
#========================================================================================#
def test():
    nAgent = 2
    # execState = """"""
    for agent in range(1, nAgent + 1):
        exec("env_%s = Simulator(seed = 101*%d, N_agent = nAgent, N_prod = 3,Tstamp = 20,costQ = np.array([[0.1, 0.1]]), costInv = np.array([[0.2, 0.2]]), costLastInv = np.array([[1, 1]]), costBack = np.array([[0.5, 0.5]]) )"%(str(agent), agent))
    print(env_1.obs_dim())

test()