import numpy as np
from scipy import stats
import os

class MASimulator:
    def __init__(self, 
                 seed = 101,
                 N_agent = 2,
                 N_prod = 2,
                 Tstamp = 20,
                 price = np.array([[1.5, 1.5]]),
                 costQ = np.array([[0.1, 0.1]]),
                 costInv = np.array([[0.2, 0.2]]),
                 costLastInv = np.array([[1, 1]]),
                 costBack = np.array([[0.5, 0.5]]) ):
        self.seed = seed
        self.N_prod = N_prod
        self.N_agent = N_agent
        self.Tstamp = Tstamp


        self.price = price
        self.costQ = costQ
        self.costInv = costInv
        self.costLastInv = costLastInv
        self.costBack = costBack

    def agent_dim(self):
        return self.N_agent
        
    def obs_dim(self):
        return self.N_prod

    def act_dim(self):
        # add another axis in the future
        return self.N_prod

    def demandGenerator(self, mu, cov, seed = 123):
        # generate multivariate log normal random variables
        # mu should have shape of N_prod * 1
        # cov should have shape of N_prod * N_prod
        np.random.seed(seed)
        self.demandVec = np.exp(np.random.multivariate_normal(mean = np.log(mu), cov = cov, size = self.Tstamp))
        return self.demandVec

    def actionSpace(self):
        # can be changed
        return 0, 10.0

    def randomActionGenerator(self, Nsample = 1):
        minAction, maxAction = self.actionSpace()
        return np.random.uniform(low = minAction, high = maxAction, size = (Nsample, self.N_prod))

    def stateSpace(self):
        return -10., 10.

    def randomStateGenerator(self, Nsample = 1):
        minInv, maxInv = self.stateSpace()
        return np.random.uniform(low = minInv, high = maxInv, size = (Nsample, self.N_prod))

    def randomInitialStateGenerator(self):
        _, maxInv = self.stateSpace()
        return np.random.uniform(low = 0, high = maxInv, size = (1, self.N_prod))


    def _reward(self, action, inventory, demand, last = False):
        # here action is a vector of q (1 by N) and is the action for timestamp t
        # inventory has shape 1 by N and is the inventory for timestamp (t+1)
        # demand has shape 1 by N and is the inventory for timestamp (t)
        if last:
            return (self.price.dot(demand.T) - (self.costQ.dot(action.T) + 
                                 self.costLastInv.dot(np.maximum(np.zeros(shape = inventory.shape), inventory).T) - 
                                 self.costBack.dot(np.minimum(np.zeros(shape = inventory.shape), inventory).T)))[0][0]
        else:
            return (self.price.dot(demand.T) - (self.costQ.dot(action.T) + 
                                 self.costInv.dot(np.maximum(np.zeros(shape = inventory.shape), inventory).T) - 
                                 self.costBack.dot(np.minimum(np.zeros(shape = inventory.shape), inventory).T)))[0][0]

    def _nextInventory(self, curInventory, curAction, curDemand):
        return curInventory + curAction - curDemand

    def step(self, action, currentInventory, curDemand, last):
        nextInventory = self._nextInventory(currentInventory, action, curDemand)
        reward = self._reward(action, nextInventory, curDemand, last)
        return nextInventory, reward



# test
# env = Simulator(seed = 1)
# demand = env.demandGenerator(mu = np.array([5,5]), cov = np.diag(np.array([0.1, 0.1])))
# randomAction = env.randomActionGenerator()
# initState = env.randomStateGenerator()

# rew = env._reward(env.randomActionGenerator(), env.randomStateGenerator(), demand[0,:], False)
# rew2 = env._reward(env.randomActionGenerator(), env.randomStateGenerator(), demand[0,:], True)

# print(rew)
# print(rew2)
# print(randomAction)
# print(initState)
# print(demand)
# print(demand[0,:])
