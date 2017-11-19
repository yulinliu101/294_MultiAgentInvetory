import numpy as np
from scipy import stats
import os

class MASimulator:
    def __init__(self, 
                 seed = 101,
                 N_agent = 2,
                 N_prod = 3,
                 Tstamp = 20,
                 costQ = np.array([[0.1, 0.1, 0.1]]),
                 costInv = np.array([[0.2, 0.2, 0.2]]),
                 costLastInv = np.array([[1, 1, 1]]),
                 costBack = np.array([[0.5, 0.5, 0.5]]) ):
        self.seed = seed
        self.N_prod = N_prod
        self.N_agent = N_agent
        self.Tstamp = Tstamp

        self.costQ = costQ
        self.costInv = costInv
        self.costLastInv = costLastInv
        self.costBack = costBack

    def agent_dim(self):
        return self.N_agent
        
    def obs_dim(self):
        return (self.N_prod + self.Tstamp, 1)

    def act_dim(self):
        # add another axis in the future
        # first column: number of product to buy
        # second column: price
        return (self.N_prod, 2)

    def demandGenerator(self, mu, cov, seed = 123):
        # generate multivariate log normal random variables
        # mu should have shape of N_prod * 1
        # cov should have shape of N_prod * N_prod
        np.random.seed(seed)
        self.demandVec = np.exp(np.random.multivariate_normal(mean = np.log(mu), cov = cov, size = self.Tstamp))
        return self.demandVec
        
    def demandGenerator_p(self, 
                          actList, 
                          M, 
                          V, 
                          sens, 
                          cov, 
                          seed = 123):
        # generate price dependent demand (no correlations between products)
        # return a N_prod * N_agent demand array
        # price: N_prod * N_agent
        # M: potential market size has shape of N_prod * 1
        # V: base utility has shape of N_prod * 1
        # sens: price sensitivity has shape of N_prod * 1
        # std: standard deviation has shape of N_prod * 1
        price = []
        for action in actList:
            price.append(action[:,1])
        price = np.array(price).T
        # print("price shape:",price.shape)
        u = V.reshape(-1,1) - np.diag(sens.flatten()).dot(price)  # N_prod * N_agent
        eu = np.exp(u)
        share = M.reshape(-1,1) * eu / (np.sum(eu, 1) + 1).reshape(-1,1)
        # using a simple normal shock
        shock = np.random.multivariate_normal(np.zeros(self.N_prod), cov = cov, size = self.N_agent).T
        demand = share + shock
        return np.maximum(np.zeros(shape = demand.shape), demand)
        

    def actionSpace(self):
        # can be changed
        return [0,0], [10.0,5.0]

    def randomActionGenerator(self, Nsample = 1):
        minAction, maxAction = self.actionSpace()
        actionDim = self.act_dim()
        return np.random.uniform(low = minAction, high = maxAction, size = (Nsample, actionDim[0], actionDim[1]))

    def stateSpace(self):
        return -10., 10.

    # def randomStateGenerator(self, Nsample = 1):
    #     minInv, maxInv = self.stateSpace()
    #     return np.random.uniform(low = minInv, high = maxInv, size = (Nsample, self.N_prod))

    def randomInitialStateGenerator(self):
        _, maxInv = self.stateSpace()
        initTime = [1] + [0] * (self.Tstamp - 1)
        return np.append(np.array([initTime]), np.random.uniform(low = 0, high = maxInv, size = (1, self.N_prod))).reshape(1, -1)


    def _reward(self, action, inventory, demand, last = False):
        # here action is a vector of q (1 by N) and is the action for timestamp t
        # inventory has shape 1 by N and is the inventory for timestamp (t+1)
        # demand has shape 1 by N and is the inventory for timestamp (t)
        productIn = action[:,0].reshape(-1,1)
        priceSell = action[:,1].reshape(1,-1)
        demand = demand.reshape(1,-1)
        if last:
            return (priceSell.dot(demand.T) - (self.costQ.dot(productIn) + 
                                 self.costLastInv.dot(np.maximum(np.zeros(shape = inventory.shape), inventory).T) - 
                                 self.costBack.dot(np.minimum(np.zeros(shape = inventory.shape), inventory).T)))[0][0]
        else:
            return (priceSell.dot(demand.T) - (self.costQ.dot(productIn) + 
                                 self.costInv.dot(np.maximum(np.zeros(shape = inventory.shape), inventory).T) - 
                                 self.costBack.dot(np.minimum(np.zeros(shape = inventory.shape), inventory).T)))[0][0]

    def _nextInventory(self, curState, curAction, curDemand):
        productIn = curAction[:,0]
        curTimestamp = curState[:self.Tstamp]
        curInventory = curState[self.Tstamp:]
        nextTimestamp = np.roll(curTimestamp, 1)
        nextInventory = (curInventory + productIn - curDemand).reshape(1,-1)

        return nextTimestamp, nextInventory

    def step(self, action, currentState, curDemand, last):
        
        nextTimestamp, nextInventory = self._nextInventory(currentState, action, curDemand)
        reward = self._reward(action, nextInventory, curDemand, last)
        nextState = np.append(nextTimestamp, nextInventory).reshape(1, -1)
        return nextState, reward



## test
# env = MASimulator(seed = 1)
# price = np.array([[2, 1], [1, 1]])
# M = np.array([5,5])
# V = np.array([3,2])
# sens = np.array([.2,.1])
# std = np.array([0.3, 0.3])
# print(env.demandGenerator_p(price, M, V, sens, std))
#demand = env.demandGenerator(mu = np.array([5,5]), cov = np.diag(np.array([0.1, 0.1])))
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
