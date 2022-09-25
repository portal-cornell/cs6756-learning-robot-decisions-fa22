from utils import *
from torch import optim

'''Learner file (BC + DAgger)'''

class BC:
    def __init__(self, net, loss_fn):
        self.net = net
        self.loss_fn = loss_fn
        
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        
    def learn(self, env, states, actions, n_steps=1e4, truncate=True):
        # TODO: Implement this method. Return the final greedy policy (argmax_policy).
        pass
    
class DAgger:
    def __init__(self, net, loss_fn, expert):
        self.net = net
        self.loss_fn = loss_fn
        self.expert = expert
        
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        
    def learn(self, env, n_steps=1e4, truncate=True):
        # TODO: Implement this method. Return the final greedy policy (argmax_policy).
        # Make sure you are making the learning process fundamentally expert-interactive.
        pass
