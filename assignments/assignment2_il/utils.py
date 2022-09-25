from typing import Type, List
import torch
from torch import nn
import numpy as np

# ====== NEURAL NET UTILS ======

def create_mlp(input_dim: int, output_dim: int, architecture: List[int], squash=False, activation: Type[nn.Module]=nn.ReLU) -> List[nn.Module]:
    '''Creates a list of modules that define an MLP.'''
    if len(architecture) > 0:
        layers = [nn.Linear(input_dim, architecture[0]), activation()]
    else:
        layers = []
        
    for i in range(len(architecture) - 1):
        layers.append(nn.Linear(architecture[i], architecture[i+1]))
        layers.append(activation())
    
    if output_dim > 0:
        last_dim = architecture[-1] if len(architecture) > 0 else input_dim
        layers.append(nn.Linear(last_dim, output_dim))
        
    if squash:
        # squashes output down to (-1, 1)
        layers.append(nn.Tanh())
    
    return layers

def create_net(input_dim: int, output_dim: int, squash=False):
    layers = create_mlp(input_dim, output_dim, architecture=[64, 64], squash=squash)
    net = nn.Sequential(*layers)
    return net

def argmax_policy(net):
    # TODO: Return a FUNCTION that takes in a state, and outputs the maximum Q value of said state.
    # Inputs:
    # - net: (type nn.Module). A neural network module, going from state dimension to number of actions. Q network.
    # Wanted output:
    # - argmax_fn: A function which takes in a state, and outputs the maximum Q value of said state.
    pass

def expert_policy(expert, s):
    '''Returns a one-hot encoded action of what the expert predicts at state s.'''
    action = expert.predict(s)[0]
    one_hot_action = np.eye(4)[action]
    return one_hot_action

# ====== ENV UTILS ======

def rollout(net, env, truncate=True):
    '''Rolls out a trajectory in the environment, with optional state masking.'''
    states = []
    actions = []
    
    ob = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        states.append(ob.reshape(-1))
        ob_tensor = torch.from_numpy(np.array(ob))
        if truncate:
            action = net(ob_tensor[:-2].float())
        else:
            action = net(ob_tensor.float())
            
        # detach action and convert to np array
        if isinstance(action, torch.FloatTensor) or isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        actions.append(action.reshape(-1))
        
        # step env
        ob, r, done, _ = env.step(np.argmax(action))
        total_reward += r
        
    states = np.array(states, dtype='float')
    action = np.array(actions, dtype='float')
    return states, actions

def expert_rollout(expert, env, truncate=False):
    '''Rolls out an expert trajectory in the environment, with optional state masking.'''
    expert_net = lambda s: expert.predict(s)[0]
    return rollout(expert_net, env, truncate=truncate)

# ====== EVAL UTILS ======

def eval_policy(policy, env, truncate=True):
    '''Evaluates policy with one trajectory in environment. Returns accumulated reward.'''
    done = False
    ob = env.reset()
    total_reward = 0
    while not done:
        if truncate:
            action = policy(ob[:-2])
        else:
            action = policy(ob)
        
        # detach action and convert to np array
        if isinstance(action, torch.FloatTensor) or isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        
        # step env and observe reward
        ob, r, done, _ = env.step(action)
        total_reward += r
    
    return total_reward