#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import pickle

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

NUM_STATES = 100
TERMINAL_STATES = [0, 100]
TERMINAL_REWARD_WIN = 1
TERMINAL_REWARD_LOSS = 0
state_action_values, policy, returns, episode_states, episode_actions, episode_rewards, state_visited = [None for x in range(7)]

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    global state_action_values, policy, returns, episode_states, episode_actions, episode_rewards, state_visited, Q

    #Action list at each state yields a 2 dimensional list list[s][a] = Q(s, a)
    state_action_values = [[0 for action in range(min(state, NUM_STATES - state) + 1)] for state in range(NUM_STATES + 1)]
    #policy = [min(state, NUM_STATES - state) for state in range(NUM_STATES + 1)]
    #policy = [for state in range(NUM_STATES + 1)]
    returns = [[[] for action in range(min(state, NUM_STATES - state) + 1)] for state in range(NUM_STATES + 1)]

    state_action_values[0] = [TERMINAL_REWARD_LOSS]
    state_action_values[-1] = [TERMINAL_REWARD_WIN]
    policy[0] = None
    policy[-1] = None


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global state_action_values, policy, returns, episode_states, episode_actions, episode_rewards, state_action_visited

    #For computing averages upon the episode's end
    episode_states = []
    episode_actions = []
    episode_rewards = []
    state_action_visited = [[False for action in range(min(state, NUM_STATES - state) + 1)] for state in range(NUM_STATES + 1)]

    action = randint(1, min(state, NUM_STATES - state) + 1)

    episode_states.append(state)
    episode_actions.append(action)

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global state_action_values, policy, returns, episode_states, episode_actions, episode_rewards, state_visited

    action = policy[state[0]]
    episode_rewards.append(reward)
    episode_states.append(state[0])
    episode_actions.append(action)
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global state_action_values, policy, returns, episode_states, episode_actions, episode_rewards, state_visited

    episode_rewards.append(reward)

    for state in episode_states:
        state_action_idx = episode_states.index(state)
        action = episode_actions[state_action_idx]
        if (state not in TERMINAL_STATES) and (state_action_visited[state][action] == False):
            G = sum(episode_rewards[state_action_idx:]) #We are dealing with an episodic task, so the return is just the reward sum
            returns[state][action].append(G)
            state_action_values[state][action] = np.mean(returns[state][action])
            state_action_visited[state][action] = True

    for state in range(1, NUM_STATES):
        for action in range(1, min(state, NUM_STATES - state)):
            if state_action_visited[state][action]:
                #print(state_action_values[state])
                #This is to prevent the 0 action from being selected for the policy if the value of all of the actions for the state end up being the same
                if all(action_value == state_action_values[state][0] for action_value in state_action_values[state]):
                    policy[state] = max(1, state_action_values[state].index(max(state_action_values[state][1:])))
                else:
                    policy[state] = max(1, state_action_values[state].index(max(state_action_values[state][1:])))

    #We need to pad the state action value array so that it is a square so that numpy.max in agent nessage will accept it
    for state in range(len(state_action_values)):
        cur_state = state_action_values[state]
        padding = [None for action in range(NUM_STATES - len(cur_state))]
        state_action_values[state] = state_action_values[state] + padding

    return None

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    Q = state_action_values
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"
