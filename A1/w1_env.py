#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Cody Rosevear
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017

  env transitions *ignore* actions, state transitions, rewards, and terminations are all random
"""

from utils import randn, randInRange, rand_un
import numpy as np

local_observation = None # local_observation: NumPy array
this_reward_observation = (None, None, None) # this_reward_observation: (floating point, NumPy array, Boolean)
nStatesSimpleEnv = 1
bandit_action_values = None

def env_init():
    global local_observation, this_reward_observation, bandit_action_values
    local_observation = np.zeros(1)

    this_reward_observation = (0.0, local_observation, False)

    #Create the bandit problem for the current run
    bandit_action_values = [randn(0.0, 1.0) for action in range(10)]


def env_start(): # returns NumPy array
    global local_observation#, this_reward_observation
    local_observation[0] = 0
    return this_reward_observation[1]

def env_step(this_action): # returns (floating point, NumPy array, Boolean), this_action: NumPy array
    global local_observation, this_reward_observation#, nStatesSimpleEnv
    episode_over = False

    #Get a reward from the current action reward distribution
    atp1 = int(this_action[0]) # how to extact action
    the_reward = randn(bandit_action_values[atp1], 1.0) # rewards drawn from (q*, 1) Gaussian

    stp1 = randInRange(nStatesSimpleEnv) # state transitions are uniform random
    #########

    local_observation[0] = stp1
    this_reward_observation = (the_reward, this_reward_observation[1], episode_over)

    return this_reward_observation

def env_cleanup():
    global bandit_action_values
    bandit_action_values = None
    return

#Note: This function has been taken and modified from its original source: Andrew Jacobsen's post (Tuesday, 12 September 2017, 8:58 PM) on the CMPUT609 course forum: https://eclass.srv.ualberta.ca/mod/forum/discuss.php?d=849395
def env_message(inMessage): # returns string, inMessage: string
    if inMessage == "what is your name?":
        return "my name is skeleton_environment!"
    elif inMessage == "get optimal action":
        return bandit_action_values.index(max(bandit_action_values))
    else:
       return "I don't know how to respond to your message"
