#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Cody Rosevear
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

from utils import randInRange, rand_un
import numpy as np
import math

local_action = None # local_action: NumPy array
this_action = None # this_action: NumPy array
last_observation = None # last_observation: NumPy array

numActions = 10

#tables
action_value_estimates = None
action_counts = None

#agent parameters
alpha = 0.10
C = 5 #fconfidence level for UCB agent

#These are set based on the current episode below in agent_start
epsilon = None
Q1 = None

#NOTE:#Agent strategies are indexed by the episode variable: We use episodes to switch
#parameter settings to ensure that agent strategies are tried on the same testbed
#for legitimate comparison
OPTIMISTIC_INIT = 0
EPSILON_GREEDY = 1

#tracker variables
time_step = 1
episode = 0

def agent_init():
    global local_action, this_action, last_observation, action_value_estimates, action_counts

    local_action = np.zeros(1)
    this_action = local_action
    last_observation = np.zeros(1)

def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global local_action, last_observation, this_action, episode, action_value_estimates, action_counts, epsilon, Q1, time_step

    #Set the parameters based on the current agent
    #We use episodes to distinguish between agent parameter settings
    if episode == OPTIMISTIC_INIT:
        epsilon = 0.0
        Q1 = 5
    elif episode == EPSILON_GREEDY:
        epsilon = 0.1
        Q1 = 0
    else:
        exit("BAD EPISODE: NO STRATEGY FOR THE CURRENT AGENT!!!")

    action_value_estimates = [Q1 for action in range(numActions)]
    action_counts = [0 for action in range(numActions)]

    stp1 = this_observation[0] # how you convert observation to a number, if state is tabular
    atp1 = randInRange(numActions)

    action_counts[atp1] += 1
    local_action[0] = atp1

    last_observation = this_observation # save observation, might be useful on the next step
    this_action = local_action

    time_step = 1
    return this_action


def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, observation_t: NumPy array
    global local_action, last_observation, this_action, action_value_estimates, action_counts, time_step, C

    #Update estimate for current action
    cur_action = int(this_action[0])
    action_value_estimates[cur_action] += alpha * (reward - action_value_estimates[cur_action])

    #Choose a new action using the parameter based agents
    stp1 = this_observation[0]
    action_selection_prob = rand_un()
    if action_selection_prob <= (1 - epsilon):
        atp1 = action_value_estimates.index(max(action_value_estimates))
    elif episode == EPSILON_GREEDY:
        atp1 = randInRange(numActions)
    else:
        print("BAD EPISODE: NO ACTION SELECTION FOR THE CURRENT AGENT!!!")
        exit()

    action_counts[atp1] += 1
    time_step += 1

    local_action[0] = atp1
    this_action = local_action
    last_observation = this_observation

    return this_action

def agent_end(reward): # reward: floating point
    global action_value_estimates, action_counts, episode

    #Reset the tables for the next agent
    action_value_estimates = None
    action_counts = None
    episode += 1
    return

def agent_cleanup():
    global episode
    #We've finished a run, so we can reset the episode counter to index through
    #all of the agent strategies again on the next run
    episode = 0
    return

def agent_message(inMessage): # returns string, inMessage: string
    # might be useful to get information from the agent

    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"
    else:
        return "I don't know how to respond to your message"
