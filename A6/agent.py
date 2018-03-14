#!/usr/bin/env python
from __future__ import division
from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import random
import json
from tiles3 import IHT, tiles

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

EPSILON = None
ALPHA = None
GAMMA = None
AGENT = None
NUM_STATES = 1000
AGGREGATE_SIZE = 100
POLY_DEGREE = 5
NUM_TILINGS = 50
TILE_WIDTH = 0.2
IHT_SIZE = 2048

def agent_init():
    global weights, iht

    if AGENT == "STATE_AGG":
        weights = np.array([0.0 for weight in range(1, NUM_STATES, AGGREGATE_SIZE)])
    elif AGENT == "TABULAR":
        weights = np.array([0.0 for weight in range(NUM_STATES)])
    elif AGENT == "POLYNOMIAL":
        weights = np.array([0.0 for weight in range(POLY_DEGREE + 1)])
    elif AGENT == "RADIAL":
        weights = np.array([0.0])
    elif AGENT == "TILE_CODING":
        iht = IHT(IHT_SIZE)
        weights = np.array([0.0 for weight in range(IHT_SIZE)])
    else:
        exit("Invalid agent selection!")

    weights = weights[np.newaxis, :]


def agent_start(state):
    global cur_state
    cur_state = state
    return 0

def agent_step(reward, state):
    global cur_state, weights

    next_state = state

    #Update the weights
    next_state_estimate = approx_value(next_state, weights)[0]
    if AGENT != "TILE_CODING":
        cur_state_estimate, semi_gradient = approx_value(cur_state, weights)
        weights += (ALPHA * (reward + (GAMMA * next_state_estimate) - cur_state_estimate)) * semi_gradient
    else:
        cur_state_estimate, tiles = approx_value(cur_state, weights)
        update = ALPHA * (reward + (GAMMA * next_state_estimate) - cur_state_estimate)
        for tile in tiles:
            weights[0][tile] += update

    cur_state = next_state
    return 0

def agent_end(reward):
    global state_estimates, weights

    if AGENT != "TILE_CODING":
        cur_state_estimate, semi_gradient = approx_value(cur_state, weights)
        weights += np.multiply(ALPHA * (reward - cur_state_estimate), semi_gradient)
    else:
        cur_state_estimate, tiles = approx_value(cur_state, weights)
        update = ALPHA * (reward - cur_state_estimate)
        for tile in tiles:
            weights[0][tile] += update

    #Get the current state value estimates for computing the cost in the experiment for the current episode
    state_estimates = [0.0] #First index of the true value functions in exp.py is just 0, so we add a 0 here to keep the arrays in sync when computing the error
    for state in range(1, NUM_STATES + 1):
        cur_state_estimate = approx_value(state, weights)[0]
        state_estimates.append(cur_state_estimate)
    return

def agent_cleanup():
    """
    This function is not used
    """
    return

def agent_message(in_message): # returns string, in_message: string
    global EPSILON, ALPHA, GAMMA, AGENT, state_estimates
    if in_message[0] == 'PARAMS':
        params = json.loads(in_message[1])
        EPSILON = params["EPSILON"]
        ALPHA = params['ALPHA']
        GAMMA = params['GAMMA']
        AGENT = params['AGENT']
    elif in_message == 'VALUE_FUNC_REQUEST':
        return state_estimates
    return

def approx_value(state, weights):
    global iht
    """
    Return the current approximated value for state given weights, and the gradient for the state,
    which is simply the feature vector for the state
    """

    if AGENT == "STATE_AGG":
        feature_vector = np.array([0.0 for weight in range(weights.shape[1])])
        state_groups = [group for group in range(1, NUM_STATES + AGGREGATE_SIZE, AGGREGATE_SIZE)]
        for i in range(len(state_groups) - 1):
            if state >= state_groups[i] and state <= state_groups[i + 1]:
                feature_vector[i] = 1
                break
    elif AGENT == "TABULAR":
        feature_vector = np.array([0.0 for weight in range(weights.shape[1])])
        feature_vector[state - 1] = 1
    elif AGENT == "POLYNOMIAL":
        feature_vector = np.array([float((state / NUM_STATES) ** degree) for degree in range(POLY_DEGREE + 1)])
    elif AGENT == "TILE_CODING":
        cur_state_rep = float((state / NUM_STATES) * (1 / TILE_WIDTH)) #Do this to get the right tile width
        cur_tiles = tiles(iht, NUM_TILINGS, [cur_state_rep])
        estimate = 0
        for tile in cur_tiles:
            estimate += weights[0][tile]
        return (estimate, cur_tiles)
    else:
        exit("Invalid agent selection!")
    feature_vector = feature_vector[np.newaxis]
    return (np.dot(weights, np.transpose(feature_vector)), feature_vector)
