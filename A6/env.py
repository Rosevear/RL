#!/usr/bin/env python

from utils import rand_norm, rand_in_range, rand_un
import numpy as np
import random
import json

START_STATE = 500
MIN_STATE = 1
MAX_STATE = 1000
STATE_TRANSITION_RANGE = 100
LEFT_TERMINAL_REWARD = -1
RIGHT_TERMINAL_REWARD = 1

def env_init():
    return

def env_start():
    global current_state
    current_state = START_STATE
    return current_state

def env_step(action):
    global current_state

    neighbours = [neighbour for neighbour in range(current_state - STATE_TRANSITION_RANGE, current_state + STATE_TRANSITION_RANGE + 1)]
    neighbours.remove(current_state)
    new_state = random.choice(neighbours)

    if new_state < MIN_STATE:
        is_terminal = True
        reward = -1
    elif new_state > MAX_STATE:
        is_terminal = True
        reward = 1
    else:
        is_terminal = False
        reward = 0

    current_state = new_state
    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}
    return result

def env_cleanup():
    return

def env_message(in_message): 
    return
