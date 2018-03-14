#!/usr/bin/env python
from rl_glue import *  # Required for RL-Glue


import numpy as np
import pickle
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import json
import random
from math import sqrt

if __name__ == "__main__":

    RLGlue("env", "agent")
    GRAPH_COLOURS = ('r', 'g', 'b', 'y', 'm', 'y', 'k')
    NUM_STATES = 1000

    #Agent parameters
    GAMMA = 1
    EPSILON = 0.10 #Not actually used anywhere
    #Copmpare all of the agents
    ALPHAS = [0.5, 0.01/50, 0.1, 0.01]
    AGENTS = ['TABULAR', 'TILE_CODING', 'STATE_AGG', "POLYNOMIAL"]
    #Compare the tile and polynomial basis
    #ALPHAS = [0.01/50, 0.0001, 0.001, 0.01, 0.1]
    #AGENTS = ['TILE_CODING', "POLYNOMIAL", "POLYNOMIAL", "POLYNOMIAL", "POLYNOMIAL"]

    num_episodes = 5000
    num_runs = 30
    state_values = np.load("TrueValueFunction.npy")

    print("Training the agents...")
    all_results = []
    for i in range(len(AGENTS)):
        print("Training the " + AGENTS[i] + " agent...")
        agent_params = {"EPSILON": EPSILON, "ALPHA": ALPHAS[i], "GAMMA": GAMMA, "AGENT": AGENTS[i]}
        RL_agent_message(('PARAMS', json.dumps(agent_params)))
        cur_agent_results = []
        for run in range(num_runs):
            #Different parts of the program use np.random (via utils.py) and others use just random,
            #seeding both with the same seed here to make sure they both start in the same place per run of the program
            np.random.seed(run)
            random.seed(run)
            run_results = []
            print "run number: ", run
            RL_init()
            for episode in range(num_episodes):
                RL_episode(0)
                state_estimates = RL_agent_message(('VALUE_FUNC_REQUEST'))
                cur_cost = []
                for i in range(1, NUM_STATES + 1):
                    cur_cost.append((state_values[i] - state_estimates[i]) ** 2)
                run_results.append(sqrt(np.mean(cur_cost)))
            RL_cleanup()
            cur_agent_results.append(run_results)
        all_results.append(cur_agent_results)

    #Averge the results for each parameter setting over the 30 runs
    avg_results = []
    for i in range(len(all_results)):
        avg_results.append([np.mean(run) for run in zip(*all_results[i])])

    print "\nPlotting the results..."
    plt.ylabel('RMSVE')
    plt.xlabel("Episodes")
    plt.axis([1, num_episodes, 0, 1])
    for i in range(len(avg_results)):
        plt.plot([episode for episode in range(num_episodes)], avg_results[i], GRAPH_COLOURS[i], label="Alpha = " + str(ALPHAS[i]) + " AGENT = " + str(AGENTS[i]))
    plt.legend(loc='center', bbox_to_anchor=(0.60,0.90))
    plt.show()
    print "\nFinished!"
