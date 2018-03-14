#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Cody Rosevear
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Cody Rosevear
  Last Modified on: 17/9/2017
"""

from __future__ import division
from rl_glue import *  # Required for RL-Glue
import argparse
import numpy as np
import sys

#Agent algorithm types
UCB = 'ucb'
OPTIMISTIC = 'optimistic'

#constants and such
EXTENSION = ".dat"
ALGO1_DATA_FILE = "RL_UCB_EXP_OUT"
ALGO2_DATA_FILE = "RL_OPTIMISTIC_EXP_OUT"

def saveResults(data, dataSize, filename): # data: floating point, dataSize: integer, filename: string
    with open(filename, "w") as dataFile:
        for i in range(dataSize):
            dataFile.write("{0}\n".format(data[i]))

#Note: This function has been taken and modified from its original source: Andrew Jacobsen's post (Tuesday, 12 September 2017, 8:58 PM) on the CMPUT609 course forum: https://eclass.srv.ualberta.ca/mod/forum/discuss.php?d=849395
def getOptimalAction():
    return int(RL_env_message("get optimal action"))

if __name__ == "__main__":
    #Determine which type of experiment is being run
    parser = argparse.ArgumentParser(description='Run the 10-armed testbed for bandit algorithms')
    parser.add_argument('--algo', choices=[OPTIMISTIC, UCB], nargs='?', const=OPTIMISTIC, type=str, default=OPTIMISTIC, help='The algorithm to be used by the agent that is compared to the epsilon greedy strategy')
    args = parser.parse_args()
    if args.algo == UCB:
        RLGlue("w1_env", "w1_ucb_agent")
    else:
        RLGlue("w1_env", "w1_agent")

    #We use the epsiodes infrastructure code to index the current agent algorithm being run
    #so that agents get compared across the same randomly generated distributions in the testbed
    #instead of generating them again for different parameter settings
    #THEY ARE NOT ACTUAL EPSIODES AS DEFINED IN RL
    numEpisodes = 2
    maxStepsInEpisode = 1000
    numRuns = 2000
    agent_actions = np.zeros((numRuns, numEpisodes, maxStepsInEpisode))
    optimal_actions = np.zeros((numRuns, 1))
    result = np.zeros((numEpisodes, maxStepsInEpisode))

    print "\nPrinting one dot for every run: {0} total Runs to complete".format(numRuns)
    for k in range(numRuns):
        RL_init()
        optimal_actions[k, 0] = getOptimalAction()

        for i in range(numEpisodes):
            RL_start()
            num_steps = 0

            while num_steps < maxStepsInEpisode:
                rl_step_result = RL_step()
                agent_actions[k, i, num_steps] = rl_step_result[2]
                num_steps += 1
            RL_agent_end(0)

        RL_cleanup()
        print ".",
        sys.stdout.flush()

    print "\nDone running the experiments"
    print "\nSaving the results to the filesystem..."

    #Compute the optimal action percentage for each time step, across all runs
    for i in range(numEpisodes):
        for j in range(maxStepsInEpisode):
            count = 0
            for optimal_action, agent_action in zip(optimal_actions, agent_actions[:, i, j]):
                if int(optimal_action) == int(agent_action):
                    count += 1
            result[i, j] = (count / numRuns)

        if args.algo == UCB:
            saveResults(result[i, :], maxStepsInEpisode, ALGO1_DATA_FILE + str(i) + EXTENSION)
        else:
            saveResults(result[i, :], maxStepsInEpisode, ALGO2_DATA_FILE + str(i) + EXTENSION)
    print "\nFinished!"
