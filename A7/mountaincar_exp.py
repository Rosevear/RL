#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent")

import numpy as np
import random
import argparse
import json
from scipy import stats
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mountain car experiment solved by the Sarsa lambda algorithm')
    parser.add_argument('--plot', choices=[True, False], default=False, nargs='?', type=bool, help='Runs the experiment to generate a 3D plot of the action values')
    parser.add_argument('-runs', nargs='?', type=int, default=50, help='Number of runs to average the results over. Default = 50')
    parser.add_argument('-e', nargs='?', type=float, default=0.0, help='Epsilon paramter value for to be used by the agent when selecting actions epsilon greedy style. Default = 0.0')
    parser.add_argument('-a', nargs='?', type=float, default=0.1, help='Alpha parameter which specifies the step size for the update rule. Default value = 0.1')
    parser.add_argument('-l', nargs='?', type=float, default=0.90, help='Lambda parameter which specifies the decay rate of the eligibility traces. Default value=0.90')
    parser.add_argument('-n', nargs='?', type=int, default=8, help='Number of tilings to use with the tile coder. Default value = 8')
    parser.add_argument('-mem', nargs='?', type=int, default=4096, help='The amount of memory allocated for hashing used by the tile coder. Default value = 4096')
    args = parser.parse_args()

    print("Running experiment for params: EPSILON: {}, ALPHA: {}, LAMBDA: {}, NUM_TILINGS: {}, IHT_SIZE: {}".format(args.e, args.a, args.l, args.n, args.mem))
    agent_params = {"EPSILON": args.e, "ALPHA": args.a, "LAMBDA": args.l, 'NUM_TILINGS': args.n, 'IHT_SIZE': args.mem}
    RL_agent_message(('PARAMS', json.dumps(agent_params)))

    if args.plot == False:
        #These are the results for the default parameters
        target_mean = -41475.08
        target_standard_error = 392.807294334

        num_episodes = 200
        num_runs = args.runs
        steps = np.zeros([num_runs, num_episodes])
        run_reward_totals = []

        for run in range(num_runs):
            print("run number : {}".format(run))
            cur_run_reward = 0
            np.random.seed(run)
            random.seed(run)
            RL_init()
            for episode in range(num_episodes):
                print("episode number : {}".format(episode))
                RL_episode(0)
                steps[run, episode] = RL_num_steps()
                cur_run_reward += RL_return()
            run_reward_totals.append(cur_run_reward)

        np.save('steps', steps)
        mean_reward = np.mean(run_reward_totals)
        standard_error = stats.sem(run_reward_totals)
        print("Results for params EPSILON: {}, ALPHA: {}, LAMBDA: {}, NUM_TILINGS: {}, IHT_SIZE: {}\n".format(args.e, args.a, args.l, args.n, args.mem))
        print("Mean Reward: {}\n".format(str(mean_reward)))
        print("Standard Error: {}\n".format(str(standard_error)))

        #Save the results and check statistical significance
        with open("sweep_results.txt", "a") as sweep_results:
            sweep_results.write("Results for params EPSILON: {}, ALPHA: {}, LAMBDA: {}, NUM_TILINGS: {}, IHT_SIZE: {}\n".format(args.e, args.a, args.l, args.n, args.mem))
            sweep_results.write("Mean Reward: {}\n".format(str(mean_reward)))
            sweep_results.write("Standard Error: {}\n".format(str(standard_error)))
            if (mean_reward > target_mean) and (abs(target_mean) - abs(mean_reward) > (2.5 * max(target_standard_error, standard_error))):
                sweep_results.write("BETTER PARAMS FOUND: EPSILON: {}, ALPHA: {}, LAMBDA: {}, NUM_TILINGS: {}, IHT_SIZE: {}\n".format(args.e, args.a, args.l, args.n, args.mem))

    else:
        num_episodes = 1000
        plot_range = 50
        np.random.seed(0)
        random.seed(0)

        print("Performing a single 1000 episode long run to compute values for the 3D plot...")
        RL_init()
        for episode in range(num_episodes):
            print("episode number : {}".format(episode))
            RL_episode(0)
        (position_values, velocity_values, plot_values) = RL_agent_message(('PLOT', plot_range))

        print("Plotting the 3D plot")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Mountain Car cost-to-go values');
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Value')
        ax.plot_wireframe(position_values, velocity_values, plot_values[:, 0:50], rstride=5, cstride=5)
        plt.show()
        print("Finished!")
