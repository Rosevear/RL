import matplotlib.pyplot as plt
import numpy as np
import argparse

numEpisodes = 2
maxStepsInEpisode = 1000
numRuns = 2000
result = np.zeros((numEpisodes, maxStepsInEpisode))
OPTIMISTIC = 'optimistic'
UCB = 'ucb'
EXTENSION = ".dat"
ALGO1_DATA_FILE = "RL_UCB_EXP_OUT"
ALGO2_DATA_FILE = "RL_OPTIMISTIC_EXP_OUT"

parser = argparse.ArgumentParser(description='Plot the data from the most recently run experiment')
parser.add_argument('--algo', choices=[OPTIMISTIC, UCB], nargs='?', const=OPTIMISTIC, type=str, default=OPTIMISTIC, help='The algorithm to be used by the agent that is compared to the epsilon greedy strategy')
args = parser.parse_args()

for i in range(numEpisodes):
    if args.algo == UCB:
        cur_result_file = ALGO1_DATA_FILE + str(i) + EXTENSION
    else:
        cur_result_file = ALGO2_DATA_FILE + str(i) + EXTENSION
    j = 0
    with open(cur_result_file, 'r') as cur_results:
        for line in cur_results:
            result[i, j] = float(line)
            j += 1

print "\nPlotting the results..."
plt.ylabel('% Optimal Action')
plt.xlabel("Steps")
plt.axis([1, 1000, 0.0, 1.0])
if args.algo == UCB:
    plt.plot(result[0, :], 'r-', label="realistic epsilon greedy: epsilon = 0.1, Q1 = 0")
    plt.plot(result[1, :], 'b-', label="UCB: c = 1")
else:
    plt.plot(result[0, :], 'r-', label="optimistic greedy: epsilon = 0, Q1 = 5")
    plt.plot(result[1, :], 'b-', label="realistic epsilon greedy: epsilon = 0.1, Q1 = 0")

plt.legend(loc='center', bbox_to_anchor=(0.6,0.25))
plt.show()
print "\nFinished!"
