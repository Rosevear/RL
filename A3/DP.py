import random
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

NUM_STATES = 100
THETA = 0.00000000000000000001
TERMINAL_REWARD = 1

#Initialize the state values
policy = [min(state, NUM_STATES - state) for state in range(NUM_STATES + 1)]
delta = None
value_estimates = [random.uniform(0, 0) for state in range(NUM_STATES)]

#Set the values for the terminal states because we know them already
value_estimates.append(TERMINAL_REWARD)
value_estimates_log = [] #To keep track of old value estimates for graphing
num_iters = 0

def compute_action_value(state, action):
    trial_success_state = state + action
    trial_fail_state = state - action
    #The reward for each non-terminal state is 0, so we only need to use the values of the possible future states
    return (PROB_HEADS * value_estimates[trial_success_state]) + (PROB_TAILS * value_estimates[trial_fail_state])

def iterate_values():
    global delta, num_iters
    delta = 0
    for state in range(1, NUM_STATES): #Don't update the 0 and 100 dollar states, as these are terminal
        old_estimate = value_estimates[state]
        action_values = []
        for action in range(min(state, NUM_STATES - state) + 1): #There are as many actions in a state as the number of dollars in a state, or however much money you need to win: +1 represents the all in state
            action_values.append(compute_action_value(state, action))
        value_estimates[state] = max(action_values)
        delta = max(delta, abs(old_estimate - value_estimates[state]))

    num_iters += 1
    if num_iters in [1, 2, 3]:
        value_estimates_log.append(list(value_estimates))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find an optimal policy and set of value functions for the Gambler\'s Problem for a given probability of flipping heads')
    parser.add_argument('p', type=float, help='User specified probability of flipping heads. Must be within 0 and 1')
    args = parser.parse_args()

    if (args.p < 0) or (args.p > 1):
        exit("Invalid probability provided: you must specify a probability value between 1 and 0, inclusive")

    PROB_HEADS = args.p
    PROB_TAILS = 1 - PROB_HEADS

    print("Estimating value functions...")
    iterate_values()
    while not delta < THETA:
        iterate_values()

    print("Getting the optimal policy...")
    for state in range(1, NUM_STATES):
        print("State: " + str(state))
        action_values = []
        for action in range(min(state, NUM_STATES - state) + 1):
            action_values.append(compute_action_value(state, action))
            print("action: " + str(action))

        #BREAK TIES METHOD ONE: TAKE THE FIRST MAXIMUM ENCOUNTERED FROM RIGHT TO LEFT
        cur_max = -1
        cur_max_action = -1
        for i in range(len(action_values) - 1, -1, -1):
            if action_values[i] > cur_max:
                cur_max = action_values[i]
                cur_max_action = i
        policy[state] =  cur_max_action

        #BREAK TIES METHOD TWO: TAKE THE FIRST ENCOUNTERED MAXIMUM FROM LEFT TO RIGHT
        #policy[state] = np.argmax(np.array(action_values[1:]))

    #Just some output to monitor what is going on as it runs
    print("Number of iterations: " + str(num_iters))
    for state in range(NUM_STATES + 1):
        print("state: " + str(state) + " estimate: " + str(value_estimates[state]) + " action: " + str(policy[state]))

    #Plot the value estimates for a few initial sweeps and the last sweep
    print("Plotting the results...")
    plt.ylabel('Value Estimates')
    plt.xlabel("Capital")
    plt.axis([1, 99, 0.0, 1.0])
    line_colours = ['r-', 'b-', 'g-']
    for i in range(3):
        plt.plot(value_estimates_log[i], line_colours[i], label=str(i + 1) + " Sweeps ")
    plt.plot(value_estimates, 'y-', label= str(num_iters) + " Sweeps")
    plt.legend(loc='upper left', bbox_to_anchor=(0.6,0.25))
    plt.show()

    #Plot the final policy
    plt.ylabel('Final policy (stake)')
    plt.xlabel("Capital")
    plt.axis([1, 99, 1, 50])
    plt.plot(policy, 'y-', label="Captial to stake")
    plt.legend(loc='upper left', bbox_to_anchor=(0.6,0.25))
    plt.show()

    print("Finished!")
