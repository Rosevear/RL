To get the plot for part 1, simply run the program: python mountaincar_exp.py
The default values and number of runs correspond the the parameters in the assignment.

To get the 3d plot, supply the --plot argument: python mountaincar_exp.py --plot
This will plot, over a single run, the cost to go function over 1000 num_episodes

To specify other parameter settings, use as follows: python mountaincar_exp.py -e epsilon -a alpha -l lambda -n num_tilings -runs num_runs
The parameters for the bonus question were averaged over 50 runs
