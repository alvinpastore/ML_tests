from __future__ import division
import optparse
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_position(environment,current_point):
    plt.figure()
    plt.plot(np.arange(len(environment)), np.zeros(len(environment))+0.5, 'bo')
    plt.axis([-0.5, len(environment) - 0.5, 0, 1])
    plt.plot(current_point, 0.4, 'rx')
    plt.title(current_point)
    plt.show()


def main():
    op = optparse.OptionParser("usage: RL_lineworld.py [options] n w \n"
                               "n = number of states\n"
                               "w = winner state (reward = 1, 0 for all other states)",
                               version="RL_lineworld 0.1, a simple lineworld example of Q-learning")

    op.add_option("-v","--verbose", dest="verbose",
                  default="False", type="string",
                  help="print the episode number and the Q-values at the end of each episode")


if __name__ == '__main__':
    main()

def game():
    environment = [0, 0, 0, 0, 0, 0, 1]

    verbose = str(sys.argv[1])
    episodes = str(sys.argv[])
    alpha = 0.1
    gamma = 0.5
    epsilon = 0.1

    Q = [[0 for s in range(2)] for a in range(len(environment))]


    for episode in range(episodes):

        # start episode at midpoint
        current_point = np.floor(len(environment)/2)

        #plot_position(environment, current_point)

        # episode terminates when state 0 or 6 are reached
        while 0 < current_point < len(environment)-1:

            # explore with probability epsilon or when the Q values are the same
            if np.random.rand() < epsilon or Q[current_point][0] == Q[current_point][1]:
                action = np.random.rand() > 0.5
            else:
                action = np.argmax(Q[current_point])

            # translate actions into movement
            if action:  # action == 1 go right
                next_point = current_point + 1
            else:       # action == 0 go left
                next_point = current_point - 1

            # get reward from environment
            reward = environment[next_point]

            # Q-learning update rule
            Q[current_point][action] += alpha * (reward + gamma * max(Q[next_point]) - Q[current_point][action])

            # move to next point
            current_point = next_point

            #plot_position(environment, current_point)

        if verbose:
            print "Episode:", episode
            print "Q-values", Q
