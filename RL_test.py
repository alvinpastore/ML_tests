from __future__ import division
import optparse
import numpy as np
import matplotlib.pyplot as plt


def plot_position(environment,current_point):
    plt.figure()
    plt.plot(np.arange(len(environment)), np.zeros(len(environment))+0.5, 'bo')
    plt.axis([-0.5, len(environment) - 0.5, 0, 1])
    plt.plot(current_point, 0.4, 'rx')
    plt.title('Current position: ' + str(current_point))
    plt.show()


def game(environment, episodes, alpha, gamma, epsilon, verbose, plot):

    # initialise q-values
    Q = [[0 for s in range(2)] for a in range(len(environment))]

    for episode in range(episodes):

        # start episode at midpoint
        current_point = int(np.floor(len(environment)/2))

        if plot:
            plot_position(environment, current_point)

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

            if plot:
                plot_position(environment, current_point)

        if verbose:
            Q_string = [['{0:.4g}'.format(flt) for flt in sublist] for sublist in Q]
            print('Episode: {0} \t Qs: {1}'.format(episode, Q_string))



def main():
    op = optparse.OptionParser("usage: %prog [options] n_states w_state n_episodes \n"
                               "n_states    = number of states, (e.g. minimum setup: [0,1,2] = 3 states)\n"
                               "w_state     = winner state (right or left, i.e. the end of lineworld)\n"
                               "n_episodes  = number of episodes of training\n"
                               "alpha       = learning rate, in [0,1]\n"
                               "gamma       = discount factor, in [0,1)\n"
                               "epsilon     = exploration factor, in [0,1] (0 greedy, 1 random)",
                               version="%prog 1.0, a simple lineworld example of Q-learning\n"
                                       "winner state has reward 1, all other states 0")

    op.add_option("-v", "--verbose", action="store_true", dest="v",
                  default=False, help="print episode number and Q-values at the end of each episode")

    op.add_option("-p", "--plot", action="store_true", dest="p",
                  default=False, help="plot simple position representation of the agent")

    (options, args) = op.parse_args()

    if len(args) != 6:
        op.error("wrong number of arguments")

    # parse number of states
    n_states = int(args[0])
    if n_states < 3:
        op.error("Need at least 3 states")
    env = [0] * n_states

    # parse winner state
    winner_state = args[1].lower()
    if winner_state not in ['right', 'left']:
        op.error("Winner state can be only right or left")
    else:
        if winner_state == "right":
            env[len(env)-1] = 1
        else:
            env[0] = 1

    # parse number of episodes
    n_episodes = int(args[2])
    if n_episodes < 10:
        op.error("Train for at least 10 episodes")

    # parse parameters
    a = float(args[3])
    g = float(args[4])
    e = float(args[5])
    if a < 0 or g < 0 or e < 0 or a > 1 or g >= 1 or e > 1:
        op.error("Parameter/s not in range")

    game(env, n_episodes, a, g, e, options.v, options.p)


if __name__ == '__main__':
    main()

