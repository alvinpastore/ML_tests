from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.close('all')

def linearise(t,c):
    return int(c * t[0] + t[1])

def randomise_initial_state(c,r):
    return np.array([np.random.randint(r),np.random.randint(c)])

# RL setup
episodes = 100
epsilon = 0.1
alpha = 0.1
gamma = 0.5

# world
rows = 4
cols = 5
nStates = rows*cols
world = np.zeros((rows,cols))
world[rows-1,cols-1] = 100
win_state = np.array([3,4])
nActions = 4 # north south west east

# figure and step counter
fig = plt.figure()
ax = fig.add_subplot(111)
plt.show()
counter = []

# optimism in the face of uncertainty (initialise qvalues high)
#Q = np.ones((nStates,nActions)) * 10
Q = [[10.0 for x1 in xrange(nActions)] for x2 in xrange(nStates)]
R = [ [0 for x1 in xrange(nActions)] for x2 in xrange(nStates)]
T = [[[0 for x1 in xrange(nStates)]  for x2 in xrange(nActions)] for x3 in xrange(nStates)]

for episode in range(episodes):
    step_counter = 0
    state = randomise_initial_state(cols,rows)
    print 'random initial state',state
    print 'episode', episode

    # keep playing until final state is reached
    while not (state[0] == win_state[0] and state[1] == win_state[1]):
        step_counter += 1
        print 'current state',state

        if np.random.rand() < epsilon:
            action = int(np.floor(np.random.rand() * 4))
            print 'picked random action'
        else:
            action = int(Q[linearise(state,cols)].index(max(Q[linearise(state,cols)])))
            print 'picked best action'

        print 'action',action

        # using action to move to next state
        if action == 0:
            next_state = [state[0],state[1]-1]      # go left (west)
        elif action == 1:
            next_state = [state[0]+1,state[1]]      # go down (south)
        elif action == 2:
            next_state = [state[0], state[1]+1]     # go right (east)
        elif action == 3:
            next_state = [state[0]-1, state[1]]     # go up (north)
        print 'next_state',next_state

        # hitting walls
        if next_state[0] < 0 or next_state[0] > rows - 1:
            reward = -1
            next_state[0] = state[0]
        elif next_state[1] < 0 or next_state[1] > cols - 1:
            reward = -1
            next_state[1] = state[1]

        # not hitting walls
        else:
            reward = world[next_state[0]][next_state[1]]

        print 'reward',reward

        # get linear indexes
        state_lin = linearise(state,cols)
        next_state_lin = linearise(next_state,cols)

        # Update Reward function, Transition Probabilities and Q Values
        R[state_lin][action] += alpha * (reward - R[state_lin][action])
        T[state_lin][action][next_state_lin] += alpha * (1 - T[state_lin][action][next_state_lin])
        sum_transitions_max_Q = 0

        for state_t in xrange(len(T[state_lin][action])):
            T[state_lin][action][state_t] += alpha * (0 - T[state_lin][action][state_t])
            sum_transitions_max_Q += T[state_lin][action][state_t] * max(Q[next_state_lin])

        Q[state_lin][action] = R[state_lin][action] + (gamma * sum_transitions_max_Q)

        # move to next state
        state = next_state

        print
        for q in Q:
            print '{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(q[0],q[1],q[2],q[3])

        # update figure
        ax.clear()
        ax.plot(state[1],state[0],'xr',markersize=20,)
        ax.imshow(world, interpolation='nearest')
        ax.set_title('Episode: ' + str(episode))
        ax.get_xaxis().set_ticks(range(cols))
        ax.get_yaxis().set_ticks(range(rows))
        plt.draw()
        plt.pause(0.01)

    counter.append(step_counter)

print 'average steps',sum(counter) / episodes

plt.waitforbuttonpress()