import numpy as np
import matplotlib.pyplot as plt

environment = [0, 0, 0, 0, 0, 0, 1]

starting_point = 3

episodes = 100
alpha = 0.1
gamma = 0.5
qvalues = [[0 for s in range(len(environment))] for a in range(2)]

for i in range(episodes):

    current_point = starting_point

    # episode terminates when state 0 or 6 are reached
    while 0 <= current_point < len(environment):

        action = np.random.rand() > 0.5
        if action:  # action == 1 go right
            next_point = current_point + 1
        else:       # action == 0 go left
            next_point = current_point - 1

        reward = environment[next_point]

        qvalues[current_point, action] += alpha * (reward + gamma * max(Q[next_point]) - Q[current_point, action])

        current_point = next_point
