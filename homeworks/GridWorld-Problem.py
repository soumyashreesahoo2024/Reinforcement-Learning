import numpy as np

# Define the gridworld environment
gridshape = 4
num_states = gridshape * gridshape
num_actions = 4
transitions = []
rewards = []

for s in range(num_states):
    row, col = divmod(s, gridshape)
    # Define transitions and rewards for non-terminal states
    if row == 0:
        # Top row
        next_state = s
        reward = -1
        transitions.append([next_state] * num_actions)
        rewards.append([reward] * num_actions)
    elif row == gridshape - 1:
        # Bottom row
        next_state = s
        reward = -1
        transitions.append([next_state] * num_actions)
        rewards.append([reward] * num_actions)
    else:
        if col == 0:
            # Left column
            next_state = s
            reward = -1
            transitions.append([next_state] * num_actions)
            rewards.append([reward] * num_actions)
        elif col == gridshape - 1:
            # Right column
            next_state = s
            reward = -1
            transitions.append([next_state] * num_actions)
            rewards.append([reward] * num_actions)
        else:
            # Middle states
            north = s - gridshape
            south = s + gridshape
            east = s + 1
            west = s - 1
            transitions.append([north, south, east, west])
            rewards.append([-1, -1, -1, -1])

    #print("rewards==", rewards)
    
    
    
# Define the transitions and rewards for the special states A and B
A_state = 0
A_reward = 10
A_transitions = [A_state] * num_actions
A_rewards = [A_reward] * num_actions
transitions[A_state] = A_transitions
rewards[A_state] = A_rewards
print("\n for state A, transition-{} \n and rewards-{}".format(transitions,rewards))




B_state = 15
B_reward = 5
B_transitions = [B_state] * num_actions
B_rewards = [B_reward] * num_actions
transitions[B_state] = B_transitions
rewards[B_state] = B_rewards
print("\n for state B, transition-{} and rewards-{}".format(transitions,rewards))




# Define the equiprobable random policy
policy = np.ones([num_states, num_actions]) / num_actions
print('\n policy==',policy)





# Define the discount factor and the convergence threshold
gamma = 1.0
theta = 1e-4





# Perform iterative policy evaluation
V = np.zeros(num_states)
while True:
    delta = 0
    for s in range(num_states):
        print("state s==", s)
        v = V[s]
        v_new = 0
        for a in range(num_actions):
            next_state = transitions[s][a]
            reward = rewards[s][a]
            prob = policy[s][a]
            v_new += prob * (reward + gamma * V[next_state])
            print(v_new)
        V[s] = v_new
        delta = max(delta, abs(v - V[s]))
        print('delta==', delta)
        print("\n iterative policy evaluated ===", V)
    if delta < theta:
        break




# Print the estimated state value function
print(V.reshape([gridshape, gridshape]))
