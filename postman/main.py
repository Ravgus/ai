import numpy as np
import matplotlib.pyplot as plt

# Define the environment boundaries and define a 3D numpy array to hold the current Q-values for each state and action pair: Q(s,a).
environment_rows = 11
environment_columns = 11

q_values = np.zeros((environment_rows, environment_columns, 4))

actions = ['up', 'right', 'down', 'left']

# Reward points
rewards = np.full((environment_rows, environment_columns), -100.)
rewards[0, 5] = 100. 

# White spaces
aisles = {}
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.
        
# To visualize environment
for row in rewards:
    print(row)
    
# Will print smth like:
#[-100. -100. -100. -100. -100.  100. -100. -100. -100. -100. -100.]
#[-100.   -1.   -1.   -1.   -1.   -1.   -1.   -1.   -1.   -1. -100.]
#[-100.   -1. -100. -100. -100. -100. -100.   -1. -100.   -1. -100.]
#[-100.   -1.   -1.   -1.   -1.   -1.   -1.   -1. -100.   -1. -100.]
#[-100. -100. -100.   -1. -100. -100. -100.   -1. -100. -100. -100.]
#[-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
#[-100. -100. -100. -100. -100.   -1. -100. -100. -100. -100. -100.]
#[-100.   -1.   -1.   -1.   -1.   -1.   -1.   -1.   -1.   -1. -100.]
#[-100. -100. -100.   -1. -100. -100. -100.   -1. -100. -100. -100.]
#[-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
#[-100. -100. -100. -100. -100. -100. -100. -100. -100. -100. -100.]

def is_terminal_state(current_row_index, current_column_index):
    """
    Function to determine if the specified location is a terminal state
    """

    if rewards[current_row_index, current_column_index] == -1.:
        return False
    else:
        return True

def get_starting_location():
    """
    Function to choose a random non-terminal starting location.
    """

    current_row_index = np.random.randint(environment_rows)

    current_column_index = np.random.randint(environment_columns)

    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)

    return current_row_index, current_column_index

def get_next_action(current_row_index, current_column_index, epsilon):
    """
    Function to choose the next action, according to the epsilon value.
    """

    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)

def get_next_location(current_row_index, current_column_index, action_index):
    """
    Function to get the next location based on the chosen action.
    """

    new_row_index = current_row_index
    new_column_index = current_column_index
    
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

def get_shortest_path(start_row_index, start_column_index):
    """
    Function that will get the shortest path between any location within the city 
    that the postman is allowed to travel and the item packaging location.
    """

    if is_terminal_state(start_row_index, start_column_index):
        return []
    else: 
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])

        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
            
    return shortest_path

epsilon = 0.9 # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 # discount factor for future rewards
learning_rate = 0.9 # the rate at which the AI agent should learn
n_training_episodes = 1000

for episode in range(n_training_episodes):

    row_index, column_index = get_starting_location()

    while not is_terminal_state(row_index, column_index):

        action_index = get_next_action(row_index, column_index, epsilon)

        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')

# Finds a few samples of shortest paths

print(get_shortest_path(3, 9)) #starting at row 3, column 9
print(get_shortest_path(5, 0)) #starting at row 5, column 0
print(get_shortest_path(9, 5)) #starting at row 9, column 5

#[[3, 9], [2, 9], [1, 9], [1, 8], [1, 7], [1, 6], [1, 5], [0, 5]]
#[[5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [4, 7], [3, 7], [2, 7], [1, 7], [1, 6], [1, 5], [0, 5]]
#[[9, 5], [9, 6], [9, 7], [8, 7], [7, 7], [7, 6], [7, 5], [6, 5], [5, 5], [5, 6], [5, 7], [4, 7], [3, 7], [2, 7], [1, 7], [1, 6], [1, 5], [0, 5]]

def visualize_game(rewards, shortest_path):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(environment_columns+1), minor=True)
    ax.set_yticks(np.arange(environment_rows+1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)

    for row_index in range(environment_rows):
        for column_index in range(environment_columns):
            if rewards[row_index, column_index] == -100.:
                color = 'black'
            elif rewards[row_index, column_index] == -1.:
                color = 'white'
            elif rewards[row_index, column_index] == 100.:
                color = 'green'
            else:
                color = 'gray'

            ax.add_patch(plt.Rectangle((column_index, row_index), 1, 1, facecolor=color, edgecolor='black'))

    for path_step in shortest_path:
        ax.add_patch(plt.Rectangle((path_step[1], path_step[0]), 1, 1, facecolor='yellow', edgecolor='black'))

    ax.add_patch(plt.Rectangle((5, 0), 1, 1, facecolor='green', edgecolor='black'))  # Target location

    plt.gca().invert_yaxis()
    plt.show()

visualize_game(rewards, get_shortest_path(9, 5))





