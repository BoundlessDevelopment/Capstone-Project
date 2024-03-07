import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def calculate(data):
    n_agents = 9  # Number of agents
    sum_ranges = 0  # To store the sum of average ranges for all valid agents
    valid_agents_count = 0  # To count agents with valid ranges

    for i in range(n_agents):
        update1 = data[i]
        update2 = data[i + n_agents] if i + n_agents < len(data) else None

        # Check if both updates are valid (not None)
        if update1 is not None and update2 is not None:
            range_x = abs(update2[0] - update1[0])
            range_y = abs(update2[1] - update1[1])
            avg_range = (range_x + range_y) / 2
            sum_ranges += avg_range
            valid_agents_count += 1

    # Calculate the average of the average ranges of valid agents
    return sum_ranges / valid_agents_count if valid_agents_count > 0 else 0

def preprocess_input(input_data):
    return calculate(input_data)

def load_data_and_labels_from_file(file_name):
    X = []
    labels = []
    with open(file_name, 'r') as file:
        for line in file:
            example_input_str, label_str = line.strip().split('*')
            example_input = eval(example_input_str)
            processed_input = preprocess_input(example_input)
            X.append(processed_input)
            labels.append(int(label_str))
    return np.array(X), np.array(labels)

# File name
file_name = '../tester/data_2_1.txt'  # Update this to the file you want to use

# Load data and labels
X, true_labels = load_data_and_labels_from_file(file_name)

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X.reshape(-1, 1))

# Plotting the data on separate number lines for each label
plt.figure(figsize=(10, 6))
for i, label in enumerate(np.unique(true_labels)):
    plt.scatter(X_normalized[true_labels == label], np.full(sum(true_labels == label), i), alpha=0.5, label=f'Label {label}')
plt.xlabel('Normalized Score')
plt.yticks([0, 1], ['Label 0', 'Label 1'])
plt.title('Color-Coded Scatterplot by Label on Separate Number Lines')
plt.legend()
plt.show()
