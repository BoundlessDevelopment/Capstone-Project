import numpy as np
import matplotlib.pyplot as plt

def calculate(data):
    n_agents = 9
    sum_ranges = 0
    valid_agents_count = 0
    #print(data)
    for i in range(n_agents):
        updates = [data[j] for j in range(i, len(data), n_agents)]  # Get all updates for the agent
        if all(update is not None for update in updates):  # Check if all updates are non-None
            # Calculate range for each pair of updates and average them
            ranges = []
            for k in range(len(updates) - 1):
                range_x = abs(updates[k+1][0] - updates[k][0])
                range_y = abs(updates[k+1][1] - updates[k][1])
                ranges.append((range_x + range_y) / 2)
            avg_range = sum(ranges) / len(ranges)
            sum_ranges += avg_range
            valid_agents_count += 1

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
file_name = '../tester/data_10.txt'  # Update this to the file you want to use

# Load data and labels
X, true_labels = load_data_and_labels_from_file(file_name)

# Plotting a histogram with actual frequency counts and a jitter plot for each label
plt.figure(figsize=(12, 6))

# Histogram plot with actual frequency counts for each label
plt.subplot(1, 2, 1)
for label in np.unique(true_labels):
    distribution_label = 'Distribution of Truthful Data' if label == 0 else 'Distribution of Adversarial Data'
    plt.hist(X[true_labels == label], bins=20, alpha=0.5, label=distribution_label, density=False)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()

# Jitter plot for each label
plt.subplot(1, 2, 2)
for label in np.unique(true_labels):
    point_label = 'Truthful Data Points' if label == 0 else 'Adversarial Data Points'
    jitter = np.random.normal(0, 0.1, sum(true_labels == label))
    plt.scatter(X[true_labels == label], jitter + label, alpha=0.5, label=point_label)
plt.title('Jitter Plot for Each Label', fontsize=14)
plt.xlabel('Score', fontsize=12)
plt.yticks(np.unique(true_labels), ['Truthful', 'Adversarial'], fontsize=10)
plt.legend()

plt.tight_layout()
plt.show()