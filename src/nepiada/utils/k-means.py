import numpy as np
import random
from sklearn.cluster import KMeans
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
    # Since each line in the file is a single data point, return a list with one random value
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

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X.reshape(-1, 1))

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_normalized)

# Predict clusters
predicted_labels = kmeans.predict(X_normalized)

# Counting the labels in each cluster
labels_in_cluster_0 = np.bincount(true_labels[predicted_labels == 0], minlength=2)
labels_in_cluster_1 = np.bincount(true_labels[predicted_labels == 1], minlength=2)

# Print label counts per cluster
print(f"Label 0's in Cluster 0: {labels_in_cluster_0[0]}")
print(f"Label 1's in Cluster 0: {labels_in_cluster_0[1]}")
print(f"Label 0's in Cluster 1: {labels_in_cluster_1[0]}")
print(f"Label 1's in Cluster 1: {labels_in_cluster_1[1]}")
