import numpy as np
from sklearn.cluster import KMeans
import random

def calculate(data):
    n_agents = 9
    sum_ranges = 0
    valid_agents_count = 0

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

# Function to load data from file
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

# Apply K-Means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X.reshape(-1, 1))

# Predict clusters
predicted_labels = kmeans.predict(X.reshape(-1, 1))

# Counting the labels in each cluster
labels_in_cluster_0 = np.bincount(true_labels[predicted_labels == 0], minlength=2)
labels_in_cluster_1 = np.bincount(true_labels[predicted_labels == 1], minlength=2)

# Print label counts per cluster
print(f"Label 0's in Cluster 0: {labels_in_cluster_0[0]}")
print(f"Label 1's in Cluster 0: {labels_in_cluster_0[1]}")
print(f"Label 0's in Cluster 1: {labels_in_cluster_1[0]}")
print(f"Label 1's in Cluster 1: {labels_in_cluster_1[1]}")
