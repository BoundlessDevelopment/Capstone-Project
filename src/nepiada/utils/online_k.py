import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

def calculate(data, num_agents):
    n_agents = num_agents
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


def preprocess_input(input_data, num_agents):
    return calculate(input_data, num_agents)

def load_data_and_labels_from_file(file_name):
    X = []
    labels = []
    with open(file_name, 'r') as file:
        for line in file:
            example_input_str, label_str = line.strip().split('*')
            example_input = eval(example_input_str)
            processed_input = preprocess_input(example_input, 9) # 9 is hard-coded here because of the data file
            X.append(processed_input)
            labels.append(int(label_str))
    return np.array(X), np.array(labels)

def predict_label(data_point, model):
    predicted_cluster = model.predict(data_point.reshape(-1, 1))[0]
    return predicted_cluster

if __name__ == "__main__":
    # File name
    file_name = '../tester/data_10_1.txt'  # Update this to the file you want to use

    # Load data and labels
    X, true_labels = load_data_and_labels_from_file(file_name)

    # Initialize MiniBatchKMeans with 2 clusters
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=42)

    # Use two dummy inputs for initial fitting
    dummy_inputs = np.array([1, 0])  # Adjust these values as needed
    kmeans.partial_fit(dummy_inputs.reshape(-1, 1))

    # Initialize the list to store predicted labels
    predicted_labels = []

    # Process each data point one by one
    for i in range(len(X)):
        data_point = X[i].reshape(-1, 1)

        # Update the model with the new data point
        kmeans.partial_fit(data_point)

        # Predict the cluster for the updated model
        predicted_label = predict_label(data_point, kmeans)
        predicted_labels.append(predicted_label)

        # Print progress
        print(f"Processed data point {i + 1} of {len(X)}, Predicted label: {predicted_label}")

    # Calculate the accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")