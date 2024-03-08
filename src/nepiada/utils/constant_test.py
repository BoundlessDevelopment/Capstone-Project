import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random

# Function to calculate the absolute difference in all adjacent updates
def calculate(data):
    n_agents = 9
    sum_ranges = 0
    valid_agents_count = 0

    for i in range(n_agents):
        updates = [data[j] for j in range(i, len(data), n_agents)]  # Get all updates for the agent
        if all(update is not None for update in updates):  # Check if all updates are non-None
            ranges = []
            for k in range(len(updates) - 1):
                range_x = abs(updates[k+1][0] - updates[k][0])
                range_y = abs(updates[k+1][1] - updates[k][1])
                ranges.append((range_x + range_y) / 2)
            avg_range = sum(ranges) / len(ranges)
            sum_ranges += avg_range
            valid_agents_count += 1

    return sum_ranges / valid_agents_count if valid_agents_count > 0 else 0

# Function to preprocess input data
def preprocess_input(input_data):
    return calculate(input_data)

def load_data_and_labels_from_file(file_name):
    X = []
    labels = []
    with open(file_name, 'r') as file:
        for line in file:
            example_input_str, label_str = line.strip().split('*')
            label = int(label_str)
            labels.append(label)
            if label == 0 and random.random() < 0.5:  # 50% probability for label 0
                example_input = [(1, 1)] * 18  # Replace with all (1,1)
            else:
                example_input = eval(example_input_str)
            processed_input = preprocess_input(example_input)
            X.append(processed_input)
    return np.array(X), np.array(labels)



# List of file names
file_names = ['../tester/data_10_1.txt', '../tester/data_10_2.txt', '../tester/data_10_3.txt', '../tester/data_10_4.txt', '../tester/data_10_5.txt']

# K-Fold Cross Validation
k_fold_accuracies = []
for i in range(len(file_names)):
    # Select the test file for this fold
    test_file = file_names[i]

    # Combine the other files for training
    train_files = [f for f in file_names if f != test_file]
    X_train, y_train = [], []
    for file in train_files:
        X, y = load_data_and_labels_from_file(file)
        X_train.extend(X)
        y_train.extend(y)
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Load test data and labels
    X_test, y_test = load_data_and_labels_from_file(test_file)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train.reshape(-1, 1), y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test.reshape(-1, 1))

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    k_fold_accuracies.append(accuracy)

    print(f"Fold {i+1} - Accuracy: {accuracy*100:.2f}%")

# Average accuracy over all folds
average_accuracy = np.mean(k_fold_accuracies)
print(f"Average Accuracy over {len(file_names)} folds: {average_accuracy*100:.2f}%")
