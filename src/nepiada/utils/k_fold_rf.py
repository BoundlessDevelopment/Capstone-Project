import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

k = 2  # Number of repetitions per agent

def preprocess_input(input_data):
    processed_data = []
    for item in input_data:
        processed_data.extend(item if item is not None else [0, 0])
    return processed_data

def load_data_from_file(file_name):
    X, y = [], []
    with open(file_name, 'r') as file:
        for line in file:
            example_input_str, label_str = line.strip().split('*')
            example_input = eval(example_input_str)
            label = int(label_str)
            processed_input = preprocess_input(example_input)
            X.append(processed_input)
            y.append(label)
    return np.array(X), np.array(y)

def create_random_forest_model():
    return RandomForestClassifier(n_estimators=100)

# List of file names - update with the actual paths
file_names = ['../tester/data_1_1.txt', '../tester/data_1_2.txt', '../tester/data_1_3.txt', '../tester/data_1_4.txt', '../tester/data_1_5.txt']

# K-Fold Cross Validation
k_fold_results = []
for i in range(len(file_names)):
    # Select the test file for this fold
    test_file = file_names[i]

    # Combine the other files for training
    train_files = [f for f in file_names if f != test_file]
    X_train, y_train = [], []
    for file in train_files:
        X, y = load_data_from_file(file)
        X_train.extend(X)
        y_train.extend(y)
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Load test data
    X_test, y_test = load_data_from_file(test_file)

    # Create and train the model
    model = create_random_forest_model()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fold {i+1} - Test Accuracy: {accuracy*100:.2f}%")
    k_fold_results.append(accuracy)

# Print the average accuracy over all folds
average_accuracy = np.mean(k_fold_results)
print(f"Average Test Accuracy over {len(file_names)} folds: {average_accuracy*100:.2f}%")
