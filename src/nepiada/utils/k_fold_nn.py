import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import random
import os

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

def create_model(input_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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
    model = create_model(len(X_train[0]))
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {i+1} - Test Accuracy: {scores[1]*100:.2f}%")
    k_fold_results.append(scores[1])

# Print the average accuracy over all folds
average_accuracy = np.mean(k_fold_results)
print(f"Average Test Accuracy over {len(file_names)} folds: {average_accuracy*100:.2f}%")
