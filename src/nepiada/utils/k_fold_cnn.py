import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import random
import os

n = 5  # Number of turns to look back (adjust this as needed)

"""

t1 = [1 0 1]
t2 = [1 0 1]
a3 = [1 1 1]

[1 1 1]
[1 0 1]
[1 0 1] y = 0
[1 1 1]

[1 0 1]
[1 0 1]
[1 0 1] y = 1
[1 1 1]
"""

def preprocess_input(input_data):
    processed_data = []
    for item in input_data:
        # Replace None with a placeholder tuple (0, 0)
        processed_data.extend(item if item is not None else (0, 0))

    # Convert the list to a NumPy array and reshape
    processed_data = np.array(processed_data).reshape(9, n, 2)

    # Flatten the tuples to single values if necessary
    processed_data = processed_data.reshape(9, n * 2)

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

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# List of file names - update with the actual paths
file_names = ['../tester/data_5_1.txt', '../tester/data_5_2.txt', '../tester/data_5_3.txt', '../tester/data_5_4.txt', '../tester/data_5_5.txt']

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

    # Create and train the CNN model
    cnn_model = create_cnn_model((9, n * 2, 1))
    cnn_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

    # Evaluate the model
    scores = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {i+1} - Test Accuracy: {scores[1]*100:.2f}%")
    k_fold_results.append(scores[1])

# Print the average accuracy over all folds
average_accuracy = np.mean(k_fold_results)
print(f"Average Test Accuracy over {len(file_names)} folds: {average_accuracy*100:.2f}%")
