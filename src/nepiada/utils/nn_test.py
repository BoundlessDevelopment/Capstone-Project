import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
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

# Load and preprocess data from the file
file_name = '../tester/data_1.txt'  # Update this path as needed
X, y = load_data_from_file(file_name)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Model
def create_model(input_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    input_size = len(X_train[0])
    model = create_model(input_size)
    
    # Fit the model with the training data
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

    # Plot training accuracy
    plt.plot(history.history['accuracy'])
    plt.title('Model Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # Evaluate the model on the test set
    scores = model.evaluate(X_test, y_test)
    print("\nTest Set Accuracy: %.2f%%" % (scores[1]*100))
