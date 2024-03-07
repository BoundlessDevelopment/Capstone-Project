import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import random

k = 2  # Number of repetitions per agent
n = 5  # Number of turns to look back (adjust this as needed)

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

# Load and preprocess data from the file
file_name = '../tester/data_5.txt'  # Update this path as needed
X, y = load_data_from_file(file_name)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Create and train the CNN model
    cnn_model = create_cnn_model((9, n * 2, 1))
    history = cnn_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

    # Plot training accuracy
    plt.plot(history.history['accuracy'])
    plt.title('Model Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # Evaluate the model on the test set
    scores = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Set Accuracy: {scores[1]*100:.2f}%")

