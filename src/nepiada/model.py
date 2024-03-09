import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import pickle

k = 2  # Number of repetitions per agent

def preprocess_input(input_data):
    processed_data = []
    for item in input_data:
        processed_data.extend(item if item is not None else [0, 0])
    return processed_data

class AgentModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probability of being adversarial

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

# Load and preprocess the data from the file
file_name = '../tester/data.txt'
X, y = load_data_from_file(file_name)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Fit the model with the training data
    agent_model = AgentModel()
    agent_model.train(X_train, y_train)

    # Initialize counters
    correct_predictions_0, correct_predictions_1 = 0, 0
    total_0, total_1 = 0, 0

    for i, test_input in enumerate(X_test):
        prob_adversarial = agent_model.predict_proba([test_input])[0]
        predicted_label = 1 if prob_adversarial >= 0.5 else 0
        actual_label = y_test[i]

        if actual_label == 0:
            total_0 += 1
            if predicted_label == actual_label:
                correct_predictions_0 += 1
        elif actual_label == 1:
            total_1 += 1
            if predicted_label == actual_label:
                correct_predictions_1 += 1

    # Calculate and print accuracies
    accuracy_0 = correct_predictions_0 / total_0 if total_0 != 0 else 0
    accuracy_1 = correct_predictions_1 / total_1 if total_1 != 0 else 0
    overall_accuracy = (correct_predictions_0 + correct_predictions_1) / len(X_test)

    print(f"Model Accuracy for label 0: {accuracy_0}")
    print(f"Model Accuracy for label 1: {accuracy_1}")
    print(f"Overall Model Accuracy: {overall_accuracy}")

    # Optionally, save the trained model
    # with open('agent_model.pkl', 'wb') as file:
    #     pickle.dump(agent_model.model, file)
