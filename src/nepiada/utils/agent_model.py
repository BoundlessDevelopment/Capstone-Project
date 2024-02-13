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

def generate_synthetic_data(num_samples, num_agents):
    X, y = [], []
    for _ in range(num_samples):
        sample = []
        for _ in range(num_agents):
            if random.random() > 0.5:
                sample.extend([(random.uniform(0, 20), random.uniform(0, 20))] * k)
            else:
                sample.extend([None] * k)
        X.append(preprocess_input(sample))
        y.append(random.choice([0, 1]))
    return np.array(X), np.array(y)

class AgentModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probability of being adversarial

# Generate synthetic data

num_samples = 100  # Number of samples
num_agents = 9     # Number of agents per sample
X, y = generate_synthetic_data(num_samples, num_agents)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
agent_model = AgentModel()
agent_model.train(X_train, y_train)


def load_data_from_file(file_name):
    X, y = [], []
    with open(file_name, 'r') as file:
        for line in file:
            example_input_str, label_str = line.strip().split('*')
            # Assuming example_input is stored as a string representation of a list
            example_input = eval(example_input_str)
            label = int(label_str)
            processed_input = preprocess_input(example_input)
            X.append(processed_input)
            y.append(label)
    return np.array(X), np.array(y)

# Load and preprocess the data from the file
file_name = '../tester/data.txt'
#X, y = load_data_from_file(file_name)

# Fit the model with this data
agent_model = AgentModel()
agent_model.train(X, y)

# Optionally, save the trained model
with open('agent_model.pkl', 'wb') as file:
    pickle.dump(agent_model.model, file)

# Example prediction with probability
base_input = [None, (14.4557, 8.9661), None, None, (20, 19), None, None, None, (5.0, 34.0)]
example_input = k * base_input
processed_input = preprocess_input(example_input)
prob_adversarial = agent_model.predict_proba([processed_input])
print("Probability of being adversarial:", prob_adversarial[0])
