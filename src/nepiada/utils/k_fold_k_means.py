import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Number of turns to look back
n = 5

def preprocess_input(input_data):
    processed_data = []
    for item in input_data:
        # Replace None with a placeholder tuple (0, 0)
        processed_data.extend(item if item is not None else (0, 0))

    # Convert the list to a NumPy array and reshape
    processed_data = np.array(processed_data).reshape(9, n, 2)

    # Flatten the tuples to single values
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

def flatten_and_normalize_data(data):
    # Flatten the data
    flattened_data = data.reshape(data.shape[0], -1)

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(flattened_data)

    return normalized_data

# Function to align and evaluate clusters with true labels
def evaluate_clusters(labels, true_labels):
    # Assuming the true labels are 0 and 1
    count_0 = sum(labels == true_labels)
    count_1 = sum(labels != true_labels)
    
    # Calculate accuracy
    if count_0 > count_1:
        accuracy = count_0 / len(true_labels)
    else:
        accuracy = count_1 / len(true_labels)

    return accuracy

# List of file names
file_names = ['../tester/data_5_1.txt', '../tester/data_5_2.txt', '../tester/data_5_3.txt', '../tester/data_5_4.txt', '../tester/data_5_5.txt']

# K-Fold Cross Validation
k_fold_accuracies = []
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

    # Load test data and labels
    X_test, y_test = load_data_from_file(test_file)

    # Prepare data
    X_train_normalized = flatten_and_normalize_data(X_train)
    X_test_normalized = flatten_and_normalize_data(X_test)

    # Apply K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train_normalized)

    # Predict clusters for the test set
    test_labels = kmeans.predict(X_test_normalized)

    # Evaluate the clustering
    accuracy = evaluate_clusters(test_labels, y_test)
    k_fold_accuracies.append(accuracy)

    print(f"Fold {i+1} - Accuracy: {accuracy*100:.2f}%")

# Average accuracy over all folds
average_accuracy = np.mean(k_fold_accuracies)
print(f"Average Accuracy over {len(file_names)} folds: {average_accuracy*100:.2f}%")
