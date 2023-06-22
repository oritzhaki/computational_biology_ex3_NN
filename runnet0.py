import numpy as np
from buildnet0 import NN, Weight, sigmoid, relu
#
# def print_model_accuracy(X_test, y_test):
#     accuracy = calculate_net_accuracy(y_test, BEST_MODEL.predict(X_test))
#     print(f"Test Accuracy: {accuracy}")
# def calculate_net_accuracy(y_train, predictions):
#     correct_predictions = np.sum(predictions == y_train)
#     accuracy = correct_predictions / len(y_train)
#     return float(accuracy)
#
#
# def measure_fitness(network, X_train, y_train):
#     predictions = network.predict(X_train)
#     return calculate_net_accuracy(y_train, predictions)
#
# def load_data_temp(path):
#     f = open(path, "r")
#     lines = f.readlines()
#
#     X, y = [], []
#
#     for line in lines:
#         values = line.rstrip('\n').split("  ")
#         X.append(values[0])
#         y.append(values[1])
#
#     size = len(lines)
#     size_train = int(size * 0.8)
#     X = np.array([list(map(int, string)) for string in X])
#     y = np.array(y).astype(int)
#     return X, y

def load_data(filename):
    """
    Load test data to check model's accuracy
    """
    f = open(filename, 'r')
    lines = f.readlines()

    test = []
    for line in lines:
        string = line.strip()
        # Converting the input string into a list of integers
        test.append([int(bit) for bit in string])

    return np.array(test)


def get_weights_for_model(W1, W2, W3):
    temp1 = Weight(1, 1, activation=lambda x: relu(x))
    temp2 = Weight(1, 1, activation=lambda x: relu(x))
    temp3 = Weight(1, 1, activation=lambda x: sigmoid(x))
    temp1.update_weights(W1)
    temp2.update_weights(W2)
    temp3.update_weights(W3)
    return temp1, temp2, temp3


def initiate_best_model(filename):
    weights = np.load(filename)
    W1 = weights['arr1']
    W2 = weights['arr2']
    W3 = weights['arr3']

    weights1, weights2, weights3 = get_weights_for_model(W1, W2, W3)

    return NN(weights1, weights2, weights3)


def write_predictions_to_file(filename):
    results = open(filename, "w")
    for label in predictions:
        results.write(str(label) + "\n")

    results.close()

if __name__ == "__main__":
    # CHECK FOR US IF IT WORKS
    # X_test, y_test = load_data_temp("testnet0.txt")
    # print_model_accuracy(X_test, y_test)


    # Load data from test file
    X_test = load_data("testnet0.txt")

    # Create best model according to best model received from GA
    BEST_MODEL = initiate_best_model("wnet0.npz")

    # Test data on best model
    predictions = BEST_MODEL.predict(X_test)

    # Write predictions to results file
    write_predictions_to_file("result0.txt")
