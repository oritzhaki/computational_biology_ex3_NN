import numpy as np

def load_data(path):
    f = open(path, "r")
    lines = f.readlines()

    X, y = [], []

    for line in lines:
        values = line.rstrip('\n').split("  ")
        X.append(values[0])
        y.append(values[1])

    X = np.array([list(map(int, string)) for string in X])
    y = np.array(y).astype(int)
    return X, y

class Perceptron:
    def __init__(self, num_features, weights):
        self.num_features = num_features
        self.weights = weights # np.random.randn(num_features) * np.sqrt(2 / num_features)
        # self.bias = 0.0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return np.where(weighted_sum >= 0, 1, -1)

    def evaluate_accuracy(self, inputs, targets):
        predictions = self.predict(inputs)
        accuracy = np.mean(predictions == targets)
        return accuracy

    def update(self, weights):
        self.weights = weights


if __name__=="__main__":
    X_train, y_train = load_data("nn1_train.txt")
    perceptron = Perceptron(num_features=16)
    accuracy = perceptron.evaluate_accuracy(X_train, y_train)
    print("Accuracy:", accuracy)