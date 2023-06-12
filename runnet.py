import sys
import numpy as np


class NN:
    def __init__(self, params):
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.init_matrices(params)

    def init_matrices(self, params):
        self.W1 = reshape(params[0], params[1])
        self.W2 = reshape(params[2], params[3])
        self.W2 = reshape(params[4], params[5])

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def feedforward(self, X):
        Z1 = np.dot(X, self.W1)
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2)
        A2 = self.relu(Z2)
        Z3 = np.dot(A2, self.W3)
        A3 = self.sigmoid(Z3)
        return A3


def reshape(vector, shape):
    array = np.array(vector)
    return array.reshape(shape[0], shape[1])


def read_file(wnet):
    """
    File 'wnet' contains in lines:
    lines[0]: weights W1
    lines[1]: shape: (input, HL1)
    lines[2]: weights W2
    lines[3]: shape: (HL1, HL2)
    lines[4]: weights W3
    lines[5]: shape: (HL2, output)
    """
    f = open(wnet, "r")
    lines = f.readlines()

    return lines[0], lines[1], lines[2], lines[3], lines[4], lines[5]


def write_to_file(prediction):
    f = open("output.txt", "w")
    f.write(prediction)
    f.close()


def main(wnet, data):
    params = read_file(wnet)
    NN_model = NN(params)
    prediction = NN_model.feedforward(data)
    write_to_file(prediction)


if __name__ == "__main__":
    args = sys.argv
    main(args[0], args[1])
