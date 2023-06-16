import sys
import numpy as np


class NN:
    def __init__(self, params):
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.init_matrices(params)

    def init_matrices(self, params):
        self.W1 = params[0]
        self.W2 = params[1]
        self.W2 = params[2]

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
    # Read the matrices from the file
    loaded_data = np.loadtxt(wnet, delimiter=',')

    # Unpack the loaded data into separate matrices
    loaded_mat1, loaded_mat2, loaded_mat3 = loaded_data

    return loaded_mat1, loaded_mat2, loaded_mat3  # return tuple of all 3 matrices

def write_to_file(prediction):
    f = open("output.txt", "w")
    # todo: make sure predictions are written into file s.t each pred is in a new line
    f.write(prediction)
    f.close()


def main(wnet, data):
    params = read_file(wnet)
    NN_model = NN(params)
    # todo: check if needs to loop over data or do it all at once - for predictions
    prediction = NN_model.feedforward(data)
    write_to_file(prediction)


if __name__ == "__main__":
    args = sys.argv
    main(args[1], args[2])
