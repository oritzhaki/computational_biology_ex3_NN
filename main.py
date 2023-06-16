# Or Itzhaki 209335058 and Tal Ishon 315242297

"""
HOW DOES IT WORK:

architecture:
- input layer size 16
- HL1 size 64
- HL2 size 32
- Output size 1

test, train = data[:20%], data[21%:]

X_train, y_train = split(train)
X_test, y_test = split(test)

init population - each p in pop is a NN

"""

import csv
import sys

import numpy as np
from ypstruct import structure
from collections import defaultdict
import time

fitness_func_counter = 0  # counts how many times the fitness metrc is called
X_train, X_test, y_train, y_test = None, None, None, None

VEC_SIZE = 3104
INPUT_SIZE = 16
HL1 = 64
HL2 = 32
OUTPUT_SIZE = 1


class NN:
    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.W3 = None

    def update(self, matrices):
        self.W1 = matrices[0]
        self.W2 = matrices[1]
        self.W2 = matrices[2]

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


neuralNet = NN()


def reshape(vector, shape):
    array = np.array(vector)
    return array.reshape(shape[0], shape[1])


def vec_to_matrix(vec):
    """
    Split the vector into 3 matrices
    """
    vector = vec.copy()
    matrices = [np.reshape(vector[:INPUT_SIZE * HL1], (INPUT_SIZE, HL1)),
                np.reshape(vector[INPUT_SIZE * HL1:INPUT_SIZE * HL1 + HL1 * HL2], (HL1, HL2)),
                np.reshape(vector[INPUT_SIZE * HL1 + HL1 * HL2:], (HL2, OUTPUT_SIZE))]
    return matrices


# Calculate the fitness score for an individual code configuration
def measure_fitness(vector):
    global fitness_func_counter
    fitness_func_counter += 1

    matrices = vec_to_matrix(vector)  # matrices: [mat1, mat2, mat3]
    neuralNet.update(matrices)

    predictions = np.zeros(shape=(X_train.shape[0]))
    for sample_idx in range(X_train.shape[0]):
        r1 = X_train[sample_idx, :]
        y_hat = neuralNet.feedforward(r1)
        predictions[sample_idx] = round(y_hat)
    correct_predictions = np.where(predictions == y_train)[0].size
    accuracy = (correct_predictions / y_train.size) * 100
    return accuracy


# todo: edit crossover to be numerical and not letter permutation sensitive
def crossover(p1, p2):
    # Generate random crossover points
    point1 = np.random.randint(1, len(p1.sequence))
    point2 = np.random.randint(point1, len(p1.sequence))
    # Create empty offspring arrays
    s1 = np.empty_like(p1.sequence)
    s2 = np.empty_like(p2.sequence)
    # Copy selected segment from parent1 to offspring1 and parent2 to offspring2
    s1[point1:point2 + 1] = p1.sequence[point1:point2 + 1]
    s2[point1:point2 + 1] = p2.sequence[point1:point2 + 1]
    # Find the mapping between parent1 and offspring2 segment
    mapping = {}
    for i in range(point1, point2 + 1):
        mapping[p1.sequence[i]] = s2[i]
    # Apply PMX crossover for the remaining values
    for i in range(len(p2.sequence)):
        if i < point1 or i > point2:
            value = p2.sequence[i]
            while value in mapping:
                value = mapping[value]
            s1[i] = value
    # Find the mapping between parent2 and offspring1 segment
    mapping = {}
    for i in range(point1, point2 + 1):
        mapping[p2.sequence[i]] = s1[i]
    # Apply PMX crossover for the remaining values
    for i in range(len(p1.sequence)):
        if i < point1 or i > point2:
            value = p1.sequence[i]
            while value in mapping:
                value = mapping[value]
            s2[i] = value
    # add new sequences to offsprings:
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    c1.sequence = s1
    c2.sequence = s2
    return c1, c2


# todo: edit mutate to be numerical and not letter permutation sensitive
def mutate(x, mu):
    # mutation == swap between two random letters
    y = x.deepcopy()
    flag = np.random.rand(*x.sequence.shape) <= mu
    ind = np.argwhere(flag)
    if len(ind) > 0:
        for i1 in ind:
            i2 = np.random.randint(len(x.sequence))
            y.sequence[i1.item()], y.sequence[i2] = y.sequence[i2], y.sequence[i1.item()]
    return y


def roulette_wheel_selection(p):
    # get index of individual for selection based on the roulette wheel mechanism. it works like this: c is accumulative
    # sum of the probability each index (representing someone from the population), and r is a number from 0 to the
    # total sum. the larger prob the person had, the more likely he is that the number will fall on him
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


def run_ga(problem, params):
    # Problem Information
    fitness_func = problem.fitness_func

    # Parameters
    maxit = params.maxit
    npop = params.npop
    pc = params.pc
    nc = int(np.round(pc * npop / 2) * 2)  # amount of offsprings
    mu = params.mu

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.sequence = None
    empty_individual.fitness = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.fitness = -np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        # initialize vector for each model in pop
        pop[i].sequence = np.random.uniform(low=-0.1, high=0.1,
                                            size=(X_train.shape[1], VEC_SIZE))
        pop[i].fitness = fitness_func(pop[i].sequence)
        if pop[i].fitness > bestsol.fitness:
            bestsol = pop[i].deepcopy()

    # Best Cost of each iteration
    bestcost = np.empty(maxit)
    avgcost = np.empty(maxit)
    bestseq = []

    should_break = 0  # counter to check if got best sequence already
    # Main Loop
    for it in range(maxit):
        print(f"Generation: {it}")
        #  create probabilities - better solutions are more likely to give offspring
        costs = np.array([x.fitness for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs / avg_cost
        probs = np.exp(2 * costs)  # todo: play with hyper param for the exp
        probs /= np.sum(probs)

        avgcost[it] = avg_cost

        popc = []  # offspring population
        for _ in range(nc // 2):  # creation of offsprings
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            c1, c2 = crossover(p1, p2)

            c1 = mutate(c1, mu)
            c2 = mutate(c2, mu)

            c1.fitness = fitness_func(c1.sequence)
            if c1.fitness > bestsol.fitness:
                bestsol = c1.deepcopy()

            c2.fitness = fitness_func(c2.sequence)
            if c2.fitness > bestsol.fitness:
                bestsol = c2.deepcopy()

            popc.append(c1)
            popc.append(c2)

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.fitness, reverse=True)  # descending
        pop = pop[:npop]  # take the population of size npop with the best fitness
        # Store Best Cost
        # print(f"Best fitness: {bestsol.fitness}")

        bestcost[it] = bestsol.fitness
        bestseq.append(bestsol.sequence)

        # check if bestcost doesn't change in the last x iterations
        if np.array_equal(bestseq[it - 1], bestseq[it]):
            should_break += 1
        else:
            should_break = 0

        # check if best solution hasn't changed in a long time and if so exit
        if should_break == 20:
            break_flag = 0
            break
        # if it > 70:
        #     break

    return bestsol.sequence, bestcost, avgcost


def load_data(path):
    f = open(path, "r")
    lines = f.readlines()

    X, y = [], []

    for line in lines:
        values = line.rstrip('\n').split("  ")
        X.append(values[0])
        y.append(values[1])

    size = len(lines)
    size_train = int(size * 0.8)
    return X[:size_train], X[size_train:], y[:size_train], y[size_train:]


def write_best_sol_to_file(best_sol):
    matrices = vec_to_matrix(best_sol)
    # Save the matrices to a file named "matrices.txt"
    np.savetxt("wnet.txt", (matrices[0], matrices[1], matrices[2]), delimiter=',')


if __name__ == '__main__':
    args = sys.argv

    # todo: make sure if input is one file that we need to split. If so - the following code is suitable here
    X_train, X_test, y_train, y_test = load_data(args[1])

    X_train = np.array(X_train).reshape(-1, 16)
    X_test = np.array(X_test).reshape(-1, 16)
    y_train = np.array(y_train).reshape(-1, 16)
    y_test = np.array(y_test).reshape(-1, 16)

    problem = structure()
    problem.fitness_func = measure_fitness
    problem.size = 3104  # 16*64 + 64*32 + 32

    # describe algorithm hyperparameters
    params = structure()
    params.maxit = 200  # number of iterations
    params.npop = 50  # size of population
    params.pc = 4  # ratio of offspring:original population (how many offspring to be created each iteration)
    params.mu = 0.05  # percentage of vector to receive mutation

    best_solution, best_fitness_array, avg_fitness_array = run_ga(problem, params)

    write_best_sol_to_file(best_solution)

    print(f"The total number of calls to fitness function: {fitness_func_counter}")

    # if (len(best_fitness_array) >= 70):
    #     with open('spreadcount.csv', 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(best_fitness_array[:70])
    #     with open('averagecount.csv', 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(avg_fitness_array[:70])
