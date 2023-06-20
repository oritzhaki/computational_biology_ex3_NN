# Or Itzhaki 209335058 and Tal Ishon 315242297
import csv
import sys
import random
import numpy as np
from ypstruct import structure
from collections import defaultdict
import time

fitness_func_counter = 0  # counts how many times the fitness metrc is called
X_train, X_test, y_train, y_test = None, None, None, None

VEC_SIZE = 16
INPUT_SIZE = 16
OUTPUT_SIZE = 1


class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = None # np.random.randn(num_features) * np.sqrt(2 / num_features)
        # self.bias = 0.0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return np.where(weighted_sum >= 0, 1, 0)

    def evaluate_accuracy(self, inputs, targets):
        predictions = self.predict(inputs)
        accuracy = np.mean(predictions == targets)
        return accuracy

    def update(self, weights):
        self.weights = weights


perceptron = Perceptron(num_features=16)

def reshape(vector, shape):
    array = np.array(vector)
    return array.reshape(shape[0], shape[1])


def vec_to_matrix(vec):
    """
    Split the vector into 3 matrices
    """
    return vec.copy()


# Calculate the fitness score for an individual code configuration
def measure_fitness(vector):
    global fitness_func_counter, X_train, y_train
    fitness_func_counter += 1

    matrices = vec_to_matrix(vector)  # matrices: [mat1, mat2, mat3]
    perceptron.update(matrices)
    return perceptron.evaluate_accuracy(X_train, y_train)


def crossover(p1, p2):
    """
    how it works example:
    p1=[-0.50368893 -0.91327073 -0.39029557 -0.88672639  0.696925   -0.79873602]
    p2=[ 0.36131945 -0.77794683 -0.80804455  0.67462618  0.73060522 -0.50873338]
    alpha=[0, 0, 0, 0, 1, 1]
    c1=[ 0.36131945 -0.77794683 -0.80804455  0.67462618  0.696925   -0.79873602]
    c2=[-0.50368893 -0.91327073 -0.39029557 -0.88672639  0.73060522 -0.50873338]
    """
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = [random.choice([0, 1]) for _ in range(len(p1.sequence))]
    for i in range(len(p1.sequence)):
        c1.sequence[i] = alpha[i] * p1.sequence[i] + (1 - alpha[i]) * p2.sequence[i]
        c2.sequence[i] = (1 - alpha[i]) * p1.sequence[i] + alpha[i] * p2.sequence[i]
    return c1, c2


def mutate(x, mu, sigma):
    """
    how it works example:
    [ 0.19814601 -0.31742422 -0.34353525  0.6856634  -0.22293646  0.16598561]
    [False False False  True  True  True]
    [ 0.19814601  -0.31742422  -0.34353525 -17.22924339  9.98403477   -4.0379242 ]
    """
    y = x.deepcopy()
    # flag = np.random.rand(*x.sequence.shape) <= mu  # creates array of T and F
    # ind = np.argwhere(flag)  # indexes of T's
    # y.sequence[ind] += sigma * np.random.randn()  # add the mutation
    for i in range(VEC_SIZE):
        if np.random.rand() < mu:
            y.sequence[i] += np.random.randn()
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
    sigma = params.sigma

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
        pop[i].sequence = np.random.randn(VEC_SIZE) * np.sqrt(1 / VEC_SIZE)
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
        costs = np.array([x.fitness*5 for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs / avg_cost
        probs = np.exp(5 * costs)  # todo: play with hyper param for the exp
        probs /= np.sum(probs)

        avgcost[it] = avg_cost

        popc = []  # offspring population
        for _ in range(nc // 2):  # creation of offsprings
            # choose two parents according to probs
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            # create two offsprings from the crossover of the two parents
            c1, c2 = crossover(p1, p2)

            # mutate the offspring
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

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
        print(f"Accuracy: {bestsol.fitness}\n")

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
    X = np.array([list(map(int, string)) for string in X])
    y = np.array(y).astype(int)
    return X[:size_train], X[size_train:], y[:size_train], y[size_train:]


def write_best_sol_to_file(best_sol):
    matrices = vec_to_matrix(best_sol)
    # Save the matrices to a file named "matrices.txt"
    np.savetxt("wnet.txt", matrices, delimiter=',')


if __name__ == '__main__':
    args = sys.argv

    # todo: make sure if input is one file that we need to split. If so - the following code is suitable here
    X_train, X_test, y_train, y_test = load_data(args[1])

    problem = structure()
    problem.fitness_func = measure_fitness
    problem.size = VEC_SIZE  # 16*64 + 64*32 + 32

    # describe algorithm hyperparameters
    params = structure()
    params.maxit = 100  # number of iterations
    params.npop = 50  # size of population
    params.pc = 4  # ratio of offspring:original population (how many offspring to be created each iteration)
    params.mu = 0.01  # percentage of vector to receive mutation
    params.sigma = 0.5  # todo: find good sigma (size of mutation)

    best_solution, best_fitness_array, avg_fitness_array = run_ga(problem, params)
    print(best_fitness_array)
    write_best_sol_to_file(best_solution)


