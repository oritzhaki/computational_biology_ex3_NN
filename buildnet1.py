import sys
import time

import numpy as np
import copy
import random
import matplotlib.pyplot as plt

####################################################
##################### GLOBALS ######################
####################################################

X_train, X_test, y_train, y_test = None, None, None, None

INPUT_SIZE = 16
HL1 = 32
HL2 = 8
OUTPUT_SIZE = 1

TRAIN_PERCENT = 0.8
BEST_FITNESS_LIST = []

POP_SIZE = 100
MUTATION_RATE = 0.3
GENERATIONS_LIMIT = 100
ELITE_PERCENT = 0.1
PERCENT_NOT_MANIPULATED = 0.05
STUCK_THRESHOLD = 10
TRY_BETTER_MUTATIONS = 6
THRESHOLD = 0.4


######################################################


def load_data(train_path, test_path):
    global X_train, X_test, y_train, y_test
    f_train = open(train_path, "r")
    lines_train = f_train.readlines()
    f_test = open(test_path, "r")
    lines_test = f_test.readlines()
    input_train_X, input_train_y, input_test_X, input_test_y = [], [], [], []

    for line in lines_train:
        values = line.rstrip('\n').split("  ")
        input_train_X.append(values[0])
        input_train_y.append(values[1])

    for line in lines_test:
        values = line.rstrip('\n').split("  ")
        input_test_X.append(values[0])
        input_test_y.append(values[1])

    X_train = np.array([list(map(int, string)) for string in input_train_X])
    y_train = np.array(input_train_y).astype(int)
    X_test = np.array([list(map(int, string)) for string in input_test_X])
    y_test = np.array(input_test_y).astype(int)


def calculate_accuracy(y, predictions):
    correct_predictions = np.sum(predictions == y)
    accuracy = correct_predictions / len(y)
    return float(accuracy)


def measure_fitness(network):
    """
    The fitness function in our GA is calculated according to the model's accuracy.
    """
    global X_train, y_train
    predictions = network.predict(X_train)
    return calculate_accuracy(y_train, predictions)


class GA:
    """
    The GA class represents an instance of a genetic algorithm (GA).
    Genetic algorithms are search and optimization algorithms inspired by the process of natural selection and evolution.
    The GA class encapsulates the logic and components necessary to run the GA, such as the population of individuals
    (represented by neural network models), mutation operators, crossover operators, and fitness evaluation methods.
    It provides an interface to initialize and execute the genetic algorithm, typically involving processes such as
    population initialization, selection, crossover, mutation, and fitness evaluation.
    """

    def __init__(self):
        self.pop_size = POP_SIZE

    def rank_selection(self, pop):
        """
         The rank_selection function is a function used to rank individuals in a population based on their fitness levels.
         In the context of neural networks, each individual represents a specific neural network model.
         The rank selection function assigns a rank or score to each individual, typically based on their performance
         or fitness, and this ranking is used to determine their probability of selection for reproduction.
         Individuals with higher fitness scores are assigned higher ranks and have a greater likelihood of being
         selected for crossover and reproduction in the next generation.
        """
        ranked_pop = sorted(pop, key=lambda network: measure_fitness(network))
        probs = [rank / len(ranked_pop) for rank in range(1, len(ranked_pop) + 1)]
        parents = random.choices(ranked_pop, weights=probs, k=len(pop))
        return parents


    def crossover(self, n1, n2):
        """
        The crossover method is a function that performs crossover on a NN as part of a genetic algorithm (GA).
        In the context of GA, crossover refers to the process of combining genetic information from two parent
        neural networks to create offspring networks with a combination of their characteristics.
        This function typically selects certain portions or attributes of the parent networks and combines them to
        generate new individuals that inherit traits from both parents.
        """
        # Create the new neural network which will be our offspring
        network = NN()

        for i in range(len(n1.weights)):
            alpha = np.random.uniform(0.0, 1.0, size=n1.weights[i].weights.shape)
            network.weights[i].weights = alpha * n1.weights[i].weights + \
                                         (1 - alpha) * n2.weights[i].weights
        return network


    def mutate(self, network):
        """
        The mutate method is a function that applies mutations to a NN as part of a genetic algorithm (GA).
        The purpose of this function is to introduce random variations or modifications to the neural network's
        parameters, such as weights or biases, in order to explore different regions of the solution space and
        potentially improve the performance or adaptability of the network.
        """
        for weight in network.weights:
            flag = np.random.rand(*weight.weights.shape) < MUTATION_RATE
            mutation_indices = np.where(flag)
            mutations_num = len(mutation_indices[0])

            if mutations_num < 2:  # less than 2 - no actual mutation will occur
                continue

            # Choose two random indices for swapping weights
            rand_indices = np.random.choice(mutations_num, size=2, replace=False)
            swap_indices = mutation_indices[0][rand_indices]

            weight.swap(swap_indices)

    def run_ga(self):
        """
        The run_ga method runs the GA according to the global variables set in the beginning of the code.
        """
        global X_train, y_train, BEST_FITNESS_LIST
        # Creating an initial pop of neural networks
        pop = []
        for _ in range(self.pop_size):
            network = NN()
            pop.append(network)

        best_fitness = 0
        stuck = 0
        for generation in range(GENERATIONS_LIMIT):
            print(f"Generation {generation}:")

            # Evaluating the fitness of each network in the current pop
            fitness_list = []
            for network in pop:
                fitness = measure_fitness(network)
                fitness_list.append(fitness)

            current_best_fit = max(fitness_list)
            print(f"Current best fitness: {current_best_fit}")
            BEST_FITNESS_LIST.append(current_best_fit)

            # Check if the current generation has achieved maximum fitness
            if current_best_fit == 1.0:
                print(f"Maximum accuracy of {current_best_fit}")
                break
            # Check for early convergence:
            if current_best_fit > best_fitness:
                best_fitness = current_best_fit
                # Reset stuck count if there's improvement
                stuck = 0
            else:
                # Increment stuck count if no improvement
                stuck += 1

            if stuck >= STUCK_THRESHOLD:
                # If no improvement for STUCK_THRESHOLD GENERATIONS_LIMIT, stop the process
                print(f"Reached convergence - FITNESS: {best_fitness}")
                break

            # Selecting the top performing networks (elites)
            sorted_indices = np.argsort(fitness_list)[::-1]
            elite_pop = [pop[i] for i in sorted_indices[:int(self.pop_size * ELITE_PERCENT)]]
            # Remaining pop after elites have been selected
            remaining_pop = list(set(pop) - set(elite_pop))

            # Creating offspring pop via crossover
            offspring_pop = []
            offsprings_len = self.pop_size - len(elite_pop)

            # Rank Selection
            selected_parents = self.rank_selection(remaining_pop)

            for _ in range(offsprings_len):
                p1 = np.random.choice(selected_parents)
                p2 = np.random.choice(elite_pop)
                offspring = self.crossover(p1, p2)
                offspring_pop.append(offspring)

            # Save some offspring not_manipulated for the next gen pop
            num_offsprings_not_manipulated = int(offsprings_len * PERCENT_NOT_MANIPULATED)
            not_manipulated_offspring = offspring_pop[:num_offsprings_not_manipulated]

            # Mutate the remaining (touched) offspring pop
            for offspring in offspring_pop[num_offsprings_not_manipulated:]:
                self.mutate(offspring)

            pop = elite_pop + not_manipulated_offspring + offspring_pop[num_offsprings_not_manipulated:]

        fitness_list = [measure_fitness(network) for network in pop]
        BEST_FITNESS_LIST.append(max(fitness_list))
        best_network = pop[np.argmax(fitness_list)]
        return best_network


class Weight:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.activation = activation

    def feedforward(self, X):
        """
        The feedforward method is a fundamental function used in neural networks to propagate input data through the
        network's layers and produce an output or prediction. It takes an input signal and passes it through the network,
        applying a series of calculations using the network's weights, biases, and activation functions.
        The function performs forward propagation, where the input signal is processed layer by layer, and the output is
        generated by the final layer of the network.
        """
        z = np.dot(X, self.weights)
        a = self.activation(z)
        return a

    def swap(self, indices):
        """
        Function performs the weight swap to introduce mutation
        """
        temp = self.weights[indices[0]]
        self.weights[indices[0]] = self.weights[indices[1]]
        self.weights[indices[1]] = temp

    def update_weights(self, weights):
        self.weights = weights


class NN:
    """
    The NN class is a programming construct used to represent individuals in the population of a genetic algorithm. 
    In genetic algorithms, a population consists of multiple individuals, each of which represents a potential solution 
    to a problem. In the context of neural networks, an individual represents a specific configuration or set of weights 
    for a neural network.
    """

    def __init__(self):
        self.W1 = Weight(INPUT_SIZE, HL1, activation=lambda x: self.relu(x))
        self.W2 = Weight(HL1, HL2, activation=lambda x: self.relu(x))
        self.W3 = Weight(HL2, OUTPUT_SIZE, activation=lambda x: self.sigmoid(x))
        self.weights = [self.W1, self.W2, self.W3]  # List to hold all weights of the neural network

    def predict(self, inputs):
        """
        The predict function is used for making predictions or classifications.
        It takes in a sample from a dataset as input and produces a predicted label or output based on the learned
        patterns and relationships within the neural network.
        """
        # Passes the inputs through each layer of the network
        outputs = inputs
        for weight in self.weights:
            outputs = weight.feedforward(outputs)
        # Converts the output of the final weight to binary predictions
        binary_predictions = (outputs > THRESHOLD).astype(int)
        return binary_predictions.flatten()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)


if __name__ == "__main__":
    load_data('nn1_train.txt', 'nn1_test.txt')
    ga = GA()
    best_net = ga.run_ga()

    ################ for plots ######################
    # all = []
    # for i in range(10):
    #     ga = GA()
    #     best_net = ga.run_ga(X_train, y_train)
    #     all.append(BEST_FITNESS_LIST.copy())
    #     BEST_FITNESS_LIST = []
    # min_length = min(len(array) for array in all)
    # trimmed_arrays = [array[:min_length] for array in all]
    # avg = np.mean(trimmed_arrays, axis=0)
    # plt.plot(avg)
    # plt.show()

    np.savez("wnet1", arr1=best_net.W1.weights, arr2=best_net.W2.weights, arr3=best_net.W3.weights)
    model_accuracy = calculate_accuracy(y_test, best_net.predict(X_test))
    print(f"Test Accuracy: {model_accuracy}")
    time.sleep(5)
    print("END!")
