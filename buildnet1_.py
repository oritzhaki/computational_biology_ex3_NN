import numpy as np
import copy
import random

####################################################
##################### GLOBALS ######################
####################################################

X_train, X_test, y_train, y_test = None, None, None, None

POP_SIZE = 200
MUTATION_RATE = 0.3
GENERATIONS_LIMIT = 100
ELITE_PERCENT = 0.05
PERCENT_NOT_MANIPULATED = 0.1
STUCK_THRESHOLD = 10
LAMARCKIAN_MUTATIONS = 3
THRESHOLD = 0.6

INPUT_SIZE = 16
OUTPUT_SIZE = 1

TRAIN_PERCENT =  0.8
best_fitness_list = []

######################################################


def load_data(path):
    f = open(path, "r")
    lines = f.readlines()

    X, y = [], []

    for line in lines:
        values = line.rstrip('\n').split("  ")
        X.append(values[0])
        y.append(values[1])

    size = len(lines)
    size_train = int(size * TRAIN_PERCENT)
    X = np.array([list(map(int, string)) for string in X])
    y = np.array(y).astype(int)
    return X[:size_train], X[size_train:], y[:size_train], y[size_train:]


def initiate_dataset():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data("nn1.txt")


def print_model_accuracy():
    accuracy = calculate_net_accuracy(y_test, best_net.predict(X_test))
    print(f"Test Accuracy: {accuracy}")



def calculate_net_accuracy(y_train, predictions):
    correct_predictions = np.sum(predictions == y_train)
    accuracy = correct_predictions / len(y_train)
    return float(accuracy)


def measure_fitness(network, X_train, y_train):
    predictions = network.predict(X_train)
    return calculate_net_accuracy(y_train, predictions)


def sign(z):
    return np.where(z >= 0, 1, 0)

class GA:
    def __init__(self):
        self.POP_SIZE = POP_SIZE

    def rank_selection(self, pop):
        ranked_pop = sorted(pop, key=lambda network: measure_fitness(network, X_train, y_train))
        probs = [rank / len(ranked_pop) for rank in range(1, len(ranked_pop) + 1)]
        parents = random.choices(ranked_pop, weights=probs, k=len(pop))
        return parents

    def run_ga(self, X_train, y_train):
        # Creating an initial pop of neural networks
        pop = []
        global best_fitness_list
        for _ in range(self.POP_SIZE):
            network = NN(Weight(INPUT_SIZE, OUTPUT_SIZE, activation=lambda x: sign(x)))
            pop.append(network)

        best_fitness = 0
        stuck = 0
        for generation in range(GENERATIONS_LIMIT):
            print(f"Generation {generation}:")

            # Evaluating the fitness of each network in the current pop
            fitnesses_list = []
            for network in pop:
                fitness = measure_fitness(network, X_train, y_train)
                fitnesses_list.append(fitness)

            current_best_fit = max(fitnesses_list)
            print(f"Current best fitness: {current_best_fit}")
            best_fitness_list.append(current_best_fit)

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
                print(f"Reachecd converge - FITNESS: {best_fitness}")
                break

            # Selecting the top performing networks (elites)
            sorted_indices = np.argsort(fitnesses_list)[::-1]
            elite_pop = [pop[i] for i in sorted_indices[:int(self.POP_SIZE * ELITE_PERCENT)]]
            # Remaining pop after elites have been selected
            remaining_pop = list(set(pop) - set(elite_pop))

            # Creating offspring pop via crossover
            offspring_pop = []
            offsprings_len = self.POP_SIZE - len(elite_pop)

            # Rank Selection
            selected_parents = self.rank_selection(remaining_pop)

            for _ in range(offsprings_len):
                p1 = np.random.choice(selected_parents)
                p2 = np.random.choice(elite_pop)
                offspring = p1.crossover(p2)
                offspring_pop.append(offspring)

            # Save some offspring untouched for the next gen pop
            num_untouched_offspring = int(offsprings_len * PERCENT_NOT_MANIPULATED)
            untouched_offspring = offspring_pop[:num_untouched_offspring]

            # Mutate the remaining (touched) offspring pop
            for offspring in offspring_pop[num_untouched_offspring:]:
                offspring.mutate()

            # Combine elites, untouched offspring and mutated offspring to create the next gen pop
            pop = elite_pop + untouched_offspring + offspring_pop[num_untouched_offspring:]


            if stuck > 3:
                new_pop = []
                # performe Lamarckian evolution on each network in the current population
                for network in pop:
                    new_pop.append(self.lamarckian(network, X_train, y_train))
                pop = new_pop


        fitnesses_list = [measure_fitness(network, X_train, y_train) for network in pop]
        best_fitness_list.append(max(fitnesses_list))
        best_network = pop[np.argmax(fitnesses_list)]
        return best_network


    def lamarckian(self, network, X_train, y_train):
        old_fitness = measure_fitness(network, X_train, y_train)
        temp_net = copy.deepcopy(network)
        for _ in range(LAMARCKIAN_MUTATIONS):
            temp_net.mutate()

        new_fitness = measure_fitness(temp_net, X_train, y_train)
        if new_fitness > old_fitness:
            return temp_net
        else:
            return network


class Weight:
    def __init__(self, input_size, output_size, activation=lambda x: sign(x)):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.activation = activation

    def feedforward(self, inputs):
        z = np.dot(inputs, self.weights)
        a = self.activation(z)
        return a

    def swap(self, indices):
        # Perform the weight swap to introduce mutation
        temp = self.weights[indices[0]]
        self.weights[indices[0]] = self.weights[indices[1]]
        self.weights[indices[1]] = temp

class NN:

    def __init__(self, weight1):
        # List to hold all weights of the neural network
        self.W1 = weight1
        self.weights = [self.W1]

    def predict(self, inputs):
        # Passes the inputs through each layer of the network
        outputs = inputs
        for weight in self.weights:
            outputs = weight.feedforward(outputs)

        return outputs.flatten()


    def crossover(self, other_network):
        # Create the new neural network which will be our offspring
        network = NN(Weight(INPUT_SIZE, OUTPUT_SIZE, activation=lambda x: sign(x)))

        for i in range(len(self.weights)):
            alpha = np.random.uniform(0.0, 1.0, size=self.weights[i].weights.shape)
            network.weights[i].weights = alpha * self.weights[i].weights + \
                                            (1 - alpha) * other_network.weights[i].weights
        return network

    def mutate(self):
        for weight in self.weights:
            flag = np.random.rand(*weight.weights.shape) < MUTATION_RATE
            mutation_indices = np.where(flag)
            mutations_num = len(mutation_indices[0])

            if mutations_num < 2:  # less than 2 - no actual mutation will accure
                continue

            # Choose two random indices for swapping weights
            rand_indices = np.random.choice(mutations_num, size=2, replace=False)
            swap_indices = mutation_indices[0][rand_indices]

            weight.swap(swap_indices)


if __name__ == "__main__":
    initiate_dataset()

    ga = GA()
    best_net = ga.run_ga(X_train, y_train)
    np.savez("wnet1", arr1=best_net.W1.weights)

    print_model_accuracy()
