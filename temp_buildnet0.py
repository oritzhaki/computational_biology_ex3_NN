# Or Itzhaki 209335058 and Tal Ishon 315242297
import sys
import random
import numpy as np
from ypstruct import structure

X_train, X_test, y_train, y_test = None, None, None, None

INPUT_SIZE = 16
HL1 = 32
HL2 = 16
OUTPUT_SIZE = 1
VEC_SIZE = (INPUT_SIZE*HL1) + (HL1*HL2) + (HL2*OUTPUT_SIZE)

LAMARK_LIMIT = 3
STUCK_LIMIT = 50
ELITE_SIZE = 0.1
breakflag = 1

class NN:
    def __init__(self):
        self.W1 = np.random.randn(INPUT_SIZE, HL1) * np.sqrt(1 / INPUT_SIZE)
        self.W2 = np.random.randn(HL1, HL2) * np.sqrt(1 / HL1)
        self.W3 = np.random.randn(HL2, OUTPUT_SIZE) * np.sqrt(1 / HL2)
        self.layers = [self.W1, self.W2, self.W3]


    def update(self, matrices):
        self.W1 = matrices[0]
        self.W2 = matrices[1]
        self.W3 = matrices[2]

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

    def measure_fitness(self):
        global X_train, y_train

        predictions = self.feedforward(X_train)
        predictions = np.round(predictions)
        predictions = np.squeeze(predictions)

        # print(f"the number of 1's in predictions: {np.count_nonzero(predictions)}")
        # print(f"the number of 0's in predictions: {len(predictions) - np.count_nonzero(predictions)}\n")
        correct_predictions = np.sum(predictions == y_train)
        accuracy = (correct_predictions / len(y_train)) * 100
        return accuracy


def reshape(vector, shape):
    array = np.array(vector)
    return array.reshape(shape[0], shape[1])


# def vec_to_matrix(vec):
#     """
#     Split the vector into 3 matrices
#     """
#     vector = vec.copy()
#     matrices = [np.reshape(vector[:INPUT_SIZE * HL1], (INPUT_SIZE, HL1)),
#                 np.reshape(vector[INPUT_SIZE * HL1:INPUT_SIZE * HL1 + HL1 * HL2], (HL1, HL2)),
#                 np.reshape(vector[INPUT_SIZE * HL1 + HL1 * HL2:], (HL2, OUTPUT_SIZE))]
#     return matrices

# def mat_to_vec(matrices):
#     """
#     combine 3 matrices into 1 vector
#     """
#     combined_vector = np.concatenate((matrices[0].flatten(), matrices[1].flatten(), matrices[2].flatten()), axis=0)
#     return combined_vector


# Calculate the fitness score for an individual code configuration
# def measure_fitness(vector):
#     global X_train, y_train
#
#     matrices = vec_to_matrix(vector)  # matrices: [mat1, mat2, mat3]
#     neuralNet.update(matrices)
#
#     predictions = neuralNet.feedforward(X_train)
#     predictions = np.round(predictions)
#     predictions = np.squeeze(predictions)
#
#     # print(f"the number of 1's in predictions: {np.count_nonzero(predictions)}")
#     # print(f"the number of 0's in predictions: {len(predictions) - np.count_nonzero(predictions)}\n")
#     correct_predictions = np.sum(predictions == y_train)
#     accuracy = (correct_predictions / len(y_train)) * 100
#     return accuracy


def crossover(p1, p2):
    """
    how it works example:
    p1=[-0.50368893 -0.91327073 -0.39029557 -0.88672639  0.696925   -0.79873602]
    p2=[ 0.36131945 -0.77794683 -0.80804455  0.67462618  0.73060522 -0.50873338]
    alpha=[0, 0, 0, 0, 1, 1]
    c1=[ 0.36131945 -0.77794683 -0.80804455  0.67462618  0.696925   -0.79873602]
    c2=[-0.50368893 -0.91327073 -0.39029557 -0.88672639  0.73060522 -0.50873338]
    """
    # c1 = p1.deepcopy()
    # c2 = p1.deepcopy()
    # alpha = np.random.uniform(0.0, 1.0, size=len(p1.sequence))
    # # alpha = [random.choice([0, 1]) for _ in range(len(p1.sequence))]
    # for i in range(len(p1.sequence)):
    #     c1.sequence[i] = alpha[i] * p1.sequence[i] + (1 - alpha[i]) * p2.sequence[i]
    #     c2.sequence[i] = (1 - alpha[i]) * p1.sequence[i] + alpha[i] * p2.sequence[i]
    # return c1, c2
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()

    # matrix1 = vec_to_matrix(c1.sequence)
    # matrix2 = vec_to_matrix(c2.sequence)
    matrix1 = c1.sequence.layers
    matrix2 = c2.sequence.layers

    new_matrix1 = []
    new_matrix2 = []

    for mat1, mat2 in zip(matrix1, matrix2):
        alpha = np.random.uniform(0.0, 1.0, size=mat1.shape)
        new_matrix1.append(alpha * mat1 + (1 - alpha) * mat2)
        new_matrix2.append(alpha * mat2 + (1 - alpha) * mat1)

    # c1.sequence = mat_to_vec(new_matrix1)
    # c2.sequence = mat_to_vec(new_matrix2)

    if len(new_matrix1) != 3:
        print("")

    nn = NN()
    nn.update(new_matrix1)
    c1.sequence = nn
    return c1


def mutate(x, mu, sigma):
    """
    how it works example:
    [ 0.19814601 -0.31742422 -0.34353525  0.6856634  -0.22293646  0.16598561]
    [False False False  True  True  True]
    [ 0.19814601  -0.31742422  -0.34353525 -17.22924339  9.98403477   -4.0379242 ]
    """
    # y = x.deepcopy()
    # flag = np.random.rand(*x.sequence.shape) <= mu  # creates array of T and F
    # ind = np.argwhere(flag)  # indexes of T's
    # y.sequence[ind] += sigma * np.random.randn(*ind.shape) * np.random.randn(*ind.shape)  # add the mutation
    # return y
    c1 = x.deepcopy()
    matrcies = c1.sequence
    # matrcies = vec_to_matrix(y.sequence)
    new_NN = []

    for matrix in matrcies.layers:
        flag = np.random.rand(*matrix.shape) < mu
        # Find the indices where mutation should occur
        mutation_indices = np.where(flag)
        num_mutations = len(mutation_indices[0])

        # If less than 2 mutations, continue to next iteration
        if num_mutations < 2:
            new_NN.append(matrix)
            continue

        # Choose two random indices for swapping weights
        random_indices = np.random.choice(num_mutations, size=2, replace=False)
        swap_indices = mutation_indices[0][random_indices]

        # Perform the weight swap to introduce mutation
        temp = matrix[swap_indices[0]]
        matrix[swap_indices[0]] = matrix[swap_indices[1]]
        matrix[swap_indices[1]] = temp

        new_NN.append(matrix)

    if len(new_NN) != 3:
        print("")
    nn = NN()
    nn.update(new_NN)
    c1.sequence = nn
    # y.sequence = mat_to_vec(matrcies)
    return c1


def roulette_wheel_selection(p):
    # get index of individual for selection based on the roulette wheel mechanism. it works like this: c is accumulative
    # sum of the probability each index (representing someone from the population), and r is a number from 0 to the
    # total sum. the larger prob the person had, the more likely he is that the number will fall on him
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    # print("")
    return ind[0][0]

    # ranked_population = sorted(p, key=lambda individual: measure_fitness(individual.sequence))
    # selection_probs = [rank / len(ranked_population) for rank in range(1, len(ranked_population) + 1)]
    # selected_parents = random.choices(range(len(ranked_population)), weights=selection_probs, k=len(p))
    # return selected_parents[0]

def elite_population(costs, pop):
    sorted_indices = np.argsort(costs)[::-1]
    return [pop[i] for i in sorted_indices[:int(len(pop) * ELITE_SIZE)]]

def normal_population(pop):
    costs = np.array([x.fitness for x in pop])
    sorted_indices = np.argsort(costs)
    ranked_population = [pop[i] for i in sorted_indices]
    selection_probs = [rank / len(ranked_population) for rank in range(1, len(ranked_population) + 1)]
    selected_parents = random.choices(ranked_population, weights=selection_probs, k=1)
    return selected_parents[0]
def mutate_again(x, num_mutations):
    y = x.deepcopy()
    indices = np.random.choice(len(x.sequence), num_mutations, replace=False)

    for i in range(num_mutations):
        i1 = indices[i]
        i2 = np.random.choice(np.setdiff1d(np.arange(len(x.sequence)), i1))
        y.sequence[i1], y.sequence[i2] = y.sequence[i2], y.sequence[i1]

    return y
def run_ga(problem, params):
    # Problem Information
    fitness_func = problem.fitness_func
    size = problem.size

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
        # pop[i].sequence = np.random.uniform(-0.5, 0.5, problem.size)  # todo: check how is best to init the models (-0.1-0.1, 1-10,...)
        pop[i].sequence = NN()
        pop[i].fitness = pop[i].sequence.measure_fitness()
        if pop[i].fitness > bestsol.fitness:
            bestsol = pop[i].deepcopy()

    # Best Cost of each iteration
    bestcost = np.empty(maxit)
    avgcost = np.empty(maxit)
    bestseq = []

    should_break = 0  # counter to check if got best sequence already
    lamarckian_count = 0

    # Main Loop
    for it in range(maxit):
        print(f"Generation: {it}")
        #  create probabilities - better solutions are more likely to give offspring
        costs = np.array([x.fitness for x in pop])
        avg_cost = np.mean(costs)
        # sum_costs = np.sum(costs)
        # # if avg_cost != 0:
        # #     costs = costs / avg_cost
        # if sum_costs != 0:
        #     costs = costs / sum_costs
        # if it%10 == 1:
        #     print("")
        # probs = np.exp(5*costs)  # todo: play with hyper param for the exp
        # probs /= np.sum(probs)

        avgcost[it] = avg_cost

        popc = []  # offspring population
        for _ in range(nc // 2):  # creation of offsprings
            #check if stuck on same best sol until lamarckian limit:
            if lamarckian_count == LAMARK_LIMIT:
                lamarckian_count = 0
                print("lamarckian")
                for i, p in enumerate(pop):
                    p_temp = p.deepcopy()
                    p_temp = mutate_again(p_temp, 5)
                    p_temp.fitness = fitness_func(p_temp.sequence)
                    if p_temp.fitness > p.fitness:
                        pop[i] = p_temp

            # choose two parents according to probs
            # p1 = pop[roulette_wheel_selection(probs)]
            # p2 = pop[roulette_wheel_selection(probs)]
            elite = elite_population(costs, pop)
            remain = [struct for struct in reversed(pop) if not any(np.array_equal(struct.sequence, elite_struct.sequence) for elite_struct in elite)]
            # for struct in reversed(population):
            #     if struct in elite:
            #         remain.append(struct)
            p1 = random.choice(elite)
            p2 = normal_population(remain)

            # create two offsprings from the crossover of the two parents
            # c1, c2 = crossover(p1, p2)
            c1 = crossover(p1, p2)

            # mutate the offspring
            c1 = mutate(c1, mu, sigma)
            # c2 = mutate(c2, mu, sigma)

            c1.fitness = c1.sequence.measure_fitness()
            if c1.fitness > bestsol.fitness:
                bestsol = c1.deepcopy()

            # c2.fitness = fitness_func(c2.sequence)
            # if c2.fitness > bestsol.fitness:
            #     bestsol = c2.deepcopy()

            popc.append(c1)
            # popc.append(c2)

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.fitness, reverse=True)  # descending
        pop = pop[:npop]  # take the population of size npop with the best fitness

        # Store Best Cost
        print(f"Best fitness: {bestsol.fitness}")
        bestcost[it] = bestsol.fitness
        bestseq.append(bestsol.sequence)

        # check if bestcost doesn't change in the last x iterations
        if np.array_equal(bestseq[it - 1], bestseq[it]):
            should_break += 1
            lamarckian_count += 1
        else:
            should_break = 0
            lamarckian_count = 0

        # check if best solution hasn't changed in a long time and if so exit
        if (should_break == STUCK_LIMIT):
            break_flag = 0
            break
        if (bestsol.fitness == 100):
            break
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
    # matrices = vec_to_matrix(best_sol)
    # Save the matrices to a file
    # np.savetxt("wnet.txt", (matrices[0], matrices[1], matrices[2]), delimiter=',')
    np.savetxt("wnet.txt", best_sol, delimiter=',')

    with open("backup.txt", 'w') as file:
        # Write the vector elements to the file
        for element in best_sol:
            file.write(str(element) + '\n')


if __name__ == '__main__':
    # todo: make sure if input is one file that we need to split. If so - the following code is suitable here
    X_train, X_test, y_train, y_test = load_data("nn0.txt")

    problem = structure()
    # problem.fitness_func = measure_fitness
    problem.size = VEC_SIZE

    # describe algorithm hyperparameters
    params = structure()
    params.maxit = 200  # number of iterations
    params.npop = 100  # size of population
    params.pc = 2  # ratio of offspring:original population (how many offspring to be created each iteration)
    params.mu = 0.3  # percentage of vector to receive mutation
    params.sigma = 1  # todo: find good sigma (size of mutation)

    best_solution, best_fitness_array, avg_fitness_array = run_ga(problem, params)
    if breakflag:
        write_best_sol_to_file(best_solution)

    # print(f"The total number of calls to fitness function: {fitness_func_counter}")

    # if (len(best_fitness_array) >= 70):
    #     with open('spreadcount.csv', 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(best_fitness_array[:70])
    #     with open('averagecount.csv', 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(avg_fitness_array[:70])
