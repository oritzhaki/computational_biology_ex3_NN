# Or Itzhaki 209335058 and Tal Ishon
import numpy as np
from ypstruct import structure

#  Mono Alphabetic code is where each letter is swapped by another - the encoding is a permutation

# enc.txt is the text which needs to be encoded. The text can include: " ", ".", ",", ";", "\n" and they are to be left
# exactly the same and are not to be switched.

# In this code by using a genetic algorithm we will discover the rules of the code.
# the rules should be outputted to perm.txt, and the decoded text to plain.txt.

# useful files: dict.txt - dictionary of popular words, LetterFreq.txt - letter and its frequency,
# Letter2Freq - two letters and their frequency together.

# Part A - genetic algorithm do decode a text (create two output files) and print the amount of steps (iterations?).

alphabet = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
coded_file = "enc.txt"
decoded_text_file = "plain.txt"
best_solution_file = "perm.txt"


def decode_text(permutation_vector):
    with open(coded_file, 'r') as file:
        text = file.read().lower()  # Read the text from the input file and convert to lowercase
    decoded_text = ''
    for char in text:
        if char in alphabet:
            index = np.where(alphabet == char)[0][0]  # Find the index of the character in the alphabet
            encoded_char = permutation_vector[index]  # Get the corresponding character from the permutation vector
            decoded_text += encoded_char
        else:
            decoded_text += char  # Preserve non-alphabetic characters as is
    return decoded_text


def create_output(solution):
    # encode text and save in correct file
    text = decoded_text_file
    with open(decoded_text_file, 'w') as file:
        file.write(text)
    # create file with the rules of the encoding
    with open(best_solution_file, 'w') as file:
        for i in range(len(alphabet)):
            file.write(f"{alphabet[i]} {solution[i]}\n")


def measure_fitness(solution):
    # idea: encode the text with the current permutation, the more words found in the dict, the higher the fitness.
    text = decode_text(solution)
    # find all words from dictionary in the text and give score accordingly
    # calculate frequencies of letters
    # calculate frequencies of pairs of letters
    # measure loss between known frequencies and apply to score
    return score


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


def mutate(x, mu):
    # mutation = swap between two random letters
    y = x.deepcopy()
    flag = np.random.rand(*x.sequence.shape) <= mu
    ind = np.argwhere(flag)
    for i1 in ind:
        i2 = np.random.randint(len(x.sequence))
        y.position[i1], y.position[i2] = y.position[i2], y.position[i1]
    return y


def roulette_wheel_selection(p):
    # get index of individual for selection based on the roulette wheel mechanism.
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


def run_ga(problem, params):
    # Problem Information
    costfunc = problem.costfunc

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
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
        pop[i].sequence = np.random.permutation(alphabet.copy())
        pop[i].fitness = costfunc(pop[i].sequence)
        if pop[i].fitness > bestsol.fitness:
            bestsol = pop[i].deepcopy()

    # Best Cost of each iteration
    bestcost = np.empty(maxit)  # need?

    # Main Loop
    for it in range(maxit):
        #  create probabilities - better solutions are more likely to give offspring
        costs = np.array([x.fitness for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs / avg_cost
        probs = np.exp(-beta * costs)

        # print(probs)

        #  print iteration:
        print(f"Generation: {it}")

        popc = []  # offspring population
        for _ in range(nc // 2):  # creation of offsprings
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            c1, c2 = crossover(p1, p2)

            c1 = mutate(c1, mu)
            c2 = mutate(c2, mu)

            c1.fitness = costfunc(c1.sequence)
            if c1.fitness > bestsol.fitness:
                bestsol = c1.deepcopy()

            c2.fitness = costfunc(c2.sequence)
            if c2.fitness > bestsol.fitness:
                bestsol = c2.deepcopy()

            popc.append(c1)
            popc.append(c2)

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.fitness)
        pop = pop[0:npop]  # take the population of size npop with the best fitness
        # Store Best Cost
        bestcost[it] = bestsol.fitness  # need?

        ### check if best solution hasn't changed in a long time and if so exit?

    return bestsol.sequence


if __name__ == '__main__':
    # define problem: we are looking for the best vector of size 26. The best vector is the permutation of the
    # alphabet that maximizes the fitness score
    problem = structure()
    problem.fitnessfunc = measure_fitness
    problem.size = 26  # letters of the alphabet

    # describe algorithm hyperparameters
    params = structure()
    params.maxit = 500  # number of iterations
    params.npop = 100  # size of population
    params.beta = 1 #not sure we need
    params.pc = 1  # ratio of offspring:original population (how many offspring to be created each iteration)
    params.mu = 0.1  # percentage of vector to receive mutation

    best_solution = run_ga(problem, params)
    create_output(best_solution)
