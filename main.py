# Or Itzhaki 209335058 and Tal Ishon
import numpy as np
from ypstruct import structure
from collections import defaultdict

#  Mono Alphabetic code is where each letter is swapped by another - the encoding is a permutation

# enc.txt is the text which needs to be encoded. The text can include: " ", ".", ",", ";", "\n" and they are to be left
# exactly the same and are not to be switched.

# In this code by using a genetic algorithm we will discover the rules of the code.
# the rules should be outputted to perm.txt, and the decoded text to plain.txt.

# useful files: dict.txt - dictionary of popular words, LetterFreq.txt - letter and its frequency,
# Letter2Freq - two letters and their frequency together.

# Part A - genetic algorithm do decode a text (create two output files) and print the amount of steps (iterations?).

alphabet = np.array(
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
     'x', 'y', 'z'])
coded_file = "enc.txt"
decoded_text_file = "plain.txt"
best_solution_file = "perm.txt"
encrypted_text = ""

dictionary = None
letter_frequencies = None
letter_pair_frequencies = None


# Load the dictionary file containing valid English words
def load_dictionary(file_path):  # todo: check if need to remove "" from dictionary
    with open(file_path, 'r') as file:
        dictionary = set(word.strip().lower() for word in file)
    return dictionary


# Load the letter frequency file
def load_letter_frequencies(file_path):
    with open(file_path, 'r') as file:
        letter_frequencies = defaultdict(float)
        for line in file:
            values = line.strip().split('\t')
            if len(values) == 2:  # check if not empty
                frequency, letter = values
                letter_frequencies[letter.lower()] = float(frequency)
            else:
                break
    return letter_frequencies


# Load the letter pair frequency file
def load_letter_pair_frequencies(file_path):
    with open(file_path, 'r') as file:
        letter_pair_frequencies = defaultdict(float)
        for line in file:
            values = line.strip().split('\t')
            if len(values) == 2:  # check if not empty
                frequency, letter_pair = values
                letter_pair_frequencies[letter_pair.lower()] = float(frequency)
            else:
                break
    return letter_pair_frequencies


# Load the encrypted text file
def load_encrypted_text():
    with open(coded_file, 'r') as file:
        text = file.read().lower()  # Read the text from the input file and convert to lowercase
    return text


def decode_text(permutation_vector):
    # decoded_text = ''
    # for char in encrypted_text:
    #     if char in alphabet:
    #         index = np.where(alphabet == char)[0][0]  # Find the index of the character in the alphabet
    #         encoded_char = permutation_vector[index]  # Get the corresponding character from the permutation vector
    #         decoded_text += encoded_char
    #     else:
    #         decoded_text += char  # Preserve non-alphabetic characters as is
    # return decoded_text
    temp_dict = dict(zip(alphabet, permutation_vector))
    decrypted_text = encrypted_text.translate(str.maketrans(temp_dict))
    return decrypted_text

def create_output(solution):
    # encode text and save in correct file
    decoded_text = decode_text(solution)

    with open(decoded_text_file, 'w') as file:
        file.write(decoded_text)
    # create file with the rules of the encoding
    with open(best_solution_file, 'w') as file:
        for i in range(len(alphabet)):
            file.write(f"{alphabet[i]} {solution[i]}\n")


# Calculate the fitness score for an individual code configuration
def measure_fitness(solution):
    text = decode_text(solution)
    fitness = 0.0

    # Calculate fitness based on letter frequencies
    for letter in text:
        fitness += letter_frequencies[letter]

    # Calculate fitness based on word occurrences in the dictionary
    # find all words from dictionary in the text and give score accordingly
    words = text.lower().split()
    for word in words:
        if word in dictionary:
            fitness += 1.0

    # Calculate fitness based on letter pair frequencies
    for i in range(len(text) - 1):
        letter_pair = text[i:i + 2]
        fitness += letter_pair_frequencies[letter_pair]

    return fitness


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
    if len(ind) > 0:
        for i1 in ind:
            i2 = np.random.randint(len(x.sequence))
            y.sequence[i1.item()], y.sequence[i2] = y.sequence[i2], y.sequence[i1.item()]
    return y


def roulette_wheel_selection(p):
    # get index of individual for selection based on the roulette wheel mechanism.
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
        pop[i].fitness = fitness_func(pop[i].sequence)
        if pop[i].fitness > bestsol.fitness:
            bestsol = pop[i].deepcopy()

    # Best Cost of each iteration
    bestcost = np.empty(maxit)  # need?
    bestseq = np.empty(maxit)

    should_break = 0  # counter to check if got best sequence already
    # Main Loop
    for it in range(maxit):
        #  create probabilities - better solutions are more likely to give offspring
        costs = np.array([x.fitness for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs / avg_cost
        # probs = np.exp(-beta * costs)
        probs = np.exp(beta * costs)
        probs /= np.sum(probs)

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
        bestcost[it] = bestsol.fitness  # need?
        # bestseq[it] = bestsol.sequence

        # # check if bestcost doesn't change in the last x iteration
        # if bestseq[it - 1] == bestseq[it]:
        #     should_break += 1
        # else:
        #     should_break = 0
        #
        # ### check if best solution hasn't changed in a long time and if so exit
        # if should_break == 20:
        #     break

    return bestsol.sequence


if __name__ == '__main__':
    # global encrypted_text, dictionary, letter_frequencies, letter_pair_frequencies
    encrypted_text = load_encrypted_text()
    dictionary = load_dictionary('dict.txt')
    letter_frequencies = load_letter_frequencies('Letter_Freq.txt')
    letter_pair_frequencies = load_letter_pair_frequencies('Letter2_Freq.txt')
    # define problem: we are looking for the best vector of size 26. The best vector is the permutation of the
    # alphabet that maximizes the fitness score
    problem = structure()
    problem.fitness_func = measure_fitness
    problem.size = 26  # letters of the alphabet

    # describe algorithm hyperparameters
    params = structure()
    params.maxit = 100  # number of iterations
    params.npop = 500  # size of population
    params.beta = 1  # not sure we need
    params.pc = 1  # ratio of offspring:original population (how many offspring to be created each iteration)
    params.mu = 0.1  # percentage of vector to receive mutation

    best_solution = run_ga(problem, params)
    create_output(best_solution)
