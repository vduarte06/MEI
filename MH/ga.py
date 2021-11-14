from random import randint, sample, shuffle, random
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def int_to_bin(i, num_of_vars):
    return "{:0{}b}".format(i, num_of_vars)


def flip_bit_at_index(s, index=0):
    flip = "0" if s[index] == "1" else "1"
    return "%s%s%s" % (s[:index], flip, s[index + 1 :])


def onemax_fitness(c):
    return c.count("1")


class Individual:
    chromosome = ""
    fitness_function = None

    def __init__(self, c, f):
        self.chromosome = c
        self.fitness_function = f

    def fitness(self):
        return self.fitness_function(self.chromosome)

    def __repr__(self):
        return "{}...: {}".format(self.chromosome[:10], self.fitness())

    def __lt__(self, other):
        return self.fitness() < other.fitness()


class Population:
    P = []  # parents
    S = []  # promissing solutions
    O = []  # offspring
    n_of_individuals = 0
    lenght_of_each_individual = 0
    crossover_probability = 0
    mutation_probability = 0

    # INITIALIZATION
    def __init__(self, N, n, Pc=0.6, Pm=1/8, fitness_function=onemax_fitness):
        """
        Genarates a population of N elements with lenght n,
        E.g. N=10 binary strings of length n=8
        """
        self.n_of_individuals = N
        self.lenght_of_each_individual = n
        self.mutation_probability = Pm
        self.crossover_probability = Pc
        self.possibile_solutions = pow(2, n) - 1

        # Initialize onemax population
        # TODO make it general porpuse
        for i in range(N):
            individual = Individual(
                int_to_bin(randint(0, self.possibile_solutions), n), fitness_function
            )
            self.P.append(individual)

    # SELECTION
    def tournament_selection(self):
        #DUVIDA: selecionamentos sempre um N igual ao da populacao?
        self.S = []
        p = deepcopy(self.P)
        for i in range(self.n_of_individuals):
            pair = sorted(sample(p, 2))
            if pair not in self.S:
                self.S.append(
                    pair[0] if pair[0].fitness() > pair[1].fitness() else pair[1]
                )
        return self.S

    # VARIATION
    def one_point_crossover(self, c1, c2):
        crossing_site = randint(0, self.lenght_of_each_individual - 1)
        result1 = c1[:crossing_site] + c2[crossing_site:]
        result2 = c2[:crossing_site] + c1[crossing_site:]
        return result1, result2

    def crossover(self):
        p_len = len(self.S)
        indices = range(p_len)

        for iteration in range(int((p_len * self.crossover_probability))):
            i, j = sample(indices, 2)
            self.S[i].chromosome, self.S[j].chromosome = self.one_point_crossover(
                self.S[i].chromosome, self.S[j].chromosome
            )

    def mutate(self, individual):
        """each bit is independently flipped with probability Pm"""
        for i in range(len(individual.chromosome)):
            if random() < self.mutation_probability:
                individual.chromosome = flip_bit_at_index(individual.chromosome, i)

        return individual

    def random_mutations(self, Pm=None):
        """
        Iterate randomly through all individuals
        and trigger mutation based on Pm
        """
        if not Pm:
            Pm = self.mutation_probability
        indexes = list(range(len(self.P)))
        shuffle(indexes)
        for i in indexes:
            i = randint(0, self.n_of_individuals - 1)
            self.mutate(self.S[i])

    def apply_variation(self):
        self.crossover()
        self.random_mutations()

    # REPLACEMENT
    def full_replacement(self):
        self.P = self.S
    
    def steady_replacement(self):
        for i , _i in enumerate(self.S):
            for j, _j in enumerate(self.P):
                if _i.fitness() >= _j.fitness():
                    self.P[j] = self.S[i]
                    break
        #print('p', self.P)

def tests():

    population = Population(10, 8)
    assert population.n_of_individuals == 10
    assert population.lenght_of_each_individual == 8

    population.tournament_selection()

    population.crossover()
    population.random_mutations()

    individual = Individual("11111111", onemax_fitness)
    assert individual.fitness() == 8
    #individual = population.mutate(individual)
    #assert individual.fitness() == 7


#tests()

def plot_performance(performance):
    
    # Data for plotting
    t = np.arange(0, len(performance))

    fig, ax = plt.subplots()
    ax.plot(t, performance)

    ax.set(xlabel='Number of generations', ylabel='Fitness improvement',
        title='Genetic Algorythms performance')
    ax.grid()

    plt.show()

if __name__ == "__main__":
    N = 100
    n = 30
    Pc = 0.6
    Pm = 1/10
    population = Population(N, n, Pc, Pm)
    #print(sorted(population.P, reverse=True))
    performance = []

    for i in range(5000):

        population.tournament_selection()
        population.crossover()
        #population.random_mutations()
        population.steady_replacement()
        
        sorted(population.P, reverse=True)
        max_fitness = population.P[0].fitness()
        performance.append(max_fitness)
        if max_fitness == n:
            break

        if i%300 == 0:
            print(max_fitness)

    #print(sorted(population.P, reverse=True))
    plot_performance(performance)
