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
    individual_lenght = 0
    crossover_probability = 0
    mutation_probability = 0

    # INITIALIZATION
    def __init__(self, N, n, Pc=0.6, Pm=1/8, fitness_function=onemax_fitness):
        """
        Genarates a population of N elements with lenght n,
        E.g. N=10 binary strings of length n=8
        """
        self.n_of_individuals = N
        self.individual_lenght = n
        self.mutation_probability = Pm
        self.crossover_probability = Pc
        self.possibile_solutions = pow(2, n) - 1
        self.P = []
        # Initialize population
        for i in range(N):
            individual = Individual(
                int_to_bin(randint(0, self.possibile_solutions), n), fitness_function
            )
            self.P.append(individual)

    # SELECTION
    def generate_roulette_wheel(self):
        p = deepcopy(self.P)            
        fitnesses = np.array([i.fitness() for i in p])
        cumulative_fitness = np.cumsum([i.fitness() for i in p])/np.sum(fitnesses)
        return  np.concatenate(([0], cumulative_fitness))

    def spin_roulette_wheel(self, roulette=None, wheel_marks=None):
        selected_ids = []
        if  roulette is None:
            roulette = self.generate_roulette_wheel()
        if not wheel_marks:
            wheel_marks = [random()]
        for mark in wheel_marks:
            for i in range(len(roulette)-1):
                if (roulette[i] <= mark <= roulette[i+1]):
                    selected_ids.append(i)
                    break

        return selected_ids
    
    def roulette_wheel_selection(self, type="classic"):
        self.S = []
        P = deepcopy(self.P)
        roulette = self.generate_roulette_wheel()
        # CLASSICAL RW: spins N times to select new population
        if type == 'classic':
            for i in range(self.n_of_individuals):
                selected_ids = self.spin_roulette_wheel(roulette)
                self.S.append(P[selected_ids[0]])
        elif type=='sus':
            # SUS: generate markers
            pointers = [i/self.n_of_individuals for i in range(self.n_of_individuals)]
            selected_ids = self.spin_roulette_wheel(roulette, pointers)
            for i in selected_ids:
                self.S.append(P[selected_ids[i]])
        return self.S

    def tournament_selection_with_replacement(self):
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
    def one_point_crossover(self, c1, c2,  crossing_site=None):
        while (crossing_site is None or crossing_site<=0 or crossing_site>=self.individual_lenght):
            crossing_site = randint(1, self.individual_lenght - 2)
        result1 = c1[:crossing_site] + c2[crossing_site:]
        result2 = c2[:crossing_site] + c1[crossing_site:]
        return result1, result2

    def crossover(self):
        p_len = len(self.P)
        indices = list(range(p_len))

        for _ in range(self.n_of_individuals):
            if random() < self.crossover_probability:
                i, j = sample(indices, 2)
                self.S[i].chromosome, self.S[j].chromosome = self.one_point_crossover(
                    self.S[i].chromosome, self.S[j].chromosome
                )

    def mutate(self, individual):
        '''each bit is independently flipped with probability Pm'''
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

    # REPLACEMENT
    def full_replacement(self):
        self.P = deepcopy(self.S)
    
    def steady_replacement2(self):
        for i , _i in enumerate(self.S):
            for j, _j in enumerate(self.P):
                if _i.fitness() >= _j.fitness():
                    self.P[j] = self.S[i]
                    break

    def steady_replacement(self, generation_gap=0.5):
        '''Some of the old solutions become part of the new population.
        whe are replacing the worst of the new solution with the best of the old solution
        the generation gap can be adjusted (percentange of replaced individuals) 
        '''
        generation_gap = int(generation_gap*self.n_of_individuals)
        best_p = sorted(self.P, reverse=True)[:generation_gap]
        O = best_p + sorted(self.S)[generation_gap:]
        self.P = deepcopy(O)

        
        #print('p', self.P)

