from genetic_algorithms.models import *
from genetic_algorithms.tests import TestGAForOnemax
import unittest




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
    #unittest.main()
    NUMBER_OF_GENERATIONS = 3000
    N = 100
    n = 100
    Pc = 0.6
    Pm = 1/10
    population = Population(N, n, Pc, Pm, onemax_fitness)

    #print(sorted(population.P, reverse=True))
    performance = []
    for i in range(NUMBER_OF_GENERATIONS):
        population.roulette_wheel_selection()
        population.crossover() #crossover tem menos impacto?
        population.random_mutations()
        population.steady_replacement(generation_gap=0.2)
        #sorted(population.P, reverse=True)
        #print(population.P)
        max_fitness = population.P[0].fitness()
        performance.append(max_fitness)
        if max_fitness == n:
            break

        if i%300 == 0:
            print(max_fitness)
    #print(sorted(population.P, reverse=True))
    plot_performance(performance)
