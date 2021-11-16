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
    NUMBER_OF_GENERATIONS = 2000
    N = 100
    n = 50
    Pc = 0.9
    Pm = 1/10
    population = Population(N, n, Pc, Pm, onemax_fitness)

    #print(sorted(population.P, reverse=True))
    performance = []
    for i in range(NUMBER_OF_GENERATIONS):
        
        population.roulette_wheel_selection('sus')
        population.crossover() #crossover tem menos impacto?
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
