from .models import *
import unittest
import numpy as np

class TestGAForOnemax(unittest.TestCase):
    def setUp(self):
        self.N = 4
        self.n = 4
        self.population = Population(self.N, self.n)
         # predefined chromosomes
        self.population.P[0].chromosome = '1111'
        self.population.P[1].chromosome = '1110'
        self.population.P[2].chromosome = '1100'
        self.population.P[3].chromosome = '1000'
    
    def test_init(self):
        self.assertEqual(self.population.n_of_individuals, self.N)
        self.assertEqual(self.population.individual_lenght, self.n)

    def test_roulette_creation(self):
        p = self.population
        
        spin_result = p.spin_roulette_wheel(wheel_marks=[4/10])
        self.assertEqual(spin_result[0], 0) # individual 0 is between 0 - 4/10
        spin_result = p.spin_roulette_wheel(wheel_marks=[0.5])
        self.assertEqual(spin_result[0], 1) # individual 1 is between 5/10 - 1
        spin_result = p.spin_roulette_wheel(wheel_marks=[1])
        self.assertEqual(spin_result[0], 3) # individual 3 is between 0.9 - 1

    def test_roulette_selection(self):
        p = Population(self.N, self.n)
        p.roulette_wheel_selection('sus')


    def test_steady_replacement(self):
        self.population.S = deepcopy(self.population.P)
        self.population.S[1].chromosome = '1111'
        self.population.steady_replacement(generation_gap=0.5)
        self.assertEqual(list(map(lambda i: i.chromosome, self.population.P)), ['1111', '1110', '1111', '1111'])
        #self.assertEqual(self.population.S, self.population.P)

    def test_crossover(self):
        c1 = self.population.P[0].chromosome
        c2 = self.population.P[3].chromosome
        i1a, i1b = self.population.one_point_crossover(c1, c2, 1)
        i2a, i2b = self.population.one_point_crossover(c1, c2, 2)
        i3a, i3b = self.population.one_point_crossover(c1, c2, 3)
        self.assertEqual([i1a, i1b], ['1000', '1111'])
        self.assertEqual([i2a, i2b], ['1100', '1011'])
        self.assertEqual([i3a, i3b], ['1110', '1001'])

