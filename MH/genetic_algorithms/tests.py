from .models import *
import unittest
import numpy as np

class TestGAForOnemax(unittest.TestCase):
    def setUp(self):
        self.N = 2
        self.n = 2
        self.population = Population(self.N, self.n)
    
    def test_init(self):
        self.assertEqual(self.population.n_of_individuals, self.N)
        self.assertEqual(self.population.lenght_of_each_individual, self.n)

    def test_roulette_selection(self):
        self.N = 4
        self.n = 4
        self.population = Population(self.N, self.n)
        p = self.population
        self.assertEqual(p.n_of_individuals, self.N)
        self.assertEqual(p.lenght_of_each_individual, self.n)

        # predefined chromosomes
        p.P[0].chromosome = '1111'
        p.P[1].chromosome = '1110'
        p.P[2].chromosome = '1100'
        p.P[3].chromosome = '1000'
        

        spin_result = p.spin_roulette_wheel(wheel_marks=[4/10])
        self.assertEqual(spin_result[0], 0) # individual 0 is between 0 - 4/10
        spin_result = p.spin_roulette_wheel(wheel_marks=[0.5])
        self.assertEqual(spin_result[0], 1) # individual 1 is between 5/10 - 1
        spin_result = p.spin_roulette_wheel(wheel_marks=[1])
        self.assertEqual(spin_result[0], 3) # individual 3 is between 0.9 - 1
        
        p.roulette_wheel_selection()

