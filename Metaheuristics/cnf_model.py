import csv
from decouple import config

DIR_NAME = config('DIR_NAME')

class CNF:
    num_of_clauses = 0
    num_of_vars = 0
    num_of_possible_solutions = 0
    clauses = []
    num_of_evaluations = 0
    max_num_of_evaluations = pow(10,3)

    def __init__(self, input_file_number):
        self.read_cnf_file(
            "{}/uf100-0{}.cnf".format(DIR_NAME, input_file_number)
        )
        self.num_of_possible_solutions = pow(2, self.num_of_vars) - 1

    def read_cnf_file(self, fname):
        with open(fname) as f:
            csv_reader = csv.reader(f, delimiter=" ")
            header = True
            cnf = []

            for row in csv_reader:
                if row[0] == "p":
                    header = False
                    header_data = list(filter(None, row))
                    num_of_vars = int(header_data[2])
                    num_of_clauses = int(header_data[3])
                    continue
                elif row[0] == "%":
                    break
                elif not header:
                    clean_row = [int(n) for n in row[:-1] if n]
                    cnf.append(clean_row)

            self.num_of_vars, self.num_of_clauses, self.clauses = num_of_vars, num_of_clauses, cnf

    def valid_clause(self, solution, clause):
        for x in clause:
            # X1 X2 ... Xi
            x, i = int(x > 0), abs(x) - 1
            if int(solution[i]) == int(x):
                return True


    def validate(self, solution):
        self.num_of_evaluations += 1
        ranking = 0
        for clause in self.clauses:
            if self.valid_clause(solution, clause):
                ranking += 1
        return ranking
