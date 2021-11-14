from cnf_model import CNF
import sys, getopt
from random import randint, shuffle
import matplotlib.pyplot as plt


def get_args():
    def help():
        print("maxsat.py -i <inputfilenumber>")
        sys.exit(-1)

    argv = sys.argv[1:]
    args = {}
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:a", ["ifile=", "algorythm="])
    except getopt.GetoptError:
        print("WRONG ARGUMENTS!:")
        print("maxsat.py -i <inputfilenumber>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i",):
            args = {"inputfilenumber": arg}
        elif opt in ("-a",):
            print(arg)
        else:
            help()
    else:
        help()

    return args


def int_to_bin(i, num_of_vars):
    return "{:0{}b}".format(i, num_of_vars)


def flip_bit_at_index(s, index=0):
    flip = "0" if s[index] == "1" else "1"
    return "%s%s%s" % (s[:index], flip, s[index + 1 :])


def systematic_search(cnf):
    solutions = []
    i = 0
    candidate_solution = "0"
    end_of_search_space = int_to_bin(cnf.num_of_possible_solutions, cnf.num_of_vars)
    while candidate_solution != end_of_search_space:
        candidate_solution = int_to_bin(i, cnf.num_of_vars)
        i += 1
        if cnf.validate(candidate_solution) >= cnf.num_of_clauses:
            solutions.append(candidate_solution)
    return solutions


def step(cnf, candidate_solution, stop_visiting=False):
    best_solution_rank = cnf.validate(candidate_solution)
    best_solution = None

    #TODO make i random
    search_space = list(range(cnf.num_of_vars))
    shuffle(search_space)
    for i in search_space:
        neighbour = flip_bit_at_index(candidate_solution, i)
        neighbour_rank = cnf.validate(neighbour)
        if neighbour_rank > best_solution_rank:
            best_solution_rank = neighbour_rank
            best_solution = neighbour
            print(best_solution_rank)
            if stop_visiting:
                break

    return best_solution_rank, best_solution


def variable_step(cnf, candidate_solution, stop_visiting=False):
    best_solution_rank = cnf.validate(candidate_solution)
    best_solution = None

    #TODO make i random
    search_space = list(range(cnf.num_of_vars))
    shuffle(search_space)
    for i in search_space:
        neighbour = flip_bit_at_index(candidate_solution, i)
        neighbour_rank = cnf.validate(neighbour)
        if neighbour_rank > best_solution_rank:
            best_solution_rank = neighbour_rank
            best_solution = neighbour
            print(best_solution_rank)
            if stop_visiting:
                break

    return best_solution_rank, best_solution


def steepest_asc(cnf):
    result = 0
    candidate_solution = int_to_bin(
        randint(0, cnf.num_of_possible_solutions), cnf.num_of_vars
    )
    while result < cnf.num_of_clauses or (cnf.num_of_evaluations <= cnf.max_num_of_evaluations):
        result, new_candidate_solution = step(cnf, candidate_solution)
        if new_candidate_solution:
            candidate_solution = new_candidate_solution
        else:
            break
    return candidate_solution

def next_asc(cnf):
    result = 0
    candidate_solution = int_to_bin(
        randint(0, cnf.num_of_possible_solutions), cnf.num_of_vars
    )
    while result < cnf.num_of_clauses or (cnf.num_of_evaluations <= cnf.max_num_of_evaluations):
        result, new_candidate_solution = step(cnf, candidate_solution, True)
        if new_candidate_solution:
            candidate_solution = new_candidate_solution
        else:
            break
    return candidate_solution

def variable_ngb_desc(cnf):
    result = 0
    candidate_solution = int_to_bin(
        randint(0, cnf.num_of_possible_solutions), cnf.num_of_vars
    )
    while result < cnf.num_of_clauses or (cnf.num_of_evaluations <= cnf.max_num_of_evaluations):
        result, new_candidate_solution = step(cnf, candidate_solution, True)
        if new_candidate_solution:
            candidate_solution = new_candidate_solution
        
    return candidate_solution

def steepest_asc_with_restart(cnf):
    result = 0
    candidate_solution = int_to_bin(
        randint(0, cnf.num_of_possible_solutions), cnf.num_of_vars
    )
    rankings = []
    while result < cnf.num_of_clauses or (cnf.num_of_evaluations <= cnf.max_num_of_evaluations):
        result, candidate_solution = step(cnf, candidate_solution)
        if not candidate_solution:
            candidate_solution = int_to_bin(
                randint(0, cnf.num_of_possible_solutions), cnf.num_of_vars
            )
        rankings.append(result)
    return rankings


if __name__ == "__main__":
    args = get_args()
    cnf_instance = CNF(args["inputfilenumber"])
    
    #solutions = systematic_search(cnf_instance)
    # s = steepest_asc(cnf_instance)
    # s = steepest_asc_with_restart(cnf_instance)
    s = next_asc(cnf_instance)
    #assert(h in solutions)

    s = steepest_asc_with_restart(cnf_instance)
    fig, ax = plt.subplots()
    ax.plot(t, s)
    plt.show()