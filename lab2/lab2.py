from aima3.agents import *
from misio.aima import *
from utils import *
import numpy as np

optilio_mode = False
dir = 'test_cases'
filename = '2015.in'

def find_front(x, y, test_case):
    pass

def preprocessing(test_case):
    # clear all fields with B or O
    mask = test_case.data[1:-1, 1:-1] > 0
    test_case.probabilities[mask] = 0
    test_case.visited[mask] = True

    mask = np.full((test_case.size[0]+2, test_case.size[1]+2), False)

    for row in range(1, test_case.data_matrix_size[0] - 1):
        for column in range(1, test_case.data_matrix_size[1] - 1):
            # clear neighbours of O
            if test_case.data[row, column] == state_to_num_mapping['O']:
                mask[row - 1, column] = True
                mask[row, column + 1] = True
                mask[row, column - 1] = True
                mask[row + 1, column] = True


            # too distant `?`
            elif test_case.data[row, column] == state_to_num_mapping['?']:
                sum_around_unknown = test_case.data[row - 1, column] + test_case.data[row, column + 1] + \
                    test_case.data[row, column - 1] + test_case.data[row + 1, column]

                # around `?` there are only `?`
                if sum_around_unknown == 0:
                    test_case.visited[row - 1, column - 1] = True

                # around `?` there are at least 3 `B`
                elif sum_around_unknown == 3 or sum_around_unknown == 4:
                    test_case.visited[row - 1, column - 1] = True
                    test_case.probabilities[row - 1, column - 1] = 1.0

    # visit all neighbors of `O`
    mask = mask[1:-1, 1:-1]
    test_case.probabilities[mask] = 0
    test_case.visited[mask] = True

def calculate_probabilities(test_case):
    preprocessing(test_case)

    # calculate probabilities
    for row in range(test_case.visited.shape[0]):
        for column in range(test_case.visited.shape[1]):
            if not test_case.visited[row, column]:


def main():
    tests = []
    if optilio_mode:
        tests = parse_test_data_from_input(optilio_mode)
    else:
        import os
        path = '{}/{}/{}'.format(os.getcwd(), dir, filename)
        tests = parse_test_data(path, optilio_mode, extended_data_matrix = True)

    test_case = tests[1]
    calculate_probabilities(test_case)
    print(test_case.visited)
    print(test_case.probabilities)

if __name__ == "__main__":
    main()