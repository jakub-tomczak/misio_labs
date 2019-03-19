from aima3.agents import *
from misio.aima import *
import numpy as np

optilio_mode = True
dir = 'test_cases'

np.set_printoptions(precision=2, floatmode='fixed')

import numpy as np

# O has value of 5 to distinguish form a sum of 4 B
state_to_num_mapping = {
    '?': 0,
    'B': 1,
    'O': 5
}

class test_case(object):
    # test_case.size is a real size of the input data
    # test_case.data_matrix_size may be equal to test_case.size or may have 1 cell offset

    def __init__(self, optilio_mode):
        self.size = (0, 0)
        self.data_matrix_size = (0,0)
        self.probability = 0.0
        self.data = None
        self.optilio_mode = optilio_mode
        self.probabilities = None
        self.visited = None
        self.extended_data_matrix = False
        self.ground_truth = None

    def parse_test_case(self, lines, extended):
        if extended:
            self.extended_data_matrix = True
            self.data_matrix_size = (self.size[0]+2, self.size[1]+2)
            self.data = np.zeros(self.data_matrix_size)
        else:
            self.data_matrix_size = self.size
            self.data = np.zeros(self.size)

        offset = 1 if self.extended_data_matrix else 0
        for row in range(self.size[0]):
            for column in range(self.size[1]):
                self.data[row + offset, column + offset] = state_to_num_mapping[lines[row][column]]

        self.probabilities = np.full(self.size, self.probability)
        self.visited = np.full(self.size, False)

    def check_differences(self):
        try:
            print(self.ground_truth - self.probabilities)
        except:
            pass

def parse_test_data(dir, filename, optilio_mode, extended_data_matrix):
    import os
    in_path = '{}/{}/{}.in'.format(os.getcwd(), dir, filename)
    out_path = '{}/{}/{}.out'.format(os.getcwd(), dir, filename)

    test_cases = []
    with open(in_path, 'r') as file:
        no_test_cases = int(file.readline())
        if not optilio_mode:
            print("file {}, test cases: {}".format(in_path, no_test_cases))
        for _ in range(no_test_cases):
            size = tuple([int(x) for x in file.readline().split(' ')])
            # read probability for the current test case
            probability = float(file.readline())
            # read all lines for the current problem
            lines = [file.readline() for _ in range(size[0])]
            case = test_case(optilio_mode)
            case.size = size
            case.probability = probability
            case.parse_test_case(lines, extended_data_matrix)
            test_cases.append(case)


    try:
        with open(out_path, 'r') as file:
            for case in test_cases:
                case.ground_truth = np.array([[float(x) for x in file.readline().split(' ')] for _ in range(case.size[0])])
    except:
        pass

    return test_cases

def parse_test_data_from_input(optilio_mode, extended_data_matrix):
    no_test_cases = int(input())
    test_cases = []
    if not optilio_mode:
        print('stdin, no of test cases {}'.format(no_test_cases))
    for _ in range(no_test_cases):
        size = tuple([int(x) for x in input().split(' ')])
        probability = float(input())
        lines = [input() for _ in range(size[0])]
        case = test_case(optilio_mode)
        case.size = size
        case.probability = probability
        case.parse_test_case(lines, extended_data_matrix)
        test_cases.append(case)

    return test_cases

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
                pass

    if optilio_mode:
        for row in range(test_case.probabilities.shape[0]):
            for column in range(test_case.probabilities.shape[1]):
                print(test_case.probabilities[row, column])
            print()
        # print(test_case.probabilities)

filename = '2019_00_small'


# def main():
tests = []
if optilio_mode:
    tests = parse_test_data_from_input(optilio_mode, extended_data_matrix = True)
else:
    tests = parse_test_data(dir, filename, optilio_mode, extended_data_matrix = True)

if optilio_mode:
    [calculate_probabilities(test_case) for test_case in tests]
else:
    test_case = tests[0]
    print(test_case.data[1:-1, 1:-1])
    calculate_probabilities(test_case)
    print(test_case.visited)
    print(test_case.probabilities)
    print()
    print(test_case.ground_truth)
    print()
    test_case.check_differences()

# if __name__ == "__main__":
#     main()