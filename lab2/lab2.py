from aima3.agents import *
from misio.aima import *
import numpy as np

optilio_mode = False
dir = 'test_cases'

np.set_printoptions(precision=2, floatmode='fixed')

import numpy as np

# O has value of 5 to distinguish form a sum of 4 B
state_to_num_mapping = {
    '?': 0,
    'B': 1,
    'O': 5
}

num_to_state_mapping = {
    0: '?',
    1: 'B',
    5: 'O',
    -1: 'None'
}
# [north, east, south, west]
neighbors_coordinates = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def debug_print(text):
    if not optilio_mode:
        print(text)

class test_case(object):
    # test_case.size is a real size of the input data
    # test_case.data_matrix_size may be equal to test_case.size or may have 1 cell offset

    def __init__(self, optilio_mode):
        self.size = (0, 0)
        self.data_matrix_size = (0, 0)
        self.probability = 0.0
        # contains loaded data with optional 1 cell offset
        self.data = None
        # contains raw loaded data without any offset
        self.raw_data = None
        self.optilio_mode = optilio_mode
        self.probabilities = None
        self.visited = None
        self.extended_data_matrix = False
        self.ground_truth = None
        self.width = 0
        self.height = 0
        self.during_check = None

    def parse_test_case(self, lines, extended):
        if extended:
            self.extended_data_matrix = True
            self.data_matrix_size = (self.size[0]+2, self.size[1]+2)
            self.data = np.zeros(self.data_matrix_size)
            self.raw_data = self.data[1:-1, 1:-1]
        else:
            self.data_matrix_size = self.size
            self.data = np.zeros(self.size)
            self.raw_data = self.data

        offset = 1 if self.extended_data_matrix else 0
        for row in range(self.size[0]):
            for column in range(self.size[1]):
                self.data[row + offset, column + offset] = state_to_num_mapping[lines[row][column]]

        self.probabilities = np.full(self.size, self.probability)
        self.visited = np.zeros(self.size, dtype=bool)
        self.width = self.size[0]
        self.height = self.size[1]

    def clear_during_check(self):
        self.during_check = np.zeros(self.size, dtype=bool)

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
        debug_print("file {}, test cases: {}".format(in_path, no_test_cases))
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
    debug_print('stdin, no of test cases {}'.format(no_test_cases))

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


# termination indicates whether we came from `B`
def find_front(row, col, front, test_case, previous):
    previous_sym, previous_coords = previous
    debug_print("find_front: y:{}, x:{}, what? {}, current front: {}, previous? {}".format(row, col, num_to_state_mapping[test_case.raw_data[row, col]], front, previous))
    pos = np.zeros(test_case.size)
    pos[row, col] = 1
    debug_print(pos)
    if test_case.during_check[row, col]:
        debug_print('{}, {} is already during checking'.format(row, col))
        return

    if test_case.visited[row, col] and (test_case.raw_data[row, col] == state_to_num_mapping['O'] or test_case.raw_data[row, col] == state_to_num_mapping['?']):
        debug_print('{}, {} is visited and is O or ?'.format(row, col))
        return

    if test_case.raw_data[row, col] == state_to_num_mapping['?'] and previous_sym == state_to_num_mapping['?']:
        debug_print('{} {} i am the second ? in a row'.format(row, col))
        return

    test_case.during_check[row, col] = True
    #
    if num_to_state_mapping[previous_sym] == 'B':
        # came from 'B' and i am '?'
        debug_print('{}, {} i am ?, you came from B, I should be in the front'.format(row, col))
        front.append((row, col))

    next_previous = (test_case.raw_data[row, col], (row, col))
    # go left
    if col > 0 and col-1 != previous_coords[1] and row!=previous_coords[0]:
        find_front(row, col - 1, front, test_case, next_previous)
    # go right
    if col + 1 < test_case.width and col+1 != previous_coords[1] and row!=previous_coords[0]:
        find_front(row, col + 1, front, test_case, next_previous)
    # go up
    if row > 0 and col != previous_coords[1] and row-1!=previous_coords[0]:
        find_front(row - 1, col, front, test_case, next_previous)
    # go down
    if row + 1 < test_case.height and col != previous_coords[1] and row+1!=previous_coords[0]:
        find_front(row + 1, col, front, test_case, next_previous)


# cell is set as visited when:
#   - cell is a neighbor of O
#   - ? cell is surrounded by ?
#   - B cell is surrounded by 3 x O and 1 x ?
def preprocessing(test_case):
    # clear all fields with B or O
    mask = test_case.raw_data > 0
    test_case.probabilities[mask] = 0
    test_case.visited[mask] = True

    # use mask with an offset to avoid corner case checking
    mask = np.full((test_case.size[0]+2, test_case.size[1]+2), False)

    for row in range(1, test_case.data_matrix_size[0] - 1):
        for column in range(1, test_case.data_matrix_size[1] - 1):
            # clear neighbours of O
            if test_case.data[row, column] == state_to_num_mapping['O']:
                mask[row - 1, column] = True
                mask[row, column + 1] = True
                mask[row, column - 1] = True
                mask[row + 1, column] = True

    # visit all neighbors of `O`
    mask = mask[1:-1, 1:-1]
    test_case.probabilities[mask] = 0
    test_case.visited[mask] = True

    for row in range(1, test_case.data_matrix_size[0] - 1):
        for column in range(1, test_case.data_matrix_size[1] - 1):
            sum_around = test_case.data[row - 1, column] + test_case.data[row, column + 1] + \
                test_case.data[row, column - 1] + test_case.data[row + 1, column]
            # too distant `?`
            if test_case.data[row, column] == state_to_num_mapping['?']:

                # around `?` there are only `?`
                if sum_around == 0:
                    test_case.visited[row - 1, column - 1] = True

                # around `?` there are at least 3 `B`
                # TO DO maybe consider only 4 B
                # this is not correct
                # elif sum_around == 3 or sum_around == 4:
                #     test_case.visited[row - 1, column - 1] = True
                #     test_case.probabilities[row - 1, column - 1] = 1.0

            elif test_case.data[row, column] == state_to_num_mapping['B'] and sum_around == 3*state_to_num_mapping['O']:
                for coords in neighbors_coordinates:
                    if test_case.data[row + coords[0], column + coords[1]] == state_to_num_mapping['?']:
                        test_case.visited[row + coords[0] - 1, column + coords[1] - 1] = True
                        test_case.probabilities[row + coords[0] - 1, column + coords[1] - 1] = 1.0

def calculate_probabilities(test_case):
    preprocessing(test_case)

    row = 0
    col = 4
    row = row + 1
    col = col + 1
    front = []
    test_case.during_check = np.zeros(test_case.size, dtype=bool)
    find_front(row-1, col-1, front, test_case, (-1, (-1, -1)))
    neighbors_sum = test_case.data[row-1, col] + test_case.data[row, col - 1] + test_case.data[row, col+1] + test_case.data[row+1, col]
    if len(front) == 0 and neighbors_sum > 0 and neighbors_sum < 5:
        test_case.probabilities[row-1, col-1] = 1.0
        test_case.visited[row-1, col-1] = True
    return

    # calculate probabilities
    for row in range(test_case.visited.shape[0]):
        for column in range(test_case.visited.shape[1]):
            if not test_case.visited[row, column]:
                front = []
                test_case.during_check = np.zeros(test_case.size, dtype=bool)
                find_front(row, column, front, test_case, (-1, (-1, -1)))
            # check only one place
            return

    if optilio_mode:
        for row in range(test_case.probabilities.shape[0]):
            for column in range(test_case.probabilities.shape[1]):
                print(test_case.probabilities[row, column], end=' ')
            print()

filename = '2015' #2019_00_small

def main():
    tests = []
    if optilio_mode:
        tests = parse_test_data_from_input(optilio_mode, extended_data_matrix = True)
    else:
        tests = parse_test_data(dir, filename, optilio_mode, extended_data_matrix = True)

    if optilio_mode:
        [calculate_probabilities(test_case) for test_case in tests]
    else:
        test_case = tests[0]
        # print(test_case.data[1:-1, 1:-1])
        calculate_probabilities(test_case)

        print('{}summary{}'.format('-'*10, '-'*10))
        print(test_case.visited)
        print(test_case.probabilities)
        print()
        print(test_case.ground_truth)
        print()
        test_case.check_differences()

if __name__ == "__main__":
    main()