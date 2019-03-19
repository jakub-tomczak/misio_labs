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

def parse_test_data(filename, optilio_mode, extended_data_matrix):
    with open(filename, 'r') as file:
        no_test_cases = int(file.readline())
        test_cases = []
        if not optilio_mode:
            print("file {}, test cases: {}".format(filename, no_test_cases))
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