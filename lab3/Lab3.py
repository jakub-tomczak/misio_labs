import sys
from collections import Counter
import numpy as np

optilio_mode = True
debug_print_enabled = False
dir = 'tests'

np.set_printoptions(precision=2)
state_to_num_mapping = {
    '.': 0, # no hole
    'J': 1, # hole
    'W': 5  # exit
}

def debug_print(text):
    if debug_print_enabled and not optilio_mode:
        print(text)
class TestCase:

    # test_case.size is a real size of the input data
    # test_case.data_matrix_size may be equal to test_case.size or may have 1 cell offset
    def __init__(self, optilio_mode):
        self.size = (0, 0)
        self.data_matrix_size = (0, 0)
        # contains loaded data with optional 1 cell offset
        self.data = None
        # contains raw loaded data without any offset
        self.raw_data = None
        self.optilio_mode = optilio_mode
        self.probability = 0.0
        self.pj = 0.0
        self.pn = 0.0
        self.width = 0
        self.height = 0
        self.instance_count = 0
        self.extended_data_matrix = False

    def parse_test_case(self, lines, extended, pj_pn):
        if extended:
            self.extended_data_matrix = True
            self.data_matrix_size = (self.size[0] + 2, self.size[1] + 2)
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

        # (y, x)
        self.height, self.width = self.size
        # (pj, pn)
        self.pj, self.pn = pj_pn


def parse_test_data(dir, filename, optilio_mode, extended_data_matrix):
    import os
    in_path = '{}/{}/{}.in'.format(os.getcwd(), dir, filename)
    out_path = '{}/{}/{}.out'.format(os.getcwd(), dir, filename)
    test_cases = []
    count = 0
    with open(in_path, 'r') as file:
        read_method = file.readline if not optilio_mode else input
        no_test_cases = int(read_method())
        debug_print("file {}, test cases: {}".format(in_path, no_test_cases))
        for _ in range(no_test_cases):
            # probability
            # pj pn
            # y x
            probability = float(read_method())
            pj_pn = tuple([float(x) for x in read_method().split(' ')])
            size = tuple([int(x) for x in read_method().split(' ')])
            lines = [read_method() for _ in range(size[0])]
            case = TestCase(optilio_mode)
            case.probability = probability
            case.size = size
            case.parse_test_case(lines, extended_data_matrix, pj_pn)
            case.instance_count = count
            test_cases.append(case)
            count = count + 1

    try:
        with open(out_path, 'r') as file:
            for case in test_cases:
                case.ground_truth = np.array([[float(x) for x in file.readline().split(' ') if x != '\n'] for _ in range(case.size[0])])
    except:
        pass

    return test_cases

def find_exit(test_case):
    pass

def main():
    filename = "2015"
    tests = parse_test_data(dir, filename, optilio_mode, extended_data_matrix=False)

    if optilio_mode:
        [find_exit(test_case) for test_case in tests]
    else:
        for i in range(len(tests)):
            test_case = tests[i]
            find_exit(test_case)

if __name__ == "__main__":
    main()