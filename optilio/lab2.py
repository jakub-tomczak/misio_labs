import sys
from collections import Counter
import numpy as np

optilio_mode = True
display_front_enabled = False
debug_print_enabled = False
dir = 'test_cases'

np.set_printoptions(precision=2)

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
    if debug_print_enabled and not optilio_mode:
        print(text)


def display_front(row, col, front, test_case):
    if not display_front_enabled:
        return
    np.set_printoptions(precision=0, floatmode='fixed')
    print("{}front for row:{} col:{}{}".format('*' * 15, row, col, '*' * 15))
    front_n = np.zeros(test_case.size)
    for fr in front.cells_in_front:
        front_n[fr.row, fr.col] = 2
        for breeze in fr.breezes_in_neighborhood:
            front_n[breeze[0], breeze[1]] = 1
    front_n[row, col] = 3

    print(front_n)
    print("{} end front {}".format('*' * 15, '*' * 15))
    np.set_printoptions(precision=2, floatmode='fixed')


class Front:
    def __init__(self):
        self.cells_in_front = []
        self.breezes = set()
        self.breezes_l = []
        self.breezes_list = []
        self.counted_breezes = None
        self.one_cell_breezes = None

    # add cell with all its breezes,
    # merge the existing breezes set with cell's breezes
    def add_cell(self, cell):
        self.cells_in_front.append(cell)
        self.breezes |= cell.breezes_in_neighborhood
        for breeze in cell.breezes_in_neighborhood:
            self.breezes_list.append(breeze)

    def count_breezes(self):
        if self.counted_breezes is None:
            self.counted_breezes = Counter(self.breezes_list)
        return self.counted_breezes

    def find_one_cell_breezes(self):
        if self.breezes_list is None:
            self.counted_breezes()
        if self.one_cell_breezes is None:
            breezes = Counter(self.breezes_list)
            self.one_cell_breezes = set([x for x, y in breezes.items() if y == 1])

    def add_breeze(self, breeze):
        self.breezes_l.append(breeze)

class CellInFront(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.breezes_in_neighborhood = set()

    def __str__(self):
        return 'row:{} col:{}, breezes{}'.format(self.row, self.col, self.breezes_in_neighborhood)

    def __hash__(self):
        return self.row * self.col + len(self.breezes_in_neighborhood)

    def __eq__(self, other):
        if not isinstance(other, CellInFront):
            return False
        elif self.row == other.row and self.col == other.col:
            return True

    def get_coordinates(self):
        return self.row, self.col


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
        self.probabilities = None
        self.probability = 0.0
        self.visited = None
        self.extended_data_matrix = False
        self.ground_truth = None
        self.width = 0
        self.height = 0
        self.during_check = None
        self.instance_count = 0
        self.certain = None

    def parse_test_case(self, lines, extended):
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

        self.probabilities = np.full(self.size, self.probability)
        self.visited = np.zeros(self.size, dtype=bool)
        self.width = self.size[1]
        self.height = self.size[0]
        self.certain = np.zeros(self.size)

    def clear_during_check(self):
        self.during_check = np.zeros(self.size, dtype=bool)

    def check_differences(self):
        if type(self.ground_truth) is not np.ndarray:
            debug_print("Error")
            return np.ones(self.probabilities.shape)
        else:
            return self.ground_truth - self.probabilities


def parse_test_data(dir, filename, optilio_mode, extended_data_matrix):
    import os
    in_path = '{}/{}/{}.in'.format(os.getcwd(), dir, filename)
    out_path = '{}/{}/{}.out'.format(os.getcwd(), dir, filename)

    test_cases = []
    count = 0
    with open(in_path, 'r') as file:
        no_test_cases = int(file.readline())
        debug_print("file {}, test cases: {}".format(in_path, no_test_cases))
        for _ in range(no_test_cases):
            size = tuple([int(x) for x in file.readline().split(' ')])
            # read probability for the current test case
            # read all lines for the current problem
            probability = float(file.readline())
            lines = [file.readline() for _ in range(size[0])]
            case = TestCase(optilio_mode)
            case.probability = probability
            case.size = size
            case.parse_test_case(lines, extended_data_matrix)
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


def parse_test_data_from_input(optilio_mode, extended_data_matrix):
    no_test_cases = int(input())
    test_cases = []
    debug_print('stdin, no of test cases {}'.format(no_test_cases))

    count = 0
    for _ in range(no_test_cases):
        size = tuple([int(x) for x in input().split(' ')])
        probability = float(input())
        lines = [input() for _ in range(size[0])]
        case = TestCase(optilio_mode)
        case.size = size
        case.probability = probability
        case.parse_test_case(lines, extended_data_matrix)
        case.instance_count = count
        test_cases.append(case)
        count = count + 1

    return test_cases


# finds all breezes around cell and returns this cell encapsulated in a class
def element_to_front(row, col, test_case):
    element = CellInFront(row, col)

    # add breezes in neighborhood to an element
    if row - 1 >= 0 and test_case.raw_data[row - 1, col] == state_to_num_mapping['B']:
        element.breezes_in_neighborhood.add((row - 1, col))
    if row + 1 < test_case.size[0] and test_case.raw_data[row + 1, col] == state_to_num_mapping['B']:
        element.breezes_in_neighborhood.add((row + 1, col))
    if col - 1 >= 0 and test_case.raw_data[row, col - 1] == state_to_num_mapping['B']:
        element.breezes_in_neighborhood.add((row, col - 1))
    if col + 1 < test_case.size[1] and test_case.raw_data[row, col + 1] == state_to_num_mapping['B']:
        element.breezes_in_neighborhood.add((row, col + 1))

    return element


def are_coords_the_same(coords1, coords2):
    return coords1[0] == coords2[0] and coords1[1] == coords2[1]


def find_front(row, col, front, test_case, previous):
    previous_sym, previous_coords = previous
    debug_print("find_front: y:{}, x:{}, what? {}, current front: {}, previous? {}"\
                .format(row, col, num_to_state_mapping[test_case.raw_data[row, col]], front, previous))

    if previous_sym == test_case.raw_data[row, col] or test_case.raw_data[row, col] == state_to_num_mapping['O']:
        debug_print('{} {} the same symbol in the row'.format(row, col))
        return

    if test_case.during_check[row, col]:
        debug_print('{} {} already during check'.format(row, col))

    test_case.during_check[row, col] = True
    if test_case.raw_data[row, col] == state_to_num_mapping['B']:
        front.add_breeze((row, col))

    if num_to_state_mapping[previous_sym] == 'B' and \
            test_case.raw_data[row, col] == state_to_num_mapping['?']:
        # came from 'B' and i am '?'
        debug_print('{}, {} i am ?, you came from B, I should be in the front'.format(row, col))
        if not test_case.visited[row, col]:
            front.add_cell(element_to_front(row, col, test_case))
        else:
            pass #print("I could not be added {} {}".format(row, col))

    next_previous = (test_case.raw_data[row, col], (row, col))
    # go left
    if col - 1 >= 0 and not test_case.during_check[row, col-1]:
        find_front(row, col - 1, front, test_case, next_previous)
    # go right
    if col + 1 < test_case.width and not test_case.during_check[row, col+1]:
        find_front(row, col + 1, front, test_case, next_previous)
    # go up
    if row - 1 >= 0 and not test_case.during_check[row-1, col]:
        find_front(row - 1, col, front, test_case, next_previous)
    # go down
    if row + 1 < test_case.height and not test_case.during_check[row+1, col]:
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
    mask = np.full((test_case.size[0] + 2, test_case.size[1] + 2), False)

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

            # corner_penalty = (0 if 0 < (row - 1) else 1) + (0 if row + 2 < test_case.data_matrix_size[0] else 1) + \
            #                  (0 if 0 < (column - 1) else 1) + (0 if column + 2 < test_case.data_matrix_size[1] else 1)
            # too distant `?`
            if test_case.data[row, column] == state_to_num_mapping['?'] and sum_around == 0:

                # around `?` there are only `?`
                test_case.visited[row - 1, column - 1] = True

            # elif test_case.data[row, column] == state_to_num_mapping['B'] and sum_around == (3 - corner_penalty) * \
            #         state_to_num_mapping[
            #             'O']:
            #     for coords in neighbors_coordinates:
            #         if 0 < row + coords[0] < test_case.data_matrix_size[0] - 1 \
            #                 and 0 < column + coords[1] < test_case.data_matrix_size[1] - 1 \
            #                 and test_case.data[row + coords[0], column + coords[1]] == state_to_num_mapping['?']:
            #             print('Setting to 1 {} {}'.format(row + coords[0] - 1, column + coords[1] - 1))
            #             test_case.visited[row + coords[0] - 1, column + coords[1] - 1] = True
            #             test_case.probabilities[row + coords[0] - 1, column + coords[1] - 1] = 1.0


def calculate_probabilities(test_case):
    preprocessing(test_case)
    debug_print(test_case.visited)

    # stores mapping from a CellInFront to front
    cell_to_front = dict()
    creating_front = 0

    for row in range(test_case.height):
        for column in range(test_case.width):
            if test_case.visited[row, column]:
                continue
            current_cell = element_to_front(row, column, test_case)

            if current_cell.get_coordinates() not in cell_to_front:
                # find a front
                # add the current cell to its front
                front = Front()
                front.add_cell(current_cell)
                creating_front = creating_front + 1
                test_case.during_check = np.zeros(test_case.size, dtype=bool)
                find_front(row, column, front, test_case, (-1, (-1, -1)))

                front.find_one_cell_breezes()

                for cell in front.cells_in_front:
                    cell_to_front[cell] = front
                    breezes_with_one_cell_in_this_cell = cell.breezes_in_neighborhood.intersection(front.one_cell_breezes)
                    if len(breezes_with_one_cell_in_this_cell) > 0 and not test_case.visited[cell.row, cell.col]\
                            and test_case.raw_data[cell.row, cell.col] == state_to_num_mapping['?']:
                        # cell is surrounded by a breeze with one cell
                        # print('certain cell {} {}'.format(cell.row, cell.col))
                        test_case.certain[cell.row, cell.col] = 1.0
                        # test_case.probabilities[cell.row, cell.col] = 1.0
                        # test_case.visited[cell.row, cell.col] = True

            if not test_case.visited[row, column]:
                # # set probability == 1 to all '?' without front and with 'B' as neighbor
                # neighbors_sum = test_case.data[row, column + 1] + test_case.data[row + 1, column] + \
                #                 test_case.data[row + 1, column + 2] + test_case.data[row + 2, column + 1]
                # # if neighbors_sum is between 0 and 5 it means that there is at least one B
                # # if front is empty and there is at least one B it means that this cell must be a hole
                # if len(front.breezes) == 0 and state_to_num_mapping['?'] < neighbors_sum < state_to_num_mapping['O']:
                #     test_case.probabilities[row, column] = 1.0
                #     test_case.visited[row, column] = True
                #     continue

                # display front
                if not optilio_mode:
                   display_front(row, column, front, test_case)

                # create mapping from ? to a position in a binary number
                cell_to_binary_position_mapping = dict()
                reversed_cell_to_binary_position_mapping = dict()
                if len(front.cells_in_front) > 16:
                    continue
                for cell, o in zip(front.cells_in_front, range(len(front.cells_in_front))):
                    debug_print(
                        '{}, {}: {},\tbreezes {}'.format(cell.row, cell.col, 1 << o, cell.breezes_in_neighborhood))
                    cell_to_binary_position_mapping[cell] = 1 << o
                    reversed_cell_to_binary_position_mapping[1 << o] = cell

                # find all legal combinations
                legal_combinations = []
                num_of_combinations = 1 << len(front.cells_in_front)
                for i in range(1, 1 + num_of_combinations):
                    new_breeze = set()
                    for f in front.cells_in_front:
                        if cell_to_binary_position_mapping[f] & i:
                            new_breeze |= f.breezes_in_neighborhood

                    # all breezes have at least one active ? field around -> get as much as possible
                    # which means that we always find all possibilities
                    # TODO check whether it is possible to take only the minimal subset of cells to combination
                    if len(new_breeze) == len(front.breezes):
                        # debug_print("legal combination is {}".format(i))
                        legal_combinations.append(i)

                # calculate the cumulative probability
                p_active = 0.0
                p_inactive = 0.0
                for combination in legal_combinations:
                    if combination & cell_to_binary_position_mapping[current_cell]:
                        p = 1.0
                        for cell in front.cells_in_front:
                            # don't add the current cell's probability
                            if cell == current_cell:
                                continue

                            prob = test_case.probability
                            to_mult = prob if combination & cell_to_binary_position_mapping[cell] else 1-prob
                            p = p * to_mult
                        p_active = p_active + p
                    else:
                        p = 1.0
                        for cell in front.cells_in_front:
                            # don't add the current cell's probability
                            if cell == current_cell:
                                continue

                            prob = test_case.probability
                            to_mult = prob if combination & cell_to_binary_position_mapping[cell] else 1 - prob
                            p = p * to_mult
                        p_inactive = p_inactive + p

                p_active = p_active * test_case.probability
                p_inactive = p_inactive * (1 - test_case.probability)

                norm = 1.0 / (p_active + p_inactive)
                test_case.probabilities[row, column] = norm * p_active

    if optilio_mode:
        np.savetxt(sys.stdout.buffer, test_case.probabilities, delimiter=' ', fmt='%.2f')
        # for row_inner in range(test_case.probabilities.shape[0]):
        #     for column_inner in range(test_case.probabilities.shape[1]):
        #         print(format(test_case.probabilities[row_inner, column_inner], ".2f"), end=' ')
        #     print()


filename = '3'  # 2019_00_small

def main():
    tests = []
    if optilio_mode:
        tests = parse_test_data_from_input(optilio_mode, extended_data_matrix=True)
    else:
        tests = parse_test_data(dir, filename, optilio_mode, extended_data_matrix=True)

    if optilio_mode:
        [calculate_probabilities(test_case) for test_case in tests]
    else:
        for i in range(len(tests)):
            test_case = tests[i]
            calculate_probabilities(test_case)

            print('{}summary{}'.format('-' * 10, '-' * 10))

            diff = abs(np.mean(test_case.check_differences()))
            if diff < 0.001:
                print('instance {} OK'.format(test_case.instance_count))
            else:
                print(print('instance {} NOT OK, diff = {}'.format(test_case.instance_count, diff)))
                print(test_case.visited)
                print(test_case.probabilities)
                print()
                print(test_case.ground_truth)
                print()
                print(test_case.check_differences())
                print()
                print(test_case.certain)


if __name__ == "__main__":
    main()
