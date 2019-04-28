import numpy as np
from main import StoreAgent


def test_store_agent(print_invalid_coords: bool = False, short_test: bool = True):
    m = 5 if short_test else 20
    agent = StoreAgent(m)
    agent.gamma = .9
    agent.l1, agent.l2, agent.l3, agent.l4 = [3, 4, 3, 2]
    agent.g = 100
    agent.c = 20
    agent.f = 5

    result = agent.run()

    if short_test:
        true_result = np.array([[0, 0, 0, 0, 0, -1],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0],
                                [2, 2, 2, 1, 1, 0]])
    else:
        true_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, -2, -2, -3, -3, -3, -3, -3, -4, -4, -4],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -2, -2, -2, -2, -2, -3, -3, -3, -3],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -2],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [4, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [4, 4, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 4, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 5, 4, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 5, 4, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 5, 4, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 5, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 5, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 5, 5, 4, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 5, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 5, 5, 4, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                [5, 5, 5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0]])

    comparison = np.where(result != true_result)
    if len(comparison):
        print('number of errors: {0}, {1:2.2f}%'.format(len(comparison[0]),
                                                        float(len(comparison[0])) / (agent.m + 1) ** 2 * 100))
        if print_invalid_coords:
            print('false values at coords:')
            for y, x in zip(*comparison):
                print(y, x)