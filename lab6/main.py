import numpy as np
import sys

OPTILIO_MODE = False


class StoreAgent(object):
    def __init__(self, m: int):
        self.m = m
        self.gamma = 0.0
        self.l1 = 0
        self.l2 = 0
        self.l3 = 0
        self.l4 = 0
        self.g = 0
        self.c = 0
        self.f = 0

    def run(self):
        arr = np.zeros((self.m + 1, self.m + 1))
        np.savetxt(sys.stdout.buffer, arr, fmt='%d')
        return arr


def run_agent():
    n_worlds = int(input())
    for world_i in range(n_worlds):
        m = int(input())
        agent = StoreAgent(m)
        agent.gamma = float(input())
        agent.l1, agent.l2, agent.l3, agent.l4 = [int(x) for x in input().split()]
        agent.g = int(input())
        agent.c = int(input())
        agent.f = int(input())

        agent.run()


def test_store_agent():
    agent = StoreAgent(20)
    agent.gamma = .9
    agent.l1, agent.l2, agent.l3, agent.l4 = [3, 4, 3, 2]
    agent.g = 100
    agent.c = 20
    agent.f = 5

    result = agent.run()

    true_result = np.array([[-1, 3, 0, 0, 2, -4, 2, 0, -4, -4, -3, 0, -1, -3, 2, -5, 2, 0, 4, -4, 2],
                            [3, -1, -1, 4, -1, -5, 4, -1, -4, -3, -1, -2, 1, 0, 0, 3, 1, -1, 1, 3, -4],
                            [-1, 0, 0, 3, -1, 2, 2, -3, -5, -4, 4, -2, -3, -4, 2, -3, -4, 3, -5, 1, -4],
                            [3, 0, 0, 1, -1, -1, 3, 1, 1, 1, -4, -3, -2, -3, 1, -3, 2, -2, -4, -2, -3],
                            [-1, -3, -4, 2, -5, 4, -2, 0, -4, -2, -1, -5, -5, 4, -3, -4, -4, 0, 0, 0, 1],
                            [1, -5, -2, 4, -5, -1, -4, 2, 0, 4, 0, 4, 2, -2, -5, 4, -3, -2, 3, -1, -4],
                            [-3, -4, -2, 4, -2, 1, 2, -3, 0, 0, -1, -4, 3, 3, -4, -2, -5, 2, 1, -3, -4],
                            [-5, 3, 0, -4, -1, -5, -1, 2, 4, 2, -5, 3, -1, -1, 0, 4, 4, -3, 2, -2, -4],
                            [-4, 2, 3, -2, 0, 4, -1, 0, -4, 2, -3, 0, 3, -2, -1, -1, 0, -1, 3, -3, 2],
                            [4, 4, -1, 2, 4, 1, -1, -1, 0, 2, -5, 2, -3, 4, -2, 4, 1, -4, -2, -4, -2],
                            [2, -1, -5, -3, -5, -5, -1, 0, 3, -3, -3, 3, 0, 0, 4, -2, 0, -5, 2, -1, 4],
                            [-2, 2, -2, -5, 3, 3, 2, -4, 1, -1, -3, -3, -3, -1, 4, 0, -3, 1, -5, 0, 1],
                            [-3, 4, -2, -2, 4, -5, 0, -5, 3, -5, -5, 4, 2, 2, -5, -2, -2, -5, 2, -4, 2],
                            [-1, -2, 1, -5, 1, 1, 4, 4, 1, 2, -3, -5, -4, -4, -1, 3, -1, 3, -2, -1, 1],
                            [-1, -5, -5, -4, -4, 3, 1, 0, -4, 4, 4, 0, -5, -2, 1, -4, -1, -2, -1, -5, 2],
                            [2, 3, -4, 2, -5, -3, 1, 4, 3, -5, -2, -1, -5, -3, 4, -2, -5, -4, -4, 3, -5],
                            [0, 0, 3, 4, 2, -3, 3, -2, -3, 0, -4, -2, -4, 4, 0, -5, 3, 0, -3, -4, -2],
                            [0, 4, 0, 0, -3, 3, 1, -2, 2, 1, -3, -2, 4, 3, -3, -1, 0, 1, 0, 1, 4],
                            [0, -4, -3, -3, -1, 4, 1, -4, -1, 1, 0, 3, -4, 2, -4, 2, -5, 2, -1, 3, -5],
                            [-3, -1, 4, -3, 2, 4, 2, 4, -4, 3, -2, 1, 3, -4, 0, -1, -1, -3, -2, 1, 1],
                            [2, -2, 1, 4, 4, 2, 0, 3, -2, 2, 2, -1, 4, 1, 2, 4, -5, -2, -3, 0, 2]])

    comparison = np.where(result != true_result)
    if len(comparison):
        print('false values at coords:')
        for y, x in zip(*comparison):
            print(y, x)


if __name__ == '__main__':
    if OPTILIO_MODE:
        run_agent()
    else:
        test_store_agent()
