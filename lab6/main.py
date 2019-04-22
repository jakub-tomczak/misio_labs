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


if __name__ == '__main__':
    if OPTILIO_MODE:
        run_agent()
    else:
        agent = StoreAgent(20)
        agent.gamma = .9
        agent.l1, agent.l2, agent.l3, agent.l4 = [3, 4, 3, 2]
        agent.g = 100
        agent.c = 20
        agent.f = 5

        agent.run()
