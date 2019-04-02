#!/bin/bash
from misio.optilio.lost_wumpus import run_agent
from misio.lost_wumpus.testing import test_locally
from misio.lost_wumpus.agents import SnakeAgent, AgentStub
from misio.lost_wumpus._wumpus import Action, Field
import numpy as np

np.set_printoptions(precision=3, suppress=True)

optilio_mode = False
local_test_file = 'tests/one.in'

class MyAgent(AgentStub):
    def __init__(self, *args, **kwargs):
        super(MyAgent, self).__init__(*args, **kwargs)
        self.reset()

    def sense(self, sensory_input: bool):
        holes_prob = np.mean(self.holes_probabilities)
        if sensory_input == Field.CAVE:
            print('as {} this is cave'.format(self.move_counter))
            if holes_prob > 0:
                self.histogram = (self.pj * self.histogram) / holes_prob
        else:
            if holes_prob < 1:
                self.histogram = (self.pn * self.histogram) / (1-holes_prob)
            print('{} this is nothing'.format(self.move_counter))

    def move(self):
        # print(self.histogram)
        self.move_counter += 1
        return np.random.choice(Action)

    def get_histogram(self):
        return self.histogram

    def reset(self):
        print("resetting")
        self.histogram = np.ones_like(self.map)
        self.where_am_i = np.ones_like(self.map)
        self.holes_probabilities = np.ones_like(self.map)
        self.move_counter = 1

if __name__ == "__main__":
    if optilio_mode:
        run_agent(SnakeAgent)
    else:
        test_locally(local_test_file, MyAgent, verbose=True, seed=1, n=1)