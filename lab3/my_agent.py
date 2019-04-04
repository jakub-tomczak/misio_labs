#!/bin/bash
from scipy.signal import convolve2d
from misio.optilio.lost_wumpus import run_agent
from misio.lost_wumpus.testing import test_locally
from misio.lost_wumpus.agents import SnakeAgent, AgentStub, RandomAgent
from misio.lost_wumpus._wumpus import Action, Field

import numpy as np

optilio_mode = True
debug_mode = False
draw_histogram = False
plot_pause_seconds = 1
local_test_file = 'tests/one.in'

if not optilio_mode and draw_histogram:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

def debug_print(text):
    if not optilio_mode and debug_mode:
        print(text)


# [0,0] UP => [4,0]
# [0,0] LEFT => [0,4]
def safe_position(x: int, boundary: int) -> int:
    return (boundary + x) % boundary

# returns position of the field 
# after performing action from field current_position
def cyclic_position(current_position: (int, int), action: Action, boundary: (int, int)) -> (int, int):
    h, w = boundary
    x, y = current_position
    if action == Action.DOWN:
        y = safe_position(y+1, h)
    elif action == Action.UP:
        y = safe_position(y-1, h)
    elif action == Action.RIGHT:
        x = safe_position(x+1, w)
    else:
        x = safe_position(x-1, w)
    return x, y

# example: x = 1, y = 4, boundary = 5, options (Action.Left, Action.Right)
# should return (2, Action.Left)
def cyclic_distance(src: int, dst: int, boundary: int, options: (Action, Action)) -> (int, Action):
    through_0_distance = min(src, dst) + boundary - max(src, dst)
    distance = abs(dst - src)
    if through_0_distance < distance:
        # it is shorter to go through 0
        # but if src < dst go left, otherwise go right
        return through_0_distance, options[0] if src < dst else options[1]
    else:
        return distance, options[1] if src < dst else options[0]

def create_convolution_masks(p1, p2):
    mask_down = np.array([
         [0, 0, 0],
         [0, 0, 0],
         [0, p2, 0],
        [p2, p1, p2],
         [0, p2, 0]
    ], dtype='float')
    mask_right = np.rot90(mask_down, 1)
    mask_up = np.rot90(mask_right, 1)
    mask_left = np.rot90(mask_up, 1)
    return {
            Action.DOWN: mask_down,
            Action.LEFT: mask_left,
            Action.RIGHT: mask_right,
            Action.UP: mask_up
    }

class MyAgent(AgentStub):
    def __init__(self, *args, **kwargs):
        super(MyAgent, self).__init__(*args, **kwargs)
        self.masks = create_convolution_masks(self.p, (1-self.p)/4)
        self.entered_cave_mask = np.where(self.map == Field.CAVE, self.pj, self.pn)
        self.entered_not_cave_mask = np.where(self.map == Field.EMPTY, self.pj, self.pn)
        self.reset()

    def sense(self, sensory_input: bool):
        if sensory_input == Field.CAVE:
            debug_print('{} this is cave'.format(self.move_counter))
            self.histogram *= self.entered_cave_mask
        else:
            self.histogram *= self.entered_not_cave_mask
            debug_print('{} this is nothing'.format(self.move_counter))

        self.histogram /= self.histogram.max()
        self.histogram[self.exit_location] = 0
        debug_print('ok')

    def move(self):
        self.move_counter += 1
        # argmax returns coordinates in one-dimensional array format
        best_place = np.unravel_index(self.histogram.argmax(), self.histogram.shape)

        chosen_direction = None

        highest_p = 0.0
        next_move = (0,0)

        # for all neighbours
        for neighbour_direction in Action:
            neighbour_position = cyclic_position(best_place, neighbour_direction, (self.h, self.w))
            # find the one with the highest probability
            if self.histogram[neighbour_position] > highest_p:
                highest_p = self.histogram[neighbour_position]
                next_move = neighbour_position
                chosen_direction = neighbour_direction
        
        self.histogram = convolve2d(self.histogram, self.masks[chosen_direction], 'same', "wrap")
        self.histogram /= self.histogram.max()
        # print("y:{}, {}; x:{} {}".format(y_diff, y_direction, x_diff, x_direction))
        debug_print("decision is {}".format(chosen_direction))
        debug_print("{}".format('-'*20))

        self.plot_location()

        return chosen_direction


    def reset(self):
        debug_print("resetting")
        self.histogram = np.ones_like(self.map)
        self.where_am_i = np.ones_like(self.map)
        self.holes_probabilities = np.ones_like(self.map)
        self.move_counter = 1
        self.exit_location = np.unravel_index(np.argmax(self.map), self.map.shape)
        debug_print("exit location {}".format(self.exit_location))

    def plot_location(self):
        if not optilio_mode and draw_histogram:
            grid = ax.imshow(self.histogram, cmap="prism")
            grid.set_data(self.histogram)
            plt.ion()
            plt.show()
            plt.pause(plot_pause_seconds)

if __name__ == "__main__":
    if optilio_mode:
        run_agent(MyAgent)
    else:
        test_locally(local_test_file, MyAgent, verbose=True, seed=1, n=10)