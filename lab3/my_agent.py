#!/bin/bash
from scipy.signal import convolve2d
from misio.optilio.lost_wumpus import run_agent
from misio.lost_wumpus.testing import test_locally
from misio.lost_wumpus.agents import SnakeAgent, AgentStub, RandomAgent
from misio.lost_wumpus._wumpus import Action, Field

import numpy as np

optilio_mode = False
debug_mode = False
draw_histogram = False
plot_pause_seconds = .5
local_test_file = 'tests/2015.in'

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
    y, x = current_position
    if action == Action.DOWN:
        y = safe_position(y+1, h)
    elif action == Action.UP:
        y = safe_position(y-1, h)
    elif action == Action.RIGHT:
        x = safe_position(x+1, w)
    else:
        x = safe_position(x-1, w)
    return y, x

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
    ])
    mask_right = np.rot90(mask_down, 1)
    mask_up = np.rot90(mask_right, 1)
    mask_left = np.rot90(mask_up, 1)
    return {
            Action.DOWN: mask_down,
            Action.LEFT: mask_left,
            Action.RIGHT: mask_right,
            Action.UP: mask_up
    }


class Move:
    def __init__(self, location, dy, dx, p, dir):
        self.location = location
        self.dy = dy
        self.dx = dx
        self.p = p
        self.dir = dir

    def coefficient(self, h, w):
        return self.p**2
        # return self.p * (h - self.location[0] + w - self.location[1])


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

    def move(self):
        self.move_counter += 1
        # argmax returns coordinates in one-dimensional array format
        best_place = np.unravel_index(self.histogram.argmax(), self.histogram.shape)
        chosen_direction = None
        highest_p = 0.0

        max_val = self.histogram.max()
        votes = {
            Action.LEFT: 0,
            Action.RIGHT: 0,
            Action.UP: 0,
            Action.DOWN: 0
        }

        if self.move_counter % 20 == 0:
            chosen_direction = np.random.choice(Action)
        else:
            for r, row in enumerate(self.histogram):
                for c, _ in enumerate(row):
                    if self.histogram[r,c] > max_val * .90:
                        dy, y_direct = cyclic_distance(r, self.exit_location[0], self.h, (Action.UP, Action.DOWN))
                        dx, x_direct = cyclic_distance(c, self.exit_location[1], self.w, (Action.LEFT, Action.RIGHT))
                        votes[y_direct] += 1
                        votes[x_direct] += 1

            votes_sorted = sorted(votes.items(), key=lambda x: x[1], reverse=True)

            if self.move_counter > 1:
                self.last_but_one_move = self.last_move

            if self.last_move == votes_sorted[0][0]:
                # votes_sorted[0][0] => direction with the highest number of votes
                if self.last_move in [Action.LEFT, Action.RIGHT] and self.the_same_direction_counter < self.w - 1 \
                    or self.last_move in [Action.UP, Action.DOWN] and self.the_same_direction_counter < self.h - 1:
                    self.the_same_direction_counter += 1
                    chosen_direction = votes_sorted[0][0]
                else:
                    # take the next move, we have chosen too many times move with the highest number of votes
                    chosen_direction = votes_sorted[1][0]
                    self.last_move = chosen_direction
                    self.the_same_direction_counter = 0
            else:
                # new last move
                chosen_direction = votes_sorted[0][0]
                self.last_move = chosen_direction
                self.the_same_direction_counter = 0

            if chosen_direction in [Action.LEFT, Action.RIGHT] and self.last_but_one_move in [Action.LEFT, Action.RIGHT]:
                self.returning += 1
                if self.returning >= 2:
                    for vote in votes_sorted:
                        if vote[0] not in [Action.LEFT, Action.RIGHT]:
                            chosen_direction = vote[0]
                            break
            elif chosen_direction in [Action.DOWN, Action.UP] and self.last_but_one_move in [Action.DOWN, Action.UP]:
                self.returning += 1
                if self.returning >= 2:
                    for vote in votes_sorted:
                        if vote[0] not in [Action.DOWN, Action.UP]:
                            chosen_direction = vote[0]
                            break
            else:
                self.returning = 0
        self.last_move = chosen_direction
        # print('{}: {}'.format(self.move_counter, chosen_direction))
        debug_print(chosen_direction)
        self.plot_location()
        self.histogram = convolve2d(self.histogram, self.masks[chosen_direction], 'same', "wrap")
        self.histogram /= self.histogram.max()
        
        debug_print("decision is {}".format(chosen_direction))
        debug_print("{}".format('-'*20))

        return chosen_direction


    def reset(self):
        # print("resetting")
        self.histogram = np.ones_like(self.map)
        self.where_am_i = np.ones_like(self.map)
        self.holes_probabilities = np.ones_like(self.map)
        self.move_counter = 1
        self.exit_location = np.unravel_index(np.argmax(self.map), self.map.shape)
        self.move_history = []
        self.the_same_direction_counter = 0
        self.last_move = None
        self.last_but_one_move = None
        self.returning = 0
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
        test_locally(local_test_file, MyAgent, verbose=True)