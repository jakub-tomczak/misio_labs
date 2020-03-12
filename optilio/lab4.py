from enum import Enum
from functools import lru_cache

class Action(Enum):
    GO = 0
    SUCK = 1


class Reward(Enum):
    GO = -1
    SUCK_DIRTY = 10
    SUCK_CLEAN = 0


class State:
    p = 0.0
    p0 = 0.0


known_step = dict()


def reward(is_clean: bool, action: Action):
    if action == Action.GO:
        return Reward.GO.value
    if action == Action.SUCK and is_clean:
        return Reward.SUCK_CLEAN.value
    return Reward.SUCK_DIRTY.value


def np(px, p_):
    return px + (1-px)*p_


@lru_cache(maxsize=None)
def step(state: (bool, int, float)):
    step_num = state[1] - 1
    px = np(state[2], State.p)

    points_go = reward(state[0], Action.GO)
    points_suck = reward(state[0], Action.SUCK)

    # points go
    points_go += calculate_or_get_from_dict((False, step_num, State.p))*px
    points_go += calculate_or_get_from_dict((True, step_num, State.p))*(1-px)


    # points suck
    points_suck += calculate_or_get_from_dict((False, step_num, px))*State.p
    points_suck += calculate_or_get_from_dict((True, step_num, px))*(1-State.p)

    res = max(points_suck, points_go)
    # known_step[state] = res
    return res


def calculate_or_get_from_dict(state):
    if state[1] <= 0:
        return 0

    return step(state)
    # if state in known_step:
    #     return known_step[state]
    # else:
    #     res = step(state)
    #     known_step[state] = res
    #
    #     return res


def calc(p0, p, steps):
    State.p = p
    State.p0 = p0

    dirty_state = (False, steps, State.p0)
    clean_state = (True, steps, State.p0)

    return calculate_or_get_from_dict(dirty_state)*State.p0 + (1-State.p0)*calculate_or_get_from_dict(clean_state)


def load_and_run():
    #global  known_step
    num_of_instances = int(input())
    # from time import time
    for _ in range(num_of_instances):
        known_step = dict()
        step.cache_clear()
        p0, p, steps = input().split(' ')
        res = calc(float(p0), float(p), int(steps))
        print(format(res, ".5f"))
        #print('{:.5f}'.format(calc(float(p0), float(p), int(steps))))


def main():
    load_and_run()


if __name__ == "__main__":
    main()
