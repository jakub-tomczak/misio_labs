from enum import Enum


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
    max_steps = 0

    def __init__(self, is_clean: bool, step_num: int, p: float):
        self.step = step_num
        self.is_clean = is_clean
        self.p = p

    def __hash__(self):
        return hash(self.step) * hash(self.is_clean) * hash(self.p)

    def __eq__(self, other):
        if other is State:
            return other.step == self.step and other.is_clean == self.is_clean and other.p == self.p
        return False

known_step = dict()


def reward(is_clean: bool, action: Action):
    if action == Action.GO:
        return Reward.GO.value
    if action == Action.SUCK and is_clean:
        return Reward.SUCK_CLEAN.value
    return Reward.SUCK_DIRTY.value


def np(px, p):
    return px + (1-px)*p


def step(state: State):
    if state.step <= 0:
        return 0

    if step in known_step:
        return known_step

    step_num = state.step - 1

    px = np(state.p, State.p)

    points_go = reward(state.is_clean, Action.GO)
    points_suck = reward(state.is_clean, Action.SUCK)

    # points go
    points_go += step(State(False, step_num, State.p))*px
    points_go += step(State(True, step_num, State.p))*(1-px)

    # points suck
    points_suck += step(State(False, step_num, px))*State.p
    points_suck += step(State(True, step_num, px))*(1-State.p)

    return max(points_suck, points_go)


def calc(p0, p, steps):
    State.p = p
    State.p0 = p0
    State.max_steps = steps

    dirty_state = State(False, steps, p)
    clean_state = State(True, steps, p)

    return step(dirty_state)*p0 + (1-p0)*step(clean_state)


def load_and_run():
    num_of_instances = int(input())
    for _ in range(num_of_instances):
        p0, p, steps = input().split(' ')
        print('{:.5f}'.format(calc(float(p0), float(p), int(steps))))


def main():
    load_and_run()


if __name__ == "__main__":
    main()
