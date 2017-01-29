import time
import argparse
import sys
from collections import deque
import os


class Directions:

    UP = 'Up'
    DOWN = 'Down'
    LEFT = 'Left'
    RIGHT = 'Right'
    ALL_MOVES = [UP, DOWN, LEFT, RIGHT]


class State:

    def __init__(self, state, parent, move):
        self.state = state
        self.parent = parent
        self.move_frm_parent = move

    def is_goal_state(self):
        return self.state == GOAL


    def __eq__(self, other):
        if other == None:
            return False
        if not self.state == other.state:
            return False
        return True

    def __hash__(self):
        return hash(self.state)

    def __str__(self):
        return self.state.__str__()

    def __unicode__(self):
        return self.state.__unicode__()

    def __repr__(self):
        return self.state.__repr__()

    def get_legal_moves(self):

        legal_moves = []
        n = int(len(self.state)**0.5)
        blank_pos = self.state.index(0)

        for d in Directions.ALL_MOVES:

            # for UP to be legal, blank should not be in top row
            if d == Directions.UP and blank_pos >= n:
                legal_moves.append(d)

            # for DOWN to be legal, blank should not be in last row
            if d == Directions.DOWN and blank_pos < n**2 - n:
                legal_moves.append(d)

            # for LEFT to be legal, blank should not be in left column
            if d == Directions.LEFT and not blank_pos % n == 0:
                legal_moves.append(d)

            # for RIGHT to be legal, blank should not in right column
            if d == Directions.RIGHT and not blank_pos % n ==  n-1:
                legal_moves.append(d)

        return legal_moves

    def neighbors(self):
        result = []
        legal_moves = self.get_legal_moves()

        for d in legal_moves:
            new_state = list(self.state)
            n = int(len(self.state) ** 0.5)
            blank_pos = self.state.index(0)
            # find the new position for blank based on move
            if d == Directions.UP:
                new_blank_pos = blank_pos - n
            elif d == Directions.DOWN:
                new_blank_pos = blank_pos + n
            elif d == Directions.LEFT:
                new_blank_pos = blank_pos - 1
            elif d == Directions.RIGHT:
                new_blank_pos = blank_pos + 1
            else:
                raise ValueError('Should never come here')

            new_state[blank_pos] = new_state[new_blank_pos]
            new_state[new_blank_pos] = 0

            result.append(State(tuple(new_state), self, d))

        return result


def bfs(initial_state):

    # start time
    st_time = time.clock()

    # node will have (board, depth)
    frontier = deque([(initial_state, 0)])
    # to have efficinet way to check for existence of a state in frontier
    visited_states = set()
    visited_states.add(initial_state)
    nodes_expanded = 0
    max_fringe_size = -1
    max_search_depth = -1

    while len(frontier) > 0:
        node = frontier.popleft()
        state = node[0] # get the board

        if state.is_goal_state():
            # return (path to goal, cost_of_path, nodes expanded, fringe_size
            path_to_goal = []
            cur_state = state
            while cur_state.parent != None:
                path_to_goal.append(cur_state.move_frm_parent)
                cur_state = cur_state.parent
            path_to_goal = path_to_goal[::-1]
            cost_of_path = len(path_to_goal)
            fringe_size = len(frontier)
            search_depth = node[1]
            runtime = time.clock() - st_time
            if not os.name == 'nt':
                import resource
                max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # get ru_maxrss
            else:
                max_ram_usage = None

            write_output(path_to_goal, cost_of_path, nodes_expanded, fringe_size, max_fringe_size,
                     search_depth, max_search_depth, runtime, max_ram_usage)
            return

        # the node out of queue is not a goal and we need to expand it
        nodes_expanded += 1

        for neighbor in state.neighbors():
            if not neighbor in visited_states:
                new_depth = node[1]+1
                frontier.append((neighbor, new_depth))
                visited_states.add(neighbor)
                max_fringe_size = max(max_fringe_size, len(frontier))
                max_search_depth = max(max_search_depth, new_depth)

    print("ERROR: bfs should never reach this point")

def dfs(initial_state):

    # start time
    st_time = time.clock()

    # node will have (board, depth)
    frontier = deque([(initial_state, 0)])
    # to have efficinet way to check for existence of a state in frontier
    visited_states = set()
    visited_states.add(initial_state)
    nodes_expanded = 0
    max_fringe_size = -1
    max_search_depth = -1

    while len(frontier) > 0:
        node = frontier.pop()
        #print(node)
        state = node[0] # get the board

        if state.is_goal_state():
            # return (path to goal, cost_of_path, nodes expanded, fringe_size
            path_to_goal = []
            cur_state = state
            while cur_state.parent != None:
                path_to_goal.append(cur_state.move_frm_parent)
                cur_state = cur_state.parent
            path_to_goal = path_to_goal[::-1]
            cost_of_path = len(path_to_goal)
            fringe_size = len(frontier)
            search_depth = node[1]
            runtime = time.clock() - st_time
            if not os.name == 'nt':
                import resource
                max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # get ru_maxrss
            else:
                max_ram_usage = None

            write_output(path_to_goal, cost_of_path, nodes_expanded, fringe_size, max_fringe_size,
                     search_depth, max_search_depth, runtime, max_ram_usage)
            return

        # the node out of queue is not a goal and we need to expand it
        nodes_expanded += 1
        print(nodes_expanded)

        ## for dfs we need to reverse the neighbour list before inserstion so
        # that popping will result in UDLR order
        for neighbor in state.neighbors()[::-1]:

            if not neighbor in visited_states:
                new_depth = node[1]+1
                frontier.append((neighbor, new_depth))
                visited_states.add(neighbor)
                max_fringe_size = max(max_fringe_size, len(frontier))
                max_search_depth = max(max_search_depth, new_depth)

    print("ERROR: dfs should never reach this point")

def write_output(path_to_goal, cost_of_path, nodes_expanded, fringe_size,
                 max_fringe_size, search_depth, max_search_depth, runtime,
                 max_ram_usage):
    output = open('output.txt', 'w')
    output.write('path_to_goal: {}\n'.format(path_to_goal))
    output.write('cost_of_path: {}\n'.format(cost_of_path))
    output.write('nodes_expanded: {}\n'.format(nodes_expanded))
    output.write('fringe_size: {}\n'.format(fringe_size))
    output.write('max_fringe_size: {}\n'.format(max_fringe_size))
    output.write('search_depth: {}\n'.format(search_depth))
    output.write('max_search_depth: {}\n'.format(max_search_depth))
    output.write('running_time: {}\n'.format(runtime))
    output.write('max_ram_usage: {}'.format(max_ram_usage))
    output.close()


def read_command( argv ):

    parser = argparse.ArgumentParser(description='Run Search.')
    parser.add_argument('method', choices=['bfs', 'dfs', 'ast', 'ida'], help='type of search')
    parser.add_argument('state', type=lambda s: tuple([int(item) for item in s.split(',')]))
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = read_command(sys.argv[1:])
    state = State(args.state, None, None)
    method = args.method
    GOAL = tuple([i for i in range(len(state.state))])
    if method == 'bfs':
        bfs(state)
    elif method == 'dfs':
        dfs(state)

