#!/usr/bin/env python3
from math import inf
import time
from typing import Optional

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class SuperModelNode:

    def __init__(self, node: Node):
        self.node = node
        self.children = None
        self.heuristic_value = heuristic(self.node.state)
        self.minimax_value = None  # this can be used to sort instead
        self.minimaxed_children = False

    def compute_and_get_children(self):
        if self.children:
            return self.children
        self.children = [SuperModelNode(child) for child in self.node.compute_and_get_children()]
        return self.sort_children()

    def sort_children(self):
        """Should only be run after compute_and_get_children"""
        if self.minimaxed_children:
            return self.children

        reverse = True if self.node.state.player == 0 else False
        # places the highest value node to be searched first
        # We sort on the minimax value (from a previous IDDFS run) if it exist, else the heuristic value
        self.children.sort(key=lambda n: n.minimax_value or n.heuristic_value, reverse=reverse)
        if all([n.minimax_value is not None for n in self.children]):
            self.minimaxed_children = True
        return self.children


# sketch:


class SuperModel:

    def __init__(self, initial_data, depth: int):
        self.init_data = initial_data
        self.depth = depth
        self.time_limit = None
        self.tree: Optional[SuperModelNode] = None

    def next_move(self, next_node: Node):
        # Start timer
        start_time = time.time()
        max_time = start_time + 0.01

        next_node = SuperModelNode(next_node)
        next_node.compute_and_get_children()
        if self.tree is not None:  # This basically caches all the work we've managed to do
            # The opponent made a move, we select the subtree that matches that move
            for child in self.tree.children:
                if child.node.state.get_hook_positions()[1] == next_node.node.state.get_hook_positions()[1]:
                    next_node = child
                    break

        # IDDFS
        best_child = None
        for depth in range(2, self.depth+1):  # from 1 to depth
            if time.time() >= max_time:
                break
            # To save resources, we do a kind of mini alphabeta at this level where we can track the best child
            max_value = -inf
            for child in next_node.compute_and_get_children():
                value = self.alphabeta(child, depth-1, -inf, inf, False)
                if value > max_value:
                    max_value = value
                    best_child = child
            if best_child is None:
                return 0, time.time() - start_time
            self.tree = best_child  # we start from here next time
        return best_child.node.move, time.time() - start_time

    def alphabeta(self, s_node: SuperModelNode, depth: int, alpha: float, beta: float, is_maximising=False) -> float:
        if depth == 0:
            return s_node.heuristic_value
        else:
            children = s_node.compute_and_get_children()
            if is_maximising:
                max_value = -inf
                for child in children:
                    max_value = max(max_value, self.alphabeta(child, depth - 1, alpha, beta, False))
                    alpha = max(alpha, max_value)
                    if beta <= alpha:
                        break
                s_node.minimax_value = max_value  # can be used to sort in future
                return max_value
            else:
                min_value = inf
                for child in children:
                    min_value = min(min_value, self.alphabeta(child, depth - 1, alpha, beta, True))
                    beta = min(beta, min_value)
                    if beta <= alpha:
                        break
                s_node.minimax_value = min_value  # can be used to sort in future
                return min_value


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move, s_time = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": s_time})

    def initialize_model(self, initial_data) -> SuperModel:
        """
        Initialize your minimax model
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: object

        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3},
          'fish1': {'score': 2, 'type': 1},
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }

        Please note that the number of fishes and their types is not fixed between test cases.
        """
        # EDIT THIS METHOD TO RETURN A MINIMAX MODEL ###
        return SuperModel(initial_data, depth=7)

    def search_best_next_move(self, model: SuperModel, initial_tree_node: Node):
        """
        Use your minimax model to find best possible next move for player 0 (green boat)
        :param model: Minimax model
        :type model: object
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE FROM MINIMAX MODEL ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        move, s_time = model.next_move(initial_tree_node)
        return ACTION_TO_STR[move], s_time


def heuristic(state):
    return manhattan_dist_value(state, 0) - manhattan_dist_value(state, 1)


def manhattan_dist_value(state: State, player):
    """smaller distance means you are closer to getting the fishes value"""
    player_pos = state.get_hook_positions()[player]

    fish_closeness_value = -inf
    for fish_num, fish_pos in state.get_fish_positions().items():
        x_abs = abs(player_pos[0] - fish_pos[0])
        dist = min(x_abs, 20 - x_abs) + abs(player_pos[1] - fish_pos[1])
        new_fish_val = state.get_fish_scores()[fish_num] * (1/max(dist, 1))
        fish_closeness_value = max(fish_closeness_value, new_fish_val)

    return fish_closeness_value


def manhattan_dist_value_verbose(state: State, player):
    """smaller distance means you are closer to getting the fishes value"""
    player_pos = state.get_hook_positions()[player]

    fish_closeness_value = -inf
    best_fish = None
    best_dist = None
    for fish_num, fish_pos in state.get_fish_positions().items():
        x_abs = abs(player_pos[0] - fish_pos[0])
        dist = (min(x_abs, 20 - x_abs)  # For the wraparound x
                + abs(player_pos[1] - fish_pos[1]))
        new_fish_val = state.get_fish_scores()[fish_num] * (1 / max(dist, 1))
        if new_fish_val > fish_closeness_value:
            fish_closeness_value = new_fish_val
            best_fish = fish_pos
            best_dist = dist

    print(f"Fish at {best_fish} is {best_dist} to player {player} hook at {player_pos} with value of {fish_closeness_value}")

    return fish_closeness_value

