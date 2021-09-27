#!/usr/bin/env python3
from __future__ import annotations

import random
from functools import lru_cache
from typing import Iterable, List, Optional

from numpy import inf

from fishing_game_core.game_tree import Node, State
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR, OPPOSITE_MOVES


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

    children: Optional[List[SuperModelNode]]

    def __init__(self, node: Node):
        self.node = node
        self.children = None
        self.heuristic_value = heuristic(self.node.state)
        self.sorted = False

    @lru_cache(maxsize=None)
    def compute_and_get_children(self):
        self.children = [SuperModelNode(child) for child in self.node.compute_and_get_children()]
        return self.children

    def sort_children(self) -> List[SuperModelNode]:
        """Should only be run after compute_and_get_children"""
        if self.sorted:
            return self.children

        reverse = True if self.node.state.player == 0 else False
        # places the highest value node to be searched first
        self.children.sort(key=lambda n: n.heuristic_value, reverse=reverse)
        self.sorted = True
        return self.children


# sketch:


class SuperModel:

    def __init__(self, initial_data, depth: int):
        self.init_data = initial_data
        self.depth = depth
        self.tree: Optional[SuperModelNode] = None

    def next_move(self, next_node: Node):
        next_node = SuperModelNode(next_node)
        next_node.compute_and_get_children()
        if self.tree is not None:  # This basically caches all the work we've managed to do
            # The opponent made a move, we select the subtree that matches that move
            for child in self.tree.children:
                if child.node.state.get_hook_positions()[1] == next_node.node.state.get_hook_positions()[1]:
                    next_node = child
                    break

        value, s_node = self.alphabeta(next_node, self.depth, -inf, inf)
        self.tree = s_node  # we start from here next time
        return s_node.node.move

    def alphabeta(self, s_node: SuperModelNode, depth: int, alpha: int, beta: int):
        if depth == 0:
            return heuristic(s_node.node.state), s_node
        else:
            children = s_node.compute_and_get_children()
            s_node.sort_children()
            player = s_node.node.state.player
            if player == 0:
                max_value = -inf
                max_child = None
                for child in children:
                    value, s_node = self.alphabeta(child, depth - 1, alpha, beta)
                    if value > max_value:
                        max_value = value
                        max_child = child
                    alpha = max(alpha, max_value)
                    if max_value >= beta:
                        break
                return max_value, max_child
            elif player == 1:
                min_value = inf
                min_child = None
                for child in children:
                    value, s_node = self.alphabeta(child, depth - 1, alpha, beta)
                    if value < min_value:
                        min_value = value
                        min_child = child
                    beta = min(beta, min_value)
                    if min_value <= alpha:
                        break
                return min_value, min_child

    def get_new_children(self, node, children: List[Node]) -> Iterable[Node]:
        """Get children, excluding the one that simply undos the prev node (excluding the root node)"""
        if node.depth == 0:
            yield from children
        else:
            for child in children:
                if child.move != OPPOSITE_MOVES[node.move]:
                    yield child


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
            best_move = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

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

        move = model.next_move(initial_tree_node)
        return ACTION_TO_STR[move]


def heuristic(state):
    val = state.player_scores[0] - state.player_scores[1]
    caught_p0, caught_p1 = state.get_caught()
    if caught_p0:
        val += state.get_fish_scores()[caught_p0]
    if caught_p1:
        val -= state.get_fish_scores()[caught_p1]
    if not caught_p1 and not caught_p0:
        val = manhattan_dist_value(state, 0) - manhattan_dist_value(state, 1)

    return val


def manhattan_dist_value(state: State, player):
    """smaller distance means you are closer to getting the fishes value"""
    player_pos = state.get_hook_positions()[player]

    fish_closeness_value = -inf
    for fish_num, fish_pos in state.get_fish_positions().items():
        dist = abs(player_pos[0] - fish_pos[0]) + abs(player_pos[1] - fish_pos[1])
        new_fish_val = state.get_fish_scores()[fish_num] * (1/max(dist, 1))
        fish_closeness_value = max(fish_closeness_value, new_fish_val)

    return fish_closeness_value
