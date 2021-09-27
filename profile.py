from numpy import inf

from fishing_game_core.game_tree import Node
from player import alphabeta


if __name__ == "__main__":
    msg =
    initial_tree_node = Node(message=msg, player=0)
    value, node = alphabeta(initial_tree_node, 9, -inf, inf)
