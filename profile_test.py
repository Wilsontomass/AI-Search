import sys

import yaml
import multiprocessing as mp

from app import FishingDerbyMinimaxApp
from fishing_game_core.game_tree import Node
from main import Application, Settings
from player import SuperModel


def receiver(receiver_pipe):
    """
    Receive message from the receiver pipe
    :return:
    """
    if not receiver_pipe.poll(None):
        sys.exit(-1)  # time limit
    msg = receiver_pipe.recv()
    return msg


if __name__ == "__main__":
    # Load the settings from the yaml file
    settings = Settings()
    settings_dictionary = yaml.safe_load(open("settings.yml", 'r'))
    settings.load_from_dict(settings_dictionary)

    game_pipe_send, player_pipe_receive = mp.Pipe()

    app = FishingDerbyMinimaxApp()
    app.load_settings(settings)
    app.set_receive_send_pipes(None, game_pipe_send)
    app.run()
    print("yay")

    model = SuperModel(receiver(player_pipe_receive), depth=7)
    msg = receiver(player_pipe_receive)
    node = Node(message=msg, player=0)
    move = model.next_move(node)
    print(move)
