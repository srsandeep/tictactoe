import logging
from game import Game
from player import Player

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    game_inst = Game(3, 2, win_count=3)
    game_inst.register_players([Player(1), Player(2)])
    game_inst.play_game_n_time(10000)

    logging.info('Exiting program')
