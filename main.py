import logging
from game import Game
from random_player import RandomPlayer
from qlearning_player import QlearningPlayer
from qlearning_sarsa_player import QlearningSARSAPlayer
from interactive_player import InteractivePlayer

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    game_inst = Game(3, 2, win_count=3)
    player1 = QlearningPlayer(1)
    player1.update_rl_parameters()

    player2 = QlearningPlayer(2)
    player1.update_rl_parameters()

    player3 = QlearningSARSAPlayer(1, 0.01)
    player3.update_rl_parameters()

    player4 = QlearningSARSAPlayer(2, 0.2)
    player4.update_rl_parameters()

    player5 = QlearningSARSAPlayer(1, 0.05)
    player5.update_rl_parameters(alpha=0.5, discount_rate=0.95, initial_q_value=0.6)
    # player6 = QlearningSARSAPlayer(2, 0.1)
    player6 = InteractivePlayer(2)
    # player6.update_rl_parameters(alpha=0.9, discount_rate=0.95, initial_q_value=0.6)

    game_inst.register_players([player5, player6])
    for each_player in game_inst.players:
        each_player.game_prestart_hook()
    game_inst.play_game_n_time(3)

    logging.info('Exiting program')
