from player import Player
import logging
import numpy as np
from template_player import TemplatePlayer

class InteractivePlayer(TemplatePlayer):

    def __init__(self, player_id):
        super().__init__(player_id)
        self.player_type = 'InteractivePlayer'


    def make_a_move(self):
        assert self.board_obj is not None, f'Player {self.player_id} not assigned a board'
        print(f'Current Board state: \n{self.board_obj.get_board_state()}')
        current_state = self.board_obj.get_board_state_id()
        action_alternatives = self.board_obj.get_empty_cell_indices()
        user_input = input('Enter move: ')
        try:
            my_selected_move = int(user_input)
            if my_selected_move not in action_alternatives:
                logging.exception('Your move is not valid. I will choose randomly!')
                my_selected_move = np.random.choice(action_alternatives)
        except ValueError:
            logging.exception('You entered invalid choice. I will choose randomly!')
            my_selected_move = np.random.choice(action_alternatives)

        logging.debug('Player {} selected: {} in state: {}'.format(self.player_id, my_selected_move, current_state))
        return my_selected_move

