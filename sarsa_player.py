from player import Player
import logging
import numpy as np

class SARSAPlayer(Player):

    def __init__(self, player_id, epsilon):
        super().__init__(player_id)
        self.epsilon = epsilon

    def make_a_move(self):
        assert self.board_obj is not None, f'Player {self.player_id} not assigned a board'
        current_state = self.board_obj.get_board_state_id()
        action_alternatives = self.board_obj.get_empty_cell_indices()
        best_q_value = max(self.get_q_value(current_state))
        best_action_alternatives = self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['qvalue']==best_q_value), 'Action'].values.tolist()
        other_suboptimal_alternatives = list(set(action_alternatives) - set(best_action_alternatives))

        if len(other_suboptimal_alternatives) > 0 and np.random.uniform() < self.epsilon:
            my_selected_move = np.random.choice(other_suboptimal_alternatives)
        else:
            my_selected_move = np.random.choice(best_action_alternatives)

        logging.debug('Player {} selected: {} in state: {}'.format(self.player_id, my_selected_move, current_state))
        return my_selected_move

