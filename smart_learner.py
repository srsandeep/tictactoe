import logging
import numpy as np
from qlearning_sarsa_player import QlearningSARSAPlayer
from qlearning_player import QlearningPlayer

class QuickLearner(QlearningSARSAPlayer):

    def __init__(self, player_id, epsilon):
        super().__init__(player_id, epsilon)
        self.player_type = 'QuickLearner'


    def make_a_move(self):
        assert self.board_obj is not None, f'Player {self.player_id} not assigned a board'
        current_state = self.board_obj.get_board_state_id()
        action_alternatives = self.board_obj.get_empty_cell_indices()
        best_q_value = max(self.get_q_value(current_state))
        min_visits_for_best_q_value = self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['qvalue']==best_q_value), 'numVisit'].min()
        best_action_alternatives_min_visit = self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['qvalue']==best_q_value) & (self.q_table['numVisit']==min_visits_for_best_q_value), 'Action'].values.tolist()
        best_action_alternatives = self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['qvalue']==best_q_value), 'Action'].values.tolist()
        other_suboptimal_alternatives = list(set(action_alternatives) - set(best_action_alternatives))
        if len(other_suboptimal_alternatives) > 0:
            min_visits_for_suboptimal_q_value = self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['Action'].isin(other_suboptimal_alternatives)), 'numVisit'].min()
            other_suboptimal_alternatives_min_visit = self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['numVisit']==min_visits_for_suboptimal_q_value), 'Action'].values.tolist()
        else:
            other_suboptimal_alternatives_min_visit = []

        if len(other_suboptimal_alternatives_min_visit) > 0 and np.random.uniform() < self.epsilon:
            my_selected_move = np.random.choice(other_suboptimal_alternatives_min_visit)
        else:
            my_selected_move = np.random.choice(best_action_alternatives_min_visit)

        logging.debug('Player {} selected: {} in state: {}'.format(self.player_id, my_selected_move, current_state))
        return my_selected_move


class ComprehensiveLearner(QlearningPlayer):

    def __init__(self, player_id):
        super().__init__(player_id)
        self.player_type = 'ComprehensiveLearner'


    def make_a_move(self):
        assert self.board_obj is not None, f'Player {self.player_id} not assigned a board'
        current_state = self.board_obj.get_board_state_id()

        # Find minimum visited action(s) in state
        min_num_visit = self.q_table.loc[self.q_table['StateID']==current_state, 'numVisit'].min()
        # Take random action for the state that matches min numVisit.
        min_visit_actions = self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['numVisit']==min_num_visit), 'Action'].values.tolist()
        my_selected_move = np.random.choice(min_visit_actions)

        logging.debug('Player {} selected: {} in state: {}'.format(self.player_id, my_selected_move, current_state))
        return my_selected_move

