import numpy as np
import math
import pandas as pd
import os
import itertools
import logging

class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.my_moves = []
        self.win_count = 0
        self.alpha = 0.7
        self.discount_rate = 0.8

    def assign_board(self, board_inst):
        self.board_obj = board_inst
        self.q_table = self.board_obj.board_state_df.copy()
        self.q_table['qvalue'] = 0

    def init_qvalue(self, initial_qvalue):
        self.q_table['qvalue'] = initial_qvalue

    def update_rl_parameters(self, alpha, discount_rate):
        self.alpha = alpha
        self.discount_rate = discount_rate

    def increment_win_count(self):
        self.win_count = self.win_count + 1

    def get_win_count(self):
        return self.win_count

    def get_q_value(self, q_state, q_action=None):
        assert q_state is not None, 'Q value requested for state None. Invalid'
        if q_action is not None:
            logging.debug(f'Q value for state:{q_state}, action:{q_action} is {self.q_table.loc[(self.q_table["StateID"]==q_state) & (self.q_table["Action"]==q_action),"qvalue"].values.tolist()[0]}')
            return self.q_table.loc[(self.q_table['StateID']==q_state) & (self.q_table['Action']==q_action),'qvalue'].values.tolist()[0]
        else:
            logging.debug(f'Q value for state:{q_state} is {self.q_table.loc[self.q_table["StateID"]==q_state, "qvalue"].values.tolist()}')
            return self.q_table.loc[self.q_table['StateID']==q_state, 'qvalue'].values.tolist()

    def set_q_value(self, q_state=None, q_action=None, q_value=0):
        assert (q_state is not None) and (q_action is not None), 'State and action both needed to set q value'
        self.q_table.loc[(self.q_table['StateID']==q_state) & (self.q_table['Action']==q_action), 'qvalue'] = q_value

    def update_q_value(self, current_state, current_action, current_reward, next_state):
        current_q_value = self.get_q_value(current_state, q_action=current_action)
        off_policy_max_sdash_adash = max(self.get_q_value(q_state=next_state))
        updated_q_value = (1 - self.alpha) * current_q_value + self.alpha * (current_reward + self.discount_rate * off_policy_max_sdash_adash)
        self.set_q_value(q_state=current_state, q_action=current_action, q_value=updated_q_value)

    def make_a_move(self):
        assert self.board_obj is not None, f'Player {self.player_id} not assigned a board'
        current_state = self.board_obj.get_board_state_id()
        # empty_indices = self.board_obj.get_empty_cell_indices()
        best_q_value = max(self.get_q_value(current_state))
        my_selected_move = np.random.choice(self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['qvalue']==best_q_value), 'Action'])
        # assert empty_indices, 'No empty cells to make a move'
        # my_selected_move = np.random.choice(np.array(empty_indices))
        logging.debug('Player {} selected: {} in state: {}'.format(self.player_id, my_selected_move, current_state))
        return my_selected_move

    def update_sars_info(self):
        self.update_q_value(self.current_state, self.current_action, self.last_reward, self.next_state)

    def update_sas_info(self, current_state, current_action, next_state):
        self.current_state = current_state
        self.current_action = current_action
        self.next_state = next_state

    def update_reward_info(self, last_reward):
        self.last_reward = last_reward

