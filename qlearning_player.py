import numpy as np
import math
import pandas as pd
import os
import itertools
import logging
from template_player import TemplatePlayer
from qtable_states import STATE_TABLE_COLUMN_NAMES_FOR_STATE as state_col_names

class QlearningPlayer(TemplatePlayer):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.update_rl_parameters(alpha=0.7,discount_rate=0.8, initial_q_value=0.6)
        self.app_init_qvalue = True
        self.q_table = None
        self.player_type = 'QlearningPlayer'

    def assign_board(self, board_inst):
        self.board_obj = board_inst
        if self.q_table is None:
            self.q_table = self.board_obj.board_state_df.copy()
            if self.app_init_qvalue:
                self.init_qvalue(self.init_q_value)

    def init_qvalue(self, initial_qvalue):
        self.q_table['qvalue'] = initial_qvalue

    def update_rl_parameters(self, alpha=0.9, discount_rate=0.95, initial_q_value=0.6):
        self.alpha = alpha
        self.discount_rate = discount_rate
        self.init_q_value = initial_q_value

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
        # self.q_table.loc[(self.q_table['StateID']==q_state) & (self.q_table['Action']==q_action), 'numVisit'] = self.q_table.loc[(self.q_table['StateID']==q_state) & (self.q_table['Action']==q_action), 'numVisit'] + 1

    def update_q_value(self, current_state, current_action, current_reward, next_state):
        current_q_value = self.get_q_value(current_state, q_action=current_action)
        off_policy_max_sdash_adash = max(self.get_q_value(q_state=next_state))
        updated_q_value = (1 - self.alpha) * current_q_value + self.alpha * (current_reward + self.discount_rate * off_policy_max_sdash_adash)
        logging.debug(f'Update Q value: state:{current_state} currentval:{current_q_value} nextvalue:{updated_q_value}')
        self.set_q_value(q_state=current_state, q_action=current_action, q_value=updated_q_value)

    def make_a_move(self):
        assert self.board_obj is not None, f'Player {self.player_id} not assigned a board'
        current_state = self.board_obj.get_board_state_id()
        best_q_value = max(self.get_q_value(current_state))
        my_selected_move = np.random.choice(self.q_table.loc[(self.q_table['StateID']==current_state) & (self.q_table['qvalue']==best_q_value), 'Action'])
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

    def save_all_info(self):
        if self.learning_mode:
            state_file = self.player_type + '_' + str(self.player_id) + '_' + 'q_table.csv'
            self.q_table.to_csv(state_file, index=False)

    def load_all_info(self):
        state_file = self.player_type + '_' + str(self.player_id) + '_' + 'q_table.csv'
        if os.path.exists(state_file):
            logging.warning(f'Loading learnt data from {state_file}')
            self.q_table = pd.read_csv(state_file)
            self.app_init_qvalue = False
