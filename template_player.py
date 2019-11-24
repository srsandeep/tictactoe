import numpy as np
import math
import pandas as pd
import os
import itertools
import logging

class TemplatePlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.my_moves = []
        self.win_count = 0

    def assign_board(self, board_inst):
        self.board_obj = board_inst

    def game_prestart_hook(self):
        self.load_all_info()

    def increment_win_count(self):
        self.win_count = self.win_count + 1

    def get_win_count(self):
        return self.win_count

    def make_a_move(self):
        assert self.board_obj is not None, f'Player {self.player_id} not assigned a board'
        current_state = self.board_obj.get_board_state_id()
        empty_indices = self.board_obj.get_empty_cell_indices()
        assert empty_indices, 'No empty cells to make a move'
        my_selected_move = np.random.choice(np.array(empty_indices))
        logging.debug('Player {} selected: {} in state: {}'.format(self.player_id, my_selected_move, current_state))
        return my_selected_move

    def update_sars_info(self):
        pass

    def update_sas_info(self, current_state, current_action, next_state):
        pass

    def update_reward_info(self, last_reward):
        pass

    def save_all_info(self):
        pass

    def load_all_info(self):
        pass
