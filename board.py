import numpy as np
import math
import pandas as pd
import os
from qtable_states import QTable, STATE_TABLE_COLUMN_NAMES_FOR_STATE
import itertools
import logging

STATE_TABLE_FILE_NAME = 'all_states_actions.csv'


class Board:

    def __init__(self, board_size: int, win_count: int = 3):
        self.board_size = board_size
        self.win_count = win_count
        self.winning_patterns = self.generate_winning_pattern()
        # self.winning_patterns = WINNING_PATTERNS
        self.board_state = np.zeros((self.board_size, self.board_size), dtype=int)

        self.state_table_file_name = str(self.board_size) + '_' + STATE_TABLE_FILE_NAME
        self.state_table_file_path = os.path.join(os.path.dirname(__file__), self.state_table_file_name)
        self.state_table_file_exists = os.path.exists(self.state_table_file_path)
        if not self.state_table_file_exists:
            qtable = QTable(self.board_size)
            qtable.create_state_action_look_up_table()
        self.board_state_df = pd.read_csv(self.state_table_file_path)

    def reset_board(self):
        self.board_state = np.zeros((self.board_size, self.board_size), dtype=int)

    def get_board_state(self):
        return self.board_state

    def get_board_state_id(self):
        found_indices = np.argwhere((self.board_state_df[STATE_TABLE_COLUMN_NAMES_FOR_STATE].values == self.board_state.flatten()).all(axis=1)).flatten().tolist()
        # assert len(found_indices) == 1, f'Found indices {found_indices} for search pattern {self.board_state.flatten()}'
        try:
            logging.debug(f'StateID: {found_indices} and {self.board_state_df.iloc[found_indices[0]]["StateID"]}')
        except Exception as e:
            logging.error(f'Found indices: {found_indices}')
            logging.error(f'DF shape: {self.board_state_df.shape}')
            logging.exception(e)
        # return found_indices[0]
        return self.board_state_df.iloc[found_indices[0]]['StateID']

    def mark_move(self, player_id, row_pos, col_pos):
        ret_status = False
        logging.debug('Player {} selected row:{} col:{}'.format(player_id, row_pos, col_pos))
        if row_pos >= self.board_size or col_pos >= self.board_size or row_pos < 0 or col_pos < 0:
            logging.error('Invalid input!')
        elif self.board_state[row_pos,col_pos] != 0:
            logging.error('Place already taken!')
        else:
            self.board_state[row_pos, col_pos] = player_id
            ret_status = True

        return ret_status

    def mark_a_move(self, player_id, cell_index):
        row_pos = int(cell_index/self.board_size)
        col_pos = cell_index % self.board_size
        assert self.mark_move(player_id, row_pos, col_pos), 'Selection of celldx: {} by player:{} failed'.format(cell_index, player_id)
        return (row_pos, col_pos)

    def show_my_moves(self, player_id):
        return [each_row_col[0] * self.board_size + each_row_col[1] for each_row_col in np.argwhere(self.board_state==player_id)]
    
    def get_empty_cell_indices(self):
        logging.debug('Empty Indices: {}'.format([each_row_col[0] * self.board_size + each_row_col[1] for each_row_col in np.argwhere(self.board_state==0)]))
        return [each_row_col[0] * self.board_size + each_row_col[1] for each_row_col in np.argwhere(self.board_state==0)]

    def is_valid_move(self, flat_board_state, select_cell):
        return True if select_cell in [each_empty_index for each_empty_index in np.argwhere(flat_board_state==0)] else False

    def register_player_ids(self, player_id_list):
        self.player_ids = player_id_list

    def is_life_saver(self, selected_pos, next_player_id):
        temp_board_state = self.board_state.copy()
        temp_board_state[int(selected_pos/self.board_size),int(selected_pos % self.board_size)] = next_player_id
        is_life_saving_move = False
        player_map = np.where(temp_board_state == next_player_id, 1, 0)
        logging.debug(f'pos:{selected_pos}, nextPid:{next_player_id},state:{temp_board_state}, map:{player_map}')
        logging.debug('Verifying life saver for player_id: {} with array: {}'.format(next_player_id, player_map))
        for each_win_pattern in self.winning_patterns:
            if np.logical_and(player_map, each_win_pattern).sum() == self.win_count:
                logging.debug('{} would have won with pattern {}'.format(next_player_id, player_map))
                is_life_saving_move = True
                break
        return is_life_saving_move


    def is_game_over(self):
        exit_criteria_satisfied = False
        winner_id = 0
        if 0 not in list(set(np.unique(self.board_state).flatten().tolist())):
            exit_criteria_satisfied = True
            logging.debug('Game over. All cells occupied!')
        for each_player_id in list(set(np.unique(self.board_state).flatten().tolist()) - set([0])):
            player_map = np.where(self.board_state == each_player_id, 1, 0)
            logging.debug('Verifying for player_id: {} with array: {}'.format(each_player_id, player_map))
            for each_win_pattern in self.winning_patterns:
                if np.logical_and(player_map, each_win_pattern).sum() == self.win_count:
                    logging.debug('{} has won with pattern {}'.format(each_player_id, player_map))
                    exit_criteria_satisfied = True
                    winner_id = each_player_id
                    break
            if winner_id != 0:
                break


        logging.debug(list(set(np.unique(self.board_state).flatten().tolist()) - set([0])))
        # return (self.board_state.size - np.count_nonzero(self.board_state)) == 0
        return exit_criteria_satisfied, winner_id

    def generate_patterns_at_pos(self, row_index=0, column_idx=0):
        pattern_list = []
        if (row_index + self.win_count <= self.board_size) or (column_idx + self.win_count <= self.board_size):

            if row_index + self.win_count <= self.board_size:
                ret_array = np.zeros((self.board_size, self.board_size), dtype=int)
                ret_array[row_index:row_index+self.win_count, column_idx] = 1
                pattern_list += [ret_array.flatten().tolist()]
                pattern_list += [ret_array.transpose().flatten().tolist()]
                pattern_list += [np.fliplr(ret_array).flatten().tolist()]
                pattern_list += [np.fliplr(ret_array.transpose()).flatten().tolist()]
                pattern_list += [np.flipud(ret_array).flatten().tolist()]
                pattern_list += [np.flipud(ret_array.transpose()).flatten().tolist()]

            if (row_index + self.win_count <= self.board_size) and (column_idx + self.win_count <= self.board_size):
                ret_array = np.zeros((self.board_size, self.board_size), dtype=int)
                ret_array[row_index:row_index+self.win_count, column_idx:column_idx+self.win_count] = \
                    np.identity(self.win_count, dtype=int)
                pattern_list += [ret_array.flatten().tolist()]
                pattern_list += [ret_array.transpose().flatten().tolist()]
                pattern_list += [np.fliplr(ret_array).flatten().tolist()]
                pattern_list += [np.fliplr(ret_array.transpose()).flatten().tolist()]
                pattern_list += [np.flipud(ret_array).flatten().tolist()]
                pattern_list += [np.flipud(ret_array.transpose()).flatten().tolist()]

            if column_idx + self.win_count <= self.board_size:
                ret_array = np.zeros((self.board_size, self.board_size), dtype=int)
                ret_array[row_index,column_idx:column_idx+self.win_count] = 1
                pattern_list += [ret_array.flatten().tolist()]
                pattern_list += [ret_array.transpose().flatten().tolist()]
                pattern_list += [np.fliplr(ret_array).flatten().tolist()]
                pattern_list += [np.fliplr(ret_array.transpose()).flatten().tolist()]
                pattern_list += [np.flipud(ret_array).flatten().tolist()]
                pattern_list += [np.flipud(ret_array.transpose()).flatten().tolist()]

            # if (row_index + self.win_count <= self.board_size) and (column_idx + self.win_count <= self.board_size):
            #     ret_array = np.zeros((self.board_size, self.board_size), dtype=int)
            #     ret_array[row_index:row_index+self.win_count, column_idx:column_idx+self.win_count] = \
            #         np.fliplr(np.identity(self.win_count, dtype=int))
            #     pattern_list += [ret_array.flatten().tolist()]
            #     pattern_list += [np.fliplr(ret_array).flatten().tolist()]
            #     pattern_list += [np.flipud(ret_array).flatten().tolist()]

        logging.debug('Pattern generated at r:{}, c: {} is {}'.format(row_index, column_idx, pattern_list))

        return pattern_list


    def generate_winning_pattern(self):
        return_patterns_list = []
        for row_idx in range(math.ceil(self.board_size/2)):
            for col_idx in range(row_idx+1):
                return_patterns_list += self.generate_patterns_at_pos(row_index=row_idx, 
                                                                column_idx=col_idx)
        return_patterns_list = pd.DataFrame(return_patterns_list).drop_duplicates().values.tolist()
        converted_to_arrays = [np.array(each_list_item, dtype=int).reshape((self.board_size, self.board_size)) for each_list_item in return_patterns_list]
        logging.debug('Winning patterns: {}'.format(converted_to_arrays))
        return converted_to_arrays

