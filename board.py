import numpy as np
import math
import pandas as pd
import os
from qtable_states import QTable
import itertools
import logging

STATE_TABLE_FILE_NAME = 'all_states.csv'
SARS_ELEMENT_REWARD_INDEX = 2
GAME_RESULT_WINNER_REWARD = 1
GAME_RESULT_LOSER_REWARD = -1
GAME_RESULT_TIE_REWARD = 0

def generate_patterns_at_pos(row_index=0, column_idx=0, out_dim=0, win_length=0):
    pattern_list = []
    if (row_index + win_length <= out_dim) or (column_idx + win_length <= out_dim):

        if row_index + win_length <= out_dim:
            ret_array = np.zeros((out_dim, out_dim), dtype=int)
            ret_array[row_index:row_index+win_length, column_idx] = 1
            pattern_list += [ret_array.flatten().tolist()]
            pattern_list += [ret_array.transpose().flatten().tolist()]
            pattern_list += [np.fliplr(ret_array).flatten().tolist()]
            pattern_list += [np.fliplr(ret_array.transpose()).flatten().tolist()]
            pattern_list += [np.flipud(ret_array).flatten().tolist()]
            pattern_list += [np.flipud(ret_array.transpose()).flatten().tolist()]

        if (row_index + win_length <= out_dim) and (column_idx + win_length <= out_dim):
            ret_array = np.zeros((out_dim, out_dim), dtype=int)
            ret_array[row_index:row_index+win_length, column_idx:column_idx+win_length] = \
                np.identity(win_length, dtype=int)
            pattern_list += [ret_array.flatten().tolist()]
            pattern_list += [np.fliplr(ret_array).flatten().tolist()]
            pattern_list += [np.flipud(ret_array).flatten().tolist()]

        if (row_index + win_length <= out_dim) and (column_idx + win_length <= out_dim):
            ret_array = np.zeros((out_dim, out_dim), dtype=int)
            ret_array[row_index:row_index+win_length, column_idx:column_idx+win_length] = \
                np.fliplr(np.identity(win_length, dtype=int))
            pattern_list += [ret_array.flatten().tolist()]
            pattern_list += [np.fliplr(ret_array).flatten().tolist()]
            pattern_list += [np.flipud(ret_array).flatten().tolist()]

    logging.debug('Pattern generated at r:{}, c: {} is {}'.format(row_index, column_idx, pattern_list))

    return pattern_list


def generate_winning_pattern(order, win_count):
    return_patterns_list = []
    for row_idx in range(math.ceil(order/2)):
        for col_idx in range(row_idx+1):
            return_patterns_list += generate_patterns_at_pos(row_index=row_idx, 
                                                            column_idx=col_idx, 
                                                            out_dim=order, 
                                                            win_length=win_count)
    return_patterns_list = pd.DataFrame(return_patterns_list).drop_duplicates().values.tolist()
    converted_to_arrays = [np.array(each_list_item, dtype=int).reshape((order, order)) for each_list_item in return_patterns_list]
    return converted_to_arrays


class Board:

    def __init__(self, board_size: int, win_count: int = 3):
        self.board_size = board_size
        self.win_count = win_count
        self.winning_patterns = generate_winning_pattern(self.board_size, self.win_count)
        # self.winning_patterns = WINNING_PATTERNS
        self.board_state = np.zeros((self.board_size, self.board_size), dtype=int)

        self.state_table_file_name = str(self.board_size) + '_' + STATE_TABLE_FILE_NAME
        self.state_table_file_path = os.path.join(os.path.dirname(__file__), self.state_table_file_name)
        self.state_table_file_exists = os.path.exists(self.state_table_file_path)
        if not self.state_table_file_exists:
            qtable = QTable(self.board_size)
        self.board_state_df = pd.read_csv(self.state_table_file_path)

    def reset_board(self):
        self.board_state = np.zeros((self.board_size, self.board_size), dtype=int)

    def get_board_state_id(self):
        found_indices = np.argwhere((self.board_state_df.values == self.board_state.flatten()).all(axis=1)).flatten().tolist()
        assert len(found_indices) == 1, f'Found indices {found_indices} for search pattern {self.board_state.flatten()}'
        return found_indices[0]

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

class Player:
    def __init__(self, player_id, board_obj):
        self.player_id = player_id
        self.my_moves = []
        self.board_obj = board_obj
        self.win_count = 0

    def increment_win_count(self):
        self.win_count = self.win_count + 1

    def make_a_move(self):
        current_state = self.board_obj.get_board_state_id()
        empty_indices = self.board_obj.get_empty_cell_indices()
        assert empty_indices, 'No empty cells to make a move'
        my_selected_move = np.random.choice(np.array(empty_indices))
        logging.debug('Player {} selected: {} in state: {}'.format(self.player_id, my_selected_move, current_state))
        return my_selected_move

class Game:
    def __init__(self, game_id, board_size, number_of_players, win_count=None):
        self.game_id = game_id
        self.num_players = number_of_players
        self.board_size = board_size
        if win_count is None:
            self.num_liner_cells_to_win = self.board_size
        else:
            self.num_liner_cells_to_win = win_count

        self.players = []
        self.board_inst = Board(self.board_size, win_count=self.board_size)
        for each_player_id in range(self.num_players):
            self.players.append(Player(each_player_id+1, self.board_inst))

        self.q_table_df = self.board_inst.board_state_df.copy()
        self.q_table_df['StateID'] = self.q_table_df.index.values
        q_table_only_df = pd.DataFrame(itertools.product(self.board_inst.board_state_df.index.tolist(), range(self.board_size*self.board_size)), columns=['StateID', 'Action'])
        self.q_table_df = pd.merge(self.q_table_df, q_table_only_df, on=['StateID'])
        self.q_table_df['move_validity'] = self.q_table_df.apply(lambda x: self.board_inst.is_valid_move(x.values[:-2],x.values[-1]), axis=1)

        self.q_table_df = self.q_table_df[self.q_table_df['move_validity']]
        self.q_table_df = self.q_table_df.drop('move_validity', axis=1)
 

    def play_one_game(self):
        self.board_inst.reset_board()
        state_action_reward_nstate = [list(), list()]
        current_player = np.random.choice(range(self.num_players))
        game_over = False
        # Initialize result as a tie
        reward_value = GAME_RESULT_TIE_REWARD
        while not game_over:
            current_state = self.board_inst.get_board_state_id()
            sel_row_id, sel_col_id = self.board_inst.mark_a_move(self.players[current_player].player_id, self.players[current_player].make_a_move())
            chosen_action = sel_row_id * self.board_size + sel_col_id
            next_state = self.board_inst.get_board_state_id()
            game_over, winner_id = self.board_inst.is_game_over()
            if game_over:
                if winner_id == 0:
                    logging.debug('Game drawn. Board state: {}'.format(self.board_inst.board_state))
                else:
                    reward_value = GAME_RESULT_WINNER_REWARD
                    self.players[current_player].increment_win_count()
                    logging.debug('Winner is player_id:{}. Board state:{}'.format(self.players[current_player].player_id, self.board_inst.board_state))
            logging.info('Current state: {}, Player: {} selected action: {} gets reward: {} with next state: {}'.format(current_state,self.players[current_player].player_id, chosen_action, reward_value, next_state))
            state_action_reward_nstate[current_player].append([current_state, chosen_action, reward_value, next_state])
            current_player = (current_player + 1) % self.num_players
            if reward_value == GAME_RESULT_WINNER_REWARD:
                state_action_reward_nstate[current_player][-1][SARS_ELEMENT_REWARD_INDEX] = GAME_RESULT_LOSER_REWARD

        logging.info(f'Player 1 SARS: {state_action_reward_nstate[0]}')
        logging.info(f'Player 2 SARS: {state_action_reward_nstate[1]}')

    def play_game_n_time(self, num_games):
        for _ in range(num_games):
            self.play_one_game()

        logging.info('Num Games: {}'.format(num_games))
        logging.info('Win counts: {}'.format(dict(zip([each_player.player_id for each_player in self.players], [each_player.win_count for each_player in self.players]))))



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    game_inst = Game(0, 3, 2, win_count=3)
    game_inst.play_game_n_time(1)

    logging.info('Exiting program')
