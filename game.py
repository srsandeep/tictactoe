import numpy as np
import math
import pandas as pd
import os
import itertools
import logging
from board import Board
from player import Player

SARS_ELEMENT_REWARD_INDEX = 2
GAME_RESULT_WINNER_REWARD = 1
GAME_RESULT_LOSER_REWARD = -1
GAME_RESULT_TIE_REWARD = 0

class Game:
    def __init__(self, board_size, number_of_players, win_count=None):
        self.head_2_head_column_names = ['gameid', 'winnerid', 'player1score', 'player2score']
        self.num_players = number_of_players
        self.board_size = board_size
        if win_count is None:
            self.num_liner_cells_to_win = self.board_size
        else:
            self.num_liner_cells_to_win = win_count

        self.players = []
        self.board_inst = Board(self.board_size, win_count=self.board_size)

        self.q_table_df = self.board_inst.board_state_df.copy()
        self.head_2_head_stats_df = pd.DataFrame([], columns=self.head_2_head_column_names)
 
    def register_players(self, player_list):
        assert self.num_players == len(player_list), 'Not enough players for a game being registered'
        for each_player_id in range(self.num_players):
            player_list[each_player_id].assign_board(self.board_inst)
            self.players.append(player_list[each_player_id])

    def play_one_game(self, gameid=1):
        self.board_inst.reset_board()
        state_action_reward_nstate = [list(), list()]
        current_player = np.random.choice(range(self.num_players))
        game_over = False
        first_move = True
        # Initialize result as a tie
        reward_value = GAME_RESULT_TIE_REWARD
        while not game_over:
            if not first_move:
                self.players[current_player].update_sars_info()
                current_player = (current_player + 1) % self.num_players
            else:
                first_move = False
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
            logging.debug('CurState: {}, Player: {} Action: {} reward: {} NextState: {}'.format(current_state,self.players[current_player].player_id, chosen_action, reward_value, next_state))
            state_action_reward_nstate[current_player].append([current_state, chosen_action, reward_value, next_state])

            self.players[current_player].update_sas_info(current_state, chosen_action, next_state)
            self.players[current_player].update_reward_info(reward_value)

            if reward_value == GAME_RESULT_WINNER_REWARD:
                state_action_reward_nstate[(current_player + 1) % self.num_players][-1][SARS_ELEMENT_REWARD_INDEX] = GAME_RESULT_LOSER_REWARD
            if game_over:
                self.players[current_player].update_sars_info()
                self.head_2_head_stats_df = pd.concat([self.head_2_head_stats_df, pd.DataFrame([[gameid,winner_id,self.players[0].get_win_count(),self.players[1].get_win_count()]], columns=self.head_2_head_column_names)], ignore_index=True)
                other_player_reward = GAME_RESULT_LOSER_REWARD if reward_value==GAME_RESULT_WINNER_REWARD else GAME_RESULT_TIE_REWARD
                self.players[(current_player + 1) % self.num_players].update_reward_info(other_player_reward)
                self.players[(current_player + 1) % self.num_players].update_sars_info()


        logging.debug(f'Player 1 SARS: {state_action_reward_nstate[0]}')
        logging.debug(f'Player 2 SARS: {state_action_reward_nstate[1]}')
        logging.info(f'Head-2-Head stats: {self.head_2_head_stats_df.tail(1).values}')

    def play_game_n_time(self, num_games):
        assert len(self.players) == self.num_players, 'Not enough players for the game'
        for i in range(num_games):
            self.play_one_game(gameid=i)
        self.players[0].q_table.to_csv('player1_q_table.csv', index=False)
        self.players[1].q_table.to_csv('player2_q_table.csv', index=False)

        logging.info('Num Games: {}'.format(num_games))
        logging.info('Win counts: {}'.format(dict(zip([each_player.player_id for each_player in self.players], [each_player.win_count for each_player in self.players]))))
