import itertools
import pandas as pd
import numpy as np
import os
import logging

STATE_TABLE_FILE_NAME = 'all_states_actions.csv'
STATE_TABLE_COLUMN_NAMES_FOR_STATE = ['TopLeft', 'TopMid', 'TopRight', 'MidLeft', 'MidMid', 'MidRight', 'BotLeft', 'BotMid', 'BotRight']
TERMINAL_ACTION = 9999

class QTable:

    def __init__(self, board_order: int):
        self.board_dim = board_order * board_order
        self.state_table_file_name = str(board_order) + '_' + STATE_TABLE_FILE_NAME
        self.state_table_file_path = os.path.join(os.path.dirname(__file__), self.state_table_file_name)
        self.state_table_file_exists = os.path.exists(self.state_table_file_path)


    def is_valid_move(self, flat_board_state, select_cell):
        ret_val = False
        empty_cell_index_list = [each_empty_index for each_empty_index in np.argwhere(flat_board_state==0)]
        if select_cell == TERMINAL_ACTION:
            if len(empty_cell_index_list) == 0:
                ret_val = True
            else:
                ret_val = False
        elif select_cell in empty_cell_index_list:
            ret_val = True
        # if select_cell in [each_empty_index for each_empty_index in np.argwhere(flat_board_state==0)]:
        #     ret_val = True

        return ret_val

    def generate_state_table(self):

        df = pd.DataFrame(list(itertools.product([0, 1, 2], repeat=9)), columns=['TopLeft', 'TopMid', 'TopRight', 'MidLeft', 'MidMid', 'MidRight', 'BotLeft', 'BotMid', 'BotRight'])
        l1 = df.columns.tolist().copy()
        df['num1'] = df.apply(lambda x: (x[l1].values==1).sum(), axis=1)
        df['num2'] = df.apply(lambda x: (x[l1].values==2).sum(), axis=1)
        df['diff'] = abs(df['num1']-df['num2'])
        df = df[df['diff']<=1]
        df = df.drop(['num1', 'num2', 'diff'], axis=1)
        df = df.reset_index().drop('index', axis=1)

        df['StateID'] = df.index.values.copy()
        q_table_only_df = pd.DataFrame(itertools.product(df['StateID'].values.tolist(), list(range(self.board_dim*self.board_dim))+[TERMINAL_ACTION]), columns=['StateID', 'Action'])
        df = pd.merge(df, q_table_only_df, on=['StateID'])
        df['move_validity'] = df.apply(lambda x: self.is_valid_move(x.values[:-2],x.values[-1]), axis=1)

        df = df[df['move_validity']]
        df = df.drop('move_validity', axis=1)

        df = df.reset_index().drop('index', axis=1)

        df.to_csv(self.state_table_file_path, index=False)

        logging.info(f'State table file generated! and has shape: {df.shape}')

    def create_state_action_look_up_table(self):
        if not self.state_table_file_exists:
            logging.info('State table File does not exist')
            self.generate_state_table()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    qtable = QTable(3)
    logging.info('Exiting the program')
