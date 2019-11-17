import itertools
import pandas as pd
import os
import logging

STATE_TABLE_FILE_NAME = 'all_states.csv'

class QTable:

    def __init__(self, board_order: int):
        self.board_dim = board_order * board_order
        self.state_table_file_name = str(board_order) + '_' + STATE_TABLE_FILE_NAME
        self.state_table_file_path = os.path.join(os.path.dirname(__file__), self.state_table_file_name)
        self.state_table_file_exists = os.path.exists(self.state_table_file_path)

        if not self.state_table_file_exists:
            logging.info('State table File does not exist')
            self.generate_state_table()

    def generate_state_table(self):

        df = pd.DataFrame(list(itertools.product([0, 1, 2], repeat=9)), columns=['TopLeft', 'TopMid', 'TopRight', 'MidLeft', 'MidMid', 'MidRight', 'BotLeft', 'BotMid', 'BotRight'])
        l1 = df.columns.tolist().copy()
        df['num1'] = df.apply(lambda x: (x[l1].values==1).sum(), axis=1)
        df['num2'] = df.apply(lambda x: (x[l1].values==2).sum(), axis=1)
        df['diff'] = abs(df['num1']-df['num2'])
        df = df[df['diff']<=1]
        df = df.drop(['num1', 'num2', 'diff'], axis=1)
        df = df.reset_index().drop('index', axis=1)

        df.to_csv(self.state_table_file_path, index=False)
        logging.info(f'State table file generated! and has shape: {df.shape}')

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    qtable = QTable(3)
    logging.info('Exiting the program')
