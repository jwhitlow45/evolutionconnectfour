import numpy as np
from kaggle_environments import make

from model import create_model
from board import isValidMove, BOARD_WIDTH, BOARD_HEIGHT

def main():
    create_model()
    isValidMove(np.zeros((6,7)), 0)
    pass

CUR_MODEL = None

def agent(observation, config):
    board = np.array(observation['board'])
    
    # get weighted results from model prediction
    weighted_results = [(weight, col) for weight, col in zip(CUR_MODEL.predict(board), range(BOARD_WIDTH))]
    # sort results to traverse in order of move preference
    weighted_results.sort(key = lambda x: x[0], reverse=True)
    
    for _, col in weighted_results:
        if isValidMove(board.reshape((BOARD_HEIGHT, BOARD_WIDTH)), col):
            return col
    return -1

def fitness_function(solution, sol_idx):
    env = make('connectx', debug=True)
    

if __name__ == '__main__':
    main()