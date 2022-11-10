import numpy as np
import pygad.kerasga
from kaggle_environments import make

from model import create_model
from board import isValidMove, countEmptySpaces, BOARD_WIDTH, BOARD_HEIGHT

def main():
    # create_model()
    # isValidMove(np.zeros((6,7)), 0)
    fitness_function('test','test')
    pass

CUR_MODEL = create_model()
AGENT_PLAYER_NUMBER = 2

def agent(observation, config):
    global CUR_MODEL

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
    global CUR_MODEL
    
    model_weights = pygad.kerasga.model_weights_as_matrix(
        model=CUR_MODEL,
        weights_vector=solution
    )
    
    CUR_MODEL.set_weights(weights=model_weights)
    
    agents = [agent, 'negamax']
    reward_index = 0
    if AGENT_PLAYER_NUMBER == 2:
        agents = agents[::-1]
        reward_index = 1
    
    env = make('connectx', debug=True)
    env.reset()
    
    # run game with agents and get result of game
    game_recap = env.run(agents)
    # get reward and end state from game recap
    reward = game_recap[-1][reward_index]['reward']
    end_state = np.array(game_recap[-1][0]['observation']['board'])
    
    output = env.render(mode='html')
    with open('game.html', 'w') as FILE:
        FILE.write(output)

    score = countEmptySpaces(end_state.reshape((BOARD_HEIGHT, BOARD_WIDTH)))
    if reward == 1: # our agent won
        return score
    if reward == -1: # opponent won
        return -score
    return 0 # draw
        

if __name__ == '__main__':
    main()