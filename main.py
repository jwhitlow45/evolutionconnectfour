import numpy as np
import pygad
import pygad.kerasga
from kaggle_environments import make

from model import create_model
from board import isValidMove, countEmptySpaces, BOARD_WIDTH, BOARD_HEIGHT

CUR_MODEL = create_model()
AGENT_PLAYER_NUMBER = 2
NUM_GENERATIONS = 1000
NUM_PARENTS_MATING = 10

def main():
    global CUR_MODEL, AGENT_PLAYER_NUMBER, NUM_GENERATIONS, NUM_PARENTS_MATING
    
    keras_ga = pygad.kerasga.KerasGA(model=CUR_MODEL, num_solutions=50)
    
    init_population = keras_ga.population_weights
    
    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS_MATING,
        initial_population=init_population,
        fitness_func=fitness_function,
        on_generation=callback_generation
    )
    
    
def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

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
    global CUR_MODEL, AGENT_PLAYER_NUMBER
    
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