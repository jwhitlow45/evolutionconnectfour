import numpy as np
import pygad
import pygad.kerasga
from kaggle_environments import make
import tensorflow as tf
from datetime import datetime
import gc

from model import create_model
from board import isValidMove, countEmptySpaces, BOARD_WIDTH, BOARD_HEIGHT

CUR_MODEL = create_model()
AGENT_PLAYER_NUMBER = 1
NUM_GENERATIONS = 800
POPULATION_SIZE = 50
NUM_PARENTS_MATING = 5

RENDER_FREQ = 100
STAT_FREQ = 1000
SAVE_FREQ = 100

NUM_GAMES = 0
NUM_WINS = 0
NUM_DRAWS = 0

STAT_OUTPUT = ''

def main():
    
    global CUR_MODEL, AGENT_PLAYER_NUMBER, NUM_GENERATIONS, NUM_PARENTS_MATING

    keras_ga = pygad.kerasga.KerasGA(model=CUR_MODEL, num_solutions=POPULATION_SIZE)
    
    init_population = keras_ga.population_weights
    
    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS_MATING,
        initial_population=init_population,
        fitness_func=fitness_function,
        on_generation=callback_generation,
    )
    ga_instance.run()
    
    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    # Fetch the parameters of the best solution.
    best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=CUR_MODEL,
                                                                weights_vector=solution)
    
    CUR_MODEL.set_weights(best_solution_weights)
    tf.keras.models.save_model(CUR_MODEL, filepath='./best.h5')
    
    
def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

def agent(observation, config):
    global CUR_MODEL
    
    board = np.array(observation['board']).reshape(1, BOARD_HEIGHT, BOARD_WIDTH)
    
    # get weighted results from model prediction
    
    weighted_results = [(weight, col) for weight, col in zip(CUR_MODEL.predict(board, verbose=0)[0], range(BOARD_WIDTH))]
    # sort results to traverse in order of move preference
    weighted_results.sort(key = lambda x: x[0], reverse=True)
    
    for _, col in weighted_results:
        if isValidMove(board[0], col):
            return col
    return -1


def fitness_function(solution, sol_idx):
    global CUR_MODEL, AGENT_PLAYER_NUMBER, NUM_GAMES, NUM_WINS, NUM_DRAWS, STAT_OUTPUT, RENDER_FREQ, STAT_FREQ, SAVE_FREQ
    
    timer = datetime.now()
    
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
    
    timer = datetime.now() - timer
    
    NUM_GAMES += 1
    print(f'games:{NUM_GAMES} gametime:{timer}')
    if NUM_GAMES == 0 or NUM_GAMES % RENDER_FREQ == 0 or NUM_GENERATIONS*POPULATION_SIZE == NUM_GAMES:
        output = env.render(mode='html')
        with open(f'./games/game-{NUM_GAMES}.html', 'w') as FILE:
            FILE.write(output)
            
    if NUM_GAMES == 0 or NUM_GAMES % STAT_FREQ == 0 or NUM_GENERATIONS*POPULATION_SIZE == NUM_GAMES:
        with open(f'./games/stats-{NUM_GAMES}.csv', 'w') as FILE:
            FILE.write(STAT_OUTPUT)
            STAT_OUTPUT = ''
            
    if NUM_GAMES == 0 or NUM_GAMES % SAVE_FREQ == 0 or NUM_GENERATIONS*POPULATION_SIZE == NUM_GAMES:
        tf.keras.models.save_model(CUR_MODEL, filepath=f'./models/model-{NUM_GAMES}.h5')
            
    garbage_collection()

    score = countEmptySpaces(end_state.reshape((BOARD_HEIGHT, BOARD_WIDTH)))
    if reward == 1: # our agent won
        NUM_WINS += 1
        STAT_OUTPUT += f'{NUM_WINS},{NUM_GAMES},{NUM_DRAWS},{NUM_WINS/NUM_GAMES}\n'
        return score
    if reward == -1: # opponent won
        STAT_OUTPUT += f'{NUM_WINS},{NUM_GAMES},{NUM_DRAWS},{NUM_WINS/NUM_GAMES}\n'
        return -score
    NUM_DRAWS += 1
    STAT_OUTPUT += f'{NUM_WINS},{NUM_GAMES},{NUM_DRAWS},{NUM_WINS/NUM_GAMES}\n'
    return 0 # draw

def garbage_collection():
    gc.collect()
    tf.keras.backend.clear_session()
    

if __name__ == '__main__':
    main()