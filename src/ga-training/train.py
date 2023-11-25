import argparse
from Environment import Environment
from Population import Population, WHITE, BLACK
from Logger import Logger
import json
from copy import deepcopy


DEFAULT_WHITES_STARTING_WEIGHTS = {
    "early": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.33, 0.33, 0.33]
    },
    "mid": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.33, 0.33, 0.33]
    },
    "late": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.33, 0.33, 0.33]
    }
}

DEFAULT_BLACKS_STARTING_WEIGHTS = {
    "early": {
        "positive": [0.33, 0.33, 0.33],
        "negative": [0.25, 0.25, 0.25, 0.25]
    },
    "mid": {
        "positive": [0.33, 0.33, 0.33],
        "negative": [0.25, 0.25, 0.25, 0.25]
    },
    "late": {
        "positive": [0.33, 0.33, 0.33],
        "negative": [0.25, 0.25, 0.25, 0.25]
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Player's parameters training")
    parser.add_argument("-s", "--server-path", type=str, default="../../tablut-server/Tablut", help="Path to the Tablut server")
    parser.add_argument("-g", "--gui", action="store_true", default=False, help="Enable board GUI")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("-i", "--indivs", type=int, required=True, help="Number of individuals per population")
    parser.add_argument("-t", "--timeout", type=int, default=15, help="Time available for an individual to make a decision")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output weights")
    parser.add_argument("--mutation-value", type=float, default=0.1, help="Value for chromosomes mutation")
    parser.add_argument("--mutation-prob", type=float, required=True, help="Probability of a mutation")
    parser.add_argument("--whites-log", type=str, default="./whites.log", help="Log file for whites")
    parser.add_argument("--blacks-log", type=str, default="./blacks.log", help="Log file for blacks")
    args = parser.parse_args()

    # TODO Mutation parameters

    env = Environment(args.server_path, gui=args.gui)
    white_population = Population(args.indivs, DEFAULT_WHITES_STARTING_WEIGHTS, WHITE, args.timeout)
    black_population = Population(args.indivs, DEFAULT_BLACKS_STARTING_WEIGHTS, BLACK, args.timeout)
    who_is_training = WHITE
    logger = Logger(args.whites_log, args.blacks_log)
    curr_best_white, curr_best_black = None, None
    mutation_prob_whites = args.mutation_prob
    mutation_prob_blacks = args.mutation_prob

    logger.update("whites", curr_best_white, white_population, 0)
    logger.update("blacks", curr_best_black, black_population, 0)

    for epoch in range(args.epochs):
        print(f"<<<<<<<<<< Epoch {epoch+1} -- training {'whites' if who_is_training == WHITE else 'blacks'} >>>>>>>>>>")

        if who_is_training == WHITE:
            print(f"Mutation probability: {mutation_prob_whites}")
            logger.update("whites", curr_best_white, white_population, epoch+1)
            
            opponent = black_population.getBestIndividual() if curr_best_black is None else curr_best_black
            num_wins = white_population.fight(env, opponent, logger, epoch+1, curr_best_white)

            epoch_best = white_population.getBestIndividual()
            if (curr_best_white is None) or (epoch_best.fitness > curr_best_white.fitness):
                curr_best_white = deepcopy(epoch_best)
                logger.update("whites", curr_best_white, white_population, epoch+1)

            white_population.crossovers()
            white_population.mutations(args.mutation_value, mutation_prob_whites)
            mutation_prob_whites = max(0.1, mutation_prob_whites - (mutation_prob_whites/args.epochs))

            if num_wins >= int(args.indivs / 2):
                who_is_training = BLACK
        else:
            print(f"Mutation probability: {mutation_prob_blacks}")
            logger.update("blacks", curr_best_black, black_population, epoch+1)
            
            opponent = white_population.getBestIndividual() if curr_best_white is None else curr_best_white
            num_wins = black_population.fight(env, opponent, logger, epoch+1, curr_best_black)

            epoch_best = black_population.getBestIndividual()
            if (curr_best_black is None) or (epoch_best.fitness > curr_best_black.fitness):
                curr_best_black = deepcopy(epoch_best)
                logger.update("blacks", curr_best_black, black_population, epoch+1)

            black_population.crossovers()
            black_population.mutations(args.mutation_value, mutation_prob_blacks)
            mutation_prob_blacks = max(0.1, mutation_prob_blacks - (mutation_prob_blacks/args.epochs))

            if num_wins >= int(args.indivs / 2):
                who_is_training = WHITE

        
        with open(args.output, "w") as f:
            json.dump({
                "epoch": epoch+1,
                "white_fitness": None if curr_best_white is None else curr_best_white.fitness,
                "black_fitness": None if curr_best_black is None else curr_best_black.fitness,
                "white": None if curr_best_white is None else curr_best_white.export(),
                "black": None if curr_best_black is None else curr_best_black.export(),
            }, f, indent=3)
