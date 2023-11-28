import argparse
from Environment import Environment
from Population import Population, WHITE, BLACK
from Logger import Logger
import json
import os


DEFAULT_WHITES_STARTING_WEIGHTS = {
    "early": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.25, 0.25, 0.25, 0.25]
    },
    "mid": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.25, 0.25, 0.25, 0.25]
    },
    "late": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.25, 0.25, 0.25, 0.25]
    }
}

DEFAULT_BLACKS_STARTING_WEIGHTS = {
    "early": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.25, 0.25, 0.25, 0.25]
    },
    "mid": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.25, 0.25, 0.25, 0.25]
    },
    "late": {
        "positive": [0.25, 0.25, 0.25, 0.25],
        "negative": [0.25, 0.25, 0.25, 0.25]
    }
}


def saveCheckpoint(dir_path, color, epoch, individual):
    try:
        out_dir = os.path.join(dir_path, color)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, f"{color}{epoch+1:03d}.json"), "w") as f:
            json.dump({
                "epoch": epoch+1,
                "fitness": individual.fitness,
                "weights": individual.export()
            }, f, indent=3)
    except:
        print(f"Could not export individual {individual}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Player's parameters training")
    parser.add_argument("-s", "--server-path", type=str, default="../../tablut-server/Tablut", help="Path to the Tablut server")
    parser.add_argument("-g", "--gui", action="store_true", default=False, help="Enable board GUI")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("-i", "--indivs", type=int, required=True, help="Number of individuals per population")
    parser.add_argument("-t", "--timeout", type=int, default=15, help="Time available for an individual to make a decision")
    parser.add_argument("-o", "--output", type=str, required=True, help="Directory where the output weights will be saved")
    parser.add_argument("--history", type=str, required=False, help="File where the population history will be saved")
    parser.add_argument("--mutation-value", type=float, default=0.1, help="Value for chromosomes mutation")
    parser.add_argument("--mutation-prob", type=float, required=True, help="Probability of a mutation")
    parser.add_argument("--whites-log", type=str, default="./whites.log", help="Log file for whites")
    parser.add_argument("--blacks-log", type=str, default="./blacks.log", help="Log file for blacks")
    args = parser.parse_args()


    env = Environment(args.server_path, gui=args.gui)
    white_population = Population(args.indivs, DEFAULT_WHITES_STARTING_WEIGHTS, WHITE, args.timeout)
    black_population = Population(args.indivs, DEFAULT_BLACKS_STARTING_WEIGHTS, BLACK, args.timeout)
    who_is_training = WHITE
    logger = Logger(args.whites_log, args.blacks_log)
    mutation_prob_whites = args.mutation_prob
    mutation_prob_blacks = args.mutation_prob

    logger.update("whites", white_population, 0)
    logger.update("blacks", black_population, 0)

    for epoch in range(args.epochs):
        print(f"<<<<<<<<<< Epoch {epoch+1} -- training {'whites' if who_is_training == WHITE else 'blacks'} >>>>>>>>>>")

        if who_is_training == WHITE:
            print(f"Mutation probability: {mutation_prob_whites}")
            logger.update("whites", white_population, epoch+1)
            
            opponent = black_population.getBestIndividual()
            num_wins = white_population.fight(env, opponent, logger, epoch+1)

            logger.update("whites", white_population, epoch+1)
            saveCheckpoint(args.output, "whites", epoch, white_population.getBestIndividual())
            if args.history is not None: logger.saveHistory(args.history, "whites", white_population, epoch+1)

            white_population.crossovers()
            white_population.mutations(args.mutation_value, mutation_prob_whites)
            mutation_prob_whites = max(0.1, mutation_prob_whites - (mutation_prob_whites/args.epochs))

            if num_wins >= int(args.indivs / 2):
                who_is_training = BLACK
        else:
            print(f"Mutation probability: {mutation_prob_blacks}")
            logger.update("blacks", black_population, epoch+1)
            
            opponent = white_population.getBestIndividual()
            num_wins = black_population.fight(env, opponent, logger, epoch+1)

            logger.update("blacks", black_population, epoch+1)
            saveCheckpoint(args.output, "blacks", epoch, black_population.getBestIndividual())
            if args.history is not None: logger.saveHistory(args.history, "blacks", black_population, epoch+1)

            black_population.crossovers()
            black_population.mutations(args.mutation_value, mutation_prob_blacks)
            mutation_prob_blacks = max(0.1, mutation_prob_blacks - (mutation_prob_blacks/args.epochs))

            if num_wins >= int(args.indivs / 2):
                who_is_training = WHITE
