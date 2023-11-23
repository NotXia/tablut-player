import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
from Environment import Environment
from Population import Population
from Player import WHITE, BLACK


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
    args = parser.parse_args()

    # TODO Mutation parameters

    env = Environment(args.server_path, gui=args.gui)
    white_population = Population(args.indivs, DEFAULT_WHITES_STARTING_WEIGHTS, WHITE, args.timeout)
    black_population = Population(args.indivs, DEFAULT_BLACKS_STARTING_WEIGHTS, BLACK, args.timeout)
    who_is_training = WHITE

    for _ in range(args.epochs):
        if who_is_training == WHITE:
            opponent = black_population.getBestIndividual()
            num_wins = white_population.fight(env, opponent)
            white_population.crossovers()
            white_population.mutations(0.1, 0.2)

            if num_wins >= int(args.nindivs / 2):
                who_is_training = BLACK
        else:
            opponent = white_population.getBestIndividual()
            num_wins = black_population.fight(env, opponent)
            black_population.crossovers()
            black_population.mutations(0.1, 0.2)

            if num_wins >= int(args.nindivs / 2):
                who_is_training = WHITE