from Individual import Individual, BLACK, WHITE
from Chromosome import Chromosome
from Environment import Environment, BLACK_WIN, WHITE_WIN, DRAW
import numpy as np
from utils import softmax



class Population:
    def __init__(self, n_individuals:int, initial_weights:dict, color:WHITE|BLACK, timeout:int) -> None:
        self.individuals = []
        self.n_individuals = n_individuals
        self.color = color

        for i in range(n_individuals):
            early_positive = (np.array(initial_weights['early']['positive']) + np.random.rand(len(initial_weights['early']['positive']))).tolist()
            early_negative = (np.array(initial_weights['early']['negative']) + np.random.rand(len(initial_weights['early']['negative']))).tolist()
            mid_positive = (np.array(initial_weights['mid']['positive']) + np.random.rand(len(initial_weights['mid']['positive']))).tolist()
            mid_negative = (np.array(initial_weights['mid']['negative']) + np.random.rand(len(initial_weights['mid']['negative']))).tolist()
            late_positive = (np.array(initial_weights['late']['positive']) + np.random.rand(len(initial_weights['late']['positive']))).tolist()
            late_negative = (np.array(initial_weights['late']['negative']) + np.random.rand(len(initial_weights['late']['negative']))).tolist()
            self.individuals.append(
                Individual({
                    "early": {
                        "positive" : Chromosome(early_positive),
                        "negative" : Chromosome(early_negative)
                    },
                    "mid": {
                        "positive" : Chromosome(mid_positive),
                        "negative" : Chromosome(mid_negative)
                    },
                    "late": {
                        "positive" : Chromosome(late_positive),
                        "negative" : Chromosome(late_negative)
                    }
                }, color, timeout)
            )

    
    """
        Makes each individual of this population to play against a given opponent.
        The fitness of each individual is updated.
    """
    def fight(self, env:Environment, opponent:Individual, _logger, _epoch, _global_best:Individual) -> int:
        num_wins = 0

        for i, indiv in enumerate(self.individuals):
            indiv.play()
            opponent.play()
            print(f"Starting game engine -- Individual {i}")
            winner, white_moves, black_moves = env.startGame()
            print(f"{'WHITE WINS' if winner == WHITE_WIN else 'BLACK WINS' if winner == BLACK_WIN else 'DRAW'} | {white_moves} white moves, {black_moves} black moves")
            indiv.fitness = self.fitness(winner, white_moves, black_moves)

            if (winner == WHITE_WIN and self.color == WHITE) or (winner == BLACK_WIN and self.color == BLACK):
                num_wins += 1

            _logger.update("whites" if self.color == WHITE else "blacks", _global_best, self, _epoch)

        return num_wins
            

    """
        Computes the fitness score given the results of a game.
    """
    def fitness(self, winner, white_moves:int, black_moves:int):
        # TODO Improve
        def score_f(x):
            return max(1, -0.18*x +10)
        if self.color == BLACK and winner == BLACK_WIN:
            return score_f(black_moves)
        elif self.color == BLACK and winner == WHITE_WIN:
            return -score_f(black_moves)
        elif self.color == WHITE and winner == WHITE_WIN:
            return score_f(white_moves)
        elif self.color == WHITE and winner == BLACK_WIN:
            return -score_f(white_moves)
        else:
            return 0
 

    """
        Creates a new population as the crossover of the current one.
    """
    def crossovers(self):
        # TODO Improve
        new_individuals = [self.getBestIndividual()]
        probabilities = softmax([i.fitness for i in self.individuals])
        
        for _ in range(self.n_individuals-1):
            parents = np.random.choice(self.individuals, p=probabilities, size=2, replace=False)
            new_individuals.append( parents[0].crossover(parents[1]) )

        self.individuals = new_individuals


    """
        Mutates the current population
    """
    def mutations(self, mutation_val, mutation_prob):
        # Skip the first as it is the best one (by crossover definition)
        for i in range(1, len(self.individuals)):
            self.individuals[i].mutation(mutation_val, mutation_prob)


    """
        Returns the individual with the best fitness.
        The first one if fitness is not available.
    """
    def getBestIndividual(self):
        try:
            curr_best = None
            for indiv in self.individuals:
                if curr_best is None or curr_best.fitness < indiv.fitness:
                    curr_best = indiv
            return curr_best
        except:
            return self.individuals[0]
        

    def __str__(self):
        out = ("")
        for i, indiv in enumerate(self.individuals):
            out += f"{i}) {indiv}\n\n"
        return out[:-2]