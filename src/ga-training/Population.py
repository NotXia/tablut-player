import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Player import WHITE, BLACK
from Individual import Individual
from Chromosome import Chromosome
from Environment import Environment, BLACK_WIN, WHITE_WIN, DRAW
import numpy as np
import random


class Population:
    def __init__(self, n_individuals:int, initial_weights:dict, color:WHITE|BLACK, timeout:int) -> None:
        self.individuals = []
        self.n_individuals = n_individuals
        self.color = color

        for i in range(n_individuals):
            early_positive = np.array(initial_weights['early']['positive']) + np.random.rand(len(initial_weights['early']['positive']))
            early_negative = np.array(initial_weights['early']['negative']) + np.random.rand(len(initial_weights['early']['negative']))
            mid_positive = np.array(initial_weights['mid']['positive']) + np.random.rand(len(initial_weights['mid']['positive']))
            mid_negative = np.array(initial_weights['mid']['negative']) + np.random.rand(len(initial_weights['mid']['negative']))
            late_positive = np.array(initial_weights['late']['positive']) + np.random.rand(len(initial_weights['late']['positive']))
            late_negative = np.array(initial_weights['late']['negative']) + np.random.rand(len(initial_weights['late']['negative']))
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
    def fight(self, env:Environment, opponent:Individual, logger, epoch) -> int:
        num_wins = 0

        for indiv in self.individuals:
            indiv.play()
            opponent.play()
            print("Starting game engine")
            winner, white_moves, black_moves = env.startGame()
            print(f"{'WHITE WINS' if winner == WHITE_WIN else 'BLACK WINS' if winner == BLACK_WIN else 'DRAW'} | {white_moves} white moves, {black_moves} black moves")
            indiv.fitness = self.fitness(winner, white_moves, black_moves)

            if (winner == WHITE_WIN and self.color == WHITE) or (winner == BLACK_WIN and self.color == BLACK):
                num_wins += 1

            logger.update("whites" if self.color == WHITE else "blacks", self, epoch)

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
        new_individuals = []
        parent1: Individual = self.getBestIndividual()
        for i in range(self.n_individuals):
            parent2: Individual = random.choice(self.individuals)
            new_individuals.append( parent1.crossover(parent2) )

        self.individuals = new_individuals


    """
        Mutates the current population
    """
    def mutations(self, mutation_val, mutation_prob):
        for indiv in self.individuals:
            indiv.mutation(mutation_val, mutation_prob)


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
        out = (
            f"--- Best ---\n{self.getBestIndividual()}\n------------\n"
        )
        for indiv in self.individuals:
            out += f"{indiv}\n\n"
        return out[:-2]