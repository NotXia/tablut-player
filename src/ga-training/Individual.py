
from __future__ import annotations
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Player import Player, WHITE, BLACK
import time
import threading
import random

class Individual:
    def __init__(self, chromosomes:dict, color:WHITE|BLACK, timeout:int) -> None:
        self.early_positive = chromosomes['early']['positive']
        self.early_negative = chromosomes['early']['negative']
        self.mid_positive = chromosomes['mid']['positive']
        self.mid_negative = chromosomes['mid']['negative']
        self.late_positive = chromosomes['late']['positive']
        self.late_negative = chromosomes['late']['negative']
        self.color = color
        self.timeout = timeout
        self.fitness = None

    def __startPlayer(self):
        time.sleep(1) # Just give some time for the server to init
        my_color_str = 'white' if self.color == WHITE else 'black'
        weights = {
            "early": {
                "positive": self.early_positive.genes,
                "negative": self.early_negative.genes
            },
            "mid": {
                "positive": self.mid_positive.genes,
                "negative": self.mid_negative.genes
            },
            "late": {
                "positive": self.late_positive.genes,
                "negative": self.late_negative.genes
            }
        }
        try:
            print(f"Starting {my_color_str} player")
            player = Player(my_color_str, weights=weights, timeout=self.timeout)
            player.play()
        except:
            print(f"Cannot start {my_color_str} player")

    """
        Starts the player.
    """
    def play(self):
        threading.Thread(target=self.__startPlayer).start()


    """
        Creates a new individual as the combination of this one and another one.
        The chromosomes of the new individual is the crossover of each chromosome of the parents.
    """
    def crossover(self, indiv:Individual):
        return Individual({
            "early": {
                "positive" : self.early_positive.crossover(indiv.early_positive),
                "negative" : self.early_negative.crossover(indiv.early_positive)
            },
            "mid": {
                "positive" : self.mid_positive.crossover(indiv.mid_positive),
                "negative" : self.mid_negative.crossover(indiv.mid_negative)
            },
            "late": {
                "positive" : self.late_positive.crossover(indiv.late_positive),
                "negative" : self.late_negative.crossover(indiv.late_negative)
            }
        }, self.color, self.timeout)
    

    """
        Randomly mutates the genes of the individual.
    """
    def mutation(self, mutation_val:float, mutation_prob:float):
        def getSign():
            return -1 if random.random() < 0.5 else 1

        if random.random() < mutation_prob: self.early_positive.mutation(getSign() * mutation_val)
        if random.random() < mutation_prob: self.early_negative.mutation(getSign() * mutation_val)
        if random.random() < mutation_prob: self.mid_positive.mutation(getSign() * mutation_val)
        if random.random() < mutation_prob: self.mid_negative.mutation(getSign() * mutation_val)
        if random.random() < mutation_prob: self.late_positive.mutation(getSign() * mutation_val)
        if random.random() < mutation_prob: self.late_negative.mutation(getSign() * mutation_val)
   