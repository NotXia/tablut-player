from __future__ import annotations
import random
from utils import normalize



class Chromosome:
    def __init__(self, genes:list[float]) -> None:
        self.genes = normalize(genes)

    """
        Creates a new cromosome as the combination
        of this chromosome and another one.
        The i-th gene of the new chromosome is randomly selected
        between the i-th genes of the parents.
    """
    def crossover(self, ch2:Chromosome) -> Chromosome:
        assert len(self.genes) == len(ch2.genes)
        new_genes = []
        for i in range(len(self.genes)):
            if random.random() < 0.5:
                new_genes.append(self.genes[i])
            else:
                new_genes.append(ch2.genes[i])
        return Chromosome(new_genes)

    """
        Randomly increases/decreases a gene.
    """
    def mutation(self, mutation_val:float):
        to_mutate_index = random.randint(0, len(self.genes)-1)

        self.genes[to_mutate_index] += mutation_val
        for i in range(len(self.genes)):
            if i == to_mutate_index: continue
            self.genes[i] -= mutation_val / (len(self.genes)-1)

        self.genes = normalize(self.genes)

    
    def __str__(self):
        return f"[{', '.join([f'{g:.3f}' for g in self.genes])}]"