import copy
import logging
import math
import random
from typing import Callable

from .individual import Individual
from .experiment_recorder import ExperimentRecorder

class Evolution:
    def __init__(
        self,
        population: list[Individual],
        fitness_fn: Callable[[Individual], float],
        experiment_recorder: ExperimentRecorder,
        crossover_instead_of_mutation_rate: float = 0.5,
        mutation_fns_and_probabilities: list[tuple[Callable[[Individual], Individual], float]] = [],
        crossover_fns_and_probabilities: list[tuple[Callable[[Individual, Individual], Individual], float]] = [],
        target_population_size: bool = 100,
        num_children_per_generation: int = 100,
        visualize_graphs: bool = True,
        **kwargs,
    ):
        """
        Initialize the evolution process
        
        Args:
            population: Initial population of individuals
            fitness_fn: Function that takes an individual and returns its fitness score
            crossover_instead_of_mutation_rate: Probability of crossover occurring instead of mutation
            mutation_fns_and_probabilities: List of mutation functions and their respective probabilities if mutation occurs
            crossover_fns_and_probabilities: List of crossover functions and their respective probabilities if crossover occurs
            target_population_size: Population size to maintain after selection
            num_children_per_generation: Number of children to generate per generation
            visualize_graphs: Whether to visualize graphs during evolution (default: True)
        """
        self.population = population
        self.fitness_fn = fitness_fn
        self.crossover_instead_of_mutation_rate = crossover_instead_of_mutation_rate
        self.mutation_fns_and_probabilities = mutation_fns_and_probabilities
        self.crossover_fns_and_probabilities = crossover_fns_and_probabilities
        self.target_population_size = target_population_size
        self.num_children_per_generation = num_children_per_generation
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_individual = None
        self.id_counter = len(self.population)
        self.visualize_graphs = visualize_graphs
        self.experiment_recorder = experiment_recorder

        self.kwargs = kwargs
        self.kwargs['visualize_graphs'] = visualize_graphs  # Add visualize_graphs to kwargs so it gets passed to crossover/mutation functions

    def run_evolution(self, num_generations: int):
        """
        Run the evolutionary process for specified number of generations
        
        Args:
            num_generations: Number of generations to evolve
        """
        for individual in self.population:
            self.experiment_recorder.record_initial_individual(individual.id, generation=self.generation)

        for individual in self.population:  # evaluate fitness for initial population
            self._evaluate(individual)
        
        self._log_generation()
        for gen in range(1, num_generations):  # first generation was initial_population
            self.generation = gen

            # Create new population through crossover and mutation
            new_children = []
            while len(new_children) < self.num_children_per_generation:
                parent1, parent2 = random.sample(self.population, 2)  # TODO should this sample with or without replacement?
                child = self._copy_individual(parent1)
                # Assign a new unique identifier immediately so downstream operations use the correct id
                child.id = self.id_counter
                self.id_counter += 1
                operations = []
                strategy = "mutation"
                try:
                    if random.random() < self.crossover_instead_of_mutation_rate:
                        strategy = "crossover"
                        child, operations = self._crossover(child, parent2)
                    else:
                        child, operations = self._mutate(child)
                    successful_child = True
                except Exception as e:
                    logging.exception("Error in crossover or mutation")
                    child.fitness = float('-inf')
                    successful_child = False
                logging.info(f"Created child {child.id}")
                new_children.append(child)

                self.experiment_recorder.record_child_creation(
                    individual_id=child.id,
                    generation=self.generation,
                    parents=(parent1.id, parent2.id),
                    operations=operations,
                    strategy=strategy,
                )

                if not successful_child:
                    self._log_individual(child)
                    continue
                self._evaluate(child)

            self.population.extend(new_children)
            
            self._selection()
            self._log_generation()

    def _evaluate(self, individual: Individual):
        individual.evaluation_metrics = None
        if hasattr(individual, "evaluation_error"):
            delattr(individual, "evaluation_error")
        try:
            self._pre_evaluation(individual)
            individual.fitness = self.fitness_fn(individual)
        except Exception as e:
            logging.exception(f"Error in fitness function: {e} for individual {individual.id}")
            individual.fitness = float('-inf')  # Lowest possible fitness since fitness is negative perplexity
            individual.evaluation_error = str(e)
            individual.evaluation_metrics = {"status": "failed"}
            try:
                self._handle_evaluation_error(individual)
            except Exception as e:
                logging.exception(f"Error in handle evaluation error: {e} for individual {individual.id}")
        
        self._log_individual(individual)

    def _pre_evaluation(self, individual: Individual):
        pass

    def _handle_evaluation_error(self, individual: Individual):
        pass

    def _log_individual(self, individual: Individual):
        """Log an individual"""
        logging.info(f"Individual {individual.id} has fitness {individual.fitness}")

    def _copy_individual(self, individual: Individual) -> Individual:
        """
        Copy an individual
        """
        return copy.deepcopy(individual)

    def _crossover(self, child: Individual, parent: Individual) -> tuple[Individual, list[dict]]:
        """
        Perform crossover between two parents
        
        Args:
            child: Child individual
            parent: Parent individual
            
        Returns:
            Child
        """
        logging.info(f"Crossover between {child.id} and {parent.id}")
        applied_operations: list[dict] = []
        for crossover_fn, probability in self.crossover_fns_and_probabilities:
            if random.random() < probability:
                logging.info(f"Crossover between {child.id} and {parent.id} with {crossover_fn.__name__}")
                crossover_fn(child, parent, **self.kwargs)
                applied_operations.append(
                    {
                        "type": "crossover",
                        "name": crossover_fn.__name__,
                        "probability": probability,
                        "with_parent_id": parent.id,
                    }
                )
        return child, applied_operations

    def _mutate(self, child: Individual) -> tuple[Individual, list[dict]]:
        """
        Mutate a single individual
        
        Args:
            child: Child individual
            
        Returns:
            Mutated child individual
        """
        logging.info(f"Mutating {child.id}")
        applied_operations: list[dict] = []
        for mutation_fn, probability in self.mutation_fns_and_probabilities:
            if random.random() < probability:
                logging.info(f"Mutating {child.id} with {mutation_fn.__name__}")
                mutation_fn(child, **self.kwargs)
                applied_operations.append(
                    {
                        "type": "mutation",
                        "name": mutation_fn.__name__,
                        "probability": probability,
                    }
                )
        return child, applied_operations
    
    def _selection(self) -> list[Individual]:
        """Select individuals for breeding based on fitness scores"""
        # Sort population by fitness
        sorted_population = sorted(
            self.population,
            key=lambda individual: (not math.isnan(individual.fitness), individual.fitness),
            reverse=True
        )
        
        self.population = sorted_population[:self.target_population_size]  # Select top performers as parents

    def _log_generation(self):
        """Log the progress of evolution"""
        current_best_fitness_in_gen = float('-inf')
        current_best_individual_in_gen = None
        fitness_sum = 0
        
        for individual in self.population:
            logging.info(f"Individual {individual.id} survived")
            fitness_sum += individual.fitness
            if individual.fitness > current_best_fitness_in_gen:
                current_best_fitness_in_gen = individual.fitness
                current_best_individual_in_gen = individual
        
        avg_fitness = fitness_sum / len(self.population)
        
        if current_best_fitness_in_gen > self.best_fitness:
            self.best_fitness = current_best_fitness_in_gen
            self.best_individual = current_best_individual_in_gen
        
        logging.info(f"Generation {self.generation}:")
        logging.info(f"  Max Fitness in Gen: {current_best_fitness_in_gen:.4f}")
        logging.info(f"  Avg Fitness in Gen: {avg_fitness:.4f}")
        if self.best_individual:
            logging.info(f"  Best Individual Overall (fitness: {self.best_individual.fitness}, id: {self.best_individual.id}): {self.best_individual.train_config}")

        population_snapshot = []
        for individual in self.population:
            fitness_value = individual.fitness
            if fitness_value is None or not math.isfinite(fitness_value):
                fitness_value = None
            population_snapshot.append({"id": individual.id, "fitness": fitness_value})

        max_fitness_value = current_best_fitness_in_gen if math.isfinite(current_best_fitness_in_gen) else None
        avg_fitness_value = avg_fitness if math.isfinite(avg_fitness) else None
        best_overall_id = self.best_individual.id if self.best_individual else None
        best_in_gen_id = current_best_individual_in_gen.id if current_best_individual_in_gen else None

        summary_payload = {
            "population_size": len(self.population),
            "max_fitness": max_fitness_value,
            "average_fitness": avg_fitness_value,
            "best_individual_id": best_in_gen_id,
            "best_overall_individual_id": best_overall_id,
            "population_snapshot": population_snapshot,
        }
        self.experiment_recorder.record_generation(self.generation, summary_payload)
