import random
from itertools import chain

import numpy as np
from deap import algorithms, base, tools
from openmdao.core.driver import Driver, RecordingDebugging

from ..utils import VariableType, listify
from .ga_utils import (
    ConstraintDominatedFitness,
    IndividualBounds,
    ObjectiveValueWithConstraintViolation,
    constraint_violation,
    convert_individual_to_design_vars,
    individual_types_sequence,
    init_population,
    make_discrete_value_mappings,
    mate_disassembled,
    mutate_disassembled,
)


class GenericNsgaDriver(Driver):
    def _declare_options(self):
        self.supports["integer_design_vars"] = True
        # self.supports["ordinal_design_vars"] = True
        # self.supports["nominal_design_vars"] = True
        self.supports["multiple_objectives"] = True

        self.options.declare("generation_count", default=100)
        self.options.declare("population_size", default=None)
        self.options.declare("crossover_prob", default=None)
        self.options.declare("mutation_prob", default=None)
        self.options.declare("random_seed", default=None)
        self.options.declare("verbose", default=False, types=bool)

    def _setup_driver(self, problem):
        super()._setup_driver(problem)

        design_var_meta = self._designvars

        for meta_key, meta_value in design_var_meta.items():
            if "type" not in meta_value:
                if meta_key in self._designvars_discrete:
                    meta_value["type"] = VariableType.INTEGER
                else:
                    meta_value["type"] = VariableType.CONTINUOUS
                meta_value["shape"] = np.shape(meta_value["lower"])

        self.num_objectives = sum(
            obj["size"] for obj in problem.model.get_objectives().values()
        )

        fitness_class = type(
            "Fitness",
            (ConstraintDominatedFitness,),
            {"weights": (-1,) * self.num_objectives},
        )

        individual_types = tuple(individual_types_sequence(design_var_meta))
        individual_bounds = IndividualBounds.from_design_var_meta(design_var_meta)
        discrete_value_mappings = dict(make_discrete_value_mappings(design_var_meta))

        self.individual_size = len(individual_types)

        mate_indpb = 0.9
        mutate_indpb = 1.0 / self.individual_size

        toolbox = base.Toolbox()
        toolbox.register(
            "population",
            init_population,
            individual_types=individual_types,
            individual_bounds=individual_bounds,
            fitness_class=fitness_class,
        )
        toolbox.decorate("population", listify)
        toolbox.register(
            "ind_to_dvs",
            convert_individual_to_design_vars,
            design_var_meta=design_var_meta,
            discrete_value_mappings=discrete_value_mappings,
        )
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register(
            "mate",
            mate_disassembled,
            individual_types=individual_types,
            individual_bounds=individual_bounds,
            int_indpb=mate_indpb,
            ord_indpb=mate_indpb,
            nom_indpb=mate_indpb,
        )
        toolbox.register(
            "mutate",
            mutate_disassembled,
            individual_types=individual_types,
            individual_bounds=individual_bounds,
            cont_indpb=mutate_indpb,
            int_indpb=mutate_indpb,
            ord_indpb=mutate_indpb,
            nom_indpb=mutate_indpb,
        )
        self.toolbox = toolbox

    def _evaluate_individual(self, individual):
        for (name, value) in self.toolbox.ind_to_dvs(individual).items():
            self.set_design_var(name, value)

        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._problem().model.run_solve_nonlinear()

        self.iter_count += 1

        return ObjectiveValueWithConstraintViolation(
            objectives=tuple(
                chain.from_iterable(
                    x.flat for x in self.get_objective_values().values()
                )
            ),
            constraint_violation=constraint_violation(
                values=self.get_constraint_values(),
                meta=self._problem().model.get_constraints(),
            ),
        )

    def evaluate_individual(self, individual):
        key = tuple(individual)

        if key in self.cache:
            return self.cache[key]
        else:
            ret = self._evaluate_individual(individual)
            self.cache[key] = ret
            return ret

    def run(self):
        self.cache = dict()

        random.seed(self.options["random_seed"])
        np.random.seed(self.options["random_seed"])

        start_population = self.toolbox.population(self.options["population_size"])
        population, logbook = nsga_main(
            population=start_population,
            toolbox=self.toolbox,
            mu=self.options["population_size"],
            ngen=self.options["generation_count"],
            verbose=self.options["verbose"],
        )

        return False


class Nsga2Driver(GenericNsgaDriver):
    def _get_name(self):
        return "Nsga2Driver"

    def _setup_driver(self, problem):
        super()._setup_driver(problem)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register(
            "vary",
            nsga2_vary,
            toolbox=self.toolbox,
            cxpb=self.options["crossover_prob"] or 1.0,
            mutpb=self.options["mutation_prob"] or 1.0,
        )

        pop_size = self.options["population_size"]
        if not pop_size:
            pop_size = self.individual_size * self.num_objectives

        # selTournamentDCD has a strict requirement on pop_size % 4 == 0
        pop_size = 4 * (pop_size // 4 + 1)

        self.options["population_size"] = pop_size


class Nsga3Driver(GenericNsgaDriver):
    def _declare_options(self):
        super()._declare_options()
        self.options.declare("reference_partitions", default=4)

    def _get_name(self):
        return "Nsga3Driver"

    def _setup_driver(self, problem):
        super()._setup_driver(problem)

        ref_points = tools.uniform_reference_points(
            self.num_objectives, p=self.options["reference_partitions"]
        )
        nsga3_select = tools.selNSGA3WithMemory(
            ref_points=ref_points,
            # The "standard" non-dominated sort algorithm uses Fitness.dominates,
            # which we have modified for constraint-domination.
            nd="standard",
        )

        self.toolbox.register(
            "vary",
            nsga3_vary,
            toolbox=self.toolbox,
            cxpb=self.options["crossover_prob"] or 1.0,
            mutpb=self.options["mutation_prob"] or 1.0,
        )
        self.toolbox.register("select", nsga3_select)

        if not self.options["population_size"]:
            num_ref_points = len(ref_points)
            # The population size should be the smallest multiple of 4, greater than num_ref_points
            self.options["population_size"] = 4 * (num_ref_points // 4 + 1)


def nsga2_vary(population, toolbox, cxpb, mutpb):
    selected = tools.selTournamentDCD(population, len(population))
    return algorithms.varAnd(selected, toolbox, cxpb, mutpb)


def nsga3_constraint_tournament(individuals, k):
    """Tournament selection based on constraint violation. The *individuals* sequence length has to
    be a multiple of 4. Starting from the beginning of the selected
    individuals, two consecutive individuals will be different (assuming all
    individuals in the input list are unique). Each individual from the input
    list won't be selected more than twice.

    This selection requires the individuals to have a :attr:`constraint_violation`
    attribute.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """

    if len(individuals) % 4 != 0:
        raise ValueError("selTournamentDCD: individuals length must be a multiple of 4")

    if k % 4 != 0:
        raise ValueError(
            "selTournamentDCD: number of individuals to select must be a multiple of 4"
        )

    def tourn(ind1, ind2):
        if ind1.fitness.constraint_violation < ind2.fitness.constraint_violation:
            return ind1
        elif ind1.fitness.constraint_violation > ind2.fitness.constraint_violation:
            return ind2
        elif random.random() <= 0.5:
            return ind1
        else:
            return ind2

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, k, 4):
        chosen.append(tourn(individuals_1[i], individuals_1[i + 1]))
        chosen.append(tourn(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(tourn(individuals_2[i], individuals_2[i + 1]))
        chosen.append(tourn(individuals_2[i + 2], individuals_2[i + 3]))

    return chosen


def nsga3_vary(population, toolbox, cxpb, mutpb):
    selected = nsga3_constraint_tournament(population, len(population))
    return algorithms.varAnd(selected, toolbox, cxpb, mutpb)


def nsga_main(population, toolbox, mu, ngen, halloffame=None, verbose=__debug__):
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("mean", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ("gen", "nevals") + tuple(stats.fields)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # in the case of NSGA-II. No actual selection is done.
    population = toolbox.select(population, len(population))

    if halloffame is not None:
        halloffame.update(population)

    # Compile statistics about the population
    logbook.record(gen=0, nevals=len(invalid_ind), **stats.compile(population))
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = toolbox.vary(population)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population from parents and offspring
        population[:] = toolbox.select(population + offspring, mu)

        # Compile statistics about the new population
        logbook.record(gen=gen, nevals=len(invalid_ind), **stats.compile(population))
        if verbose:
            print(logbook.stream)

    return population, logbook
