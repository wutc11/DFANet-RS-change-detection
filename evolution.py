import random
import pickle
from deap import tools


def varAnd(population, toolbox, cxpb, mutpb):
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb and offspring[i - 1].skill_factor[0] == offspring[i].skill_factor[0]:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness1.values, offspring[i].fitness1.values
            del offspring[i - 1].fitness2.values, offspring[i].fitness2.values
            del offspring[i - 1].fitness3.values, offspring[i].fitness3.values
            del offspring[i - 1].fitness4.values, offspring[i].fitness4.values

        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            offspring[i-1], = toolbox.mutate(offspring[i-1])
            del offspring[i - 1].fitness1.values, offspring[i].fitness1.values
            del offspring[i - 1].fitness2.values, offspring[i].fitness2.values
            del offspring[i - 1].fitness3.values, offspring[i].fitness3.values
            del offspring[i - 1].fitness4.values, offspring[i].fitness4.values

    return offspring

def eaSimple(population, toolbox, cxpb, mutpb, ngen, task_num, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # for i in range(1,task_num+1):
        # Evaluate the individuals with an invalid fitness
        #invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate(), invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit
    print("\n")
    print("-"*10, "初始种群计算fitness中", "-"*10)
    print("\n")
    print("一共", len(population), "个个体")
    for i, individual in enumerate(population):
        print("\n")
        print("-"*10,"第", i, "个个体", "-"*10)
        individual.fitness1.values, cd1 = toolbox.evaluate(individual, 1)
        individual.fitness2.values, cd2 = toolbox.evaluate(individual, 2)
        individual.fitness3.values, cd3 = toolbox.evaluate(individual, 3)
        individual.fitness4.values, cd4 = toolbox.evaluate(individual, 4)

    inpopulation = population

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        print("\n")
        print("-"*10, "第", gen, "代进化进行中", "-"*10)
        print("\n")

        fit1list = []
        fit2list = []
        fit3list = []
        fit4list = []
        for individual in inpopulation:
            individual.factorial_ranks = []
            fit1list.append(individual.fitness1.values)
            fit2list.append(individual.fitness2.values)
            fit3list.append(individual.fitness3.values)
            fit4list.append(individual.fitness4.values)

        sorted_id1 = []
        sorted_id1 = sorted(range(len(fit1list)), key=lambda k: fit1list[k], reverse=True)
        sorted_id2 = []
        sorted_id2 = sorted(range(len(fit2list)), key=lambda k: fit2list[k], reverse=True)
        sorted_id3 = []
        sorted_id3 = sorted(range(len(fit3list)), key=lambda k: fit3list[k], reverse=True)
        sorted_id4 = []
        sorted_id4 = sorted(range(len(fit4list)), key=lambda k: fit4list[k], reverse=True)

        pop = 1
        for i in range(len(inpopulation)):
            inpopulation[sorted_id1[i]].factorial_ranks.append(pop)
            pop += 1

        pop = 1
        for i in range(len(inpopulation)):
            inpopulation[sorted_id2[i]].factorial_ranks.append(pop)
            pop += 1

        pop = 1
        for i in range(len(inpopulation)):
            inpopulation[sorted_id3[i]].factorial_ranks.append(pop)
            pop += 1

        pop = 1
        for i in range(len(inpopulation)):
            inpopulation[sorted_id4[i]].factorial_ranks.append(pop)
            pop += 1

        # 分配scalar_factor
        # 基于factorial_ranks, 判断个体更偏向处理哪一个任务
        # 将不擅长的任务的cost置为无穷大
        for individual in inpopulation:
            xxx = min(individual.factorial_ranks)
            yyy = [i+1 for (i,j) in enumerate(individual.factorial_ranks) if j==xxx]
            individual.scalar_fitness.values = (1 / xxx),
            if len(yyy) > 1:
                individual.skill_factor = random.sample(yyy, 1)
                task = int(individual.skill_factor[0])
                if task == 1:
                    individual.fitness.values = individual.fitness1.values
                    individual.fitness2.values = 0,
                    individual.fitness3.values = 0,
                    individual.fitness4.values = 0,
                elif task == 2:
                    individual.fitness.values = individual.fitness2.values
                    individual.fitness1.values = 0,
                    individual.fitness3.values = 0,
                    individual.fitness4.values = 0,
                elif task == 3:
                    individual.fitness.values = individual.fitness3.values
                    individual.fitness1.values = 0,
                    individual.fitness2.values = 0,
                    individual.fitness4.values = 0,
                elif task == 4:
                    individual.fitness.values = individual.fitness4.values
                    individual.fitness1.values = 0,
                    individual.fitness3.values = 0,
                    individual.fitness2.values = 0,
                else:
                    print('task is wrong', task)

            else:
                individual.skill_factor = yyy
                if yyy[0] == 1:
                    individual.fitness.values = individual.fitness1.values
                    individual.fitness2.values = 0,
                    individual.fitness3.values = 0,
                    individual.fitness4.values = 0,
                elif yyy[0] == 2:
                    individual.fitness.values = individual.fitness2.values
                    individual.fitness1.values = 0,
                    individual.fitness3.values = 0,
                    individual.fitness4.values = 0,
                elif yyy[0] == 3:
                    individual.fitness.values = individual.fitness3.values
                    individual.fitness1.values = 0,
                    individual.fitness2.values = 0,
                    individual.fitness4.values = 0,
                elif yyy[0] == 4:
                    individual.fitness.values = individual.fitness4.values
                    individual.fitness1.values = 0,
                    individual.fitness3.values = 0,
                    individual.fitness2.values = 0,
                else:
                    print('yyy is wrong', yyy)


        # Select the next generation individuals
        offspring = toolbox.select(inpopulation, len(population),fit_attr="scalar_fitness")

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness1.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit
        
        if len(invalid_ind)>0:
            for i, individual in enumerate(invalid_ind):
                print("\n")
                print("-"*10,"第", i, "个后代", "-"*10)
                individual.fitness1.values, _ = toolbox.evaluate(individual, 1)
                individual.fitness2.values, _ = toolbox.evaluate(individual, 2)
                individual.fitness3.values, _ = toolbox.evaluate(individual, 3)
                individual.fitness4.values, _ = toolbox.evaluate(individual, 4)
        # print("\n"+"cd11:",cd1)
        # print("\n"+"cd22:",cd2)
        # print("\n"+"cd33:",cd3)
        # print("\n"+"cd44:",cd4)


        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        inpopulation = inpopulation + offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        inpopulation[:] = toolbox.select(inpopulation, len(population),fit_attr="scalar_fitness")

    for individual in inpopulation:
        print("individual:",individual,"\n")
        print("individual fitness:", individual.fitness.values, "\n")
        print("individual fitness1:", individual.fitness1.values, "\n")
        print("individual fitness2:", individual.fitness2.values, "\n")
        print("individual fitness3:", individual.fitness3.values, "\n")
        print("individual fitness4:", individual.fitness4.values, "\n")
        print("individual factorial ranks:", individual.factorial_ranks, "\n")
        print("individual skill factor:", individual.skill_factor, "\n")

    # with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump(population, f)
    return inpopulation, logbook, cd1, cd2, cd3, cd4