import pygad
import numpy as np

def fitness_func(ga_instance, solution, solution_idx):
    output = np.sum(solution * function_inputs)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness

if __name__ == '__main__':
    function_inputs = [4, -2, 3.5, 5, -11, -4.7] # The weights applied to a potential solution.
    desired_output = 44

    fitness_function = fitness_func

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = len(function_inputs) # Use this to control the number of feature selection potential solutions is used.

    init_range_low = -2
    init_range_high = 5

    parent_selection_type = "sss" #steady-state selection, meaning it selects the parents with the highest fitness.
    keep_parents = 1

    crossover_type = "single_point" # Swaps the chromosomes from a certain index onwards between the parents.

    mutation_type = "random"
    mutation_percent_genes = 20
    for num_of_generations in range(10, 51, 10):
        ga_instance = pygad.GA(num_generations=num_of_generations,
                               num_parents_mating=num_parents_mating, # Num of parents to select each generation.
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop, # Number of solutions per population.
                               num_genes=num_genes, # Effectively, the thing that is tweaked for each generation.
                               # gene_type=list[float], # The type of gene, meaning of each value inside a chromosome. Supports list.
                               init_range_low=init_range_low, # dependent on the gene type, the range of values to be generated.
                               init_range_high=init_range_high,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents, # Number of parents to keep from current population.
                               # keep_elitism = 1, # The number of the solutions with the best fitness that will be kept for next generation.
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes, # The probablity that each gene will be mutated
                               # crossover_type=crossover_func, Can be used to customize a crossover func.
                               # mutation_type=mutation_func, Can be used to customize a mutation func.
                               )

        ga_instance.run()
        print('--------------------------------------------------')
        print(f'Generation: {num_of_generations}')
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        prediction = np.sum(np.array(function_inputs) * solution)
        print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))