import pygad
import numpy as np

# y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6 where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44

def fitness_func(ga_instance, solution, solution_idx):
    output = np.sum(solution*function_inputs)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness

if __name__ == '__main__':
    function_inputs = [4, -2, 3.5, 5, -11, -4.7]
    desired_output = 44

    fitness_function = fitness_func

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = len(function_inputs)

    init_range_low = -2
    init_range_high = 5

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 20

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           save_solutions=True)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = np.sum(np.array(function_inputs) * solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    ga_instance.plot_fitness()
    ga_instance.plot_fitness(plot_type="scatter")
    ga_instance.plot_fitness(plot_type="bar")
    ga_instance.plot_new_solution_rate()
    ga_instance.plot_new_solution_rate(plot_type="scatter")
    ga_instance.plot_new_solution_rate(plot_type="bar")
    ga_instance.plot_genes(graph_type="plot")