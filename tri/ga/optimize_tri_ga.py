import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
import os
import random
import importlib
from write_model_ga import *
import main_tri_repo_prof_ga
import model_custom #import the module here, so that it can be reloaded.
import matplotlib.pyplot as plt

seed = 60
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)

def optimize_tri_func(custom_list):
    #print("optimize_raw: custom_list")
    #print(custom_list)
    print("optimize: custom_list")
    #custom_list = list(set(custom_list))
    #custom_list.sort()
    print(custom_list)
    write_file(custom_list)
    importlib.reload(model_custom)
    score = main_tri_repo_prof_ga.main(custom_list)
    print("score")
    print(score)
    return score
    
varbound=np.array([[0,4]]*25)
algorithm_param = {'max_num_iteration': 3,\
                   'population_size': 100,\
                   'mutation_probability':0.5,\
                   'elit_ratio': 0.1,\
                   'parents_portion': 0.2,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=optimize_tri_func,dimension=25,variable_type='int',variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model.run(seed=60, no_plot = True)
re = np.array(model.report)
plt.plot(re)
plt.xlabel('Iteration')
plt.title('Genetic Algorithm')
plt.savefig('generation_scores.png')
best_sol = model.result.variable
f = open('ga_result.txt', 'w')
f.write('The best solution found is: \n')
f.write(str(best_sol))
f.close()