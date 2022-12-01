import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score

import SpFixedIS


#################################################
# ##### CLASSIFICATION EXAMPLE  #################
#################################################

df = load_breast_cancer()
x, y = df.data, df.target

# specify a skearn metric to maximise,
# ensuring that the metric is compatible for the problem type

scoring = accuracy_score

# specify a wrapper to use
# ensuring that the wrapper is compatible for the problem type
wrapper = DecisionTreeClassifier()

# set the engine parameters
sp_engine = SpFixedIS.SpFixedIS(x, y, scoring=scoring, wrapper=wrapper)

# set the spFSR engine
# avaliable engine parameters (with default values in parentheses):
# 1. num_instances: number of instances to select - there is no default value
# 2. iter_max (100): maximum number of iterations
# 3. stall_limit (35): when to restart the search (up to iter_max)
# 4. num_grad_avg (10): number of gradient estimates to be averaged for determining search direction.
# 5. num_gain_smoothing (1): number of iterations to smooth the gain values across
# 6. stall_tolerance (1e-8): tolerance for the objective function change in stalling
# 7. print_freq (10): iteration print frequency for the algorithm output
# 8. display_rounding (5): number of digits to display during algorithm execution
# 9. random_state (999): seed for controlling randomness in the execution of the algorithm
# 10. instances_to_keep_indicies (None): A numpy array of the indicies of the instances to keep.
# 11. is_debug (False): whether detailed search information should be displayed each iteration
sp_run = sp_engine.run(num_instances=100, iter_max=50)

# get the results of the run
sp_results = sp_run.results

# list of available keys in the engine output
print('Available keys:', sp_results.keys())

# performance value of the selected instances in predicting the non-selected instances
print('Best value:', sp_results.get('best_value'))

# indices of selected features
print('Indices of selected instances: ', sp_results.get('instances'))

# importance of selected instances
print('Importance of selected instances: ', sp_results.get('importance').round(3))

# number of iterations for the optimal set
print('Total iterations for the optimal instance set:', sp_results.get('total_iter_for_opt'))

print('\n')

#################################################
# ##### REGRESSION EXAMPLE  #####################
#################################################

df = load_boston()

x, y = df.data, df.target

# set the engine parameters
sp_engine = SpFixedIS.SpFixedIS(x, y, scoring=r2_score, wrapper=DecisionTreeRegressor())
sp_run = sp_engine.run(num_instances=100, iter_max=50)
sp_results = sp_run.results

print('Best value:', sp_results.get('best_value'))
print('Indices of selected instances: ', sp_results.get('instances'))
print('Importance of selected instances: ', sp_results.get('importance').round(3))
print('Total iterations for the optimal instance set:', sp_results.get('total_iter_for_opt'))
