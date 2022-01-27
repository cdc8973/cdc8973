# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:42:49 2021

@author: Cesar Diaz 
cdiazcarav@miners.utep.edu
cesar.dc1509@gmail.com

The University of Texas at El Paso

Code created as part of the project "Machine Learning the Ground State and Free
Energies of Iron-Vanadium Alloys via Cluster Expansions" during the Summer of 
2021, under the mentorship of Prof. Jorge Munoz and funded by the Campus Office
of Undergraduate Research Initiatives. 

The code takes a matrix with the correlation functions obtained for a set of 
Iron-Vanadium Compositions, and a matrix with the energies calculated for each
compositions. Then, it calculates the basis vectors, calculates a ground-state
energy prediction for all random compositions between pure Fe and pure V,
and returns a plot and the data.

After the ground-state energy calculations are compleated, the code uses the 
values to calculate the free energy of all the random compositions, and then 
determines at what temperature the phase transition between ordered and random
structure occurs for a certain specified ordered compositions. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math 


total_atoms = 2000

# Boltzmann constant in eV/K
kb = 8.617333262145e-5

energies = np.array([[-8579.997851], #0
                     [-8954.275246], #0.125 ord
                     [-8935.656182], #0.125 ran
                     [-9313.921962], #0.25 ord
                     [-9249.100596], #0.25 ran
                     [-9523.932546], #0.375
                     [-9779.418502], #0.5 ord
                     [-9767.379430], #0.5 ran
                     [-9983.912115], #0.625
                     [-10230.54084], #0.75 ord
                     [-10182.50037], #0.75 ran
                     [-10387.44556], #0.875 ord
                     [-10380.82660], #0.875 ran
                     [-10599.98662]]) #1

energies_model_test = np.array([[-8579.997851],       #0
                                [-8935.656182],       #0.125 ran
                                [-9249.100596],       #0.25
                                [-9523.932546],       #0.375
                                [-9767.379430],       #0.5
                                [-9983.912115],       #0.625
                                [-10182.50037],       #0.75
                                [-10380.82660],       #0.875
                                [-10599.98662]])      #1
# All energies in energies_model_test array correspond to those found under 
# random distribution

energies_ordered = np.array([[-8579.997851], #0
                             [-8954.275246], #0.125 ord
                             [-9313.921962], #0.25 ord
                             [-9779.418502], #0.5 ord
                             [-10230.54084], #0.75 ord
                             [-10387.44556], #0.875 ord
                             [-10599.98662]]) #1

#%% Ordered energies do not change with temperature because entropy is always 
# zero for ordered configurations

compositions_ordered = np.array([0, 0.125, 0.25, 0.5, 0.75, 0.875, 1])

correlation_functions = np.array([[1, 1, 1, 1],                 #0
                                  [1, 0.75, 0.5, 1],            #0.125 ord
                                  [1, 0.75, 0.5625, 0.5625],    #0.125 ran
                                  [1, 0.5, 0, 1],               #0.25 ord
                                  [1, 0.5, 0.25, 0.25],         #0.25 ran
                                  [1, 0.25, 0.0625, 0.0625],    #0.375
                                  [1, 0, -1, 1],                #0.5 ord
                                  [1, 0, 0, 0],                 #0.5 ran
                                  [1, -0.25, 0.0625, 0.0625],   #0.625
                                  [1, -0.5, 0, 1],              #0.75 ord
                                  [1, -0.5, 0.25, 0.25],        #0.75 ran
                                  [1, -0.75, 0.5, 1],           #0.875 ord
                                  [1, -0.75, 0.5625, 0.5625],   #0.875 ran
                                  [1, -1, 1, 1]])               #1

energies_ordered_per_atom = energies_ordered / total_atoms

print(energies_ordered_per_atom)

j0 = 0
j1 = 0
j2 = 0
j3 = 0

basis_vectors = np.array([[j0], 
                         [j1],
                         [j2], 
                         [j3]])

#%% Calculating correlation functions matrix pseudo-inverse and solving for 
# basis vectors
correlation_functions_inv = np.linalg.pinv(correlation_functions)
basis_vectors = np.matmul(correlation_functions_inv, energies)

#%% Creating list with all testing compositions appended
compositions = []
start_value = 0

while start_value <= 1:
    compositions.append(round(start_value, 3))
    start_value += 0.001

compositions.append(1)

#%% Calculating correlation functions with random formula for arbitrary 
# compositions
def cf_random(composition, n):
    return (1 - 2*composition) ** n


cf_array = np.empty((0, 4), int)

for composition in compositions:
    
    point_cf = cf_random(composition, 1)
    pair_cf = cf_random(composition, 2)
    
    cf_array_stack = np.array([1, point_cf, pair_cf, pair_cf])
    cf_array = np.vstack([cf_array, cf_array_stack])
    
#%% Converting the compositions into a single column array for stacking (hstack)
compositions_array = np.array(compositions)
compositions_array = np.reshape(compositions_array, (len(compositions_array), 1))

#%% Calculating predictions of the cluster expansions model 
predictions_array = np.empty((0, 1), float)

for cf in cf_array:
   
    prediction = np.matmul(cf, basis_vectors)
    predictions_array = np.vstack([predictions_array, prediction])

#%% Creating an array with correlation functions and the compositions as index
cf_array = np.hstack([compositions_array, cf_array])

#df_cf_array = pd.DataFrame(cf_array)
#df_cf_array.to_csv('Correlation functions.csv')

#%% Creating an array with the predictions and the compositions as index
predictions_array = np.hstack([compositions_array, predictions_array])

#%% Calculating energy per atom in predictions array
predictions_per_atom = predictions_array[:, 1] / total_atoms

predictions_per_atom_hstack = np.empty((0, 1), float)

for energy in predictions_per_atom:
    predictions_per_atom_hstack = np.vstack([predictions_per_atom_hstack, 
                                             energy])

predictions_array = np.hstack([predictions_array, predictions_per_atom_hstack])

#%% Selecting the compositions we trained on for comparation
predictions_train = predictions_array[:, 1]
predictions_train = predictions_train[np.array([0, 125, 250, 375, 500, 625, 750, 875, 
                                          1000])]

predictions_train_hstack = np.empty((0, 1), float)

for energy in predictions_train:
    predictions_train_hstack = np.vstack([predictions_train_hstack, energy])

comparation_array = np.hstack([energies_model_test, predictions_train_hstack])

df_comparation_train = pd.DataFrame(comparation_array)
df_comparation_train.to_csv('Comparation array.csv')


#%% Entropy calculations (calculated with aproximation from Kittel formula 80)
def entropy_fundamental(composition):
    return -(((1 - composition) * math.log(1 - composition)) + 
            (composition * math.log(composition)))


fundamental_entropies = np.empty((0, 1), float)

for composition in compositions:
    
    if composition == 0 or composition == 1:
        entropy = 0
    else:
        entropy = entropy_fundamental(composition)

    fundamental_entropies = np.vstack([fundamental_entropies, entropy])

#%% Calculating entropies in SI units (multiplying by boltzmann constant)
entropies = fundamental_entropies * kb

#%% Apending the temperatures to calculate the entropy times temperature 
temperatures = []
t = 0

while t <= 1500:
    temperatures.append(t)
    t += 100

#%% Calculating Helmholtz free energy
free_energies = np.empty((0, 1), float)

i = 0

free_energies_all_temp = np.empty((1001, 0), float)

for temperature in temperatures:

    for internal_energy in predictions_array[:, 2]:
    
        free_energy = internal_energy - (entropies[i] * temperature)
        free_energies = np.vstack([free_energies, free_energy])
        
        i += 1
    
    free_energies_all_temp = np.hstack([free_energies_all_temp, free_energies])
    free_energies = np.empty((0, 1), float)
    
    i = 0    

predictions_array = np.hstack([predictions_array, fundamental_entropies, 
                               entropies])

df_predictions_array = pd.DataFrame(predictions_array, columns = ('Composition',
                                                                  'Total Energy (eV)',
                                                                  'Energy per atom (eV per atom)',
                                                                  'Fundamental Entropy (ln Ω)',
                                                                  'Entropy per atom (eV/K per atom)'))
df_predictions_array.to_csv('Predictions array for all compositions.csv')

df_free_energies_all_temp = pd.DataFrame(free_energies_all_temp,
                                         index = compositions,
                                         columns = temperatures)
df_free_energies_all_temp.to_csv('Free energies for specified temperatures.csv')

#%% Defining features for annotation with temperature in the graph (matplotlib stuff)
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5)

#%% Plotting a single energy per atom vs lattice parameter graph
plt.style.use('default')

plt.figure(figsize=(6, 4))
ax_plot = plt.subplot()
plt.plot(compositions, free_energies_all_temp[:, 0], label='Random Compositions Energies', 
         color = 'r', linewidth=2)
plt.legend()
plt.xlabel('Vanadium Composition')
plt.ylabel('Ground-State Energy (eV)')
#plt.title('Ground-State Energy for all Compositions Predicted with Cluster Expansions Model',
  #        fontsize = 15)
ax_plot.text(0.02, 0.076, 'Temperature = 10 K',
                 transform=ax_plot.transAxes, verticalalignment='top', bbox=props)
plt.xlim([-0.03, 1.03])
plt.show()

#%% Plotting the ordered vs disordered Helmholtz Free Energy at different temperatures
j = 0

for temperature in temperatures:
    
    plt.figure(figsize=(6, 4))
    ax_plot = plt.subplot()
    plt.plot(compositions_ordered, energies_ordered_per_atom,
             label='Ordered Compositions Energies', marker='o', color='tab:blue', 
             linewidth=2)
    plt.plot(compositions, free_energies_all_temp[:, j], 
             label='Random Compositions Energies', color='r', linewidth=2)
    plt.legend()
    plt.xlabel('Vanadium Composition')
    plt.ylabel('Free Energy (eV)')
    #plt.title('Ordered Free Energy and Random Free Energy vs Composition', fontsize = 15)
    ax_plot.text(0.02, 0.076, 'Temperature = {} ºK'.format(temperature),
                      transform=ax_plot.transAxes, verticalalignment='top', bbox=props)
    
    plt.show()
    
    j += 1
    
#%% Indexing Free Energies of the compositions in list compositions_ordered from 
# the free energies array predicted by the cluster expansion model

# Note: the free_energies_all_temp array includes energies per atom predicted 
# for all possible random configurations. List of energies indexed represent 
# the predictions for the random versions of the ordered configurations

free_energies_0 = free_energies_all_temp[0, :].reshape((1, len(temperatures)))
free_energies_125 = free_energies_all_temp[125, :].reshape((1, len(temperatures)))
free_energies_250 = free_energies_all_temp[250, :].reshape((1, len(temperatures)))
free_energies_500 = free_energies_all_temp[500, :].reshape((1, len(temperatures)))
free_energies_750 = free_energies_all_temp[750, :].reshape((1, len(temperatures)))
free_energies_875 = free_energies_all_temp[875, :].reshape((1, len(temperatures)))
free_energies_1000 = free_energies_all_temp[1000, :].reshape((1, len(temperatures)))

free_energies_comparation = np.empty((0, len(temperatures)), float)

free_energies_comparation = np.vstack([free_energies_0, 
                                       free_energies_125, 
                                       free_energies_250, 
                                       free_energies_500, 
                                       free_energies_750, 
                                       free_energies_875, 
                                       free_energies_1000])

df_free_energies_comparation = pd.DataFrame(free_energies_comparation, 
                                            columns = temperatures)
df_free_energies_comparation.to_csv('Free energies comparation.csv')

#%% Calculating and plotting the difference between ordered and random 
# compositions free energies in all temperatures simulated

diff_ord_ran_energies_array = np.empty((7, 0), float)

k = 0

for temperature in temperatures:
    
    free_energies_random_config_stack = np.empty((0, 1), float)
    
    for energy in free_energies_comparation[:, k]:
        free_energies_random_config_stack = np.vstack([free_energies_random_config_stack, 
                                                       energy])
    
    diff_ord_ran_energies = energies_ordered_per_atom - free_energies_random_config_stack
    
    diff_ord_ran_energies_array = np.hstack([diff_ord_ran_energies_array, 
                                                   diff_ord_ran_energies])
    
    k += 1
    
    ax_plot = plt.subplot()
    plt.plot(compositions_ordered, diff_ord_ran_energies)
    plt.xlabel('Vanadium Composition')
    plt.ylabel('Free Energy Difference (eV)')
    plt.xticks(ticks = [0, 0.25, 0.5, 0.75, 1], labels = [0, 0.25, 0.5, 0.75, 1])
    plt.title('Difference Between Ordered and Random Free Energy vs Composition', 
              fontsize = 15)
    ax_plot.text(0.02, 0.076, 'Temperature = {} ºK'.format(temperature),
                      transform=ax_plot.transAxes, verticalalignment='top', bbox=props)
    plt.style.use('fivethirtyeight')
    
    plt.show()
    
i = 0

temperatures = np.array(temperatures).reshape(-1, 1)

x_intercepts = np.empty((0, 1), float)

for composition in compositions_ordered:
    
    if composition != 0 and composition != 1:
    
        model = linear_model.LinearRegression()
        model.fit(temperatures, diff_ord_ran_energies_array[i, :])
        m = model.coef_ 
        b = model.intercept_
        
        x_intercept = -b / m
        x_intercepts = np.vstack([x_intercepts, x_intercept])
        
        ax_plot = plt.subplot()
        plt.plot(temperatures, diff_ord_ran_energies_array[i, :])
        plt.title('Difference Between Ord and Ran Free Energies vs Composition')
        plt.xlabel('Free Energy (eV)')
        plt.ylabel('Temperature (K)')
        ax_plot.text(0.02, 0.076, 'Composition = {}'.format(compositions_ordered[i]),
                          transform=ax_plot.transAxes, verticalalignment='top', bbox=props)
        plt.style.use('fivethirtyeight')
        
        plt.show()
        
    i += 1

phase_transition_diagram_data = np.array([[0.125], 
                                          [0.25], 
                                          [0.5], 
                                          [0.75]])
                                          
phase_transition_diagram_data = np.hstack([phase_transition_diagram_data, 
                                           x_intercepts[0:-1]])

df_phase_transition_diagram_data = pd.DataFrame(phase_transition_diagram_data)
df_phase_transition_diagram_data.to_csv('Phase transition diagram data.csv')

plt.plot(phase_transition_diagram_data[:, 0], phase_transition_diagram_data[:, 1])
plt.xlabel('Vanadium Composition')
plt.ylabel('Temperature (K)')
plt.title('Phase Transition Temperature vs Vanadium Composition')
plt.xticks(ticks = [0.125, 0.25, 0.5, 0.75], labels = [0.125, 0.25, 0.5, 0.75])

plt.show()

df_diff_ord_ran_energies_array = pd.DataFrame(diff_ord_ran_energies_array, 
                                              index = compositions_ordered,
                                              columns = temperatures)
df_diff_ord_ran_energies_array.to_csv('Differences between ordered and random free energies at different temperatures.csv')

