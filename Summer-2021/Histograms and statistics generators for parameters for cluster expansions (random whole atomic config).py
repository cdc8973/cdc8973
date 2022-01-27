# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:29:43 2021

@author: cesar
"""

from ase import Atoms
from ase.visualize import view
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import math

vanadium_compositions = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]

means_array = np.empty((0, 4), float)
stds_array = np.empty((0, 4), float)

for vanadium_composition in vanadium_compositions:

    vanadium_atoms_unit_cell = int(vanadium_composition * 16)
    
    #specifies lattice parameter value for each composition
    if vanadium_composition == 0:
        lattice_parameter = 2.8664
        
    elif vanadium_composition == 0.125:
        lattice_parameter = 2.8742776
        
    elif vanadium_composition == 0.25:
        lattice_parameter = 2.8832316
        
    elif vanadium_composition == 0.375:
        lattice_parameter = 2.8943574
        
    elif vanadium_composition == 0.5:
        lattice_parameter = 2.889
        
    elif vanadium_composition == 0.625:
        lattice_parameter = 2.9334862
        
    elif vanadium_composition == 0.75:
        lattice_parameter = 2.9641786
        
    elif vanadium_composition == 0.875:
        lattice_parameter = 2.9989696
        
    elif vanadium_composition == 1:
        lattice_parameter = 3.041

    # Size of the system cell in unit cells
    # assuming a cubic cell, starting at the origin
    system_size = 2
    
    # Defining total atoms and total vanadium atoms
    total_atoms = 16 * (system_size ** 3)
    
    total_vanadium_atoms = int(vanadium_composition * total_atoms)
    
    parameters_array = np.empty((0, 4), float)
    random_positions_array = np.empty((0, total_vanadium_atoms), float)
    
    for i in range(1000):
        
        # Unit cell length for FeV
        unit_cell_length = 2*lattice_parameter
        
        # Defining unit cell
        basis = np.array([[1.0, 0.0, 0.0], 
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])*unit_cell_length
        
        base_atoms = np.array([[0.0, 0.0, 0.0], 
                               [0.5, 0.0, 0.0], 
                               [0.0, 0.5, 0.0],
                               [0.5, 0.5, 0.0],
                               [0.0, 0.0, 0.5],
                               [0.0, 0.5, 0.5],
                               [0.5, 0.0, 0.5],
                               [0.5, 0.5, 0.5],
                               [0.75, 0.25, 0.25],
                               [0.25, 0.75, 0.25],
                               [0.25, 0.25, 0.75],
                               [0.75, 0.75, 0.75],
                               [0.25, 0.25, 0.25],
                               [0.75, 0.25, 0.75],
                               [0.25, 0.75, 0.75],
                               [0.75, 0.75, 0.25]])*unit_cell_length
        
        
        # Generate atom positions (expands unit cell to desired amount of atoms)
        positions = []
        for i in range(system_size):
            for j in range(system_size):
                for k in range(system_size):
                    base_position = np.array([i, j, k])
                    cart_position = np.inner(basis.T, base_position)
                    for atom in base_atoms:
                        positions.append(cart_position + atom)
        
        # Defines random positions of vanadium atoms in whole atomic configuration 
        range_total_atoms = np.arange(1, total_atoms + 1)
        random_int_array = np.random.choice(range_total_atoms, 
                                            total_vanadium_atoms, replace=False)
        
        random_positions_array = np.vstack([random_positions_array, random_int_array])
        
        # Creates V and Fe atoms arrays for stack
        atom_position_index = 1
        
        v_atoms_positions = np.empty((0, 3), int)
        fe_atoms_positions = np.empty((0, 3), int)
        
        total_fe_atoms = 0
        total_v_atoms = 0
        
        #Classifies each atom as iron or vanadium
        for i,pos in enumerate(positions):
                        if atom_position_index in random_int_array:
                            v_atoms_stack = np.full((1, 3), (pos[0], pos[1], pos[2]))
                            v_atoms_positions = np.vstack([v_atoms_positions, v_atoms_stack])
                            atom_position_index += 1
                            total_v_atoms += 1
                        else:
                            fe_atoms_stack = np.full((1, 3), (pos[0], pos[1], pos[2]))
                            fe_atoms_positions = np.vstack([fe_atoms_positions, fe_atoms_stack])
                            atom_position_index += 1
                            total_fe_atoms += 1
        
        #Creation of atoms from both elements with ASE
        fe_atoms = Atoms('Fe{}'.format(total_fe_atoms), positions = fe_atoms_positions)
        v_atoms = Atoms('V{}'.format(total_v_atoms), positions = v_atoms_positions)
        
        atoms = deepcopy(fe_atoms)
        atoms.extend(v_atoms)
        atoms.set_cell(basis)
        
        #Getting coordinates for central atom of the atomic configuration
        x_center_coordinate = (unit_cell_length * system_size)/2
        y_center_coordinate = (unit_cell_length * system_size)/2
        z_center_coordinate = (unit_cell_length * system_size)/2
        
        center_atom_coordinates = np.array([x_center_coordinate, y_center_coordinate,
                                   z_center_coordinate])
        
        atom_center_index = 0
        
        #Lines for getting index of central atom (index in the atoms object)
        for atom in atoms:
            
            logical_0 = (atom.position[0] == center_atom_coordinates[0])
            logical_1 = (atom.position[1] == center_atom_coordinates[1])
            logical_2 = (atom.position[2] == center_atom_coordinates[2])
            
            if logical_0 and logical_1 and logical_2:
                break
            else:
                atom_center_index += 1
        
        # Getting distances of each atom from central atom
        
        def distance(atom1, atom2):
            return round(math.sqrt((atom1[0] - atom2[0]) ** 2 + (atom1[1] - atom2[1]) ** 2 +
                             (atom1[2] - atom2[2]) ** 2), 8)
        
        
        atoms_index_distances = 0
        distances_array = np.empty((0, 1), float)
        indexes_distance_array = np.empty((0, 1), dtype =  int)
        
        for atom in atoms:
        
            position = atom.position
            dist = distance(position, center_atom_coordinates)
            
            distances_array = np.vstack([distances_array, dist])
            indexes_distance_array = np.vstack([indexes_distance_array, atoms_index_distances])
            
            atoms_index_distances += 1
        
        distances_array = np.hstack([indexes_distance_array, distances_array])
        indexing_array = np.argsort(distances_array[:, 1])
        
        distances_array = distances_array[indexing_array]
        
        #Classifying fnn's and snn's 
        first_nearest_neighbors_dist = np.empty((0, 2), float)
        second_nearest_neighbors_dist = np.empty((0, 2), float)
        
        for atom in distances_array:
          
            if atom[1] > 0:
                if atom[1] < lattice_parameter:
                    first_nearest_neighbors_dist = np.vstack([first_nearest_neighbors_dist, atom])
                elif atom[1] == lattice_parameter:
                    second_nearest_neighbors_dist = np.vstack([second_nearest_neighbors_dist, atom])
        
        atomic_number_central_atom = atoms.numbers[atom_center_index]
        
        if atomic_number_central_atom == 26:
            coefficient_central_atom = 1
        else:
            coefficient_central_atom = -1
        
        
        #first nearest neighbors pair correlation functions (n = 2, k = 1)
        first_nearest_neighbors_indexes = first_nearest_neighbors_dist[:, 0]
        
        coefficients_fnn = np.empty((0, 1), float)
        
        for index in first_nearest_neighbors_indexes:
            
            atomic_number = atoms.numbers[int(index)]
            
            if atomic_number == 26:
                coefficients_fnn = np.vstack([coefficients_fnn, np.array([1])])
            else:
                coefficients_fnn = np.vstack([coefficients_fnn, np.array([-1])])
        
        sumatory_pairs_fnn = 0
        total_pairs = len(coefficients_fnn)
        
        for coefficient in coefficients_fnn:
            
            product = coefficient_central_atom * coefficient
            sumatory_pairs_fnn += product 
        
        pair_fnn_correlation_function = float(sumatory_pairs_fnn / total_pairs)
        
        
        #second nearest neighbors pair correlation functions (n = 2, k = 2)
        second_nearest_neighbors_indexes = second_nearest_neighbors_dist[:, 0]
        
        coefficients_snn = np.empty((0, 1), float)
        
        for index in second_nearest_neighbors_indexes:
            
            atomic_number = atoms.numbers[int(index)]
            
            if atomic_number == 26:
                coefficients_snn = np.vstack([coefficients_snn, np.array([1])])
            else:
                coefficients_snn = np.vstack([coefficients_snn, np.array([-1])])
        
        
        sumatory_pairs_snn = 0
        total_pairs_snn = len(coefficients_snn)
        
        for coefficient in coefficients_snn:
            
            product = coefficient_central_atom * coefficient
            sumatory_pairs_snn += product 
            
        pair_snn_correlation_function = float(sumatory_pairs_snn / total_pairs_snn)
        
        
        #point correlation function (n = 1, k = 1)
        atoms_counter = 0
        sumatory_points = 0
        
        for atom in atoms:
            
            atomic_number = atom.number
        
            if atomic_number == 26:
                sumatory_points += 1
            elif atomic_number == 23:
                sumatory_points += -1
        
            atoms_counter += 1
        
        point_correlation_function = float(sumatory_points / atoms_counter) 
        
        #empty correlation function (n = 0, k = 0). One by definition
        empty_correlation_function  = 1
        
        correlation_functions = np.full((1, 4), (empty_correlation_function, 
                                                 point_correlation_function, 
                                                 pair_fnn_correlation_function, 
                                                 pair_snn_correlation_function))
        
        parameters_array = np.vstack([parameters_array, correlation_functions])
    
    
    #view(atoms)
    
    df_random_positions = pd.DataFrame(random_positions_array)
    df_random_positions.to_csv('Random unit cell positions array composition {}.csv'.format(vanadium_composition))
    
    df_parameters = pd.DataFrame(parameters_array)
    df_parameters.to_csv('Correlation functions array. Composition {}.csv'.format(vanadium_composition))
    
    mean_empty_cf = np.mean(parameters_array[:, 0])
    mean_point_cf = np.mean(parameters_array[:, 1])
    mean_pair_fnn_cf = np.mean(parameters_array[:, 2])
    mean_pair_snn_cf = np.mean(parameters_array[:, 3])
    
    std_empty_cf = np.std(parameters_array[:, 0])
    std_point_cf = np.std(parameters_array[:, 1])
    std_pair_fnn_cf = np.std(parameters_array[:, 2])
    std_pair_snn_cf = np.std(parameters_array[:, 3])
    
    means_array_stack = np.array([mean_empty_cf, mean_point_cf, mean_pair_fnn_cf, 
                            mean_pair_snn_cf])
    stds_array_stack = np.array([std_empty_cf, std_point_cf, std_pair_fnn_cf, 
                                 std_pair_snn_cf])
    
    means_array = np.vstack([means_array, means_array_stack])
    stds_array = np.vstack([stds_array, stds_array_stack])
    
    bins = [-1, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0, 
            0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    xticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    
    plt.hist(parameters_array, bins = bins, edgecolor = 'black', label = ['Empty', 'Point', 'Pair fnn', 'Pair snn'])
    plt.title('Histogram Correlation Functions V composition {}'.format(vanadium_composition))
    plt.xlabel('Value for correlation')
    plt.ylabel('Amount of cases')
    plt.xticks(xticks, fontsize = 10)
    plt.grid(True)
    
    plt.axvline(mean_empty_cf, color = 'b', label = 'Mean Empty', linestyle = '--', linewidth = 2)
    plt.axvline(mean_point_cf, color = 'r', label = 'Mean Point', linestyle = '--', linewidth = 2)
    plt.axvline(mean_pair_fnn_cf, color = 'g', label = 'Mean Pair FNN', linestyle = '--', linewidth = 2)
    plt.axvline(mean_pair_snn_cf, color = 'y', label = 'Mean Pair SNN', linestyle = '--', linewidth = 2)
    plt.legend()
    
    plt.show()
    
    plt.hist(parameters_array[:, 0], bins = bins, edgecolor = 'black', label = ['Empty'], color = 'b')
    plt.title('Histogram Correlation Functions V composition {} n = 0, k = 0'.format(vanadium_composition))
    plt.xlabel('Value for correlation')
    plt.ylabel('Amount of cases')
    plt.xticks(xticks, fontsize = 10)
    plt.grid(True)
    plt.axvline(mean_empty_cf, color = 'b', label = 'Mean Empty', linestyle = '--', linewidth = 2)
    plt.legend()
    
    plt.show()
    
    plt.hist(parameters_array[:, 1], bins = bins, edgecolor = 'black', label = ['Point'], color = 'r')
    plt.title('Histogram Correlation Functions V composition {} n = 1, k = 1'.format(vanadium_composition))
    plt.xlabel('Value for correlation')
    plt.ylabel('Amount of cases')
    plt.xticks(xticks, fontsize = 10)
    plt.grid(True)
    plt.axvline(mean_point_cf, color = 'r', label = 'Mean Point', linestyle = '--', linewidth = 2)
    plt.legend()
    
    plt.show()
    
    plt.hist(parameters_array[:, 2], bins = bins, edgecolor = 'black', label = ['Pair fnn'], color = 'y')
    plt.title('Histogram Correlation Functions V composition {} n = 2,k = 1'.format(vanadium_composition))
    plt.xlabel('Value for correlation')
    plt.ylabel('Amount of cases')
    plt.xticks(xticks, fontsize = 10)
    plt.grid(True)
    plt.axvline(mean_pair_fnn_cf, color = 'y', label = 'Mean Pair fnn', linestyle = '--', linewidth = 2)
    plt.legend()
    
    plt.show()
    
    plt.hist(parameters_array[:, 3], bins = bins, edgecolor = 'black', label = ['Pair snn'], color = 'g')
    plt.title('Histogram Correlation Functions V composition {} n = 2, k = 2'.format(vanadium_composition))
    plt.xlabel('Value for correlation')
    plt.ylabel('Amount of cases')
    plt.xticks(xticks, fontsize = 10)
    plt.grid(True)
    plt.axvline(mean_pair_snn_cf, color = 'g', label = 'Mean Pair snn', linestyle = '--', linewidth = 2)
    plt.legend()
    
    plt.show()
    
    print('V{} composition median n = 1 {} \n'.format(vanadium_composition, mean_point_cf))
    print('V{} composition median n = 2 k = 1 {} \n'.format(vanadium_composition, mean_pair_fnn_cf))
    print('V{} composition median n = 2 k = 1 {} \n'.format(vanadium_composition, mean_pair_fnn_cf))
    
    print('V{} composition median n = 2 k = 1 {} \n'.format(vanadium_composition, mean_pair_fnn_cf))

df_mean = pd.DataFrame(means_array)
df_stds = pd.DataFrame(stds_array)

df_mean.to_csv('df_mean.csv')
df_stds.to_csv('df_stds.csv')

