# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 07:02:13 2021

@author: cesar
"""

import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt

lattice_parameters = []
vanadium_composition = 0.25
lattice_value_for = 2.889

vanadium_atoms_unit_cell = int(vanadium_composition * 16)

for i in range(1):
    lattice_parameters.append(lattice_value_for)
    lattice_value_for += 0.01

energy_array = np.empty((1, 2))

for lattice_parameter in lattice_parameters:
    
    sub_directory = 'Simulation_Lattice_{}'.format(round(lattice_parameter, 2))
    parent_dir = os.getcwd()
    
    new_directory = os.path.join(parent_dir, sub_directory)
    os.chdir(new_directory)
    
    f = open('log.lammps', 'r')
    lines = f.readlines()
    
    enum = list(enumerate(lines))
    
    initial_pe = enum[54][1]
    initial_pe = initial_pe.split()
    print(initial_pe)
    initial_pe = initial_pe[0]
    print(initial_pe)
    
    energy_array_stack = np.full((1, 2), [round(lattice_parameter, 2), initial_pe])
    
    energy_array = np.vstack([energy_array, energy_array_stack])
    
    os.chdir(parent_dir)


energy_array = np.delete(energy_array, 0, 0)

df = pd.DataFrame(energy_array, columns=('Lattice Parameter', 'Initial Potential Energy'))
df.to_csv('Lattice vs Energy Fe.csv')


