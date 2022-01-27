# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 07:02:13 2021

@author: Cesar Diaz
cdiazcarav@miners.utep.edu
cesar.dc1509@gmail.com

The University of Texas at El Paso

Code created as part of the project "Machine Learning the Ground State and Free
Energies of Iron-Vanadium Alloys via Cluster Expansions" during the Summer of 
2021, under the mentorship of Prof. Jorge Munoz and funded by the Campus Office
of Undergraduate Research Initiatives. 

The program reads the values of the ground state energies for a set of already
simulated compositions, using the code "lammps script random config atoms 
running positions and simulation.py". The user needs to define the directories
and lattice parameters from which the code will go over. This ground state 
energy value is located in the log.lammps file, which is found in each 
directory (and for each lattice). The program automatically locates and reads 
those inputs in the second for loop. 

"""

import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt


lattice_parameters = []
lattice_value_for = 2.86

vanadium_composition = 0.5

iron_composition = 1 - vanadium_composition

#%% Defining the lattice parameters to run the simulations in
for i in range(9):
    lattice_parameters.append(lattice_value_for)
    lattice_value_for += 0.01

energies_array = np.empty((1, 2))

#%% Reading the energies from the specified Lattice Parameters directories

for lattice_parameter in lattice_parameters:
    
    # Defining the list of directories to read the energies from
    sub_directory = 'Simulation_Lattice_{}'.format(round(lattice_parameter, 2))
    parent_dir = os.getcwd()
    
    new_directory = os.path.join(parent_dir, sub_directory)
    os.chdir(new_directory)
    
    # Oppening the log lammps file (place where the energy number is located)
    f = open('log.lammps', 'r')
    lines = f.readlines()
    
    # Lines for reading the energy - initial energy always in index 54 of 
    # lines variable (readlines method applied to log.lammps file)
    enum = list(enumerate(lines))
    
    print(enum)
    
    initial_pe = enum[121][1] 
    #print(initial_pe)
    initial_pe = initial_pe.split()
    #print(initial_pe)
    initial_pe = initial_pe[4]
    
    energies_array_stack = np.full((1, 2), [round(lattice_parameter, 2), float(initial_pe)])
    
    energies_array = np.vstack([energies_array, energies_array_stack])
    
    os.chdir(parent_dir)
    
energies_array = np.delete(energies_array, 0, 0)
print(energies_array)

#%% Finding Minimum from the energies list

model = np.poly1d(np.polyfit(energies_array[:, 0], energies_array[:, 1], 20))
lattices_linspace = np.linspace(lattice_parameters[0], 
                                lattice_parameters[len(lattice_parameters) - 1], 500)
lattices_linspace = lattices_linspace.reshape((len(lattices_linspace), 1))

energies_regression_array = model(lattices_linspace).reshape((len(lattices_linspace), 1))
energies_regression_array = np.hstack([lattices_linspace, energies_regression_array])

print(energies_regression_array)

min_energy = min(energies_regression_array[:, 1])

i = 0

for energy in energies_regression_array[:, 1]:
    if(energy == min_energy):
        break
    
    i = i + 1
    
min_lattice_parameter = float(lattices_linspace[i])
print(min_lattice_parameter)

#%% Props for bbox parameter in the std text box. Matplotlib related stuff.
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5)

#%% Plotting the data from the simulations and the polynomial fit 

ax = plt.subplot()
plt.plot(energies_regression_array[:, 0], energies_regression_array[:, 1], 
         label = 'Regression Line')
plt.title('Lattice Parameter vs Ground-State Energy Composition {}'.format(vanadium_composition))
plt.scatter(energies_array[:, 0], energies_array[:, 1], label = 'Simulations Data', 
            s = 10)
ax.text(0.71, 0.80,'Minimum Lattice: \n {}'.format(round(min_lattice_parameter, 6)),
             transform=ax.transAxes, verticalalignment='top', bbox=props)
ax.text(0.71, 0.67,'Minimum Energy: \n {}'.format(round(min_energy, 6)),
             transform=ax.transAxes, verticalalignment='top', bbox=props)
plt.legend()
plt.show()


df = pd.DataFrame(energies_array, columns=('Lattice Parameter', 'Initial Potential Energy'))
df.to_excel('Lattice vs Energy Fe{}V{}.xlsx'.format(iron_composition, 
                                                 vanadium_composition))




