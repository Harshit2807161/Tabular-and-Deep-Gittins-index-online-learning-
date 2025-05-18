# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:36:37 2024

@author: Harshit
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
restart_df = pd.read_csv('C:\Intern\percent_wrong_QWINN_largetoy.csv')
DGN_df = pd.read_csv('C:\Intern\percent_wrong_DGN_largetoy.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_restart = restart_df['percent_wrong']
percent_wrong_DGN = DGN_df['percent_wrong']

# Plotting the data
plt.figure(figsize=(10,10))

plt.plot(percent_wrong_restart, label='QWINN')
plt.plot(percent_wrong_DGN, label='DGN')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=18)
plt.ylabel('Cumm. % of suboptimal arms chosen',fontsize=18)

# Adding a legend
plt.legend(fontsize = 10)

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\wrong_arms_largetoy_new.png")
plt.show()


restart_df = pd.read_csv('C:\Intern\BRE_QWINN_largetoy.csv')
QWI_df = pd.read_csv('C:\Intern\BRE_DGN_largetoy.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_restart = restart_df['BRE']
percent_wrong_QWI = QWI_df['BRE']

# Plotting the data
plt.figure(figsize=(10, 10))

plt.plot(percent_wrong_restart, label='QWINN')
plt.plot(percent_wrong_QWI, label='DGN')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=25)
plt.ylabel('Bellman Relative Error',fontsize=25)

# Adding a legend
plt.legend(fontsize=20)
plt.tick_params(axis='both', labelsize=16)  # Adjust 'labelsize' to your desired size
# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\bellman_error_largetoy_new.png")
plt.show()