# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:36:37 2024

@author: Harshit
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
retire_df = pd.read_csv('C:\Intern\percent_wrong_QGI_constHR.csv')
restart_df = pd.read_csv('C:\Intern\percent_wrong_restart_constHR.csv')
DGN_df = pd.read_csv('C:\Intern\percent_wrong_DGN_constHR.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_retire = retire_df['percent_wrong']
percent_wrong_restart = restart_df['percent_wrong']
percent_wrong_DGN = DGN_df['percent_wrong']

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(percent_wrong_retire, label='QGI')
plt.plot(percent_wrong_restart, label='Restart-in-state')
plt.plot(percent_wrong_DGN, label='DGN')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=15)
plt.ylabel('Cumm. % of suboptimal arms chosen',fontsize=15)

# Adding a legend
plt.legend()

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\wrong_arms_constHR.png")
plt.show()


retire_df = pd.read_csv('C:\Intern\BRE_QGI_constHR.csv')
restart_df = pd.read_csv('C:\Intern\BRE_restart_constHR.csv')
QWI_df = pd.read_csv('C:\Intern\BRE_DGN_constHR.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_retire = retire_df['BRE']
percent_wrong_restart = restart_df['BRE']
percent_wrong_QWI = QWI_df['BRE']

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(percent_wrong_retire, label='QGI')
plt.plot(percent_wrong_restart, label='Restart-in-state')
plt.plot(percent_wrong_QWI, label='DGN')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=15)
plt.ylabel('Bellman Relative Error',fontsize=15)

# Adding a legend
plt.legend()

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\bellman_error_constHR.png")
plt.show()


