# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:36:37 2024

@author: Harshit
"""
'''
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
retire_df = pd.read_csv('C:\Intern\percent_wrong_QGI_decHR.csv')
restart_df = pd.read_csv('C:\Intern\percent_wrong_restart_decHR.csv')
DGN_df = pd.read_csv('C:\Intern\percent_wrong_DGN_decHR.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_retire = retire_df['percent_wrong']
percent_wrong_restart = restart_df['percent_wrong']
percent_wrong_DGN = DGN_df['percent_wrong']

# Plotting the data
plt.figure(figsize=(10, 10))

plt.plot(percent_wrong_retire, label='QGI')
plt.plot(percent_wrong_restart, label='Restart-in-state')
plt.plot(percent_wrong_DGN, label='DGN')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=25)
plt.ylabel('Cumm. % of suboptimal arms chosen',fontsize=25)

plt.legend(fontsize=22)
plt.tick_params(axis='both', labelsize=18)  # Adjust 'labelsize' to your desired size

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\wrong_arms_decHR.png")
plt.show()


retire_df = pd.read_csv('C:\Intern\BRE_QGI_decHR.csv')
restart_df = pd.read_csv('C:\Intern\BRE_restart_decHR.csv')
QWI_df = pd.read_csv('C:\Intern\BRE_DGN_decHR.csv')

# Extract the percent_wrong column from each dataframe
BRE_retire = retire_df['BRE']
BRE_restart = restart_df['BRE']
BRE_QWI = QWI_df['BRE']

# Plotting the data
plt.figure(figsize=(10, 10))

plt.plot(BRE_retire, label='QGI')
plt.plot(BRE_restart, label='Restart-in-state')
plt.plot(BRE_QWI, label='DGN')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=25)
plt.ylabel('Bellman Relative Error',fontsize=25)



plt.legend(fontsize=22)
plt.tick_params(axis='both', labelsize=18)  # Adjust 'labelsize' to your desired size

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\bellman_error_decHR.png")
plt.show()



plt.figure(figsize=(10, 10))
plt.plot(percent_wrong_retire, label='QGI (Wrong arms)')
plt.plot(percent_wrong_restart, label='Restart-in-state (Wrong arms)')
plt.plot(percent_wrong_DGN, label='DGN (Wrong arms)')
plt.plot(BRE_retire, label='QGI (BRE)')
plt.plot(BRE_restart, label='Restart-in-state (BRE)')
plt.plot(BRE_QWI, label='DGN (BRE)')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=27)
plt.ylabel('Bellman error and Cumm % wrong arms',fontsize=27)

# Adding a legend
plt.legend(fontsize=22)
plt.tick_params(axis='both', labelsize=18)  # Adjust 'labelsize' to your desired size
# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\all_decHR.png")
plt.show()

'''
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
retire_df = pd.read_csv('C:\Intern\percent_wrong_QGI_incHR.csv')
restart_df = pd.read_csv('C:\Intern\percent_wrong_restart_monoHR.csv')
DGN_df = pd.read_csv('C:\Intern\percent_wrong_DGN_monoHR.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_retire = retire_df['percent_wrong']
percent_wrong_restart = restart_df['percent_wrong']
percent_wrong_DGN = DGN_df['percent_wrong']

# Plotting the data
plt.figure(figsize=(10, 10))

plt.plot(percent_wrong_retire, label='QGI')
plt.plot(percent_wrong_restart, label='Restart-in-state')
plt.plot(percent_wrong_DGN, label='DGN')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=25)
plt.ylabel('Cumm. % of suboptimal arms chosen',fontsize=25)

plt.legend(fontsize=22)
plt.tick_params(axis='both', labelsize=18)  # Adjust 'labelsize' to your desired size

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\wrong_arms_monoHR.png")
plt.show()


retire_df = pd.read_csv('C:\Intern\BRE_QGI_incHR.csv')
restart_df = pd.read_csv('C:\Intern\BRE_restart_monoHR.csv')
QWI_df = pd.read_csv('C:\Intern\BRE_DGN_monoHR.csv')

# Extract the percent_wrong column from each dataframe
BRE_retire = retire_df['BRE']
BRE_restart = restart_df['BRE'] 
BRE_QWI = QWI_df['BRE']

# Plotting the data
plt.figure(figsize=(10, 10))

plt.plot(BRE_retire, label='QGI')
plt.plot(BRE_restart, label='Restart-in-state')
plt.plot(BRE_QWI, label='DGN')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=25)
plt.ylabel('Bellman Relative Error',fontsize=25)



plt.legend(fontsize=22)
plt.tick_params(axis='both', labelsize=18)  # Adjust 'labelsize' to your desired size

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\bellman_error_monoHR.png")
plt.show()



plt.figure(figsize=(10, 10))
plt.plot(percent_wrong_retire, label='QGI (Wrong arms)')
plt.plot(percent_wrong_restart, label='Restart-in-state (Wrong arms)')
plt.plot(percent_wrong_DGN, label='DGN (Wrong arms)')
plt.plot(BRE_retire, label='QGI (BRE)')
plt.plot(BRE_restart, label='Restart-in-state (BRE)')
plt.plot(BRE_QWI, label='DGN (BRE)')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=25)
plt.ylabel('Bellman error and Cumm % wrong arms',fontsize=15)

# Adding a legend
plt.legend()

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\all_decHR.png")
plt.show()