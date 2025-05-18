# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:34:03 2024

@author: Harshit
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:36:37 2024

@author: Harshit
"""
'''
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
retire_df = pd.read_csv('percent_wrong_retire.csv')
restart_df = pd.read_csv('percent_wrong_restart.csv')
QWI_df = pd.read_csv('percent_wrong_QWI.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_retire = retire_df['percent_wrong']
percent_wrong_restart = restart_df['percent_wrong']
percent_wrong_QWI = QWI_df['percent_wrong']

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(percent_wrong_retire, label='QGI')
plt.plot(percent_wrong_restart, label='Restart-in-state')
plt.plot(percent_wrong_QWI, label='QWI')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=15)
plt.ylabel('Cumm. % of wrong actions',fontsize=15)
plt.title('Cumm. % of wrong actions vs Time step (n)',fontsize=15)

# Adding a legend
plt.legend()

# Display the plot
plt.show()
'''


import csv
import matplotlib.pyplot as plt

# List of CSV file paths and their labels for the legend
csv_file_paths = [
    ('restart_toy.csv', 'Restart-in-state'),
    ('qgi_toy.csv', 'QGI'),
    ('QWI_toy.csv', 'QWI')
]

# State column names
state_columns = ['State0', 'State1', 'State2', 'State3']

# Plot data for each state
for i in range(4):
    state_key = state_columns[i]
    plt.figure()
    
    for csv_file_path, label in csv_file_paths:
        state_values = []

        # Read data from the current CSV file
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                state_values.append(float(row[state_key]))

        # Plot the data for the current state
        plt.plot(state_values[:5000], label=label)

    plt.xlabel('Time step (n)', fontsize=15)
    plt.ylabel('Gittins index', fontsize=15)
    plt.legend()
    plt.grid(True)
    
    # Save the plot for the current state
    plt.savefig(f'{state_key}_plot_NN.png')
    plt.show()
    plt.close()

print("Plots saved successfully.")
