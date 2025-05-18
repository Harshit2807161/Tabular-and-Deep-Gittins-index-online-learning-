import csv
import matplotlib.pyplot as plt
import numpy as np

# List of CSV file paths and their labels for the legend
csv_file_paths = [
    #('DGN_toy.csv', 'DGN'),
    #('QWINN_toy.csv', 'QWINN'),
    ('restart_toy.csv', 'Restart-in-state'),
    ('QGI_toy.csv', 'QGI'),
    ('QWI_toy.csv', 'QWI')
]

# Initialize lists to hold data for each state
state_data = {f'State{i}': [] for i in range(5)}

# Read data from each CSV file
for csv_file_path, label in csv_file_paths:
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row if present
        for row in csv_reader:
            for i in range(5):
                state_data[f'State{i}'].append(float(row[i]))

# Plot data for each state
for i in range(5):
    state_key = f'State{i}'
    plt.figure()
    for csv_file_path, label in csv_file_paths:
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header row if present
            state_values = []
            for row in csv_reader:
                state_values.append(float(row[i]))
            plt.plot(state_values, label=label)

    plt.xlabel('Time step (n)',fontsize=15)
    plt.ylabel('Gittins index',fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{state_key}_plot_NN.png')
    plt.show()
    plt.close()

print("Plots saved successfully.")
