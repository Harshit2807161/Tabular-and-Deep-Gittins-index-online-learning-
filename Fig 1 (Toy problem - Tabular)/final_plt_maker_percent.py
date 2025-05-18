import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
retire_df = pd.read_csv('C:\Intern\percent_wrong_QGI.csv')
restart_df = pd.read_csv('C:\Intern\percent_wrong_restart.csv')
QWI_df = pd.read_csv('C:\Intern\percent_wrong_QWI.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_retire = retire_df['percent_wrong']
percent_wrong_restart = restart_df['percent_wrong']
percent_wrong_QWI = QWI_df['percent_wrong']

# Plotting the data
plt.figure(figsize=(10, 10))

plt.plot(percent_wrong_retire, label='QGI')
plt.plot(percent_wrong_restart, label='Restart-in-state')
plt.plot(percent_wrong_QWI, label='QWI')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=20)
plt.ylabel('Cumm. % of suboptimal arms chosen',fontsize=20)
#plt.title('Cumm. % of wrong actions vs Time step (n)',fontsize=15)

# Adding a legend
plt.legend()

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\wrong_arms_toy_prob.png")
plt.show()


retire_df = pd.read_csv('C:\Intern\BRE_QGI.csv')
restart_df = pd.read_csv('C:\Intern\BRE_restart.csv')
QWI_df = pd.read_csv('C:\Intern\BRE_QWI.csv')

# Extract the percent_wrong column from each dataframe
percent_wrong_retire = retire_df['BRE'][:3000]
percent_wrong_restart = restart_df['BRE'][:3000]
percent_wrong_QWI = QWI_df['BRE'][:3000]

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(percent_wrong_retire, label='QGI')
plt.plot(percent_wrong_restart, label='Restart-in-state')
plt.plot(percent_wrong_QWI, label='QWI')

# Adding labels and title
plt.xlabel('Time step (n)',fontsize=20)
plt.ylabel('Bellman Relative Error',fontsize=20)

# Adding a legend
plt.legend(fontsize = 18)

# Display the plot
plt.savefig("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 1 (Toy problem - Tabular)\\bellman_error_toy_prob_eps1_0.7.png")
plt.show()


