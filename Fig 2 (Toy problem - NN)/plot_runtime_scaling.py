import matplotlib.pyplot as plt

# Convert time in 'hours:min:seconds' to seconds
# Convert time in 'hours:min:seconds' or 'min:seconds' to seconds
def time_to_seconds(time_str):
    time_str = time_str.replace('hr', '').replace('min', '').replace('sec', '').strip()
    time_parts = time_str.split()
    
    if len(time_parts) == 3:
        h, m, s = map(int, time_parts)
        return h * 3600 + m * 60 + s
    elif len(time_parts) == 2:
        m, s = map(int, time_parts)
        return m * 60 + s
    else:
        return int(time_parts[0])  # If only seconds are present (unlikely in this case)

# Now the rest of the code remains the same


# Data
episodes = [5000, 10000, 20000, 50000, 100000]
qwin_times = ["9 min 32 sec", "17 min 43 sec", "35 min 24 sec", "1 hr 31 min 21 sec", "3 hr 2 min 34 sec"]
dgn_times = ["3 min 10 sec", "6 min 30 sec", "13 min 24 sec", "32 min 16 sec", "1 hr 4 min 48 sec"]

# Convert times to seconds
qwin_seconds = [time_to_seconds(t) for t in qwin_times]
dgn_seconds = [time_to_seconds(t) for t in dgn_times]

plt.figure(figsize=(10, 10))
# Plotting
plt.plot(episodes, qwin_seconds, label='QWINN', marker='o', linestyle='-')
plt.plot(episodes, dgn_seconds, label='DGN', marker='s', linestyle='--')

# Labels and Title
plt.xlabel('Number of Episodes',fontsize = 25)
plt.ylabel('Time (in seconds)',fontsize = 25)
plt.legend(fontsize=20)
plt.tick_params(axis='both', labelsize=16)  # Adjust 'labelsize' to your desired size
plt.savefig("C:\\Intern\\Gittin\'s plots\\All codes directory\\Fig 2 (Toy problem - NN)\\runtime_scaling_NN.png")
plt.show()
