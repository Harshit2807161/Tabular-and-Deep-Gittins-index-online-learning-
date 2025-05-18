# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:01:26 2024

@author: Harshit
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
inc_restart = pd.read_csv('inc_restart.csv')
print(inc_restart)
inc_retire = pd.read_csv('inc_retire.csv')
dec_restart = pd.read_csv('dec_restart.csv')
dec_retire = pd.read_csv('dec_retire.csv')
const_restart = pd.read_csv('const_restart.csv')
const_retire = pd.read_csv('const_retire.csv')

# Plot 1: Increasing Restart vs Retire
plt.figure(figsize=(8, 6))
plt.plot(inc_restart, label='Restart-in-state')
plt.plot(inc_retire, label='QGI')
plt.legend()
plt.xlabel('Time step (n)')  # Customize as needed
plt.ylabel('Gittins index')  # Customize as needed
plt.show()

# Plot 2: Decreasing Restart vs Retire
plt.figure(figsize=(8, 6))
plt.plot(dec_restart, label='Restart-in-state')
plt.plot(dec_retire, label='QGI')
plt.legend()
plt.xlabel('Time step (n)')  # Customize as needed
plt.ylabel('Gittins index')  # Customize as needed
plt.show()

# Plot 3: Constant Restart vs Retire
plt.figure(figsize=(8, 6))
plt.plot(const_restart, label='Restart-in-state')
plt.plot(const_retire, label='QGI')
plt.legend()
plt.xlabel('Time step (n)')  # Customize as needed
plt.ylabel('Gittins index')  # Customize as needed
plt.show()