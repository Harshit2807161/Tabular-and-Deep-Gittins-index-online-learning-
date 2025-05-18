# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:39:15 2024

@author: Harshit
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import DGNnn_new_hetro as DGNnn
import torch
import copy
import csv
import random
from tqdm import tqdm 

#initialise-
action = np.zeros(5)
reward = np.zeros(5)
state = np.zeros(5)
arm = np.zeros(5)
Q = np.zeros((50,2,50))

hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []
hist = [[] for _ in range(50)]

M = np.zeros((50))

def calculate_v_values(Q_values_action1, Q_values_action2):
    V_values = np.zeros((50))
    for i in range(50):
            V_values[i] = max(Q_values_action1[i],Q_values_action2[i])  # Use keepdims=True to maintain the 2D shape
    return V_values/10



V_true = np.array([ 7.4522438,   7.4522438,   7.4522438,   7.4522438,   7.4522438,   7.53629608,
  7.57353058,  7.61076508,  7.6480011,   7.6852356,   7.72247086,  7.75970612,
  7.79694061,  7.83417511,  7.87141113,  7.90864563,  7.94588089,  7.98311539,
  8.02035065,  8.05758591,  8.0948204,   8.13205566,  8.16929016,  8.20652542,
  8.24376068,  8.28099518,  8.34963226,  8.42621613,  8.50279999,  8.57938385,
  8.65596695,  8.73255157,  8.80913467,  8.8857193,   8.96230316,  9.03888702,
  9.11547089,  9.19205322,  9.26863861,  9.34522095,  9.42180481,  9.49838867,
  9.57497253,  9.6515564,   9.72814178,  9.80472565,  9.88130798,  9.95789185,
 10.03447571, 10.11105957])

def bellman_relative_error(V_approx, V_true):
    nonzero_indices = V_true != 0
    if np.any(nonzero_indices):
        #relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]) / V_true[nonzero_indices])
        relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]))
        return np.mean(relative_errors)
    
def check_best_action(state):
    '''
    Given:
      - state: a list indicating the current state (0..4) for each of 5 arms
      - G:     a 5x5 matrix of Gittins indices, where G[state][arm]

    Returns:
      The index of the arm that has the highest Gittins index
      for its current state.
    '''
    best_arm = np.argmax(state)
    return best_arm

def save_list_to_csv(data_list, filename):
    """
    Save a list to a CSV file, where each element of the list is in a separate row of the same column.

    Parameters:
    data_list (list): The list to be saved.
    filename (str): The name of the CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each element of the list as a new row
        for item in data_list:
            writer.writerow([item])  # Each item is placed in a new row in a single column


def get_reward(state,action):
    rewards = []
    for i in range(5):
        if action[i]==1:
            rewards.append(5+((state[i]+1))/10)
        else:
            rewards.append(0)
    return rewards


def transition(state, action):
    P1 = np.zeros((50, 50))
    P0 = np.eye(50)
    
    for i in range(50):
        P1[i] = np.random.dirichlet(np.ones(50))  
    next_statee = []
    for i in range(len(state)):
        if action[i] == 1:
            next_statee.append(np.random.choice(50, p=P1[state[i]])) 
        else:
            next_statee.append(np.random.choice(50, p=P0[state[i]]))  
    return next_statee

def transition_uniform(state, action):
    P1 = np.full((50, 50), 1/50) 
    P0 = np.eye(50)
    
    next_statee = []
    for i in range(len(state)):
        if action[i] == 1:
            next_statee.append(np.random.choice(50, p=P1[state[i]]))  
        else:
            next_statee.append(np.random.choice(50, p=P0[state[i]])) 
    
    return next_statee


def select_action(s,M,epsilon):
    r = random.random()
    if r < epsilon:
        action = np.random.choice(5)
    else:
        gre = [0]*len(s)
        for i in range(len(s)):
            gre[i] = M[s[i]]
            
        action = np.argmax(gre)
    actione = [0]*len(s)
    actione[action] = 1
    return actione

def gettask(action):
        task = 0
        for i in range(5):
            if(action[i]==1):
                task = i
                break
        return task

import time
def game():
    start_time = time.time()
    gamma = 0.9
    epsilon = 0.4
    episodes = 5000
    rate = 1
    learning_rate = 0.1
    beta = 0.6
    agent = DGNnn.Agent(2,2,-1)
    eps = 1
    state = np.random.choice(50,5)
    cumm_wrong_steps = []
    plt_wrong_actions = []
    BRE = []
    for episode_no in tqdm(range(episodes)):
        if episode_no % 20 == 0:
            state = np.random.choice(50,5)
        if episode_no==0:
            learning_rate = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        if episode_no>=1:
            learning_rate = 1 / math.ceil(episode_no / 5000)
            if(episode_no%1==0):
                t = episode_no
                beta = 0.6 / (1 + math.ceil(t* math.log(t) / 5000))
            else:
                beta = 0
        
        eps = max(eps*0.9995,0.1)
        arm = select_action(copy.copy(state),M,eps)
        action = arm
        task = gettask(action)
        task_eps = task
        task_opt = check_best_action(state)
        if task_eps != task_opt:
            cumm_wrong_steps.append(1)
        else:
            cumm_wrong_steps.append(0)

        plt_wrong_actions.append(np.mean(cumm_wrong_steps)*100)
        Q0 = np.zeros((50))
        for i in range(50):
                input_tensor = torch.FloatTensor(np.array([i,i]))
                output_values = agent.qnetwork_local.forward(input_tensor)
                Q0[i] = output_values[1] 
        V_values = calculate_v_values(Q0,M)
        BRE.append(bellman_relative_error(V_values,V_true))

        next_state = transition(copy.copy(state),action)
        reward = get_reward(copy.copy(state),action)
        '''print(state)
        print(next_state)
        print(reward)
        print(action)'''     
        for i in range(5):
            if action[i]==1:
                for k in range(50):
                        agent.step(np.array([state[i],k]), action[i], reward[i], np.array([next_state[i],k]),False,M[k])
        #for i in range(5):
        for k in range(50):
                input_tensor = torch.FloatTensor(np.array([k,k]))
                output_values = agent.qnetwork_local.forward(input_tensor)
                M[k]+=beta*(output_values[1]-M[k])
        current_time = time.time()-start_time
        state = next_state
        for i in range(50):
            hist[i].append(M[i])

    print(current_time)
    print(V_values)
    # Specify the CSV file path
    csv_file_path = 'DGN_toy.csv'

    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2','State3','State4'])  # Optional header row
        csv_writer.writerows(zip(hist0, hist1, hist2, hist3, hist4))

    print(f"The two lists have been saved to {csv_file_path}")

    filename = 'C:\\Intern\\percent_wrong_DGN_largetoy.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['percent_wrong'])  # Writing the header
        for value in plt_wrong_actions:
            writer.writerow([value])  # Writing each value in a new row
    
    filename = 'C:\\Intern\\BRE_DGN_largetoy.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['BRE'])  # Writing the header
        for value in BRE:
            writer.writerow([value])  # Writing each value in a new row

    
    
game()

print(M)
plt.figure(figsize=(6,6))
plt.title('M vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('M',fontsize = 'xx-large')
plt.plot(hist[0],'-',c='black',label='State 0')
plt.plot(hist[10],'-',c='yellow',label='State 10')
plt.plot(hist[20],'-',c='blue',label='State 20')
plt.plot(hist[30],'-',c='green',label='State 30')
plt.plot(hist[40],'-',c='red',label='State 40')
plt.legend()

plt.show()

# For one time scale- change objective function (include min of that set)
# For two time scale and one output of NN- change Q_target_next