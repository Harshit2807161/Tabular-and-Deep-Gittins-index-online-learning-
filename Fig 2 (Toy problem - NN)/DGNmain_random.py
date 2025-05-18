# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:39:15 2024

@author: Harshit
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import DGNnn_new as DGNnn
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

hist = [[] for _ in range(50)]

M = np.zeros(50)

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
    np.save("C:\\Intern\\Gittin's plots\\All codes directory\\Fig 2 (Toy problem - NN)\\P1_matrix.npy", P1)  
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


import time
def game():
    start_time = time.time()
    gamma = 0.9
    epsilon = 0.4
    episodes = 10000
    rate = 1
    learning_rate = 0.1
    beta = 0.2
    agent = DGNnn.Agent(2,2,-1)
    eps = 1
    state = np.random.choice(50,5)
    for episode_no in tqdm(range(episodes)):
        if episode_no % 20 == 0:
            state = np.random.choice(50,5)
        if episode_no==0:
            learning_rate = 1
        if episode_no>=1:
            learning_rate = 1 / math.ceil(episode_no / 5000)
            if(episode_no%10==0):
                t = episode_no
                beta = 0.2 / (1 + math.ceil(t* math.log(t) / 1000))
            else:
                beta = 0
        
        #eps = max(eps*0.9995,0.1)
        arm = select_action(copy.copy(state),M,1)
        action = arm
        next_state = transition(copy.copy(state),action)
        # next_state = transition_uniform(copy.copy(state),action)
        reward = get_reward(copy.copy(state),action)
        '''print(state)
        print(next_state)
        print(reward)
        print(action)'''     
        for i in range(5):
            if action[i]==1:
                for k in range(50):
                        agent.step(state[i], action[i], reward[i], next_state[i], False, M)

        for k in range(50):
            input_tensor = torch.FloatTensor(np.array([k,k]))
            output_values = agent.qnetwork_local.forward(input_tensor)
            M[k]+=beta*(output_values[1]-M[k])
        current_time = time.time()-start_time
        
        for i in range(50):
            hist[i].append(0.1*M[i])
        
        state = next_state
    
    print(current_time)
        
    print(M)
    '''
    # Specify the CSV file path
    csv_file_path = 'DGN_toy.csv'
    
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2','State3','State4'])  # Optional header row
        csv_writer.writerows(zip(hist0, hist1, hist2, hist3, hist4))

    print(f"The two lists have been saved to {csv_file_path}")
    '''
    
game()


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