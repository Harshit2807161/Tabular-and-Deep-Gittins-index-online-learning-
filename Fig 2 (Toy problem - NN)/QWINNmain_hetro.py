import numpy as np
import math
import matplotlib.pyplot as plt
import QWINNnn
import torch
import copy
from tqdm import tqdm 
import csv
import random
#initialise-
action = np.zeros(5)
reward = np.zeros(5)
state = np.zeros(5)
arm = np.zeros(5)
Q = np.zeros((5,2,5,5))
hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []
M = np.zeros((5,5))


def get_reward(state,action):
    rewards = []
    for i in range(5):
        if action[i]==1:
            if state[i]==0:
                rewards.append(0.9*(i+1))
            elif state[i]==1:
                rewards.append(0.81*(i+1))
            elif state[i]==2:
                rewards.append(0.729*(i+1))
            elif state[i]==3:
                rewards.append((0.9**(state[i]+1))*(i+1))
            else:
                rewards.append((0.9**(state[i]+1))*(i+1))
        else:
            rewards.append(0)
    return rewards

def calculate_v_values(Q_values_action1, Q_values_action2):
    V_values = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            V_values[i][j] = max(Q_values_action1[i][j],Q_values_action2[i][j])  # Use keepdims=True to maintain the 2D shape
    return V_values

G = np.array([[ 7.33592224, 18.90987015, 26.69592094, 35.69182587, 44.8101387 ],
 [ 7.33525562, 17.18063354, 24.65020943, 32.10348129, 39.66909409],
 [ 7.28526926, 15.45191193, 22.9175663,  30.34776306, 37.39906311],
 [ 7.09461737, 13.71846485, 21.17184258, 28.18720436, 35.12771225],
 [ 6.83058023, 13.32402802, 19.00160789, 25.93294907, 32.86553574]])

V_true = np.array([[ 9.96415329, 19.04043198, 27.26326752, 35.67977524, 44.4648819 ],
 [ 8.04321384, 17.43393517, 24.19148254, 32.01085663, 40.72146606],
 [ 6.76452732, 15.83143044, 22.5439682,  29.34253311, 36.97805786],
 [ 6.76441097, 14.2289257,  20.89645004, 27.69501877, 34.49358749],
 [ 6.76441097, 12.62641907, 19.24893379, 26.06292915, 32.84607697]])

V_true = 0.1*V_true

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
    best_arm = 0
    best_val = -float('inf')
    for arm in range(len(state)):
        s = state[arm]  # current state of this arm
        # Lookup the Gittins index:
        gittins_value = G[s][arm]
        if gittins_value > best_val:
            best_val = gittins_value
            best_arm = arm
    return best_arm



def transition(state,action):
    P1 = [(0.1,0.9,0,0,0),
            (0.1,0,0.9,0,0),
            (0.1,0,0,0.9,0),
            (0.1,0,0,0,0.9),
            (0.1,0,0,0,0.9)]
    P0 = [(1,0,0,0,0),
          (0,1,0,0,0),
          (0,0,1,0,0),
          (0,0,0,1,0),
          (0,0,0,0,1)]
    next_statee = []
    for i in range(5):
        if action[i]==1:
            next_statee.append(np.random.choice(5,p=P1[state[i]]));
        else:
            next_statee.append(np.random.choice(5,p=P0[state[i]]));
    return next_statee

def select_action(s,M,epsilon):
    r = random.random()
    if r < epsilon:
        action = np.random.choice(5)
    else:
        gre = [0]*len(s)
        for i in range(len(s)):
            gre[i] = M[s[i]][i]
            
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
    episodes = 5000
    start_time = time.time()
    rate = 1
    beta = 0.6
    eps = 1
    agent = QWINNnn.Agent(3,2,-1)
    state = np.random.choice(5,5)
    cumm_wrong_steps = []
    plt_wrong_actions = []
    BRE = []
    for episode_no in tqdm(range(episodes)):
        if episode_no==0:
            learning_rate = 0.1
        if episode_no>=1:
            learning_rate = 1 / math.ceil(episode_no / 5000)
            if(episode_no%10==0):
                t = episode_no
                beta = 0.2 / (1 + math.ceil(t* math.log(t) / 5000))
            else:
                beta = 0
        eps = max(eps*0.9995,0.1)
        arm = select_action(copy.copy(state),M,eps)
        action = arm
        next_state = transition(copy.copy(state),action)
        reward = get_reward(copy.copy(state),action)
        task = gettask(action)
        task_eps = task
        task_opt = check_best_action(state)
        if task_eps != task_opt:
            cumm_wrong_steps.append(1)
        else:
            cumm_wrong_steps.append(0)

        plt_wrong_actions.append(np.mean(cumm_wrong_steps)*100)
        Q0 = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                input_tensor = torch.FloatTensor(np.array([i,i,j]))
                output_values = agent.qnetwork_local.forward(input_tensor)
                Q0[i][j] = output_values[1] 
        V_values = calculate_v_values(Q0,M)
        BRE.append(bellman_relative_error(V_values,V_true))

        '''print(state)
        print(next_state)
        print(reward)
        print(action)'''
        for i in range(5):
            for k in range(5):
                agent.step(np.array([state[i],k,i]), action[i], reward[i], np.array([next_state[i],k,i]), False,M[k][i])
        for i in range(5):
            for k in range(5):
                input_tensor = torch.FloatTensor(np.array([k,k,i]))
                output_values = agent.qnetwork_local.forward(input_tensor)
                M[k][i] += beta*(output_values[1]-output_values[0])
        state = next_state
        hist0.append(M[0][0])
        hist1.append(M[1][1])
        hist2.append(M[2][2])
        hist3.append(M[3][3])
        hist4.append(M[4][4])
        
        current_time = time.time()-start_time
    print(current_time)
    print(V_values)
    # Specify the CSV file path
    csv_file_path = 'QWINN_toy.csv'
    
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2','State3','State4'])  # Optional header row
        csv_writer.writerows(zip(hist0, hist1, hist2,hist3,hist4))

    print(f"The two lists have been saved to {csv_file_path}")

    filename = 'C:\\Intern\\percent_wrong_QWINN_largetoy.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['percent_wrong'])  # Writing the header
        for value in plt_wrong_actions:
            writer.writerow([value])  # Writing each value in a new row
    
    filename = 'C:\\Intern\\BRE_QWINN_largetoy.csv'

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
plt.plot(hist4,'-',c='black',label='State 4')
plt.plot(hist3,'-',c='yellow',label='State 3')
plt.plot(hist2,'-',c='blue',label='State 2')
plt.plot(hist1,'-',c='green',label='State 1')
plt.plot(hist0,'-',c='red',label='State 0')
plt.legend()

plt.show()
