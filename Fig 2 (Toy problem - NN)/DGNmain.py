import numpy as np
import math
import matplotlib.pyplot as plt
import DGNnn
import torch
import copy
import csv
from tqdm import tqdm 
#initialise-
action = np.zeros(2)
reward = np.zeros(2)
state = np.zeros(2)
arm = np.zeros(2)
Q = np.zeros((3,2,3))

hist0 = []
hist1 = []
hist2 = []

M = np.zeros(3)

def get_reward(state,action):
    rewards = []
    for i in range(2):
        if action[i]==1:
            if state[i]==0:
                rewards.append(0.9)
            elif state[i]==1:
                rewards.append(0.81)
            else:
                rewards.append(0.729)
        else:
            rewards.append(0)
    return rewards

def transition(state,action):
    P1 = [(0.1,0.9,0),
          (0.1,0,0.9),
          (0.1,0,0.9)]
    P0 = [(1,0,0),
          (0,1,0),
          (0,0,1)]
    next_statee = []
    for i in range(2):
        if action[i]==1:
            next_statee.append(np.random.choice(3,p=P1[state[i]]));
        else:
            next_statee.append(np.random.choice(3,p=P0[state[i]]));
    return next_statee

def choose_arm(s,We,epsilon):
    wl= []
    p = np.array([0,0])
    x = 0 
    if np.random.random() < epsilon:
        x = np.random.choice(2)
        p[x]=1
        p = p.tolist()
        return p
    else:
        indv =0;
        ind = 0;
        p = [0,0];
        for i in range(2):
            wl.append(We[s[i]])
        for i in range(2):
            indv = max(wl[i],indv)
        for i in range(2):
            if(indv==wl[i]):
                ind = i;
                break
        p[ind]=1
        return p
            
def game():
    gamma = 0.9
    epsilon = 0.4
    state = np.array([0,0])
    episodes = 3500
    rate = 1
    beta = 1
    agent = DGNnn.Agent(2,2,-1)
    eps = 1
    state = np.random.choice(3,2)
    for episode_no in tqdm(range(episodes)):
        state = np.random.choice(3,2)
        if episode_no==0:
            learning_rate = 0.1
        if episode_no>=1:
            learning_rate = 1 / math.ceil(episode_no / 5000)
            if(episode_no%20==0):
                beta = 0.7
            else:
                beta = 0
        eps = max(eps*0.9995,0.1)
        arm = choose_arm(s=copy.copy(state),We=M,epsilon=eps)
        action = arm
        next_state = transition(copy.copy(state),action)
        reward = get_reward(copy.copy(state),action)
        '''print(state)
        print(next_state)
        print(reward)
        print(action)'''
        if episode_no%10==0:
            rate = rate - 0.00002        
        for i in range(2):
            if action[i]==1:
                for k in range(3):
                        agent.step(np.array([state[i],k]), action[i], reward[i], np.array([next_state[i],k]), False,M[k])

        for k in range(3):
            input_tensor = torch.FloatTensor(np.array([k,k]))
            output_values = agent.qnetwork_local.forward(input_tensor)
            M[k]+=beta*(output_values[1]-M[k])
        hist0.append(0.1*M[0])
        hist1.append(0.1*M[1])
        hist2.append(0.1*M[2])
    print(M)
    
    # Specify the CSV file path
    csv_file_path = 'DGN_toy.csv'
    
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2'])  # Optional header row
        csv_writer.writerows(zip(hist0, hist1, hist2))

    print(f"The two lists have been saved to {csv_file_path}")

    
game()


plt.figure(figsize=(6,6))
plt.title('M vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('M',fontsize = 'xx-large')
plt.plot(hist2,'-',c='blue',label='State 2')
plt.plot(hist1,'-',c='green',label='State 1')
plt.plot(hist0,'-',c='red',label='State 0')
plt.legend()

plt.show()

# For one time scale- change objective function (include min of that set)
# For two time scale and one output of NN- change Q_target_next