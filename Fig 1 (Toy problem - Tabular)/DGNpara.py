import numpy as np
import math
import matplotlib.pyplot as plt
import DGNnn
import torch
import copy
import csv
from tqdm import tqdm 
import time
import pandas as pd
#initialise-
action = np.zeros(2)
reward = np.zeros(2)
state = np.zeros(2)
arm = np.zeros(2)
Q = np.zeros((5,2,5))

hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []

M = np.zeros(5)

def calculate_v_values(Q_values_action1, Q_values_action2):
    Q_values = np.column_stack((Q_values_action1, Q_values_action2))
    V_values = np.max(Q_values, axis=1)
    return V_values

V_true = [8.99999135, 8.14648002, 7.45888848, 7.05713998, 6.7344719 ]

def bellman_relative_error(V_approx, V_true):
    nonzero_indices = V_true != 0
    if np.any(nonzero_indices):
        relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]) / V_true[nonzero_indices])
        return np.mean(relative_errors)
    else:
        return np.nan

reg = []
def regret(s, next_state):
    change = np.subtract(next_state, s)
    print("s_input",s)
    print("s_output",next_state)
    print("change",change)
    if np.nonzero(change)[0] == np.argmax(s):
        reg.append(1)
        print("append 1")
    else:
        reg.append(0)
        print("append 0")
    # for i in range(5):
    #     if (change[i] != 0) and s[i] != np.max(s):
    #         reg.append(0)
    #         break
    #     elif change.all() == 0:
    #         reg.append(0)
    #         break
    #     else:
    #         reg.append(1)
    #         break

def get_reward(state,action):
    rewards = []
    for i in range(5):
        if action[i]==1:
            if state[i]==0:
                rewards.append(0.9)
            elif state[i]==1:
                rewards.append(0.81)
            elif state[i]==2:
                rewards.append(0.729)
            elif state[i]==3:
                rewards.append(0.9**3)
            elif state[i]==4:
                rewards.append(0.9**4)
        else:
            rewards.append(0)
    return rewards

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

def choose_arm(s,We,epsilon):
    wl= []
    p = np.array([0,0,0,0,0])
    x = 0 
    if np.random.random() < epsilon:
        x = np.random.choice(5)
        p[x]=1
        p = p.tolist()
        return p
    else:
        indv =0;
        ind = 0;
        p = [0,0,0,0,0];
        for i in range(5):
            wl.append(We[s[i]])
        for i in range(5):
            indv = max(wl[i],indv)
        for i in range(5):
            if(indv==wl[i]):
                ind = i;
                break
        p[ind]=1
        return p
            
BRE = []
histV = []

def game():
    gamma = 0.9
    epsilon = 0.4
    state = np.array([0,0,0,0,0])
    episodes = 25000
    rate = 1
    beta = 1
    agent = DGNnn.Agent(2,2,-1)
    eps = 1
    state = np.random.choice(5,5)
    start_time = time.time()
    for episode_no in tqdm(range(episodes)):
        state = np.random.choice(5,5)
        if episode_no==0:
            learning_rate = 0.1
        if episode_no>=1:
            learning_rate = 1 / math.ceil(episode_no / 5000)
            if(episode_no%20==0):
                beta = 0.7
            else:
                beta = 0
        eps = max(eps*0.9995,0.1)
        # eps = eps * 0.999
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
        for i in range(5):
            if action[i]==1:
                for k in range(5):
                        agent.step(np.array([state[i],k]), action[i], reward[i], np.array([next_state[i],k]), False,M[k])
        Q0 = []
        Q1 = []
        for k in range(5):
            input_tensor = torch.FloatTensor(np.array([k,k]))
            output_values = agent.qnetwork_local.forward(input_tensor)
            M[k]+=beta*(output_values[1]-M[k])
            Q0.append(output_values[0].detach().numpy())
            Q1.append(output_values[1].detach().numpy())

        V_values = calculate_v_values(Q0, Q1)
        print("Vval: ", V_values)
        BRE.append(bellman_relative_error(V_values, V_true))
        histV.append(V_values)
        regret(state,next_state)

        hist0.append(0.1*M[0])
        hist1.append(0.1*M[1])
        hist2.append(0.1*M[2])
        hist3.append(0.1*M[3])
        hist4.append(0.1*M[4])

    end_time = time.time()
    print(1000*(end_time - start_time))
    print(M)
    
    # Specify the CSV file path
    csv_file_path = 'DGN_toy.csv'
    
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2','State3','State4'])  # Optional header row
        csv_writer.writerows(zip(hist0, hist1, hist2, hist3, hist4))

    print(f"The two lists have been saved to {csv_file_path}")

    
game()


plt.figure(figsize=(6,6))
plt.title('M vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('M',fontsize = 'xx-large')
plt.plot(hist0,'-',c='blue',label='State 0')
plt.plot(hist1,':',c='green',label='State 1')
plt.plot(hist2,'--',c='red',label='State 2')
plt.plot(hist3,'-.',c='black',label='State 3')
plt.plot(hist4,'-',c='yellow',label='State 4')
plt.legend()

plt.show()

# For one time scale- change objective function (include min of that set)
# For two time scale and one output of NN- change Q_target_next

print(histV[len(histV)-1])

plt.figure(figsize=(6,6))
plt.title('BRE vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('BRE',fontsize = 'xx-large')
plt.plot(BRE,'-',c='blue',label='BRE')
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
plt.title('V(s) vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('V(S)',fontsize = 'xx-large')
plt.plot(histV,'-',c='blue',label='reg')
plt.legend()
plt.show()


cumu_regret = []
cumu_regret.append(reg[0])
avg_cum_reg = []

for i in range(len(reg)):
    add = (cumu_regret[i-1] + reg[i])
    cumu_regret.append(add)
    avg_cum_reg.append(cumu_regret[i]/(i+1))

plt.figure(figsize=(6,6))
plt.title('V(s) vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('regret',fontsize = 'xx-large')
plt.plot(avg_cum_reg,'-',c='blue',label='reg')
plt.legend()
plt.show()


hist0
hist1
hist2
hist3
hist4
BRE
avg_cum_reg

data = {
    'DGN_0': hist0,
    'DGN_1': hist1,
    'DGN_2': hist2,
    'DGN_3': hist3,
    'DGN_4': hist4,
    'DGN_BRE': BRE,
    'DGN_reg': avg_cum_reg
}

df = pd.DataFrame(data)
df.to_csv('DGN_toy.csv', index=False)