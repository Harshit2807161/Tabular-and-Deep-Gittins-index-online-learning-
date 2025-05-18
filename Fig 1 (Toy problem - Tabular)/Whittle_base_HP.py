#!/usr/bin/env python
# coding: utf-8
#Deteriorating arms - 
#for restart paper toy problem. 3 states, 2 arms.
#0,1,2 states 2 has highest gittins. so when you are passive you setup TPM such that you can only visit worse off (lower gittins) states. if 0,1,2 are in increasing gittins, you can only visit 0 from 1 if that arm is passive.
# In[114]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tqdm import tqdm
# In[115]:


#initialise-
action = np.zeros(2)
reward = np.zeros(2)
state = np.zeros(2)
arm = np.zeros(2)
Q = np.zeros((3,2,3))

hist0 = []
hist1 = []
hist2 = []

W = np.zeros(3)



# In[116]:


def get_reward(state,action):
    rewards = []
    for i in range(len(state)):
        if action[i]==1:
            rewards.append((0.9)**(state[i]+1))
        else:
            rewards.append(0)
    return rewards


def transition(state,action,ct0,ct1,ct2):
    P1 = [(0.1,0.9,0),
          (0.1,0,0.9),
          (0.1,0,0.9)]
    P0 = [(1,0,0),
          (0,1,0),
          (0,0,1)]
    next_statee = []
    for i in range(len(state)):
        if action[i]==1:
            next_statee.append(np.random.choice(3,p=P1[state[i]]));
            if state[action[i]]==1:
                ct1+=1
            elif state[action[i]]==0:
                ct0+=1
            else:
                ct2+=1
        else:
            next_statee.append(np.random.choice(3,p=P0[state[i]]));
    return next_statee,ct0,ct1,ct2

def choose_arm(s,We,epsilon,num_0,num_1,num_2):
    wl= []
    p = np.array([0,0])
    x = 0 
    if np.random.random() < epsilon:
        x = np.random.choice(2)
        p[x]=1
        p = p.tolist()
        if s[x] == 0:
            num_0+=1
        elif s[x] == 1:
            num_1+=1
        else:
            num_2+=1
        return p,num_0,num_1,num_2
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
            
    


# In[117]:

hist = [[[],[],[],[],[],[]]]*3
whittles_value_data = []
best_HPs = []

def game(qlr,wlr,phi,den):
    gamma = 0.9
    ct0,ct1,ct2=0,0,0
    epsilon = 1
    state = np.array([0,0])
    episodes = 10000
    beta = 1
    epsilon = 1
    num_0,num_1,num_2 = 0,0,0
    for episode_no in range(episodes):
        
        
        # Tracking whittles estimates
        hist0.append(W[0])
        hist1.append(W[1])
        hist2.append(W[2])
        
        # Scheduling learning rates
        if episode_no==0:
            learning_rate = 1
        if episode_no>=1:
            learning_rate = qlr / math.ceil(episode_no / den)
            if(episode_no%phi==0):
                beta = wlr / (1 + math.ceil(episode_no * math.log(episode_no) / den))
            else:
                beta = 0
        
        # Transition
        arm,num_0,num_1,num_2 = choose_arm(s=state,We=W,epsilon=1,num_0 = num_0,num_1 = num_1, num_2=num_2)
        epsilon = max(0.1,epsilon*0.999)
        action = arm
        next_state,ct0,ct1,ct2 = transition(state,action,ct0,ct1,ct2)
        reward = get_reward(state,action)
        
        # Updates
        for i in range(2):
            for k in range(3):
                Q[state[i]][action[i]][k] += learning_rate*((1-action[i])*(W[k])+action[i]*reward[i]+gamma*(max(Q[next_state[i]][0][k],Q[next_state[i]][1][k]))-Q[state[i]][action[i]][k])
                
        for k in range(3):
            W[k] += beta*(Q[k][1][k]-Q[k][0][k])

        
        # Tracking Q-values
        for i in range(3):
            for j in range(3):
                    hist[i][j].append(Q[j][0][i])
                    hist[i][j+3].append(Q[j][1][i])
        
        
        
        state = next_state
        
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(1,201):
        sum0 += hist0[-i]
        sum1 += hist1[-i]
        sum2 += hist2[-i]
    processed_W = [sum0/200,sum1/200,sum2/200]
    
    if abs(processed_W[0] - 0.9) <= 0.025 and abs(processed_W[1] - 0.815) <= 0.025 and abs(processed_W[2] - 0.7511) <= 0.025:
        print("Yes")
        HP_dict = {"qlr":qlr,"wlr":wlr,"phi":phi,"den":den,"W[0]":processed_W[0],"W[1]":processed_W[1],"W[2]":processed_W[2]}
        best_HPs.append(HP_dict)
    
    whittles_value_data.append(processed_W)
    plt.figure(figsize=(6,6))
    plt.title(f'QWI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, den = {den}',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('W',fontsize = 'xx-large')
    plt.plot(hist2,'-',c='blue',label='State 2')
    plt.plot(hist1,'-',c='green',label='State 1')
    plt.plot(hist0,'-',c='red',label='State 0')
    plt.legend()

    plt.savefig(f'Plots/plot_{qlr}_{wlr}_{phi}_{den}_10000.png')
    plt.clf()
    plt.close()

        

# In[118]:

QLRs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
WLRs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
MODs = [10,25,50,75,100,200]
dens = [100,500,1000,2500,5000,7500,10000]

for qlr in tqdm(range(len(QLRs))):
    for wlr in WLRs:
        for phi in MODs:
            for den in dens:
                action = np.zeros(2)
                reward = np.zeros(2)
                state = np.zeros(2)
                arm = np.zeros(2)
                Q = np.zeros((3,2,3))

                hist0 = []
                hist1 = []
                hist2 = []

                W = np.zeros(3)
                hist = [[[],[],[],[],[],[]]]*3
                game(QLRs[qlr],wlr,phi,den)
                
df = pd.DataFrame(whittles_value_data, columns=['state 0', 'state 1', 'state 2'])
df.to_csv('whittles_data_10000.csv', index=False)

# In[119]:


print(W)


plt.figure(figsize=(6,6))
plt.title('Q values for k = 0',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[0][0],'-',c='blue',label='Q[0][0][0]')
plt.plot(hist[0][1],'-',c='green',label='Q[1][0][0]')
plt.plot(hist[0][2],'-',c='red',label='Q[2][0][0]')
plt.plot(hist[0][3],'-',c='cyan',label='Q[0][1][0]')
plt.plot(hist[0][4],'-',c='orange',label='Q[1][1][0]')
plt.plot(hist[0][5],'-',c='black',label='Q[2][1][0]')
plt.legend()


plt.figure(figsize=(6,6))
plt.title('Q values for k = 1',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[1][0],'-',c='blue',label='Q[0][0][1]')
plt.plot(hist[1][1],'-',c='green',label='Q[1][0][1]')
plt.plot(hist[1][2],'-',c='red',label='Q[2][0][1]')
plt.plot(hist[1][3],'-',c='cyan',label='Q[0][1][1]')
plt.plot(hist[1][4],'-',c='orange',label='Q[1][1][1]')
plt.plot(hist[1][5],'-',c='black',label='Q[2][1][1]')
plt.legend()


plt.figure(figsize=(6,6))
plt.title('Q values for k = 2',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[2][0],'-',c='blue',label='Q[0][0][2]')
plt.plot(hist[2][1],'-',c='green',label='Q[1][0][2]')
plt.plot(hist[2][2],'-',c='red',label='Q[2][0][2]')
plt.plot(hist[2][3],'-',c='cyan',label='Q[0][1][2]')
plt.plot(hist[2][4],'-',c='orange',label='Q[1][1][2]')
plt.plot(hist[2][5],'-',c='black',label='Q[2][1][2]')


plt.legend()


# In[120]:


plt.figure(figsize=(6,6))
plt.title('QWI',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('W',fontsize = 'xx-large')
plt.plot(hist2,'-',c='blue',label='State 2')
plt.plot(hist1,'-',c='green',label='State 1')
plt.plot(hist0,'-',c='red',label='State 0')
plt.legend()

plt.show()


# In[113]:


print("Whittle's index for state 0",hist0[len(hist0)-1])
print("Whittle's index for state 1",hist1[len(hist1)-1])
print("Whittle's index for state 2",hist2[len(hist2)-1])

print(best_HPs)

with open('HP_data_10000.csv', 'w', newline='') as csv_file:
    # Assuming all dictionaries have the same keys
    fieldnames = best_HPs[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()  # Write the header
    writer.writerows(best_HPs)  # Write the data
# In[ ]:










