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



# In[116]:


def get_reward(state,action):
    rewards = []
    for i in range(5):
        if action[i]==1:
            if state[i]==0:
                rewards.append(0.9)
            elif state[i]==1:
                rewards.append(0.9**2)
            elif state[i]==2:
                rewards.append(0.9**3)
            elif state[i]==3:
                rewards.append(0.9**4)
            elif state[i]==4:
                rewards.append(0.9**5)
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
        p = [0,0];
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
            
            
    


# In[117]:

hist = [[[],[],[],[],[],[]]]*3
whittles_value_data = []
best_HPs = []

def game(qlr,wlr,phi,den,ct):
    gamma = 0.9
    episodes = 25000
    beta = wlr
    state = np.random.choice(5,5)
    for episode_no in range(episodes):
        #state = np.random.choice(5,5)
        hist0.append(W[0])
        hist1.append(W[1])
        hist2.append(W[2])
        hist3.append(W[3])
        hist4.append(W[4])
        
        
        # Scheduling learning rates
        if episode_no==0:
            learning_rate = 1
        if episode_no>=1:
            learning_rate = qlr / math.ceil(episode_no / den)
            if(episode_no%phi==0):
                beta = wlr / (1 + math.ceil(episode_no * math.log(episode_no) / den))
            else:
                beta = 0
        
        if any(w>10 for w in W):
            plt.figure(figsize=(6,6))
            print("OUT")
            plt.title(f'QWI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, den = {den}',fontsize='xx-large')
            plt.xlabel('Time step', fontsize = 'xx-large')
            plt.ylabel('W',fontsize = 'xx-large')
            plt.plot(hist4,'-',c='yellow',label='State 4')
            plt.plot(hist3,'-',c='black',label='State 3')
            plt.plot(hist2,'-',c='blue',label='State 2')
            plt.plot(hist1,'-',c='green',label='State 1')
            plt.plot(hist0,'-',c='red',label='State 0')
            plt.legend()

            plt.savefig(f'Plots/QWI_finaltoy/HP/plot_{qlr}_{wlr}_{phi}_{den}.png')
            plt.clf()
            plt.close()
            return
        
        # Transition
        arm = choose_arm(s=state,We=W,epsilon=1)
        #epsilon = max(0.1,epsilon*0.999)
        action = arm
        next_state = transition(state,action)
        reward = get_reward(state,action)
        

        for i in range(5):
            for k in range(5):
                Q[state[i]][action[i]][k] += learning_rate*((1-action[i])*(W[k])+action[i]*reward[i]+gamma*(max(Q[next_state[i]][0][k],Q[next_state[i]][1][k]))-Q[state[i]][action[i]][k])
        
        for k in range(5):
            W[k] += beta*(Q[k][1][k]-Q[k][0][k])

        
        state = next_state
        
    sum0 = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0 
    for i in range(1,201):
        sum0 += hist0[-i]
        sum1 += hist1[-i]
        sum2 += hist2[-i]
        sum3 += hist3[-i]
        sum4 += hist4[-i]
    processed_W = [sum0/200,sum1/200,sum2/200,sum3/200,sum4/200]
    

    
    if abs(processed_W[0] - 0.9) <= 0.02 and abs(processed_W[1] - 0.819) <= 0.02 and abs(processed_W[2] - 0.74804) <= 0.02 and abs(processed_W[3] - 0.6909) <= 0.02 and abs(processed_W[4] - 0.64565) <= 0.02:
                print("YES")
                HP_dict = {"qlr":qlr,"wlr":wlr,"phi":phi,"den":den,"W[0]":processed_W[0],"W[1]":processed_W[1],"W[2]":processed_W[2],"W[3]":processed_W[3],"W[4]":processed_W[4]}
                best_HPs.append(HP_dict)
                ct = 1

    
    plt.figure(figsize=(6,6))
    plt.title(f'QWI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, den = {den}, ct = {ct}',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('W',fontsize = 'xx-large')
    plt.plot(hist4,'-',c='yellow',label='State 4')
    plt.plot(hist3,'-',c='black',label='State 3')
    plt.plot(hist2,'-',c='blue',label='State 2')
    plt.plot(hist1,'-',c='green',label='State 1')
    plt.plot(hist0,'-',c='red',label='State 0')
    plt.legend()

    plt.savefig(f'Plots/QWI_finaltoy/HP/plot_{qlr}_{wlr}_{phi}_{den}.png')
    plt.clf()
    plt.close()
    

        

# In[118]:

df = pd.read_csv('QWI_HP_data_final_toy.csv')

# Extract the columns into lists
QLRs = 
WLRs = 
MODs = 
dens = 
cts = [0]*6
for qlr in tqdm(range(len(QLRs))):
                        Q = np.zeros((5,2,5))
                        
                        hist0 = []
                        hist1 = []
                        hist2 = []
                        hist3 = []
                        hist4 = []
        
                        W = np.zeros(5)
        
                        hist = [[[],[],[],[],[],[]]]*5
                        game(QLRs[qlr],WLRs[qlr],MODs[qlr],dens[qlr],cts[qlr])


print(best_HPs)

with open('QWI_HP_data_final_toy_refined.csv', 'w', newline='') as csv_file:
    # Assuming all dictionaries have the same keys
    fieldnames = best_HPs[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()  # Write the header
    writer.writerows(best_HPs)  # Write the data










