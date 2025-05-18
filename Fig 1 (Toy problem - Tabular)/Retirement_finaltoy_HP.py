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
import random
import csv
from tqdm import tqdm
from copy import deepcopy


# In[116]:



class envir():
    def __init__(self):
        self.P =[(0.1,0.9,0,0,0),
                (0.1,0,0.9,0,0),
                (0.1,0,0,0.9,0),
                (0.1,0,0,0,0.9),
                (0.1,0,0,0,0.9)]
        self.rewards = np.array([0.9,0.9**2,0.9**3,0.9**4, 0.9**5])
    def step(self,s,action):
        next_state = deepcopy(s)
        #print(s)
        #print(action)
        next_state[action]= np.random.choice(5,p=self.P[s[action]])
        reward = self.rewards[s[action]]
        return next_state,reward
            

class Agent():
    def __init__(self,alpha,gamma):
        self.Q_values = np.zeros((5,5))
        self.alpha = alpha
        #self.c= c
        self.gamma = gamma
        
    def select_action(self,s,M,epsilon):
        r = random.random()
        if r < epsilon:
            action = np.random.choice(5)
        else:
            gre = [0]*len(s)
            for i in range(len(s)):
                gre[i] = M[s[i]]
                
            action = np.argmax(gre)
            
        return action
            
    


# In[117]:

hist = [[[],[],[],[],[],[]]]*3
whittles_value_data = []
best_HPs = []

def game(qlr,wlr,phi,den):
    beta = wlr
    t = 0
    epsilon = 1
    env = envir()
    s = np.random.choice(5,5)
    while True:
        
        hist0.append(0.1*M[0])
        hist1.append(0.1*M[1])
        hist2.append(0.1*M[2])
        hist3.append(0.1*M[3])
        hist4.append(0.1*M[4])
        
        if phi!= 0:
            # Scheduling learning rates
            if t==0:
                learning_rate = qlr
            if t>=1:
                learning_rate = qlr / math.ceil(t / den)
                if(t%phi==0):
                    beta = wlr / (1 + math.ceil(t * math.log(t) / den))
                else:
                    beta = 0
                    
        else:
            if t==0:
                learning_rate = 1
            if t>=1:
                learning_rate = qlr / math.ceil(t / den)
                beta = wlr / (1 + math.ceil(t * math.log(t) / den))
        
        t = t+1
        #Calculating Q values over state space for a given arm for the given M vector through Q learning
        # for episode_no in range(1):
        action = agent.select_action(s,M,epsilon)
        #epsilon = max(0.1,epsilon*0.99)
        #print("epsilon", epsilon)
        

        next_state, R = env.step(s,action)
        for k in range(5):
            agent.Q_values[int(s[action])][k] += learning_rate*(R+agent.gamma*(max(M[k],agent.Q_values[int(next_state[action])][k]))-agent.Q_values[int(s[action])][k])    


        s = next_state

      #Update M
        #beta = 0.2
        for i in range(5):
            M[i] += beta*(agent.Q_values[i][i] - M[i])      
      #Stopping criteria
        if t>=20000:
            break

        
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
    
    if abs(processed_W[0] - 0.9) <= 0.01 and abs(processed_W[1] - 0.819) <= 0.01 and abs(processed_W[2] - 0.74804) <= 0.01 and abs(processed_W[3] - 0.6909) <= 0.01 and abs(processed_W[4] - 0.64565) <= 0.01:
            print("YES")
            HP_dict = {"qlr":qlr,"wlr":wlr,"phi":phi,"den":den,"M[0]":processed_W[0],"M[1]":processed_W[1],"M[2]":processed_W[2],"M[3]":processed_W[3],"M[4]":processed_W[4]}
            best_HPs.append(HP_dict)
    
    
    plt.figure(figsize=(6,6))
    plt.title(f'QWI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, den = {den}',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('W',fontsize = 'xx-large')
    plt.plot(hist4,'-',c='yellow',label='State 4')
    plt.plot(hist3,'-',c='black',label='State 3')
    plt.plot(hist2,'-',c='blue',label='State 2')
    plt.plot(hist1,'-',c='green',label='State 1')
    plt.plot(hist0,'-',c='red',label='State 0')
    plt.legend()

    plt.savefig(f'Plots/QGI_finaltoy/HP/plot_{qlr}_{wlr}_{phi}_{den}.png')
    plt.clf()
    plt.close()
    

        

# In[118]:

df = pd.read_csv('QGI_HP_data_final_toy.csv')

# Extract the columns into lists
QLRs = df['qlr'].tolist()
WLRs = df['wlr'].tolist()
MODs = df['phi'].tolist()
dens = df['den'].tolist()

for qlr in tqdm(range(len(QLRs))):
                    agent = Agent(alpha = 0.3,gamma = 0.9)
                    
                    hist0 = []
                    hist1 = []
                    hist2 = []
                    hist3 = []
                    hist4 = []
                    M = np.array([0,0,0,0,0], dtype=np.float64)
    
                    hist = [[[],[],[],[],[],[]]]*5
                    game(QLRs[qlr],WLRs[qlr],MODs[qlr],dens[qlr])
                


print(best_HPs)

with open('QGI_HP_data_final_toy_refined.csv', 'w', newline='') as csv_file:
    # Assuming all dictionaries have the same keys
    fieldnames = best_HPs[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()  # Write the header
    writer.writerows(best_HPs)  # Write the data
# In[ ]:



















