# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 08:26:20 2024

@author: Harshit
"""

#!/usr/bin/env python
# coding: utf-8
#Deteriorating arms - 
#for restart paper toy problem. 3 states, 2 arms.
#0,1,2 states 2 has highest gittins. so when you are passive you setup TPM such that you can only visit worse off (lower gittins) states. if 0,1,2 are in increasing gittins, you can only visit 0 from 1 if that arm is passive.
# In[114]:


import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import random
import csv
# In[115]:


#initialise-
Q = np.zeros((2,2,2,2))

hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []


hist00 = []
hist01 = []
hist10 = []
hist11 = []

W = np.zeros((2,2))


# Define the default factory function
def default_float_value():
    return 0.0

# Create the defaultdict
V = defaultdict(default_float_value)
V_true = defaultdict(default_float_value)

def calculate_bellman_final(state,W):  
    s_full = tuple(state)
    v_vals = [0]*5
    actions = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
    for i in range(len(state)):
        reward = get_reward(state, actions[i])
        next_state,prob = transition(state,actions[i])
        ns_full = tuple(next_state)
        v_vals[i] = next(x for x in reward if x != 0) + 0.9*(prob)*(V[ns_full])
        
    #V[s_full] = np.max(v_vals)
    
    whittle_state = []
    for i in range(len(state)):
        whittle_state.append(W[state[i]])
    max_arm = np.argmax(whittle_state)
    
    V[s_full] = v_vals[max_arm]


def bellman_relative_error(V, V_true):
    relative_errors = []
    for s_full in V_true.keys():
        if V_true[s_full] !=0:
            relative_errors.append(np.abs((V[s_full] - V_true[s_full])) / V_true[s_full])
        else:
            relative_errors.append(0)
    return np.mean(relative_errors)

reg = []
def regret(s, next_state):
    change = np.subtract(next_state, s)
    #print("s_input",s)
    #print("s_output",next_state)
    #print("change",change)
    if np.nonzero(change)[0] == np.argmax(s):
        reg.append(1)
        #print("append 1")
    else:
        reg.append(0)
        #print("append 0")

# In[116]:
from copy import deepcopy 

class envir():
    def __init__(self):
        self.P_zero =[(0.3,0.7),
                      (0.7,0.3)]
        self.P_one = [(0.9,0.1),
                      (0.1,0.9)]
        self.rewards = [(1,10),
                        (1,10)]
    def step(self,s,action):
        next_state = deepcopy(s)
        reward = [0]*len(s)
        
        if action == 0:
            next_state[action]= np.random.choice(2,p=self.P_zero[s[action]])
        else:
            next_state[action]= np.random.choice(2,p=self.P_one[s[action]])
        reward[action] = self.rewards[s[action]][next_state[action]]
        return next_state,reward
    
    

def select_action(s,M,epsilon):
    r = random.random()
    if r < epsilon:
        action = np.random.choice(2)
    else:
        gre = [0]*len(s)
        for i in range(len(s)):
            gre[i] = M[s[i]][i]
            
        action = np.argmax(gre)
        
    return action

            
def check_best_action(state):
    if state[1] == 1:
        return 1
    elif state[0] == 0:
        return 0
    elif state[0] == 1:
        return 0 
    else:
        return 1 


    


# In[117]:

hist = [[[],[],[],[],[],[],[],[],[],[]]]*5
print(hist)

histV = []
BRE = []
plt_wrong_actions = []
def game():
    gamma = 0.9
    episodes = 10000
    rate = 0.3
    env = envir()
    epsilon = 0.2
    state = np.random.choice(2,2)
    cumm_wrong_steps = []
    '''
    for episode_no in range(100000):   
        s_full = tuple(state)
        v_vals = [0]*5
        state = np.random.choice(5,5)
        actions = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        for i in range(len(state)):
            reward = get_reward(state, actions[i])
            next_state,prob = transition(state,actions[i])
            ns_full = tuple(next_state)
            v_vals[i] = next(x for x in reward if x != 0) + 0.9*(prob)*(V_true[ns_full])
            
        V_true[s_full] = np.max(v_vals)
        
    print(V_true)
    state = np.random.choice(5,5)
    episode_no = 0
    '''
    
    for episode_no in range(episodes):
        state = np.random.choice(2,2)
    
        hist00.append(W[0][0])
        hist01.append(W[1][0])
        hist10.append(W[0][1])
        hist11.append(W[1][1])
        
        if episode_no==0:
            learning_rate = 0.3
            beta = 0.2
        if episode_no>=1:
            learning_rate = 0.3 / math.ceil(episode_no / 5000)
            if(episode_no%20==0):
                beta = 0.2 / (1 + math.ceil(episode_no * math.log(episode_no) / 5000))
            else:
                beta = 0
        
        task = select_action(s=state,M=W,epsilon=epsilon)
        #epsilon = max(0.1,epsilon*0.999)
        action = [0]*len(state)
        action[task] = 1
        task_opt = check_best_action(state)
        if task != task_opt:
            cumm_wrong_steps.append(1)
        else:
            cumm_wrong_steps.append(0)
        next_state, reward = env.step(state,task)
        plt_wrong_actions.append(np.mean(cumm_wrong_steps)*100)
       # calculate_bellman_final(state,W)

        for i in range(2):
            for k in range(2):
                Q[state[i]][action[i]][k][i] += learning_rate*((1-action[i])*(W[k][i])+action[i]*reward[i]+gamma*(max(Q[next_state[i]][0][k][i],Q[next_state[i]][1][k][i]))-Q[state[i]][action[i]][k][i])
        
        for i in range(2):
            for j in range(2):
                W[j][i] += beta*(Q[j][1][j][i]-Q[j][0][j][i])
        
        
        state = next_state
        '''
        #V_values = calculate_v_values(Q0, Q1)
        BRE.append(bellman_relative_error(V, V_true))
        #histV.append(V_values)
        regret(state,next_state)
        '''
    csv_file_path = 'QWI_toy.csv'
    print("CWS: ",sum(cumm_wrong_steps))
    
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2','State3'])  # Optional header row
        csv_writer.writerows(zip(hist00, hist01, hist10, hist11))

    print(f"The two lists have been saved to {csv_file_path}")
    
    
    filename = 'percent_wrong_QWI.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['percent_wrong'])  # Writing the header
        for value in plt_wrong_actions:
            writer.writerow([value])  # Writing each value in a new row
    
    
            
    
        

# In[118]:


game()
'''

data = {
    'W[0]': hist0,
    'W[1]': hist1,
    'W[2]': hist2,
    'W[3]': hist3,
    'W[4]': hist4
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('best_run_QWI.csv', index=False)
# In[119]:

'''
print(W)

"""
plt.figure(figsize=(6,6))
plt.title('Q values for k = 0',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[0][0],'-',c='blue',label='Q[0][0][0]')
plt.plot(hist[0][1],'-',c='green',label='Q[1][0][0]')
plt.plot(hist[0][2],'-',c='red',label='Q[2][0][0]')
plt.plot(hist[0][3],'-',c='cyan',label='Q[0][1][0]')
#plt.plot(hist[0][4],'-',c='orange',label='Q[1][1][0]')
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
"""

# In[120]:


plt.figure(figsize=(6,6))
plt.title('QWI',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Gittins index',fontsize = 'xx-large')
plt.plot(hist00,'-',c='blue',label='Task 0 / State 0')
plt.plot(hist01,':',c='green',label='Task 0 / State 1')
plt.plot(hist10,'--',c='red',label='Task 1 / State 0')
plt.plot(hist11,'-.',c='black',label='Task 1 / State 1')
plt.legend()
plt.show()


plt.figure(figsize=(6,6))
plt.title('QWI',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('W',fontsize = 'xx-large')
plt.plot(plt_wrong_actions,'-',c='blue',label='State 0')
plt.legend()

plt.show()

# In[113]:


# In[ ]:


'''
print("Data has been written to output.csv")

#print(histV[len(histV)-1])

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



# In[ ]:

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

'''

