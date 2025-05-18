# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 08:26:20 2024

@author: Harshit
"""

#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
from collections import defaultdict
import csv
from numpy import random
from mpmath import mp
import time 
from copy import deepcopy
import copy 
import matplotlib.pyplot as plt
# no. of tasks = phi
# no. of states = S
# current task = task (0 or 1)
# current state = s (a list of cardinality phi)
# continue action = 0
# restart action = 1

# In[97]:


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
        if action == 0:
            next_state[action]= np.random.choice(2,p=self.P_zero[s[action]])
        else:
            next_state[action]= np.random.choice(2,p=self.P_one[s[action]])
        reward = self.rewards[s[action]][next_state[action]]
        return next_state,reward
            



# In[98]:


class Agent():
    def __init__(self,alpha,T,gamma):
        self.Q_values = np.zeros((2,2,2,2))
        self.phi = 2
        self.alpha = alpha
        self.T = T
        self.gamma = gamma
    
    def activate_task(self,s,T):        
        summ = 0 
        p = [0]*len(s)
        prob = [0]*len(s)
        for i in range(len(s)):
            summ += mp.exp((self.Q_values[s[i]][0][s[i]])/T)
            p[i] = mp.exp((self.Q_values[s[i]][0][s[i]])/T)
        for i in range(len(s)):
            prob[i] = p[i]/summ
        task = random.choice(5,1,p=prob)
        return task[0]
    
    def activate_task_eps_greedy(self,s,epsilon):
        r = random.random()
        if r < epsilon:
            action = np.random.choice(2)
        else:
            gre = [0]*len(s)
            for i in range(len(s)):
                gre[i] = self.Q_values[s[i]][0][s[i]][i]
            action = np.argmax(gre)
        return action
    
    def check_best_action(self,state):
        if state[1] == 1:
            return 1
        elif state[0] == 0:
            return 0
        elif state[0] == 1:
            return 0 
        else:
            return 1 

        
    def act_greedy(self,s,restart_prob,task):
        return max((self.Q_values[s[task]][0][restart_prob][task]),(self.Q_values[s[task]][1][restart_prob][task]))
    
    def update(self,s,next_state,reward,task):
        for k in range(2):
            self.Q_values[s[task]][0][k][task] = (1-self.alpha)*(self.Q_values[s[task]][0][k][task]) + self.alpha*(reward+self.gamma*(self.act_greedy(next_state,k,task)))
            self.Q_values[k][1][s[task]][task] = (1-self.alpha)*(self.Q_values[k][1][s[task]][task]) + self.alpha*(reward+self.gamma*(self.act_greedy(next_state,s[task],task)))
    
# In[99]:


hist00 = []
hist01 = []
hist10 = []
hist11 = []
t = []
plt_wrong_actions = []
class pull():
        start_time = time.time()
        env = envir()
        agent = Agent(alpha=0.2,T=1000,gamma=0.9)
        episode = 10000
        agent.alpha = 0.2
        agent.gamma = 0.9
        Tmax = 90000
        Tmin = 0.5
        phi = 3
        T = Tmax
        b = 0.998
        eps = 0.2
        cumm_wrong_steps = []
        for ep_no in range(episode):
            s = np.random.choice(2,2)
            #T = Tmin + b*(T - Tmin) 
            #print(T)
            #task = agent.activate_task(s,T)
            #agent.alpha = agent.alpha - 0.00002
            task = agent.activate_task_eps_greedy(s,eps)
            task_opt = agent.check_best_action(s)
            if task != task_opt:
                cumm_wrong_steps.append(1)
            else:
                cumm_wrong_steps.append(0)
                
            plt_wrong_actions.append(np.mean(cumm_wrong_steps)*100)
            current_time = time.time()-start_time
            next_state,reward = env.step(s,task)
            agent.update(s,next_state,reward,task)
            t.append(current_time)
            hist00.append(0.1*agent.Q_values[0][0][0][0])  
            hist01.append(0.1*agent.Q_values[1][0][1][0])              
            hist10.append(0.1*agent.Q_values[0][0][0][1])
            hist11.append(0.1*agent.Q_values[1][0][1][1])
            s = next_state
        csv_file_path = 'restart_toy.csv'
        print("CWS: ",sum(cumm_wrong_steps))
        # Write the two lists to the CSV file side by side
        
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['State0','State1','State2','State3'])  # Optional header row
            csv_writer.writerows(zip(hist00, hist01, hist10, hist11))
        '''
        print(f"The two lists have been saved to {csv_file_path}")
        
        for x in range(5):
            print(agent.Q_values[x][0][x],agent.Q_values[x][1][x])
        #print(current_time)
        '''
        
        filename = 'percent_wrong_restart.csv'

        # Writing to CSV file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['percent_wrong'])  # Writing the header
            for value in plt_wrong_actions:
                writer.writerow([value])  # Writing each value in a new row
        


plt.figure(figsize=(6,6))
plt.title('Restart-in-state',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Gittins index',fontsize = 'xx-large')
plt.plot(hist00,'-',c='blue',label='Task 0 / State 0')
plt.plot(hist01,':',c='green',label='Task 0 / State 1')
plt.plot(hist10,'--',c='red',label='Task 1 / State 0')
plt.plot(hist11,'-.',c='black',label='Task 1 / State 1')
plt.legend()
plt.show()


plt.title('Q(s)(c)(s) vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q(s)(c)(s)',fontsize = 'xx-large')
plt.plot(plt_wrong_actions,'-',c='blue',label='State 0')
plt.legend()
plt.show()

# In[80]:


# In[ ]:





# In[ ]:


