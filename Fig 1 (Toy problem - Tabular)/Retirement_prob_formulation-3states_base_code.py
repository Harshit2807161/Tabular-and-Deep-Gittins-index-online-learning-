#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import random
from copy import deepcopy,copy 
import csv
# In[121]:

reg = []

def regret(s, next_state):
    change = np.subtract(deepcopy(next_state), deepcopy(s))
    #print("s_input",s)
    #print("s_output",next_state)
    #print("change",change)
    if np.nonzero(change)[0] == np.argmax(s):
        reg.append(1)
        #print("append 1")
    else:
        reg.append(0)
        #print("append 0")
    


class envir():
    def __init__(self):
        self.P =[(0.3,0.7,0,0,0),
                (0.3,0,0.7,0,0),
                (0.3,0,0,0.7,0),
                (0.3,0,0,0,0.7),
                (0.3,0,0,0,0.7)]
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
    
    def check_best_action(self,state):
        ind = np.argmin(state)
        all_inds = []
        for i in range(len(state)):
            if state[ind] == state[i]:
                all_inds.append(i)
        return all_inds


# In[122]:


histm0 = []
histm1 = []
histm2 = []
histm3 = []
histm4 = []
hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []
histV0 = []
histV = []

def calculate_v_values(Q_values_action1, Q_values_action2):
    Q_values = np.column_stack((Q_values_action1, Q_values_action2))
    V_values = np.max(Q_values, axis=1)
    return V_values

def calculate_v_values_stochastic(Q_values, action_probabilities):
    V_values = np.sum(Q_values * action_probabilities, axis=1)
    return V_values

V_true = [8.99999135, 8.14648002, 7.45888848, 7.05713998, 6.7344719]
# V_true = [9, 8.15241315, 7.47595723, 6.90163656, 6.49113047]

def bellman_relative_error(V_approx, V_true):
    nonzero_indices = V_true != 0
    if np.any(nonzero_indices):
        #relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]) / V_true[nonzero_indices])
        relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]))
        return np.mean(relative_errors)
    else:
        return np.nan



# In[123]:

hist = [[[],[],[],[],[],[],[],[],[],[]]]*5
BRE = []
plt_wrong_actions = []
def game():
    t = 0
    M = np.array([0,0,0,0,0], dtype=np.float64)
    env = envir()
    agent = Agent(alpha = 0.3,gamma = 0.9)
    epsilon = 1
    #epsilon = 0.2
    cumm_wrong_steps = []
    s = np.random.choice(5,5)
    cumm_rew = []
    episode_rew = 0
    V_vals = [0]*5
    while True:
        if t==0:
            learning_rate = 0.2
            beta = 0.6
        
        if t>=1:
            learning_rate = 0.2 / math.ceil(t/ 5000)
            if(t%10==0):
                beta = 0.6 / (1 + math.ceil(t* math.log(t) / 5000))
            else:
                beta = 0
        if t%25==0:
            if len(cumm_rew)==0:
                cumm_rew.append(episode_rew)
            else:
                cumm_rew.append(cumm_rew[-1]+episode_rew)
            episode_rew = 0
            s = np.random.choice(5,5)
        
        #s = np.random.choice(5,5)
        t = t+1
        #Calculating Q values over state space for a given arm for the given M vector through Q learning
        # for episode_no in range(1):
        action = agent.select_action(s,M,epsilon)
        #epsilon = max(0.1,epsilon*0.99)
        #print("epsilon", epsilon)
        task_opt = agent.check_best_action(s)
        if action not in task_opt:
            cumm_wrong_steps.append(1)
        else:
            cumm_wrong_steps.append(0)
            
        plt_wrong_actions.append(np.mean(cumm_wrong_steps)*100)
        
        next_state, R = env.step(s,action)
        episode_rew += R
        for k in range(5):
            agent.Q_values[int(s[action])][k] += learning_rate*(R+agent.gamma*(max(M[k],agent.Q_values[int(next_state[action])][k]))-agent.Q_values[int(s[action])][k])    

        sold = s[action] 
        if sold == 0: 
            hist0.append(agent.Q_values[int(sold)][int(sold)])
        elif sold == 1:
            hist1.append(agent.Q_values[int(sold)][int(sold)])
        elif sold == 2:
            hist2.append(agent.Q_values[int(sold)][int(sold)])
        elif sold == 3:
            hist3.append(agent.Q_values[int(sold)][int(sold)])
        elif sold == 4:
            hist4.append(agent.Q_values[int(sold)][int(sold)])
        Q0 = []
        for i in range(5):
            Q0.append(agent.Q_values[i][i])

        V_values = calculate_v_values(Q0,M)
        BRE.append(bellman_relative_error(V_values,V_true))
        '''
        V_values = calculate_v_values(Q0, M)
        BRE.append(bellman_relative_error(V_values, V_true))
        regret(s,next_state)
        histV.append(V_values)
        '''
        s = next_state
      #Update M
        #beta = 0.2
        for i in range(5):
            M[i] += beta*(agent.Q_values[i][i] - M[i])      
      #Stopping criteria
        if t>=10000:
            break
        histm0.append(0.1*M[0])
        histm1.append(0.1*M[1])
        histm2.append(0.1*M[2])
        histm3.append(0.1*M[3])
        histm4.append(0.1*M[4])
        
        for i in range(len(s)):
            for j in range(len(s)):
                    hist[i][j].append(agent.Q_values[j][i])
    csv_file_path = 'QGI_toy.csv'
    
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2','State3','State4'])  # Optional header row
        csv_writer.writerows(zip(histm0, histm1, histm2, histm3, histm4))

    print(f"The two lists have been saved to {csv_file_path}")
    print(0.1*M)
    print("CWS: ",sum(cumm_wrong_steps))

    plt.plot(cumm_rew)
    plt.show()


    filename = 'C:\\Intern\\percent_wrong_QGI.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['percent_wrong'])  # Writing the header
        for value in plt_wrong_actions:
            writer.writerow([value])  # Writing each value in a new row
    
    filename = 'C:\Intern\BRE_QGI.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['BRE'])  # Writing the header
        for value in BRE:
            writer.writerow([value])  # Writing each value in a new row

    filename = 'C:\Intern\cumm_rew_QGI.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cumm_rew'])  # Writing the header
        for value in cumm_rew:
            writer.writerow([value])  # Writing each value in a new row            


# In[124]:


game()


'''
plt.figure(figsize=(6,6))
plt.title('Q values for k = 0',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[0][0],'-',c='blue',label='Q[0][0]')
plt.plot(hist[0][1],'-',c='green',label='Q[1][0]')
plt.plot(hist[0][2],'-',c='red',label='Q[2][0]')
plt.plot(hist[0][3],'-',c='green',label='Q[3][0]')
plt.plot(hist[0][4],'-',c='red',label='Q[4][0]')
plt.legend()

plt.figure(figsize=(6,6))
plt.title('Q values for k = 1',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[1][0],'-',c='blue',label='Q[0][1]')
plt.plot(hist[1][1],'-',c='green',label='Q[1][1]')
plt.plot(hist[1][2],'-',c='red',label='Q[2][1]')
plt.plot(hist[1][3],'-',c='green',label='Q[3][1]')
plt.plot(hist[1][4],'-',c='red',label='Q[4][1]')
plt.legend()

plt.figure(figsize=(6,6))
plt.title('Q values for k = 2',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[2][0],'-',c='blue',label='Q[0][2]')
plt.plot(hist[2][1],'-',c='green',label='Q[1][2]')
plt.plot(hist[2][2],'-',c='red',label='Q[2][2]')
plt.plot(hist[2][3],'-',c='green',label='Q[3][2]')
plt.plot(hist[2][4],'-',c='red',label='Q[4][2]')
plt.legend()
'''
# In[125]:


plt.figure(figsize=(6,6))
plt.title('QGI',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Gittins index',fontsize = 'xx-large')
plt.plot(histm0,'-',c='blue',label='State 0')
plt.plot(histm1,':',c='green',label='State 1')
plt.plot(histm2,'--',c='red',label='State 2')
plt.plot(histm3,'-.',c='black',label='State 3')
plt.plot(histm4,'-',c='yellow',label='State 4')
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
plt.title('QGI',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Gittins index',fontsize = 'xx-large')
plt.plot(plt_wrong_actions,'-',c='blue',label='State 0')
plt.legend()
plt.show()

# In[126]:

plt.figure(figsize=(6,6))
plt.title('BE vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('BE',fontsize = 'xx-large')
plt.plot(BRE,'-',c='blue',label='BRE')
plt.legend()
plt.show()
'''
plt.figure(figsize=(6,6))
plt.title('V(s) vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('regret',fontsize = 'xx-large')
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
'''

'''fig=plt.figure()
plt.figure(figsize=(6,6))
plt.title('Q[s][pull] vs episodes plot',fontsize='xx-large')
plt.xlabel('Episodes', fontsize = 'xx-large')
plt.ylabel('Q[s][pull]',fontsize = 'xx-large')
plt.plot(hist0,'-',c='green')
plt.plot(hist1,'-',c='blue')
plt.plot(hist2,'-',c='cyan')

plt.show()


# In[ ]:
'''




# In[ ]:


