#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
from collections import defaultdict
import csv
from numpy import random
from mpmath import mp
import time 
import copy 
import matplotlib.pyplot as plt
# no. of tasks = phi
# no. of states = S
# current task = task (0 or 1)
# current state = s (a list of cardinality phi)
# continue action = 0
# restart action = 1

# In[97]:

def calculate_v_values(Q_values_action1, Q_values_action2):
    Q_values = np.column_stack((Q_values_action1, Q_values_action2))
    V_values = np.max(Q_values, axis=1)
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




class envir():
    def __init__(self):
        self.phi = 5

    def step(self,s,action):
        self.P =np.array([(0.3,0.7,0,0,0),
                (0.3,0,0.7,0,0),
                (0.3,0,0,0.7,0),
                (0.3,0,0,0,0.7),
                (0.3,0,0,0,0.7)])
        self.rewards = np.array([0.9,0.9**2,0.9**3,0.9**4, 0.9**5])
        next_state = copy.deepcopy(s)
        next_state[action]= np.random.choice(5,p=self.P[s[action]])
        reward = self.rewards[s[action]]
        return next_state,reward
            


# In[98]:


class Agent():
    def __init__(self,alpha,T,gamma):
        self.Q_values = np.zeros((5,2,5))
        self.phi = 5
        self.S = np.array([5,5,5,5,5])
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
            action = np.random.choice(5)
        else:
            gre = [0]*len(s)
            for i in range(len(s)):
                gre[i] = self.Q_values[s[i]][0][s[i]]
            action = np.argmax(gre)
        return action

        
    def act_greedy(self,s,restart_prob,task):
        return max((self.Q_values[s[task]][0][restart_prob]),(self.Q_values[s[task]][1][restart_prob]))
    
    def update(self,s,next_state,reward,task):
        for k in range(5):
            self.Q_values[s[task]][0][k] = (1-self.alpha)*(self.Q_values[s[task]][0][k]) + self.alpha*(reward+self.gamma*(self.act_greedy(next_state,k,task)))
            self.Q_values[k][1][s[task]] = (1-self.alpha)*(self.Q_values[k][1][s[task]]) + self.alpha*(reward+self.gamma*(self.act_greedy(next_state,s[task],task)))
    
    def check_best_action(self,state):
        ind = np.argmin(state)
        all_inds = []
        for i in range(len(state)):
            if state[ind] == state[i]:
                all_inds.append(i)
        return all_inds
# In[99]:


hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []
t = []
BRE = []
plt_wrong_actions = []
cumm_rew = []
class pull():
        start_time = time.time()
        env = envir()
        agent = Agent(alpha=0.1,T=1000,gamma=0.9)
        episode = 5000
        agent.alpha = 0.1
        agent.gamma = 0.9
        Tmax = 90000    
        Tmin = 0.5
        phi = 3
        T = Tmax
        episode_rew = 0
        V_vals = [0]*5
        b = 0.998
        #eps = 0.2
        eps = 1
        s = np.random.choice(5,5)
        cumm_wrong_steps = []
        for ep_no in range(episode):
            if ep_no%25==0:
                if len(cumm_rew)==0:
                    cumm_rew.append(episode_rew)
                else:
                    cumm_rew.append(cumm_rew[-1]+episode_rew)
                episode_rew = 0
                s = np.random.choice(5,5)
            
            #T = Tmin + b*(T - Tmin) 
            #print(T)
            #task = agent.activate_task(s,T)
            task = agent.activate_task_eps_greedy(s,eps)
            task_opt = agent.check_best_action(s)
            if task not in task_opt:
                cumm_wrong_steps.append(1)
            else:
                cumm_wrong_steps.append(0)
                
            plt_wrong_actions.append(np.mean(cumm_wrong_steps)*100)
            current_time = time.time()-start_time
            next_state,reward = env.step(s,task)
            episode_rew += reward
            agent.update(s,next_state,reward,task)
            t.append(current_time)
            Q0 = []
            M = []
            for i in range(5):
                Q0.append(agent.Q_values[i][0][i])
                M.append(agent.Q_values[i][1][i])
            V_values = calculate_v_values(Q0,M)
            BRE.append(bellman_relative_error(V_values,V_true))
            hist0.append(0.1*agent.Q_values[0][0][0])  
            hist1.append(0.1*agent.Q_values[1][0][1])              
            hist2.append(0.1*agent.Q_values[2][0][2])
            hist3.append(0.1*agent.Q_values[3][0][3])
            hist4.append(0.1*agent.Q_values[4][0][4])
            s = next_state
        csv_file_path = 'restart_toy.csv'
        print("CWS: ",sum(cumm_wrong_steps))
        # Write the two lists to the CSV file side by side
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['State0','State1','State2','State3','State4'])  # Optional header row
            csv_writer.writerows(zip(hist0, hist1, hist2, hist3, hist4))
    
        print(f"The two lists have been saved to {csv_file_path}")
        
        for x in range(5):
            print(agent.Q_values[x][0][x],agent.Q_values[x][1][x])
        #print(current_time)
        
        
        filename = 'C:\\Intern\\percent_wrong_restart.csv'

        # Writing to CSV file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['percent_wrong'])  # Writing the header
            for value in plt_wrong_actions:
                writer.writerow([value])  # Writing each value in a new row

        filename = 'C:\\Intern\\BRE_restart.csv'

        # Writing to CSV file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['BRE'])  # Writing the header
            for value in BRE:
                writer.writerow([value])  # Writing each value in a new row

        filename = 'C:\\Intern\\cumm_rew_restart.csv'

        # Writing to CSV file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['cumm_rew'])  # Writing the header
            for value in cumm_rew:
                writer.writerow([value])  # Writing each value in a new row     

# In[101]:

plt.plot(cumm_rew)
plt.show()
'''
plt.title('Q(s)(c)(s) vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q(s)(c)(s)',fontsize = 'xx-large')
plt.plot(hist0,'-',c='blue',label='State 0')
plt.plot(hist1,'-',c='green',label='State 1')
plt.plot(hist2,'-',c='red',label='State 2')
plt.plot(hist3,'-',c='yellow',label='State 3')
plt.plot(hist4,'-',c='black',label='State 4')
plt.legend()
plt.show()


plt.title('Q(s)(c)(s) vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q(s)(c)(s)',fontsize = 'xx-large')
plt.plot(plt_wrong_actions,'-',c='blue',label='State 0')
plt.legend()
plt.show()
'''
# In[80]:

plt.figure(figsize=(6,6))
plt.title('BE vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('BE',fontsize = 'xx-large')
plt.plot(BRE,'-',c='blue',label='BRE')
plt.legend()
plt.show()

print("Gittin's index for state 0",hist0[len(hist0)-1])
print("Gittin's index for state 1",hist1[len(hist1)-1])
print("Gittin's index for state 2",hist2[len(hist2)-1])
print("Gittin's index for state 3",hist3[len(hist3)-1])
print("Gittin's index for state 4",hist4[len(hist4)-1])


# In[ ]:





# In[ ]:




