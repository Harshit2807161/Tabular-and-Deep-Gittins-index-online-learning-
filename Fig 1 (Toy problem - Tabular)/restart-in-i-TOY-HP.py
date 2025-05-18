
#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
from collections import defaultdict
from tqdm import tqdm
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


class envir():
    def __init__(self):
        self.phi = 5
    def step(self,s,action):
        self.P =np.array([(0.1,0.9,0,0,0),
                (0.1,0,0.9,0,0),
                (0.1,0,0,0.9,0),
                (0.1,0,0,0,0.9),
                (0.1,0,0,0,0.9)])
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


# In[99]:


hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []
t = []
def game(alpha):
        start_time = time.time()
        env = envir()
        agent = Agent(alpha=0.3,T=1000,gamma=0.9)
        episode = 10000
        agent.alpha = alpha
        agent.gamma = 0.9
        Tmax = 90000
        Tmin = 0.5
        phi = 3
        T = Tmax
        b = 0.998
        eps = 1
        for ep_no in range(episode):
            s = np.random.choice(5,5)
            #T = Tmin + b*(T - Tmin) 
            #print(T)
            #task = agent.activate_task(s,T)
            task = agent.activate_task_eps_greedy(s,eps)
            current_time = time.time()-start_time
            next_state,reward = env.step(s,task)
            agent.update(s,next_state,reward,task)
            t.append(current_time)

            hist0.append(0.1*agent.Q_values[0][0][0])  
            hist1.append(0.1*agent.Q_values[1][0][1])              
            hist2.append(0.1*agent.Q_values[2][0][2])
            hist3.append(0.1*agent.Q_values[3][0][3])
            hist4.append(0.1*agent.Q_values[4][0][4])
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
        processed_M = [sum0/200,sum1/200,sum2/200,sum3/200,sum4/200]
        
        if abs(processed_M[0] - 0.9) <= 0.01 and abs(processed_M[1] - 0.819) <= 0.01 and abs(processed_M[2] - 0.74804) <= 0.01 and abs(processed_M[3] - 0.6909) <= 0.01 and abs(processed_M[4] - 0.64565) <= 0.01:
            print(f"Alpha is : {alpha}, M is : {processed_M}")
            print(abs(processed_M[0] - 0.9)+abs(processed_M[1] - 0.819)+abs(processed_M[2] - 0.74804)+abs(processed_M[3] - 0.6909)+abs(processed_M[4] - 0.64565))
            print("############")
            print("\n")
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
            
# In[101]:

alphas = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
for i in tqdm(range(len(alphas))):
                    hist0 = []
                    hist1 = []
                    hist2 = []
                    hist3 = []
                    hist4 = []
                    game(alphas[i])
    


# In[80]:


print("Gittin's index for state 0",hist0[len(hist0)-1])
print("Gittin's index for state 1",hist1[len(hist1)-1])
print("Gittin's index for state 2",hist2[len(hist2)-1])
print("Gittin's index for state 3",hist3[len(hist3)-1])
print("Gittin's index for state 4",hist4[len(hist4)-1])


# In[ ]:





# In[ ]:




