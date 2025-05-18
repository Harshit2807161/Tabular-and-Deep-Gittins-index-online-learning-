#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import random
from collections import deque
from copy import copy 

# In[121]:


class envir():
    def __init__(self):
        self.P =[(0.1,0.9,0),
                 (0.1,0,0.9),
                 (0.1,0,0.9)]
        self.rewards = np.array([0.9,0.81,0.729])
    def step(self,s,action):
        next_state = copy(s)
        next_state[action]= np.random.choice(3,p=self.P[s[action]])
        reward = self.rewards[s[action]]
        return next_state,reward
            

class Agent():
    def __init__(self,alpha,gamma):
        self.Q_values = np.zeros((3,3))
        self.alpha = alpha
        #self.c= c
        self.gamma = gamma
        
    def select_action(self,s,M,epsilon):
        r = random.random()
        if r < epsilon:
            action = random.randint(0,1)
        else:
            gre = [0]*len(s)
            for i in range(len(s)):
                gre[i] = M[s[i]]
                
            action = np.argmax(gre)
            
        return action


# In[122]:


histm0 = []
histm1 = []
histm2 = []
hist0 = []
hist1 = []
hist2 = []
histV0 = []


# In[123]:

hist = [[[],[],[],[],[],[]]]*3

def game(lr,beta,e):
    t = 0
    M0 = deque(maxlen=300)
    M1 = deque(maxlen=300)
    M2 = deque(maxlen=300)
    M = np.array([0,0,0], dtype=np.float64)
    env = envir()
    agent = Agent(alpha = 0.3,gamma = 0.9)
    epsilon = 1
    s = np.random.choice(3,2)
    while True:
        if t==0:
            learning_rate = 1
        if t>=1:
            learning_rate = 1 / math.ceil(t/ 100)
            if(t%10==0):
                beta = 0.1 / (1 + math.ceil(t* math.log(t) / 100))
            else:
                beta = 0
        #s = np.random.choice(3,2)
        t = t+1
        #Calculating Q values over state space for a given arm for the given M vector through Q learning
        action = agent.select_action(s,M,epsilon)
        epsilon = max(0.1,epsilon*0.999)
        next_state, R = env.step(s,action)
        for k in range(3):
            agent.Q_values[int(s[action])][k] += learning_rate*(R+agent.gamma*(max(M[k],agent.Q_values[int(next_state[action])][k]))-agent.Q_values[int(s[action])][k])    
        '''if((t%100)==0):
            agent.alpha = agent.alpha-0.0002'''
        sold = s[action] 
        if sold == 0: 
            hist0.append(agent.Q_values[int(sold)][int(sold)])
        elif sold == 1:
            hist1.append(agent.Q_values[int(sold)][int(sold)])
        elif sold == 2:
            hist2.append(agent.Q_values[int(sold)][int(sold)])
            
        s = next_state
      #Print values

      #Update M
        for i in range(3):
            M[i] += beta*(agent.Q_values[i][i] - M[i])      
      #Stopping criteria
        if t>=2500:
            if abs((0.1*sum(M0)/len(M0))-0.9)<=e and abs((0.1*sum(M1)/len(M1))-0.81774245)<=e and abs((0.1*sum(M2)/len(M2))-0.75981343)<=e:
                
                print(f"The LRs which converge index in epsilon neighbourhood of true values are {lr} and {beta}")
                
                print(M)
                '''
                plt.figure(figsize=(6,6))
                plt.title('Q values for k = 0',fontsize='xx-large')
                plt.xlabel('Time step', fontsize = 'xx-large')
                plt.ylabel('Q',fontsize = 'xx-large')
                plt.plot(hist[0][0],'-',c='blue',label='Q[0][0]')
                plt.plot(hist[0][1],'-',c='green',label='Q[1][0]')
                plt.plot(hist[0][2],'-',c='red',label='Q[2][0]')
                plt.legend()

                plt.figure(figsize=(6,6))
                plt.title('Q values for k = 1',fontsize='xx-large')
                plt.xlabel('Time step', fontsize = 'xx-large')
                plt.ylabel('Q',fontsize = 'xx-large')
                plt.plot(hist[1][0],'-',c='blue',label='Q[0][1]')
                plt.plot(hist[1][1],'-',c='green',label='Q[1][1]')
                plt.plot(hist[1][2],'-',c='red',label='Q[2][1]')
                plt.legend()

                plt.figure(figsize=(6,6))
                plt.title('Q values for k = 2',fontsize='xx-large')
                plt.xlabel('Time step', fontsize = 'xx-large')
                plt.ylabel('Q',fontsize = 'xx-large')
                plt.plot(hist[2][0],'-',c='blue',label='Q[0][2]')
                plt.plot(hist[2][1],'-',c='green',label='Q[1][2]')
                plt.plot(hist[2][2],'-',c='red',label='Q[2][2]')
                plt.legend()

                '''
                plt.figure(figsize=(6,6))
                plt.title(f"QGI with HPs: alpha = {lr} and beta = {beta}",fontsize='xx-large')
                plt.xlabel('Time step', fontsize = 'xx-large')
                plt.ylabel('Gittins index',fontsize = 'xx-large')
                plt.plot(histm2,'-',c='blue',label='State 2')
                plt.plot(histm1,'-',c='green',label='State 1')
                plt.plot(histm0,'-',c='red',label='State 0')
                plt.legend()
                plt.show()
                
                return M
            
            
            break
        histm0.append(0.1*M[0])
        histm1.append(0.1*M[1])
        histm2.append(0.1*M[2])
        
        M0.append(M[0])
        M1.append(M[1])
        M2.append(M[2])
        
        for i in range(3):
            for j in range(3):
                    hist[i][j].append(agent.Q_values[j][i])
        
        
        
      #For M[i]
        '''if(t>=10):
            if(F[0]==0 and (histV0[len(histV0)-1]-histV0[len(histV0)-5]<0.1)):
                ct=1;
                m=M[0]   
      #For algorithm
        print(" ")
        histm0.append(M[0])
        histm1.append(M[1])
        if ((np.linalg.norm(F)<.9) or t>=9000) and (t>=2 and np.linalg.norm(V-Vold)<0.1):
            print("Gittin's index for state 0 is",0.2*min(m,(M[0])))
            print("Gittin's index for state 1 is",0.2*(M[1]))
            break'''

    print(0.1*M)
            


# In[124]:

lrs = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]    
betas = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]

for lr in lrs:
    for beta in betas:
        e = 0.001        
        histm0 = []
        histm1 = []
        histm2 = []
        hist0 = []
        hist1 = []
        hist2 = []
        histV0 = []
        M = game(lr,beta,e)

'''

plt.figure(figsize=(6,6))
plt.title('Q values for k = 0',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[0][0],'-',c='blue',label='Q[0][0]')
plt.plot(hist[0][1],'-',c='green',label='Q[1][0]')
plt.plot(hist[0][2],'-',c='red',label='Q[2][0]')
plt.legend()

plt.figure(figsize=(6,6))
plt.title('Q values for k = 1',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[1][0],'-',c='blue',label='Q[0][1]')
plt.plot(hist[1][1],'-',c='green',label='Q[1][1]')
plt.plot(hist[1][2],'-',c='red',label='Q[2][1]')
plt.legend()

plt.figure(figsize=(6,6))
plt.title('Q values for k = 2',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[2][0],'-',c='blue',label='Q[0][2]')
plt.plot(hist[2][1],'-',c='green',label='Q[1][2]')
plt.plot(hist[2][2],'-',c='red',label='Q[2][2]')
plt.legend()

# In[125]:


plt.figure(figsize=(6,6))
plt.title('QGI',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Gittins index',fontsize = 'xx-large')
plt.plot(histm2,'-',c='blue',label='State 2')
plt.plot(histm1,'-',c='green',label='State 1')
plt.plot(histm0,'-',c='red',label='State 0')
plt.legend()
plt.show()



# In[ ]:


'''

