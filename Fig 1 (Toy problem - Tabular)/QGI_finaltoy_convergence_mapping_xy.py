#!/usr/bin/env python
# coding: utf-8
#Deteriorating arms - 
#for restart paper toy problem. 3 states, 2 arms.
#0,1,2 states 2 has highest gittins. so when you are passive you setup TPM such that you can only visit worse off (lower gittins) states. if 0,1,2 are in increasing gittins, you can only visit 0 from 1 if that arm is passive.
# In[114]:

import numpy as np
import math
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from copy import deepcopy 
# In[115]:





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



def game(qlr,wlr,phi,den,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,x,y,igloo):
    t = 0
    beta = wlr
    epsilon = 1
    s = np.random.choice(5,5)
    env = envir()
    while True:
        
        hist0.append(0.1*M[0])
        hist1.append(0.1*M[1])
        hist2.append(0.1*M[2])
        hist3.append(0.1*M[3])
        hist4.append(0.1*M[4])
        
        if t==0:
            learning_rate = 1
        if t>=1:
            learning_rate = qlr / math.ceil(t/ den)
            if(t%phi==0):
                beta = wlr / (1 + math.ceil(t* math.log(t) / den))
            else:
                beta = 0
        
        t = t+1
        
        # Transitions 
        action = agent.select_action(s,M,epsilon)

        next_state, R = env.step(s,action)
        for k in range(5):
            agent.Q_values[int(s[action])][k] += learning_rate*(R+agent.gamma*(max(M[k],agent.Q_values[int(next_state[action])][k]))-agent.Q_values[int(s[action])][k])    

        # One long episode
        s = next_state


        # Updating M
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
    
    if processed_W[0] > processed_W[1] > processed_W[2] > processed_W[3] > processed_W[4]:
        if abs(processed_W[0] - 0.9) <= 0.01 and abs(processed_W[1] - 0.819) <= 0.01 and abs(processed_W[2] - 0.74804) <= 0.01 and abs(processed_W[3] - 0.6909) <= 0.01 and abs(processed_W[4] - 0.64565) <= 0.01:
            print("yes")
            zone_1_conv[y][x] += 1 
    
        if abs(processed_W[0] - 0.9) <= 0.025 and abs(processed_W[1] - 0.819) <= 0.025 and abs(processed_W[2] - 0.74804) <= 0.025 and abs(processed_W[3] - 0.6909) <= 0.025 and abs(processed_W[4] - 0.64565) <= 0.025:
            print("yes2")
            zone_2_conv[y][x] += 1 
    
        if abs(processed_W[0] - 0.9) <= 0.05 and abs(processed_W[1] - 0.819) <= 0.05 and abs(processed_W[2] - 0.74804) <= 0.05 and abs(processed_W[3] - 0.6909) <= 0.05 and abs(processed_W[4] - 0.64565) <= 0.05:
            print("yes3")
            zone_3_conv[y][x] += 1 
            
        
        print("yeso")
        order_conv[y][x] += 1 
        
    
    #whittles_value_data.append(processed_W)
    plt.figure(figsize=(6,6))
    plt.title(f'QGI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, den = {den}',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('W',fontsize = 'xx-large')
    plt.plot(hist4,'-',c='blue',label='State 4')
    plt.plot(hist3,'-',c='black',label='State 3')
    plt.plot(hist2,'-',c='yellow',label='State 2')
    plt.plot(hist1,'-',c='green',label='State 1')
    plt.plot(hist0,'-',c='red',label='State 0')
    plt.legend()

    plt.savefig(f'Plots/QGI_finaltoy/XY/plot_{qlr}_{wlr}_{igloo}.png')
    plt.clf()
    plt.close()


        

# In[118]:

X = [1.2,1.1,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
Y = [0.1,0.2,0.3,0.4,0.50,0.60,0.70,0.80,0.90,1,1.1,1.2]
phi = 10
den = 5000

X.reverse()
Y.reverse()

zone_1_conv = np.zeros((len(Y),len(X)))
zone_2_conv = np.zeros((len(Y),len(X)))
zone_3_conv = np.zeros((len(Y),len(X)))
zone_4_conv = np.zeros((len(Y),len(X)))
order_conv = np.zeros((len(Y),len(X)))

for i in tqdm(range(10)):
    for y in range(len(Y)):
        for x in range(len(X)):
                    #initialise-
                    agent = Agent(alpha = 0.3,gamma = 0.9)
                    
                    hist0 = []
                    hist1 = []
                    hist2 = []
                    hist3 = []
                    hist4 = []
                    M = np.array([0,0,0,0,0], dtype=np.float64)
    
                    hist = [[[],[],[],[],[],[]]]*5

                    game(X[x],Y[y],phi,den,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,x,y,i)
                

# Call the function with the example array

title = "Convergence map for X and Y"
xlabel = "X"
ylabel = "Y"

listX = [1.2,1.1,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
listY = [0.1,0.2,0.3,0.4,0.50,0.60,0.70,0.80,0.90,1,1.1,1.2]
special_point = (8,6)
listX.reverse()
listY.reverse()

def plot_heatmap(matrix, title, xlabel, ylabel, special_point, listX, listY, zone):
    """
    Plots a heatmap of the matrix with values between 0 and 10. The colormap ranges from blue (0) to red (10).
    
    Parameters:
    - matrix: numpy array with values between 0 and 10
    - title: title of the graph
    - xlabel: label for the x-axis
    - ylabel: label for the y-axis
    - listX: list of labels for x-axis
    - listY: list of labels for y-axis
    - zone: zone identifier for the filename
    - bound: bound identifier for the filename
    """
    
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    if len(listX) != matrix.shape[1] or len(listY) != matrix.shape[0]:
        raise ValueError("Length of listX and listY must match the dimensions of the matrix")

    # Create a new figure
    plt.figure(figsize=(8, 8))
    
    # Create heatmap
    cax = plt.imshow(matrix, cmap='coolwarm', vmin=0, vmax=10)
    
    # Add color bar
    cbar = plt.colorbar(cax, ticks=np.linspace(0, 10, 11))
    cbar.ax.set_yticklabels([f'{i/10:.1f}' for i in range(11)], fontsize='large')
    
    #plt.scatter(special_point[1], special_point[0], color='green', s=100, edgecolors='black')
    
    # Set the ticks and labels
    plt.xticks(np.arange(matrix.shape[1]), labels=listX, fontsize='large')
    plt.yticks(np.arange(matrix.shape[0]), labels=listY, fontsize='large')
    
    # Set graph title and axis labels
    plt.title(title, fontsize='xx-large')
    plt.xlabel(xlabel, fontsize='xx-large')
    plt.ylabel(ylabel, fontsize='xx-large')

    # Save and show the heatmap
    plt.savefig(f'Conv_maps/QGI_finaltoy/XY/heatmap_zone{zone}.png')
    plt.show()
    plt.clf()
    plt.close()


zone = 1
plot_heatmap(zone_1_conv, title, xlabel, ylabel, special_point, listX, listY, zone)

zone = 2 
plot_heatmap(zone_2_conv, title, xlabel, ylabel, special_point, listX, listY, zone)

zone = 3 
plot_heatmap(zone_3_conv, title, xlabel, ylabel, special_point, listX, listY, zone)

zone = 0
plot_heatmap(order_conv, title, xlabel, ylabel, special_point, listX, listY, zone)
