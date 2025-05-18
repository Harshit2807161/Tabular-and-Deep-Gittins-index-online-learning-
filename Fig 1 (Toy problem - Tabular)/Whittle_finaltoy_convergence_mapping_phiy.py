# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:21:37 2024

@author: Harshit
"""

#!/usr/bin/env python
# coding: utf-8
#Deteriorating arms - 
#for restart paper toy problem. 3 states, 2 arms.
#0,1,2 states 2 has highest gittins. so when you are passive you setup TPM such that you can only visit worse off (lower gittins) states. if 0,1,2 are in increasing gittins, you can only visit 0 from 1 if that arm is passive.
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
print(hist)
def game(qlr,wlr,phi,den,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,x,y,igloo):
    gamma = 0.9
    ct0,ct1,ct2=0,0,0
    epsilon = 1
    state = np.array([0,0])
    episodes = 20000
    beta = wlr
    epsilon = 1
    num_0,num_1,num_2 = 0,0,0
    state = np.random.choice(5,5)
    for episode_no in range(episodes):
        
        
        # Tracking whittles estimates
        hist0.append(W[0])
        hist1.append(W[1])
        hist2.append(W[2])
        hist3.append(W[3])
        hist4.append(W[4])
        if phi!=0:
            # Scheduling learning rates
            if episode_no==0:
                learning_rate = qlr
                beta = wlr
            if episode_no>=1:
                learning_rate = qlr / math.ceil(episode_no / den)
                if(episode_no%phi==0):
                    beta = wlr / (1 + math.ceil(episode_no * math.log(episode_no) / den))
                else:
                    beta = 0
        else:
            if episode_no==0:
                learning_rate = qlr
                beta = wlr
            if episode_no>=1:
                learning_rate = qlr / math.ceil(episode_no / den)
                beta = wlr / (1 + math.ceil(episode_no * math.log(episode_no) / den))
        
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
    plt.title(f'QWI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, den = {den}',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('W',fontsize = 'xx-large')
    plt.plot(hist4,'-',c='blue',label='State 4')
    plt.plot(hist3,'-',c='black',label='State 3')
    plt.plot(hist2,'-',c='yellow',label='State 2')
    plt.plot(hist1,'-',c='green',label='State 1')
    plt.plot(hist0,'-',c='red',label='State 0')
    plt.legend()

    plt.savefig(f'Plots/QWI_finaltoy/PHIY/plot_{wlr}_{igloo}_{phi}.png')
    plt.clf()
    plt.close()

        

# In[118]:

x = 0.1
PHIs =  [100,90,80,70,60,50,40,30,20,10,0]
Y = [0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]   
den = 5000

PHIs.reverse()
Y.reverse()

zone_1_conv = np.zeros((len(Y),len(PHIs)))
zone_2_conv = np.zeros((len(Y),len(PHIs)))
zone_3_conv = np.zeros((len(Y),len(PHIs)))
zone_4_conv = np.zeros((len(Y),len(PHIs)))
order_conv = np.zeros((len(Y),len(PHIs)))

for i in tqdm(range(10)):
    for y in range(len(Y)):
        for phi in range(len(PHIs)):
                    action = np.zeros(5)
                    reward = np.zeros(5)
                    state = np.zeros(5)
                    arm = np.zeros(5)
                    Q = np.zeros((5,2,5))
    
                    hist0 = []
                    hist1 = []
                    hist2 = []
                    hist3 = []
                    hist4 = []
                    
                    W = np.zeros(5)

                    game(x,Y[y],PHIs[phi],den,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,phi,y,i)
                
                    

# Call the function with the example array

title = "Convergence map for PHIs and Y"
xlabel = "PHIs"
ylabel = "Y"

listPHIs = [100,90,80,70,60,50,40,30,20,10,0]
listY = [0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]  
listPHIs.reverse()
listY.reverse()

def plot_heatmap(matrix, title, xlabel, ylabel, listPHIs, listY, zone):
    """
    Plots a heatmap of the matrix with values between 0 and 10. The colormap ranges from blue (0) to red (10).
    
    Parameters:
    - matrix: numpy array with values between 0 and 10
    - title: title of the graph
    - xlabel: label for the x-axis
    - ylabel: label for the y-axis
    - listPHIs: list of labels for x-axis
    - listY: list of labels for y-axis
    - zone: zone identifier for the filename
    - bound: bound identifier for the filename
    """
    
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    if len(listPHIs) != matrix.shape[1] or len(listY) != matrix.shape[0]:
        raise ValueError("Length of listPHIs and listY must match the dimensions of the matrix")

    # Create a new figure
    plt.figure(figsize=(8, 8))
    
    # Create heatmap
    cax = plt.imshow(matrix, cmap='coolwarm', vmin=0, vmax=10)
    
    # Add color bar
    cbar = plt.colorbar(cax, ticks=np.linspace(0, 10, 11))
    cbar.ax.set_yticklabels([f'{i/10:.1f}' for i in range(11)], fontsize='large')
    

    # Set the ticks and labels
    plt.xticks(np.arange(matrix.shape[1]), labels=listPHIs, fontsize='large')
    plt.yticks(np.arange(matrix.shape[0]), labels=listY, fontsize='large')
    
    # Set graph title and axis labels
    plt.title(title, fontsize='xx-large')
    plt.xlabel(xlabel, fontsize='xx-large')
    plt.ylabel(ylabel, fontsize='xx-large')

    # Save and show the heatmap
    plt.savefig(f'Conv_maps/QWI_finaltoy/PHIY/heatmap_zone{zone}.png')
    plt.show()
    plt.clf()
    plt.close()

# Example usage
zone = 1
plot_heatmap(zone_1_conv, title, xlabel, ylabel, listPHIs, listY, zone)

zone = 2 
plot_heatmap(zone_2_conv, title, xlabel, ylabel, listPHIs, listY, zone)

zone = 3 
plot_heatmap(zone_3_conv, title, xlabel, ylabel, listPHIs, listY, zone)

zone = 0
plot_heatmap(order_conv, title, xlabel, ylabel, listPHIs, listY, zone)




