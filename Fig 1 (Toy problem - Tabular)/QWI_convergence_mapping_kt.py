# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:36:41 2024

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
import pandas as pd
import csv
from tqdm import tqdm
# In[115]:

#initialise-
action = np.zeros(2)
reward = np.zeros(2)
state = np.zeros(2)
arm = np.zeros(2)
Q = np.zeros((3,2,3))

hist0 = []
hist1 = []
hist2 = []

W = np.zeros(3)



# In[116]:


def get_reward(state,action):
    rewards = []
    for i in range(len(state)):
        if action[i]==1:
            rewards.append((0.9)**(state[i]+1))
        else:
            rewards.append(0)
    return rewards


def transition(state,action,ct0,ct1,ct2):
    P1 = [(0.1,0.9,0),
          (0.1,0,0.9),
          (0.1,0,0.9)]
    P0 = [(1,0,0),
          (0,1,0),
          (0,0,1)]
    next_statee = []
    for i in range(len(state)):
        if action[i]==1:
            next_statee.append(np.random.choice(3,p=P1[state[i]]));
            if state[action[i]]==1:
                ct1+=1
            elif state[action[i]]==0:
                ct0+=1
            else:
                ct2+=1
        else:
            next_statee.append(np.random.choice(3,p=P0[state[i]]));
    return next_statee,ct0,ct1,ct2

def choose_arm(s,We,epsilon,num_0,num_1,num_2):
    wl= []
    p = np.array([0,0])
    x = 0 
    if np.random.random() < epsilon:
        x = np.random.choice(2)
        p[x]=1
        p = p.tolist()
        if s[x] == 0:
            num_0+=1
        elif s[x] == 1:
            num_1+=1
        else:
            num_2+=1
        return p,num_0,num_1,num_2
    else:
        indv =0;
        ind = 0;
        p = [0,0];
        for i in range(2):
            wl.append(We[s[i]])
        for i in range(2):
            indv = max(wl[i],indv)
        for i in range(2):
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
def game(qlr,wlr,K,T,phi,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,ti,ki,igloo):
    gamma = 0.9
    ct0,ct1,ct2=0,0,0
    epsilon = 1
    state = np.array([0,0])
    episodes = 5000
    beta = wlr
    epsilon = 1
    num_0,num_1,num_2 = 0,0,0
    for episode_no in range(episodes):
        
        
        # Tracking whittles estimates
        hist0.append(W[0])
        hist1.append(W[1])
        hist2.append(W[2])
        
        # Scheduling learning rates
        if episode_no==0:
            learning_rate = qlr
            beta = wlr
        if episode_no>=1:
            learning_rate = qlr / math.ceil(episode_no / T)
            if(episode_no%phi==0):
                beta = wlr / (1 + math.ceil(episode_no * math.log(episode_no) / K))
            else:
                beta = 0
        
        # Transition
        arm,num_0,num_1,num_2 = choose_arm(s=state,We=W,epsilon=1,num_0 = num_0,num_1 = num_1, num_2=num_2)
        epsilon = max(0.1,epsilon*0.999)
        action = arm
        next_state,ct0,ct1,ct2 = transition(state,action,ct0,ct1,ct2)
        reward = get_reward(state,action)
        
        # Updates
        for i in range(2):
            for k in range(3):
                Q[state[i]][action[i]][k] += learning_rate*((1-action[i])*(W[k])+action[i]*reward[i]+gamma*(max(Q[next_state[i]][0][k],Q[next_state[i]][1][k]))-Q[state[i]][action[i]][k])
                
        for k in range(3):
            W[k] += beta*(Q[k][1][k]-Q[k][0][k])

        
        # Tracking Q-values
        for i in range(3):
            for j in range(3):
                    hist[i][j].append(Q[j][0][i])
                    hist[i][j+3].append(Q[j][1][i])
        

        state = next_state
        
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(1,201):
        sum0 += hist0[-i]
        sum1 += hist1[-i]
        sum2 += hist2[-i]
    processed_W = [sum0/200,sum1/200,sum2/200]
    
    true = [0.9,0.815,0.7511]
    
    if abs(processed_W[0] - 0.9) <= 0.01 and abs(processed_W[1] - 0.815) <= 0.01 and abs(processed_W[2] - 0.7511) <= 0.01:
        print("yes")
        zone_1_conv[ki][ti] += 1 

    if abs(processed_W[0] - 0.9) <= 0.025 and abs(processed_W[1] - 0.815) <= 0.025 and abs(processed_W[2] - 0.7511) <= 0.025:
        print("yes2")
        zone_2_conv[ki][ti] += 1 

    if abs(processed_W[0] - 0.9) <= 0.05 and abs(processed_W[1] - 0.815) <= 0.05 and abs(processed_W[2] - 0.7511) <= 0.05:
        print("yes3")
        zone_3_conv[ki][ti] += 1 

    if ((abs(processed_W[0] - true[0]) <= 0.01 and abs(processed_W[1] - true[1]) <= 0.01 and abs(processed_W[2] - true[2]) <= 0.05) or
    (abs(processed_W[0] - true[0]) <= 0.01 and abs(processed_W[2] - true[2]) <= 0.01 and abs(processed_W[1] - true[1]) <= 0.05) or
    (abs(processed_W[1] - true[1]) <= 0.01 and abs(processed_W[2] - true[2]) <= 0.01 and abs(processed_W[0] - true[0]) <= 0.05)):
        print("yes4")
        zone_4_conv[ki][ti] += 1 
        
    if processed_W[0] > processed_W[1] and processed_W[1] > processed_W[2]:
        print("yeso")
        order_conv[ki][ti] += 1 
        
    
    #whittles_value_data.append(processed_W)
    plt.figure(figsize=(6,6))
    plt.title(f'QWI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, kappa = {K}, theta = {T}',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('W',fontsize = 'xx-large')
    plt.plot(hist2,'-',c='blue',label='State 2')
    plt.plot(hist1,'-',c='green',label='State 1')
    plt.plot(hist0,'-',c='red',label='State 0')
    plt.legend()

    plt.savefig(f'Plots/QWI/KT/plot_{T}_{K}_{igloo}.png')
    plt.clf()
    plt.close()

        

# In[118]:

x = 0.1
y = 0.2  
K = [1000,1500,2000,2500,3000,3500,4000,4500,5000]
T = [5000,4500,4000,3500,3000,2500,2000,1500,1000]
phi = 100

K.reverse()
T.reverse()

zone_1_conv = np.zeros((len(K),len(T)))
zone_2_conv = np.zeros((len(K),len(T)))
zone_3_conv = np.zeros((len(K),len(T)))
zone_4_conv = np.zeros((len(K),len(T)))
order_conv = np.zeros((len(K),len(T)))

for i in tqdm(range(10)):
    for k in range(len(K)):
        for t in range(len(T)):
                    action = np.zeros(2)
                    reward = np.zeros(2)
                    state = np.zeros(2)
                    arm = np.zeros(2)
                    Q = np.zeros((3,2,3))
    
                    hist0 = []
                    hist1 = []
                    hist2 = []
    
                    W = np.zeros(3)
                    hist = [[[],[],[],[],[],[]]]*3
                    game(x,y,K[k],T[t],phi,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,t,k,i)
                
                    


title = "Convergence map for K and T"
xlabel = "T"
ylabel = "K"

Klist = [1000,1500,2000,2500,3000,3500,4000,4500,5000]
Tlist = [5000,4500,4000,3500,3000,2500,2000,1500,1000]
special_point = (0,8)
Klist.reverse()
Tlist.reverse()
def plot_heatmap(matrix, title, xlabel, ylabel, special_point, listT, listK, zone):
    """
    Plots a heatmap of the matrix with values between 0 and 10. The colormap ranges from blue (0) to red (10).
    
    Parameters:
    - matrix: numpy array with values between 0 and 10
    - title: title of the graph
    - xlabel: label for the x-axis
    - ylabel: label for the y-axis
    - listT: list of labels for x-axis
    - listK: list of labels for y-axis
    - zone: zone identifier for the filename
    - bound: bound identifier for the filename
    """
    
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    if len(listT) != matrix.shape[1] or len(listK) != matrix.shape[0]:
        raise ValueError("Length of listT and listK must match the dimensions of the matrix")

    # Create a new figure
    plt.figure(figsize=(8, 8))
    
    # Create heatmap
    cax = plt.imshow(matrix, cmap='coolwarm', vmin=0, vmax=10)
    
    # Add color bar
    cbar = plt.colorbar(cax, ticks=np.arange(0, 1.1, 0.1))
    cbar.set_label('Value', fontsize='large')
    cbar.ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.1)], fontsize='large')
    
    plt.scatter(special_point[1], special_point[0], color='green', s=100, edgecolors='black')
    
    # Set the ticks and labels
    plt.xticks(np.arange(matrix.shape[1]), labels=listT, fontsize='large')
    plt.yticks(np.arange(matrix.shape[0]), labels=listK, fontsize='large')
    
    # Set graph title and axis labels
    plt.title(title, fontsize='xx-large')
    plt.xlabel(xlabel, fontsize='xx-large')
    plt.ylabel(ylabel, fontsize='xx-large')

    # Save and show the heatmap
    plt.savefig(f'Conv_maps/QWI/KT/heatmap_zone{zone}.png')
    plt.show()
    plt.clf()
    plt.close()

# Example usage
zone = 1
plot_heatmap(zone_1_conv, title, xlabel, ylabel, special_point, Tlist, Klist, zone)

zone = 2 
plot_heatmap(zone_2_conv, title, xlabel, ylabel, special_point, Tlist, Klist, zone)

zone = 3 
plot_heatmap(zone_3_conv, title, xlabel, ylabel, special_point, Tlist, Klist, zone)

zone = 0
plot_heatmap(order_conv, title, xlabel, ylabel, special_point, Tlist, Klist, zone)

'''
zone = 1
bound = 7
plot_array_as_graph(convmap_zone1_7,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 2
bound = 7
plot_array_as_graph(convmap_zone2_7,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 3
bound = 7
plot_array_as_graph(convmap_zone3_7,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 4
bound = 7
plot_array_as_graph(convmap_zone4_7,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 0
bound = 7
plot_array_as_graph(convmap_order_7,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 1
bound = 5
plot_array_as_graph(convmap_zone1_5,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 2
bound = 5
plot_array_as_graph(convmap_zone2_5,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 3
bound = 5
plot_array_as_graph(convmap_zone3_5,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 4
bound = 5
plot_array_as_graph(convmap_zone4_5,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 0
bound = 5
plot_array_as_graph(convmap_order_5,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 1
bound = 10
plot_array_as_graph(convmap_zone1_10,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 2
bound = 10
plot_array_as_graph(convmap_zone2_10,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 3
bound = 10
plot_array_as_graph(convmap_zone3_10,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 4
bound = 10
plot_array_as_graph(convmap_zone4_10,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

zone = 0
bound = 10
plot_array_as_graph(convmap_order_10,title,xlabel,ylabel,special_point,listT,listK,zone,bound)

'''



