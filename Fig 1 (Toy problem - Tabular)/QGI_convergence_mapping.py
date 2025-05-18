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
from copy import copy 
# In[115]:





# In[116]:


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
    


# In[117]:



def game(qlr,wlr,phi,den,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,x,y,igloo):
    t = 0
    beta = wlr
    epsilon = 1
    s = np.random.choice(3,2)
    while True:
        if t==0:
            learning_rate = 1
        if t>=1:
            learning_rate = qlr / math.ceil(t/ den)
            if(t%phi==0):
                beta = wlr / (1 + math.ceil(t* math.log(t) / den))
            else:
                beta = 0
        
        #s = np.random.choice(3,2)
        
        t = t+1
        
        # Transitions 
        action = agent.select_action(s,M,1)
        epsilon = max(0.1,epsilon*0.999)
        next_state, R = env.step(s,action)
        
        # Q-value updates
        for k in range(3):
            agent.Q_values[int(s[action])][k] += learning_rate*(R+agent.gamma*(max(M[k],agent.Q_values[int(next_state[action])][k]))-agent.Q_values[int(s[action])][k])    

        # One long episode
        s = next_state


        # Updating M
        for i in range(3):
            M[i] += beta*(agent.Q_values[i][i] - M[i])      
      
        # Stopping criteria
        if t>=5000:
            break
        
        # Tracking M
        histm0.append(0.1*M[0])
        histm1.append(0.1*M[1])
        histm2.append(0.1*M[2])
        

        
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(1,201):
        sum0 += histm0[-i]
        sum1 += histm1[-i]
        sum2 += histm2[-i]
    processed_M = [sum0/200,sum1/200,sum2/200]
    
    true = [0.9,0.815,0.7511]
    
    if abs(processed_M[0] - 0.9) <= 0.01 and abs(processed_M[1] - 0.815) <= 0.01 and abs(processed_M[2] - 0.7511) <= 0.01:
        print("yes")
        zone_1_conv[y][x] += 1 
        
    if abs(processed_M[0] - 0.9) <= 0.025 and abs(processed_M[1] - 0.815) <= 0.025 and abs(processed_M[2] - 0.7511) <= 0.025:
        print("yes2")
        zone_2_conv[y][x] += 1 

    if abs(processed_M[0] - 0.9) <= 0.05 and abs(processed_M[1] - 0.815) <= 0.05 and abs(processed_M[2] - 0.7511) <= 0.05:
        print("yes3")
        zone_3_conv[y][x] += 1 

    if ((abs(processed_M[0] - true[0]) <= 0.01 and abs(processed_M[1] - true[1]) <= 0.01 and abs(processed_M[2] - true[2]) <= 0.05) or
    (abs(processed_M[0] - true[0]) <= 0.01 and abs(processed_M[2] - true[2]) <= 0.01 and abs(processed_M[1] - true[1]) <= 0.05) or
    (abs(processed_M[1] - true[1]) <= 0.01 and abs(processed_M[2] - true[2]) <= 0.01 and abs(processed_M[0] - true[0]) <= 0.05)):
        print("yes4")
        zone_4_conv[y][x] += 1 
        
    if processed_M[0] > processed_M[1] and processed_M[1] > processed_M[2]:
        print("yeso")
        order_conv[y][x] += 1 
        
    

    plt.figure(figsize=(6,6))
    plt.title(f'QWI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, den = {den}',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('W',fontsize = 'xx-large')
    plt.plot(histm2,'-',c='blue',label='State 2')
    plt.plot(histm1,'-',c='green',label='State 1')
    plt.plot(histm0,'-',c='red',label='State 0')
    plt.legend()

    plt.savefig(f'Plots/QGI/XY/plot_{x}_{y}_{igloo}_{phi}.png')
    plt.clf()
    plt.close()

        

# In[118]:

X = [1.3,1.25,1.20,1.15,1.10,1.05,1,0.95,0.9,0.85,0.8,0.75,0.7]
Y = [0.2,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]    
phi = 100
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
                    M = np.array([0,0,0], dtype=np.float64)
                    agent = Agent(alpha = 0.3,gamma = 0.9)
                    env = envir()
                    histm0 = []
                    histm1 = []
                    histm2 = []

                    game(X[x],Y[y],phi,den,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,x,y,i)
                
                    
convmap_zone1_7 = np.zeros((len(Y),len(X)))
convmap_zone2_7 = np.zeros((len(Y),len(X)))
convmap_zone3_7 = np.zeros((len(Y),len(X)))
convmap_zone4_7 = np.zeros((len(Y),len(X)))
convmap_order_7 = np.zeros((len(Y),len(X)))

convmap_zone1_5 = np.zeros((len(Y),len(X)))
convmap_zone2_5 = np.zeros((len(Y),len(X)))
convmap_zone3_5 = np.zeros((len(Y),len(X)))
convmap_zone4_5 = np.zeros((len(Y),len(X)))
convmap_order_5 = np.zeros((len(Y),len(X)))

convmap_zone1_10 = np.zeros((len(Y),len(X)))
convmap_zone2_10 = np.zeros((len(Y),len(X)))
convmap_zone3_10 = np.zeros((len(Y),len(X)))
convmap_zone4_10 = np.zeros((len(Y),len(X)))
convmap_order_10 = np.zeros((len(Y),len(X)))

print(zone_1_conv)
print(zone_2_conv)
print(zone_3_conv)
print(zone_4_conv)
print(order_conv)


for i in range(len(Y)):
    for j in range(len(X)): 
        if zone_1_conv[i][j] > 5:
            convmap_zone1_5[i][j] = 1
        if zone_1_conv[i][j] > 7:
            convmap_zone1_7[i][j] = 1
        if zone_1_conv[i][j] == 10:
            convmap_zone1_10[i][j] = 1
            
        if zone_2_conv[i][j] > 5:
            convmap_zone2_5[i][j] = 1
        if zone_2_conv[i][j] > 7:
            convmap_zone2_7[i][j] = 1
        if zone_2_conv[i][j] == 10:
            convmap_zone2_10[i][j] = 1
            
        if zone_4_conv[i][j] > 5:
            convmap_zone4_5[i][j] = 1
        if zone_4_conv[i][j] > 7:
            convmap_zone4_7[i][j] = 1
        if zone_4_conv[i][j] == 10:
            convmap_zone4_10[i][j] = 1
            
        if zone_3_conv[i][j] > 5:
            convmap_zone3_5[i][j] = 1
        if zone_3_conv[i][j] > 7:
            convmap_zone3_7[i][j] = 1
        if zone_3_conv[i][j] == 10:
            convmap_zone3_10[i][j] = 1
            
            
        if order_conv[i][j] > 5:
            convmap_order_5[i][j] = 1
        if order_conv[i][j] > 7:
            convmap_order_7[i][j] = 1
        if order_conv[i][j] == 10:
            convmap_order_10[i][j] = 1
                



def plot_array_as_graph(array, title, xlabel, ylabel, special_point, listX, listY, zone, bound):
    """
    Plots a binary array with 1s as 'X' and marks a specific point in red. Also labels axes from provided lists.
    
    Parameters:
    - array: numpy array of 0s and 1s
    - title: title of the graph
    - xlabel: label for the x-axis
    - ylabel: label for the y-axis
    - special_point: tuple (i, j) indicating the position to mark in red
    - listX: list of labels for x-axis
    - listY: list of labels for y-axis
    """
    
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    if len(listX) != array.shape[1] or len(listY) != array.shape[0]:
        raise ValueError("Length of listX and listY must match the dimensions of the array")

    
    # Create a new figure
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot 'X' for each 1 in the array
    for (i, j), value in np.ndenumerate(array):
        if value == 1:
            color = 'red' if (i, j) == special_point else 'black'
            ax.text(j, i, 'X', ha='center', va='center', fontsize=16, color=color)

    # Set the limits
    ax.set_xlim(-0.5, array.shape[1] - 0.5)
    ax.set_ylim(array.shape[0] - 0.5, -0.5)
    
    # Set the ticks and labels
    ax.set_xticks(np.arange(array.shape[1]))
    ax.set_xticklabels(listX, fontsize='large')
    ax.set_yticks(np.arange(array.shape[0]))
    ax.set_yticklabels(listY, fontsize='large')
    ax.invert_yaxis()  # Invert the y-axis to match the array indexing
    
    # Set graph title and axis labels
    ax.set_title(title, fontsize='xx-large')
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(ylabel, fontsize='xx-large')

    # Display gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Show ticks on both x and y axes
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    # save and close
    plt.savefig(f'Conv_maps/QGI/XY/plot_zone{zone}_bound{bound}.png')
    plt.clf()
    plt.close()



# Call the function with the example array

title = "Convergence map for X and Y"
xlabel = "X"
ylabel = "Y"

listX = [1.3,1.25,1.20,1.15,1.10,1.05,1,0.95,0.9,0.85,0.8,0.75,0.7]
listY = [0.2,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]  
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
    cbar = plt.colorbar(cax, ticks=np.arange(0, 11))
    cbar.set_label('Value', fontsize='large')
    cbar.ax.set_yticklabels([str(i) for i in range(11)], fontsize='large')
    
    plt.scatter(special_point[1], special_point[0], color='green', s=100, edgecolors='black')
    
    # Set the ticks and labels
    plt.xticks(np.arange(matrix.shape[1]), labels=listX, fontsize='large')
    plt.yticks(np.arange(matrix.shape[0]), labels=listY, fontsize='large')
    
    # Set graph title and axis labels
    plt.title(title, fontsize='xx-large')
    plt.xlabel(xlabel, fontsize='xx-large')
    plt.ylabel(ylabel, fontsize='xx-large')

    # Save and show the heatmap
    plt.savefig(f'Conv_maps/QGI/XY/heatmap_zone{zone}_100.png')
    plt.show()
    plt.clf()
    plt.close()


zone = 1
plot_heatmap(zone_1_conv, title, xlabel, ylabel, special_point, listX, listY, zone)

zone = 2 
plot_heatmap(zone_2_conv, title, xlabel, ylabel, special_point, listX, listY, zone)

zone = 3 
plot_heatmap(zone_3_conv, title, xlabel, ylabel, special_point, listX, listY, zone)

zone = 4
plot_heatmap(zone_4_conv, title, xlabel, ylabel, special_point, listX, listY, zone)

zone = 0
plot_heatmap(order_conv, title, xlabel, ylabel, special_point, listX, listY, zone)

'''

zone = 1
bound = 7
plot_array_as_graph(convmap_zone1_7,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 2
bound = 7
plot_array_as_graph(convmap_zone2_7,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 3
bound = 7
plot_array_as_graph(convmap_zone3_7,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 4
bound = 7
plot_array_as_graph(convmap_zone4_7,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 0
bound = 7
plot_array_as_graph(convmap_order_7,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 1
bound = 5
plot_array_as_graph(convmap_zone1_5,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 2
bound = 5
plot_array_as_graph(convmap_zone2_5,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 3
bound = 5
plot_array_as_graph(convmap_zone3_5,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 4
bound = 5
plot_array_as_graph(convmap_zone4_5,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 0
bound = 5
plot_array_as_graph(convmap_order_5,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 1
bound = 10
plot_array_as_graph(convmap_zone1_10,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 2
bound = 10
plot_array_as_graph(convmap_zone2_10,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 3
bound = 10
plot_array_as_graph(convmap_zone3_10,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 4
bound = 10
plot_array_as_graph(convmap_zone4_10,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

zone = 0
bound = 10
plot_array_as_graph(convmap_order_10,title,xlabel,ylabel,special_point,listX,listY,zone,bound)

'''