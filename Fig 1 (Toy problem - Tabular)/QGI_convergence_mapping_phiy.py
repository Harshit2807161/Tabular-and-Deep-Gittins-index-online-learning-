#!/usr/bin/env python
# coding: utf-8
#Deteriorating arms - 
#for restart paper toy problem. 3 states, 2 arms.
#0,1,2 states 2 has highest gittins. so when you are passive you setup TPM such that you can only visit worse off (lower gittins) states. if 0,1,2 are in increasing gittins, you can only visit 0 from 1 if that arm is passive.
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



def game(qlr,wlr,phi,den,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,phit,y,igloo):
    t = 0
    beta = wlr
    epsilon = 1
    s = np.random.choice(3,2)
    while True:
        
        if phi != 0:
            if t==0:
                learning_rate = 1
            if t>=1:
                learning_rate = qlr / math.ceil(t/ den)
                if(t%phi==0):
                    beta = wlr / (1 + math.ceil(t* math.log(t) / den))
                else:
                    beta = 0
        else:
            if t==0:
                learning_rate = 1
            if t>=1:
                learning_rate = qlr / math.ceil(t/ den)
                beta = wlr / (1 + math.ceil(t* math.log(t) / den))

        
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
        zone_1_conv[y][phit] += 1 
        
    if abs(processed_M[0] - 0.9) <= 0.025 and abs(processed_M[1] - 0.815) <= 0.025 and abs(processed_M[2] - 0.7511) <= 0.025:
        print("yes2")
        zone_2_conv[y][phit] += 1 

    if abs(processed_M[0] - 0.9) <= 0.05 and abs(processed_M[1] - 0.815) <= 0.05 and abs(processed_M[2] - 0.7511) <= 0.05:
        print("yes3")
        zone_3_conv[y][phit] += 1 

    if ((abs(processed_M[0] - true[0]) <= 0.01 and abs(processed_M[1] - true[1]) <= 0.01 and abs(processed_M[2] - true[2]) <= 0.05) or
    (abs(processed_M[0] - true[0]) <= 0.01 and abs(processed_M[2] - true[2]) <= 0.01 and abs(processed_M[1] - true[1]) <= 0.05) or
    (abs(processed_M[1] - true[1]) <= 0.01 and abs(processed_M[2] - true[2]) <= 0.01 and abs(processed_M[0] - true[0]) <= 0.05)):
        print("yes4")
        zone_4_conv[y][phit] += 1 
        
    if processed_M[0] > processed_M[1] and processed_M[1] > processed_M[2]:
        print("yeso")
        order_conv[y][phit] += 1 
        
    

    plt.figure(figsize=(6,6))
    plt.title(f'QWI for qlr = {qlr}, wrl = {wlr}, phi = {phi}, den = {den}',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('W',fontsize = 'xx-large')
    plt.plot(histm2,'-',c='blue',label='State 2')
    plt.plot(histm1,'-',c='green',label='State 1')
    plt.plot(histm0,'-',c='red',label='State 0')
    plt.legend()

    plt.savefig(f'Plots/QGI/PHIY/plot_{phi}_{y}_{igloo}.png')
    plt.clf()
    plt.close()

        

# In[118]:

x = 1
Y = [0.2,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]    
PHIs = [100,90,80,70,60,50,40,30,20,10,0]
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
                    #initialise-
                    M = np.array([0,0,0], dtype=np.float64)
                    agent = Agent(alpha = 0.3,gamma = 0.9)
                    env = envir()
                    histm0 = []
                    histm1 = []
                    histm2 = []

                    game(x,Y[y],PHIs[phi],den,zone_1_conv,zone_2_conv,zone_3_conv,zone_4_conv,order_conv,phi,y,i)
                
                    


# Call the function with the example array

title = "Convergence map for PHIs and Y"
xlabel = "PHIs"
ylabel = "Y"

listPHIs = [100,90,80,70,60,50,40,30,20,10,0]
listY = [0.2,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]  
special_point = (0,0)
listPHIs.reverse()
listY.reverse()

def plot_heatmap(matrix, title, xlabel, ylabel, special_point, listPHIs, listY, zone):
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
    cbar = plt.colorbar(cax, ticks=np.arange(0, 1.1, 0.1))
    cbar.set_label('Value', fontsize='large')
    cbar.ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.1)], fontsize='large')
    
    plt.scatter(special_point[1], special_point[0], color='green', s=100, edgecolors='black')
    
    # Set the ticks and labels
    plt.xticks(np.arange(matrix.shape[1]), labels=listPHIs, fontsize='large')
    plt.yticks(np.arange(matrix.shape[0]), labels=listY, fontsize='large')
    
    # Set graph title and axis labels
    plt.title(title, fontsize='xx-large')
    plt.xlabel(xlabel, fontsize='xx-large')
    plt.ylabel(ylabel, fontsize='xx-large')

    # Save and show the heatmap
    plt.savefig(f'Conv_maps/QGI/PHIY/heatmap_zone{zone}.png')
    plt.show()
    plt.clf()
    plt.close()


zone = 1
plot_heatmap(zone_1_conv, title, xlabel, ylabel, special_point, listPHIs, listY, zone)

zone = 2 
plot_heatmap(zone_2_conv, title, xlabel, ylabel, special_point, listPHIs, listY, zone)

zone = 3 
plot_heatmap(zone_3_conv, title, xlabel, ylabel, special_point, listPHIs, listY, zone)

zone = 4
plot_heatmap(zone_4_conv, title, xlabel, ylabel, special_point, listPHIs, listY, zone)

zone = 0
plot_heatmap(order_conv, title, xlabel, ylabel, special_point, listPHIs, listY, zone)


'''

zone = 1
bound = 7
plot_array_as_graph(convmap_zone1_7,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 2
bound = 7
plot_array_as_graph(convmap_zone2_7,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 3
bound = 7
plot_array_as_graph(convmap_zone3_7,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 4
bound = 7
plot_array_as_graph(convmap_zone4_7,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 0
bound = 7
plot_array_as_graph(convmap_order_7,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 1
bound = 5
plot_array_as_graph(convmap_zone1_5,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 2
bound = 5
plot_array_as_graph(convmap_zone2_5,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 3
bound = 5
plot_array_as_graph(convmap_zone3_5,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 4
bound = 5
plot_array_as_graph(convmap_zone4_5,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 0
bound = 5
plot_array_as_graph(convmap_order_5,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 1
bound = 10
plot_array_as_graph(convmap_zone1_10,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 2
bound = 10
plot_array_as_graph(convmap_zone2_10,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 3
bound = 10
plot_array_as_graph(convmap_zone3_10,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 4
bound = 10
plot_array_as_graph(convmap_zone4_10,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

zone = 0
bound = 10
plot_array_as_graph(convmap_order_10,title,xlabel,ylabel,special_point,listPHIs,listY,zone,bound)

'''