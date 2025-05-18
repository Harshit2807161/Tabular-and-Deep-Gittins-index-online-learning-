# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:27:34 2024

@author: Harshit
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import QWINNnn_mono_HR as QWINNnn
import torch
import copy
from tqdm import tqdm 
import csv
from numpy import random
#initialise-
hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []
action = np.zeros(9)
reward = np.zeros(9)
state = np.zeros(9)
arm = np.zeros(9)
Q = np.zeros((100,2,100,9))

hist0 = []
hist2 = []

M = np.zeros((100,9))
def get_prob(step,p1,lamda):
    pi = [0]*step
    for i in range(1,step):
        pi[i]=(1-(1-p1)*((lamda)**(1/i)))
    mul = pi[step-1]
    return mul     

class envir():
    def __init__(self):
        self.phi = 9
        self.lamda = 0.8 
        self.p1=[]
        for i in range(9):
            self.p1.append(0.1*(i+1))
            
    def step(self,s,task):
            next_state = copy.copy(s)
            reward = [0]*9
            if s[task] == 0:
                next_state[task] = 0
            else:
                #print(s,task,get_prob(s[task],self.p1[task],self.lamda))
                next_state[task] = random.choice([0,s[task]+1],1,p=[get_prob(s[task],self.p1[task],self.lamda),1-get_prob(s[task],self.p1[task],self.lamda)])[0]
            if s[task]!=0 and next_state[task] == 0:
                reward[task] = 1
            elif s[task]==0:
                reward[task] = -10000
            else:
                reward[task] = 0
            return next_state,reward
        
        
def choose_arm(s,W,epsilon):
        wl= []
        p = np.array([0,0,0,0,0,0,0,0,0])
        if np.random.random() < epsilon:
            for i in range(len(p)):
                if s[i]!=0:
                    wl.append(i)
            arm_to_pull = np.random.choice(wl,1)[0]
            p[arm_to_pull] = 1
            return p
        else:
            p2 = {}
            for i in range(len(p)):
                if s[i]!=0:
                    p2[i] = W[s[i]][i]
            max_key = max(p2, key=p2.get)
            p[max_key] = 1
            return p
    
def gettask(action):
        task = 0
        for i in range(len(action)):
            if(action[i]==1):
                task = i
                break
        return task
        
def save_list_to_csv(data_list, filename):
    """
    Save a list to a CSV file, where each element of the list is in a separate row of the same column.

    Parameters:
    data_list (list): The list to be saved.
    filename (str): The name of the CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each element of the list as a new row
        for item in data_list:
            writer.writerow([item])  # Each item is placed in a new row in a single column

import time
def game():
    episodes = 2000
    start_time = time.time()
    rate = 1
    env = envir()
    beta = 0.6
    eps = 1
    agent = QWINNnn.Agent(3,2,-1)
    t = []
    for episode_no in tqdm(range(episodes)):
        state = np.array([1,1,1,1,1,1,1,1,1])
        if episode_no==0:
            learning_rate = 0.1
        if episode_no>=1:
            learning_rate = 1 / math.ceil(episode_no / 5000)
            if(episode_no%10==0):
                t = episode_no
                #beta = 0.2 / (1 + math.ceil(t* math.log(t) / 5000))
                beta = 0.2
            else:
                beta = 0
        eps = eps*0.9995
        while(state[0]!=0 or state[1]!=0 or state[2]!=0 or state[3]!=0 or state[4]!=0 or state[5]!=0 or state[6]!=0 or state[7]!=0 or state[8]!=0):
            arm = choose_arm(s=copy.copy(state),W=M,epsilon=eps)
            action = arm
            task = gettask(action)
            next_state,reward = env.step(copy.copy(state),task)
            '''print(state)
            print(next_state)
            print(reward)
            print(action)'''
            for i in range(9):
                for k in range(10):
                    agent.step(np.array([state[i],k,i]), action[i], reward[i], np.array([next_state[i],k,i]), False,M[k][task])
            for k in range(10):
                input_tensor = torch.FloatTensor(np.array([k,k,task]))
                output_values = agent.qnetwork_local.forward(input_tensor)
                M[k][task] += beta*(output_values[1]-output_values[0])
            state = next_state
            curr_time = time.time()-start_time
            #t.append(curr_time)
        hist1.append(0.01*M[3][3])
    plt.figure(figsize=(6,6))
    plt.title('M vs time step plot',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('M',fontsize = 'xx-large')
    #plt.plot(hist2,'-',c='blue',label='State 2')
    plt.plot(hist1,'-',c='blue')
    #plt.plot(hist0,'-',c='red',label='State 0')
    plt.legend()
    plt.show()
    
    
    
        
game()

print(M)

plt.figure(figsize=(6,6))
plt.title('M vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('M',fontsize = 'xx-large')
plt.plot(hist4,'-',c='black',label='State 4')
plt.plot(hist3,'-',c='yellow',label='State 3')
plt.plot(hist2,'-',c='blue',label='State 2')
plt.plot(hist1,'-',c='green',label='State 1')
plt.plot(hist0,'-',c='red',label='State 0')
plt.legend()

plt.show()
