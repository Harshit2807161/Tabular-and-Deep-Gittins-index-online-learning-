# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:27:25 2024

@author: Harshit
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:27:34 2024

@author: Harshit
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import DGNnn_dist 
import torch
import copy
from tqdm import tqdm 
import csv
from numpy import random
import pandas as pd
#initialise-


df = pd.read_csv('all_distributions_samples.csv')
binomial_sample_values = df['binomial_sample'].to_numpy()
geometric_sample_values = df['geometric_sample'].to_numpy()
poisson_sample_values = df['poisson_sample'].to_numpy()
all_dists = ['poisson_sample', 'binomial_sample', 'geometric_sample']




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
M = np.zeros((100,9))

class envir():
    def __init__(self):
        self.phi = 4
        self.S = np.array([1,1,1,1])
    
    def step(self,task,s,jobsize,dist):
        s_old = s[task]
        reward = 0
#         print(s_old,jobsize[task])
        if(s_old==jobsize[task] or s_old==0):
            s_new = 0
        else:
            s_new = s_old+1
        
        if(s_old==0):
            reward = -1000
        elif(s_old!=0 and s_new!=0):
            reward = 0
        else:
            reward = 1

        new_state = [0,0,0,0]
        for i in range(4):
            if i==task:
                new_state[i] = s_new
            else:
                new_state[i] = s[i]
                
        r = [0]*len(s)
        r[task] = reward
        #func(s, new_state, dist)
        return new_state,r
    
    def getjob(self, dist, job_counter):
        
        if dist == 'binomial_sample':
            ret = binomial_sample_values[job_counter * 4 : job_counter * 4 + 4]
             
        elif dist == 'geometric_sample':
            ret = geometric_sample_values[job_counter * 4 : job_counter * 4 + 4]
        
        elif dist == 'poisson_sample':
            ret = poisson_sample_values[job_counter * 4 : job_counter * 4 + 4]
            
        return ret
        
        
def choose_arm(s,W,epsilon):
        wl= []
        p = np.array([0,0,0,0])
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
                    p2[i] = W[s[i]]
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

def calculate_count(service, size, new_service):
    count = 0
    max_value = np.max(service)
    flag = 0
    count_max = np.sum(service == max_value)
    if count_max == 1:
        for i in range(4):
            if np.argmax(service) == i:
                if service[i]+1 == new_service[i] or new_service[i] == 0:
                    count += 0
                else:
                    count += 1
    else:
        for i in range(4):
            if np.max(service) == service[i]:
                if service[i]+1 == new_service[i] or new_service[i] == 0:
                    flag = 1
        if flag == 0:
            count += 1
    #print(count)
    return count

hist01_b = [ []*2 for i in range(6)]
ti_b = []
hist01_p = [ []*2 for i in range(6)]
ti_p = []
hist01_g = [ []*2 for i in range(6)]
ti_g = []
regret_arr_b = []
regret_arr_g = []
regret_arr_p = []

wrong_steps_b = []
wrong_steps_g = []
wrong_steps_p = []



import time
def game():
    for dist in all_dists:
        action = np.zeros(4)
        reward = np.zeros(4)
        arm = np.zeros(4)
        Q = np.zeros((6,2,6))
        M = np.zeros((6))
        episodes = 5000
        start_time = time.time()
        rate = 1
        env = envir()
        beta = 0.6
        eps = 1
        agent = DGNnn_dist.Agent(2,2,-1)
        t = []
        job_counter = 0
        for episode_no in tqdm(range(episodes)):
            jobsize = env.getjob(dist,job_counter)
            #print(jobsize)
            state = np.array([1,1,1,1])
            count = 0
            eps = eps*0.9995
            regret_sum = 0
            while(state[0]!=0 or state[1]!=0 or state[2]!=0 or state[3]!=0):
                if episode_no==0:
                    learning_rate = 0.1
                if episode_no>=1:
                    learning_rate = 1 / math.ceil(episode_no / 5000)
                    if(episode_no%5==0):
                        t = episode_no
                        beta = 0.4
                    else:
                        beta = 0
                arm = choose_arm(s=copy.copy(state),W=M,epsilon=eps)
                action = arm
                task = gettask(action)
                next_state,reward = env.step(task,copy.copy(state),jobsize,dist)
                counter = calculate_count(state, jobsize, next_state)
                regret_sum += counter
                '''print(state)
                print(next_state)
                print(reward)
                print(action)'''
                if dist == 'binomial_sample':
                    wrong_steps_b.append(counter)
                elif dist == 'poisson_sample':
                    wrong_steps_p.append(counter)
                elif dist == 'geometric_sample':
                    wrong_steps_g.append(counter)

                for i in range(4):
                    if action[i]==1:
                        for k in range(1,6):
                            agent.step(np.array([state[i],k]), action[i], reward[i], np.array([next_state[i],k]), False,M[k])
                for k in range(1,6):
                    input_tensor = torch.FloatTensor(np.array([k,k]))
                    output_values = agent.qnetwork_local.forward(input_tensor)
                    M[k] += beta*(output_values[1]-M[k])
                state = next_state
                curr_time = time.time()-start_time
            if dist == 'binomial_sample':
                regret_arr_b.append(regret_sum)
            elif dist == 'poisson_sample':
                regret_arr_p.append(regret_sum)
            elif dist == 'geometric_sample':
                regret_arr_g.append(regret_sum)
                
            for i in range(6):     
                if dist == 'binomial_sample':
                    hist01_b[i].append(M[i])
                elif dist == 'poisson_sample':
                    hist01_p[i].append(M[i])
                elif dist == 'geometric_sample':
                    hist01_g[i].append(M[i])

    plt.title('Gittins index of geometric',fontsize='xx-large')
    plt.xlabel('Time', fontsize = 'xx-large')
    plt.ylabel('Gittins index',fontsize = 'xx-large')
    for i in range(6):
        plt.plot(hist01_g[i],'-', label=str(i), linewidth=0.5)
    plt.legend()
    plt.show()
    
    plt.title('Gittins index of poisson',fontsize='xx-large')
    plt.xlabel('Time', fontsize = 'xx-large')
    plt.ylabel('Gittins index',fontsize = 'xx-large')
    for i in range(6):
        plt.plot(hist01_p[i],'-', label=str(i), linewidth=0.5)
    plt.legend()
    plt.show()
    
    plt.title('Gittins index of binomial',fontsize='xx-large')
    plt.xlabel('Time', fontsize = 'xx-large')
    plt.ylabel('Gittins index',fontsize = 'xx-large')
    for i in range(6):
        plt.plot(hist01_b[i],'-', label=str(i), linewidth=0.5)
    plt.legend()
    plt.show()

        
game()

cum_regret_p = []
cum_regret_p.append(regret_arr_p[0])
cum_regret_b = []
cum_regret_b.append(regret_arr_b[0])
cum_regret_g = []
cum_regret_g.append(regret_arr_g[0])
for i in range(1,5000):
    cum_regret_p.append(cum_regret_p[i-1] + regret_arr_p[i])
    cum_regret_b.append(cum_regret_b[i-1] + regret_arr_b[i])
    cum_regret_g.append(cum_regret_g[i-1] + regret_arr_g[i])
plt.plot(cum_regret_p, label = 'poisson')
plt.plot(cum_regret_b, label = 'binom')
plt.plot(cum_regret_g, label = 'geom')
plt.xlim([0, 20000])
plt.legend()
plt.show()


def calculate_percentage(array):
    percentages = []
    cumulative_sum = 0

    for i, value in enumerate(array):
        if value==0:
            cumulative_sum += 1
        else:
            cumulative_sum += 0 
        if i == 0:
            percentage = 100 if value == 0 else 0
        else:
            percentage = (cumulative_sum / (i + 1)) * 100
        percentages.append(percentage)

    return percentages

percentages_b = calculate_percentage(wrong_steps_b)
percentages_g = calculate_percentage(wrong_steps_g)
percentages_p = calculate_percentage(wrong_steps_p)
plt.plot(percentages_b, linewidth=0.7, label = 'binomial')
plt.plot(percentages_g, linewidth=0.7, label = 'geometric')
plt.plot(percentages_p, linewidth=0.7, label = 'poisson')
plt.xlim([0, 5000])
#plt.ylim([70, 100])
plt.legend()
plt.show

