# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:39:15 2024

@author: Harshit
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import DGNnn_new as DGNnn
import torch
import copy
import csv
import random
import time
from tqdm import tqdm 
import argparse
#initialise-
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

M = np.zeros(5)

def get_reward(state,action):
    rewards = []
    for i in range(5):
        if action[i]==1:
            if state[i]==0:
                rewards.append(0.9)
            elif state[i]==1:
                rewards.append(0.81)
            elif state[i]==2:
                rewards.append(0.729)
            elif state[i]==3:
                rewards.append(0.9**(state[i]+1))
            else:
                rewards.append(0.9**(state[i]+1))
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

def select_action(s,M,epsilon):
    r = random.random()
    if r < epsilon:
        action = np.random.choice(5)
    else:
        gre = [0]*len(s)
        for i in range(len(s)):
            gre[i] = M[s[i]]
            
        action = np.argmax(gre)
    actione = [0]*len(s)
    actione[action] = 1
    return actione



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HP tuning")
    
    parser.add_argument("--beta_num",type=float,default=0.2)
    parser.add_argument("--beta_phi",type=int,default=5)
    parser.add_argument("--eps_coeff",type=float,default=0.9995)
    parser.add_argument("--lr",type=float,default=5e-3)
    parser.add_argument("--tau",type=float,default=1e-3)
    
    args = parser.parse_args()

    for trial_no in tqdm(range(10)):
        
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

        M = np.zeros(5)
        
        start_time = time.time()
        cumm_rew = 0 
        gamma = 0.9
        epsilon = 0.4
        episodes = 5000
        rate = 1
        learning_rate = 0.1
        beta = args.beta_num
        agent = DGNnn.Agent(2,2,-1,args.tau,args.lr)
        eps = 1
        state = np.random.choice(5,5)
        ep_rew = 0
        rews = []
        for episode_no in tqdm(range(episodes)):
            if episode_no==0:
                learning_rate = 1
            if episode_no>=1:
                learning_rate = 1 / math.ceil(episode_no / 5000)
                if(episode_no%args.beta_phi==0):
                    t = episode_no
                    beta = args.beta_num / (1 + math.ceil(t* math.log(t) / 1000))
                else:
                    beta = 0
            
            eps = max(eps*args.eps_coeff,0)
            arm = select_action(copy.copy(state),M,0)
            action = arm
            next_state = transition(copy.copy(state),action)
            reward = get_reward(copy.copy(state),action)    
            idx = np.argmax(action)
            cumm_rew+=reward[idx]
            for i in range(5):
                if action[i]==1:
                    for k in range(5):
                            agent.step(state[i], action[i], reward[i], next_state[i], False, M)
    
            for k in range(5):
                input_tensor = torch.FloatTensor(np.array([k,k]))
                output_values = agent.qnetwork_local.forward(input_tensor)
                M[k]+=beta*(output_values[1]-M[k])
            
            current_time = time.time()-start_time
            rews.append(cumm_rew)
            hist0.append(0.1*M[0])
            hist1.append(0.1*M[1])
            hist2.append(0.1*M[2])
            hist3.append(0.1*M[3])
            hist4.append(0.1*M[4])
            
            state = next_state
    
    
        plt.figure(figsize=(10,10))
        plt.title('M vs time step plot',fontsize='xx-large')
        plt.xlabel('Time step', fontsize = 'xx-large')
        plt.ylabel(f'{0.1*M[0]:0.2f}_{0.1*M[1]:0.2f}_{0.1*M[2]:0.2f}_{0.1*M[3]:0.2f}_{0.1*M[4]:0.2f}',fontsize = 'xx-large')
        plt.plot(hist4,'-',c='black',label='State 4')
        plt.plot(hist3,'-',c='yellow',label='State 3')
        plt.plot(hist2,'-',c='blue',label='State 2')
        plt.plot(hist1,'-',c='green',label='State 1')
        plt.plot(hist0,'-',c='red',label='State 0')
        plt.legend()
        save_path = f"Conv_fin/conv_plot_DGN_{args.beta_num}_{args.beta_phi}_{args.eps_coeff}_{args.lr}_{args.tau}_{trial_no}.png"
        plt.savefig(save_path)
        plt.show()
        plt.clf()
        plt.close()
        
        plt.figure(figsize=(6,6))
        plt.title('Cumm reward vs time step plot',fontsize='xx-large')
        plt.xlabel('Time step', fontsize = 'xx-large')
        plt.ylabel('M',fontsize = 'xx-large')
        plt.plot(rews,'-',c="red")
        plt.legend()
        plt.savefig(f"Rew/reward_plot_DGN_{args.beta_num}_{args.beta_phi}_{args.eps_coeff}_{args.lr}_{args.tau}_{cumm_rew}.png")
        #plt.show()
        plt.clf()
        plt.close()
        

# For one time scale- change objective function (include min of that set)
# For two time scale and one output of NN- change Q_target_next