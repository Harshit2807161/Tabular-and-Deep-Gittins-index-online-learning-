import numpy as np
import math
import matplotlib.pyplot as plt
import QWINNnn
import torch
import copy
from tqdm import tqdm 
import csv
import random
#initialise-
action = np.zeros(10)
reward = np.zeros(10)
state = np.zeros(10)
arm = np.zeros(10)
Q = np.zeros((2,2,2,10))
hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []
M = np.zeros((2,10))


class envir():
    def __init__(self):
        self.phi = 9
        self.p1=[]
        for i in range(10):
            self.p1.append(0.05*(i+1))
    def step(self,s,task):
            reward = [0]*10
            next_state = copy.copy(s)
            if s[task] == 0:
                next_state[task] = 0
            else:
                #print(s,task,get_prob(s[task],self.p1[task],self.lamda))
                next_state[task] = np.random.choice([0,s[task]],1,p=[self.p1[task],1-self.p1[task]])[0]
            if s[task]!=0 and next_state[task] == 0:
                reward[task] = 1
            elif s[task]==0:
                reward[task] = -10000
            else:
                reward[task] = 0
            return next_state,reward


def select_action(s,M,epsilon):
    r = random.random()
    if r < epsilon:
        action = np.random.choice(10)
    else:
        gre = [0]*len(s)
        for i in range(len(s)):
            gre[i] = M[s[i]][i]
            
        action = np.argmax(gre)
    actione = [0]*len(s)
    actione[action] = 1
    return actione

def gettask(action):
        task = 0
        for i in range(10):
            if(action[i]==1):
                task = i
                break
        return task

import time
def game():
    episodes = 1000
    start_time = time.time()
    rate = 1
    beta = 0.6
    eps = 1
    agent = QWINNnn.Agent(3,2,-1)
    env = envir()
    state = np.array([1,1,1,1,1,1,1,1,1,1])
    for episode_no in tqdm(range(episodes)):
        state = np.array([1,1,1,1,1,1,1,1,1,1])
        if episode_no==0:
            learning_rate = 0.1
        if episode_no>=1:
            learning_rate = 1 / math.ceil(episode_no / 5000)
            if(episode_no%10==0):
                t = episode_no
                beta = 0.2 / (1 + math.ceil(t* math.log(t) / 5000))
            else:
                beta = 0
        eps = max(eps*0.997,0.1)
        while(state[0]!=0 or state[1]!=0 or state[2]!=0 or state[3]!=0 or state[4]!=0 or state[5]!=0 or state[6]!=0 or state[7]!=0 or state[8]!=0 or state[9]!=0):
            arm = select_action(copy.copy(state),M,1)
            action = arm
            task = gettask(action)
            next_state, reward = env.step(copy.copy(state),task)
            '''print(state)
            print(next_state)
            print(reward)
            print(action)'''
            for i in range(10):
                for k in range(2):
                    agent.step(np.array([state[i],k,i]), action[i], reward[i], np.array([next_state[i],k,i]), False,M[k][i])
            for i in range(10):
                for k in range(2):
                    input_tensor = torch.FloatTensor(np.array([k,k,i]))
                    output_values = agent.qnetwork_local.forward(input_tensor)
                    M[k][i] += beta*(output_values[1]-output_values[0])
            state = next_state
            current_time = time.time()-start_time
        hist1.append(0.01*M[1][5])
    print(current_time)
    # Specify the CSV file path
    plt.figure(figsize=(6,6))
    plt.title('M vs time step plot',fontsize='xx-large')
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('M',fontsize = 'xx-large')
    #plt.plot(hist2,'-',c='blue',label='State 2')
    plt.plot(hist1,'-',c='blue')
    #plt.plot(hist0,'-',c='red',label='State 0')
    plt.legend()
    plt.show()
    
    csv_file_path = 'QWINN_toy.csv'
    '''
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2','State3','State4'])  # Optional header row
        csv_writer.writerows(zip(hist0, hist1, hist2,hist3,hist4))

    print(f"The two lists have been saved to {csv_file_path}")
    ''' 
game()

print(M)
'''
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
'''