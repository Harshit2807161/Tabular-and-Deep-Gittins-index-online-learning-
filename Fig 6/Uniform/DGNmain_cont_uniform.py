import numpy as np
import math
import matplotlib.pyplot as plt
import DGNnn_cont_uniform as DGNnn
import torch
import copy
import csv
from tqdm import tqdm 
import time 
import math
from tqdm import tqdm 
from copy import copy


#initialise-
action = np.zeros(4)
reward = np.zeros(4)
state = np.zeros(4)
arm = np.zeros(4)
Q = np.zeros((101,2,101))

hist0 = []
hist1 = []
hist2 = []

M = np.zeros(101)

class envir():
    def __init__(self):
        self.phi = 4
        self.S = np.array([1,1,1,1])
    
    def step(self,task,s,jobsize):
        s_old = s[task]
        reward = [0,0,0,0]
        if(s_old==jobsize[task] or s_old==0):
            s_new = 0
        else:
            s_new = s_old+1
        
        if(s_old==0):
            reward[task] = -1000
            print("dfsd")
        elif(s_old!=0 and s_new!=0):
            reward[task] = 0
        else:
            reward[task] = 1

        new_state = [0,0,0,0]
        for i in range(4):
            if i==task:
                new_state[i] = s_new
            else:
                new_state[i] = s[i]
        return new_state,reward
    
    def getjob(self):
        num_samples = 4
        discretized_values = np.arange(1, 11, 0.1)
        # Sample from log-normal distribution and clip
        sampled_values_clipped = np.clip(np.random.uniform(1, 10, num_samples), 1, 10)

        # Digitize the clipped values
        sampled_values = discretized_values[np.digitize(sampled_values_clipped, discretized_values)]
        j = [0]*len(sampled_values)
        #print(sampled_values)
        for i in range(len(sampled_values)):
            j[i] = math.floor((sampled_values[i]-1)/(0.1) + 1)        
        return j
    
    
class Agent():
    def __init__(self,alpha,gamma):
        self.Q_values = np.zeros((101,101))
        self.phi = 4
        self.S = np.array([6,6,6,6])
        self.alpha = alpha
        self.gamma = gamma
def select_task(s,M):
        k = []
        for i in range(4):
            k.append(M[s[i]])
        #print(k)
        max_index = np.argmax(k)  # Get the index of the maximum value
        return max_index
    
def choose_arm(s,W,epsilon):
        wl= []
        p = np.array([0,0,0,0])
        if np.random.random() < epsilon:
            for i in range(len(p)):
                if s[i]!=0:
                    wl.append(i)
            arm_to_pull = np.random.choice(wl,1)[0]
            p[arm_to_pull]=1
            return p,arm_to_pull
        else:
            p2 = {}
            for i in range(len(p)):
                if s[i]!=0:
                    p2[i] = W[s[i]]
            max_key = max(p2, key=p2.get)
            p[max_key] = 1
            return p,max_key
    
def game():
    gamma = 0.9
    epsilon = 0.4
    state = np.array([1,1,1,1])
    episodes = 10000
    rate = 1
    beta = 1
    agent = DGNnn.Agent(2,2,-1)
    eps = 1
    state = np.random.choice(3,2)
    env = envir()
    for episode_no in tqdm(range(episodes)):
        state = np.array([1,1,1,1])
        jobsize = env.getjob()
        while (state[0]!=0 or state[1]!=0 or state[2]!=0 or state[3]!=0):
            eps = max(eps*0.9995,0.1)
            arm,task = choose_arm(s=copy(state),W=M,epsilon=eps)
            action = arm
            next_state,reward = env.step(task,copy(state),jobsize) 
            for i in range(4):
                if action[i]==1:
                    for k in range(1,101):
                            agent.step(np.array([state[i],k]), action[i], reward[i], np.array([next_state[i],k]), False,M[k])
    
            for k in range(1,101):
                input_tensor = torch.FloatTensor(np.array([k,k]))
                output_values = agent.qnetwork_local.forward(input_tensor)
                M[k]+=0.7*(output_values[1]-M[k])
            state = next_state
            hist1.append(0.01*M[34])
    print(0.01*M)
    
    # Specify the CSV file path
    csv_file_path = 'DGN_cont_uniform2.csv'
    
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State42'])  # Optional header row
        csv_writer.writerows(map(lambda x: [x], hist1))

    print(f"The two lists have been saved to {csv_file_path}")

    
game()


plt.figure(figsize=(6,6))
plt.title('M vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('M',fontsize = 'xx-large')
plt.plot(hist1,'-',c='green',label='State 1')
plt.legend()

plt.show()

# For one time scale- change objective function (include min of that set)
# For two time scale and one output of NN- change Q_target_next