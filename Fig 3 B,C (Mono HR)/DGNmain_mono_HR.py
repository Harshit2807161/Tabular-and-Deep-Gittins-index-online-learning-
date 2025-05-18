import numpy as np
import math
import matplotlib.pyplot as plt
import dgnn_mono_HR_3s as DGNnn
import torch
import copy
import csv
from tqdm import tqdm 
from numpy import random


def load_csv_with_counter(file_name):
    row_counter = 0
    rowwise_lists = []

    # Open and read the CSV file
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header row

        # Iterate through each row in the CSV file
        for row in reader:
            row_as_int = [int(value) for value in row]  # Convert each value to integer
            rowwise_lists.append(row_as_int)
            row_counter += 1  # Increment the counter
    return rowwise_lists

file_name = "C:\\Intern\\Gittin\'s plots\\All codes directory\\Fig 3 B,C (Mono HR)\\jobsize_data_decHR.csv"  # Replace with the actual file name
rowwise_lists = load_csv_with_counter(file_name)


#initialise-
action = np.zeros(9)
reward = np.zeros(9)
state = np.zeros(9)
arm = np.zeros(9)
Q = np.zeros((100,2,100,9))

hist0 = []
hist1 = []
hist2 = []

def calculate_v_values(Q_values_action1, Q_values_action2):
    V_values = np.zeros((10,9))
    for i in range(10):
        for j in range(9):
            V_values[i][j] = max(Q_values_action1[i][j],Q_values_action2[i][j])  # Use keepdims=True to maintain the 2D shape
    return V_values

V_true = np.zeros((100,9))


V_true[1,:] = [18.39001075, 24.63113997, 28.37029607, 32.74919392, 36.6070096,  39.95416782,
  42.26659913, 45.21643186, 47.6777546 ]
V_true[2,:] =  [28.20408841, 38.31997852, 43.99755973, 52.84045075, 60.77817213, 69.34942784,
  74.86945361, 83.47013721, 92.19408469]
V_true[3,:] =  [23.39255981, 33.64927941, 41.51103384, 49.29737536, 60.0681885,  65.63925207,
  73.01061097, 78.24495162, 73.41496629]
V_true[4,:] =  [20.27661416, 31.27369285, 41.65217067, 48.66837802, 54.16685682, 57.77537873,
  54.99415751, 38.90083371, 13.70105255]
'''
V_true[5,:] =  [ 86.58998871,  99.1035614,  111.94316864, 124.13070679, 132.8927002,
  135.60003662, 138.30737305, 141.33862305, 143.7220459 ]

V_true[6,:] =  [107.53770447, 120.05126953, 132.96072388, 145.07841492, 157.59199524,
  166.28613281, 168.9934845,  172.08056641, 174.40815735]

V_true[7,:] =    [128.48545837, 141.00390625, 153.97828674, 166.02618408, 178.5397644,
  191.05331421, 199.67964172, 202.82250977, 205.09431458]

V_true[8,:] =  [149.43321228, 161.96673584, 174.99584961, 186.97392273, 199.48748779,
  212.00108337, 225.48692322, 233.56445312, 235.78044128]
V_true[9,:] =  [170.38093567, 182.92956543, 196.01341248, 207.92163086, 220.43519592,
  232.94877625, 246.63449097, 259.03582764, 266.46652222]
'''
# V_true = [9, 8.15241315, 7.47595723, 6.90163656, 6.49113047]




def bellman_relative_error(V_approx, V_true):
    nonzero_indices = V_true != 0
    if np.any(nonzero_indices):
        #relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]) / V_true[nonzero_indices])
        relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]))
        return np.mean(relative_errors)
    else:
        return np.nan
    
def check_best_action(state):
    inds = np.argmax(state)
    all_inds = []
    for i in range(len(state)):
        if state[inds] == state[i]:
            all_inds.append(i)
    return all_inds

def choose_arm(s,W,epsilon):
        wl= []
        p = np.array([0,0,0,0,0,0,0,0,0])
        if np.random.random() < epsilon:
            for i in range(len(p)):
                if s[i]!=0:
                    wl.append(i)
            arm_to_pull = np.random.choice(wl,1)[0]
            p[arm_to_pull] = 1
            return p,arm_to_pull
        else:
            p2 = {}
            for i in range(len(p)):
                if s[i]!=0:
                    p2[i] = W[s[i]][i]
            max_key = max(p2, key=p2.get)
            p[max_key]= 1
            return p,max_key


def get_prob(step,p1,lamda):
    pi = [0]*step
    for i in range(1,step):
        pi[i]=(1-(1-p1)*((lamda)**(i)))
    mul = pi[step-1]
    return mul     

class envir():
    def __init__(self):
        self.phi = 9
        self.lamda = 0.8 
        self.p1=[]
        for i in range(9):
            self.p1.append(0.1*(i+1))
            
    def step(self,s,task,jobsize):
            s_old = s[task]
            next_state = copy.copy(s)
            reward = [0]*9
            if (s[task] == jobsize[task] or s[task] == 0):
                next_state[task] = 0
            else:
                #print(s,task,get_prob(s[task],self.p1[task],self.lamda))
                next_state[task]+= 1 
            
            if s[task]!=0 and next_state[task] == 0:
                reward[task] = 1
            elif s[task]==0:
                reward[task] = -10000
            else:
                reward[task] = 0
            return next_state,reward
        
    def getjobsize(self, job_counter):
        jobsize = rowwise_lists[job_counter]
        return jobsize

            
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
        
class Agent():
    def __init__(self,alpha,gamma):
        self.Q_values = np.zeros((100,100,9))
        self.alpha = alpha
        #self.c= c
        self.gamma = gamma
        
import time
def game():
    gamma = 0.99
    epsilon = 0.4
    state = np.array([0,0])
    episodes = 2500
    rate = 1
    beta = 1
    agent = DGNnn.Agent(3,2,-1)
    eps = 1
    M = np.zeros((100,9))
    env = envir()
    cumm_rew = []
    BRE = []
    plt_wrong_actions = []
    cumm_wrong_steps = []
    start_time = time.time()
    step_no = 0
    for episode_no in tqdm(range(episodes)):
        state = np.array([1,1,1,1,1,1,1,1,1])
        eps = eps*0.9985
        episode_rew = 0
        jobsize = env.getjobsize(episode_no)
        while (state[0]!=0 or state[1]!=0 or state[2]!=0 or state[3]!=0 or state[4]!=0 or state[5]!=0 or state[6]!=0 or state[7]!=0 or state[8]!=0):
            if step_no==0:
                beta = 0.4
            if step_no>=1:
                if(step_no%15==0):               
                    beta = 0.4
                else:
                    beta = 0
            arm,task = choose_arm(s=copy.copy(state),W=M,epsilon=eps)
            action = arm
            next_state,reward = env.step(copy.copy(state),task,jobsize)     
            step_no+=1
            if len(cumm_rew)==0:
                cumm_rew.append(episode_rew)
            else:
                cumm_rew.append(cumm_rew[-1]+episode_rew)
            task_eps = task
            task_opt = check_best_action(state)
            if task_eps not in task_opt:
                cumm_wrong_steps.append(1)
            else:
                cumm_wrong_steps.append(0)

            plt_wrong_actions.append(np.mean(cumm_wrong_steps)*100)
            Q0 = np.zeros((10,9))
            for i in range(10):
                for j in range(9):
                    input_tensor = torch.FloatTensor(np.array([i,i,j]))
                    output_values = agent.qnetwork_local.forward(input_tensor)
                    Q0[i][j] = output_values[1] 
            V_values = calculate_v_values(Q0,M)
            V_values_full = np.zeros((100,9))
            V_values_full[:10, :] = V_values
            BRE.append(bellman_relative_error(V_values_full,V_true))

            for i in range(9):
                if action[i]==1:
                    for k in range(10):
                        agent.step(np.array([state[i],k,i]), action[i], reward[i], np.array([next_state[i],k,i]), False,M[k][task])

            for k in range(10):
                input_tensor = torch.FloatTensor(np.array([k,k,task]))
                output_values = agent.qnetwork_local.forward(input_tensor)
                M[k][task]+=beta*(output_values[1]-M[k][task])
            state = next_state
        #hist1.append(0.01*M[3][3])
        hist1.append(0.01*M[4][1])
        current_time = time.time() - start_time
    #print(current_time)
    V_values = calculate_v_values(Q0,M)
    print(V_values)
    #print(M)
    save_list_to_csv(hist1, "mono_DGN_inc.csv")

    filename = 'C:\\Intern\\percent_wrong_DGN_decHR.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['percent_wrong'])  # Writing the header
        for value in plt_wrong_actions:
            writer.writerow([value])  # Writing each value in a new row
    
    filename = 'C:\\Intern\\BRE_DGN_decHR.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['BRE'])  # Writing the header
        for value in BRE:
            writer.writerow([value])  # Writing each value in a new row
    
    filename = 'C:\\Intern\\cumm_rew_DGN_decHR.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cumm_rew'])  # Writing the header
        for value in cumm_rew:
            writer.writerow([value])  # Writing each value in a new row         
    
game()

plt.figure(figsize=(6,6))
plt.title('M vs time step plot',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('M',fontsize = 'xx-large')
plt.plot(hist1,'-',c='green',label='Arm 4, State 4')
plt.legend()

plt.show()

# For one time scale- change objective function (include min of that set)
# For two time scale and one output of NN- change Q_target_next