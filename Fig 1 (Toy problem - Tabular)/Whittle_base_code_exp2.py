#!/usr/bin/env python
# coding: utf-8
#Deteriorating arms - 
#for restart paper toy problem. 3 states, 2 arms.
#0,1,2 states 2 has highest gittins. so when you are passive you setup TPM such that you can only visit worse off (lower gittins) states. if 0,1,2 are in increasing gittins, you can only visit 0 from 1 if that arm is passive.
# In[114]:


import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import csv
# In[115]:


#initialise-
Q = np.zeros((5,2,5))

hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []

W = np.zeros(5)


# Define the default factory function
def default_float_value():
    return 0.0

# Create the defaultdict
V = defaultdict(default_float_value)
V_true = defaultdict(default_float_value)

def calculate_v_values(Q_values_action1, Q_values_action2):
    Q_values = np.column_stack((Q_values_action1, Q_values_action2))
    V_values = np.max(Q_values, axis=1)
    return V_values

V_true = [8.99999135, 8.14648002, 7.45888848, 7.05713998, 6.7344719]
# V_true = [9, 8.15241315, 7.47595723, 6.90163656, 6.49113047]

def bellman_relative_error(V_approx, V_true):
    nonzero_indices = V_true != 0
    if np.any(nonzero_indices):
        #relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]) / V_true[nonzero_indices])
        relative_errors = np.abs((V_approx[nonzero_indices] - V_true[nonzero_indices]))
        return np.mean(relative_errors)
    else:
        return np.nan


reg = []
def regret(s, next_state):
    change = np.subtract(next_state, s)
    #print("s_input",s)
    #print("s_output",next_state)
    #print("change",change)
    if np.nonzero(change)[0] == np.argmax(s):
        reg.append(1)
        #print("append 1")
    else:
        reg.append(0)
        #print("append 0")

# In[116]:


def get_reward(state,action):
    rewards = []
    for i in range(2):
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
    P1 = [(0.3,0.7,0,0,0),
        (0.3,0,0.7,0,0),
        (0.3,0,0,0.7,0),
        (0.3,0,0,0,0.7),
        (0.3,0,0,0,0.7)]
    P0 = [(1,0,0,0,0),
          (0,1,0,0,0),
          (0,0,1,0,0),
          (0,0,0,1,0),
          (0,0,0,0,1)]
    next_statee = []
    prob = 0
    for i in range(2):
        if action[i]==1:
            next_statee.append(np.random.choice(5,p=P1[state[i]]));
            prob = P1[state[i]][next_statee[-1]]
        else:
            next_statee.append(np.random.choice(5,p=P0[state[i]]));
    return next_statee,prob

def choose_arm(s,We,epsilon):
    wl= []
    p = np.array([0,0,0,0,0])
    x = 0 
    if np.random.random() < epsilon:
        x = np.random.choice(2)
        p[x]=1
        p = p.tolist()
        return p
    else:
        indv =0;
        ind = 0;
        p = [0,0,0,0,0];
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
            
def check_best_action(state):
    ind = np.argmin(state)
    all_inds = []
    for i in range(len(state)):
        if state[ind] == state[i]:
            all_inds.append(i)
    return all_inds


    


# In[117]:

hist = [[[],[],[],[],[],[],[],[],[],[]]]*5
print(hist)

histV = []
BRE = []
plt_wrong_actions = []
cumm_rew = []
def game():
    gamma = 0.9
    #epsilon = 1
    episodes = 500
    rate = 0.3
    epsilon = 0.2
    state = np.random.choice(5,2)
    cumm_wrong_steps = []
    episode_rew = 0
    V_vals = [0]*5

    '''
    for episode_no in range(100000):   
        s_full = tuple(state)
        v_vals = [0]*5
        state = np.random.choice(5,5)
        actions = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        for i in range(len(state)):
            reward = get_reward(state, actions[i])
            next_state,prob = transition(state,actions[i])
            ns_full = tuple(next_state)
            v_vals[i] = next(x for x in reward if x != 0) + 0.9*(prob)*(V_true[ns_full])
            
        V_true[s_full] = np.max(v_vals)
        
    print(V_true)
    state = np.random.choice(5,5)
    episode_no = 0
    '''
    
    for episode_no in range(episodes):
        if episode_no%25==0:
            if len(cumm_rew)==0:
                cumm_rew.append(episode_rew)
            else:
                cumm_rew.append(cumm_rew[-1]+episode_rew)
            episode_rew = 0
            s = np.random.choice(5,2)
        
    
        hist0.append(W[0])
        hist1.append(W[1])
        hist2.append(W[2])
        hist3.append(W[3])
        hist4.append(W[4])
        
        
        if episode_no==0:
            learning_rate = 0.1
            beta = 0.2
        if episode_no>=1:
            learning_rate = 0.1 / math.ceil(episode_no / 5000)
            if(episode_no%10==0):
                beta = 0.2 / (1 + math.ceil(episode_no * math.log(episode_no) / 5000))
            else:
                beta = 0
        
        arm = choose_arm(s=state,We=W,epsilon=epsilon)
        #epsilon = max(0.1,epsilon*0.999)
        action = arm
        task = np.argmax(action)
        task_opt = check_best_action(state)
        if task not in task_opt:
            cumm_wrong_steps.append(1)
        else:
            cumm_wrong_steps.append(0)
        next_state, prob = transition(state,action)
        reward = get_reward(state,action)
        episode_rew += np.sum(reward)
        plt_wrong_actions.append(np.mean(cumm_wrong_steps)*100)
       # calculate_bellman_final(state,W)

        for i in range(2):
            for k in range(5):
                Q[state[i]][action[i]][k] += learning_rate*((1-action[i])*(W[k])+action[i]*reward[i]+gamma*(max(Q[next_state[i]][0][k],Q[next_state[i]][1][k]))-Q[state[i]][action[i]][k])
        
        for k in range(5):
            W[k] += beta*(Q[k][1][k]-Q[k][0][k])

        Q0 = []
        M = []
        for i in range(5):
            Q0.append(Q[i][0][i])
            M.append(Q[i][1][i])
        V_values = calculate_v_values(Q0,M)
        BRE.append(bellman_relative_error(V_values,V_true))
        
        for i in range(len(state)):
            for j in range(len(state)):
                    hist[i][j].append(Q[j][0][i])
                    hist[i][j+5].append(Q[j][1][i])
        
        Q0 = []
        Q1 = []
        for i in range(5):
                Q0.append(Q[i][0][i])
                Q1.append(Q[i][1][i])
        state = next_state
        '''
        #V_values = calculate_v_values(Q0, Q1)
        BRE.append(bellman_relative_error(V, V_true))
        #histV.append(V_values)
        regret(state,next_state)
        '''
    csv_file_path = 'QWI_toy.csv'
    print("CWS: ",sum(cumm_wrong_steps))
    
    # Write the two lists to the CSV file side by side
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['State0','State1','State2','State3','State4'])  # Optional header row
        csv_writer.writerows(zip(hist0, hist1, hist2, hist3, hist4))

    print(f"The two lists have been saved to {csv_file_path}")
    
    
    filename = 'C:\Intern\percent_wrong_QWI.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['percent_wrong'])  # Writing the header
        for value in plt_wrong_actions:
            writer.writerow([value])  # Writing each value in a new row

    
    filename = 'C:\Intern\BRE_QWI.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['BRE'])  # Writing the header
        for value in BRE:
            writer.writerow([value])  # Writing each value in a new row

    filename = 'C:\Intern\cumm_rew_QWI.csv'

    # Writing to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cumm_rew'])  # Writing the header
        for value in cumm_rew:
            writer.writerow([value])  # Writing each value in a new row


    
    
            
    
        

# In[118]:


game()

plt.plot(cumm_rew)
plt.show()


'''

data = {
    'W[0]': hist0,
    'W[1]': hist1,
    'W[2]': hist2,
    'W[3]': hist3,
    'W[4]': hist4
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('best_run_QWI.csv', index=False)
# In[119]:

'''
print(W)

"""
plt.figure(figsize=(6,6))
plt.title('Q values for k = 0',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[0][0],'-',c='blue',label='Q[0][0][0]')
plt.plot(hist[0][1],'-',c='green',label='Q[1][0][0]')
plt.plot(hist[0][2],'-',c='red',label='Q[2][0][0]')
plt.plot(hist[0][3],'-',c='cyan',label='Q[0][1][0]')
#plt.plot(hist[0][4],'-',c='orange',label='Q[1][1][0]')
plt.plot(hist[0][5],'-',c='black',label='Q[2][1][0]')
plt.legend()

plt.figure(figsize=(6,6))
plt.title('Q values for k = 1',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[1][0],'-',c='blue',label='Q[0][0][1]')
plt.plot(hist[1][1],'-',c='green',label='Q[1][0][1]')
plt.plot(hist[1][2],'-',c='red',label='Q[2][0][1]')
plt.plot(hist[1][3],'-',c='cyan',label='Q[0][1][1]')
plt.plot(hist[1][4],'-',c='orange',label='Q[1][1][1]')
plt.plot(hist[1][5],'-',c='black',label='Q[2][1][1]')
plt.legend()

plt.figure(figsize=(6,6))
plt.title('Q values for k = 2',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('Q',fontsize = 'xx-large')
plt.plot(hist[2][0],'-',c='blue',label='Q[0][0][2]')
plt.plot(hist[2][1],'-',c='green',label='Q[1][0][2]')
plt.plot(hist[2][2],'-',c='red',label='Q[2][0][2]')
plt.plot(hist[2][3],'-',c='cyan',label='Q[0][1][2]')
plt.plot(hist[2][4],'-',c='orange',label='Q[1][1][2]')
plt.plot(hist[2][5],'-',c='black',label='Q[2][1][2]')
plt.legend()
"""

# In[120]:


plt.figure(figsize=(6,6))
plt.title('QWI',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('W',fontsize = 'xx-large')
plt.plot(hist0,'-',c='blue',label='State 0')
plt.plot(hist1,':',c='green',label='State 1')
plt.plot(hist2,'--',c='red',label='State 2')
plt.plot(hist3,'-.',c='black',label='State 3')
plt.plot(hist4,'-',c='yellow',label='State 4')
plt.legend()

plt.show()


plt.figure(figsize=(6,6))
plt.title('QWI',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('W',fontsize = 'xx-large')
plt.plot(plt_wrong_actions,'-',c='blue',label='State 0')
plt.legend()

plt.show()

# In[113]:

plt.figure(figsize=(6,6))
plt.title('BE vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('BE',fontsize = 'xx-large')
plt.plot(BRE,'-',c='blue',label='BRE')
plt.legend()
plt.show()

print("Whittle's index for state 0",hist0[len(hist0)-1])
print("Whittle's index for state 1",hist1[len(hist1)-1])
print("Whittle's index for state 2",hist2[len(hist2)-1])
print("Whittle's index for state 3",hist3[len(hist3)-1])
print("Whittle's index for state 4",hist4[len(hist4)-1])


# In[ ]:


'''
print("Data has been written to output.csv")

#print(histV[len(histV)-1])

plt.figure(figsize=(6,6))
plt.title('BRE vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('BRE',fontsize = 'xx-large')
plt.plot(BRE,'-',c='blue',label='BRE')
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
plt.title('V(s) vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('V(S)',fontsize = 'xx-large')
plt.plot(histV,'-',c='blue',label='reg')
plt.legend()
plt.show()



# In[ ]:

cumu_regret = []
cumu_regret.append(reg[0])
avg_cum_reg = []

for i in range(len(reg)):
    add = (cumu_regret[i-1] + reg[i])
    cumu_regret.append(add)
    avg_cum_reg.append(cumu_regret[i]/(i+1))

plt.figure(figsize=(6,6))
plt.title('V(s) vs Time Step',fontsize='xx-large')
plt.xlabel('Time step', fontsize = 'xx-large')
plt.ylabel('regret',fontsize = 'xx-large')
plt.plot(avg_cum_reg,'-',c='blue',label='reg')
plt.legend()
plt.show()

'''

