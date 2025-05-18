import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
#initialise-
action = np.zeros(5)
reward = np.zeros(5)
state = np.zeros(5)
action = np.zeros(5)
arm = np.zeros(5)
Q = np.zeros((5,2,5))

hist0 = []
hist1 = []
hist2 = []
hist3 = []
hist4 = []
W = np.zeros(5)

def get_reward(state,action):
    rewards = []
    for i in range(5):
        if action[i]==1:
            rewards.append(0)
        else:
            rewards.append((0.9)**(state[i]+1))
    return rewards

def transition(state,action):
    P0 = [(0.1, 0.9, 0, 0, 0), 
          (0.1, 0, 0.9, 0, 0), 
          (0.1, 0, 0, 0.9, 0), 
          (0.1, 0, 0, 0, 0.9), 
          (0.1, 0, 0, 0, 0.9)]
    next_statee = []
    for i in range(5):
        if action[i]==1:
            next_statee.append(0);
        else:
            next_statee.append(np.random.choice(5,p=P0[state[i]]));
    return next_statee

def choose_arm(s,We,epsilon):
    wl= []
    p = np.array([0,0,0,0,0])
    x = 0 
    if np.random.random() < epsilon:
        x = np.random.choice(5)
        p[x]=1
        p = p.tolist()
        return p
    else:
      indv =0;
      ind = 0;
      p = [0,0,0,0,0];
      for i in range(5):
        wl.append(We[s[i]])
      for i in range(5):
        indv = max(wl[i],indv)
      for i in range(5):
        if(indv==wl[i]):
          ind = i;
          break
      p[ind]=1
      return p
            
    
def game():
    start_time = time()
    gamma = 0.9
    eps = 1
    state = np.array([1,0,0,4,2])
    episodes = 20000
    learning_rate = 1
    beta = 1
    for episode_no in range(episodes):
        if episode_no>=1:
            learning_rate = 1 / math.ceil(episode_no / 5000)
            if(episode_no%100==0):
                beta =  1 / (1 + math.ceil(episode_no * math.log(episode_no) / 5000))
            else:
                beta = 0
                
        arm = choose_arm(s=state,We = W,epsilon=1)
        eps = max(0.1,eps*0.9999)
        action = arm
        next_state = transition(state,action)
        reward = get_reward(state,action)
        for i in range(5):
            for k in range(5):
                Q[state[i]][action[i]][k] += learning_rate*((1-action[i])*(reward[i]+W[k])+gamma*(max(Q[next_state[i]][0][k],Q[next_state[i]][1][k]))-Q[state[i]][action[i]][k])
        
        hist0.append(W[0])
        hist1.append(W[1])
        hist2.append(W[2])
        hist3.append(W[3])
        hist4.append(W[4])
        
        for k in range(5):
            W[k] += beta*(Q[k][1][k]-Q[k][0][k])
        

        state = next_state
    time_taken = time()-start_time
    print(time_taken)
        
# 0.010, 0.050, 0.075, 0.10, 0.10        
    
game()

print(W)



plt.style.use('dark_background')
plt.figure(figsize=(6,6))
plt.title('Whittles indices vs episodes plot',fontsize='xx-large')
plt.xlabel('Episode', fontsize = 'xx-large')
plt.ylabel('W',fontsize = 'xx-large')
plt.plot(hist0,'-',c='green')
plt.plot(hist1,'-',c='blue')
plt.plot(hist2,'-',c='red')
plt.plot(hist3,'-',c='cyan')
plt.plot(hist4,'-',c='white')
plt.show()
