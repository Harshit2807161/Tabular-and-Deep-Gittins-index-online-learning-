import copy
from numpy import random

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



import csv
env = envir()

def simulate_mab_environment(trials=7000):
    env = envir()
    jobsize_data = []

    # Run simulation for each trial
    for trial in range(trials):
        state = [1] * 9  # Start each arm at state 1
        jobsize = [0] * 9  # Track number of pulls for each arm

        # Continue until all arms reach state 0
        while any(state):
            for arm in range(9):
                if state[arm] != 0:
                    state, reward = env.step(state, arm)
                    jobsize[arm] += 1

        jobsize_data.append(jobsize)

    # Save jobsize data to CSV
    with open("C:\Intern\Gittin\'s plots\All codes directory\Fig 3 B,C (Mono HR)\jobsize_data_decHR.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([f"Arm {i}" for i in range(9)])  # Write header
        writer.writerows(jobsize_data)

# Run the simulation
simulate_mab_environment()
