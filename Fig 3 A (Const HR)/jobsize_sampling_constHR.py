import copy
import random
import csv

class Envir:
    def __init__(self):
        self.phi = 9
        self.p1 = []
        for i in range(10):
            self.p1.append(0.05 * (i + 1))

    def step(self, s, task):
        next_state = copy.copy(s)
        if s[task] == 0:
            next_state[task] = 0
        else:
            next_state[task] = random.choices([0, s[task]], weights=[self.p1[task], 1 - self.p1[task]])[0]
        if s[task] != 0 and next_state[task] == 0:
            reward = 1
        elif s[task] == 0:
            reward = -10000
        else:
            reward = 0
        return next_state, reward

def simulate_trials(num_trials):
    env = Envir()
    num_arms = 10
    all_trials_jobsize = []

    for trial in range(num_trials):
        s = [1] * num_arms  # Start with all arms at state 1
        jobsize = [0] * num_arms  # Initialize jobsize array to count pulls per arm
        
        while sum(s) > 0:  # Continue until all arms are in state 0
            for task in range(num_arms):
                if s[task] != 0:  # Only pull the arm if it's not already in state 0
                    s, reward = env.step(s, task)
                    jobsize[task] += 1

        all_trials_jobsize.append(jobsize)

    # Save jobsize results to CSV
    with open('C:\Intern\Gittin\'s plots\All codes directory\Fig 3 A (Const HR)\jobsizes_constHR.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'Arm{i}' for i in range(num_arms)])  # Header
        writer.writerows(all_trials_jobsize)

if __name__ == "__main__":
    num_trials = 2500
    simulate_trials(num_trials)
