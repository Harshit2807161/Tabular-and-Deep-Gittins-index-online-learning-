import os 
from tqdm.contrib.itertools import product
from tqdm import tqdm
from time import time
'''
BETA_NUM = [0.1,0.5,0.8,1]
BETA_PHI = [1,5,25]
EPS_COEFF = [0,0.9995,0.9998,1]
LR = [5e-3,8e-3,1e-2,5e-2,8e-2]
TAU = [0,0.5,1]
'''

BETA_NUM = [0.5,0.5,0.5,0.5,0.5,0.8,1]
BETA_PHI = [1,1,1,1,1,1,0]
EPS_COEFF = [0,0,0,0.9998,1,0.9998,0]
LR = [0.008,0.01,0.05,0.005,0.005,0.005,0.005]
TAU = [0,0,0,0,0,0,0]
# 3
'''
for i,j,k,l,m in product(BETA_NUM,BETA_PHI,EPS_COEFF,LR,TAU):
    command_template = f"python DGNmain_new.py --beta_num={i} --beta_phi={j} --eps_coeff={k} --lr={l} --tau={m}"
    command = command_template
    os.system(command)
    
'''
    
for i in tqdm(range(len(BETA_NUM))):
    command_template = f"python DGNmain_new.py --beta_num={BETA_NUM[i]} --beta_phi={BETA_PHI[i]} --eps_coeff={EPS_COEFF[i]} --lr={LR[i]} --tau={TAU[i]}"
    command = command_template
    os.system(command)
    

# 0.5_1_0_0.008_0
# 0.5_1_0_0.01_0
# 0.5_1_0.0_0.05_0.0
# 0.5_1_0.9998_0.005_0.0
# 0.5_1_1.0_0.005_0.0
# 0.8_1_0.9998_0.005_0
# 1_1_0_0.005_0