import pandas as pd
import matplotlib.pyplot as plt

df_res = pd.read_csv('restart.csv')
df_ret = pd.read_csv('retire.csv')

df_res['Restart_G_1'] = 0.01 * df_res['Restart_G_1']
df_res['Restart_G_2'] = 0.01 * df_res['Restart_G_2']
df_res['Restart_G_3'] = 0.01 * df_res['Restart_G_3']
df_ret['Retire_G_1'] = 0.01 * df_ret['Retire_G_1']
df_ret['Retire_G_2'] = 0.01 * df_ret['Retire_G_2']
df_ret['Retire_G_3'] = 0.01 * df_ret['Retire_G_3']

# graph for state 0
plt.figure(figsize=(10,10))
# plt.title('State 0',fontsize='xx-large')
plt.xlabel('Episodes', fontsize = 25)
plt.ylabel('Gittins Index',fontsize = 25)
# plt.plot(df_w['W_0'], '-',c='red',label='QWI')
plt.plot(df_res['Restart_G_1'], '-',c='blue',label='Restart-in-state(1)')
plt.plot(df_ret['Retire_G_1'], '-',c='green',label='QGI (1)')
plt.plot(df_res['Restart_G_2'], '--',c='blue',label='Restart-in-state(2)')
plt.plot(df_ret['Retire_G_2'], '--',c='green',label='QGI (2)')
plt.plot(df_res['Restart_G_3'], ':',c='blue',label='Restart-in-state(3)')
plt.plot(df_ret['Retire_G_3'], ':',c='green',label='QGI (3)')
plt.xlim([-50, 1000])
plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')
plt.legend(fontsize = 20)
plt.savefig("geom.png")
plt.show()


'''
plt.figure(figsize=(7,7))
# plt.title('State 0',fontsize='xx-large')
plt.xlabel('Episodes', fontsize = 'xx-large')
plt.ylabel('Gittins Index',fontsize = 'xx-large')
# plt.plot(df_w['W_1'], '-',c='red',label='QWI')
plt.plot(df_res['Restart_B_2'], '-',c='blue',label='Restart')
plt.plot(df_ret['Retire_B_2'], '-',c='green',label='Retirement')
plt.xlim([-50, 1000])
plt.legend()
plt.show()
'''

'''
plt.figure(figsize=(7,7))
# plt.title('State 0',fontsize='xx-large')
plt.xlabel('Episodes', fontsize = 'xx-large')
plt.ylabel('Gittins Index',fontsize = 'xx-large')
# plt.plot(df_w['W_2'], '-',c='red',label='QWI')
plt.plot(df_res['Restart_B_3'], '-',c='blue',label='Restart')
plt.plot(df_ret['Retire_B_3'], '-',c='green',label='Retirement')
plt.xlim([-50, 1000])
plt.legend()
plt.show()
'''

'''
plt.figure(figsize=(10,10))
# plt.title('State 0',fontsize='xx-large')
plt.xlabel('Time Step (n)', fontsize = 25)
plt.ylabel('Percent',fontsize = 25)
# plt.plot(df_w['W_2'], '-',c='red',label='QWI')
plt.plot(df_res['Restart_P_Per'], '-',c='blue',label='Restart (Poisson)')
plt.plot(df_ret['Retire_P_Per'], '-',c='green',label='QGI (Poisson)')
plt.plot(df_res['Restart_G_Per'], ':',c='blue',label='Restart (Geometric)')
plt.plot(df_ret['Retire_G_Per'], ':',c='green',label='QGI (Geometric)')
plt.plot(df_res['Restart_B_Per'], '--',c='blue',label='Restart (Binomial)')
plt.plot(df_ret['Retire_B_Per'], '--',c='green',label='QGI (Binomial)')
plt.xlim([-50, 10000])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize = 20, loc='lower right')
plt.savefig("percent.png")
plt.show()

plt.figure(figsize=(10,10))
# plt.title('State 0',fontsize='xx-large')
plt.xlabel('Episodes', fontsize = 25)
plt.ylabel('Cummulative Episodic Regret',fontsize = 25)
# plt.plot(df_w['W_2'], '-',c='red',label='QWI')
plt.plot(df_res['Restart_Reg_P'], '-',c='blue',label='Restart (Poisson)')
plt.plot(df_ret['Retire_Reg_P'], '-',c='green',label='QGI (Poisson)')
plt.plot(df_res['Restart_Reg_G'], ':',c='blue',label='Restart (Geometric)')
plt.plot(df_ret['Retire_Reg_G'], ':',c='green',label='QGI (Geometric)')
plt.plot(df_res['Restart_Reg_B'], '--',c='blue',label='Restart (Binomial)')
plt.plot(df_ret['Retire_Reg_B'], '--',c='green',label='QGI (Binomial)')
plt.xlim([-50, 7000])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize = 20, loc='upper right')
plt.savefig("regret.png")
plt.show()
'''
'''
plt.figure(figsize=(7,7))
# plt.title('State 0',fontsize='xx-large')
plt.xlabel('Episodes', fontsize = 'xx-large')
plt.ylabel('Gittins Index',fontsize = 'xx-large')
# plt.plot(df_w['W_2'], '-',c='red',label='QWI')
plt.plot(df_res['Restart_B_Per'], '-',c='blue',label='Restart')
plt.plot(df_ret['Retire_B_Per'], '-',c='green',label='Retirement')
plt.xlim([-50, 1000])
plt.legend()
plt.show()

plt.figure(figsize=(7,7))
# plt.title('State 0',fontsize='xx-large')
plt.xlabel('Episodes', fontsize = 'xx-large')
plt.ylabel('Gittins Index',fontsize = 'xx-large')
# plt.plot(df_w['W_2'], '-',c='red',label='QWI')
plt.plot(df_res['Restart_G_Per'], '-',c='blue',label='Restart')
plt.plot(df_ret['Retire_G_Per'], '-',c='green',label='Retirement')
plt.xlim([-50, 1000])
plt.legend()
plt.show()
'''