
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from env.mjenv_reach_site import MjEnv

episode_horizon = 100000
num_test_episodes = 1000

def calculate_dynamic_friction(sim_state):
        tau_f = [0,0,0,0,0,0,0]
        FI_10 = 0.54615
        FI_11 = 0.87224
        FI_12 = 0.64068
        FI_13 = 1.2794
        FI_14 = 0.83904
        FI_15 = 0.30301
        FI_16 = 0.56489

        FI_20 = 5.1181
        FI_21 = 9.0657
        FI_22 = 10.136
        FI_23 = 5.5903
        FI_24 = 8.3469
        FI_25 = 17.133
        FI_26 = 10.336

        FI_30 = 0.039533
        FI_31 = 0.025882
        FI_32 = -0.04607
        FI_33 = 0.036194
        FI_34 = 0.026226
        FI_35 = -0.021047
        FI_36 = 0.0035526

        TAU_F_CONST_0 = FI_10/(1+np.exp(-FI_20*FI_30))
        TAU_F_CONST_1 = FI_11/(1+np.exp(-FI_21*FI_31))
        TAU_F_CONST_2 = FI_12/(1+np.exp(-FI_22*FI_32))
        TAU_F_CONST_3 = FI_13/(1+np.exp(-FI_23*FI_33))
        TAU_F_CONST_4 = FI_14/(1+np.exp(-FI_24*FI_34))
        TAU_F_CONST_5 = FI_15/(1+np.exp(-FI_25*FI_35))
        TAU_F_CONST_6 = FI_16/(1+np.exp(-FI_26*FI_36))

        tau_f[0] =  FI_10/(1+np.exp(-FI_20*(sim_state["joint_vel"][0]+FI_30))) - TAU_F_CONST_0
        tau_f[1] =  FI_11/(1+np.exp(-FI_21*(sim_state["joint_vel"][1]+FI_31))) - TAU_F_CONST_1
        tau_f[2] =  FI_12/(1+np.exp(-FI_22*(sim_state["joint_vel"][2]+FI_32))) - TAU_F_CONST_2
        tau_f[3] =  FI_13/(1+np.exp(-FI_23*(sim_state["joint_vel"][3]+FI_33))) - TAU_F_CONST_3
        tau_f[4] =  FI_14/(1+np.exp(-FI_24*(sim_state["joint_vel"][4]+FI_34))) - TAU_F_CONST_4
        tau_f[5] =  FI_15/(1+np.exp(-FI_25*(sim_state["joint_vel"][5]+FI_35))) - TAU_F_CONST_5
        tau_f[6] =  FI_16/(1+np.exp(-FI_26*(sim_state["joint_vel"][6]+FI_36))) - TAU_F_CONST_6

        return tau_f

def calculate_static_friction(action, sim_state):
   if sim_state["joint_vel"][0] < 0.03 and sim_state["joint_vel"][0] > -0.03 and action[0]<0.65 and action[0]>-0.62:
      action[0] = 0
   if sim_state["joint_vel"][1] < 0.03 and sim_state["joint_vel"][1] > -0.03 and action[1]<0.88 and action[1]>-0.81:
      action[1] = 0
   if sim_state["joint_vel"][2] < 0.03 and sim_state["joint_vel"][2] > -0.03 and action[2]<0.51 and action[2]>-0.42:
      action[2] = 0
   if sim_state["joint_vel"][3] < 0.03 and sim_state["joint_vel"][3] > -0.03 and action[3]<0.61 and action[3]>-0.51:
      action[3] = 0
   if sim_state["joint_vel"][4] < 0.03 and sim_state["joint_vel"][4] > -0.03 and action[4]<0.44 and action[4]>-0.67:
      action[4] = 0
   if sim_state["joint_vel"][5] < 0.03 and sim_state["joint_vel"][5] > -0.03 and action[5]<0.40 and action[5]>-0.23:
      action[5] = 0
   if sim_state["joint_vel"][6] < 0.03 and sim_state["joint_vel"][6] > -0.03 and action[6]<0.32 and action[6]>-0.35:
      action[6] = 0
   return action

# Create Environment
sim = MjEnv(
   
   env_name = "panda_torque",
   max_episode_length=10000,
   init_joint_config = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315, 0.029840531355804868, 1.5411935298621688, 0.7534486589746342]
   )

mean_reward={}
std_reward={}

#for cycle to load and evalute all the best models saved in the folder
#model = TQC.load(folder + title, env=env, verbose=1)


#model= SAC.load("/home/lar/Davide/cercagrigliona/data/best_model/best_model_1.zip", env=env, verbose=1)
# Enjoy trained agent
j=0
theta=[]
velocity=[]
#coppie = np.load('/home/ws5/DavideBargellini/cercagrigliona/data/distruggipls/coppie.npy').T
#eef = np.load('/home/ws5/DavideBargellini/cercagrigliona/data/distruggipls/eefpos.npy')
"""
for i in range (0,25):
   action = [8,0,0,0,0,0,0]
   env.step(action)
   env.render()

while True:
   env.render()
"""
for i in range (0,1000):
   sim_state = sim.get_state()
   if i < 200:
      action = [0,0,0,1,0,0,0]
   else:
      action = [0,0,0,0,0,0,0]
   #here i calculate static friction
   action=calculate_static_friction(action, sim_state)
   #here i calculate dynamic friction
   tau = calculate_dynamic_friction(sim_state)

   new_action = [action[0]-tau[0], action[1]-tau[1], action[2]-tau[2], action[3]-tau[3], action[4]-tau[4], action[5]-tau[5], action[6]-tau[6]]

   sim.execute(new_action)
   sim.render()
