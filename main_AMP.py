'''
    This is a benchmark to see how DDPG works with different GNN for AMP
    改了画图
'''

import torch
import numpy as np
import os
import gymnasium as gym

import pickle  

from ckt_graphs import GraphAMPNMCF
from ddpg import DDPGAgent
from datetime import datetime

from utils import ActionNormalizer, OutputParser2
from models import ActorCriticPVTGAT
from AMP_NMCF import AMPNMCFEnv
from pvt_graph import PVTGraph

date = datetime.today().strftime('%Y-%m-%d')
PWD = os.getcwd()
SPICE_NETLIST_DIR = f'{PWD}/simulations'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

pvtGraph = PVTGraph()
# 清理已存在的PVT角点目录
pvtGraph._clean_pvt_dirs()
# 创建PVT角点目录和文件
pvtGraph._create_pvt_dirs()
pvtGraph._create_pvt_netlists()
pvtGraph._create_pvt_netlists_tran()
CktGraph = GraphAMPNMCF
GNN = ActorCriticPVTGAT # you can select other GNN

""" Run intial op experiment """

run_intial = False
if run_intial == True:
    env = AMPNMCFEnv()
    env._init_random_sim(100)
    
""" Regsiter the environemnt to gymnasium """
from gymnasium.envs.registration import register

env_id = 'sky130AMP_NMCF-v0'
env_dict = gym.envs.registration.registry.copy()

for env in env_dict:
    if env_id in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry[env]

print("Register the environment")
register(
        id = env_id,
        entry_point = 'AMP_NMCF:AMPNMCFEnv',
        max_episode_steps = 50,
        )
env = gym.make(env_id)  
#remote


# parameters
num_steps = 5000
memory_size = 5100
batch_size = 128
noise_sigma = 2 # noise volume
noise_sigma_min = 0.1
noise_sigma_decay = 0.9995 # if 1 means no decay
initial_random_steps = 200
noise_type = 'uniform' 

agent = DDPGAgent(
    env, 
    CktGraph(),
    PVTGraph(),
    GNN().Actor(CktGraph(), PVTGraph()),
    GNN().Critic(CktGraph()),
    memory_size, 
    batch_size,
    noise_sigma,
    noise_sigma_min,
    noise_sigma_decay,
    initial_random_steps=initial_random_steps,
    noise_type=noise_type, 
)

# train the agent
agent.train(num_steps)

print("Replay the best results")
memory = agent.memory
# 获取所有角点的最佳结果
best_reward = float('-inf')
best_action = None
best_corner = None


# 遍历所有角点buffer找到最佳结果
for corner_idx, buffer in memory.corner_buffers.items():
    rewards = buffer['reward'][:buffer['size']]
    if len(rewards) > 0:
        max_reward = np.max(rewards)
        if max_reward > best_reward:
            best_reward = max_reward
            idx = np.argmax(rewards)
            best_action = buffer['action'][idx]
            best_corner = corner_idx

if best_action is not None:
    # 在最佳角点上运行仿真
    results_dict, flag, terminated, truncated, info = agent.env.step((best_action, np.arange(pvtGraph.num_corners)))
    

# saved agent's actor and critic network, memory buffer, and agent
save = True
if save == True:
    model_weight_actor = agent.actor.state_dict()
    save_name_actor = f"Actor_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}.pth"
      
    model_weight_critic = agent.critic.state_dict()
    save_name_critic = f"Critic_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}.pth"
      
    torch.save(model_weight_actor, PWD + "/saved_weights/" + save_name_actor)
    torch.save(model_weight_critic, PWD + "/saved_weights/" + save_name_critic)
    print("Actor and Critic weights have been saved!")

    # save memory
    with open(f'./saved_memories/memory_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}.pkl', 'wb') as memory_file:
        pickle.dump(memory, memory_file)

    # save agent
    with open(f'./saved_agents/DDPGAgent_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}.pkl', 'wb') as agent_file:
        pickle.dump(agent, agent_file)