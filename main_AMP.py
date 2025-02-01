'''
    This is a benchmark to see how DDPG works with different GNN for AMP
    改了画图
'''

import torch
import numpy as np
import os
import gymnasium as gym
import multiprocessing

import pickle  

from ckt_graphs import GraphAMPNMCF
from ddpg import DDPGAgent
from datetime import datetime

from utils import ActionNormalizer, OutputParser2
from models import ActorCriticPVTGAT
from AMP_NMCF import AMPNMCFEnv
from pvt_graph import PVTGraph

if __name__ == '__main__':
    # 你现有的主程序代码
    multiprocessing.freeze_support()  # Windows可能需要
    
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

    # parameters
    continue_training = True  # 是否加载已保存的agent
    laststeps = 2000
    old=True
    agent_folder = './saved_results/02-01_22-51_steps9_corners-5_reward--3.39'  # 已保存agent的文件夹路径

    load_buffer = False
    load_buffer_size = 0
    buffer_path = './saved_memories/memory_GraphAMPNMCF_2025-01-31_noise=uniform_reward=-3.61_ActorCriticPVTGAT.pkl'  

    plot_interval = 50
    print_interval = 5

    sample_num = 8
    num_steps = 3000
    initial_random_steps = 0
    batch_size = 128

    noise_sigma = 2 # noise volume
    noise_sigma_min = 0.1
    noise_sigma_decay = 0.9995 # if 1 means no decay
    noise_type = 'uniform' 
    THREAD_NUM = 2
    memory_size = laststeps + num_steps+ 10 + load_buffer_size

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
            max_episode_steps = 100000,   #!!!no limit of steps
            kwargs={'THREAD_NUM': THREAD_NUM ,'print_interval':print_interval}
            )
    env = gym.make(env_id)  
    #remote

                
    # 创建新agent实例
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
        sample_num=sample_num,
        agent_folder= agent_folder,
        old = old
    )
    
    if load_buffer == True:
        agent.load_replay_buffer(buffer_path)
    
    # train the agent
    agent.train(num_steps, plot_interval ,continue_training=continue_training)

    print("********Replay the best results********")
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
        results_dict, flag, terminated, truncated, info = agent.env.step(
            (best_action, np.arange(pvtGraph.num_corners), True)
        )
        

    # saved agent's actor and critic network, memory buffer, and agent
    save = True
    if save == True:
        num_steps = agent.total_step
        # 创建以时间戳和参数命名的文件夹
        current_time = datetime.now().strftime('%m-%d_%H-%M')
        folder_name = f"{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}"
        save_dir = os.path.join(PWD, 'saved_results', folder_name)
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        results_file_name = f"opt_result_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}"
        results_path = os.path.join(save_dir, results_file_name)
        with open(results_path, 'w') as f:
            f.writelines(agent.env.unwrapped.get_saved_results)  

        # 保存模型权重
        model_weight_actor = agent.actor.state_dict()
        save_name_actor = f"Actor_weight_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}.pth"
        
        model_weight_critic = agent.critic.state_dict()
        save_name_critic = f"Critic_weight_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}.pth"
        
        torch.save(model_weight_actor, os.path.join(save_dir, save_name_actor))
        torch.save(model_weight_critic, os.path.join(save_dir, save_name_critic))
        print("Actor and Critic weights have been saved!")

        # 保存 memory
        memory_path = os.path.join(save_dir, f'memory_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}.pkl')
        with open(memory_path, 'wb') as memory_file:
            pickle.dump(memory, memory_file)

        # 保存 agent
        agent_path = os.path.join(save_dir, f'DDPGAgent_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}.pkl')
        with open(agent_path, 'wb') as agent_file:
            pickle.dump(agent, agent_file)
            
        print(f"All results have been saved in: {save_dir}")


