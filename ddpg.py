import numpy as np

from typing import List, Tuple, Dict
from copy import deepcopy
import torch
import os
from torch.nn import LazyLinear
import torch.nn.functional as F
import torch.optim as optim
import pickle

from utils import trunc_normal

from IPython.display import clear_output
import matplotlib
# 使用 Agg backend，这是一个非交互式后端，不需要图形界面
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pvt_graph import PVTGraph

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, CktGraph, PVT_Graph, size: int, batch_size: int = 32):
        self.num_node_features = CktGraph.num_node_features
        self.action_dim = CktGraph.action_dim
        self.num_nodes = CktGraph.num_nodes
        self.pvt_dim = PVT_Graph.corner_dim
        self.num_corners = PVT_Graph.num_corners
        self.pvt_graph = PVT_Graph

        # 为每个角点创建独立的buffer
        self.corner_buffers = {}  # 格式: {corner_idx: {obs, info, reward}}
        
        # 总体reward和终止标志
        self.total_rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self._init_corner_buffer()

    def _init_corner_buffer(self):
        """为新的角点初始化buffer"""
        for corner_idx , corner_name in enumerate(self.pvt_graph.pvt_corners.keys()):
            self.corner_buffers[corner_idx] = {
                # 原始角点数据
                'name': corner_name,
                'obs': np.zeros([self.max_size, self.num_nodes, self.num_node_features], dtype=np.float32),
                'next_obs': np.zeros([self.max_size, self.num_nodes, self.num_node_features], dtype=np.float32),
                'info': np.zeros([self.max_size], dtype=object),
                'reward': np.zeros([self.max_size], dtype=np.float32),
                
                # PVT图状态
                'pvt_state': np.zeros([self.max_size, self.num_corners, self.pvt_dim], dtype=np.float32),
                'next_pvt_state': np.zeros([self.max_size, self.num_corners, self.pvt_dim], dtype=np.float32),
                
                # 动作相关
                'action': np.zeros([self.max_size, self.action_dim], dtype=np.float32),
                'corner_indices': [],  # 存储采样的角点索引列表
                'attention_weights': np.zeros([self.max_size, self.num_corners], dtype=np.float32),
                
                # 其他信息
                'total_reward': np.zeros([self.max_size], dtype=np.float32),
                'done': np.zeros([self.max_size], dtype=np.float32),
                
                'ptr': 0,  # 每个角点buffer的独立指针
                'size': 0  # 每个角点buffer的数据量
            }

    def store(
        self,
        pvt_state: np.ndarray,
        action: np.ndarray,
        results_dict: dict,
        next_pvt_state: np.ndarray,
        corner_indices: list,
        attention_weights: np.ndarray,
        total_reward: float,
        done: bool,
    ):
        """存储一个transition"""
        if len(attention_weights) != self.num_corners:
            attention_weights = np.pad(attention_weights, (0, self.num_corners - len(attention_weights)))
        
        # 存储每个角点的结果
        for corner_idx, result in results_dict.items():
            buffer = self.corner_buffers[corner_idx]
            ptr = buffer['ptr']
            
            # 存储角点特定数据
            buffer['obs'][ptr] = result['observation']
            buffer['next_obs'][ptr] = result['observation']  # 需要更新为真实的next_obs
            buffer['info'][ptr] = result['info']
            buffer['reward'][ptr] = result['reward']
            
            # 存储共享数据
            buffer['pvt_state'][ptr] = pvt_state
            buffer['next_pvt_state'][ptr] = next_pvt_state
            buffer['action'][ptr] = action
            buffer['corner_indices'].append(corner_indices)
            buffer['attention_weights'][ptr] = attention_weights
            buffer['total_reward'][ptr] = total_reward
            buffer['done'][ptr] = done
            
            # 更新指针和大小
            buffer['ptr'] = (ptr + 1) % self.max_size
            buffer['size'] = min(buffer['size'] + 1, self.max_size)

    def sample_corner_batch(self, corner_idx: int) -> Dict[str, np.ndarray]:
        """从特定角点的buffer中采样一批数据"""
        if corner_idx not in self.corner_buffers:
            return None
            
        buffer = self.corner_buffers[corner_idx]
        size = buffer['size']
        if size < self.batch_size:
            return None
            
        idxs = np.random.choice(size, size=self.batch_size, replace=False)
        
        return {
            # 角点特定数据
            'obs': buffer['obs'][idxs],
            'next_obs': buffer['next_obs'][idxs],
            'info': buffer['info'][idxs],
            'reward': buffer['reward'][idxs],
            # 共享数据
            'pvt_state': buffer['pvt_state'][idxs],
            'next_pvt_state': buffer['next_pvt_state'][idxs],
            'action': buffer['action'][idxs],
            'corner_indices': [buffer['corner_indices'][i] for i in idxs],
            'attention_weights': buffer['attention_weights'][idxs],
            'total_reward': buffer['total_reward'][idxs],
            'done': buffer['done'][idxs]
        }


class DDPGAgent:
    """DDPGAgent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        noise: noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(
        self,
        env,
        CktGraph,
        PVT_Graph,
        Actor,
        Critic,
        memory_size: int,
        batch_size: int,
        noise_sigma: float,
        noise_sigma_min: float,
        noise_sigma_decay: float,
        noise_type: str,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        sample_num: int = 3
    ):
        super().__init__()
        """Initialize."""
        self.noise_sigma = noise_sigma
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_decay = noise_sigma_decay
        self.action_dim = CktGraph.action_dim
        self.env = env
        self.memory = ReplayBuffer(CktGraph, PVT_Graph, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.sample_num = sample_num

        self.episode = 0
        self.device = CktGraph.device
        print(self.device)
        self.actor = Actor.to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic.to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=3e-4, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=3e-4, weight_decay=1e-4)
        self.transition = list()
        self.total_step = 0

        self.noise_type = noise_type
        self.is_test = False

        self.plots_dir = 'plots'  # 保存图片的文件夹
        self.plots_rewards_dir = 'plots_rewards'  # 保存图片的文件夹

        # 如果文件夹不存在，则创建
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir )

        if not os.path.exists(self.plots_rewards_dir):
            os.makedirs(self.plots_rewards_dir)


        # 添加PVT相关属性
        self.pvt_corners = {}  # 记录每个角点的性能
        self.best_corner = None
        self.best_corner_reward = -float('inf')

        # 添加归一化参数
        self.perf_norm_params = {
            'phase_margin': {'target': 60, 'scale': 60},
            # 'CL': {'target': 100, 'scale': 100}, 
            'dcgain': {'target': 130, 'scale': 130},
            'PSRP': {'target': -80, 'scale': 80},
            'PSRN': {'target': -80, 'scale': 80},
            'cmrrdc': {'target': -80, 'scale': 80}, 
            'vos': {'target': 0.06e-3, 'scale': 0.06e-3},
            'TC': {'target': 10e-6, 'scale': 10e-6},
            'settlingTime': {'target': 1e-6, 'scale': 1e-6},
            'FOML': {'target': 160, 'scale': 160},
            'FOMS': {'target': 300, 'scale': 300},
            'Active_Area': {'target': 150, 'scale': 150},
            'Power': {'target': 0.3, 'scale': 0.3},
            'GBW': {'target': 1.2e6, 'scale': 1.2e6},
            'sr': {'target': 0.6, 'scale': 0.6}
        }

        
        # PVT编码的归一化参数
        self.pvt_norm_params = {
            'vdd': {'min': 1.62, 'max': 1.98},  # ±10% of 1.8V
            'temp': {'min': -40, 'max': 125},   # 温度范围
            'process': {'min': 0, 'max': 1}     # process corners (0,1,2)
        }

        # 添加reward历史记录字典
        self.corner_rewards_history = {}
        # 为不同角点设置不同颜色
        self.corner_colors = plt.cm.rainbow(np.linspace(0, 1, self.actor.pvt_graph.num_corners))  # 27个角点的不同颜色

        # 添加跳过步数计数器
        self.skipped_steps = 0

    def _normalize_pvt_graph_state(self, state: torch.Tensor) -> torch.Tensor:
        """归一化PVT图状态
        
        Args:
            state: shape [num_corners, feature_dim]
            feature_dim包含: [vdd, temp, process, perf_1, perf_2,..., reward]
        """
        state = state.clone()  # 避免修改原始数据
        
        # 归一化PVT编码部分
        state[:, 6] = (state[:, 6] - self.pvt_norm_params['vdd']['min']) / (self.pvt_norm_params['vdd']['max'] - self.pvt_norm_params['vdd']['min'])
        state[:, 5] = (state[:, 5] - self.pvt_norm_params['temp']['min']) / (self.pvt_norm_params['temp']['max'] - self.pvt_norm_params['temp']['min'])
        
        # state[:, 2] = state[:, 2] / self.pvt_norm_params['process']['max']
        
        # 性能指标归一化 - 分类处理
        # 1. 相位裕度(phase margin): 通常在0-180度之间,60度为良好值
        state[:, 7] = torch.sigmoid((state[:, 7] - 45) / 15)  # 以60度为中心,使用sigmoid函数归一化
        
        # 2. 增益相关指标(dcgain, PSRP, PSRN, cmrrdc): 使用对数归一化
        state[:, 8] = torch.sigmoid((state[:, 8] - 120) / 10)  # dcgain
        state[:, 9] = torch.sigmoid((state[:, 9] + 80) / 10)  # PSRP 
        state[:, 10] = torch.sigmoid((state[:, 10] + 80) / 10)  # PSRN
        state[:, 11] = torch.sigmoid((state[:, 11] + 80) / 10)  # cmrrdc
        
        # 3. 时间相关指标(TC, settlingTime): 使用对数归一化
        state[:, 12] = torch.sigmoid((-torch.log10(state[:, 12]) - 5) / 0.5)  # TC
        state[:, 13] = torch.sigmoid((-torch.log10(state[:, 13]) - 6) / 0.5)  # settlingTime
        
        # 4. 偏移电压(vos): 越小越好
        state[:, 14] = torch.sigmoid((-torch.log10(state[:, 14]) - 3) / 0.5)  # vos
        
        # 5. 性能指标(FOML, FOMS): 越大越好
        state[:, 15] = torch.sigmoid((state[:, 15] - 150) / 10)   # FOML
        state[:, 16] = torch.sigmoid((state[:, 16] - 280) / 20) # FOMS
        
        # 6. 面积和功耗(Active_Area, Power): 越小越好
        state[:, 17] = torch.sigmoid((150 - state[:, 17]) / 15)  # Active_Area
        state[:, 18] = torch.sigmoid((0.3 - state[:, 18]) / 0.03)  # Power

        # 7. 带宽和压摆率(GBW, sr): 使用对数归一化
        state[:, 19] = torch.sigmoid((torch.log10(state[:, 19]) - 6) / 0.5)  # GBW
        state[:, 20] = torch.sigmoid((state[:, 20] - 0.5) / 0.1)  # sr
        
        # # 归一化性能指标 (中间维度)
        # perf_names = list(self.perf_norm_params.keys())
        # for i, perf_name in enumerate(perf_names):
        #     state[:, 3+i] = state[:, 3+i] / self.perf_norm_params[perf_name]['scale']
        
        # 归一化reward (最后一维)
        state[:, 21] = (state[:, 21] - (-10)) / (2 - (-10))  # 映射到[0,1]
        
        return state

    def select_action(self, pvt_graph_state: torch.Tensor) -> np.ndarray:
        """基于PVT图状态选择动作"""
        # 归一化状态
        normalized_state = self._normalize_pvt_graph_state(pvt_graph_state)
        
        if self.is_test == False:
            if self.total_step < self.initial_random_steps:
                print('*** Random actions ***')
                selected_action = np.random.uniform(-1, 1, self.action_dim)
            else:
                print(f'*** Actions with Noise sigma = {self.noise_sigma} ***')
                
                selected_action = self.actor(
                    normalized_state  # 使用归一化后的状态
                ).detach().cpu().numpy()
                selected_action = selected_action.flatten()
                if self.noise_type == 'uniform':
                    print(""" uniform distribution noise """)
                    selected_action = np.random.uniform(np.clip(
                        selected_action-self.noise_sigma, -1, 1), np.clip(selected_action+self.noise_sigma, -1, 1))

                if self.noise_type == 'truncnorm':
                    print(""" truncated normal distribution noise """)
                    selected_action = trunc_normal(selected_action, self.noise_sigma)
                    selected_action = np.clip(selected_action, -1, 1)
                
                self.noise_sigma = max(
                    self.noise_sigma_min, self.noise_sigma*self.noise_sigma_decay)

        else:   
            selected_action = self.actor(
                normalized_state  # 使用归一化后的状态
            ).detach().cpu().numpy()
            selected_action = selected_action.flatten()

        print(f'selected action: {selected_action}')
        self.transition = [normalized_state, selected_action]
        return selected_action

    # def step(self, action: np.ndarray, pvt_corners) -> Tuple[np.ndarray, Dict, bool]:
    #     """Take an action and return the response of the env."""
    #     next_state, rewards, terminated, truncated, infos = self.env.step(action, pvt_corners)

    #     # 记录每个角点的性能
    #     for corner_idx, reward in rewards.items():
    #         self.pvt_corners[corner_idx] = {
    #             'reward': reward,
    #             'performance': infos[corner_idx]['performance']
    #         }
            
    #         # 更新最佳角点
    #         if reward > self.best_corner_reward:
    #             self.best_corner = corner_idx
    #             self.best_corner_reward = reward

    #     if self.is_test == False:
    #         # 在transition中添加角点信息
    #         self.transition += [reward, next_state, terminated, info]
    #         self.memory.store(*self.transition)

    #     return next_state, rewards, terminated, truncated, infos

    def update_model(self) -> torch.Tensor:
        print("*** Update the model by gradient descent. ***")
        
        # 获取当前选择的角点
        _, corner_indices = self.actor.sample_corners(num_samples=self.sample_num)
        
        # 从每个选定的角点缓冲区采样数据
        corner_batches = {}
        for corner_idx in corner_indices:
            batch = self.memory.sample_corner_batch(corner_idx)
            if batch is not None:
                corner_batches[corner_idx] = batch
        
        if not corner_batches:  # 如果没有足够的数据
            return None, None
            
        # 计算critic loss
        critic_losses = []
        for corner_idx, batch in corner_batches.items():
            # 转换数据到tensor
            obs = torch.FloatTensor(batch['obs']).to(self.device)
            next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
            action = torch.FloatTensor(batch['action']).to(self.device)
            reward = torch.FloatTensor(batch['reward']).reshape(-1, 1).to(self.device)
            done = torch.FloatTensor(batch['done']).reshape(-1, 1).to(self.device)
            pvt_state = torch.FloatTensor(batch['pvt_state']).to(self.device)
            next_pvt_state = torch.FloatTensor(batch['next_pvt_state']).to(self.device)
            
            # # 计算目标Q值
            # masks = 1 - done
            # next_action = self.actor_target(next_pvt_state)
            # next_value = self.critic_target(next_obs, next_action)
            # curr_return = reward + self.gamma * next_value * masks
            
            # 计算当前Q值和loss
            values = self.critic(obs, action)
            critic_loss = F.mse_loss(values, reward)
            critic_losses.append(critic_loss)
        
        # 更新critic
        total_critic_loss = sum(critic_losses) / len(critic_losses)
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()
        
        # 计算actor loss
        actor_losses = []
        for corner_idx, batch in corner_batches.items():
            # 转换数据到tensor
            obs = torch.FloatTensor(batch['obs']).to(self.device)
            pvt_state = torch.FloatTensor(batch['pvt_state']).to(self.device)
            
            # 修改这里：获取attention_weight的方式
            attention_weights = []
            for i in range(len(batch['corner_indices'])):  # 遍历每个batch
                # 使用numpy的where函数找到corner_idx的位置
                corner_indices = np.array(batch['corner_indices'][i])  # 确保是numpy数组
                idx_positions = np.where(corner_indices == corner_idx)[0] 
                if len(idx_positions) > 0:
                    # 如果找到了corner_idx，使用第一个位置的权重
                    idx = idx_positions[0]
                    weight = batch['attention_weights'][i][idx]
                else:
                    # 如果没找到，使用0权重
                    print(f"Corner index {corner_idx} not found in batch {i}")
                attention_weights.append(weight)
            
            attention_weight = torch.FloatTensor(attention_weights).reshape(-1, 1).to(self.device)
            
            # 计算加权policy loss
            value = self.critic(obs, self.actor(pvt_state))
            actor_loss = -(attention_weight * value).mean()
            actor_losses.append(actor_loss)
        
        # 更新actor
        total_actor_loss = sum(actor_losses) / len(actor_losses)
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        # self._target_soft_update()
        
        return total_actor_loss.data, total_critic_loss.data

    def train(self, num_steps: int, plotting_interval: int = 1):
        """Train the agent."""
        self.is_test = False           
        results_dict= self.env.reset()
        
        # 更新PVT图
        for corner_idx, result in results_dict.items():
            self.actor.pvt_graph.update_performance_and_reward_r(
                corner_idx,
                result['info'],
                result['reward']
            )
            
        pvt_graph_state = self.actor.pvt_graph.get_graph_features()
        
        actor_losses = []              
        critic_losses = []
        scores = []
        score = 0

        # 初始化reward历史记录
        for corner_name in self.actor.pvt_graph.pvt_corners.keys():
            self.corner_rewards_history[corner_name] = []

        for self.total_step in range(1, num_steps + 1):
            print(f'*** Step: {self.total_step} | Episode: {self.episode} ***')
            
            # 使用PVT图状态选择动作
            action = self.select_action(pvt_graph_state)
            
            if self.total_step >= self.initial_random_steps:
                attention_weights, corner_indices = self.actor.sample_corners(num_samples=self.sample_num)
                print(f'*** corner_indices: {corner_indices} ***')
            else:
                # 在随机探索阶段,使用所有角点
                corner_indices = np.arange(self.actor.num_PVT)
                # 创建均匀分布的权重,每个角点权重相等
                num_corners = len(corner_indices)
                attention_weights = torch.ones(num_corners, device=self.device) / num_corners  # 每个角点权重为1/总角点数
            # 在采样的角点上执行动作
            results_dict, reward_no, terminated, truncated, info = self.env.step((action, corner_indices))
            
            # 添加错误处理
            if results_dict is None:
                print("Warning: results_dict is None, skipping this step")
                self.skipped_steps += 1  # 增加计数
                continue
                
            # 更新PVT图并存储转换
            for corner_idx, result in results_dict.items():
                # 更新PVT图中角点的性能和reward
                self.actor.update_pvt_performance(
                    corner_idx,
                    result['info'],
                    result['reward']
                )
                
            # 更新reward历史记录
            for idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
                if idx in results_dict:  # 如果这个角点在当前step被采样
                    reward = results_dict[idx]['reward']
                else:  # 如果没有被采样，使用上一次的值
                    reward = self.corner_rewards_history[corner_name][-1]
                self.corner_rewards_history[corner_name].append(reward)

            # 更新状态
            pvt_graph_state = self.actor.pvt_graph.get_graph_features()
            normalized_state = self._normalize_pvt_graph_state(pvt_graph_state)

            total_reward = 0
            # 遍历采样的角点及其对应的权重
            for weight, corner_idx in zip(attention_weights, corner_indices):
                # 获取该角点的 reward
                reward = results_dict[corner_idx]['reward']
                # 加权累加
                total_reward += weight * reward

            print(f'*** total_reward: {total_reward} ***')

            if total_reward > 0:  # 使用最佳角点的reward判断是否终止
                terminated = True
            else:
                terminated = False

            score += total_reward

            # 存储转换,包括每个角点的信息
            if not self.is_test:
                self.transition += [
                    results_dict,
                    normalized_state,
                    corner_indices,
                    attention_weights,
                    total_reward,
                    terminated
                ]
                self.memory.store(*self.transition)
                
            if terminated or truncated:
                results_dict = self.env.reset()
                # 更新PVT图
                for corner_idx, result in results_dict.items():
                    self.actor.update_pvt_performance(
                        corner_idx,
                        result['info'],
                        result['reward']
                    )
                
                pvt_graph_state = self.actor.pvt_graph.get_graph_features()
                self.episode += 1
                scores.append(score)
                score = 0

            print(f'*** The progress of the PVT graph ***')
            # 打印PVT图进度
            print("\nPVT Graph Rewards:")
            for corner_idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
                reward = pvt_graph_state[corner_idx][21]  # reward在第21维
                print(f"{corner_name}: reward = {reward:.4f}")
            print()

            # 如果满足训练条件则更新模型
            if  self.total_step > self.initial_random_steps:
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # 绘图
            if self.total_step % plotting_interval == 0:
                self._plot(
                    self.total_step,
                    scores,
                    actor_losses,
                    critic_losses,
                )
            self.plot_corner_rewards()

        print(f"\nTraining completed:")
        print(f"Total steps: {num_steps}")
        print(f"Skipped steps: {self.skipped_steps}")
        
        self.env.close()

    def test(self):
        """Test the agent."""
        self.is_test = True
        results_dict = self.env.reset()
        
        # 更新PVT图
        for corner_idx, result in results_dict.items():
            self.actor.pvt_graph.update_performance_and_reward_r(
                corner_idx,
                result['info'],
                result['reward']
            )
            
        pvt_graph_state = self.actor.pvt_graph.get_graph_features()
        truncated = False
        terminated = False
        score = 0
        
        performance_list = []
        while not (truncated or terminated):    
            # 选择动作
            action = self.select_action(pvt_graph_state)
            
            # 采样PVT角点并获取注意力权重
            attention_weights, corner_indices = self.actor.sample_corners(num_samples=self.sample_num)
            
            # 在采样的角点上执行动作 - 修改这里，分别传入两个参数
            results_dict, terminated, truncated = self.env.step((action, corner_indices))
            
            # 更新PVT图
            for corner_idx, result in results_dict.items():
                self.actor.update_pvt_performance(
                    corner_idx,
                    result['info'],
                    result['reward']
                )
            
            # 记录性能
            performance_list.append({
                'action': action,
                'corner_indices': corner_indices,
                'attention_weights': attention_weights,
                'results': {idx: {
                    'info': result['info'],
                    'reward': result['reward']
                } for idx, result in results_dict.items()}
            })
            
            # 计算加权reward
            total_reward = 0
            for weight, corner_idx in zip(attention_weights, corner_indices):
                reward = results_dict[corner_idx]['reward']
                total_reward += weight * reward
            
            # 更新状态和分数
            pvt_graph_state = self.actor.pvt_graph.get_graph_features()
            score += total_reward
            
        print(f"score: {score}")
        print("Performance in each corner:")
        for corner_idx, result in results_dict.items():
            print(f"Corner {corner_idx}:")
            print(f"Info: {result['info']}")
            print(f"Reward: {result['reward']}")
            
        self.env.close()
        return performance_list

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau      
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
        self,
        step: int,
        scores: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
    ):
        """Plot the training progresses."""

        
        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"step {step}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        
        # 保存图像
        filename = os.path.join(self.plots_dir, f"step_{step:06d}.png")
        plt.savefig(filename)
        plt.close('all')  # 确保关闭所有图形

    def plot_corner_rewards(self):
        """绘制PVT角点reward变化图"""
        plt.figure(figsize=(10, 6))
        plt.title('PVT Corner Rewards vs Steps')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        
        for idx, (corner_name, rewards) in enumerate(self.corner_rewards_history.items()):
            plt.plot(range(len(rewards)), rewards, 
                    label=corner_name, 
                    color=self.corner_colors[idx],
                    alpha=0.7)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # 保存图像
        filename = os.path.join(self.plots_rewards_dir, f"corner_rewards_step_{self.total_step:06d}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def load_replay_buffer(self, buffer_path):
        """加载已保存的replay buffer并转移数据到当前agent的buffer中"""
        if not os.path.exists(buffer_path):
            print(f"No saved buffer found at {buffer_path}")
            return
            
        print(f"\nLoading replay buffer from {buffer_path}")
        
        # 打印当前buffer的角点信息
        print("\nCurrent buffer corners:")
        for idx, corner_name in enumerate(self.actor.pvt_graph.pvt_corners.keys()):
            print(f"Index {idx}: {corner_name}")
            
        # 加载保存的buffer
        with open(buffer_path, 'rb') as f:
            saved_buffer = pickle.load(f)
        
        # 打印保存的buffer的角点信息    
        print("\nSaved buffer corners:")
        for corner_idx, saved_corner_buffer_idx in enumerate(saved_buffer.corner_buffers.keys()):
            print(f"Index {corner_idx}: {saved_corner_buffer_idx} : {saved_buffer.corner_buffers[saved_corner_buffer_idx]['name']}")
            
        # 转移数据到新的buffer
        for corner_idx, saved_corner_buffer in saved_buffer.corner_buffers.items():
            buffer = self.memory.corner_buffers[corner_idx]
            
            # 复制所有数据
            buffer['obs'][:saved_corner_buffer['size']] = saved_corner_buffer['obs'][:saved_corner_buffer['size']]
            buffer['next_obs'][:saved_corner_buffer['size']] = saved_corner_buffer['next_obs'][:saved_corner_buffer['size']]
            buffer['info'][:saved_corner_buffer['size']] = saved_corner_buffer['info'][:saved_corner_buffer['size']]
            buffer['reward'][:saved_corner_buffer['size']] = saved_corner_buffer['reward'][:saved_corner_buffer['size']]
            buffer['pvt_state'][:saved_corner_buffer['size']] = saved_corner_buffer['pvt_state'][:saved_corner_buffer['size']]
            buffer['next_pvt_state'][:saved_corner_buffer['size']] = saved_corner_buffer['next_pvt_state'][:saved_corner_buffer['size']]
            buffer['action'][:saved_corner_buffer['size']] = saved_corner_buffer['action'][:saved_corner_buffer['size']]
            buffer['corner_indices'] = saved_corner_buffer['corner_indices'][:saved_corner_buffer['size']]
            buffer['attention_weights'][:saved_corner_buffer['size']] = saved_corner_buffer['attention_weights'][:saved_corner_buffer['size']]
            buffer['total_reward'][:saved_corner_buffer['size']] = saved_corner_buffer['total_reward'][:saved_corner_buffer['size']]
            buffer['done'][:saved_corner_buffer['size']] = saved_corner_buffer['done'][:saved_corner_buffer['size']]
            
            buffer['ptr'] = saved_corner_buffer['ptr']
            buffer['size'] = saved_corner_buffer['size']
            
            print(f"Loaded data for corner {corner_idx} ({saved_corner_buffer['name']}) to ({buffer['name']})")
            print(f"  Size: {buffer['size']}")
            
        print(f"\nSuccessfully loaded replay buffer with {len(saved_buffer.corner_buffers)} corners")