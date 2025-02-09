import numpy as np

from typing import List, Tuple, Dict

import torch
from torch.nn import LazyLinear
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, Linear
import torch.nn.functional as F   
import torch.optim as optim

from utils import trunc_normal

from IPython.display import clear_output
import matplotlib.pyplot as plt

import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class ActorCriticPVTGAT:
    class GuidedGATConv(MessagePassing):
        def __init__(self, in_channels, out_channels, heads, layer_idx ,concat=True,
                     dropout=0, feature_guidance_weight=0, **kwargs):
            super().__init__(node_dim=0, **kwargs)
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.heads = heads
            self.concat = concat
            self.dropout = dropout
            self.layer_idx = layer_idx
            self.feature_guidance_weight = feature_guidance_weight
            self.output_dim = heads * out_channels if concat else out_channels

            # 网络层
            self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
            self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))
            
            # 初始化
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.xavier_uniform_(self.att_l)
            nn.init.xavier_uniform_(self.att_r)
            
            # 存储注意力权重
            self.attention_weights = None

        def forward(self, x, edge_index):
            # 确保edge_index是tensor类型
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, device=x.device)
            
            reward_weights = None
            if self.layer_idx == 0:  # 仅第一层计算reward权重
                rewards = x[:, -1:]
                reward_weights = 1.0 / (torch.abs(rewards)**3 + 1e-6)
                reward_weights = F.normalize(reward_weights, p=1, dim=0)


            x = self.lin(x).view(-1, self.heads, self.out_channels)
            return self.propagate(edge_index, x=x, reward_weights=reward_weights)

        def message(self, edge_index_i, x_i, x_j, reward_weights_j, size_i):
            # 计算注意力分数
            alpha = (x_i * self.att_l).sum(-1) + (x_j * self.att_r).sum(-1)
            alpha = F.leaky_relu(alpha)
            
            if reward_weights_j is not None:
                # 确保reward_weights_j维度正确
                if reward_weights_j.dim() == 1:
                    reward_weights_j = reward_weights_j.unsqueeze(-1)
            
                guided_alpha = alpha + self.feature_guidance_weight * reward_weights_j

                alpha = softmax(guided_alpha, edge_index_i, num_nodes=size_i)
                self.attention_weights = alpha.detach().mean(dim=1)
            else:
                guided_alpha = alpha

                alpha = softmax(guided_alpha, edge_index_i, num_nodes=size_i)
            

            
            # Dropout
            # if self.training and self.dropout > 0:
            #     alpha = F.dropout(alpha, p=self.dropout, training=True)

            return x_j * alpha.view(-1, self.heads, 1)

        def update(self, aggr_out):
            if self.concat:
                aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
            else:
                aggr_out = aggr_out.mean(dim=1)
            return aggr_out

    class GATActor(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim, heads=4, dropout=0, guidance_weight=1.0):
            """
            Args:
                input_dim: 输入特征维度
                hidden_dims: 隐藏层维度列表, 如[32, 32, 16]
                output_dim: 输出维度
                heads: 注意力头数
                dropout: Dropout率
                guidance_weight: 注意力引导权重
            """
            super().__init__()
            
            # 构建GAT层
            self.layers = nn.ModuleList()
            dims = [input_dim] + hidden_dims
            
            for i in range(len(dims)-1):
                self.layers.append(
                    ActorCriticPVTGAT.GuidedGATConv(
                        dims[i] * (heads if i > 0 else 1),
                        dims[i+1],
                        heads=heads,
                        layer_idx=i,
                        concat=(i < len(dims)-2),  # 最后一层不concat
                        dropout=dropout,
                        feature_guidance_weight=guidance_weight
                    )
                )
            
            # 输出层
            # last_dim = hidden_dims[-1] * (heads if len(hidden_dims) == 1 else 1)
            self.fc = LazyLinear(output_dim)
            # self.dropout = dropout
        

        def forward(self, x, edge_index):
            for layer in self.layers:
                x = layer(x, edge_index)
                x = F.elu(x)
            x = self.fc(torch.flatten(x))
            x = torch.tanh(x)    
            return x
        
        def get_attention_weights(self):
            """获取注意力权重"""
            # 只返回最后一层的注意力权重
            return self.layers[0].attention_weights


    class Actor(nn.Module):
        def __init__(self, CktGraph, PVT_Graph, hidden_dims=[32, 64, 128, 64, 32 ], heads=4):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
            self.num_PVT = PVT_Graph.num_corners
            self.edge_index_PVT = PVT_Graph.edge_index
            self.node_features_PVT = PVT_Graph.node_features
            self.pvt_graph = PVT_Graph  # 保存PVT_Graph实例
            
            # GAT网络配置
            self.gat = ActorCriticPVTGAT.GATActor(
                input_dim=self.node_features_PVT.shape[1],  # PVT图节点特征维度
                hidden_dims=hidden_dims,
                output_dim=self.action_dim,  # 直接输出action_dim维度
                heads=heads,
                dropout=0,
                guidance_weight=1.5##here
            )
            
            # 当前选中的角点
            self.current_corner_idx = None
            
            # 添加初始注意力权重
            self.attention_weights = None
        
        def forward(self, state):
            if len(state.shape) == 2:  # 如果不是batch数据
                state = state.reshape(1, state.shape[0], state.shape[1])
            
            batch_size = state.shape[0]
            edge_index_PVT = self.edge_index_PVT
            device = self.device
            actions = torch.tensor(()).to(device)
            
            for i in range(batch_size):
                x = state[i]
                # 通过GAT网络生成动作
                action = self.gat(x, edge_index_PVT)  # 已经是正确维度的动作
                action = action.reshape(1, -1)
                actions = torch.cat((actions, action), axis=0)
            
            return actions  # 直接返回GAT的输出，它已经经过tanh激活
        
        def sample_corners(self, num_samples=3):
            """基于注意力权重采样PVT角点"""

            
            # 获取最后一层的注意力权重
            attention_weights = self.gat.get_attention_weights()
            
            # 创建节点重要性得分张量
            node_importance = torch.zeros(self.num_PVT, device=attention_weights.device)
            
            # 只计算每个节点作为源节点的重要性
            for node in range(self.num_PVT):
                # 获取所有从该节点发出的边
                source_edges = (self.edge_index_PVT[0] == node).nonzero().squeeze()
                # 计算平均注意力权重（如果有出边的话）
                if len(source_edges.shape) > 0:
                    node_importance[node] = attention_weights[source_edges].mean()
            
            # 获取前num_samples个最重要的节点
            top_values, indices = torch.topk(node_importance, num_samples)
            
            return top_values, indices.cpu().numpy()  # 返回top_values和indices
        
        def update_pvt_performance(self, corner_idx, performance, reward):
            """更新PVT图中角点的性能和reward"""
            self.pvt_graph.update_performance_and_reward(corner_idx, performance, reward)

        def update_pvt_performance_r(self, corner_idx, performance, reward):
            """强制更新PVT图中角点的性能和reward"""
            self.pvt_graph.update_performance_and_reward_r(corner_idx, performance, reward)
            
        def get_current_corner(self):
            """获取当前选中的角点"""
            return self.current_corner_idx
            
        def set_current_corner(self, corner_idx):
            """设置当前选中的角点"""
            self.current_corner_idx = corner_idx
            
        def get_best_corner(self):
            """获取最佳性能的角点"""
            return self.pvt_graph.get_best_corner()

        def get_pvt_state(self):
            """获取PVT图的状态特征"""
            return self.pvt_graph.get_graph_features()

        
    class Critic(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            # self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            # self.edge_index = CktGraph.edge_index
            # self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.action_dim + 7
            self.out_channels = 1
            self.mlp1 = Linear(self.in_channels, 64)
            self.mlp2 = Linear(64, 128)
            self.mlp3 = Linear(128, 64)
            self.mlp4 = Linear(64, 32)
            self.mlp5 = Linear(32,16)
            self.mlp6 = Linear(16,8)
            self.mlp7 = Linear(8, self.out_channels)
    
        def forward(self, pvt_corner, action):
            batch_size = action.shape[0]
            device = self.device

            data = torch.cat((action, pvt_corner), axis=1)
            
    
            values = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.mlp1(x))
                x = F.relu(self.mlp2(x))
                x = F.relu(self.mlp3(x))
                x = F.relu(self.mlp4(x))
                x = F.relu(self.mlp5(x))
                x = F.relu(self.mlp6(x))
                x = F.relu(self.mlp6(x))
                values = torch.cat((values, x), axis=0)
    
            return values
        
        # def forward(self, pvt_corner, action):
        #     batch_size = action.shape[0]
        #     device = self.device
            
        #     # 处理pvt_corner
        #     pvt_corner = torch.tensor(pvt_corner, device=self.device)  # [7]
        #     pvt_corner = pvt_corner.unsqueeze(0)  # [1, 7]
        #     pvt_corner = pvt_corner.repeat(batch_size, 1)  # [batch_size, 7]
            
        #     # 连接输入
        #     data = torch.cat((action, pvt_corner), axis=1)  # [batch_size, action_dim + 7]
            
        #     # 直接对整个batch进行前向传播 
        #     x = F.relu(self.mlp1(data))
        #     x = F.relu(self.mlp2(x))
        #     x = F.relu(self.mlp3(x))
        #     x = F.relu(self.mlp4(x))
        #     x = F.relu(self.mlp5(x))
        #     x = F.relu(self.mlp6(x))
            
        #     return x