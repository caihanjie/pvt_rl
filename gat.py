import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#gat加权回归 注意节点提取：简单数据集

# GuidedGATConv层的实现保持不变
class GuidedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, concat=True,
                 dropout=0.2, feature_guidance_weight=0, **kwargs):
        super(GuidedGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.feature_guidance_weight = feature_guidance_weight

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if concat:
            self.output_dim = heads * out_channels
        else:
            self.output_dim = out_channels

        self.reset_parameters()

        # 添加一个属性来存储注意力权重
        self.attention_weights = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index):
        # 获取reward值(最后一维)
        rewards = x[:, -1:]
        
        # 计算reward引导权重 - reward越小权重越大
        reward_weights = 1.0 / (torch.abs(rewards) + 1e-6)
        reward_weights = F.normalize(reward_weights, p=1, dim=0)

        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)

        return self.propagate(edge_index, x=x,
                            reward_weights=reward_weights,
                            size=None)

    def message(self, edge_index_i, x_i, x_j, reward_weights_j, size_i):
        alpha = (x_i * self.att_l).sum(-1) + (x_j * self.att_r).sum(-1)
        alpha = F.leaky_relu(alpha)

        # 使用reward引导注意力
        guided_alpha = alpha + self.feature_guidance_weight * torch.log(reward_weights_j)
        alpha = softmax(guided_alpha, edge_index_i, size_i)
        
        # 保存注意力权重
        self.attention_weights = alpha.detach().mean(dim=1)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        return aggr_out


# 修改后的GNN模型，用于回归任务
class GuidedGATRegression(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.2, feature_guidance_weight=0):
        super(GuidedGATRegression, self).__init__()

        # 第一层GAT
        self.conv1 = GuidedGATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            feature_guidance_weight=feature_guidance_weight,

        )

        # 第二层GAT
        self.conv2 = GuidedGATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            feature_guidance_weight=feature_guidance_weight,

        )

        # 添加一个全连接层来生成最终的输出向量
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, x, edge_index):
        # 第一层GAT
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层GAT
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # 最终的全连接层，不需要激活函数因为是回归任务
        x = self.fc(x)

        return x

    def get_attention_weights(self):
        """获取两层的注意力权重"""
        return {
            'layer1': self.conv1.attention_weights,
            'layer2': self.conv2.attention_weights
        }


# 生成用于回归任务的随机图数据
def generate_random_graph_data(num_nodes=5, num_features=4, output_dim=3, avg_degree=4):
    """
    生成随机图数据，目标是一个output_dim维的向量
    """
    # 生成节点特征（刻意使一些节点的特征值较小）
    x = torch.randn(num_nodes, num_features)
    small_features_mask = torch.randint(0, 2, (num_nodes,)).bool()
    x[small_features_mask] *= 0.1

    # 生成随机边
    num_edges = int(num_nodes * avg_degree)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # 生成目标向量（使用一个简单的非线性函数生成）
    y = torch.sin(x[:, :output_dim] * 0.5) + torch.cos(x[:, :output_dim] * 0.3)
    y = y + torch.randn_like(y) * 0.1  # 添加一些噪声

    # 创建PyG数据对象
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


# 训练函数
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    # 计算预测值
    pred = model(data.x, data.edge_index)
    # 使用MSE损失
    loss = F.mse_loss(pred, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


# 评估函数
@torch.no_grad()
def evaluate(model, data, mask=None):
    model.eval()
    pred = model(data.x, data.edge_index)
    if mask is not None:
        pred = pred[mask]
        y = data.y[mask]
    else:
        y = data.y
    # 计算MSE和MAE
    mse = F.mse_loss(pred, y).item()
    mae = torch.mean(torch.abs(pred - y)).item()
    return mse, mae


def analyze_attention(model, edge_index, epoch, num_nodes):
    """分析最后一层的注意力权重，找出最重要的节点"""
    attention_weights = model.conv2.attention_weights  # 只获取第二层的权重
    
    if attention_weights is not None:
        # 创建节点重要性得分张量
        node_importance = torch.zeros(num_nodes)
        
        # 只计算每个节点作为源节点的重要性
        for node in range(num_nodes):
            # 获取所有从该节点发出的边
            
            source_edges = (edge_index[0] == node).nonzero().squeeze()
            # 计算平均注意力权重（如果有出边的话）
            if len(source_edges.shape) > 0:
                node_importance[node] = attention_weights[source_edges].mean()
        
        # 获取前3个最重要的节点
        top_k = 3
        top_values, top_indices = torch.topk(node_importance, min(top_k, num_nodes))
        
        print(f"\n在 epoch {epoch} 的前{top_k}个最重要节点:")
        for node_idx, importance in zip(top_indices, top_values):
            print(f"节点 {node_idx.item()}, 重要性得分: {importance:.4f}")
            # 打印该节点的度
            out_degree = sum(edge_index[0] == node_idx.item()).item()
            print(f"   出度: {out_degree}")


def main():
    # 设置随机种子确保可复现性
    torch.manual_seed(12345)
    np.random.seed(12345)

    # 设置输出向量维度
    output_dim = 3

    # 生成数据
    data = generate_random_graph_data(output_dim=output_dim)

    # 划分训练集和测试集
    num_nodes = data.x.size(0)
    node_indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(node_indices, test_size=0.2, random_state=42)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # 创建模型
    model = GuidedGATRegression(
        input_dim=data.x.size(1),
        hidden_dim=32,
        output_dim=output_dim,
        heads=4,
        dropout=0, # 回归任务通常使用较小的dropout
        feature_guidance_weight = 10
    )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 训练循环
    epochs = 200
    train_losses = []
    train_mses = []
    test_mses = []

    print(f"开始训练... 目标是预测{output_dim}维向量")
    for epoch in range(epochs):
        # 训练
        loss = train(model, data, optimizer)

        # 评估
        train_mse, train_mae = evaluate(model, data, train_mask)
        test_mse, test_mae = evaluate(model, data, test_mask)

        # 记录指标
        train_losses.append(loss)
        train_mses.append(train_mse)
        test_mses.append(test_mse)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}, '
                  f'Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, '
                  f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')
            # 分析注意力权重
        analyze_attention(model, data.edge_index, epoch + 1, num_nodes)

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')

    plt.subplot(1, 2, 2)
    plt.plot(train_mses, label='Train MSE')
    plt.plot(test_mses, label='Test MSE')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 输出一些预测样本进行对比
    with torch.no_grad():
        model.eval()
        pred = model(data.x, data.edge_index)
        print("\n预测样本对比 (前5个测试节点):")
        test_indices = test_idx[:5]
        print("真实值:")
        print(data.y[test_indices].numpy())
        print("预测值:")
        print(pred[test_indices].numpy())


if __name__ == '__main__':
    main()