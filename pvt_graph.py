import torch
import numpy as np
import os
import shutil

class PVTGraph:
    def __init__(self):
        # PVT角点定义
        self.pvt_corners = {
            'tt_027C_1v80': [1, 0, 0, 0, 0, 27 ,1.8],  
            'ff_027C_1v80': [0, 1, 0, 0, 0, 27 ,1.8],
            'ss_027C_1v80': [0, 0, 1, 0, 0, 27 ,1.8]
            # 'fs_027C_1v80': [0, 0, 0, 1, 0, 27 ,1.8],
            # 'sf_027C_1v80': [0, 0, 0, 0, 1, 27 ,1.8]
        }
        
        self.PWD = os.getcwd()
        self.SPICE_NETLIST_DIR = f'{self.PWD}/simulations'
        
        
        
        # 初始化PVT图的节点特征
        self.num_corners = len(self.pvt_corners)
        self.corner_dim = 22
        self.node_features = np.zeros((self.num_corners, self.corner_dim))  
        
        # 初始化每个角点的特征
        for i, (corner, pvt_code) in enumerate(self.pvt_corners.items()):
            self.node_features[i, :7] = pvt_code  # PVT编码
            # 初始化性能指标和reward为最差值
            self.node_features[i, 7:21] = -np.inf  # 性能指标
            self.node_features[i, 21] = -np.inf    # reward
            
        # 定义图的边 - 完全图
        edges = []
        for i in range(self.num_corners):
            for j in range(self.num_corners):
                if i != j:
                    edges.append([i, j])
        self.edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # 设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        


    def _clean_pvt_dirs(self):
        # 定义需要清理的目录前缀
        corner_prefixes = ['ss', 'ff', 'tt', 'sf', 'fs']
        
        # 遍历指定目录中的所有文件夹
        for corner in os.listdir(self.SPICE_NETLIST_DIR):
            # 获取完整的目录路径
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            if os.path.isdir(corner_dir) and any(corner.startswith(prefix) for prefix in corner_prefixes):
                print(f"Removing existing corner directory: {corner_dir}")
                shutil.rmtree(corner_dir)

    def _create_pvt_dirs(self):
        """为每个PVT角点创建目录"""
        for corner in self.pvt_corners.keys():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            os.makedirs(corner_dir)
            
            # 创建.spiceinit文件
            spiceinit_content ="""* ngspice initialization for sky130
* assert BSIM compatibility mode with "nf" vs. "W"
set ngbehavior=hsa
* "nomodcheck" speeds up loading time
set ng_nomodcheck
set num_threads=8"""
            
            spiceinit_path = os.path.join(corner_dir, '.spiceinit')
            with open(spiceinit_path, 'w') as f:
                f.write(spiceinit_content)
                
    def _create_pvt_netlists(self):
        """为每个PVT角点创建仿真文件"""
        # 读取原始网表文件
        with open(f'{self.SPICE_NETLIST_DIR}/AMP_NMCF_ACDC.cir', 'r') as f:
            netlist_content = f.readlines()
            
        # 为每个角点创建网表
        for corner, params in self.pvt_corners.items():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            # 复制必要的文件
            self._copy_support_files(corner_dir)
            
            # 修改网表内容
            corner_netlist = []

            # 解析角点名称获取工艺角
            process = corner.split('_')[0]  # tt/ff/ss/fs/sf
            
            # 遍历原始网表内容并根据PVT参数修改
            for line in netlist_content:
                if line.startswith('.temp'):
                    # 修改温度
                    corner_netlist.append(f'.temp {params[5]}\n')
                elif line.startswith('.include') and 'tt.spice' in line:
                    # 修改工艺角路径
                    corner_netlist.append(f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/{process}.spice\n')
                elif line.startswith('.PARAM supply_voltage'):
                    # 修改电源电压
                    corner_netlist.append(f'.PARAM supply_voltage = {params[6]}\n')
                else:
                    corner_netlist.append(line)
            corner_netlist.insert(1, f'* PVT Corner: {corner}\n')
            # 保存修改后的网表
            netlist_path = os.path.join(corner_dir, f'AMP_NMCF_ACDC_{corner}.cir')
            with open(netlist_path, 'w') as f:
                f.writelines(corner_netlist)
    
    def _create_pvt_netlists_tran(self):
        """为每个PVT角点创建仿真文件"""
        # 读取原始网表文件
        with open(f'{self.SPICE_NETLIST_DIR}/AMP_NMCF_Tran.cir', 'r') as f:
            netlist_content = f.readlines()
            
        # 为每个角点创建网表
        for corner, params in self.pvt_corners.items():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            # 复制必要的文件
            self._copy_support_files(corner_dir)
            
            # 修改网表内容
            corner_netlist = []

            # 解析角点名称获取工艺角
            process = corner.split('_')[0]  # tt/ff/ss/fs/sf
            
            # 遍历原始网表内容并根据PVT参数修改
            for line in netlist_content:
                if line.startswith('.temp'):
                    # 修改温度
                    corner_netlist.append(f'.temp {params[5]}\n')
                elif line.startswith('.include') and 'tt.spice' in line:
                    # 修改工艺角路径
                    corner_netlist.append(f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/{process}.spice\n')
                elif line.startswith('.PARAM supply_voltage'):
                    # 修改电源电压
                    corner_netlist.append(f'.PARAM supply_voltage = {params[6]}\n')
                else:
                    corner_netlist.append(line)
            corner_netlist.insert(1, f'* PVT Corner: {corner}\n')
            # 保存修改后的网表
            netlist_path = os.path.join(corner_dir, f'AMP_NMCF_Tran_{corner}.cir')
            with open(netlist_path, 'w') as f:
                f.writelines(corner_netlist)

    def _copy_support_files(self, corner_dir):
        """复制支持文件到角点目录"""
        support_files = [
            # 'AMP_NMCF_vars.spice',
            # 'AMP_NMCF.sp',
            # 添加其他需要的支持文件
        ]
        
        for file in support_files:
            src = os.path.join(self.SPICE_NETLIST_DIR, file)
            dst = os.path.join(corner_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
    def get_corner_netlist_path(self, corner_idx):
        """获取指定角点的网表路径"""
        corner_name = list(self.pvt_corners.keys())[corner_idx]
        return os.path.join(self.SPICE_NETLIST_DIR, corner_name, f'AMP_NMCF_ACDC_{corner_name}.cir')

    def update_performance_and_reward(self, corner_idx, new_performance, new_reward):
        """
        更新指定角点的最佳性能和reward
        corner_idx: PVT角点的索引
        new_performance: 新的性能指标列表 [phase_margin, dcgain, PSRP, ...]
        new_reward: 新的reward值
        """
        current_reward = self.node_features[corner_idx, 17]
        # 只在新reward更好时更新性能和reward
        if new_reward > current_reward:
            self.node_features[corner_idx, 7:21] = new_performance  # 更新性能指标
            self.node_features[corner_idx, 21] = new_reward        # 更新reward

    def update_performance_and_reward_r(self, corner_idx, info_dict, reward):
        """
        强制更新指定角点的性能和reward
        corner_idx: PVT角点的索引
        info_dict: 性能指标字典 {'phase_margin': float, 'dcgain': float, ...}
        reward: 新的reward值
        """
        # 从info字典中提取性能指标列表
        performance = list(info_dict.values())
        self.node_features[corner_idx, 7:21] = performance  # 更新性能指标
        self.node_features[corner_idx, 21] = reward        # 更新reward

    def get_corner_name(self, idx):
        """根据索引获取角点名称"""
        return list(self.pvt_corners.keys())[idx]
    
    def get_corner_idx(self, corner_name):
        """根据角点名称获取索引"""
        return list(self.pvt_corners.keys()).index(corner_name)

    def get_best_corner(self):
        """
        获取当前reward最高的角点
        返回: (corner_idx, best_reward)
        """
        rewards = self.node_features[:, 21]
        best_idx = np.argmax(rewards)
        return best_idx, rewards[best_idx]

    def get_graph_features(self):
        """
        获取PVT图的特征表示
        返回: 节点特征矩阵(torch.Tensor)
        """
        # 返回转换为tensor的节点特征矩阵
        return torch.tensor(self.node_features, dtype=torch.float32).to(self.device) 