import os
import time
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# 导入自定义模块
import overlap_block
import bond_embedding
import atom_embedding
import nodes_embedding
import edges_embedding
import EGAT_Layer
import GAT_Layer
from Hamiltonian_block import Hamiltonian_diagonal_block, Hamiltonian_off_diagonal_block


class HamiltonianBlockGenLayer(nn.Module):
    def __init__(self, mol2, nodes_feature, edges_feature, atom_block, atom_in_block):
        super().__init__()
        self.mol2 = mol2
        self.nodes_feature = nodes_feature
        self.edges_feature = edges_feature
        self.atom_block = atom_block
        self.atom_in_block = atom_in_block
        self.total_orbitals = sum(block.shape[0] for block in atom_block)
        
        # 原子嵌入与周期划分
        self.atom_embedding = atom_embedding.Atom_Embedding(mol2)
        self.atom_type_list = self.atom_embedding.get_atom_type()
        self.atom_pairs, self.atom_pairs_index = overlap_block.overlap_block_dec(mol2).get_atom_pairs()
        
        # 定义周期原子符号
        self.atomic_symbol = [
            'H', 'He',  # Period 1
            'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',  # Period 2
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',  # Period 3
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'  # Period 4
        ]
        self.atomic_symbol_period_1 = self.atomic_symbol[:2]
        self.atomic_symbol_period_2 = self.atomic_symbol[2:10]
        self.atomic_symbol_period_3 = self.atomic_symbol[10:18]
        self.atomic_symbol_period_4 = self.atomic_symbol[18:36]
        
        # 初始化各周期基组大小
        self.basis_set_size_period_1 = atom_block[0].shape[0] if atom_block else 0
        self.basis_set_size_period_2 = self.basis_set_size_period_1
        self.basis_set_size_period_3 = self.basis_set_size_period_1
        self.basis_set_size_period_4 = self.basis_set_size_period_1
        
        # 根据原子类型更新各周期基组大小
        for atom_blk, atom_type in zip(self.atom_block, self.atom_type_list):
            if atom_type in self.atomic_symbol_period_1:
                self.basis_set_size_period_1 = atom_blk.shape[0]
            elif atom_type in self.atomic_symbol_period_2:
                self.basis_set_size_period_2 = atom_blk.shape[0]
            elif atom_type in self.atomic_symbol_period_3:
                self.basis_set_size_period_3 = atom_blk.shape[0]
            elif atom_type in self.atomic_symbol_period_4:
                self.basis_set_size_period_4 = atom_blk.shape[0]
            else:
                raise ValueError(f"Unknown atom type: {atom_type}")
        
        # 初始化对角哈密顿量子模块（按周期）
        self.diag_blocks = {
            1: Hamiltonian_diagonal_block(self.basis_set_size_period_1, nodes_feature.shape[1]),
            2: Hamiltonian_diagonal_block(self.basis_set_size_period_2, nodes_feature.shape[1]),
            3: Hamiltonian_diagonal_block(self.basis_set_size_period_3, nodes_feature.shape[1]),
            4: Hamiltonian_diagonal_block(self.basis_set_size_period_4, nodes_feature.shape[1])
        }
        
        # 初始化非对角哈密顿量子模块（按周期对）
        self.off_diag_blocks = {
            (1,1): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_1, self.basis_set_size_period_1, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (2,2): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_2, self.basis_set_size_period_2, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (3,3): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_3, self.basis_set_size_period_3, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (4,4): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_4, self.basis_set_size_period_4, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (1,2): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_1, self.basis_set_size_period_2, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (1,3): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_1, self.basis_set_size_period_3, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (1,4): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_1, self.basis_set_size_period_4, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (2,3): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_2, self.basis_set_size_period_3, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (2,4): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_2, self.basis_set_size_period_4, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            ),
            (3,4): Hamiltonian_off_diagonal_block(
                self.basis_set_size_period_3, self.basis_set_size_period_4, 
                nodes_feature.shape[1], nodes_feature.shape[1]
            )
        }
        
    def get_period(self, atom_type):
        """获取原子类型所属的周期"""
        if atom_type in self.atomic_symbol_period_1:
            return 1
        elif atom_type in self.atomic_symbol_period_2:
            return 2
        elif atom_type in self.atomic_symbol_period_3:
            return 3
        elif atom_type in self.atomic_symbol_period_4:
            return 4
        else:
            raise ValueError(f"Unknown atom type: {atom_type}")

    def forward(self):
        # 初始化哈密顿量矩阵
        H_total = torch.zeros(self.total_orbitals, self.total_orbitals, dtype=torch.float32, device=self.nodes_feature.device)
        
        # 计算轨道偏移量
        orbital_offsets = [0]
        for block in self.atom_block:
            orbital_offsets.append(orbital_offsets[-1] + block.shape[0])

        # 处理对角块
        for atom_idx, (atom_blk, atom_type) in enumerate(zip(self.atom_block, self.atom_type_list)):
            node_feat = self.nodes_feature[atom_idx]
            edge_feat = self.edges_feature[atom_idx, atom_idx]
            atom_blk = torch.tensor(atom_blk, dtype=torch.float32, device=self.nodes_feature.device)
            
            # 获取周期并选择对应的对角块模块
            period = self.get_period(atom_type)
            H_diag_atom = self.diag_blocks[period](node_feat, edge_feat, atom_blk)

            # 放置对角块到总哈密顿量
            start_idx = orbital_offsets[atom_idx]
            end_idx = orbital_offsets[atom_idx + 1]
            H_total[start_idx:end_idx, start_idx:end_idx] = H_diag_atom
        
        # 处理非对角块
        for atom_pair_type, off_diag_blk, (atom_index_i, atom_index_j) in zip(
                self.atom_pairs, self.atom_in_block, self.atom_pairs_index):
            
            # 跳过对角元素（已处理）
            if atom_index_i == atom_index_j:
                continue
                
            # 获取节点和边特征
            node_feat_i = self.nodes_feature[atom_index_i]
            node_feat_j = self.nodes_feature[atom_index_j]
            edge_feat_ij = self.edges_feature[atom_index_i, atom_index_j]
            off_diag_blk = torch.tensor(off_diag_blk, dtype=torch.float32, device=self.nodes_feature.device)
            
            # 确定原子对所属周期
            atom_type_i, atom_type_j = atom_pair_type
            period_i = self.get_period(atom_type_i)
            period_j = self.get_period(atom_type_j)
            
            # 获取对应的非对角块模块
            if (period_i, period_j) in self.off_diag_blocks:
                H_off_diag = self.off_diag_blocks[(period_i, period_j)](
                    node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk
                )
            elif (period_j, period_i) in self.off_diag_blocks:
                # 转置处理
                off_diag_blk_t = off_diag_blk.T
                H_off_diag = self.off_diag_blocks[(period_j, period_i)](
                    node_feat_j, node_feat_i, edge_feat_ij, off_diag_blk_t
                ).T
            else:
                raise ValueError(f"Unknown period pair: ({period_i}, {period_j})")
            
            # 放置非对角块到总哈密顿量（对称矩阵）
            start_idx_i = orbital_offsets[atom_index_i]
            end_idx_i = orbital_offsets[atom_index_i + 1]
            start_idx_j = orbital_offsets[atom_index_j]
            end_idx_j = orbital_offsets[atom_index_j + 1]
            
            H_total[start_idx_i:end_idx_i, start_idx_j:end_idx_j] = H_off_diag
            H_total[start_idx_j:end_idx_j, start_idx_i:end_idx_i] = H_off_diag.T
   
        return H_total


def main():
    # -------------------------- 配置参数 --------------------------
    data_root = "/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset"  # 数据根目录
    mol_dir = os.path.join(data_root, "mol")  # 分子结构文件目录
    ham_dir = os.path.join(data_root, "Hamiltonian")  # 哈密顿量文件目录
    epochs = 100  # 训练轮数
    lr = 1e-4  # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
    save_path = "best_model.pth"  # 最佳模型保存路径
    
    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path))

    # -------------------------- 数据准备 --------------------------
    # 获取所有分子文件列表并排序
    mol_files = sorted([f for f in os.listdir(mol_dir) if f.endswith('.mol2')])
    ham_files = sorted([f for f in os.listdir(ham_dir) if f.endswith('_h.npz')])
    
    # 确保分子文件和哈密顿量文件数量匹配
    assert len(mol_files) == len(ham_files), "分子文件和哈密顿量文件数量不匹配"
    
    # 划分训练集（前20个）和验证集（后3个）
    train_size = 10
    val_size = 3
    
    train_mol = [os.path.join(mol_dir, f) for f in mol_files[1:train_size+1]]
    train_ham = [os.path.join(ham_dir, f) for f in ham_files[:train_size]]
    val_mol = [os.path.join(mol_dir, f) for f in mol_files[train_size+1:train_size+val_size+1]]
    val_ham = [os.path.join(ham_dir, f) for f in ham_files[train_size:train_size+val_size]]
    
    print(f"训练集: {len(train_mol)}个分子")
    print(f"验证集: {len(val_mol)}个分子")

    # -------------------------- 模型初始化 --------------------------
    # 使用第一个分子初始化模型参数
    sample_mol = train_mol[0]
    
    # 初始化节点嵌入
    nodes_embed = nodes_embedding.Nodes_Embedding(sample_mol)
    nodes_feat = nodes_embed.forward().to(device)
    
    # 初始化边嵌入
    edges_embed = edges_embedding.Edges_Embedding(sample_mol)
    edges1, edges2, degree_tensor = edges_embed.forward()
    edges1, edges2, degree_tensor = edges1.to(device), edges2.to(device), degree_tensor.to(device)
    
    # 初始化EGAT层
    egat_layer = EGAT_Layer.EGATlayer(
        nodes_feat.shape[1], 
        nodes_feat.shape[1], 
        1
    ).to(device)
    
    # 获取原子块和相互作用
    atom_block, atom_interactions, _ = overlap_block.overlap_block_dec(sample_mol).get_atom_block()
    
    # 初始化哈密顿量生成层
    ham_gen_layer = HamiltonianBlockGenLayer(
       sample_mol,
        nodes_feat,
        torch.zeros_like(edges_embed.forward()[2]).to(device),
        atom_block,
        atom_interactions
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(
        list(egat_layer.parameters()) + list(ham_gen_layer.parameters()),
        lr=lr
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # -------------------------- 训练函数 --------------------------
    def train_one_epoch():
        egat_layer.train()
        ham_gen_layer.train()
        total_loss = 0.0
        
        for mol_path, ham_path in zip(train_mol, train_ham):
            optimizer.zero_grad()
            
            try:
                # 1. 计算节点嵌入
                nodes_embed = nodes_embedding.Nodes_Embedding(mol_path)
                nodes_feat = nodes_embed.forward().to(device)
                
                # 2. 计算边嵌入
                edges_embed = edges_embedding.Edges_Embedding(mol_path)
                edges1, edges2, degree_tensor = edges_embed.forward()
                edges1, edges2, degree_tensor = edges1.to(device), edges2.to(device), degree_tensor.to(device)
                
                # 3. 计算键方向
                edge_dir = torch.tensor(
                    bond_embedding.Bond_Embedding(mol_path).get_atom_pairs_direction(),
                    dtype=torch.float32,
                    device=device
                )
                
                # 4. 通过EGAT层更新节点特征
                out_nodes_feat, connectivity_mask = egat_layer.forward(
                    nodes_feat,
                    edges1,
                    edges2,
                    edge_dir,
                    degree_tensor,
                    cut_off=5.0
                )
                
                # 5. 获取原子块和相互作用
                atom_block, atom_interactions, _ = overlap_block.overlap_block_dec(mol_path).get_atom_block()
                
                # 6. 更新哈密顿量生成层的输入
                ham_gen_layer.mol2 = mol_path
                ham_gen_layer.nodes_feature = out_nodes_feat
                ham_gen_layer.edges_feature = connectivity_mask
                ham_gen_layer.atom_block = atom_block
                ham_gen_layer.atom_in_block = atom_interactions
                ham_gen_layer.atom_embedding = atom_embedding.Atom_Embedding(mol_path)
                ham_gen_layer.atom_type_list = ham_gen_layer.atom_embedding.get_atom_type()
                ham_gen_layer.atom_pairs, ham_gen_layer.atom_pairs_index = overlap_block.overlap_block_dec(mol_path).get_atom_pairs()
                ham_gen_layer.total_orbitals = sum(block.shape[0] for block in atom_block)
                
                # 7. 生成预测哈密顿量
                pred_ham = ham_gen_layer.forward()
                
                # 8. 加载真实哈密顿量
                true_ham = sp.load_npz(ham_path)
                true_ham = torch.tensor(true_ham.toarray(), dtype=torch.float32, device=device)
                
                # 9. 确保形状匹配
                if pred_ham.shape != true_ham.shape:
                    print(f"警告: 预测形状 {pred_ham.shape} 与真实形状 {true_ham.shape} 不匹配，跳过此样本")
                    continue
                
                # 10. 计算损失
                loss = F.mse_loss(pred_ham, true_ham)
                total_loss += loss.item()
                
                # 11. 反向传播和参数更新
                loss.backward()
                optimizer.step()
                
            except Exception as e:
                print(f"处理分子 {mol_path} 时出错: {str(e)}")
                continue
        
        return total_loss / len(train_mol)
    
    # -------------------------- 验证函数 --------------------------
    def validate():
        egat_layer.eval()
        ham_gen_layer.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for mol_path, ham_path in zip(val_mol, val_ham):
                try:
                    # 1. 计算节点嵌入
                    nodes_embed = nodes_embedding.Nodes_Embedding(mol_path)
                    nodes_feat = nodes_embed.forward().to(device)
                    
                    # 2. 计算边嵌入
                    edges_embed = edges_embedding.Edges_Embedding(mol_path)
                    edges1, edges2, degree_tensor = edges_embed.forward()
                    edges1, edges2, degree_tensor = edges1.to(device), edges2.to(device), degree_tensor.to(device)
                    
                    # 3. 计算键方向
                    edge_dir = torch.tensor(
                        bond_embedding.Bond_Embedding(mol_path).get_atom_pairs_direction(),
                        dtype=torch.float32,
                        device=device
                    )
                    
                    # 4. 通过EGAT层更新节点特征
                    out_nodes_feat, connectivity_mask = egat_layer.forward(
                        nodes_feat,
                        edges1,
                        edges2,
                        edge_dir,
                        degree_tensor,
                        cut_off=5.0
                    )
                    
                    # 5. 获取原子块和相互作用
                    atom_block, atom_interactions, _ = overlap_block.overlap_block_dec(mol_path).get_atom_block()
                    
                    # 6. 更新哈密顿量生成层的输入
                    ham_gen_layer.mol2 = mol_path
                    ham_gen_layer.nodes_feature = out_nodes_feat
                    ham_gen_layer.edges_feature = connectivity_mask
                    ham_gen_layer.atom_block = atom_block
                    ham_gen_layer.atom_in_block = atom_interactions
                    ham_gen_layer.atom_embedding = atom_embedding.Atom_Embedding(mol_path)
                    ham_gen_layer.atom_type_list = ham_gen_layer.atom_embedding.get_atom_type()
                    ham_gen_layer.atom_pairs, ham_gen_layer.atom_pairs_index = overlap_block.overlap_block_dec(mol_path).get_atom_pairs()
                    ham_gen_layer.total_orbitals = sum(block.shape[0] for block in atom_block)
                    
                    # 7. 生成预测哈密顿量
                    pred_ham = ham_gen_layer.forward()
                    
                    # 8. 加载真实哈密顿量
                    true_ham = sp.load_npz(ham_path)
                    true_ham = torch.tensor(true_ham.toarray(), dtype=torch.float32, device=device)
                    
                    # 9. 确保形状匹配
                    if pred_ham.shape != true_ham.shape:
                        print(f"警告: 预测形状 {pred_ham.shape} 与真实形状 {true_ham.shape} 不匹配，跳过此样本")
                        continue
                    
                    # 10. 计算损失
                    loss = F.mse_loss(pred_ham, true_ham)
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"验证分子 {mol_path} 时出错: {str(e)}")
                    continue
        
        return total_loss / len(val_mol)
    
    # -------------------------- 训练循环 --------------------------
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练
        train_loss = train_one_epoch()
        
        # 验证
        val_loss = validate()
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'egat_state_dict': egat_layer.state_dict(),
                'ham_gen_state_dict': ham_gen_layer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"保存最佳模型到 {save_path}")
        
        # 打印训练信息
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 时间: {epoch_time:.2f}s")
    
    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
