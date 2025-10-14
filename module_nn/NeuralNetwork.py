from Hamiltonian_block import Hamiltonian_diagonal_block
from Hamiltonian_block import Hamiltonian_off_diagonal_block
import overlap_block
import bond_embedding
import atom_embedding
import nodes_embedding
import edges_embedding
import EGAT_Layer
import GAT_Layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class HamiltonianBlockGenLayer(nn.Module):
    def __init__(self, mol2=f'/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/module_nn/net.mol2'):
        super().__init__()
        self.mol2 = mol2
        self.nodes_feature_size = 16
        self.atom_block, self.atom_in_block, self.off_diag_num = overlap_block.overlap_block_dec(self.mol2).get_atom_block()
        self.total_orbitals = sum(block.shape[0] for block in self.atom_block)
        #single atom block classified by atom type
        self.atom_embedding = atom_embedding.Atom_Embedding(self.mol2)
        self.atom_type_list = self.atom_embedding.get_atom_type()
        self.atom_pairs, self.atom_pairs_index = overlap_block.overlap_block_dec(self.mol2).get_atom_pairs()
        self.atomic_symbol = [
                                # Period 1
                                'H', 'He',
                                # Period 2
                                'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                                # Period 3
                                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                                # Period 4
                                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'
                            ]


        self.atomic_symbol_period_1 = self.atomic_symbol[:2]
        self.atomic_symbol_period_2 = self.atomic_symbol[2:10]
        self.atomic_symbol_period_3 = self.atomic_symbol[10:18]
        self.atomic_symbol_period_4 = self.atomic_symbol[18:36]
        self.basis_set_size_period_1 = self.atom_block[0].shape[0]
        self.basis_set_size_period_2 = self.atom_block[0].shape[0]
        self.basis_set_size_period_3 = self.atom_block[0].shape[0]
        self.basis_set_size_period_4 = self.atom_block[0].shape[0]
        
        for atom_block, atom_type in zip(self.atom_block, self.atom_type_list):
            if atom_type in self.atomic_symbol_period_1:
                self.basis_set_size_period_1 = atom_block.shape[0]
            elif atom_type in self.atomic_symbol_period_2:
                self.basis_set_size_period_2 = atom_block.shape[0]
            elif atom_type in self.atomic_symbol_period_3:
                self.basis_set_size_period_3 = atom_block.shape[0]
            elif atom_type in self.atomic_symbol_period_4:
                self.basis_set_size_period_4 = atom_block.shape[0]
            else:
                raise ValueError(f"Unknown atom type: {atom_type}")

        periods = [1,2,3,4]
        basis_set_size_period = [self.basis_set_size_period_1, self.basis_set_size_period_2, self.basis_set_size_period_3, self.basis_set_size_period_4]
        nodes_feature_size = self.nodes_feature_size
        
        self.diagonal_block_nn = nn.ModuleList([Hamiltonian_diagonal_block(atom_block_size=bs, nodes_feature_size=nodes_feature_size)
    for bs in basis_set_size_period])
  
        self.same_period_off_diag_blocks = nn.ModuleList([
            Hamiltonian_off_diagonal_block(
            atom_in_block_size_rows=bs,
            atom_in_block_size_cols=bs,
            nodes_feature_i_size=nodes_feature_size,
            nodes_feature_j_size=nodes_feature_size
        )
        for bs in basis_set_size_period])

        cross_period_pairs = [(0,1), (1,2), (2,3), (0,2), (1,3), (0,3)]  # 使用索引代替周期数
        self.cross_period_off_diag_blocks = nn.ModuleList([
            Hamiltonian_off_diagonal_block(
            atom_in_block_size_rows=basis_set_size_period[i],
            atom_in_block_size_cols=basis_set_size_period[j],
            nodes_feature_i_size=nodes_feature_size,
            nodes_feature_j_size=nodes_feature_size
        )
        for i, j in cross_period_pairs])
        
    def forward(self, nodes_features, connectivity_mask):
        self.edges_feature = connectivity_mask
        self.nodes_feature = nodes_features
        H_total = torch.randn(self.total_orbitals, self.total_orbitals, 
                            dtype=torch.float32)
        orbital_offsets = [0]
        for block in self.atom_block:
            orbital_offsets.append(orbital_offsets[-1] + block.shape[0])

        for atom_idx, (atom_blk, atom_type) in enumerate(zip(self.atom_block, self.atom_type_list)):
            #print(f'atom{atom_idx} type: {atom_type}, block shape: {atom_blk.shape}')
            node_feat = self.nodes_feature[atom_idx].clone().detach().requires_grad_(True) #node feature of atom_idx
            edge_feat = self.edges_feature[atom_idx, atom_idx].clone().detach().requires_grad_(True) #edge feature of atom_idx
            atom_blk = torch.tensor(atom_blk, dtype=torch.float32)
            
            if atom_type in self.atomic_symbol_period_1:
                H_diag_atom = self.diagonal_block_nn[0].forward(node_feat, edge_feat, atom_blk)
            elif atom_type in self.atomic_symbol_period_2:
                H_diag_atom = self.diagonal_block_nn[1].forward(node_feat, edge_feat, atom_blk)
                
            elif atom_type in self.atomic_symbol_period_3:
                H_diag_atom = self.diagonal_block_nn[2].forward(node_feat, edge_feat, atom_blk)
            elif atom_type in self.atomic_symbol_period_4:
                H_diag_atom = self.diagonal_block_nn[3].forward(node_feat, edge_feat, atom_blk)
            else:
                raise ValueError(f"Unknown atom type: {atom_type}")

            start_idx = orbital_offsets[atom_idx]
            end_idx = orbital_offsets[atom_idx + 1]
            H_total[start_idx:end_idx, start_idx:end_idx] = H_diag_atom
        
        for atom_pair_type, off_diag_blk, (atom_index_i, atom_index_j) in zip(self.atom_pairs, self.atom_in_block, self.atom_pairs_index):
            node_feat_i = self.nodes_feature[atom_index_i].clone().detach().requires_grad_(True) #node feature of atom_index_i
            node_feat_j = self.nodes_feature[atom_index_j].clone().detach().requires_grad_(True) #node feature of atom_index_j
            edge_feat_ij = self.edges_feature[atom_index_i, atom_index_j].clone().detach().requires_grad_(True) #edge feature of atom_index_i and atom_index_j
            off_diag_blk = torch.tensor(off_diag_blk, dtype=torch.float32)
            atom_type_i = atom_pair_type[0]
            atom_type_j = atom_pair_type[1]
            if atom_type_i in self.atomic_symbol_period_1 and atom_type_j in self.atomic_symbol_period_2:#period 1-2
                    H_off_diag = self.cross_period_off_diag_blocks[0].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_2 and atom_type_j in self.atomic_symbol_period_3:#period 2-3
                H_off_diag = self.cross_period_off_diag_blocks[1].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_3 and atom_type_j in self.atomic_symbol_period_4:#period 3-4
                H_off_diag = self.cross_period_off_diag_blocks[2].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_1 and atom_type_j in self.atomic_symbol_period_3:#period 1-3
                H_off_diag = self.cross_period_off_diag_blocks[3].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_2 and atom_type_j in self.atomic_symbol_period_4:#period 2-4
                H_off_diag = self.cross_period_off_diag_blocks[4].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_1 and atom_type_j in self.atomic_symbol_period_4:#period 1-4
                H_off_diag = self.cross_period_off_diag_blocks[5].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_2 and atom_type_j in self.atomic_symbol_period_1:#period 2-1
                off_diag_blk = off_diag_blk.T
                H_off_diag = self.cross_period_off_diag_blocks[0].forward(node_feat_j, node_feat_i, edge_feat_ij, off_diag_blk).T
            elif atom_type_i in self.atomic_symbol_period_3 and atom_type_j in self.atomic_symbol_period_2:#period 3-2
                off_diag_blk = off_diag_blk.T
                H_off_diag = self.cross_period_off_diag_blocks[1].forward(node_feat_j, node_feat_i, edge_feat_ij, off_diag_blk).T
            elif atom_type_i in self.atomic_symbol_period_4 and atom_type_j in self.atomic_symbol_period_3:#period 4-3
                off_diag_blk = off_diag_blk.T
                H_off_diag = self.cross_period_off_diag_blocks[2].forward(node_feat_j, node_feat_i, edge_feat_ij, off_diag_blk).T
            elif atom_type_i in self.atomic_symbol_period_3 and atom_type_j in self.atomic_symbol_period_1:#period 3-1
                off_diag_blk = off_diag_blk.T
                H_off_diag = self.cross_period_off_diag_blocks[3].forward(node_feat_j, node_feat_i, edge_feat_ij, off_diag_blk).T
            elif atom_type_i in self.atomic_symbol_period_4 and atom_type_j in self.atomic_symbol_period_2:#period 4-2
                off_diag_blk = off_diag_blk.T
                H_off_diag = self.cross_period_off_diag_blocks[4].forward(node_feat_j, node_feat_i, edge_feat_ij, off_diag_blk).T
            elif atom_type_i in self.atomic_symbol_period_4 and atom_type_j in self.atomic_symbol_period_1:#period 4-1
                off_diag_blk = off_diag_blk.T
                H_off_diag = self.cross_period_off_diag_blocks[5].forward(node_feat_j, node_feat_i, edge_feat_ij, off_diag_blk).T
                
            elif atom_type_i in self.atomic_symbol_period_1 and atom_type_j in self.atomic_symbol_period_1:#period 1-1
                H_off_diag = self.same_period_off_diag_blocks[0].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_2 and atom_type_j in self.atomic_symbol_period_2:#period 2-2
                H_off_diag = self.same_period_off_diag_blocks[1].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_3 and atom_type_j in self.atomic_symbol_period_3:#period 3-3
                H_off_diag = self.same_period_off_diag_blocks[2].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)
            elif atom_type_i in self.atomic_symbol_period_4 and atom_type_j in self.atomic_symbol_period_4:#period 4-4
                H_off_diag = self.same_period_off_diag_blocks[3].forward(node_feat_i, node_feat_j, edge_feat_ij, off_diag_blk)            
            else:
                raise ValueError(f"Unknown atom pair type: {atom_pair_type}")
            
            
            start_idx_i = orbital_offsets[atom_index_i]
            end_idx_i = orbital_offsets[atom_index_i + 1]
            start_idx_j = orbital_offsets[atom_index_j]
            end_idx_j = orbital_offsets[atom_index_j + 1]
            H_total[start_idx_i:end_idx_i, start_idx_j:end_idx_j] = H_off_diag
            H_total[start_idx_j:end_idx_j, start_idx_i:end_idx_i] = H_off_diag.T
   
        return H_total
class GraphAttentionLayer(nn.Module):
    def __init__(self, node_in_dim=128, num_out_dim=128, num_heads=8, num_gat_layers=2,num_egat_layers=2, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.num_egat_layers = num_egat_layers
        self.num_gat_layers = num_gat_layers
        self.num_out_dim = num_out_dim
        self.dropout = dropout
        
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            self.gnn_layers.append(
                GAT_Layer.GATlayer(
                    num_in_features=node_in_dim,
                    num_out_features=num_out_dim,
                    num_heads=num_heads,
                    concat=False,
                    activation=nn.ELU() if i < num_gat_layers - 1 else None,
                    add_skip_connection=True
                )
            )
        for i in range(num_egat_layers):

            self.gnn_layers.append(
                EGAT_Layer.EGATlayer(
                    num_in_features=node_in_dim,
                    num_out_features=num_out_dim,
                    num_heads=num_heads,
                    concat=False,
                    activation=nn.ELU() if i < num_egat_layers - 1 else None,
                    add_skip_connection=True
                )
            )
    def forward(self, node_feat, edges1, edges2, edges_direction, degree_tensor, cut_off=5.0):
        for i,layer in enumerate(self.gnn_layers):
            if i < self.num_gat_layers:
                node_feat, connectivity_mask = layer.forward(node_feat, edges1, edges2, degree_tensor, cut_off)
            else:
                node_feat, connectivity_mask = layer.forward(node_feat, edges1, edges2, edges_direction, degree_tensor, cut_off)
            if i < self.num_egat_layers + self.num_gat_layers - 1:
                node_feat = F.dropout(node_feat, p=self.dropout, training=self.training)
        return node_feat, connectivity_mask
    
class MolEmbeddingLayer(nn.Module):
    def __init__(self, mol2):
        super(MolEmbeddingLayer, self).__init__()
        self.mol2 = mol2
        self.nodes = nodes_embedding.Nodes_Embedding(mol2)
        self.edges = edges_embedding.Edges_Embedding(mol2)
    def forward(self):
        node_feat = self.nodes.forward()
        edge_feat_dis, edge_feat_bond, degree_tensor, edges_direction = self.edges.forward()
        return node_feat, edge_feat_dis, edge_feat_bond, degree_tensor, edges_direction
    
class DeepMolH(nn.Module):
    def __init__(self, mol2=f'/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/module_nn/net.mol2', num_egat_layers=2, num_gat_layers=0, num_heads=8, dropout=0.6):
        super(DeepMolH, self).__init__()
        self.mol2 = mol2
        self.nodes = nodes_embedding.Nodes_Embedding(mol2)
        self.nodes_feat, self.edges_feat_dis,self.edges_feat_bond, self.degree_tensor, self.edges_direction = MolEmbeddingLayer(mol2).forward()
        self.gatlayer = GraphAttentionLayer(node_in_dim=self.nodes_feat.shape[1], num_out_dim=self.nodes_feat.shape[1], num_heads=num_heads, num_gat_layers=num_gat_layers, num_egat_layers=num_egat_layers, dropout=dropout)
        self.HamiltonianBlockGenLayer = HamiltonianBlockGenLayer()
        self.edges_direction = torch.tensor(self.edges_direction, dtype=torch.float32)
    def forward(self):
        out_nodes_features, connectivity_mask = self.gatlayer.forward(self.nodes_feat, self.edges_feat_dis, self.edges_feat_bond, self.edges_direction, self.degree_tensor)
        Hamiltonian_block = self.HamiltonianBlockGenLayer.forward(out_nodes_features, connectivity_mask)
    #NOTE: GATLayer seemly not work well, we use EGATLayer instead or correct the GATLayer !!!!
        return Hamiltonian_block
        
    
    
    
if __name__ == '__main__':
    import scipy.sparse as sp
    mol2 = '/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/dataset/mol/10.mol2'
    Hamiltonian_block_target = sp.load_npz('/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/dataset/Hamiltonian/1_h.npz')

    atom_block, atom_interactions, off_diag_num = overlap_block.overlap_block_dec(mol2).get_atom_block()
    nodes_embed = nodes_embedding.Nodes_Embedding(mol2)
    edges_embedding_embed = edges_embedding.Edges_Embedding(mol2)   
    
    nodes_features = nodes_embed.forward()
    degree_matrix = edges_embedding_embed.get_degree_matrix()
    degree_tensor = torch.from_numpy(degree_matrix)
    edges1, edges2, degree_tensor, edges_direction = edges_embedding_embed.forward()
    
    egat_layer = EGAT_Layer.EGATlayer(nodes_features.shape[1], nodes_features.shape[1], 1)
    egde_vec = bond_embedding.Bond_Embedding(mol2).get_atom_pairs_direction()
    edges_direction = torch.tensor(egde_vec, dtype=torch.float32)
    out_nodes_features, connectivity_mask = egat_layer.forward(nodes_features, edges1, edges2, edges_direction ,degree_tensor, cut_off=5.0)
    
    Hamiltonian_block = DeepMolH().forward()
    
    print(DeepMolH(mol2))


