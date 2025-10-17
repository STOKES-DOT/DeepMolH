import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn


class Hamiltonian_diagonal_block(nn.Module):
    def __init__(self, atom_block_size=None, nodes_feature_size=None):
        super().__init__()
        self.nodes_size = nodes_feature_size
        self.orbital_size = atom_block_size
        self.num_off_diag_elements = self.orbital_size * (self.orbital_size - 1) // 2
        self.nodes_feature_net = nn.Sequential(
            nn.Linear(self.nodes_size, self.orbital_size**2),
            nn.ELU(),
            nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),
            nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),
            nn.Linear(self.orbital_size**2, self.orbital_size),
        )
        
        self.H_diag_net_diag = nn.Sequential(
            nn.Linear(self.orbital_size, self.orbital_size**2),
            nn.ELU(),
            nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),
              nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),
              nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),
              nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),   
              nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),   
              nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),   
            nn.Linear(self.orbital_size**2, self.orbital_size),
        )
        
        self.H_diag_net_off_diag = nn.Sequential(
            nn.Linear(self.orbital_size, self.orbital_size**2),
            nn.ELU(),
            nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),
            nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),
            nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),   
            nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),   
            nn.Linear(self.orbital_size**2, self.orbital_size**2),   
            nn.ELU(),
            nn.Linear(self.orbital_size**2, self.orbital_size**2),
            nn.ELU(),   
            nn.Linear(self.orbital_size**2, self.num_off_diag_elements)
        )
        
        self.coupling_strength = nn.Parameter(torch.tensor(0.1))
        self.off_diag_coupling_strength = nn.Parameter(torch.tensor(0.1))
        self.edges_contribution = nn.Parameter(torch.tensor(0.1))
        self._initialize_weights()
        self.orbital_coupling_strength = nn.Parameter(torch.tensor(0.1))
        self.off_diag_orbital_coupling_strength = nn.Parameter(torch.tensor(0.1))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, nodes_feature, edges_feature, atom_block):
    
        
        nodes_feature = self.nodes_feature_net(
            nodes_feature + self.edges_contribution * edges_feature
        )
  
        temp_diag_block = self.H_diag_net_diag(nodes_feature)
        diag_block = self.coupling_strength * temp_diag_block
        diag_block = torch.eye(self.orbital_size, device=diag_block.device, dtype=torch.float32) * diag_block
        diag_block = diag_block.unsqueeze(0)
        
   
        base_H = self.orbital_coupling_strength * atom_block.unsqueeze(0)
        H_diag = diag_block + base_H
        

        off_diag_elements = self.H_diag_net_off_diag(nodes_feature)
        off_diag_elements = self.off_diag_coupling_strength * off_diag_elements
        
      
        off_diag_matrix = torch.zeros(self.orbital_size, self.orbital_size, 
                                    device=off_diag_elements.device, dtype=torch.float32)
        
        triu_indices = torch.triu_indices(self.orbital_size, self.orbital_size, offset=1)
        
        if off_diag_elements.shape[-1] != self.num_off_diag_elements:
            off_diag_elements = off_diag_elements.view(-1, self.num_off_diag_elements)
        
        off_diag_matrix[triu_indices[0], triu_indices[1]] = off_diag_elements.squeeze()
        
    
        off_diag_matrix = off_diag_matrix.unsqueeze(0)
        H_diag = H_diag + off_diag_matrix
        
        H_diag = (H_diag + H_diag.transpose(1, 2)) * 0.5
        return H_diag
    
class Hamiltonian_off_diagonal_block(nn.Module):
    def __init__(self, atom_in_block_size_rows=None, atom_in_block_size_cols=None, nodes_feature_i_size=None, nodes_feature_j_size=None):
        super().__init__()
        self.nodes_feature_i_size = nodes_feature_i_size
        self.nodes_feature_j_size = nodes_feature_j_size
        self.nodes_size = nodes_feature_i_size
        self.rows, self.cols = atom_in_block_size_rows, atom_in_block_size_cols
        self.orbital_size_cols = self.cols
        self.orbital_size_rows = self.rows
        
        self.nodes_feature_net_i = nn.Sequential(
            nn.Linear(self.nodes_size, self.orbital_size_cols),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols, self.orbital_size_cols),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols, self.orbital_size_cols),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols, self.orbital_size_cols),
           
        )
        self.nodes_feature_net_j = nn.Sequential(
            nn.Linear(self.nodes_size, self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_rows, self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_rows, self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_rows, self.orbital_size_rows),
            
        )
        
        self.H_off_diag_net = nn.Sequential(
            nn.Linear(self.orbital_size_cols+self.orbital_size_rows, self.orbital_size_cols+self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols+self.orbital_size_rows, self.orbital_size_cols+self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols+self.orbital_size_rows, self.orbital_size_cols+self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols+self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            nn.ELU(),
            nn.Linear(self.orbital_size_cols*self.orbital_size_rows, self.orbital_size_cols*self.orbital_size_rows),
            
        )
    
        self.off_diag_coupling_strength = nn.Parameter(torch.tensor(0.1))
        self.nodes_contribution_i = nn.Parameter(torch.tensor(0.1))
        self.nodes_contribution_j = nn.Parameter(torch.tensor(0.1))
        self.edges_contribution_nodes_i = nn.Parameter(torch.tensor(0.1))
        self.edges_contribution_nodes_j = nn.Parameter(torch.tensor(0.1))
        self._initialize_weights()
        self.orbital_coupling_strength = nn.Parameter(torch.tensor(0.1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -0.1)
    def forward(self, nodes_feature_i, nodes_feature_j, edges_feature, atom_in_block):
        
        nodes_feature_i = nodes_feature_i.clone().detach().requires_grad_(True).reshape(1, -1)
        nodes_feature_j = nodes_feature_j.clone().detach().requires_grad_(True).reshape(1, -1)
        
        edges_feature = edges_feature.clone().detach().requires_grad_(True).reshape(1, -1)
        
        nodes_feature_i = self.nodes_feature_net_i(nodes_feature_i+self.edges_contribution_nodes_i*edges_feature)
        nodes_feature_j = self.nodes_feature_net_j(nodes_feature_j+self.edges_contribution_nodes_j*edges_feature)
        
        node_edge_contribution = torch.cat((nodes_feature_i, nodes_feature_j), dim=1)
        
        H_off_diag = self.H_off_diag_net(node_edge_contribution).view(self.orbital_size_rows, self.orbital_size_cols)
        H_off_diag = H_off_diag +self.off_diag_coupling_strength*atom_in_block
        
        return H_off_diag

#NOTE: We need a spherical harmonics basis to represent the E(3)-equivariant MLP, So that the structure of network can be restricted to the shell structure of the atoms and atom-pairs !!!