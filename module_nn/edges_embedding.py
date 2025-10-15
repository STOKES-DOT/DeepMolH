import numpy as np
import torch
import torch.nn as nn
import bond_embedding

class Edges_Embedding(nn.Module):#edge feature embedding with MLP
    def __init__(self):
        super().__init__()
        self.mol2 = f'/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/module_nn/net.mol2'
        self.bond_embed = bond_embedding.Bond_Embedding(self.mol2)
        self.gb_matrix = self.bond_embed.gaussian_basis_matrix()
   
        if len(self.gb_matrix.shape) != 2:
            raise ValueError(f"Expected 2D gb_matrix, got shape {self.gb_matrix.shape}")
        
        self.edges_size = self.gb_matrix.shape[1]
        self.hidden_dim = 10
        
        self.features_dim = self.gb_matrix.shape[1]
        self.flatten = nn.Flatten()
        self.edges_embedding_dist = nn.Sequential(
            nn.Linear(self.features_dim,self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim,self.features_dim),
            nn.LeakyReLU(),
        )
        self.edges_embedding_type = nn.Sequential(
            nn.Linear(self.features_dim,self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim,self.features_dim),
            nn.LeakyReLU(),
        )

    def forward(self,mol2):
        bond_embed = bond_embedding.Bond_Embedding(mol2)
        gb_matrix = bond_embed.gaussian_basis_matrix()
        
        edge_info1 = self.edges_embedding_dist(torch.tensor(gb_matrix, dtype=torch.float32))
        bond_type_matrix = bond_embed.get_bond_type()
        edge_info2 = self.edges_embedding_type(torch.tensor(bond_type_matrix, dtype=torch.float32))
        
        edge_info1 = (edge_info1 + edge_info1.transpose(0, 1)) / 2
        edge_info2 = (edge_info2 + edge_info2.transpose(0, 1)) / 2
        
        edges_direction = bond_embed.forward()[2]
        num_atoms = bond_type_matrix.shape[0]
        degree_matrix = np.zeros((num_atoms,num_atoms))
        for i in range(num_atoms):
            for j in range(num_atoms):
                if bond_type_matrix[i,j] != 0:
                    degree_matrix[i,j] = 1
        return edge_info1, edge_info2, degree_matrix, edges_direction


if __name__ == '__main__':
    edges_embedding = Edges_Embedding()
    mol2 = f'/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/module_nn/net.mol2'
    edge_info1, edge_info2, degree_matrix, edges_direction = edges_embedding.forward(mol2)
    print(edge_info1.shape)
    print(edge_info2.shape)
    print(degree_matrix.shape)
    print(torch.tensor(edges_direction).shape)
