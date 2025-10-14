import numpy as np
import torch
import torch.nn as nn
import atom_embedding
class Nodes_Embedding(nn.Module):#node feature embedding with MLP
    def __init__(self,mol2):
        super().__init__()
        self.mol2 = mol2
        self.atom_embed = atom_embedding.Atom_Embedding(self.mol2)
        atom_type_vector = self.atom_embed.atom_type_to_vector()
        atom_charge = self.atom_embed.atom_charge()
        self.register_buffer('atom_type_vector', torch.tensor(atom_type_vector, dtype=torch.float32))
        self.register_buffer('atom_charge', torch.tensor(atom_charge, dtype=torch.float32))
        self.num_atoms = len(self.atom_embed.get_atom_type())
        self.atom_part = self.atom_embed.atom_molecular_part()
        self.flatten = nn.Flatten()
        self.nodes_embedding = nn.Sequential(
            nn.Linear(16,16),
            nn.ELU(),
            nn.Linear(16,16),
            nn.ELU(),
            nn.Linear(16,16),
            nn.ELU(),
        )
        
    def forward(self):
        atom_part_embed = self.atom_embed.atom_molecular_part()
        nodes1=[]
        for atom_v,atom_part in zip(self.atom_type_vector,self.atom_part):
            atom_v = atom_v.clone().detach()
            atom_part_tensor = torch.full((16,), atom_part, dtype=torch.float32)
            atom_charge_tensor = torch.full((16,), self.atom_charge[atom_part], dtype=torch.float32)
            nodes1.append(self.nodes_embedding(atom_v) + atom_part_tensor + atom_charge_tensor)
        return torch.stack(nodes1)
    
if __name__ == '__main__':
    mol2 = '/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/mol/2000.mol2'
    nodes_embed = Nodes_Embedding(mol2)
    print(nodes_embed.forward())