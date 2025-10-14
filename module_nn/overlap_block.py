import torch
import torch.nn as nn
import torch.nn.functional as F
import atom_embedding
from pyscf import gto

class overlap_block_dec(nn.Module):
    def __init__(self, mol2, basis='sto-3g'):
        super().__init__()
        self.mol2 = mol2
        self.xyz = self._mol2_to_xyz()
        self.overlap, self.atom_slices = self.get_overlap()
        self.basis = basis

    def _mol2_to_xyz(self):
        xyz_lines = []
        with open(self.mol2, 'r') as f:
            lines = f.readlines()
        atom_section = False
        for line in lines:
            if line.startswith('@<TRIPOS>ATOM'):
                atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                atom_section = False
                continue
            if atom_section and line.strip():
                parts = line.split()
                if len(parts) >= 6: 
                    atom_symbol = parts[1][:2].title()
                    x, y, z = parts[2:5] 
                    xyz_lines.append(f"{atom_symbol} {x} {y} {z}")
        return xyz_lines
    
    def get_overlap(self):
        mol = gto.M(atom=self.xyz,basis='sto-3g')
        S_ao = mol.intor('int1e_ovlp')
        atom_slices = mol.aoslice_by_atom()
        return S_ao, atom_slices
    def get_atom_block(self):
        atom_blocks = []
        atom_interactions = []
        off_diag_num = []
        for i in range(len(self.atom_slices)):
            start_i, end_i = self.atom_slices[i, 2], self.atom_slices[i, 3]
            for j in range(i,len(self.atom_slices)):
                start_j, end_j = self.atom_slices[j, 2], self.atom_slices[j, 3]
                if i==j:
                    atom_blocks.append(self.overlap[start_i:end_i, start_j:end_j])
                else:
                    atom_interactions.append(self.overlap[start_i:end_i, start_j:end_j])
                    off_diag_num.append([i,j])
        return atom_blocks, atom_interactions, off_diag_num

    def get_atom_pairs(self):
        atom_pairs = []
        atom_pairs_num = []
        atom_types = atom_embedding.Atom_Embedding(self.mol2).get_atom_type()
        for i in range(len(self.atom_slices)):
            for j in range(i,len(self.atom_slices)):
                if i!=j:
                    atom_pairs.append([atom_types[i],atom_types[j]])
                    atom_pairs_num.append([i,j])
        return atom_pairs, atom_pairs_num
    
    def get_single_atom_type_block_size(self):
        atom_types = atom_embedding.Atom_Embedding(self.mol2).get_atom_type()
        atom_types_list = []
        atom_type_block_size = []
        for atom_type, i in zip(atom_types, range(len(self.atom_slices))):
            if atom_type not in atom_types_list:
                atom_types_list.append(atom_type)
                start_i, end_i = self.atom_slices[i, 2], self.atom_slices[i, 3]
                atom_type_block_size.append(end_i - start_i)
        atom_type_block_size_dict = {}
        for atom_type, block_size in zip(atom_types_list, atom_type_block_size):
            atom_type_block_size_dict[atom_type] = block_size
        return atom_type_block_size_dict
    
    def get_atoms_pairs_block_size(self):
        atom_pairs_types = self.get_atom_pairs()[0]
        atom_interactions = self.get_atom_block()[1]
        atom_interactions_size = []
        atom_pairs_types_list = []
        for atom_pair_type, interaction in zip(atom_pairs_types, atom_interactions):
            if atom_pair_type not in atom_pairs_types_list:
                atom_pairs_types_list.append(atom_pair_type)
                atom_interactions_size.append(interaction.shape)
    
        return atom_pairs_types_list, atom_interactions_size

