from module_nn.Atom_embedding import Atom_Embedding
from module_nn.bond_embedding import Bond_Embedding
from module_nn.Hamiltonian_gen import Hamiltonian_gen
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
class Data_gen():
    def __init__(self,mol2,files,index):
        self.mol2 = mol2
        self.files = files
        self.index = index
        self.atom_embed = Atom_Embedding(self.mol2)
        self.mol_embed = Bond_Embedding(self.mol2)
        self.hamiltonian_gen = Hamiltonian_gen(self.mol2,self.files,self.index)
    def data_save(self):
        
