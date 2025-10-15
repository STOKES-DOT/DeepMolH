import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import matplotlib.pyplot as plt
from NeuralNetwork import DeepMolH
import os
class Tranning_module(nn.Module):
    def __init__(self, mol_dir, target_Hamiltonian):
        super().__init__()
        self.model = DeepMolH()
        self.mol_dir = mol_dir
        self.target_Hamiltonian = target_Hamiltonian
    def training_data_load(self):
        self.mol2 = os.path.join(self.mol_dir, self.mol2)
        self.target_Hamiltonian = torch.from_numpy(self.target_Hamiltonian).float()
        mol2 = os.path.join(self.mol_dir, self.mol2)

