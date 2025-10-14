import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import matplotlib.pyplot as plt
from NeuralNetwork import DeepMolH

mol2 = '/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/dataset/mol/10.mol2'
target_Hamiltonian = sp.load_npz('/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/dataset/Hamiltonian/10_h.npz').toarray()


model = DeepMolH(mol2)
Hamiltonian_block = model.forward()


