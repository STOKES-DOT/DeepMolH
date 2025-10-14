from pyscf import gto, scf, dft
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, save_npz

class Hamiltonian_gen():
    def __init__(self,mol2,files,index):
        self.mol2=mol2
        self.xyz = self._mol2_to_xyz()
        self.overlap = self.get_overlap()
        self.hamiltonian = self.get_hamiltonian()
        self.files = files
        self.index = index
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
    def get_overlap(self,basis='sto-3g'):
        mol = gto.M(atom=self.xyz,basis=basis)
        S_ao = mol.intor('int1e_ovlp')
        return S_ao
    def get_hamiltonian(self,basis='sto-3g',xc='lda'):
        mol = gto.M(atom=self.xyz,basis=basis)
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.kernel()
        fock = mf.get_fock()
        H_ao = fock
        return H_ao
    def save_matrix(self):
        Hamiltonian_matrix = csr_matrix(self.hamiltonian)
        save_npz(f'{self.files}/{self.index}_h.npz',Hamiltonian_matrix)
        Overlap_matrix =csr_matrix(self.overlap)
        save_npz(f'{self.files}/{self.index}_o.npz',Overlap_matrix)
        
if __name__ == '__main__':
    for i in range(60):
        mol2 = f'/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/mol/{i+1}.mol2'
        hamiltonian_gen = Hamiltonian_gen(mol2,'/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/Hamiltonian',i+1)
        print(f'index {i+1}th molecule')
        hamiltonian_gen.get_hamiltonian(basis='sto-3g')#the test data with B3LYP/STO-3G
        hamiltonian_gen.get_overlap(basis='sto-3g')
        hamiltonian_gen.save_matrix()

