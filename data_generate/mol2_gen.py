import pandas as pd
from openbabel import pybel
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import AllChem
smiles = pd.read_csv('data_generate\smiles.csv')
smiles = smiles['smiles']

def smi_to_mol(smi,index,files):
    #rdkit block for 3D conformation
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())#ETKDGv3 module to search for the best conformation
    AllChem.MMFFOptimizeMolecule(mol)#optimize the conformation using MMFF94 force field
    mol_block = Chem.MolToMolBlock(mol)
    Chem.MolToMolFile(mol, f'{files}\{index}.mol')
    #openbabel block for mol2 conversion
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol", "mol2")
    ob_mol = openbabel.OBMol()
    obConversion.ReadString(ob_mol, mol_block)
    obConversion.WriteFile(ob_mol, f'{files}\{index}.mol2')
    print(f'{index}.mol2 saved')
    
i=1
for smi in enumerate(smiles):
    print(smi[1])
    smi_to_mol(smi[1],i,'dataset\mol')
    i+=1
