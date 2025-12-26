# DeepMolH

A deep learning framework for predicting molecular Hamiltonians using Graph Neural Networks (GNNs) with equivariant message passing.

## Overview

DeepMolH is a PyTorch-based neural network architecture that predicts quantum mechanical Hamiltonian matrices for molecular systems. The model combines Graph Attention Networks (GAT) and Equivariant Graph Attention Networks (EGAT) to learn molecular representations and generate Hamiltonian matrices directly from molecular structures.

### Key Features

- **Graph Neural Network Architecture**: Combines GAT and EGAT layers for molecular representation learning
- **Equivariant Message Passing**: Leverages E(3)-equivariant operations to respect rotational symmetry
- **Hamiltonian Prediction**: Directly predicts quantum mechanical Hamiltonian matrices from molecular graphs
- **Quantum Chemistry Integration**: Uses PySCF for reference Hamiltonian calculations
- **Flexible Training**: Supports hyperparameter optimization with Optuna
- **Multiple Atomic Periods**: Handles elements from periods 1-4 of the periodic table

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install e3nn
pip install pyscf
pip install rdkit
pip install openbabel
pip install optuna
pip install scipy numpy pandas matplotlib seaborn
pip install tqdm joblib
```

### Setup

1. Clone the repository:

```bash
git clone https://github.com/STOKES-DOT/DeepMolH.git
cd DeepMolH
```

2. Install dependencies (see above)

3. Prepare your molecular dataset in MOL2 format or generate from SMILES

## Usage

### 1. Generate Molecular Data from SMILES

Use the data generation module to convert SMILES strings to 3D molecular structures:

```python
from data_generate.mol2_gen import smi_to_mol
import pandas as pd

# Load SMILES data
smiles_df = pd.read_csv('data_generate/smiles.csv')

# Generate MOL2 files
for idx, smi in enumerate(smiles_df['smiles']):
    smi_to_mol(smi, idx, 'dataset/mol')
```

### 2. Generate Hamiltonian Ground Truth

Generate reference Hamiltonian matrices using PySCF:

```python
from module_nn.Hamiltonian_gen import Hamiltonian_gen

mol2_file = 'dataset/mol/1.mol2'
ham_gen = Hamiltonian_gen(mol2_file, 'dataset/Hamiltonian', 1)
ham_gen.save_matrix()  # Saves Hamiltonian and overlap matrices
```

### 3. Train the Model

Train the DeepMolH model on your dataset:

```python
from module_nn.Tranning_module import CompleteDeepMolHTrainer

# Initialize trainer
trainer = CompleteDeepMolHTrainer(
    data_dir='dataset',
    use_optuna=True  # Enable hyperparameter optimization
)

# Load dataset
train_data, val_data, test_data = trainer.load_dataset(
    train_ratio=0.7,
    val_ratio=0.15
)

# Train with Optuna optimization
best_params = trainer.run_optuna_optimization(
    train_data=train_data,
    val_data=val_data,
    n_trials=50
)

# Train final model
best_model = trainer.train_final_model(
    train_data=train_data,
    val_data=val_data,
    best_params=best_params
)
```

### 4. Use Pre-trained Model for Prediction

Load a trained model and predict Hamiltonians:

```python
from module_nn.NeuralNetwork import DeepMolH
import torch

# Load model
model = DeepMolH(
    num_egat_layers=10,
    num_gat_layers=10,
    num_heads=8,
    dropout=0.1
)
model.load_state_dict(torch.load('best_deepmolh_model.pth'))
model.eval()

# Predict Hamiltonian for a molecule
mol2_file = 'path/to/molecule.mol2'
with torch.no_grad():
    hamiltonian = model(mol2_file)
    
print(f"Predicted Hamiltonian shape: {hamiltonian.shape}")
```

## Project Structure

```
DeepMolH/
├── module_nn/              # Neural network modules
│   ├── NeuralNetwork.py   # Main DeepMolH model architecture
│   ├── GAT_Layer.py       # Graph Attention Network layer
│   ├── EGAT_Layer.py      # Equivariant GAT layer
│   ├── Hamiltonian_block.py        # Hamiltonian block generators
│   ├── E(3)_Hamiltonian_block.py   # E(3)-equivariant blocks
│   ├── Hamiltonian_gen.py          # PySCF Hamiltonian generation
│   ├── Tranning_module.py          # Training and optimization
│   ├── atom_embedding.py           # Atomic feature embeddings
│   ├── bond_embedding.py           # Bond feature embeddings
│   ├── nodes_embedding.py          # Node feature generation
│   ├── edges_embedding.py          # Edge feature generation
│   └── overlap_block.py            # Overlap matrix handling
├── data_generate/          # Data generation utilities
│   ├── mol2_gen.py        # SMILES to MOL2 conversion
│   ├── data_gen.py        # Dataset preparation
│   └── smiles.csv         # Sample SMILES data
├── dataset/                # Training data
│   ├── mol/               # MOL2 molecular structures
│   ├── Hamiltonian/       # Reference Hamiltonian matrices
│   └── edges/             # Edge features
└── best_deepmolh_model.pth # Pre-trained model weights
```

## Model Architecture

DeepMolH uses a multi-stage architecture:

1. **Molecular Embedding Layer**: Converts MOL2 files into node and edge features
2. **Graph Attention Layers**: Multiple GAT layers for local structure learning
3. **Equivariant Graph Attention Layers**: EGAT layers with E(3) equivariance
4. **Hamiltonian Block Generation**: Predicts diagonal and off-diagonal Hamiltonian blocks
   - Separate networks for different atomic periods (1-4)
   - Cross-period interaction blocks for multi-element molecules

### Supported Elements

The model supports atoms from periods 1-4 of the periodic table:
- **Period 1**: H, He
- **Period 2**: Li, Be, B, C, N, O, F, Ne
- **Period 3**: Na, Mg, Al, Si, P, S, Cl, Ar
- **Period 4**: K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr

## Training Features

- **Loss Function**: Mean Squared Error (MSE) between predicted and reference Hamiltonians
- **Optimizer**: Adam optimizer with configurable learning rate
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Early Stopping**: Based on validation loss
- **Checkpointing**: Automatic saving of best models
- **Training Curves**: Visualization of training progress

## Input Data Format

### MOL2 Files
Molecular structures in Tripos MOL2 format with atomic coordinates and bond information.

### Hamiltonian Matrices
Sparse matrices stored in NPZ format:
- `{id}_h.npz`: Hamiltonian matrix (Fock matrix from DFT)
- `{id}_o.npz`: Overlap matrix (basis function overlaps)

### SMILES Input
CSV file with columns:
- `smiles`: SMILES string representation
- `omega`: Target property (optional)

## Performance Tips

- **GPU Acceleration**: Training on GPU is highly recommended
- **Batch Processing**: Process multiple molecules in parallel when possible
- **Memory Management**: Large molecules may require memory optimization
- **Basis Set**: Default is STO-3G; adjust in `Hamiltonian_gen.py` for higher accuracy

## Getting Help

For questions and discussions:
- Open an issue on [GitHub Issues](https://github.com/STOKES-DOT/DeepMolH/issues)
- Check existing issues for solutions to common problems

## Citation

If you use DeepMolH in your research, please cite this repository:

```bibtex
@software{deepmolh,
  title = {DeepMolH: Deep Learning for Molecular Hamiltonian Prediction},
  author = {STOKES-DOT},
  url = {https://github.com/STOKES-DOT/DeepMolH},
  year = {2024}
}
```

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and request features via [GitHub Issues](https://github.com/STOKES-DOT/DeepMolH/issues)
- Submit pull requests with improvements
- Share your use cases and results

## License

Please check the repository for license information.

## Acknowledgments

This project builds on:
- **PyTorch**: Deep learning framework
- **E3NN**: Equivariant neural network library
- **PySCF**: Python-based quantum chemistry package
- **RDKit**: Cheminformatics toolkit
- **PyTorch Geometric**: Graph neural network library

---

**Note**: This is an active research project. The model architecture and API may evolve over time.
