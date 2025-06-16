import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Helper functions (unchanged)
def augment_smiles(smiles, num_augmentations):
    mol = Chem.MolFromSmiles(smiles)
    augmented_smiles = [smiles]
    if mol is not None:
        for _ in range(num_augmentations - 1):
            augmented_smiles.append(Chem.MolToSmiles(mol, doRandom=True))
    return augmented_smiles

def smiles_to_features(smiles, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits))
    else:
        return np.zeros((nBits,))

