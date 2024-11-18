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

def load_data_with_smiles(file_path, target_columns):
    data = pd.read_excel(file_path)
    all_smiles = []  # List to hold all SMILES, including augmented ones
    features = []
    targets = {col: [] for col in target_columns}  # Initialize a dictionary for each target

    for _, row in data.iterrows():
        original_smiles = row['Cation'] + '.' + row['Anion']
        augmented_smiles = augment_smiles(original_smiles, num_augmentations=200)
        for smiles in augmented_smiles:
            all_smiles.append(smiles)  # Append each augmented SMILES
            features.append(smiles_to_features(smiles))
            for col in target_columns:
                targets[col].append(row[col])  # Append target for each augmented SMILES

    features = np.array(features)
    # Convert targets into a format suitable for training
    targets_array = np.column_stack([targets[col] for col in target_columns])
    return all_smiles, features, targets_array
