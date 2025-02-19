def custom_tokenize_smiles(smiles_list):
    """
    Custom tokenization for SMILES strings.
    """
    tokenized_smiles = []
    special_tokens = {
        #must be the elements and characters from periodic table
        }

    for smiles in smiles_list:
        i = 0
        tokens = []
        while i < len(smiles):
            two_char = smiles[i:i+2]
            if two_char in special_tokens:
                tokens.append(two_char)
                i += 2
            elif smiles[i] in special_tokens:
                tokens.append(smiles[i])
                i += 1
            else:
                # Single characters and numbers
                tokens.append(smiles[i])
                i += 1
        tokenized_smiles.append(tokens)
    return tokenized_smiles

def calculate_token_variance(tokenized_smiles):
    """
    Calculate the variance of each token's across the dataset.
    """
    # Flatten the list of tokens to get overall frequency
    all_tokens = [token for sublist in tokenized_smiles for token in sublist]
    token_counts = Counter(all_tokens)

    # Calculate frequencies
    num_samples = len(tokenized_smiles)
    token_frequencies = {token: count / num_samples for token, count in token_counts.items()}

    # Calculate variance of token presence across samples
    variances = {}
    for token in token_counts.keys():
        presence = np.array([token in sample for sample in tokenized_smiles])
        variances[token] = np.var(presence)

    return variances

# Select tokens based on variance threshold
def select_informative_tokens(variances, threshold=0.01):
    """
    Select tokens with a variance above a certain threshold.
    """
    return [token for token, var in variances.items() if var > threshold]