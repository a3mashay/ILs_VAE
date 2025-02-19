# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:42:26 2025

@author: mashi
"""

def custom_tokenize_smiles(smiles_list):
    """
    Custom tokenization for SMILES strings to better capture chemical structures.
    """
    tokenized_smiles = []
    special_tokens = {'C', '(', ')', 'F', 'c', 'O', '[', ']', '=', '1', 'N', '-', '+', 'n', 'S', 'B', 'H',
                        '@', '2', 'r', '#', 'P', 'l', '3', 'I', '/', 'i', 'A', 's', 'o', '4', 'b',
                      'CC', ')(', 'F)', ')C', '(=', '(F', 'C(', 'O)', '=O', '(C', 'cc', 'C)', '+]', '-]',
                       '[N', ')F', 'c1', 'C[', '](', ')[', '1c', '=C', 'S(', '1)', ']1', '(c', 'C1', '[n',
                       'n+', 'N+', 'OC', 'C=', '[O', 'O-', 'FC', 'N-', '[B', 'CO', '1C', ']S', 'Br', '(S',
                       'n(', 'r-', 'cn', 'c(', '[C', '])', 'NH', 'H]', '@H', 'n1', 'C@', '1=', 'c[', '#N',
                       'c2', '([', ')O', '[P', ']C', 'Cc', 'Cn', 'Cl', ')c', '1(', 'C#', 'N(', 'B-', '@@',
                       'F[', 'N)', 'H+', 'CN', ')N', '(O', 'P+', 'NC', '2c', '=N', '=[', '2)', ')=', '(#',
                       'P-', 'H2', '2+', 'N1', '#C', 'C2', 'l-', '(N', 'O=', ']=', 'N=', '1n', 'l)', '2C',
                       '[S', ')S', 'N#', 'H3', '3+', 'O[', '1N', '2=', '=1', 'OS', 'P(', '1[', 'CS', 'nc',
                       '(n', '[I', 'I-', 'N[', 'lC', 'r)', 'SC', 'C-', 'Oc', ')B', ']c', ')n', '12', 'S+',
                       'FS', 'N2', 'N@', 'l(', '@+', 'nH', 'H-', '1O', 'S-', 'CH', 'C3', 'NN', '/C', 'O1',
                       ']2', 'nn', '2n', 'C/', 'Si', 'i]', '(B', 'OP', '[A', 'rC', 'Al', ']P', 'c3', 'n[',
                       'lc', '3C', 'Sc', 'CP', '][', 'n2', '3(', '1S', 'cs', 'CB', 'O2', 'Nc', '1F', 'On',
                       'I)', '2(', '=c', 'O(', '3c', 'rc', 'S)', ')/', 'Nn', 'sc', '3)', ']n', '@]', '1I',
                       'S1', '(P', 'PO', 'r[', ']N', '2[', '-c', '=S', ')P', '/[', 'n-', ')-', '2N', '=P',
                       'P[', 'P@', '[H', 'PH', 'ON', 's1', 'S@', '/N', '1B', ')I', 'S/', ']/', ']O', 's2',
                       '(I', 'OH', '2-', '(o', 'o2', 'NO', 'co', 'o1', 'Fc', 'c/', '/1', '1s', '(/', 'NS',
                       '3B', '2B', 'H4', '4+', 'P1', 'C+', 'I[', ']I', ']B', 'IC', '1o', 'oc',
                       ')s', '3n', '/c', 'lS', '1-', '/B', 'Sn', '3=', 'As', 's-', '2F', 'l[', 'Sb', 'b-',
                       'SS', '[F', 'F-'}

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