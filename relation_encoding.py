import pandas as pd
import numpy as np
import tqdm
import pickle
import os

poke_types = pd.read_csv("pokemon_types.csv")

types = ["grass", "fire", "dark", "fairy", "ground", "ice", "water", "steel", "electric",
         "rock", "psychic", "poison", "flying", "ghost", "normal", "bug", "dragon", "fighting"]

def similarity(row1, row2, type_eq_scalar=0.33):
    if(row1["name"] == row2["name"]):
        return 1.0
    eqs = 0
    s1 = 0
    s2 = 0
    for t in types:
        if(1 == row1[t] == row2[t]):
            eqs += 1
        s1 += row1[t]
        s2 += row2[t]

    eqs /= s1
    eqs /= s2

    return eqs * type_eq_scalar

for i, row in tqdm.tqdm(poke_types.iterrows()):
    vector = []
    for j, row_2 in poke_types.iterrows():
        vector.append(similarity(row, row_2))

    pickle.dump(np.array(vector), open(f"relation_vecs/{row['name']}.txt", "wb+"))
