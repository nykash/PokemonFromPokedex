import pandas as pd
import tqdm

games = ["Red", "Blue", "Yellow", "Gold", "Silver", "Crystal", "Ruby","Sapphire","Emerald", "FireRed",
         "LeafGreen", "Diamond", "Pearl", "Platinum", "HeartGold", "SoulSilver", "Black", "White", "Black 2", "White 2",
         "X", "Y", "Omega Ruby",  "Sun", "Moon", "Ultra Sun", "Ultra Moon", "Alpha Sapphire",
         "Let's Go Pikachu", "Let's Go Eevee", "Sword", "Shield",
         "Brilliant Diamond", "Shining Pearl", "Legends: Arceus"]


df = pd.read_csv("pokemon_pokedex.csv")
data = {"entry": [], "name": []}


for i in tqdm.tqdm(range(df.shape[0])):
    row = df.iloc[i]
    done = []
    for game in games:
        if(len(row[game])<=0):
            continue
        if(row[game] not in done):
            data["entry"].append(row[game])
            data["name"].append(row["names"])
            done.append(row[game])

new_df = pd.DataFrame(data)
new_df.to_csv("inverted_dex.csv")