from bs4 import BeautifulSoup
import requests
import pickle
import pandas as pd
import tqdm


games = ["Red", "Blue", "Yellow", "Gold", "Silver", "Crystal", "Ruby","Sapphire","Emerald", "FireRed",
         "LeafGreen", "Diamond", "Pearl", "Platinum", "HeartGold", "SoulSilver", "Black", "White", "Black 2", "White 2",
         "X", "Y", "Omega Ruby",  "Sun", "Moon", "Ultra Sun", "Ultra Moon", "Alpha Sapphire",
         "Let's Go Pikachu", "Let's Go Eevee", "Sword", "Shield",
         "Brilliant Diamond", "Shining Pearl", "Legends: Arceus"]

def in_game(x):
    for game in games:
        if(game in x):
            return True
    return False

def get_pokemon_data(name):
    r = requests.get(f"https://pokemondb.net/pokedex/{name}")
    soup = BeautifulSoup(r.content, features="html.parser")

    tag = soup.find("h2", text="Pokédex entries")

    while(not tag.name == "table"):
        tag = tag.next_element

    entry_words = tag.get_text(separator="\n").split()
    entries = {}
    cur_entry = ""
    prev_word = ""
    skip = 0
    game_enc = []
    other = False

    for i, word in enumerate(entry_words):
        if(skip > 0):
            skip -= 1
            continue

        if (i < len(entry_words) - 2 and word + " " + entry_words[i + 1]+" "+entry_words[i+2] in games):
            game_enc.append(word + " " + entry_words[i + 1]+" "+entry_words[i+2])
            if (other):
                w = word
                for g in game_enc:
                    entries[g] = cur_entry.rstrip()

                cur_entry = ""
                game_enc = [w]
                other = False
            skip = 2
            continue

        if (i < len(entry_words) - 1 and word + " " + entry_words[i + 1] in games):
            game_enc.append(word + " " + entry_words[i + 1])
            if (other):
                w = word
                for g in game_enc:
                    entries[g] = cur_entry.rstrip()

                cur_entry = ""
                game_enc = [w]
                other = False
            skip = 1
            continue
        if(word in games):
            if(other):
                w = word
                if(entry_words[i-1]+" "+word in games):
                    w = entry_words[i-1]+" "+word
                    cur_entry = " ".join(cur_entry.split()[:-1])
                for g in game_enc:
                    entries[g] = cur_entry.rstrip()

                cur_entry = ""
                game_enc = [w]
                other = False
            else:
                if(i < len(entry_words)-1 and word+" "+entry_words[i+1] in games):
                    game_enc.append(word+" "+entry_words[i+1])
                    skip = 1
                    continue
                game_enc.append(word)

            continue

        else:
            other = True

        word = word if word.lower() != name.lower() else "it"
        cur_entry += word + " "

    for g in game_enc:
        entries[g] = cur_entry.rstrip()


    return entries

# dat = get_pokemon_data("bulbasaur")
# for key, val in dat.items():
#     print(key, val)
# exit()
game_entries = [[] for g in games]
pokemon = pickle.load(open("all_pokemon.txt", "rb"))
for poke in tqdm.tqdm(pokemon):
   # print(poke)
    fillers = []
    fill_with = None
    entries = get_pokemon_data(poke)
    for i,game in enumerate(games):
        if (game not in entries):
            fillers.append(i)
        else:
            game_entries[i].append(entries[game])
            if(fill_with is None):
                fill_with = (i, entries[game])
    for f in fillers:
        if(fill_with == None):
            print(poke, entries)
        game_entries[f].append(fill_with[1])

df_dict = {"names":pokemon}
for game_name, entry in zip(games, game_entries):
    df_dict[game_name] = entry

df = pd.DataFrame(df_dict)
df.to_csv("pokemon_pokedex.csv")






