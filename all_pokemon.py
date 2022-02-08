from bs4 import BeautifulSoup
import requests
import tqdm
import pickle
import unidecode

def process(texts):
    if(texts[1].lower() in types):
        return proc(texts[0].lower())

    text = proc(texts[0].lower()+"-"+texts[1].lower())
    return text

def proc(text):
    if "♀" in text:
        text = text.replace("♀", "-f")
    if "♂" in text:
        text = text.replace("♂", "-m")
    text = text.replace("'", "")
    text = text.replace(".", "")
    text = text.rstrip("-")
    text = text.replace(":", "")
    text = unidecode.unidecode(text)
    return text

types = ["grass", "fire", "dark", "fairy", "ground", "ice", "water", "steel", "electric",
         "rock", "psychic", "poison", "flying", "ghost", "normal", "bug", "dragon", "fighting"]

r = requests.get("https://pokemondb.net/pokedex/national")
soup = BeautifulSoup(r.content, features="html.parser")

tags = soup.find_all("div", class_="infocard-list infocard-list-pkmn-lg")
pokemon = []
for info in tags:
    pkmn_entries = info.text.split("\n")

    for row in tqdm.tqdm(pkmn_entries):
        splat = row.split()
        if(len(splat) < 2):
            continue

        pokemon.append(process(splat[1:3]))

    print(pokemon)
    print(len(pokemon))

pickle.dump(pokemon, open("all_pokemon.txt", "wb+"))
