from bs4 import BeautifulSoup
import requests
import tqdm
import pickle
import unidecode
import os

def get_pokemon_img(name):
    url = f"https://img.pokemondb.net/artwork/{name}.jpg"
    response = requests.get(url, stream=True)

    with open(f'pokemon/{name}.jpg', 'wb+') as handle:
        if not response.ok:
            url_2 = f"https://pokemondb.net/pokedex/{name}"
            r = requests.get(url_2, stream=True)
            soup = BeautifulSoup(r.content, features="html.parser")

            tags = soup.find_all("img")
            url = tags[0]["src"]

            response = requests.get(url, stream=True)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)


pokemon = pickle.load(open("all_pokemon.txt", "rb"))
for poke in tqdm.tqdm(pokemon):
    get_pokemon_img(poke)
