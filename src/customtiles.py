import json
from .paths import CUSTOM_TILES_DB
from typing import TypedDict, Literal
import discord

class CustomTile(TypedDict):
    name: str
    description: str
    value: str
    author: int

with open(CUSTOM_TILES_DB) as f:
    customtiles: dict[str, CustomTile] = json.load(f)

def save_customtiles():
    with open(CUSTOM_TILES_DB, "w") as f:
        json.dump(customtiles, f)

# these functions are easy to make but i'm using them to allow me to switch to sqlite if i want to do that
def new_customtile(name: str, description: str, value: str, author: discord.User):
    assert name not in customtiles, f"The custom tile `{name}` already exists!"
    customtiles[name] = {
        "name": name,
        "description": description,
        "value": value,
        "author": author.id
    }
    save_customtiles()

def edit_customtile(name: str, attribute: Literal["description", "value"], new: str, user: discord.User):
    assert name in customtiles, f"The custom tile `{name}` doesn't exist!"
    assert user.id == customtiles[name]["author"], "You must own the custom tile to edit it!"
    customtiles[name][attribute] = new
    save_customtiles()

def del_customtile(name: str, user: discord.User):
    assert name in customtiles, f"The custom tile `{name}` doesn't exist!"
    assert user.id == customtiles[name]["author"], "You must own the custom tile to delete it!"
    del customtiles[name]
    save_customtiles()

def get_customtile(name: str):
    assert name in customtiles, f"The custom tile `{name}` doesn't exist!"
    return customtiles[name]