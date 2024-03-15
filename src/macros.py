import json
from .paths import MACRO_DB
from typing import TypedDict, Literal, NotRequired
import discord

class Macro(TypedDict):
    name: str
    description: str
    value: str
    author: int

with open(MACRO_DB) as f:
    macros: dict[str, Macro] = json.load(f)

def save_macros():
    with open(MACRO_DB, "w") as f:
        json.dump(macros, f)

# these functions are easy to make but i'm using them to allow me to switch to sqlite if i want to do that
def new_macro(name: str, description: str, value: str, author: discord.User):
    assert name not in macros, f"The macro `{name}` already exists!"
    macros[name] = {
        "name": name,
        "description": description,
        "value": value,
        "author": author.id,
    }
    save_macros()

def edit_macro(name: str, attribute: Literal["description", "value"], new: str, user: discord.User):
    assert name in macros, f"The macro `{name}` doesn't exist!"
    assert user.id == macros[name]["author"], "You must own the macro to edit it!"
    macros[name][attribute] = new
    save_macros()

def del_macro(name: str, user: discord.User):
    assert name in macros, f"The macro `{name}` doesn't exist!"
    assert user.id == macros[name]["author"], "You must own the macro to delete it!"
    del macros[name]
    save_macros()

def get_macro(name: str):
    assert name in macros, f"The macro `{name}` doesn't exist!"
    return macros[name]