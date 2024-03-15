import numpy as np
from .tiles import Tile, tiles as tile_data
from PIL import Image
from typing import Literal

def recolor(tile: Tile, color: tuple[int, int, int, int]): # stole this directly from richolas
    for wobble in range(3):
        image = tile.images[wobble]
        arr = np.multiply(image, np.array(color) / 255, casting="unsafe").astype(np.uint8)
        tile.images[wobble] = Image.fromarray(arr)
    tile.applied = True

def flatten_to_color_array(x: np.ndarray):
    return x.reshape(-1, 4)

def get_colors(x: np.ndarray):
    f = flatten_to_color_array(x)
    return np.unique(f[f[:, 3] != 0], axis=0)

def get_all_colors(tile: Tile):
    array = np.concatenate(tile.images)
    return get_colors(array)

def count_instances_of_color(x, color):
    f = flatten_to_color_array(x)
    return np.count_nonzero((f[:] == color).all(1))

def count_all_instances_of_color(tile: Tile, color):
    array = np.concatenate(tile.images)
    f = flatten_to_color_array(array)
    return np.count_nonzero((f[:] == color).all(1))

def only_color(x: np.ndarray, color: np.ndarray):
    f = flatten_to_color_array(x).copy()
    f[(f[:] != color).any(1)] = [0, 0, 0, 0]
    return f.reshape(x.shape)

def dynamictiling(tiles: list[list[list[Tile]]]):
    for i, row in enumerate(tiles):
        for j, stack in enumerate(row):
            for tile in stack:
                data = tile_data[tile.name]
                if data["tiling"] != 1:
                    continue
                surrounds = 0b00000000
                tile_with = [tile.name, "level"]
                if j < len(row) - 1:
                    for tile2 in row[j+1]:
                        if tile2.name in tile_with:
                            surrounds |= 0b10000000
                            break
                if j > 0:
                    for tile2 in row[j-1]:
                        if tile2.name in tile_with:
                            surrounds |= 0b00100000
                            break
                if i > 0 and j < len(tiles[i-1]):
                    for tile2 in tiles[i-1][j]:
                        if tile2.name in tile_with:
                            surrounds |= 0b01000000
                            break
                if i < len(tiles) - 1 and j < len(tiles[i+1]):
                    for tile2 in tiles[i+1][j]:
                        if tile2.name in tile_with:
                            surrounds |= 0b00010000
                            break
                if surrounds & 0b11000000 == 0b11000000 and i > 0 and j < len(tiles[i-1]) - 1:
                    for tile2 in tiles[i-1][j+1]:
                        if tile2.name in tile_with:
                            surrounds |= 0b00001000 #???
                            break
                if surrounds & 0b10010000 == 0b10010000 and i < len(tiles) - 1 and j < len(tiles[i+1]) - 1:
                    for tile2 in tiles[i+1][j+1]:
                        if tile2.name in tile_with:
                            surrounds |= 0b00000001
                            break
                if surrounds & 0b01100000 == 0b01100000 and i > 0 and 0 < j < len(tiles[i-1]):
                    for tile2 in tiles[i-1][j-1]:
                        if tile2.name in tile_with:
                            surrounds |= 0b00000100
                            break
                if surrounds & 0b00110000 == 0b00110000 and i < len(tiles) - 1 and 0 < j < len(tiles[i+1]) + 1:
                    for tile2 in tiles[i+1][j-1]:
                        if tile2.name in tile_with:
                            surrounds |= 0b00000010 #???
                            break
                tile.surrounds = surrounds
                tile.set_tiling()
                tile.set_images()

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches
