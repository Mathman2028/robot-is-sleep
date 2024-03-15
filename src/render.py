from PIL import Image
from .utils import recolor, dynamictiling
from .tiles import Tile
from .colors import parse_color
from .errors import BadInputError
import numpy as np

def render(tiles: list[list[list[Tile]]], *, flags: dict[str, tuple[str, ...]]):
    palette = flags.get("palette", ("default",))
    assert len(palette) == 1, "Palette flag only has one arg"
    palette, = palette
    
    bg = flags.get("background", ("#00000000",))
    assert len(bg) < 2, "Background flag only has one arg"
    if len(bg) == 0:
        bg = "0,4"
    else:
        bg, = bg
    bgcolor = parse_color(bg)
    
    deterministic = "deterministic" in flags
    
    nowobble = "wobble" in flags
    
    size = flags.get("size", (24,))
    assert len(size) < 2, "Size flag only has one arg"
    if len(size) == 0:
        size = 32
    else:
        size, = size
    try:
        size = int(size)
    except ValueError:
        raise BadInputError("badbad")
    
    dynamictiling(tiles)
    height = len(tiles)
    width = max(len(i) for i in tiles)
    stack_size = max([max([len(i) for i in row] + [0]) for row in tiles] + [0])
    renders: list[Image.Image] = []
    for row in tiles:
        for stack in row:
            for tile in stack:
                for var in tile.variants:
                    var_func = var[0]
                    var_args = var[1:]
                    var_func(tile, *var_args, flags=flags)
                if not tile.applied:
                    recolor(tile, tile.color.get_rgba(tile.palette))
    for wobble in range(1 if nowobble else 3):
        base_image = Image.new("RGBA", (width * size, height * size), bgcolor.get_rgba(palette))
        for k in range(stack_size):
            for i, row in enumerate(tiles):
                for j, stack in enumerate(row):
                    if len(stack) > k:
                        tile = stack[k]
                        image = tile.images[(wobble + ((i + j) if deterministic else tile.wobble_offset)) % 3]
                        base_image.alpha_composite(image, (j * size + tile.displacement[0], i * size + tile.displacement[1]))
        if "raw" in flags:
            renders.append(base_image)
        else:
            renders.append(base_image.resize((base_image.width * 2, base_image.height * 2), resample=Image.Resampling.NEAREST))
    return renders
