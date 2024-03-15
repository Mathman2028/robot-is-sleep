from .tiles import Tile
from .colors import Color, parse_color
import inspect
from itertools import zip_longest
from typing import Literal, get_origin, get_args
from .variants import variants
from .utils import recolor
from PIL import Image, ImageMath
from .errors import BadInputError
from .customtiles import get_customtile
from types import GenericAlias
from .tiles import tiles
import random
import re
from random import randint

PREFIX_CHARS = {
    "=": "",
    "$": "text",
    "#": "glyph",
    ";": "event",
    "^": "node",
    "|": "txt",
}

def split_outside_brackets(string: str, sep: str) -> list[str]:
    first_split = string.split(sep)
    final = []
    current = []
    brackets = 0
    for i in first_split:
        current.append(i)
        brackets += i.count("[")
        brackets -= i.count("]")
        assert brackets >= 0, "Invalid brackets"
        if brackets == 0:
            if current:
                final.append(sep.join(current))
                current = []
    if current:
        final.append(sep.join(current))
    return final

def parse_arg(annotation, arg: str, *, flags: dict[str, tuple[str, ...]]):
    if arg.startswith("[") and arg.endswith("]"):
        arg = arg[1:-1]
    if annotation == Tile:
        tile = parse_tile(arg, flags=flags)
        for var in tile.variants:
            var_func = var[0]
            var_args = var[1:]
            var_func(tile, *var_args, flags=flags)
        if not tile.applied:
            recolor(tile, tile.color.get_rgba(tile.palette))
        if any(tile.displacement):
            for wobble in range(3):
                image = tile.images[wobble]
                new_img = Image.new("RGBA", (image.width + 2 * abs(tile.displacement[0]), image.height + 2 * abs(tile.displacement[1])))
                new_img.paste(image, (2 * max(tile.displacement[0], 0), 2 * max(tile.displacement[1], 0)))
                tile.images[wobble] = new_img
        return tile
    elif annotation == Color:
        return parse_color(arg)
    elif get_origin(annotation) is Literal:
        assert arg in get_args(annotation), "invalid arg"
        return arg
    elif annotation == bool:
        assert arg in ("true", "True", "false", "False"), "Booleans must be true or false"
        return arg in ("true", "True")
    elif isinstance(annotation, GenericAlias):
        origin, (arg_type,) = get_origin(annotation), get_args(annotation)
        output = []
        if origin == list:
            args = split_outside_brackets(arg, ",")
            for i in args:
                output.append(parse_arg(arg_type, i, flags=flags))
            return output
        else:
            raise NotImplementedError("Whoops! Mathman you forgot to parse that idiot")
    else:
        try:
            return annotation(arg)
        except ValueError as e:
            raise BadInputError(f"Could not convert `{arg}` to {annotation.__name__}.") from e

def parse_variant(var_string: str, *, flags: dict[str, tuple[str, ...]]):
    var = split_outside_brackets(var_string, "/")
    name, *args = var
    assert name in variants, f"Variant `{name}` not found!"
    var_func = variants[name]
    signature = inspect.signature(var_func)
    params = list(signature.parameters.values())[1:]
    converted_args = [var_func] 
    star = None
    for arg, param in zip_longest(args, params):
        if star:
            annotation = star
        elif param is None:
            raise BadInputError("too many args!")
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            continue
        else:
            annotation = param.annotation
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                star = annotation
        assert arg is not None or star or param.default != inspect.Parameter.empty or param.kind == inspect.Parameter.VAR_POSITIONAL, f"missing arg: {param.name}"
        if arg is None:
            break
        converted_args.append(parse_arg(annotation, arg, flags=flags))
    return converted_args

def parse_tile(tile_string: str, *, flags: dict[str, tuple[str, ...]], override_prefix = None):
    if tile_string == "":
        tile_string = "-"
    while match := re.search(r"%(-?\d+)\|(-?\d+)", tile_string):
        min_val, max_val = [int(i) for i in match.groups()]
        val = randint(min_val, max_val)
        start, end = match.span()
        tile_string = tile_string[:start] + str(val) + tile_string[end:]
    
    while match := re.search(r"\+{([^\{\}]+)}", tile_string):
        expr = match.group(1)
        try:
            replacewith = str(ImageMath.eval(expr))
        except Exception as e:
            raise BadInputError(f"Your expression raised a {type(e).__name__}: {e.args[0]}") from e
        start, end = match.span()
        tile_string = tile_string[:start] + str(replacewith) + tile_string[end:]
        
    prefix = flags.get("text", ("",))
    assert len(prefix) < 2, "Text flag only has one arg"
    if len(prefix) == 0:
        prefix = "text"
    else:
        prefix, = prefix
    
    name, *variants = split_outside_brackets(tile_string, ":")
    
    palette = flags.get("palette", ("default",))
    assert len(palette) == 1, "Palette flag only has one arg"
    palette, = palette
    
    if override_prefix is not None:
        prefix = override_prefix
    
    if name[0] in PREFIX_CHARS:
        prefix = PREFIX_CHARS[name[0]]
        name = name[1:]
    if prefix != "":
        prefix += "_"
    if name in ["-", "."]:
        prefix = ""
    
    if name.startswith("random"):
        stuff = name.split("/")
        assert len(stuff) < 4, "too many args"
        name = stuff[0]
        if len(stuff) == 1 or not stuff[1]:
            tiling = None
            include_all_tiling = True
        else:
            include_all_tiling = False
            try:
                tiling = int(stuff[1])
            except ValueError:
                raise BadInputError("thats not an int")
            assert -1 <= tiling <= 4, "tiling must be legal"
        if len(stuff) < 3 or not stuff[2]:
            req_prefix = None
            include_all_prefix = True
        else:
            include_all_prefix = False
            req_prefix = stuff[2]
        name = random.choice([i for i in tiles if (include_all_tiling or tiles[i]["tiling"] == tiling) and (include_all_prefix or i.startswith(req_prefix))])
    
    if name.startswith("ct!"):
        ct_name, *ct_args = split_outside_brackets(name.removeprefix("ct!"), "/")
        ct_name = prefix + ct_name
        customtile = get_customtile(ct_name)
        ct_value = customtile["value"]
        for i, v in enumerate(ct_args):
            if v.startswith("[") and v.endswith("]"):
                v = v[1:-1]
            ct_value = ct_value.replace(f"${i + 1}", v)
        tile = parse_tile(ct_value, flags=flags, override_prefix="")
    else:
        tile = Tile(prefix + name, palette=palette, flags=flags)
    
    for i in variants:
        if not i:
            continue
        tile.variants.append(parse_variant(i, flags=flags))
    return tile

def parse(command: str, *, flags: dict[str, tuple[str, ...]]):
    rows = split_outside_brackets(command, "\n")
    result: list[list[list[Tile]]] = []
    for i in rows:
        row: list[list[Tile]] = []
        i = i.strip()
        for j in split_outside_brackets(i, " "):
            stack: list[Tile] = []
            j = j.strip()
            for k in split_outside_brackets(j, "&"):
                stack.append(parse_tile(k, flags=flags))
            row.append(stack)
        result.append(row)
    return result