from PIL import Image
from abc import ABC, abstractmethod
from .paths import PALETTES_PATH
from .errors import BadInputError
class Color(ABC):
    @abstractmethod
    def get_rgba(self, palette_name): ...
    
class HexColor(Color):
    def __init__(self, rgba: tuple[int, int, int, int] = (0xff, 0xff, 0xff, 0xff)):
        self.rgba = rgba
    
    def get_rgba(self, _):
        return self.rgba

class PaletteColor(Color):
    def __init__(self, index: tuple[int, int]):
        self.palette_index = index
    
    def get_rgba(self, palette_name):
        try:
            palette = Image.open(PALETTES_PATH / f"{palette_name}.png").convert("RGBA")
        except FileNotFoundError:
            raise BadInputError(f"Palette `{palette_name}` not found!")
        return palette.getpixel(self.palette_index)

def parse_color(arg: str):
    if arg.startswith("#"): # hexcode
        color_str = arg[1:]
        assert len(color_str) in (6, 8), "hexcodes must be in #RRGGBB or #RRGGBBAA format"
        color_list: list[int] = []
        for i in range(0, len(color_str), 2):
            channel_str = color_str[i:i+2]
            try:
                num = int(channel_str, base=16)
            except ValueError:
                raise BadInputError(f"`{channel_str}` doesn't seem to parse as a hexadecimal number.")
            color_list.append(num)
        if len(color_list) == 3:
            color_list.append(0xff)
        return HexColor(tuple(color_list))
    elif "," in arg: # palette index
        return PaletteColor(tuple(int(i) for i in arg.split(",", maxsplit=1)))
    else:
        assert arg in COLOR_NAMES, f"Color name `{arg}` not found."
        return COLOR_NAMES[arg]

COLOR_NAMES: dict[str, tuple[int, int]] = { # stolen from richolas but
    "maroon": PaletteColor((2, 1)),
    "gold": PaletteColor((6, 2)),
    "teal": PaletteColor((1, 2)),
    "red": PaletteColor((2, 2)),
    "orange": PaletteColor((2, 3)),
    "yellow": PaletteColor((2, 4)),
    "lime": PaletteColor((5, 3)),
    "green": PaletteColor((5, 2)),
    "cyan": PaletteColor((1, 4)),
    "blue": PaletteColor((3, 2)),
    "purple": PaletteColor((3, 1)),
    "pink": PaletteColor((4, 1)),
    "rosy": PaletteColor((4, 2)),
    "grey": PaletteColor((0, 1)),
    "gray": PaletteColor((0, 1)),
    "black": PaletteColor((0, 4)),
    "silver": PaletteColor((0, 2)),
    "white": PaletteColor((0, 3)),
    "brown": PaletteColor((6, 1)),
}