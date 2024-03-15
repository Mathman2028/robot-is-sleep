from dataclasses import dataclass, field
from PIL import Image
from .paths import SPRITES_PATH, CUSTOM_SPRITES_PATH, JSONS_PATH
from random import randrange
from .colors import Color, PaletteColor
import json
from typing import TypedDict, NotRequired
import glob
from .errors import BadInputError

TILING_VARIANTS: dict[int, int] = {
    # R, U, L, D, E, Q, Z, C
    # Straightforward so far, easy to compute with a bitfield
    0b00000000: 0,
    0b10000000: 1,
    0b01000000: 2,
    0b11000000: 3,
    0b00100000: 4,
    0b10100000: 5,
    0b01100000: 6,
    0b11100000: 7,
    0b00010000: 8,
    0b10010000: 9,
    0b01010000: 10,
    0b11010000: 11,
    0b00110000: 12,
    0b10110000: 13,
    0b01110000: 14,
    0b11110000: 15,
    # Messy from here on, requires hardcoding
    0b11001000: 16,
    0b11101000: 17,
    0b11011000: 18,
    0b11111000: 19,
    0b01100100: 20,
    0b11100100: 21,
    0b01110100: 22,
    0b11110100: 23,
    0b11101100: 24,
    0b11111100: 25,
    0b00110010: 26,
    0b10110010: 27,
    0b01110010: 28,
    0b11110010: 29,
    0b11111010: 30,
    0b01110110: 31,
    0b11110110: 32,
    0b11111110: 33,
    0b10010001: 34,
    0b11010001: 35,
    0b10110001: 36,
    0b11110001: 37,
    0b11011001: 38,
    0b11111001: 39,
    0b11110101: 40,
    0b11111101: 41,
    0b10110011: 42,
    0b11110011: 43,
    0b11111011: 44,
    0b11110111: 45,
    0b11111111: 46
}

class TileData(TypedDict):
    name: str
    tags: NotRequired[list[str]]
    color: list[str]
    active: NotRequired[list[str]]
    tiling: int
    diagonal_tiling: NotRequired[bool]
    sprite: str
    type: str
    source: str

tiles: dict[str, TileData] = {}

for path in glob.glob("*.json", root_dir=JSONS_PATH):
    with open(JSONS_PATH / path) as f:
        data = json.load(f)
        for i in data.values():
            i["source"] = path.removesuffix(".json")
        tiles.update(data)

class MayResize:
    def __init__(self, tile: "Tile", flags: dict):
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
        self.halfsize = size // 2
        self.tile = tile
        self.base_disp_x, self.base_disp_y = (self.halfsize - tile.images[0].width // 2, self.halfsize - tile.images[0].height // 2)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        new_disp_x, new_disp_y = (self.halfsize - self.tile.images[0].width // 2, self.halfsize - self.tile.images[0].height // 2)
        self.tile.displacement = (self.tile.displacement[0] - self.base_disp_x + new_disp_x, self.tile.displacement[1] - self.base_disp_y + new_disp_y)

@dataclass
class Tile:
    name: str
    images: list[Image.Image] = field(init=False)
    displacement: tuple[int, int] = (0, 0)
    anim_frame: int = 0
    applied: bool = False
    color: Color = field(init=False)
    variants: list[list] = field(init=False)
    palette: str = "default"
    data: TileData = field(init=False)
    wobble_offset: int = field(init=False)
    surrounds: int = 0b00000000 # RULDEQZC
    variant_count: int = 0
    flags: dict = None
    
    def set_tiling(self):
        if self.data["tiling"] == 1:
            self.anim_frame = TILING_VARIANTS[self.surrounds]
    def set_images(self, *, surrounds_reset = True):
        with self.may_resize():
            if self.name not in ["-", "."]:
                try:
                    for wobble in range(3):
                        if self.data["source"] == "baba":
                            image = Image.open(SPRITES_PATH / f"{self.data["sprite"]}_{self.anim_frame}_{wobble + 1}.png")
                        else:
                            image = Image.open(CUSTOM_SPRITES_PATH / self.data["source"] / f"{self.data["sprite"]}_{self.anim_frame}_{wobble + 1}.png")
                        self.images[wobble] = image.convert("RGBA")
                except FileNotFoundError as e:
                    if surrounds_reset and self.data["tiling"] == 1:
                        self.surrounds &= 0b11110000
                        self.anim_frame = TILING_VARIANTS[self.surrounds]
                        self.set_images(surrounds_reset=False)
                        return
                    raise BadInputError(f"`{self.data["name"]}` has no frame `{self.anim_frame}`") from e
            self.applied = False
    
    def __post_init__(self):
        assert self.name in tiles, f"Tile `{self.name}` not found!"
        self.data = tiles[self.name]
        size = self.flags.get("size", (24,))
        assert len(size) < 2, "Size flag only has one arg"
        if len(size) == 0:
            size = 32
        else:
            size, = size
        try:
            size = int(size)
        except ValueError:
            raise BadInputError("badbad")
        self.images = [Image.new("RGBA", (size, size)), Image.new("RGBA", (size, size)), Image.new("RGBA", (size, size))]
        self.variants = []
        self.set_images()
        self.color = PaletteColor(tuple(self.data.get("active", None) or self.data["color"]))
        self.wobble_offset = randrange(3)
    
    def may_resize(self):
        return MayResize(self, self.flags)