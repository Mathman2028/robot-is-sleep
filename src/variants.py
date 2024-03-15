from types import FunctionType
from typing import Literal, cast, TypeAlias
from .tiles import Tile
from .colors import Color, PaletteColor, HexColor
from .utils import recolor, get_all_colors, count_all_instances_of_color, only_color, prereplace
from PIL import Image, ImageFilter, ImageMath, ImageEnhance, ImageDraw
from .tiles import tiles
import numpy as np
import cv2
from .errors import BadInputError
import sys
from math import sin, cos, tan, radians
import re
from random import randint
from .paths import OVERLAYS_PATH

variants: dict[str, FunctionType] = {}
true_name_variants: dict[str, FunctionType] = {}


def variant(aliases: list | None = None):
    aliases = aliases or []

    def decorator(func: FunctionType):
        variants[func.__name__] = func
        true_name_variants[func.__name__] = (func, aliases)
        for i in aliases:
            variants[i] = func
        def wrapper(tile: Tile, *args, **kwargs):
            tile.variant_count += 1
            assert tile.variant_count < 100, "Max of 100 variants on 1 tile!"
            func(tile, *args, **kwargs)
        return wrapper

    return decorator

MetaKernel: TypeAlias = Literal["full", "edge", "msb", "diag", "sides"]
def gen_kernel(name: MetaKernel, size: int):
    ksize = 2 * size + 1
    if name in ("full", "edge", "msb", "corners"):
        ker = np.ones((ksize, ksize))
    else:
        ker = np.zeros((ksize, ksize))
    if name == "full":
        ker[size, size] = -(ksize**2) + 1
    elif name == "edge":
        ker[size, size] = -(ksize**2) + 5
        ker[0, 0] = 0
        ker[0, ksize - 1] = 0
        ker[ksize - 1, ksize - 1] = 0
        ker[ksize - 1, 0] = 0
    elif name == "msb":
        ker[size, size] = -(ksize**2) + 3
        ker[0, 0] = 0
        ker[0, ksize - 1] = 0
    elif name == "diag":
        np.fill_diagonal(ker, 1)
        ker[size, size] = -ksize + 1
    elif name == "sides":
        ker[size, ...] = 1
        ker[size, size] = -ksize + 1
    return ker

# LET THE VARIANTS BEGIN!

@variant(aliases=["c"])
def color(tile: Tile, color: Color, palette: str = None, *, flags: dict):
    """Recolor a tile. This color will be applied at the end instead of the original color."""
    tile.color = color
    tile.palette = palette or "default"


@variant(aliases=["~", "ac"])
def apply(tile: Tile, *, flags: dict):
    """Apply the color to a tile immediately."""
    recolor(tile, tile.color.get_rgba(tile.palette))


@variant(aliases=["f"])
def frame(tile: Tile, frame: int, *, flags: dict):
    """Set the frame of a tile."""
    tile.anim_frame = frame
    tile.set_images()


@variant(aliases=["d"])
def direction(
    tile: Tile,
    dir: Literal["r", "right", "u", "up", "l", "left", "d", "down"],
    *,
    flags: dict,
):
    """Set a tile's direction."""
    if tiles[tile.name]["tiling"] not in (0, 2, 3):
        return
    frame = tile.anim_frame + 1
    frame &= 0b00111
    if dir[0] == "r":
        frame |= 0
    elif dir[0] == "u":
        frame |= 8
    elif dir[0] == "l":
        frame |= 16
    elif dir[0] == "d":
        frame |= 24
    tile.anim_frame = (frame - 1) % 32
    tile.set_images()


@variant(aliases=["a"])
def animate(tile: Tile, anim: int, *, flags: dict):
    """Set the animation state of a tile. Use -1 for sleeping."""
    assert -1 <= anim < 4, "anim frame must be in range -1 to 3"
    if tiles[tile.name]["tiling"] not in (2, 3, 4):
        return
    assert (
        tiles[tile.name]["tiling"] == 2 or anim != -1
    ), "sleep is only available for characters"
    frame = (tile.anim_frame & 0b11000) + 1
    frame += anim
    tile.anim_frame = (frame - 1) % 32
    tile.set_images()


@variant(aliases=["disp"])
def displace(tile: Tile, x: int, y: int, *, flags: dict):
    """Displace a tile. This displacement is not applied immediately."""
    orig_x, orig_y = tile.displacement
    tile.displacement = (orig_x + x, orig_y + y)


@variant(aliases=["s"])
def scale(tile: Tile, x: float, y: float = None, *, flags: dict):
    """Scale a tile by a factor. If Y is not given, X will be used for both."""
    with tile.may_resize():
        if y is None:
            y = x
        for wobble in range(3):
            image = tile.images[wobble]
            image = image.resize(
                (int(image.width * x), int(image.width * y)), Image.Resampling.NEAREST
            )
            tile.images[wobble] = image
    


@variant(aliases=["scalepx", "st", "spx"])
def scaleto(tile: Tile, x: int, y: int = None, *, flags: dict):
    """Scale a tile to a size in pixels. If Y is not given, X will be used for both."""
    with tile.may_resize():
        if y is None:
            y = x
        for wobble in range(3):
            image = tile.images[wobble]
            image = image.resize((x, y), Image.Resampling.NEAREST)
            tile.images[wobble] = image

@variant(aliases=["rot"])
def rotate(tile: Tile, angle: float, *, flags: dict):
    """Rotate a tile clockwise by a number of degrees."""
    for wobble in range(3):
        image = tile.images[wobble]
        image = image.rotate(-angle)
        tile.images[wobble] = image


@variant()
def crop(tile: Tile, left: int, top: int, right: int, bottom: int, *, flags: dict):
    """Crop a tile to the box shown."""
    for wobble in range(3):
        image = tile.images[wobble]
        image = image.crop((left, top, right, bottom))
        tile.images[wobble] = image
    orig_x, orig_y = tile.displacement
    tile.displacement = (orig_x + left, orig_y + top)


@variant()
def cut(tile: Tile, other: Tile, *, flags: dict):
    """Cut out the shape of a tile from this tile."""
    for wobble in range(3):
        image = tile.images[wobble]
        image2 = other.images[wobble]
        width = max(image.width, image2.width)
        height = max(image.height, image2.height)
        padded = Image.new("RGBA", (width, height))
        padded2 = Image.new("RGBA", (width, height))
        padded.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        padded2.paste(
            image2, ((width - image2.width) // 2, (height - image2.height) // 2)
        )
        base = np.asarray(padded).copy().astype(np.float64, casting="unsafe")
        overlay = np.asarray(padded2).copy()
        base[..., 3] *= 1 - (overlay[..., 3] / 255)
        base[base[..., 3] == 0] = 0
        base = base.astype(np.uint8, casting="unsafe")
        image = Image.fromarray(base)
        tile.images[wobble] = image


@variant()
def mask(tile: Tile, other: Tile, *, flags: dict):
    """Use another tile as a mask for this tile."""
    for wobble in range(3):
        image = tile.images[wobble]
        image2 = other.images[wobble]
        width = max(image.width, image2.width)
        height = max(image.height, image2.height)
        padded = Image.new("RGBA", (width, height))
        padded2 = Image.new("RGBA", (width, height))
        padded.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        padded2.paste(
            image2, ((width - image2.width) // 2, (height - image2.height) // 2)
        )
        base = np.asarray(padded).copy().astype(np.float64, casting="unsafe")
        overlay = np.asarray(padded2).copy()
        base[..., 3] *= overlay[..., 3] / 255
        base[base[..., 3] == 0] = 0
        base = base.astype(np.uint8, casting="unsafe")
        image = Image.fromarray(base)
        tile.images[wobble] = image


@variant()
def stack(tile: Tile, other: Tile, *, flags: dict):
    """Stack another tile on top of this tile."""
    with tile.may_resize():
        for wobble in range(3):
            image = tile.images[wobble]
            image2 = other.images[wobble]
            width = max(image.width, image2.width)
            height = max(image.height, image2.height)
            padded = Image.new("RGBA", (width, height))
            padded.paste(
                image,
                ((width - image.width) // 2, (height - image.height) // 2),
                mask=image.getchannel("A"),
            )
            padded.paste(
                image2,
                ((width - image2.width) // 2, (height - image2.height) // 2),
                mask=image2.getchannel("A"),
            )
            tile.images[wobble] = padded

@variant(aliases=["m"])
def meta(
    tile: Tile,
    level: int = 1,
    kernel: MetaKernel = "full",
    size: int = 1,
    *,
    flags: dict,
):
    """Apply a meta effect to a tile."""
    assert size > 0, "Meta size must be positive!"
    for wobble in range(3):
        image = tile.images[wobble]
        padding = max(level * size, 0)  # stolen from richolas
        orig = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
        # check_size(*orig.shape[size::-1])
        base = orig[..., 3]
        if level < 0:
            base = 255 - base
        ker = gen_kernel(kernel, size)
        for _ in range(abs(level)):
            base = cv2.filter2D(src=base, ddepth=-1, kernel=ker)
        base = np.dstack((base, base, base, base))
        mask = orig[..., 3] > 0
        if not (level % 2) and level > 0:
            base[mask, ...] = orig[mask, ...]
        else:
            base[mask ^ (level < 0), ...] = 0
        tile.images[wobble] = Image.fromarray(base)
    x, y = tile.displacement
    x -= padding
    y -= padding
    tile.displacement = (x, y)


@variant()
def csel(tile: Tile, index: int, *, flags: dict):
    """Selects a certain color based on occurance."""
    colors = get_all_colors(tile)
    colors = sorted(colors, key=lambda color: count_all_instances_of_color(tile, color))
    try:
        color = colors[index]
    except IndexError as e:
        raise BadInputError(
            f"Index `{index}` out of range for tile `{tile.name}`."
        ) from e
    for wobble in range(3):
        array = np.asarray(tile.images[wobble])
        array = only_color(array, color)
        tile.images[wobble] = Image.fromarray(array)


@variant()
def neon(tile: Tile, strength: float = 0.714, *, flags: dict):
    """Darkens the inside of each region of color."""
    CARD_KERNEL = np.array(((0, 1, 0), (1, 0, 1), (0, 1, 0)))
    OBLQ_KERNEL = np.array(((1, 0, 1), (0, 0, 0), (1, 0, 1)))
    unique_colors = get_all_colors(tile)
    for wobble in range(3):
        image = tile.images[wobble]
        sprite = np.asarray(image).copy()
        final_mask = np.ones(
            sprite.shape[:2], dtype=np.float64
        )  # rick stealing yet again
        for color in unique_colors:
            mask = (sprite == color).all(axis=2)
            float_mask = mask.astype(np.float64)
            card_mask = cv2.filter2D(src=float_mask, ddepth=-1, kernel=CARD_KERNEL)
            oblq_mask = cv2.filter2D(src=float_mask, ddepth=-1, kernel=OBLQ_KERNEL)
            final_mask[card_mask == 4] -= strength / 2
            final_mask[oblq_mask == 4] -= strength / 2
        if strength < 0:
            final_mask = np.abs(1 - final_mask)
        sprite[:, :, 3] = np.multiply(
            sprite[:, :, 3], np.clip(final_mask, 0, 1), casting="unsafe"
        )
        tile.images[wobble] = Image.fromarray(sprite)


@variant()
def convolve(tile: Tile, *nums: list[float], flags: dict):
    """Applies a convolution to a tile."""
    with tile.may_resize():
        for wobble in range(3):
            array = np.asarray(tile.images[wobble]).copy()
            try:
                filter = np.asarray(nums)
            except ValueError as e:
                raise BadInputError("Filter must be rectangular")
            result = cv2.filter2D(array, -1, filter)
            tile.images[wobble] = Image.fromarray(result)


@variant()
def filter(
    tile: Tile,
    filter: Literal[
        "blur",
        "contour",
        "detail",
        "edge_enhance",
        "edge_enhance_more",
        "emboss",
        "find_edges",
        "sharpen",
        "smooth",
        "smooth_more",
    ],
    *,
    flags: dict,
):
    for wobble in range(3):
        image = tile.images[wobble]
        image = image.filter(getattr(ImageFilter, filter.upper()))
        tile.images[wobble] = image


@variant()
def crystallize(tile: Tile, crystals: int = 48, *, flags: dict):
    for wobble in range(3):
        im = np.asarray(tile.images[wobble]).copy()
        # Make output image same size
        res = np.zeros_like(im)
        h, w = im.shape[:2]
        # Generate some randomly placed crystal centres
        nx = np.random.randint(0, w, crystals, dtype=np.uint16)
        ny = np.random.randint(0, h, crystals, dtype=np.uint16)
        # Pick up colours at those locations from source image
        sRGB = []
        for i in range(crystals):
            sRGB.append(im[ny[i], nx[i]])

        # Iterate over image
        for y in range(h):
            for x in range(w):
                # Find nearest crystal centre...
                dmin = sys.float_info.max
                for i in range(crystals):
                    d = (y - ny[i]) * (y - ny[i]) + (x - nx[i]) * (x - nx[i])
                    if d < dmin:
                        dmin = d
                        j = i
                # ... and copy colour of original image to result
                res[y, x, :] = sRGB[j]
        tile.images[wobble] = Image.fromarray(res)


@variant()
def eval(tile: Tile, channels: str, expr: str, *tiles: Tile, flags: dict):
    """Evaluates expressions on images. The base tile is called `src` and the other tiles are called `t1`, `t2`, and so on."""
    assert all(i in "RGBA" for i in channels), "Channels must be RGBA"
    channels = set(channels)
    for wobble in range(3):
        image = tile.images[wobble]
        r, g, b, a = image.split()
        channel_ims = {"R": r, "G": g, "B": b, "A": a}
        for channel in channels:
            channel_im = channel_ims[channel]
            other_ims = {
                "t" + str(i + 1): v.images[wobble].getchannel(channel)
                for i, v in enumerate(tiles)
            }
            try:
                evaled = ImageMath.eval(expr, other_ims, src=channel_im, sin=sin, cos=cos, tan=tan, radians=radians)
            except Exception as e:
                raise BadInputError(
                    f"Your expression raised a {type(e).__name__}: {e.args[0]}"
                )
            assert isinstance(evaled, Image.Image), "operation must return an image"
            channel_ims[channel] = evaled.convert("L")
        tile.images[wobble] = Image.merge("RGBA", [channel_ims[i] for i in "RGBA"])


@variant(aliases=["mc"])
def macro(tile: Tile, name: str, *args: str, flags: dict):
    from .macros import get_macro
    from .parsing import (
        parse_variant,
        split_outside_brackets,
    )  # hacky fix for circular import

    macro = get_macro(name)
    value = macro["value"]
    for i, v in enumerate(args):
        value = value.replace(f"${i+1}", v)
    value = value.replace("$N", tile.name)
    value = value.replace("$W", str(tile.images[0].width))
    value = value.replace("$H", str(tile.images[0].height))
    value = value.replace("$A", str(tile.anim_frame))
    value = prereplace(value)
    splitted = split_outside_brackets(value, ":")
    macro_vars = [parse_variant(i, flags=flags) for i in splitted]
    for i in macro_vars:
        func, *var_args = i
        func(tile, *var_args, flags=flags)


@variant()
def median(
    tile: Tile,
    size: int = 3,
    kind: Literal["min", "median", "max", "mode"] = "median",
    *,
    flags: dict,
):
    filters: dict[str, type] = {
        "min": ImageFilter.MinFilter,
        "median": ImageFilter.MedianFilter,
        "max": ImageFilter.MaxFilter,
        "mode": ImageFilter.ModeFilter,
    }
    assert size % 2 == 1, "Size must be odd"
    for wobble in range(3):
        image = tile.images[wobble]
        image = image.filter(filters[kind](size))
        tile.images[wobble] = image


@variant()
def enhance(
    tile: Tile,
    kind: Literal["color", "contrast", "brightness", "sharpness"],
    factor: float,
    *,
    flags: dict,
):
    assert 0 <= factor <= 2, "Factor must be between 0 and 2"
    kind = kind.title()
    enhancer: ImageEnhance._Enhance = getattr(ImageEnhance, kind)
    for wobble in range(3):
        image = tile.images[wobble]
        image = enhancer(image).enhance(factor)
        tile.images[wobble] = image


@variant(aliases=["flood", "fill"])
def floodfill(tile: Tile, color: Color, inside: bool = True, *, flags: dict):
    rgba = color.get_rgba(tile.palette)
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble]).copy()
        sprite[sprite[:, :, 3] == 0] = 0  # Optimal
        sprite_alpha = sprite[:, :, 3]  # Stores the alpha channel separately
        sprite_alpha[sprite_alpha > 0] = (
            -1
        )  # Sets all nonzero numbers to a number that's neither 0 nor 255.
        # Pads the alpha channel by 1 on each side to allow flowing past
        # where the sprite touches the edge of the bounding box.
        sprite_alpha = np.pad(sprite_alpha, ((1, 1), (1, 1)))
        sprite_flooded = cv2.floodFill(
            image=sprite_alpha, mask=None, seedPoint=(0, 0), newVal=255
        )[1]
        mask = sprite_flooded != (inside * 255)
        sprite_flooded[mask] = (not inside) * 255
        mask = mask[1:-1, 1:-1]
        if inside:
            sprite_flooded = 255 - sprite_flooded
        # Crops the alpha channel back to the original size and positioning
        sprite[:, :, 3][mask] = sprite_flooded[1:-1, 1:-1][mask].astype(np.uint8)
        sprite[(sprite[:, :] == [0, 0, 0, 255]).all(2)] = rgba
        tile.images[wobble] = Image.fromarray(sprite)


@variant()
def croppoly(tile: Tile, *xy: int, flags: dict):
    """Crops the sprite to the specified polygon."""
    assert len(xy) % 2 == 0, "Must have the same number of X and Y!"
    assert len(xy) >= 6, "Must have at least 3 points to make a polygon!"
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble])
        x_y = np.reshape(xy, (-1, 2))
        pts = np.array([x_y], dtype=np.int32).reshape((1, -1, 2))[:, :, ::-1]
        clip_poly = cv2.fillPoly(np.zeros(sprite.shape[:2], dtype=np.float32), pts, 1)
        clip_poly = np.tile(clip_poly, (4, 1, 1)).T
        tile.images[wobble] = Image.fromarray(
            np.multiply(sprite, clip_poly, casting="unsafe").astype(np.uint8)
        )


@variant(aliases=["mm", "matrix"])
def matmul(
    tile: Tile,
    red: list[float],
    green: list[float],
    blue: list[float],
    alpha: list[float],
    *,
    flags: dict,
):
    matrix = [red, green, blue, alpha]
    assert [len(i) == 4 for i in matrix], "Matrix must be 4x4"
    matrix = np.array(matrix)
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble]).astype(np.float64) / 255
        aftermul = sprite.reshape(-1, 4) @ matrix
        aftermul = (np.clip(aftermul, 0.0, 1.0) * 255).astype(np.uint8)
        tile.images[wobble] = Image.fromarray(aftermul.reshape(sprite.shape))


@variant(aliases=["cvt"])
def convert(
    tile: Tile,
    direction: Literal["to", "from"],
    space: Literal["BGR", "HSV", "HLS", "YUV", "YCrCb", "XYZ", "Lab", "Luv"],
    *,
    flags: dict,
):
    space_conversion = {
        "to": {
            "BGR": cv2.COLOR_RGB2BGR,
            "HSV": cv2.COLOR_RGB2HSV,
            "HLS": cv2.COLOR_RGB2HLS,
            "YUV": cv2.COLOR_RGB2YUV,
            "YCrCb": cv2.COLOR_RGB2YCrCb,
            "XYZ": cv2.COLOR_RGB2XYZ,
            "Lab": cv2.COLOR_RGB2Lab,
            "Luv": cv2.COLOR_RGB2Luv,
        },
        "from": {
            "BGR": cv2.COLOR_BGR2RGB,
            "HSV": cv2.COLOR_HSV2RGB,
            "HLS": cv2.COLOR_HLS2RGB,
            "YUV": cv2.COLOR_YUV2RGB,
            "YCrCb": cv2.COLOR_YCrCb2RGB,
            "XYZ": cv2.COLOR_XYZ2RGB,
            "Lab": cv2.COLOR_Lab2RGB,
            "Luv": cv2.COLOR_Luv2RGB,
        },
    }
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble]).copy()
        sprite[:, :, :3] = cv2.cvtColor(
            sprite[:, :, :3], space_conversion[direction][space]
        )
        tile.images[wobble] = Image.fromarray(sprite)


@variant(aliases=["ot"])
def outline(
    tile: Tile,
    color: Color = PaletteColor((0, 4)),
    kernel: MetaKernel = "full",
    size: int = 1,
    *,
    flags: dict,
):
    """Apply an outline effect to a tile."""
    assert size > 0, "Outline size must be positive!"
    for wobble in range(3):
        image = tile.images[wobble]
        padding = max(size, 0)  # stolen from richolas
        orig = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
        # check_size(*orig.shape[size::-1])
        base = orig[..., 3]
        ker = gen_kernel(kernel, size)
        base = cv2.filter2D(src=base, ddepth=-1, kernel=ker)
        base = np.dstack((base, base, base, base))
        base = np.multiply(
            base, np.array(color.get_rgba(tile.palette)) / 255, casting="unsafe"
        ).astype(np.uint8)
        mask = orig[..., 3] > 0
        base[mask, ...] = orig[mask, ...]
        tile.images[wobble] = Image.fromarray(base.astype(np.uint8))
    x, y = tile.displacement
    x -= padding
    y -= padding
    tile.displacement = (x, y)


@variant()
def melt(
    tile: Tile,
    side: Literal["left", "top", "right", "bottom"] = "bottom",
    *,
    flags: dict,
):
    """Removes transparent pixels from each row/column and shifts the remaining ones to the end."""
    is_vertical = side in ("top", "bottom")
    at_end = side in ("right", "bottom")
    for wobble in range(3):
        image = tile.images[wobble]
        sprite = np.asarray(image).copy()
        if is_vertical:
            sprite = np.swapaxes(sprite, 0, 1)
        for i in range(sprite.shape[0]):
            sprite_slice = sprite[i, sprite[i, :, 3] != 0]
            sprite[i] = np.pad(
                sprite_slice,
                (
                    (sprite[i].shape[0] - sprite_slice.shape[0], 0)[:: 2 * at_end - 1],
                    (0, 0),
                ),
            )
        if is_vertical:
            sprite = np.swapaxes(sprite, 0, 1)
        tile.images[wobble] = Image.fromarray(sprite)


@variant()
def affine(tile: Tile, row1: list[float], row2: list[float], *, flags: dict):
    """Applies an affine transformation to a tile."""
    try:
        matrix = np.array((row1, row2))
    except ValueError as e:
        raise BadInputError("Matrix must be 2x3")
    assert matrix.shape == (2, 3), "Matrix must be 2x3"
    for wobble in range(3):
        image = tile.images[wobble]
        image = cv2.warpAffine(
            np.asarray(image),
            matrix,
            (image.width, image.height),
            flags=cv2.INTER_NEAREST,
        )
        tile.images[wobble] = Image.fromarray(image)


@variant()
def pad(
    tile: Tile,
    left: int,
    top: int = None,
    right: int = None,
    bottom: int = None,
    *,
    flags: dict,
):
    """Pads a sprite by a certain amount."""
    if top is None:
        top = left
    if right is None:
        right = left
    if bottom is None:
        bottom = top
    new_size = (
        tile.images[0].width + left + right,
        tile.images[1].height + top + bottom,
    )
    for wobble in range(3):
        image = tile.images[wobble]
        new_image = Image.new("RGBA", new_size)
        new_image.paste(image, (left, top))
        tile.images[wobble] = new_image
    old_x, old_y = tile.displacement
    tile.displacement = (old_x - left, old_y - top)


@variant()
def glitch(tile: Tile, distance: int, *, flags: dict):
    """Randomly spreads pixels in a tile."""
    for wobble in range(3):
        image = tile.images[wobble]
        tile.images[wobble] = image.effect_spread(distance)


@variant(aliases=["rstdisp"])
def resetdisp(tile: Tile, *, flags: dict):
    """Sets a tile's displacement to 0. This re-anchors the image's top left corner to the tile's top-left corner in the render."""
    tile.displacement = (0, 0)


@variant(aliases=["t", "tile"])
def connect(
    tile: Tile,
    dir: Literal["r", "u", "l", "d", "ru", "ur", "ul", "lu", "ld", "dl", "dr", "rd"],
    *,
    flags: dict,
):
    """Manually connect a tile to another."""
    if "r" in dir:
        tile.surrounds |= 0b10000000  # RULDEQZC
    if "u" in dir:
        tile.surrounds |= 0b01000000
    if "u" in dir:
        tile.surrounds |= 0b00100000
    if "u" in dir:
        tile.surrounds |= 0b00010000
    if dir in ("ru", "ur"):
        tile.surrounds |= 0b00001000
    if dir in ("ul", "lu"):
        tile.surrounds |= 0b00000100
    if dir in ("dl", "ld"):
        tile.surrounds |= 0b00000010
    if dir in ("dr", "rd"):
        tile.surrounds |= 0b00000001
    tile.set_tiling()
    tile.set_images()


@variant(aliases=["tearpx"])
def tearpixels(
    tile: Tile, distance: int, chance: float = 1.0, angle: int = 0, seed: int = None, *, flags: dict
):
    """Randomly displace pixels in a tile in one direction."""
    assert distance >= 0, "distance must be > 0"
    radian_angle = radians(angle)
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble])
        dst = np.indices(sprite.shape[:2], dtype=np.float32).transpose(0, 2, 1)
        rng = np.random.default_rng(seed)
        displacement = rng.uniform(0, distance, sprite.shape[:2])
        mask = rng.uniform(0, 1, displacement.shape)
        displacement[mask > chance] = 0
        displacement = np.reshape(displacement, displacement.shape + (1,))
        displacement = cast(np.ndarray, np.pad(displacement, ((0, 0), (0, 0), (0, 1))))
        newdisp = (displacement.reshape(-1, 2) @ np.asarray([[cos(radian_angle), sin(radian_angle)],[-sin(radian_angle), cos(radian_angle)]])).reshape(displacement.shape)
        newdisp = newdisp.transpose(2, 0, 1)
        dst += newdisp
        new = cv2.remap(
            sprite,
            dst[0],
            dst[1],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_WRAP,
        )
        tile.images[wobble] = Image.fromarray(new)

@variant()
def tear(
    tile: Tile, distance: int, chance: float = 1.0, seed: int = None, *, flags: dict
):
    """Randomly displace pixels in a tile in one direction."""
    assert distance >= 0, "distance must be > 0"
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble])
        dst = np.indices(sprite.shape[:2], dtype=np.float32).transpose(0, 2, 1)
        rng = np.random.default_rng(seed)
        displacement = rng.uniform(0, distance, (sprite.shape[0],))
        mask = rng.uniform(0, 1, displacement.shape)
        displacement[mask > chance] = 0
        displacement = np.reshape(displacement, displacement.shape + (1,))
        dst[0] += displacement
        new = cv2.remap(
            sprite,
            dst[0],
            dst[1],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_WRAP,
        )
        tile.images[wobble] = Image.fromarray(new)

@variant(aliases=["rc", "rcol"])
def randomcolor(tile: Tile, *, flags: dict):
    """A random hex color."""
    tile.color = HexColor((randint(0, 255), randint(0, 255), randint(0, 255), randint(0, 255)))


@variant(aliases=["rp"])
def repeat(tile: Tile, amount: int, *variants: str, flags: dict):
    from .parsing import parse_variant
    assert amount > 0, "Amount must be >0"
    assert amount <= 20, "Repeat may only run 20 times!"
    parsed_vars = []
    for i in variants:
        var = parse_variant(i, flags=flags)
        assert var[0].__name__ != "repeat", "Repeat cannot call itself."
        parsed_vars.append(var)
    for _ in range(amount):
        for var in parsed_vars:
            var_func = var[0]
            var_args = var[1:]
            var_func(tile, *var_args, flags=flags)

@variant(aliases=["st"])
def stretch(tile: Tile, axis: Literal["x", "y"], size: int, offset: int = 0, *, flags: dict):
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble])
        if axis == "x":
            sprite = sprite.swapaxes(0, 1)
        lhalf = sprite[:(sprite.shape[0] // 2 + offset), ...]
        rhalf = sprite[(sprite.shape[0] // 2 + offset):, ...]
        sprite = np.concatenate([lhalf, lhalf[-1, ...].reshape(1, *lhalf.shape[1:]).repeat(size, 0), rhalf[0, ...].reshape(1, *rhalf.shape[1:]).repeat(size, 0), rhalf], 0)
        if axis == "x":
            sprite = sprite.swapaxes(0, 1)
        tile.images[wobble] = Image.fromarray(sprite)
    
    tile.displacement = (tile.displacement[0]-(size if axis == "x" else 0), tile.displacement[1]-(size if axis == "y" else 0))

@variant(aliases=["th"])
def threshold(tile: Tile, channels: str, value: float, below: bool = False, *, flags: dict):
    assert all(i in "RGBA" for i in channels), "Channels must be RGBA"
    channels = set(channels)
    for wobble in range(3):
        image = tile.images[wobble]
        r, g, b, a = image.split()
        channel_ims = {"R": r, "G": g, "B": b, "A": a}
        sprite = np.asarray(image).copy()
        keep = np.ones_like(sprite[..., 0], np.bool_)
        for channel in channels:
            channel_im = channel_ims[channel]
            csprite = np.asarray(channel_im).copy()
            if below:
                keepfilter = csprite < (value * 255)
            else:
                keepfilter = csprite > (value * 255)
            keep &= keepfilter
        sprite[keep, ...] = 0
        tile.images[wobble] = Image.fromarray(sprite)
    
@variant(aliases=["o"])
def overlay(tile: Tile, name: Literal["ace", "aro", "bi", "enby", "fluid", "gay", "lesbian", "pan", "poly", "trans"], *, flags: dict):
    for wobble in range(3):
        image = tile.images[wobble]
        overlay = Image.open(OVERLAYS_PATH / f"{name}_{wobble + 1}.png")
        overlay.putalpha(Image.new("L", (overlay.width, overlay.height), 255))
        tile.images[wobble] = Image.fromarray(np.multiply(np.asarray(image) / 255, overlay.resize((image.width, image.height), Image.Resampling.NEAREST), casting="unsafe").astype("uint8"))
    tile.applied = True

@variant(aliases=["iso"])
def isolate(tile: Tile, color: Color, *, flags: dict):
    col_array = np.asarray(color.get_rgba(tile.palette))
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble])
        sprite = only_color(sprite, col_array)
        tile.images[wobble] = Image.fromarray(sprite)

@variant(aliases=["in"])
def inactive(tile: Tile, *, flags: dict):
    tile.color = PaletteColor(tuple(tile.data["color"]))

@variant()
def snip(tile: Tile, left: int, top: int, right: int, bottom: int, *, flags: dict):
    """Snip the given box out of a tile."""
    for wobble in range(3):
        image = tile.images[wobble]
        draw = ImageDraw.Draw(image)
        draw.rectangle(((left, top), (right, bottom)), fill=0)

@variant()
def wrap(tile: Tile, x: int, y: int, *, flags: dict):
    for wobble in range(3):
        sprite = np.asarray(tile.images[wobble])
        tile.images[wobble] = Image.fromarray(np.roll(sprite, (y, x), (0, 1)))