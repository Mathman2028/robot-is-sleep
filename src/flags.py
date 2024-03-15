from discord.ext import commands

flags = {
    "palette",
    "background",
    "raw",
    "text",
    "speed",
    "combine",
    "deterministic",
    "format",
    "wobble",
    "size",
    "export",
}

shorthands = {
    "p": "palette",
    "b": "background",
    "r": "raw",
    "t": "text",
    "s": "speed",
    "c": "combine",
    "d": "deterministic",
    "f": "format",
    "w": "wobble",
    "S": "size",
    "e": "export"
}

class FlagConverter(commands.Converter):
    async def convert(self, ctx: commands.Context, argument: str):
        if argument == "-":
            raise ValueError("empty flag")
        if not argument.startswith("-"):
            raise commands.BadArgument("flags must start with -")
        argument = argument.removeprefix("-")
        if argument.startswith("-"):
            args = argument.removeprefix("-").split("=")
            if args[0] not in flags:
                raise commands.BadArgument("invalid flag")
            return {args[0]: tuple(args[1:])}
        else:
            if "=" in argument:
                # just one
                args = argument.split("=")
                if args[0] not in shorthands:
                    raise commands.BadArgument("invalid shorthand")
                return {shorthands[args[0]]: tuple(args[1:])}
            else:
                my_flags = {}
                for char in argument:
                    if char not in shorthands:
                        raise commands.BadArgument("invalid shorthand")
                    my_flags[shorthands[char]] = ()
                return my_flags