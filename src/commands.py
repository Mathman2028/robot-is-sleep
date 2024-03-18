from discord.ext import commands
from discord.ext import tasks
from discord import ui
import discord
from .errors import BadInputError
from .flags import FlagConverter
from .render import render
from .parsing import parse
from .variants import true_name_variants
from .macros import get_macro, new_macro, edit_macro, del_macro, save_macros
from .customtiles import get_customtile, new_customtile, edit_customtile, del_customtile, save_customtiles
import io
import inspect
from typing import Literal, get_origin, get_args, cast
import types
from PIL import Image, ImageSequence
import zipfile

class Commands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        super().__init__()
    
    async def cog_unload(self):
        save_macros()
        save_customtiles()
    
    @tasks.loop(minutes=5)
    async def save_stuff(self):
        save_macros()
        save_customtiles()
    
    @commands.command(name="render", aliases=["r"])
    async def render_image(self, ctx: commands.Context, flags: commands.Greedy[FlagConverter], *, data: str):
        parsed_flags = {}
        for i in flags:
            parsed_flags |= i
        parsed = parse(data.replace("\\:", ":"), flags=parsed_flags)
        rendered = render(parsed, flags=parsed_flags)
        
        speed = parsed_flags.get("speed", (200,))
        assert len(speed) == 1, "Speed flag has exactly 1 arg"
        try:
            speed = int(speed[0])
        except ValueError as e:
            raise BadInputError(f"Invalid integer literal: {speed[0]}") from e
        speed = max(min(speed, 65535), 20)
        
        combine = "combine" in parsed_flags
        if combine:
            reference = ctx.message.reference
            if reference is None:
                message = [i async for i in ctx.channel.history(limit=2)][1]
            else:
                message = reference.resolved
                assert isinstance(message, discord.Message), "Your replied message was deleted."
            assert len(message.attachments) > 0, "That message has no attachments!"
            attachment = message.attachments[0]
            with io.BytesIO() as image_binary:
                await attachment.save(image_binary)
                image = Image.open(image_binary)
                rendered = ImageSequence.all_frames(image) + rendered
        
        format = parsed_flags.get("format", ("gif",))
        assert len(format) == 1, "Format flag has exactly 1 arg"
        format = format[0]
        assert format in ["png", "gif", "pdf", "webp"], "Invalid file format"
        
        with io.BytesIO() as image_binary:
            rendered[0].save(
                image_binary, format, save_all=True, append_images=rendered[1:], loop=0, duration=speed, disposal=2, optimize=False
            )
            image_binary.seek(0)
            await ctx.message.reply(file=discord.File(fp=image_binary, filename=f"render.{format}"), mention_author=False)
        
        if "export" in parsed_flags:
            assert len(parsed_flags["export"]) == 1, "Export flag has 1 arg"
            exportname, = parsed_flags["export"]
            with io.BytesIO() as zipbinary:
                file = zipfile.PyZipFile(zipbinary, "x")
                for i, img in enumerate(rendered):
                    with io.BytesIO() as buffer:
                        img.save(buffer, "PNG")
                        buffer.seek(0)
                        file.writestr(
                            f"{exportname}_{i // 3}_{(i % 3) + 1}.png",
                            buffer.getvalue())
                file.close()
                zipbinary.seek(0)
                await ctx.message.reply(file=discord.File(fp=zipbinary, filename=f"render.zip"), mention_author=False)
        

    def get_var_name(self, annotation):
        if get_origin(annotation) is Literal:
            return " | ".join(get_args(annotation))
        elif type(annotation) == types.GenericAlias:
            return self.get_var_name(get_args(annotation)[0]) + "[]"
        else:
            return annotation.__name__

    def get_var_signature(self, name: str, func: types.FunctionType):
        signature = inspect.signature(func)
        params = list(signature.parameters.values())[1:]
        param_str = name
        for param in params:
            annotation = param.annotation
            default = param.default != inspect.Parameter.empty
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                break
            middle = self.get_var_name(annotation)
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                middle += "[]"
            param_str += f"/{"[" if default else "<"}{middle} {param.name}{"]" if default else ">"}"
        return param_str

    @commands.command(name="variants", aliases=["vars"])
    async def vars(self, ctx: commands.Context):
        var_list = sorted(true_name_variants.items())
        page = 0
        async def next_page(interaction: discord.Interaction):
            nonlocal page
            page += 1
            await update(interaction)
        async def prev_page(interaction: discord.Interaction):
            nonlocal page
            page -= 1
            await update(interaction)
        def gen_embed():
            embed = discord.Embed(title="Variants")
            for i in range(page * 5, min(page * 5 + 5, len(var_list))):
                name, (func, aliases) = var_list[i]
                param_str = self.get_var_signature(name, func)
                embed.add_field(name=param_str + (f" (Aliases: {", ".join(aliases)})" if aliases else ""), value=func.__doc__)
            embed.set_footer(text="<> = required, [] = optional -- do not include <> or [] in usage")
            view = ui.View()
            prev_button = ui.Button(emoji="⬅️", disabled=page == 0)
            prev_button.callback = prev_page
            view.add_item(prev_button)
            next_button = ui.Button(emoji="➡️", disabled=page * 5 + 5 >= len(var_list))
            next_button.callback = next_page
            view.add_item(next_button)
            return embed, view
        async def update(interaction: discord.Interaction):
            embed, view = gen_embed()
            await interaction.response.defer()
            await interaction.message.edit(embed=embed, view=view)
        embed, view = gen_embed()
        await ctx.message.reply(embed=embed, view=view)
    
    @commands.group(aliases=["m", "mc"])
    async def macro(self, ctx: commands.Context):
        """A frontend for adding macros."""
        if ctx.subcommand_passed is None:
            await ctx.send(embed=discord.Embed(title="Macros", description=
"""
This is a frontend for users (that means you!) to create macros.
Macros can be applied using the macro variant.
Subcommands:
""").add_field(name="macro new <name> <value> <description>", value="Create a new macro.\nAliases: create, make, mk", inline=False
               ).add_field(name="macro edit <name> <attribute> <new>", value='Edit a macro. Attributes can be "value" or "description".\nAliases: e', inline=False
                           ).add_field(name="macro delete <name>", value="Delete a macro.\nAliases: del, remove, rm", inline=False
                                       ).add_field(name="macro info <name>", value="Get information about a macro.\nAliases: i, get", inline=False))
    
    @macro.command(name="new", aliases=["make", "mk", "create"])
    async def mcnew(self, ctx: commands.Context, name: str, value: str, *, description: str):
        """Create a new macro."""
        new_macro(name, description, value, ctx.author)
        await ctx.send("Macro created!")
    
    @macro.command(name="edit", aliases=["e"])
    async def mcedit(self, ctx: commands.Context, name: str, attribute: Literal["description", "value"], new: str):
        """Edit a macro."""
        edit_macro(name, attribute, new, ctx.author)
        await ctx.send("Macro edited!")
    
    @macro.command(name="delete", aliases=["del", "remove", "rm"])
    async def mcdelete(self, ctx: commands.Context, name: str):
        """Delete a macro."""
        del_macro(name, ctx.author)
        await ctx.send("Macro deleted!")
    
    @macro.command(name="info", aliases=["i", "get"])
    async def mcinfo(self, ctx: commands.Context, name: str):
        """Get information about a macro."""
        macro = get_macro(name)
        user = await self.bot.fetch_user(macro["author"])
        embed = discord.Embed(title=name, description=f"{macro["description"]}\n```\n{macro["value"]}\n```").set_footer(text=user.name, icon_url=user.avatar.url if user.avatar else None)
        await ctx.send(embed=embed)
    
    @commands.group(aliases=["ct", "ctile"])
    async def customtile(self, ctx: commands.Context):
        if ctx.subcommand_passed is None:
            await ctx.send(embed=discord.Embed(title="Custom tiles", description=
'''
This is a frontend for users (that means you!) to create custom tiles.
To use a custom tile, prefix the custom tile's name with the prefix "ct!".
Subcommands:
''').add_field(name="customtile new <name> <value> <description>", value="Create a new custom tile.\nAliases: create, make, mk", inline=False
               ).add_field(name="customtile edit <name> <attribute> <new>", value='Edit a custom tile. Attributes can be "value" or "description".\nAliases: e', inline=False
                           ).add_field(name="customtile delete <name>", value="Delete a custom tile.\nAliases: del, remove, rm", inline=False
                                       ).add_field(name="customtile info <name>", value="Get information about a custom tile.\nAliases: i, get", inline=False))
    
    @customtile.command(name="new", aliases=["make", "mk", "create"])
    async def ctnew(self, ctx: commands.Context, name: str, value: str, *, description: str):
        new_customtile(name, description, value, ctx.author)
        await ctx.send("Custom tile created!")
    
    @customtile.command(name="edit", aliases=["e"])
    async def ctedit(self, ctx: commands.Context, name: str, attribute: Literal["description", "value"], new: str):
        edit_customtile(name, attribute, new, ctx.author)
        await ctx.send("Custom tile edited!")
    
    @customtile.command(name="delete", aliases=["del", "remove", "rm"])
    async def ctdelete(self, ctx: commands.Context, name: str):
        del_customtile(name, ctx.author)
        await ctx.send("Custom tile deleted!")
    
    @customtile.command(name="info", aliases=["i", "get"])
    async def ctinfo(self, ctx: commands.Context, name: str):
        customtile = get_customtile(name)
        user = await self.bot.fetch_user(customtile["author"])
        embed = discord.Embed(title=name, description=f"{customtile["description"]}\n```\n{customtile["value"]}\n```").set_footer(text=user.name, icon_url=user.avatar.url if user.avatar else None)
        await ctx.send(embed=embed)
        
    @commands.command()
    async def interp(self, ctx: commands.Context, start: int, end: int, frame_count: int):
        output = ""
        for i in range(frame_count):
            t = i / frame_count
            output += str(((1 - t) ** 3) * start + (1 - (1 - t) ** 3) * end) + " "
        await ctx.send(output)
    
    @commands.command()
    async def support(self, ctx: commands.Context):
        await ctx.send("This command will be updated with more helpful information later. For now, join the [support server](https://discord.gg/8R2vg9Chvy) for help!")
    
    @commands.command()
    async def cmds(self, ctx: commands.Context):
        await ctx.send_help()

    async def cog_command_error(self, ctx: commands.Context, error: Exception):
        while isinstance(error, commands.CommandInvokeError):
            error = error.original
        if isinstance(error, commands.CommandNotFound):
            pass
        elif isinstance(error, (AssertionError, BadInputError)):
            await ctx.message.reply(error.args[0])
        elif isinstance(error, commands.BadLiteralArgument):
            await ctx.message.reply(f"{error.argument} is not included in {error.literals}")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.message.reply(f"The argument `{error.param.name}` is missing!")
        else:
            await ctx.message.reply(embed=discord.Embed(color=discord.Color.red(), title="An error occured!", description=error))
            raise error

async def setup(bot: commands.Bot):
    await bot.add_cog(Commands(bot))
    
