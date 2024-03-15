import discord
from discord.ext import commands
import asyncio

intents = discord.Intents.default()
intents.message_content = True

allowed_mentions = discord.AllowedMentions(everyone=False, users=False, roles=False, replied_user=True)

bot = commands.Bot(">", intents=intents, allowed_mentions=allowed_mentions, activity=discord.Activity(type=discord.ActivityType.listening, name=">r", help_command=None))

def main():
    with open("TOKEN.txt") as f:
        token = f.read()
    asyncio.run(bot.load_extension("src.commands"))
    bot.run(token)

@bot.command()
async def reload(ctx: commands.Context):
    await bot.reload_extension("src.commands")
    await ctx.send("Reloaded!")

if __name__ == "__main__":
    main()