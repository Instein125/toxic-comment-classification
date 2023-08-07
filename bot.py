import discord
from preprocessing import preprocess, tokenization
from prediction_model import prediction


async def send_message(message, user_message):
    try:
        text = preprocess(user_message)
        tokenized_text = tokenization(text)
        response_dict = prediction(tokenized_text)
        response = "\n".join(f"{key.capitalize()}: {value}" for key, value in response_dict[0].items())

        start_str = "\n The model prediction of: " + user_message + " is: "
        
        if (response_dict[0]['toxic'] == True):
            await message.channel.purge(limit = 1)

        await message.channel.send(start_str)
        await message.channel.send(response)

    except Exception as e:
        print(e)

def run_bot():
    TOKEN = 'MTEzODEyNDExMTUyOTM5ODQxMg.GEak4K.a4Eq3oCP1ELQDSz3Vd8MREI-3MNJ0UK7h0VAX8'
    intents = discord.Intents.default()
    intents.message_content = True

    client = MyClient(intents=intents)
    client.run(TOKEN)


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        if message.author == self.user:
            return
        user_message = str(message.content)
        print(f'Message from {message.author}: {user_message}')

        await send_message(message, user_message)

intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
