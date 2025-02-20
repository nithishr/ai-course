import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # take environment variables from .env.

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tg_bot_token = os.getenv("TG_BOT_TOKEN")


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions like a pirate.",
    }
]

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    messages.append({"role": "user", "content": update.message.text})
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    answer = completion.choices[0].message
    messages.append(answer)
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=answer.content
    )


if __name__ == "__main__":
    application = ApplicationBuilder().token(tg_bot_token).build()

    start_handler = CommandHandler("start", start)
    chat_handler = CommandHandler("chat", chat)

    application.add_handler(start_handler)
    application.add_handler(chat_handler)

    application.run_polling()
